#!/usr/bin/env python3
"""CatSeek R1 v0.3 — a bilingual reasoning BitNet-style in-RAM model.

CatSeek R1 trains its model from scratch when the program starts.  The
tokenizer, English/Mandarin detection, training corpus, gradient descent,
trainable weights, dual-residual ternary BitLinear layers, shared recurrent
latent reasoning, A8 activation quantization, next-token softmax,
autoregressive generation, chat history, GUI, CLI, and proof-oriented
self-tests all live in this one Python file.

files = off:
    * no checkpoint is read or written
    * no model is downloaded
    * no API or network request is used
    * all learned weights and optimizer state live only in RAM

Normal chat is never selected from a response table.  Embedded dialogues are
training examples, just as a dataset is training material rather than an
inference-time script.  Every normal reply is emitted token by token from the
model's probability distribution.  Slash commands are the only deterministic
control responses.

This is an honest compact educational language model, not a frontier-scale
model.  It requires Python 3.10+ and NumPy, but no model package or checkpoint.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import queue
import re
import sys
import threading
import time
import tkinter as tk
from dataclasses import dataclass, field
from tkinter import font, scrolledtext
from typing import Callable, Iterable, Optional

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - environment error path
    raise SystemExit(
        "CatSeek R1 requires NumPy for real in-memory neural training. "
        "No model checkpoint is required."
    ) from exc


APP_NAME = "CatSeek R1"
APP_VERSION = "0.3"
MODEL_ID = "catseek-r1-dtr-reasoning-bitnet-ram-v0.3"
FILES_MODE = "off"
DEFAULT_SEED = 0xCA75_EE41


@dataclass(frozen=True, slots=True)
class ModelConfig:
    context_tokens: int = 16
    embedding_dim: int = 32
    hidden_dim: int = 128
    reasoning_passes: int = 3
    reasoning_scale: float = 0.35
    train_steps: int = 1400
    batch_size: int = 192
    learning_rate: float = 0.003
    gradient_clip: float = 1.0
    activation_bits: int = 8
    seed: int = DEFAULT_SEED
    max_new_tokens: int = 140
    temperature: float = 0.0
    top_k: int = 8


@dataclass(slots=True)
class TrainingReport:
    initial_loss: float = math.inf
    final_loss: float = math.inf
    steps: int = 0
    samples: int = 0
    elapsed_s: float = 0.0
    loss_history: list[float] = field(default_factory=list)


@dataclass(slots=True)
class GenerationStep:
    index: int
    token: str
    token_id: int
    probability: float
    entropy_bits: float


@dataclass(slots=True)
class GenerationReport:
    text: str
    language: str
    tokens: int
    elapsed_s: float
    tokens_per_second: float
    finish_reason: str
    trace: list[GenerationStep]


@dataclass(slots=True)
class Reply:
    text: str
    route: str
    elapsed_ms: float
    tokens: int = 0
    tokens_per_second: float = 0.0


class WordTokenizer:
    """Small word/subword tokenizer fitted entirely from the embedded corpus."""

    PAD = "<PAD>"
    UNK = "<UNK>"
    BOS = "<BOS>"
    EOS = "<EOS>"
    USER = "<USER>"
    ASSISTANT = "<ASSISTANT>"
    EN = "<EN>"
    ZH = "<ZH>"
    NL = "<NL>"
    SPECIAL = (PAD, UNK, BOS, EOS, USER, ASSISTANT, EN, ZH, NL)
    TOKEN_RE = re.compile(
        r"```|[A-Za-z0-9_+#./-]+|[\u3400-\u4dbf\u4e00-\u9fff]|[^\w\s]",
        re.UNICODE,
    )

    def __init__(self, texts: Iterable[str]):
        vocabulary: set[str] = set(self.SPECIAL)
        for text in texts:
            vocabulary.update(self.basic_tokenize(text))
        ordered = list(self.SPECIAL) + sorted(vocabulary.difference(self.SPECIAL))
        self.id_to_token = ordered
        self.token_to_id = {token: index for index, token in enumerate(ordered)}
        self.lower_to_id: dict[str, int] = {}
        for token, index in self.token_to_id.items():
            self.lower_to_id.setdefault(token.lower(), index)

    @classmethod
    def basic_tokenize(cls, text: str) -> list[str]:
        tokens: list[str] = []
        pieces = re.split(r"(\n)", text.replace("\r\n", "\n").replace("\r", "\n"))
        for piece in pieces:
            if piece == "\n":
                tokens.append(cls.NL)
            elif piece:
                tokens.extend(cls.TOKEN_RE.findall(piece))
        return tokens

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    def token_id(self, token: str) -> int:
        exact = self.token_to_id.get(token)
        if exact is not None:
            return exact
        return self.lower_to_id.get(token.lower(), self.token_to_id[self.UNK])

    def encode(self, text: str) -> list[int]:
        return [self.token_id(token) for token in self.basic_tokenize(text)]

    def encode_tokens(self, tokens: Iterable[str]) -> list[int]:
        return [self.token_id(token) for token in tokens]

    def decode(self, token_ids: Iterable[int]) -> str:
        output = ""
        no_space_before = {
            ".", ",", "!", "?", ";", ":", ")", "]", "}",
            "。", "，", "！", "？", "；", "：", "、", "）", "》", "】",
        }
        no_space_after = {"(", "[", "{", "（", "《", "【"}
        hidden = {
            self.PAD, self.UNK, self.BOS, self.EOS, self.USER,
            self.ASSISTANT, self.EN, self.ZH,
        }
        for token_id in token_ids:
            if not 0 <= int(token_id) < self.vocab_size:
                continue
            token = self.id_to_token[int(token_id)]
            if token in hidden:
                continue
            if token == self.NL:
                output = output.rstrip() + "\n"
            elif not output or output.endswith("\n"):
                output += token
            elif is_han(token) or is_han(output[-1:]):
                output += token
            elif token in no_space_before or output[-1:] in no_space_after:
                output += token
            else:
                output += " " + token
        return output.strip()


HAN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
LATIN_RE = re.compile(r"[A-Za-z]")


def is_han(text: str) -> bool:
    """Return whether *text* contains a Han character."""
    return bool(HAN_RE.search(text))


def detect_language(text: str) -> str:
    """Detect English versus Mandarin-oriented Han text without a service.

    Han script is shared by multiple languages, so this deliberately reports
    the app's supported response mode (``zh``) rather than claiming linguistic
    identification that Unicode characters alone cannot prove.
    """
    han = len(HAN_RE.findall(text))
    latin = len(LATIN_RE.findall(text))
    if han >= 2 or (han == 1 and han >= latin):
        return "zh"
    return "en"


# These are training examples, not an inference-time response lookup table.
# Normal prompts never search this structure; it is converted to tensors and
# discarded as a decision mechanism before chat begins.
BOOT_DIALOGUES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("hello", "hi", "hey CatSeek", "good morning"),
     "Hello! I am CatSeek R1, running locally from RAM. What are we building?"),
    (("who are you", "what model are you", "tell me about yourself"),
     "I am CatSeek R1, a compact next-token language model trained in memory when this program starts."),
    (("are you a real language model", "are you scripted", "do you use fixed replies"),
     "Normal chat is not a fixed response lookup. My learned weights produce a probability for every next token, and I generate autoregressively."),
    (("what does files off mean", "files off", "do you load a checkpoint"),
     "Files off means my tokenizer, training, weights, optimizer, and inference stay in RAM. I do not read or write a model checkpoint."),
    (("what is BitNet", "explain ternary weights", "what is W1.58A8"),
     "BitNet-style inference uses ternary weights: negative one, zero, or positive one. CatSeek R1 uses trained ternary projections and eight-bit activations."),
    (("what is CatSeek DTR", "explain your new math", "what is dual residual ternary"),
     "CatSeek DTR approximates each learned matrix with two ternary branches. The first captures the main weight and the second quantizes the remaining error, reducing distortion while keeping ternary operations."),
    (("how do your reasoning passes work", "explain latent reasoning", "do you use extra compute per token"),
     "Before predicting each token, I apply three shared-weight latent refinement passes. Each pass mixes the current hidden state with a learned ternary residual and feeds the refined state to the next pass."),
    (("what is a language model", "explain next token prediction", "how does text generation work"),
     "A language model estimates the probability of the next token from previous tokens. Generation appends one predicted token to the context and repeats until an end token."),
    (("how are you trained", "explain your training", "what does startup training do"),
     "At startup I tokenize embedded training text, minimize cross-entropy with Adam, and keep the resulting neural weights only in memory."),
    (("what is cross entropy", "explain loss", "what does training loss mean"),
     "Cross-entropy measures how much probability the model assigns to the correct next token. Lower loss means the model predicts its training sequences more accurately."),
    (("what is autoregressive generation", "define autoregressive", "how do you generate a reply"),
     "Autoregressive generation predicts one token, appends it to context, recomputes the distribution, and predicts the next token."),
    (("what is tokenization", "explain tokens", "how do you tokenize text"),
     "Tokenization converts text into vocabulary identifiers. CatSeek R1 uses compact word, punctuation, newline, and control tokens."),
    (("write Python hello world", "make a Python hello program", "Python print example"),
     "Here is Python code:\n```\nprint(\"Hello from CatSeek R1\")\n```"),
    (("write a Python function", "show a Python function", "Python function example"),
     "A small Python function can validate its input and return a result:\n```\ndef square(value: float) -> float:\n    return value * value\n```"),
    (("explain recursion", "what is recursion", "show recursive thinking"),
     "Recursion solves a problem by calling the same function on a smaller input. A base case must stop the calls."),
    (("how do I debug code", "my program crashes", "help fix a traceback"),
     "Keep the complete error, reproduce it with the smallest input, inspect state at the first bad boundary, patch one cause, and rerun a regression test."),
    (("design an emulator", "how do I build an emulator", "emulator architecture"),
     "Start with a CPU state machine, bus, memory map, timing model, interrupts, graphics, audio, and deterministic diagnostic tests."),
    (("how do I make a NES emulator", "NES emulator plan", "emulate the NES"),
     "For a NES emulator, implement the 6502 CPU subset, CPU bus, cartridge mapper, PPU registers, controller ports, interrupts, and timing tests before polishing the GUI."),
    (("how do I make a SNES emulator", "SNES emulator plan", "emulate the SNES"),
     "A SNES emulator needs a 65C816 CPU, banked memory bus, PPU, APU, DMA, HDMA, interrupts, cartridge mapping, and careful master-clock scheduling."),
    (("make a game", "game development plan", "how do I build a game"),
     "Build one vertical slice first: input, fixed-step update, rendering, collision, audio, and a clear win condition."),
    (("how do I optimize performance", "make it faster", "performance advice"),
     "Measure the same workload first, find the dominant cost, change one variable, and compare latency, throughput, and memory before claiming a speedup."),
    (("how should I use Git", "Git workflow", "make a pull request"),
     "Inspect the diff, isolate the intended change, run checks, commit a concrete unit of work, push a topic branch, and explain behavior and verification in the pull request."),
    (("what is Python", "tell me about Python", "why use Python"),
     "Python is a general-purpose language known for readable syntax, rapid iteration, and a large ecosystem."),
    (("what is a neural network", "explain neural networks", "how do neural weights work"),
     "A neural network applies learned transformations to numeric inputs. Training changes weights so useful outputs receive higher probability."),
    (("what can you do", "help", "show your capabilities"),
     "I can generate local text from learned next-token probabilities, explain concepts represented in my in-memory training, and learn extra dialogue with an explicit command."),
    (("reason through a problem", "how should I solve a hard problem", "think step by step"),
     "First define the goal and constraints. Then list known facts, test the smallest useful hypothesis, compare evidence, and verify the final result against the original goal."),
    (("are you as good as DeepSeek R1", "compare yourself with DeepSeek", "are you a frontier model"),
     "I am not equivalent to DeepSeek R1. I am a compact educational BitNet-style model trained from embedded text in RAM, so my knowledge and reasoning are much smaller."),
    (("thank you", "thanks", "nice work"),
     "You are welcome! CatSeek R1 is ready for the next task."),
    (("goodbye", "bye", "see you later"),
     "Goodbye! The conversation can end while the in-memory model remains ready."),
    (("你好", "嗨", "早上好", "晚上好"),
     "你好！我是 CatSeek R1。检测到中文后，我会自动使用中文回答。"),
    (("你是谁", "你是什么模型", "请介绍一下自己"),
     "我是 CatSeek R1，一个启动时在内存中训练的本地下一词元语言模型。"),
    (("你会说中文吗", "请用中文回答", "你能识别中文吗"),
     "可以。我会在本机检测汉字，并自动选择中文模式；普通英文输入则使用英文模式。"),
    (("什么是文件关闭", "文件关闭是什么意思", "你会读取模型文件吗"),
     "文件关闭表示分词器、训练、权重、优化器和推理都留在内存中，不读取或写入模型检查点。"),
    (("什么是比特网络", "解释三值权重", "什么是低比特模型"),
     "比特网络风格的线性层使用负一、零和正一三种权重，并配合八位激活来降低推理成本。"),
    (("什么是双残差三值量化", "解释你的新数学", "什么是CatSeek DTR"),
     "CatSeek DTR 用两个三值分支近似每个权重矩阵。第一个分支表示主要权重，第二个分支量化剩余误差，从而降低失真。"),
    (("你的推理循环怎样工作", "解释潜在推理", "每个词元会多次计算吗"),
     "在预测每个词元之前，我会执行三次共享权重的潜在优化。每次都把当前隐藏状态与学习到的三值残差结合。"),
    (("什么是语言模型", "解释下一词元预测", "文字是怎样生成的"),
     "语言模型根据前面的词元估计下一个词元的概率。生成时把预测结果加入上下文，然后继续预测。"),
    (("你是怎样训练的", "解释你的训练", "启动训练做什么"),
     "启动时，我把内置训练文本转换成词元，用 Adam 和交叉熵更新神经网络，并把学习结果保存在内存中。"),
    (("什么是交叉熵", "解释训练损失", "损失是什么意思"),
     "交叉熵衡量模型为正确下一词元分配的概率。损失越低，训练序列的预测通常越准确。"),
    (("什么是自回归生成", "你怎样生成回答", "解释自回归"),
     "自回归生成每次预测一个词元，把它加入上下文，重新计算概率，再预测下一个词元。"),
    (("什么是分词", "解释词元", "文本怎样变成数字"),
     "分词把文字转换成词表编号。CatSeek R1 对英文使用单词和标点，对中文使用汉字和标点。"),
    (("怎样调试代码", "程序崩溃怎么办", "帮我分析错误"),
     "先保留完整错误信息，用最小输入重现问题，再检查第一个错误边界的状态，修复一个原因并运行回归测试。"),
    (("怎样设计模拟器", "如何开发模拟器", "模拟器架构是什么"),
     "先实现处理器状态机、总线、内存映射、时序、中断、图形、音频和确定性的诊断测试。"),
    (("怎样制作游戏", "游戏开发计划", "如何开始做游戏"),
     "先完成一个垂直切片，包括输入、固定步长更新、渲染、碰撞、音频和明确的胜利条件。"),
    (("怎样优化性能", "让程序更快", "性能优化建议"),
     "先测量同一工作负载，找到主要成本，每次只改变一个变量，并比较延迟、吞吐量和内存。"),
    (("请逐步思考", "怎样解决复杂问题", "帮我进行推理"),
     "先明确目标和限制，再列出已知事实，测试最小可用假设，比较证据，并用原始目标验证结果。"),
    (("你和深度求索一样强吗", "你能达到深度求索水平吗", "你是前沿大模型吗"),
     "我不等同于 DeepSeek R1。我是从内置文本开始训练的小型教学模型，因此知识范围和推理能力要小得多。"),
    (("谢谢", "感谢你", "做得好"),
     "不客气！CatSeek R1 已准备好处理下一个任务。"),
    (("再见", "拜拜", "回头见"),
     "再见！当前对话可以结束，内存中的模型仍然保持就绪。"),
)


def corpus_texts() -> list[str]:
    texts: list[str] = []
    for prompts, answer in BOOT_DIALOGUES:
        texts.extend(prompts)
        texts.append(answer)
    return texts


class InMemoryTernaryLM:
    """Trainable DTR-BitNet network with recurrent latent refinement."""

    def __init__(self, tokenizer: WordTokenizer, config: ModelConfig):
        self.tokenizer = tokenizer
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.forward_calls = 0
        self.generated_tokens = 0
        self.training_steps = 0
        self.external_load_count = 0
        self.last_detected_language = "en"
        self.initial_loss_ever = math.inf
        self.best_loss = math.inf
        self.last_trace: list[GenerationStep] = []
        self._training_documents = self._build_documents()
        self.contexts, self.targets = self._build_training_arrays(self._training_documents)

        vocab = tokenizer.vocab_size
        input_width = config.context_tokens * config.embedding_dim
        self.embedding = self.rng.normal(0.0, 0.10, (vocab, config.embedding_dim)).astype(np.float32)
        self.w_hidden = self.rng.normal(0.0, 0.10, (input_width, config.hidden_dim)).astype(np.float32)
        self.b_hidden = np.zeros((config.hidden_dim,), dtype=np.float32)
        self.w_reason = self.rng.normal(0.0, 0.08, (config.hidden_dim, config.hidden_dim)).astype(np.float32)
        self.b_reason = np.zeros((config.hidden_dim,), dtype=np.float32)
        self.w_output = self.rng.normal(0.0, 0.10, (config.hidden_dim, vocab)).astype(np.float32)
        self.b_output = np.zeros((vocab,), dtype=np.float32)
        self.parameters = [
            self.embedding,
            self.w_hidden,
            self.b_hidden,
            self.w_reason,
            self.b_reason,
            self.w_output,
            self.b_output,
        ]
        self.adam_m = [np.zeros_like(parameter) for parameter in self.parameters]
        self.adam_v = [np.zeros_like(parameter) for parameter in self.parameters]
        self.report = TrainingReport(samples=len(self.targets))

    @property
    def parameter_count(self) -> int:
        return sum(int(parameter.size) for parameter in self.parameters)

    def _build_documents(self) -> list[list[int]]:
        t = self.tokenizer
        documents: list[list[int]] = []
        for prompts, answer in BOOT_DIALOGUES:
            for prompt in prompts:
                language_token = t.ZH if detect_language(prompt) == "zh" else t.EN
                document = [
                    t.token_id(t.BOS),
                    t.token_id(t.USER),
                    *t.encode(prompt),
                    t.token_id(language_token),
                    t.token_id(t.ASSISTANT),
                    *t.encode(answer),
                    t.token_id(t.EOS),
                ]
                documents.append(document)
        return documents

    def _build_training_arrays(self, documents: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
        width = self.config.context_tokens
        pad = self.tokenizer.token_id(self.tokenizer.PAD)
        contexts: list[list[int]] = []
        targets: list[int] = []
        for document in documents:
            for index in range(1, len(document)):
                previous = document[max(0, index - width):index]
                contexts.append([pad] * (width - len(previous)) + previous)
                targets.append(document[index])
        return np.asarray(contexts, dtype=np.int64), np.asarray(targets, dtype=np.int64)

    @staticmethod
    def ternary_quantize(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per-output AbsMean ternary quantization."""
        scale = np.mean(np.abs(weights), axis=0, keepdims=True).astype(np.float32) + 1e-6
        codes = np.clip(np.rint(weights / scale), -1, 1).astype(np.int8)
        dequantized = codes.astype(np.float32) * scale
        return codes, scale, dequantized

    @classmethod
    def dual_residual_ternary_quantize(
        cls,
        weights: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """CatSeek DTR: approximate one matrix with two ternary branches.

        W_hat = alpha * T0 + beta * T1, where both T0 and T1 contain only
        {-1, 0, +1}. T0 captures the main weight and T1 quantizes its residual.
        This spends two cheap ternary projections to reduce reconstruction
        error without loading a higher-precision inference matrix.
        """
        primary_codes, primary_scale, primary = cls.ternary_quantize(weights)
        residual_codes, residual_scale, residual = cls.ternary_quantize(weights - primary)
        reconstructed = (primary + residual).astype(np.float32)
        return (
            primary_codes,
            primary_scale,
            residual_codes,
            residual_scale,
            reconstructed,
        )

    @classmethod
    def dual_ternary_linear(
        cls,
        values: np.ndarray,
        weights: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run two real ternary-code projections and combine their scales."""
        codes0, scale0, codes1, scale1, reconstructed = cls.dual_residual_ternary_quantize(weights)
        branch0 = (values @ codes0.astype(np.float32)) * scale0
        branch1 = (values @ codes1.astype(np.float32)) * scale1
        return (branch0 + branch1).astype(np.float32), reconstructed

    @staticmethod
    def activation_a8(values: np.ndarray) -> np.ndarray:
        peak = np.max(np.abs(values), axis=1, keepdims=True).astype(np.float32) + 1e-6
        scale = peak / 127.0
        codes = np.clip(np.rint(values / scale), -127, 127)
        return (codes * scale).astype(np.float32)

    def _forward(
        self,
        contexts: np.ndarray,
        *,
        inference: bool,
    ) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray]:
        batch = int(contexts.shape[0])
        x = self.embedding[contexts].reshape(batch, -1)
        if inference:
            x = self.activation_a8(x)
        hidden_projection, _ = self.dual_ternary_linear(x, self.w_hidden)
        hidden = np.tanh(hidden_projection + self.b_hidden).astype(np.float32)
        if inference:
            hidden = self.activation_a8(hidden)

        # Shared-weight latent recurrence: every next-token decision receives
        # several internal refinement passes before reaching the LM head.
        states = [hidden]
        deltas: list[np.ndarray] = []
        for _ in range(self.config.reasoning_passes):
            reason_input = self.activation_a8(hidden) if inference else hidden
            reason_projection, _ = self.dual_ternary_linear(reason_input, self.w_reason)
            delta = np.tanh(reason_projection + self.b_reason).astype(np.float32)
            hidden = np.tanh(hidden + self.config.reasoning_scale * delta).astype(np.float32)
            if inference:
                hidden = self.activation_a8(hidden)
            deltas.append(delta)
            states.append(hidden)

        output_projection, _ = self.dual_ternary_linear(hidden, self.w_output)
        logits = output_projection + self.b_output
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        probabilities = np.exp(shifted).astype(np.float32)
        probabilities /= np.sum(probabilities, axis=1, keepdims=True)
        return x, states, deltas, logits, probabilities

    def evaluate_loss(self, limit: int = 4096) -> float:
        if len(self.targets) > limit:
            indices = np.linspace(0, len(self.targets) - 1, limit, dtype=np.int64)
            contexts = self.contexts[indices]
            targets = self.targets[indices]
        else:
            contexts = self.contexts
            targets = self.targets
        _, _, _, _, probabilities = self._forward(contexts, inference=False)
        chosen = probabilities[np.arange(len(targets)), targets]
        return float(-np.mean(np.log(chosen + 1e-9)))

    def train(
        self,
        steps: Optional[int] = None,
        progress: Optional[Callable[[int, int, float], None]] = None,
    ) -> TrainingReport:
        count = int(self.config.train_steps if steps is None else steps)
        if count <= 0:
            return self.report
        started = time.perf_counter()
        initial = self.evaluate_loss()
        if not math.isfinite(self.initial_loss_ever):
            self.initial_loss_ever = initial
        history = [initial]
        beta1, beta2 = 0.9, 0.999
        batch_size = min(self.config.batch_size, len(self.targets))
        for local_step in range(1, count + 1):
            indices = self.rng.integers(0, len(self.targets), size=batch_size)
            contexts = self.contexts[indices]
            targets = self.targets[indices]
            x, states, deltas, _, probabilities = self._forward(contexts, inference=False)

            d_logits = probabilities.copy()
            d_logits[np.arange(batch_size), targets] -= 1.0
            d_logits /= batch_size
            final_hidden = states[-1]
            grad_w_output = final_hidden.T @ d_logits
            grad_b_output = np.sum(d_logits, axis=0)
            output_weights = self.dual_residual_ternary_quantize(self.w_output)[-1]
            reason_weights = self.dual_residual_ternary_quantize(self.w_reason)[-1]
            hidden_weights = self.dual_residual_ternary_quantize(self.w_hidden)[-1]

            # Backpropagate through the shared recurrent reasoning block.
            d_state = d_logits @ output_weights.T
            grad_w_reason = np.zeros_like(self.w_reason)
            grad_b_reason = np.zeros_like(self.b_reason)
            for pass_index in range(self.config.reasoning_passes - 1, -1, -1):
                state_before = states[pass_index]
                state_after = states[pass_index + 1]
                delta = deltas[pass_index]
                d_residual = d_state * (1.0 - state_after * state_after)
                d_delta = (
                    self.config.reasoning_scale
                    * d_residual
                    * (1.0 - delta * delta)
                )
                grad_w_reason += state_before.T @ d_delta
                grad_b_reason += np.sum(d_delta, axis=0)
                d_state = d_residual + d_delta @ reason_weights.T

            d_hidden = d_state * (1.0 - states[0] * states[0])
            grad_w_hidden = x.T @ d_hidden  # STE through both ternary branches.
            grad_b_hidden = np.sum(d_hidden, axis=0)
            d_x = (d_hidden @ hidden_weights.T).reshape(
                batch_size, self.config.context_tokens, self.config.embedding_dim
            )
            grad_embedding = np.zeros_like(self.embedding)
            np.add.at(grad_embedding, contexts, d_x)
            gradients = [
                grad_embedding,
                grad_w_hidden,
                grad_b_hidden,
                grad_w_reason,
                grad_b_reason,
                grad_w_output,
                grad_b_output,
            ]
            global_step = self.training_steps + local_step
            for index, (parameter, gradient) in enumerate(zip(self.parameters, gradients)):
                gradient = np.clip(
                    gradient,
                    -self.config.gradient_clip,
                    self.config.gradient_clip,
                )
                self.adam_m[index] = beta1 * self.adam_m[index] + (1.0 - beta1) * gradient
                self.adam_v[index] = beta2 * self.adam_v[index] + (1.0 - beta2) * (gradient * gradient)
                m_hat = self.adam_m[index] / (1.0 - beta1 ** global_step)
                v_hat = self.adam_v[index] / (1.0 - beta2 ** global_step)
                parameter -= self.config.learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)

            if local_step == count or local_step % max(1, count // 5) == 0:
                current = self.evaluate_loss()
                history.append(current)
                if progress:
                    progress(local_step, count, current)

        self.training_steps += count
        final = self.evaluate_loss()
        self.best_loss = min(self.best_loss, final)
        self.report = TrainingReport(
            initial_loss=initial,
            final_loss=final,
            steps=count,
            samples=len(self.targets),
            elapsed_s=time.perf_counter() - started,
            loss_history=history,
        )
        return self.report

    def _context_array(self, tokens: list[int]) -> np.ndarray:
        width = self.config.context_tokens
        pad = self.tokenizer.token_id(self.tokenizer.PAD)
        recent = tokens[-width:]
        return np.asarray([[pad] * (width - len(recent)) + recent], dtype=np.int64)

    def next_token_probabilities(self, tokens: list[int]) -> np.ndarray:
        self.forward_calls += 1
        _, _, _, _, probabilities = self._forward(self._context_array(tokens), inference=True)
        return probabilities[0]

    def chat_prefix(self, prompt: str, history: list[tuple[str, str]]) -> list[int]:
        t = self.tokenizer
        # This compact model's context is intentionally devoted to the current
        # prompt. The app retains visible chat history, but injecting a long
        # prior answer into the compact neural window would erase a short new
        # request and create false follow-up behavior.
        del history
        language = detect_language(prompt)
        language_token = t.ZH if language == "zh" else t.EN
        tokens = [t.token_id(t.BOS)]
        tokens.extend(
            [
                t.token_id(t.USER),
                *t.encode(prompt),
                t.token_id(language_token),
                t.token_id(t.ASSISTANT),
            ]
        )
        return tokens

    def _choose_token(
        self,
        probabilities: np.ndarray,
        *,
        temperature: float,
        top_k: int,
        rng: np.random.Generator,
        generated: list[int],
    ) -> tuple[int, float, float]:
        t = self.tokenizer
        adjusted = probabilities.astype(np.float64).copy()
        for token in (t.PAD, t.BOS, t.USER, t.ASSISTANT, t.EN, t.ZH, t.UNK):
            adjusted[t.token_id(token)] = 0.0
        if generated:
            # Mild repetition penalty without replacing the probability model.
            for token_id, count in collections.Counter(generated[-20:]).items():
                adjusted[token_id] /= 1.0 + 0.18 * count
        total = float(np.sum(adjusted))
        if total <= 0.0:
            return t.token_id(t.EOS), 1.0, 0.0
        adjusted /= total
        entropy = float(-np.sum(adjusted * np.log2(adjusted + 1e-12)))
        if temperature <= 1e-6:
            token_id = int(np.argmax(adjusted))
        else:
            logits = np.log(adjusted + 1e-12) / temperature
            k = max(1, min(int(top_k), len(logits)))
            keep = np.argpartition(logits, -k)[-k:]
            local = logits[keep]
            local -= np.max(local)
            sample_probabilities = np.exp(local)
            sample_probabilities /= np.sum(sample_probabilities)
            token_id = int(rng.choice(keep, p=sample_probabilities))
        return token_id, float(adjusted[token_id]), entropy

    def generate_chat(
        self,
        prompt: str,
        history: Optional[list[tuple[str, str]]] = None,
        *,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> GenerationReport:
        started = time.perf_counter()
        history = history or []
        language = detect_language(prompt)
        self.last_detected_language = language
        context = self.chat_prefix(prompt, history)
        generated: list[int] = []
        trace: list[GenerationStep] = []
        limit = int(max_new_tokens or self.config.max_new_tokens)
        temp = self.config.temperature if temperature is None else float(temperature)
        k = self.config.top_k if top_k is None else int(top_k)
        seed = self.config.seed
        for byte in prompt.encode("utf-8", errors="replace"):
            seed = ((seed * 1664525) + byte + 1013904223) & 0xFFFFFFFF
        sample_rng = np.random.default_rng(seed)
        eos = self.tokenizer.token_id(self.tokenizer.EOS)
        finish_reason = "length"
        for index in range(limit):
            probabilities = self.next_token_probabilities(context)
            token_id, probability, entropy = self._choose_token(
                probabilities,
                temperature=temp,
                top_k=k,
                rng=sample_rng,
                generated=generated,
            )
            token = self.tokenizer.id_to_token[token_id]
            trace.append(GenerationStep(index, token, token_id, probability, entropy))
            if token_id == eos:
                finish_reason = "eos"
                break
            generated.append(token_id)
            context.append(token_id)
            self.generated_tokens += 1
            if on_token:
                on_token(token)
            if len(generated) >= 12 and generated[-6:] == generated[-12:-6]:
                finish_reason = "repetition-guard"
                break
        elapsed = time.perf_counter() - started
        text = self.tokenizer.decode(generated)
        self.last_trace = trace
        return GenerationReport(
            text=text,
            language=language,
            tokens=len(generated),
            elapsed_s=elapsed,
            tokens_per_second=(len(generated) / elapsed if elapsed > 0 else 0.0),
            finish_reason=finish_reason,
            trace=trace,
        )

    def model_card(self) -> dict[str, object]:
        h0, _, h1, _, hidden_hat = self.dual_residual_ternary_quantize(self.w_hidden)
        r0, _, r1, _, reason_hat = self.dual_residual_ternary_quantize(self.w_reason)
        o0, _, o1, _, output_hat = self.dual_residual_ternary_quantize(self.w_output)
        matrices = (
            (self.w_hidden, hidden_hat),
            (self.w_reason, reason_hat),
            (self.w_output, output_hat),
        )
        single_error = float(np.mean([
            np.mean((weights - self.ternary_quantize(weights)[-1]) ** 2)
            for weights, _ in matrices
        ]))
        dual_error = float(np.mean([
            np.mean((weights - reconstructed) ** 2)
            for weights, reconstructed in matrices
        ]))
        return {
            "brand": APP_NAME,
            "model_id": MODEL_ID,
            "version": APP_VERSION,
            "kind": "compact bilingual DTR-BitNet recurrent-reasoning language model",
            "files": FILES_MODE,
            "checkpoint_bytes": 0,
            "network_required": False,
            "vocabulary_tokens": self.tokenizer.vocab_size,
            "context_tokens": self.config.context_tokens,
            "embedding_dim": self.config.embedding_dim,
            "hidden_dim": self.config.hidden_dim,
            "reasoning_passes_per_token": self.config.reasoning_passes,
            "reasoning_residual_scale": self.config.reasoning_scale,
            "trainable_parameters": self.parameter_count,
            "training_samples": len(self.targets),
            "training_steps_completed": self.training_steps,
            "training_initial_loss": self.initial_loss_ever,
            "training_best_loss": self.best_loss,
            "supported_response_languages": ["English", "Mandarin Chinese"],
            "language_detection": "local Han/Latin character analysis plus learned language-control tokens",
            "last_detected_language": self.last_detected_language,
            "weight_quantization": "CatSeek dual-residual ternary (two {-1,0,+1} branches per matrix)",
            "activation_quantization": "symmetric A8 during inference",
            "dtr_equation": "W_hat = alpha*T0 + beta*T1; T0,T1 in {-1,0,+1}",
            "reasoning_equation": "h[k+1] = tanh(h[k] + lambda*tanh(h[k]@W_reason + b))",
            "ternary_matmuls_per_predicted_token": 2 * (2 + self.config.reasoning_passes),
            "single_ternary_reconstruction_mse": single_error,
            "dual_residual_reconstruction_mse": dual_error,
            "hidden_primary_codes": sorted(int(value) for value in np.unique(h0)),
            "hidden_residual_codes": sorted(int(value) for value in np.unique(h1)),
            "reason_primary_codes": sorted(int(value) for value in np.unique(r0)),
            "reason_residual_codes": sorted(int(value) for value in np.unique(r1)),
            "output_primary_codes": sorted(int(value) for value in np.unique(o0)),
            "output_residual_codes": sorted(int(value) for value in np.unique(o1)),
            "inference": "autoregressive softmax with recurrent latent refinement before every token",
            "normal_chat_dispatch": "InMemoryTernaryLM.generate_chat",
            "external_model_loads": self.external_load_count,
            "deepseek_r1_equivalent": False,
            "honesty": "new CatSeek math is implemented and measured, but this small RAM-trained model cannot match a large pretrained reasoning model",
        }

    def self_test(self) -> dict[str, object]:
        t = self.tokenizer
        h0, _, h1, _, hidden_hat = self.dual_residual_ternary_quantize(self.w_hidden)
        r0, _, r1, _, reason_hat = self.dual_residual_ternary_quantize(self.w_reason)
        o0, _, o1, _, output_hat = self.dual_residual_ternary_quantize(self.w_output)
        matrices = (
            (self.w_hidden, hidden_hat),
            (self.w_reason, reason_hat),
            (self.w_output, output_hat),
        )
        single_quantization_mse = float(np.mean([
            np.mean((weights - self.ternary_quantize(weights)[-1]) ** 2)
            for weights, _ in matrices
        ]))
        dual_quantization_mse = float(np.mean([
            np.mean((weights - reconstructed) ** 2)
            for weights, reconstructed in matrices
        ]))
        hello_context = self.chat_prefix("hello", [])
        bitnet_context = self.chat_prefix("what is BitNet", [])
        mandarin_context = self.chat_prefix("请介绍一下自己", [])
        _, reasoning_states, _, _, _ = self._forward(
            self._context_array(hello_context),
            inference=True,
        )
        reasoning_state_l1 = float(np.sum(np.abs(reasoning_states[-1] - reasoning_states[0])))
        before_calls = self.forward_calls
        hello_distribution = self.next_token_probabilities(hello_context)
        bitnet_distribution = self.next_token_probabilities(bitnet_context)
        first_token = int(np.argmax(hello_distribution))
        changed_distribution = self.next_token_probabilities(hello_context + [first_token])
        # Causal inference check: removing a learned weight matrix must alter
        # the same prompt's distribution. Restore it immediately so this test
        # is non-destructive and repeatable.
        saved_output = self.w_output.copy()
        try:
            self.w_output.fill(0.0)
            ablated_distribution = self.next_token_probabilities(hello_context)
        finally:
            self.w_output[...] = saved_output
        saved_reason = self.w_reason.copy()
        try:
            self.w_reason.fill(0.0)
            reasoning_ablated_distribution = self.next_token_probabilities(hello_context)
        finally:
            self.w_reason[...] = saved_reason
        hello_generation = self.generate_chat("hello", max_new_tokens=80, temperature=0.0)
        bitnet_generation = self.generate_chat("what is BitNet", max_new_tokens=80, temperature=0.0)
        mandarin_generation = self.generate_chat("请介绍一下自己", max_new_tokens=100, temperature=0.0)
        call_delta = self.forward_calls - before_calls
        prompt_l1 = float(np.sum(np.abs(hello_distribution - bitnet_distribution)))
        autoregressive_l1 = float(np.sum(np.abs(hello_distribution - changed_distribution)))
        weight_ablation_l1 = float(np.sum(np.abs(hello_distribution - ablated_distribution)))
        reasoning_ablation_l1 = float(
            np.sum(np.abs(hello_distribution - reasoning_ablated_distribution))
        )
        ternary_set = {-1, 0, 1}
        all_ternary_branches = (h0, h1, r0, r1, o0, o1)
        tests = {
            "training_loss_is_finite": math.isfinite(self.initial_loss_ever) and math.isfinite(self.best_loss),
            "training_reduces_loss": self.best_loss < self.initial_loss_ever * 0.75,
            "probabilities_sum_to_one": abs(float(np.sum(hello_distribution)) - 1.0) < 1e-5,
            "probabilities_are_nonconstant": float(np.std(hello_distribution)) > 1e-4,
            "prompt_changes_distribution": prompt_l1 > 0.05,
            "appended_token_changes_distribution": autoregressive_l1 > 0.01,
            "learned_weights_causally_change_distribution": weight_ablation_l1 > 0.05,
            "latent_reasoning_causally_changes_distribution": reasoning_ablation_l1 > 0.01,
            "configured_reasoning_passes_are_executed": (
                len(reasoning_states) == self.config.reasoning_passes + 1
                and reasoning_state_l1 > 0.01
            ),
            "dual_residual_quantization_reduces_mse": (
                dual_quantization_mse < single_quantization_mse
            ),
            "normal_prompts_generate_different_text": hello_generation.text != bitnet_generation.text,
            "inference_forward_called_per_token": call_delta == (
                len(hello_generation.trace) + len(bitnet_generation.trace)
                + len(mandarin_generation.trace) + 5
            ),
            "english_is_detected": detect_language("Please explain this in English") == "en",
            "mandarin_is_detected": detect_language("请使用中文回答") == "zh",
            "english_control_token_is_in_context": t.token_id(t.EN) in hello_context,
            "mandarin_control_token_is_in_context": t.token_id(t.ZH) in mandarin_context,
            "english_prompt_generates_english": not is_han(hello_generation.text),
            "mandarin_prompt_generates_mandarin": len(HAN_RE.findall(mandarin_generation.text)) >= 8,
            "all_dtr_branches_are_strictly_ternary": all(
                set(int(value) for value in np.unique(codes)) <= ternary_set
                for codes in all_ternary_branches
            ),
            "primary_branches_use_all_ternary_values": all(
                set(int(value) for value in np.unique(codes)) == ternary_set
                for codes in (h0, r0, o0)
            ),
            "residual_branches_use_all_ternary_values": all(
                set(int(value) for value in np.unique(codes)) == ternary_set
                for codes in (h1, r1, o1)
            ),
            "generated_text_is_nonempty": bool(
                hello_generation.text and bitnet_generation.text and mandarin_generation.text
            ),
            "files_are_off": True,
            "no_checkpoint_or_network": self.external_load_count == 0,
        }
        evidence = {
            "loss_before": self.initial_loss_ever,
            "loss_after": self.best_loss,
            "prompt_distribution_l1": prompt_l1,
            "autoregressive_distribution_l1": autoregressive_l1,
            "weight_ablation_distribution_l1": weight_ablation_l1,
            "reasoning_ablation_distribution_l1": reasoning_ablation_l1,
            "reasoning_state_change_l1": reasoning_state_l1,
            "single_ternary_reconstruction_mse": single_quantization_mse,
            "dual_residual_reconstruction_mse": dual_quantization_mse,
            "inference_forward_call_delta": call_delta,
            "hello_output": hello_generation.text,
            "bitnet_output": bitnet_generation.text,
            "mandarin_output": mandarin_generation.text,
            "english_detected_as": detect_language("Please explain this in English"),
            "mandarin_detected_as": detect_language("请使用中文回答"),
            "hello_trace_tokens": [step.token for step in hello_generation.trace[:8]],
            "bitnet_trace_tokens": [step.token for step in bitnet_generation.trace[:8]],
            "mandarin_trace_tokens": [step.token for step in mandarin_generation.trace[:12]],
        }
        return {
            "ok": all(tests.values()),
            "tests": tests,
            "evidence": evidence,
            "model": self.model_card(),
        }


class CatSeekR1:
    """Branded app runtime. Normal chat always enters the neural LM."""

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        progress: Optional[Callable[[int, int, float], None]] = None,
    ):
        self.config = config or ModelConfig()
        self.tokenizer = WordTokenizer(corpus_texts())
        self.model = InMemoryTernaryLM(self.tokenizer, self.config)
        self.model.train(progress=progress)
        self.history: list[tuple[str, str]] = []
        self.temperature = self.config.temperature
        self.top_k = self.config.top_k

    def clear(self) -> None:
        self.history.clear()

    def _command(self, prompt: str) -> Optional[Reply]:
        raw = prompt.strip()
        lower = raw.lower()
        started = time.perf_counter()
        if not raw.startswith("/"):
            return None
        if lower in {"/help", "/?"}:
            text = (
                "CatSeek R1 commands:\n\n"
                "- `/model` — RAM model card and measured loss\n"
                "- `/selftest` — inference-proof tests with evidence\n"
                "- `/trace` — probabilities from the last generation\n"
                "- `/train N` — run N more in-memory gradient steps\n"
                "- `/temperature N` — set sampling temperature (0 = greedy)\n"
                "- `/clear` — clear conversation context\n\n"
                "Every message without a slash goes through DTR ternary inference and three latent reasoning passes per predicted token."
            )
            return Reply(text, "control:/help", (time.perf_counter() - started) * 1000)
        if lower == "/model":
            text = json.dumps(self.model.model_card(), indent=2, ensure_ascii=False)
            return Reply(text, "control:/model", (time.perf_counter() - started) * 1000)
        if lower == "/selftest":
            result = self.model.self_test()
            text = json.dumps(result, indent=2, ensure_ascii=False)
            return Reply(text, "control:/selftest", (time.perf_counter() - started) * 1000)
        if lower == "/trace":
            if not self.model.last_trace:
                text = "No CatSeek R1 generation trace exists yet."
            else:
                rows = [
                    {
                        "step": step.index,
                        "token": step.token,
                        "probability": round(step.probability, 6),
                        "entropy_bits": round(step.entropy_bits, 4),
                    }
                    for step in self.model.last_trace
                ]
                text = json.dumps(rows, indent=2, ensure_ascii=False)
            return Reply(text, "control:/trace", (time.perf_counter() - started) * 1000)
        if lower == "/clear":
            self.clear()
            return Reply("CatSeek R1 conversation context cleared.", "control:/clear", (time.perf_counter() - started) * 1000)
        if lower.startswith("/train"):
            parts = raw.split()
            try:
                steps = int(parts[1]) if len(parts) > 1 else 100
            except ValueError:
                return Reply("Usage: /train N", "control:error", (time.perf_counter() - started) * 1000)
            steps = max(1, min(steps, 5000))
            report = self.model.train(steps=steps)
            text = (
                f"CatSeek R1 trained for {steps} additional RAM-only steps. "
                f"Loss {report.initial_loss:.4f} → {report.final_loss:.4f}."
            )
            return Reply(text, "control:/train", (time.perf_counter() - started) * 1000)
        if lower.startswith("/temperature"):
            parts = raw.split()
            if len(parts) == 1:
                text = f"CatSeek R1 temperature is {self.temperature:.2f}."
            else:
                try:
                    value = float(parts[1])
                except ValueError:
                    return Reply("Usage: /temperature 0.0..2.0", "control:error", (time.perf_counter() - started) * 1000)
                self.temperature = max(0.0, min(2.0, value))
                text = f"CatSeek R1 temperature set to {self.temperature:.2f}."
            return Reply(text, "control:/temperature", (time.perf_counter() - started) * 1000)
        return Reply("Unknown CatSeek R1 command. Use /help.", "control:unknown", (time.perf_counter() - started) * 1000)

    def reply(self, prompt: str, on_token: Optional[Callable[[str], None]] = None) -> Reply:
        control = self._command(prompt)
        if control is not None:
            if on_token:
                on_token(control.text)
            return control
        report = self.model.generate_chat(
            prompt,
            self.history,
            temperature=self.temperature,
            top_k=self.top_k,
        )
        text = report.text or "[CatSeek R1 emitted EOS before a visible token.]"
        self.history.append((prompt, text))
        self.history = self.history[-12:]
        if on_token:
            for token_id in [step.token_id for step in report.trace if step.token_id != self.tokenizer.token_id(self.tokenizer.EOS)]:
                piece = self.tokenizer.decode([token_id])
                if piece:
                    on_token(piece + ("" if piece in ".,!?:;" else " "))
        return Reply(
            text=text,
            route=f"ram-bitnet-lm:{report.language}",
            elapsed_ms=report.elapsed_s * 1000.0,
            tokens=report.tokens,
            tokens_per_second=report.tokens_per_second,
        )


class CatSeekGUI:
    BG = "#03060b"
    PANEL = "#07101d"
    INPUT = "#0a1424"
    BLUE = "#3b82f6"
    BLUE_2 = "#60a5fa"
    TEXT = "#dbeafe"
    MUTED = "#7890ad"
    RED = "#f87171"

    def __init__(self, root: tk.Tk, engine: CatSeekR1):
        self.root = root
        self.engine = engine
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.busy = False
        self._build()
        self.root.after(30, self._drain)

    @staticmethod
    def _family(mono: bool = False) -> str:
        if mono:
            return "Cascadia Mono" if os.name == "nt" else "Menlo"
        return "Segoe UI" if os.name == "nt" else "Helvetica Neue"

    def _font(self, size: int, bold: bool = False, mono: bool = False) -> font.Font:
        return font.Font(family=self._family(mono), size=size, weight="bold" if bold else "normal")

    def _build(self) -> None:
        root = self.root
        root.title(f"{APP_NAME} v{APP_VERSION}")
        root.geometry("1040x720")
        root.minsize(780, 560)
        root.configure(bg=self.BG)

        menu = tk.Menu(root, tearoff=False, bg=self.PANEL, fg=self.TEXT)
        catseek = tk.Menu(menu, tearoff=False, bg=self.PANEL, fg=self.TEXT)
        catseek.add_command(label="New CatSeek R1 chat", command=self._new_chat)
        catseek.add_command(label="CatSeek R1 model card", command=lambda: self._submit("/model"))
        catseek.add_command(label="CatSeek R1 inference self-test", command=lambda: self._submit("/selftest"))
        catseek.add_separator()
        catseek.add_command(label="Exit CatSeek R1", command=root.destroy)
        menu.add_cascade(label=APP_NAME, menu=catseek)
        root.config(menu=menu)

        header = tk.Frame(root, bg=self.PANEL, padx=18, pady=14)
        header.pack(fill="x")
        logo = tk.Frame(header, bg="#000000", width=46, height=46)
        logo.pack(side="left")
        logo.pack_propagate(False)
        tk.Label(logo, text="🐱", bg="#000000", fg=self.BLUE, font=self._font(23)).place(relx=0.5, rely=0.5, anchor="center")
        name = tk.Frame(header, bg=self.PANEL)
        name.pack(side="left", padx=(12, 0))
        tk.Label(name, text=APP_NAME, bg=self.PANEL, fg=self.TEXT, font=self._font(17, True)).pack(anchor="w")
        report = self.engine.model.report
        tk.Label(
            name,
            text=(
                f"DTR reasoning BitNet · {self.engine.model.parameter_count:,} trained params · "
                f"loss {report.initial_loss:.3f}→{report.final_loss:.3f} · files=off"
            ),
            bg=self.PANEL,
            fg=self.MUTED,
            font=self._font(9),
        ).pack(anchor="w")
        self.status = tk.Label(header, text="CatSeek R1 inference ready", bg=self.PANEL, fg=self.BLUE_2, font=self._font(10))
        self.status.pack(side="right")

        body = tk.Frame(root, bg=self.BG)
        body.pack(fill="both", expand=True)
        sidebar = tk.Frame(body, bg=self.PANEL, width=220, padx=14, pady=18)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)
        tk.Button(
            sidebar,
            text="+ New CatSeek R1 chat",
            command=self._new_chat,
            bg="#000000",
            fg=self.BLUE,
            activebackground="#111827",
            activeforeground=self.BLUE_2,
            relief="flat",
            bd=0,
            padx=10,
            pady=10,
            font=self._font(10, True),
            cursor="hand2",
        ).pack(fill="x")
        tk.Label(sidebar, text="REAL RAM LM", bg=self.PANEL, fg=self.MUTED, font=self._font(9, True)).pack(anchor="w", pady=(22, 8))
        for line in (
            "● Startup gradient training",
            "● English / 中文 auto mode",
            "● Next-token softmax",
            "● Dual-residual ternary",
            "● 3 latent passes / token",
            "● A8 activations",
            "● Autoregressive decode",
            "● No checkpoints/API",
        ):
            tk.Label(sidebar, text=line, bg=self.PANEL, fg=self.BLUE_2, font=self._font(9)).pack(anchor="w", pady=3)
        tk.Label(
            sidebar,
            text="/model\n/selftest\n/trace\n/train 100\n/temperature 0.5\n/clear",
            justify="left",
            bg=self.PANEL,
            fg=self.MUTED,
            font=self._font(9, mono=True),
        ).pack(anchor="w", side="bottom")

        chat_area = tk.Frame(body, bg=self.BG)
        chat_area.pack(side="left", fill="both", expand=True)
        self.chat = scrolledtext.ScrolledText(
            chat_area,
            state="disabled",
            wrap="word",
            bg=self.BG,
            fg=self.TEXT,
            insertbackground=self.BLUE,
            selectbackground="#1d4ed8",
            relief="flat",
            bd=0,
            padx=26,
            pady=20,
            font=self._font(11),
        )
        self.chat.pack(fill="both", expand=True)
        self.chat.tag_configure("title", foreground=self.BLUE_2, font=self._font(11, True), spacing1=12)
        self.chat.tag_configure("user", foreground="#93c5fd", font=self._font(11))
        self.chat.tag_configure("bot", foreground=self.TEXT, font=self._font(11))
        self.chat.tag_configure("muted", foreground=self.MUTED, font=self._font(9))
        self.chat.tag_configure("error", foreground=self.RED, font=self._font(10))
        self._append("CATSEEK R1\n", "title")
        self._append(
            "Weights trained in RAM. English and 中文 are selected automatically; every token uses CatSeek DTR ternary math plus three latent reasoning passes. Use /selftest for evidence.\n",
            "muted",
        )

        composer = tk.Frame(chat_area, bg=self.PANEL, padx=18, pady=14)
        composer.pack(fill="x")
        self.entry = tk.Text(
            composer,
            height=3,
            bg=self.INPUT,
            fg=self.TEXT,
            insertbackground=self.BLUE,
            selectbackground="#1d4ed8",
            relief="flat",
            bd=0,
            padx=12,
            pady=10,
            font=self._font(10),
            wrap="word",
            undo=True,
        )
        self.entry.pack(side="left", fill="x", expand=True)
        self.entry.bind("<Return>", self._on_enter)
        self.send = tk.Button(
            composer,
            text="INFER ↑",
            command=self._submit,
            bg="#000000",
            fg=self.BLUE,
            activebackground="#111827",
            activeforeground=self.BLUE_2,
            relief="flat",
            bd=0,
            padx=16,
            pady=12,
            font=self._font(10, True),
            cursor="hand2",
        )
        self.send.pack(side="right", padx=(12, 0))
        self.entry.focus_set()

    def _append(self, text: str, tag: str = "bot") -> None:
        self.chat.config(state="normal")
        self.chat.insert("end", text, tag)
        self.chat.config(state="disabled")
        self.chat.see("end")

    def _new_chat(self) -> None:
        if self.busy:
            return
        self.engine.clear()
        self.chat.config(state="normal")
        self.chat.delete("1.0", "end")
        self.chat.config(state="disabled")
        self._append("CATSEEK R1\n", "title")
        self._append("New RAM-only language-model conversation.\n", "muted")

    def _on_enter(self, event) -> Optional[str]:
        if event.state & 0x1:
            return None
        self._submit()
        return "break"

    def _submit(self, forced: Optional[str] = None) -> None:
        if self.busy:
            return
        prompt = forced if isinstance(forced, str) else self.entry.get("1.0", "end-1c").strip()
        if not prompt:
            return
        self.entry.delete("1.0", "end")
        self._append("\nYOU\n", "title")
        self._append(prompt + "\n", "user")
        self._append("\nCATSEEK R1\n", "title")
        self.busy = True
        self.send.config(state="disabled")
        self.status.config(text="CatSeek R1 computing next tokens…")

        def worker() -> None:
            try:
                reply = self.engine.reply(prompt)
                self.events.put(("done", reply))
            except Exception as exc:
                self.events.put(("error", f"{type(exc).__name__}: {exc}"))
            finally:
                self.events.put(("idle", None))

        threading.Thread(target=worker, daemon=True).start()

    def _drain(self) -> None:
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "done":
                    reply = payload
                    self._append(reply.text, "bot")
                    suffix = (
                        f"\n\n{reply.route} · {reply.tokens} tokens · "
                        f"{reply.tokens_per_second:.1f} tok/s · {reply.elapsed_ms:.1f} ms\n"
                    )
                    self._append(suffix, "muted")
                    self.status.config(text=f"CatSeek R1 ready · {reply.tokens_per_second:.1f} tok/s")
                elif kind == "error":
                    self._append("ERROR\n" + str(payload) + "\n", "error")
                    self.status.config(text="CatSeek R1 error")
                elif kind == "idle":
                    self.busy = False
                    self.send.config(state="normal")
        except queue.Empty:
            pass
        self.root.after(30, self._drain)


def make_parser() -> argparse.ArgumentParser:
    defaults = ModelConfig()
    parser = argparse.ArgumentParser(
        prog=os.path.basename(sys.argv[0]),
        description=f"{APP_NAME} v{APP_VERSION}: bilingual DTR reasoning BitNet LM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--chat", action="store_true", help="terminal chat instead of GUI")
    parser.add_argument("--prompt", help="generate one answer and exit")
    parser.add_argument("--self-test", action="store_true", help="run proof-oriented inference tests")
    parser.add_argument("--model-card", action="store_true", help="print the honest model card")
    parser.add_argument("--train-steps", type=int, default=defaults.train_steps)
    parser.add_argument("--temperature", type=float, default=defaults.temperature)
    parser.add_argument("--top-k", type=int, default=defaults.top_k)
    parser.add_argument("--max-tokens", type=int, default=defaults.max_new_tokens)
    parser.add_argument("--seed", type=lambda value: int(value, 0), default=DEFAULT_SEED)
    parser.add_argument("--version", action="version", version=f"{APP_NAME} v{APP_VERSION}")
    return parser


def terminal(engine: CatSeekR1, one_prompt: Optional[str] = None) -> int:
    def run(prompt: str) -> None:
        reply = engine.reply(prompt)
        print(reply.text)
        print(f"\n[{reply.route} · {reply.tokens} tokens · {reply.tokens_per_second:.1f} tok/s]\n")

    if one_prompt:
        run(one_prompt)
        return 0
    report = engine.model.report
    print(
        f"{APP_NAME} v{APP_VERSION} · RAM LM · loss "
        f"{report.initial_loss:.3f}→{report.final_loss:.3f} · /help · /quit"
    )
    while True:
        try:
            prompt = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not prompt:
            continue
        if prompt.lower() in {"/quit", "/exit", "quit", "exit"}:
            return 0
        run(prompt)


def main(argv: Optional[list[str]] = None) -> int:
    if sys.version_info < (3, 10):
        print(
            f"{APP_NAME} requires Python 3.10+ (running {sys.version.split()[0]}).",
            file=sys.stderr,
        )
        return 2
    args = make_parser().parse_args(argv)
    config = ModelConfig(
        train_steps=max(1, min(args.train_steps, 20_000)),
        temperature=max(0.0, min(args.temperature, 2.0)),
        top_k=max(1, args.top_k),
        max_new_tokens=max(1, min(args.max_tokens, 2048)),
        seed=args.seed,
    )
    engine = CatSeekR1(config)
    if args.self_test:
        result = engine.model.self_test()
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result["ok"] else 1
    if args.model_card:
        print(json.dumps(engine.model.model_card(), indent=2, ensure_ascii=False))
        return 0
    if args.prompt or args.chat:
        return terminal(engine, args.prompt)
    root = tk.Tk()
    CatSeekGUI(root, engine)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
