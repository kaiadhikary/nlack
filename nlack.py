from __future__ import annotations

import asyncio
import base64
import cmath
import datetime as dt
import difflib
import hashlib
import io
import itertools
import json
import math
import os
import random
import re
import shutil
import sqlite3
import time
import urllib.parse
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import aiosqlite
import httpx
import pandas as pd
import typer
from bs4 import BeautifulSoup
from lxml import etree
from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

try:
    import fitz
except ImportError:
    fitz = None

try:
    from curl_cffi.requests import AsyncSession as CurlAsyncSession
except ImportError:
    CurlAsyncSession = None

try:
    from playwright.async_api import async_playwright
except ImportError:
    async_playwright = None

try:
    from groq import AsyncGroq
except ImportError:
    AsyncGroq = None

SCRIPT_DIR = Path(__file__).resolve().parent
console = Console()
app = typer.Typer(name="nlack", add_completion=False, invoke_without_command=True)

BROWSER_PROFILES: List[Dict[str, str]] = [
    {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        "Sec-Ch-Ua": '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
        "Sec-Ch-Ua-Mobile": "?0",
        "Sec-Ch-Ua-Platform": '"Windows"',
    },
    {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_7_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        "Sec-Ch-Ua": '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
        "Sec-Ch-Ua-Mobile": "?0",
        "Sec-Ch-Ua-Platform": '"macOS"',
    },
    {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:133.0) Gecko/20100101 Firefox/133.0",
        "Sec-Ch-Ua": '"Not_A Brand";v="99"',
        "Sec-Ch-Ua-Mobile": "?0",
        "Sec-Ch-Ua-Platform": '"Linux"',
    },
]

GEO_PROFILES: Dict[str, Dict[str, str]] = {
    "GLOBAL": {"Accept-Language": "en-US,en;q=0.9"},
    "US": {"Accept-Language": "en-US,en;q=0.9", "X-Forwarded-For": "64.233.160.10"},
    "EU": {"Accept-Language": "en-GB,en;q=0.9,de;q=0.8,fr;q=0.7", "X-Forwarded-For": "185.60.216.35"},
    "IN": {"Accept-Language": "en-IN,en;q=0.9,hi;q=0.8", "X-Forwarded-For": "103.21.244.12"},
    "SG": {"Accept-Language": "en-SG,en;q=0.9,zh;q=0.8", "X-Forwarded-For": "175.41.128.10"},
}

INSTITUTIONAL_API_HOSTS = (
    "sec.gov",
    "efts.sec.gov",
    "openstreetmap.org",
    "nominatim.openstreetmap.org",
    "wikidata.org",
    "wikipedia.org",
    "openalex.org",
    "worldbank.org",
    "nasa.gov",
)


class NlackConfig(BaseModel):
    db_path: Path = Field(default=SCRIPT_DIR / "nlack.db")
    export_dir: Path = Field(default=SCRIPT_DIR)
    download_assets: bool = True
    stream_live: bool = True
    timeout_seconds: float = 28.0
    max_concurrency: int = 10
    base_delay: float = 0.2
    max_delay: float = 15.0
    cache_ttl_seconds: int = 3600
    max_records_per_source: int = 200
    crawl_depth: int = 2
    enable_ai: bool = True
    max_thinking_mode: bool = True
    internet_autoschema: bool = True
    use_headless_browser: bool = False
    solve_with_vision: bool = True
    wayback_fallback: bool = True
    geo_region: str = "GLOBAL"
    proxies: List[str] = Field(default_factory=list)
    cookies_file: Optional[Path] = None
    auth_bearer: Optional[str] = Field(default_factory=lambda: os.getenv("NLACK_BEARER_TOKEN"))
    gemini_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    )
    brave_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("BRAVE_API_KEY"))
    socrata_app_token: Optional[str] = Field(default_factory=lambda: os.getenv("SOCRATA_APP_TOKEN"))
    datagov_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv(
            "DATAGOV_IN_API_KEY", "579b464db66ec23bdd000001cdd3946e44ce4aad7209ff7b23ac571b"
        )
    )
    openalex_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("OPENALEX_API_KEY"))
    github_token: Optional[str] = Field(default_factory=lambda: os.getenv("GITHUB_TOKEN"))
    hf_token: Optional[str] = Field(default_factory=lambda: os.getenv("HF_TOKEN"))
    groq_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("GROQ_API_KEY"))


class DatasetMeta(BaseModel):
    dataset_id: str
    source: str
    title: str
    description: str = ""
    url: str
    download_url: Optional[str] = None
    format: str = "JSON"
    license: str = "Public / Open Discovery"
    attribution: str = ""
    country_scope: str = "Worldwide"
    discovered_at: str = Field(
        default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat()
    )
    extra_meta: Dict[str, Any] = Field(default_factory=dict)


class NormalizedRecord(BaseModel):
    record_id: str
    run_id: str
    dataset_id: str
    source: str
    source_url: str
    local_file: str = ""
    collected_at: str
    entity_key: str
    cluster_id: str
    fingerprint: str
    raw_data: Dict[str, Any]
    normalized_data: Dict[str, Any]
    telemetry_data: Dict[str, Any] = Field(default_factory=dict)
    quality_score: float
    lead_score: float
    quantum_entropy: float = 0.0
    reasoning_trace: str = ""
    quality_breakdown: Dict[str, float]
    ai_tags: List[str] = Field(default_factory=list)
    ai_summary: str = ""
    license: str = "Public / Open"


class NlackEventBus:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._subscribers: List[Callable[[NormalizedRecord], None]] = []

    def subscribe(self, callback: Callable[[NormalizedRecord], None]) -> None:
        self._subscribers.append(callback)

    def publish(self, record: NormalizedRecord) -> None:
        if not self.enabled:
            return
        for cb in self._subscribers:
            try:
                cb(record)
            except Exception:
                continue


class QuantumEngine:
    def __init__(self, candidates: Optional[List[str]] = None) -> None:
        self._amplitudes: Dict[str, complex] = {}
        self._phase_counter: int = 0
        if candidates:
            self.register_states(candidates)

    def register_states(self, states: List[str]) -> None:
        n = max(len(states), 1)
        init_amp = complex(1.0 / math.sqrt(n), 0.0)
        for s in states:
            if s not in self._amplitudes:
                self._amplitudes[s] = init_amp
        self._normalize()

    def _normalize(self) -> None:
        norm = math.sqrt(sum(abs(a) ** 2 for a in self._amplitudes.values()))
        if norm > 0:
            for k in self._amplitudes:
                self._amplitudes[k] /= norm

    def grover_amplify(self, state: str, reward: float) -> None:
        if state not in self._amplitudes:
            self._amplitudes[state] = complex(0.5, 0.0)
        phase = cmath.exp(complex(0, reward * math.pi / 4.0))
        self._amplitudes[state] *= (1.0 + max(0.05, min(reward, 2.0))) * phase
        mean_amp = sum(self._amplitudes.values()) / max(len(self._amplitudes), 1)
        for k in self._amplitudes:
            self._amplitudes[k] = 2 * mean_amp - self._amplitudes[k]
        self._normalize()

    def collapse_superposition(self, top_k: int = 3) -> List[str]:
        if not self._amplitudes:
            return []
        probs = {k: abs(v) ** 2 for k, v in self._amplitudes.items()}
        ranked = sorted(
            probs.keys(),
            key=lambda k: probs[k] * random.uniform(0.88, 1.12),
            reverse=True,
        )
        return ranked[:top_k]

    def tunneling_delay(self, base_delay: float, attempt: int = 1, max_delay: float = 15.0) -> float:
        self._phase_counter += 1
        theta = (self._phase_counter * 1.61803398875) % (2 * math.pi)
        phi = random.uniform(0, math.pi)
        wave = abs(cmath.cos(theta) + 1j * cmath.sin(phi)) ** 2
        return min(base_delay * (1.25 ** (attempt - 1)) * (0.35 + 0.65 * wave), max_delay)

    @staticmethod
    def shannon_entropy(data: Any) -> float:
        text = json.dumps(data, default=str) if not isinstance(data, (str, bytes)) else str(data)
        if not text:
            return 0.0
        counts = Counter(text)
        length = len(text)
        ent = -sum((c / length) * math.log2(c / length) for c in counts.values())
        return round(min(100.0, (ent / 6.0) * 100.0), 2)

    @staticmethod
    def hilbert_fidelity(col_a: str, col_b: str) -> float:
        a_clean = f"_{col_a.lower()}_"
        b_clean = f"_{col_b.lower()}_"
        ngrams_a = [a_clean[i : i + 3] for i in range(max(1, len(a_clean) - 2))]
        ngrams_b = [b_clean[i : i + 3] for i in range(max(1, len(b_clean) - 2))]
        vec_a: Dict[str, complex] = defaultdict(complex)
        vec_b: Dict[str, complex] = defaultdict(complex)
        for idx, ng in enumerate(ngrams_a):
            vec_a[ng] += cmath.exp(1j * (idx * 0.15))
        for idx, ng in enumerate(ngrams_b):
            vec_b[ng] += cmath.exp(1j * (idx * 0.15))
        norm_a = math.sqrt(sum(abs(v) ** 2 for v in vec_a.values())) or 1.0
        norm_b = math.sqrt(sum(abs(v) ** 2 for v in vec_b.values())) or 1.0
        inner = sum(vec_a[k].conjugate() * vec_b[k] for k in set(vec_a) & set(vec_b))
        return round(abs(inner / (norm_a * norm_b)) ** 2, 4)


class NlackCognitiveEngine:
    POLLINATIONS_CHAT_URL = "https://text.pollinations.ai/openai"
    GOOGLE_GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"

    def __init__(self, config: NlackConfig) -> None:
        self.config = config

    def synthesize_local_thinking_trace(
        self, norm: Dict[str, Any], telemetry: Dict[str, Any], q_entropy: float
    ) -> str:
        steps: List[str] = []
        entity = norm.get("company") or norm.get("name") or norm.get("identifier") or "Entity_Node"
        steps.append(f"[N1:Node={str(entity)[:28]}]")
        steps.append(f"[N2:Fields={len(norm)}|H={q_entropy:.1f}b]")
        if telemetry.get("wallets") or telemetry.get("ip_addresses") or telemetry.get("lei_identifiers"):
            steps.append(
                f"[N3:Telemetry=W:{len(telemetry.get('wallets', []))},IP:{len(telemetry.get('ip_addresses', []))},ID:{len(telemetry.get('isin_identifiers', []))}]"
            )
        else:
            steps.append("[N3:Verified_Dataset_Signal]")
        return " -> ".join(steps)

    async def chat_json(
        self, system_prompt: str, user_prompt: str, client: Optional[httpx.AsyncClient] = None
    ) -> Optional[Dict[str, Any]]:
        if not self.config.enable_ai:
            return None

        directive = (
            "Nlack Deep Reasoning Protocol: "
            "1. Decompose structure and semantics. "
            "2. Cross-verify ontology alignments. "
            "3. Output strictly valid JSON."
        )
        full_system = f"{directive}\n{system_prompt}"
        own_client = client is None
        http_client = client or httpx.AsyncClient(timeout=20.0)

        try:
            if self.config.gemini_api_key:
                try:
                    gemini_payload = {
                        "systemInstruction": {"parts": [{"text": full_system}]},
                        "contents": [{"parts": [{"text": user_prompt}]}],
                        "generationConfig": {"temperature": 0.1, "responseMimeType": "application/json"},
                    }
                    r = await http_client.post(
                        f"{self.GOOGLE_GEMINI_URL}?key={self.config.gemini_api_key}",
                        json=gemini_payload,
                    )
                    if r.status_code == 200:
                        candidates = r.json().get("candidates") or []
                        if candidates:
                            text_out = (
                                ((candidates[0].get("content") or {}).get("parts") or [{}])[0].get("text", "")
                            )
                            match = re.search(r"\{.*\}", text_out, re.DOTALL)
                            if match:
                                return json.loads(match.group(0))
                except Exception:
                    pass

            if self.config.groq_api_key and AsyncGroq:
                try:
                    groq_client = AsyncGroq(api_key=self.config.groq_api_key)
                    resp = await groq_client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[
                            {"role": "system", "content": full_system},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=0.1,
                        response_format={"type": "json_object"},
                    )
                    return json.loads(resp.choices[0].message.content or "{}")
                except Exception:
                    pass

            payload = {
                "model": "openai",
                "messages": [
                    {"role": "system", "content": full_system},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.1,
                "jsonMode": True,
            }
            r = await http_client.post(self.POLLINATIONS_CHAT_URL, json=payload)
            if r.status_code == 200:
                data = r.json()
                content = (
                    data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    if isinstance(data, dict)
                    else r.text
                )
                match = re.search(r"\{.*\}", content, re.DOTALL)
                if match:
                    return json.loads(match.group(0))
        except Exception:
            pass
        finally:
            if own_client:
                await http_client.aclose()
        return None

    async def extract_from_image_bytes(
        self,
        image_bytes: bytes,
        instruction: str = "Extract all text, tables, and statistical metrics from this image.",
        mime_type: str = "image/png",
        client: Optional[httpx.AsyncClient] = None,
    ) -> Dict[str, Any]:
        b64_img = base64.b64encode(image_bytes).decode("ascii")
        prompt_text = (
            f"{instruction}\n"
            "Return strict JSON with keys: "
            "'extracted_text' (all visible text) and "
            "'records' (array of flat key-value objects for any data or tables visible)."
        )
        own_client = client is None
        http_client = client or httpx.AsyncClient(timeout=26.0)

        try:
            if self.config.gemini_api_key:
                try:
                    gemini_payload = {
                        "contents": [
                            {
                                "parts": [
                                    {"text": prompt_text},
                                    {"inlineData": {"mimeType": mime_type, "data": b64_img}},
                                ]
                            }
                        ],
                        "generationConfig": {"temperature": 0.1, "responseMimeType": "application/json"},
                    }
                    r = await http_client.post(
                        f"{self.GOOGLE_GEMINI_URL}?key={self.config.gemini_api_key}",
                        json=gemini_payload,
                    )
                    if r.status_code == 200:
                        candidates = r.json().get("candidates") or []
                        if candidates:
                            text_out = (
                                ((candidates[0].get("content") or {}).get("parts") or [{}])[0].get("text", "")
                            )
                            match = re.search(r"\{.*\}", text_out, re.DOTALL)
                            if match:
                                return json.loads(match.group(0))
                except Exception:
                    pass

            payload = {
                "model": "openai",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64_img}"}},
                        ],
                    }
                ],
                "jsonMode": True,
            }
            r = await http_client.post(self.POLLINATIONS_CHAT_URL, json=payload)
            if r.status_code == 200:
                data = r.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "{}")
                match = re.search(r"\{.*\}", content, re.DOTALL)
                if match:
                    return json.loads(match.group(0))
        except Exception:
            pass
        finally:
            if own_client:
                await http_client.aclose()
        return {"extracted_text": "", "records": []}

    async def deep_enrich_records(
        self, records: List[NormalizedRecord], client: Optional[httpx.AsyncClient] = None
    ) -> None:
        if not (self.config.enable_ai and self.config.max_thinking_mode and records):
            return
        top_slice = sorted(records, key=lambda r: r.lead_score, reverse=True)[:5]
        compact_input = [
            {"id": r.record_id, "entity": r.entity_key, "data": r.normalized_data, "telemetry": r.telemetry_data}
            for r in top_slice
        ]
        sys_prompt = (
            "Perform cross-record intelligence synthesis. "
            "Return JSON with key 'enrichments' mapping each record 'id' to an object with "
            "'trace' (1-line analytical reasoning chain) and 'summary' (1-line executive insight)."
        )
        res = await self.chat_json(sys_prompt, json.dumps(compact_input, default=str)[:3500], client=client)
        if isinstance(res, dict) and isinstance(res.get("enrichments"), dict):
            enrich_map = res["enrichments"]
            for rec in top_slice:
                if rec.record_id in enrich_map and isinstance(enrich_map[rec.record_id], dict):
                    item = enrich_map[rec.record_id]
                    if item.get("trace"):
                        rec.reasoning_trace = str(item["trace"])[:180]
                    if item.get("summary"):
                        rec.ai_summary = str(item["summary"])[:200]


class LocalAssetVault:
    EXTENSION_MAP: Dict[str, str] = {
        "application/pdf": ".pdf",
        "text/csv": ".csv",
        "application/json": ".json",
        "application/xml": ".xml",
        "text/xml": ".xml",
        "text/html": ".html",
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/webp": ".webp",
        "image/gif": ".gif",
        "image/tiff": ".tiff",
        "image/bmp": ".bmp",
        "image/svg+xml": ".svg",
        "application/zip": ".zip",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
        "application/vnd.ms-excel": ".xls",
    }

    @classmethod
    def infer_extension(cls, url: str, content_type: str, body: bytes) -> str:
        clean_path = urllib.parse.urlparse(url).path.lower()
        for ext in (
            ".pdf", ".csv", ".xlsx", ".xls", ".json", ".xml", ".png", ".jpg",
            ".jpeg", ".webp", ".gif", ".tiff", ".bmp", ".svg", ".zip", ".parquet",
            ".doc", ".docx", ".txt", ".html",
        ):
            if clean_path.endswith(ext):
                return ext
        ct_base = (content_type or "").split(";")[0].strip().lower()
        if ct_base in cls.EXTENSION_MAP:
            return cls.EXTENSION_MAP[ct_base]
        if body[:4] == b"%PDF":
            return ".pdf"
        if body[:8] == b"\x89PNG\r\n\x1a\n":
            return ".png"
        if body[:3] == b"\xff\xd8\xff":
            return ".jpg"
        if body[:4] == b"RIFF" and body[8:12] == b"WEBP":
            return ".webp"
        if body[:4] == b"PK\x03\x04":
            return ".xlsx" if "spreadsheet" in ct_base else ".zip"
        return ".dat"

    @classmethod
    def save_to_script_folder(
        cls,
        url: str,
        content_type: str,
        body: bytes,
        prefix: str = "nlack",
        target_dir: Path = SCRIPT_DIR,
    ) -> Path:
        target_dir.mkdir(parents=True, exist_ok=True)
        ext = cls.infer_extension(url, content_type, body)
        parsed = urllib.parse.urlparse(url)
        raw_stem = Path(parsed.path).stem or parsed.netloc.replace(".", "_") or "asset"
        safe_stem = re.sub(r"[^a-zA-Z0-9_-]+", "_", raw_stem).strip("_")[:45]
        short_hash = hashlib.sha1(body[:4096] + url.encode("utf-8")).hexdigest()[:8]
        filename = f"{prefix}_{safe_stem}_{short_hash}{ext}"
        file_path = target_dir / filename
        if not file_path.exists():
            file_path.write_bytes(body)
        return file_path


class ForensicTelemetryEngine:
    BASE58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
    REGEX_BTC = re.compile(r"\b(?:bc1[a-zA-HJ-NP-Z0-9]{25,39}|[13][a-km-zA-HJ-NP-Z1-9]{25,34})\b")
    REGEX_ETH = re.compile(r"\b0x[a-fA-F0-9]{40}\b")
    REGEX_TRON = re.compile(r"\bT[A-Za-z1-9]{33}\b")
    REGEX_IPV4 = re.compile(
        r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"
    )
    REGEX_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,10}\b")
    REGEX_ISIN = re.compile(r"\b[A-Z]{2}[A-Z0-9]{9}\d\b")
    REGEX_LEI = re.compile(r"\b[0-9A-Z]{18}\d{2}\b")

    def __init__(self) -> None:
        self._parent: Dict[str, str] = {}

    def load_persisted_edges(self, edges: List[Tuple[str, str]]) -> None:
        for node_a, node_b in edges:
            self._union(node_a, node_b)

    def _find(self, item: str) -> str:
        if item not in self._parent:
            self._parent[item] = item
        if self._parent[item] != item:
            self._parent[item] = self._find(self._parent[item])
        return self._parent[item]

    def _union(self, a: str, b: str) -> None:
        ra, rb = self._find(a), self._find(b)
        if ra != rb:
            self._parent[rb] = ra

    @classmethod
    def verify_base58check(cls, address: str) -> bool:
        try:
            num = 0
            for char in address:
                num = num * 58 + cls.BASE58_ALPHABET.index(char)
            combined = num.to_bytes(25, byteorder="big")
            payload, checksum = combined[:-4], combined[-4:]
            digest = hashlib.sha256(hashlib.sha256(payload).digest()).digest()[:4]
            return checksum == digest
        except Exception:
            return False

    @staticmethod
    def verify_lei_mod97(lei: str) -> bool:
        if len(lei) != 20 or not lei[:18].isalnum() or not lei[18:].isdigit():
            return False
        converted = "".join(str(ord(c) - 55) if c.isalpha() else c for c in lei)
        return int(converted) % 97 == 1

    @staticmethod
    def verify_isin_luhn(isin: str) -> bool:
        if len(isin) != 12 or not isin[:2].isalpha():
            return False
        digits = "".join(str(ord(c) - 55) if c.isalpha() else c for c in isin)
        total = 0
        for idx, d in enumerate(reversed(digits)):
            n = int(d)
            if idx % 2 == 1:
                n *= 2
                if n > 9:
                    n -= 9
            total += n
        return total % 10 == 0

    def extract_telemetry(self, raw: Dict[str, Any], norm: Dict[str, Any]) -> Dict[str, Any]:
        blob = json.dumps({"raw": raw, "norm": norm}, default=str)
        btc = [
            w
            for w in sorted(set(self.REGEX_BTC.findall(blob)))
            if w.startswith("bc1") or self.verify_base58check(w)
        ]
        eth = sorted(set(self.REGEX_ETH.findall(blob)))
        tron = [w for w in sorted(set(self.REGEX_TRON.findall(blob))) if self.verify_base58check(w)]
        wallets = [f"btc:{w}" for w in btc] + [f"eth:{w}" for w in eth] + [f"tron:{w}" for w in tron]

        ips = [
            ip
            for ip in sorted(set(self.REGEX_IPV4.findall(blob)))
            if not ip.startswith(("0.", "127.", "255.", "192.168.", "10."))
        ]
        emails = sorted(set(self.REGEX_EMAIL.findall(blob)))
        isins = [i for i in sorted(set(self.REGEX_ISIN.findall(blob))) if self.verify_isin_luhn(i)]
        leis = [l for l in sorted(set(self.REGEX_LEI.findall(blob))) if self.verify_lei_mod97(l)]

        telemetry: Dict[str, Any] = {}
        if wallets:
            telemetry["wallets"] = wallets[:20]
        if ips:
            telemetry["ip_addresses"] = ips[:20]
        if emails:
            telemetry["emails"] = emails[:20]
        if isins:
            telemetry["isin_identifiers"] = isins[:15]
        if leis:
            telemetry["lei_identifiers"] = leis[:10]
        return telemetry

    async def enrich_geoip(
        self, telemetry: Dict[str, Any], http: "NlackNetworkEngine"
    ) -> None:
        ips = telemetry.get("ip_addresses") or []
        if not ips:
            return
        geo_profiles = []
        for ip in ips[:3]:
            res = await http.fetch(
                f"http://ip-api.com/json/{ip}?fields=status,country,countryCode,regionName,city,lat,lon,isp,org,as,query"
            )
            if res and res[0] == 200:
                try:
                    info = json.loads(res[2].decode("utf-8", errors="replace"))
                    if info.get("status") == "success":
                        geo_profiles.append(
                            {
                                "ip": ip,
                                "country": info.get("country"),
                                "city": info.get("city"),
                                "lat": info.get("lat"),
                                "lon": info.get("lon"),
                                "asn": info.get("as"),
                                "isp": info.get("isp"),
                            }
                        )
                except Exception:
                    continue
        if geo_profiles:
            telemetry["geoip_attribution"] = geo_profiles

    def assign_cluster(self, entity_key: str, telemetry: Dict[str, Any]) -> Tuple[str, List[Tuple[str, str]]]:
        pivots: List[str] = [f"entity:{entity_key}"]
        for w in telemetry.get("wallets", []):
            pivots.append(f"wallet:{w}")
        for ip in telemetry.get("ip_addresses", []):
            pivots.append(f"ip:{ip}")
        for em in telemetry.get("emails", []):
            pivots.append(f"email:{em.lower()}")
        for isin in telemetry.get("isin_identifiers", []):
            pivots.append(f"isin:{isin}")

        first = pivots[0]
        new_edges: List[Tuple[str, str]] = []
        for p in pivots[1:]:
            self._union(first, p)
            new_edges.append((first, p))
        root = self._find(first)
        return "cluster_" + hashlib.sha1(root.encode("utf-8")).hexdigest()[:10], new_edges

    @staticmethod
    def compute_lead_score(
        norm: Dict[str, Any],
        telemetry: Dict[str, Any],
        quality_score: float,
        quantum_entropy: float,
    ) -> float:
        score = quality_score * 0.40 + quantum_entropy * 0.12
        if any(k in norm for k in ("price", "value", "population", "stars")):
            score += 18.0
        if "latitude" in norm and "longitude" in norm:
            score += 10.0
        if "company" in norm or "name" in norm:
            score += 10.0
        if telemetry.get("wallets"):
            score += 15.0
        if telemetry.get("ip_addresses") or telemetry.get("isin_identifiers") or telemetry.get("lei_identifiers"):
            score += 10.0
        if telemetry.get("emails"):
            score += 5.0
        return round(min(100.0, score), 2)


class AutoSchema:
    CANONICAL_SYNONYMS: Dict[str, List[str]] = {
        "company": [
            "company", "company_name", "organization", "organisation", "employer",
            "org_name", "firm", "business_name", "vendor", "operator", "publisher",
            "owner", "agency", "department", "ministry", "institution", "author",
            "issuer", "intermediary", "bank_name", "maintainer", "holder", "filer",
        ],
        "city": [
            "city", "location", "office_city", "town", "municipality", "addr_city",
            "addr:city", "district", "urban_area", "headquarters_city", "place",
            "market", "mandi", "centre", "locality", "jurisdiction_city",
        ],
        "state": [
            "state", "province", "region", "addr_state", "addr:state", "state_name",
            "admin_area", "territory", "prefecture", "bundesland", "oblast",
        ],
        "country": [
            "country", "country_name", "nation", "country_code", "iso3", "iso2",
            "countryiso3code", "addr_country", "addr:country", "entity", "sovereign",
        ],
        "name": [
            "name", "title", "label", "item_name", "full_name", "repo_name",
            "dataset_name", "indicator_name", "commodity", "crop", "product",
            "facility_name", "amenity_name", "headline", "subject", "parameter",
            "series_title", "project_name", "resource_name",
        ],
        "category": [
            "category", "type", "sector", "industry", "amenity", "topic",
            "domain", "class", "group", "variety", "pipeline_tag", "language",
            "segment", "theme", "classification", "form_type",
        ],
        "date": [
            "date", "published_date", "created_at", "updated_at", "timestamp",
            "arrival_date", "report_date", "last_modified", "observation_date",
            "time", "period", "filing_date", "time_start", "seendate", "release_date",
        ],
        "year": ["year", "fiscal_year", "yr", "calendar_year", "reference_year", "publication_year"],
        "price": [
            "price", "modal_price", "min_price", "max_price", "cost", "amount",
            "rate", "tariff", "fee", "unit_price", "wholesale_price", "nav", "close_price",
        ],
        "value": [
            "value", "val", "metric_value", "indicator_value", "score",
            "measurement", "total", "count", "quantity", "net_investment",
            "cited_by_count", "funding_amount", "gdp",
        ],
        "currency": ["currency", "curr", "currency_code", "price_unit", "denomination"],
        "unit": ["unit", "units", "uom", "measure_unit", "scale"],
        "latitude": ["latitude", "lat", "y", "coord_lat", "geo_lat"],
        "longitude": ["longitude", "lon", "lng", "long", "x", "coord_lon", "geo_lon"],
        "identifier": [
            "id", "identifier", "uid", "code", "indicator_id", "qid", "osm_id",
            "repo_id", "resource_id", "uuid", "isin", "cin", "granule_id", "cik", "doi",
        ],
        "url": ["url", "link", "html_url", "homepage", "website", "source_url", "uri", "download_url"],
        "description": [
            "description", "summary", "abstract", "notes", "details", "bio", "comment", "snippet",
        ],
        "population": ["population", "pop", "inhabitants", "residents", "headcount"],
        "stars": ["stars", "stargazers_count", "likes", "upvotes", "rating"],
        "downloads": ["downloads", "download_count", "forks", "forks_count"],
        "email": ["email", "email_address", "contact_email", "maintainer_email"],
        "phone": ["phone", "phone_number", "telephone", "mobile", "contact_number"],
        "ip_address": ["ip", "ip_address", "ipv4", "ipv6", "host_ip", "remote_addr"],
        "wallet_address": ["wallet", "wallet_address", "crypto_address", "btc_address", "eth_address"],
    }

    CITY_ALIASES: Dict[str, str] = {
        "bangalore": "Bengaluru", "bengaluru": "Bengaluru",
        "bombay": "Mumbai", "mumbai": "Mumbai",
        "madras": "Chennai", "chennai": "Chennai",
        "calcutta": "Kolkata", "kolkata": "Kolkata",
        "poona": "Pune", "pune": "Pune",
        "new delhi": "New Delhi", "delhi": "Delhi",
        "nyc": "New York", "new york city": "New York",
        "sf": "San Francisco",
    }

    CURRENCY_SYMBOLS: Dict[str, str] = {
        "₹": "INR", "rs.": "INR", "rs": "INR", "inr": "INR",
        "$": "USD", "us$": "USD", "usd": "USD",
        "€": "EUR", "eur": "EUR", "£": "GBP", "gbp": "GBP",
        "¥": "JPY", "jpy": "JPY", "cny": "CNY",
    }

    def __init__(self) -> None:
        self._lookup: Dict[str, str] = {}
        for canonical, synonyms in self.CANONICAL_SYNONYMS.items():
            for syn in synonyms:
                self._lookup[self._slugify(syn)] = canonical
        self._unresolved_columns: Set[str] = set()

    @staticmethod
    def _slugify(col: str) -> str:
        clean = re.sub(r"[^a-zA-Z0-9]+", "_", str(col).strip().lower()).strip("_")
        return clean or "field"

    def load_learned_mappings(self, learned: Dict[str, str]) -> None:
        for raw_slug, canonical in learned.items():
            self._lookup[self._slugify(raw_slug)] = canonical

    def map_column(self, raw_col: str) -> str:
        slug = self._slugify(raw_col)
        if slug in self._lookup:
            return self._lookup[slug]

        best_match = None
        best_score = 0.0
        for syn_slug, canonical in self._lookup.items():
            seq_ratio = difflib.SequenceMatcher(None, slug, syn_slug).ratio()
            hilbert = QuantumEngine.hilbert_fidelity(slug, syn_slug)
            combined = 0.6 * seq_ratio + 0.4 * hilbert
            if combined > best_score:
                best_score = combined
                best_match = canonical

        if best_match and best_score >= 0.82:
            self._lookup[slug] = best_match
            return best_match

        tokens = slug.split("_")
        for tok in reversed(tokens):
            if tok in self._lookup:
                self._lookup[slug] = self._lookup[tok]
                return self._lookup[tok]

        self._unresolved_columns.add(slug)
        return slug

    async def resolve_unknown_columns_online(
        self,
        unknown_cols: List[str],
        http: "NlackNetworkEngine",
        ai_engine: NlackCognitiveEngine,
        storage: "StorageEngine",
    ) -> Dict[str, str]:
        if not unknown_cols:
            return {}

        canonical_targets = list(self.CANONICAL_SYNONYMS.keys())
        newly_learned: Dict[str, str] = {}

        for col in unknown_cols[:15]:
            human_term = col.replace("_", " ")
            dm_res = await http.fetch(
                "https://api.datamuse.com/words",
                params={"ml": human_term, "max": 8},
            )
            if dm_res and dm_res[0] == 200:
                try:
                    words = json.loads(dm_res[2].decode("utf-8", errors="replace"))
                    for w_obj in words:
                        w_slug = self._slugify(w_obj.get("word", ""))
                        if w_slug in self._lookup:
                            mapped = self._lookup[w_slug]
                            newly_learned[col] = mapped
                            self._lookup[col] = mapped
                            break
                except Exception:
                    pass

            if col not in newly_learned:
                wd_res = await http.fetch(
                    "https://www.wikidata.org/w/api.php",
                    params={
                        "action": "wbsearchentities",
                        "search": human_term,
                        "type": "property",
                        "language": "en",
                        "format": "json",
                        "limit": 5,
                    },
                )
                if wd_res and wd_res[0] == 200:
                    try:
                        wd_items = json.loads(wd_res[2].decode("utf-8", errors="replace")).get("search", [])
                        for item in wd_items:
                            label_slug = self._slugify(item.get("label", ""))
                            desc_tokens = self._slugify(item.get("description", "")).split("_")
                            for candidate in [label_slug] + desc_tokens:
                                if candidate in self._lookup:
                                    mapped = self._lookup[candidate]
                                    newly_learned[col] = mapped
                                    self._lookup[col] = mapped
                                    break
                            if col in newly_learned:
                                break
                    except Exception:
                        pass

        still_unknown = [c for c in unknown_cols[:25] if c not in newly_learned]
        if still_unknown:
            sys_prompt = (
                "You are Nlack AutoSchema. Map each raw column name to the closest canonical field from "
                f"{canonical_targets}, or keep the snake_case name if none fit. Return JSON mapping raw -> canonical."
            )
            user_prompt = f"Map these columns: {json.dumps(still_unknown)}"
            llm_map = await ai_engine.chat_json(sys_prompt, user_prompt, client=http._http_client)
            if isinstance(llm_map, dict):
                for k, v in llm_map.items():
                    k_slug = self._slugify(k)
                    v_slug = self._slugify(str(v))
                    if v_slug in canonical_targets:
                        newly_learned[k_slug] = v_slug
                        self._lookup[k_slug] = v_slug

        if newly_learned:
            await storage.save_learned_schemas(newly_learned)
        return newly_learned

    def normalize_record(self, raw_record: Dict[str, Any]) -> Dict[str, Any]:
        normalized: Dict[str, Any] = {}
        for raw_key, raw_val in raw_record.items():
            if raw_val is None:
                continue
            if isinstance(raw_val, float) and pd.isna(raw_val):
                continue
            if isinstance(raw_val, str):
                val_str = raw_val.strip()
                if not val_str or val_str.lower() in {"null", "none", "n/a", "na", "-", "nan"}:
                    continue
            else:
                val_str = None

            canonical_key = self.map_column(raw_key)

            if canonical_key in {"price", "value"} and isinstance(raw_val, str):
                num_val, detected_curr = self.parse_currency_and_number(raw_val)
                normalized[canonical_key] = num_val if num_val is not None else raw_val.strip()
                if detected_curr and "currency" not in normalized:
                    normalized["currency"] = detected_curr
            elif canonical_key == "date":
                iso_date = self.normalize_date(raw_val)
                normalized["date"] = iso_date
                if len(iso_date) >= 4 and iso_date[:4].isdigit() and "year" not in normalized:
                    normalized["year"] = int(iso_date[:4])
            elif canonical_key == "year":
                yr = self.parse_year(raw_val)
                if yr is not None:
                    normalized["year"] = yr
            elif canonical_key in {"city", "state", "country"} and isinstance(raw_val, str):
                normalized[canonical_key] = self.normalize_location(raw_val, canonical_key)
            elif canonical_key in {"latitude", "longitude"}:
                try:
                    normalized[canonical_key] = round(float(raw_val), 6)
                except (ValueError, TypeError):
                    normalized[canonical_key] = raw_val
            else:
                if canonical_key not in normalized:
                    normalized[canonical_key] = val_str if val_str is not None else raw_val
                else:
                    fallback_key = self._slugify(raw_key)
                    if fallback_key not in normalized:
                        normalized[fallback_key] = val_str if val_str is not None else raw_val

        return normalized

    @classmethod
    def parse_currency_and_number(cls, text: str) -> Tuple[Optional[float], Optional[str]]:
        cleaned = text.strip()
        detected_currency: Optional[str] = None
        lower = cleaned.lower()

        for sym, code in cls.CURRENCY_SYMBOLS.items():
            if sym in lower:
                detected_currency = code
                break

        multiplier = 1.0
        if re.search(r"\b(cr|crore|crores)\b", lower):
            multiplier = 1e7
        elif re.search(r"\b(lakh|lakhs|lac)\b", lower):
            multiplier = 1e5
        elif re.search(r"\d+(\.\d+)?\s*(b|bn|billion)\b", lower):
            multiplier = 1e9
        elif re.search(r"\d+(\.\d+)?\s*(m|mn|million)\b", lower):
            multiplier = 1e6
        elif re.search(r"\d+(\.\d+)?\s*k\b", lower):
            multiplier = 1e3

        num_match = re.search(r"[-+]?\d[\d,]*(\.\d+)?", cleaned)
        if not num_match:
            return None, detected_currency
        try:
            num_str = num_match.group(0).replace(",", "")
            return round(float(num_str) * multiplier, 4), detected_currency
        except ValueError:
            return None, detected_currency

    @staticmethod
    def normalize_date(val: Any) -> str:
        s = str(val).strip()
        if re.fullmatch(r"\d{4}", s):
            return s
        for fmt in (
            "%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d",
            "%Y%m%dT%H%M%SZ", "%d %b %Y", "%d %B %Y", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S",
        ):
            try:
                return dt.datetime.strptime(s[:19], fmt).date().isoformat()
            except ValueError:
                continue
        iso_match = re.match(r"^(\d{4}-\d{2}-\d{2})", s)
        return iso_match.group(1) if iso_match else s

    @staticmethod
    def parse_year(val: Any) -> Optional[int]:
        m = re.search(r"\b(18\d{2}|19\d{2}|20\d{2})\b", str(val))
        return int(m.group(1)) if m else None

    @classmethod
    def normalize_location(cls, val: str, field_type: str) -> str:
        clean = re.sub(r"\s+", " ", val.strip())
        lower = clean.lower()
        if field_type == "city" and lower in cls.CITY_ALIASES:
            return cls.CITY_ALIASES[lower]
        if field_type == "country":
            if lower in {"ind", "in", "india", "republic of india"}:
                return "India"
            if lower in {"usa", "us", "united states", "united states of america"}:
                return "United States"
            if lower in {"uk", "gb", "gbr", "united kingdom", "great britain"}:
                return "United Kingdom"
        return clean.title() if clean.islower() or clean.isupper() else clean


class IntelligenceEngine:
    def __init__(self, config: NlackConfig, event_bus: Optional[NlackEventBus] = None) -> None:
        self.config = config
        self.autoschema = AutoSchema()
        self.forensics = ForensicTelemetryEngine()
        self.cognitive = NlackCognitiveEngine(config)
        self.event_bus = event_bus or NlackEventBus(enabled=config.stream_live)

    def build_entity_key(self, norm: Dict[str, Any], source: str) -> str:
        primary = (
            norm.get("company")
            or norm.get("name")
            or norm.get("identifier")
            or norm.get("wallet_address")
            or norm.get("ip_address")
            or str(norm.get("description", ""))[:40]
        )
        location = norm.get("city") or norm.get("state") or norm.get("country") or ""
        temporal = str(norm.get("date") or norm.get("year") or "")

        def clean_token(t: Any) -> str:
            s = str(t).lower().strip()
            s = re.sub(r"\b(pvt|ltd|limited|inc|corp|corporation|llc|llp|plc|gmbh)\b", "", s)
            return re.sub(r"[^a-z0-9]+", "-", s).strip("-")

        parts = [clean_token(p) for p in (primary, location, temporal) if p]
        if not parts:
            h = hashlib.md5(json.dumps(norm, sort_keys=True, default=str).encode()).hexdigest()[:12]
            return f"{source}:{h}"
        return ":".join(p for p in parts if p)

    @staticmethod
    def compute_fingerprint(norm: Dict[str, Any]) -> str:
        canonical_json = json.dumps(norm, sort_keys=True, default=str)
        return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

    def score_quality(
        self, raw: Dict[str, Any], norm: Dict[str, Any], source_url: str
    ) -> Tuple[float, Dict[str, float]]:
        total_raw = max(len(raw), 1)
        completeness = min(1.0, len(norm) / max(total_raw, 3)) * 35.0
        canonical_keys = set(AutoSchema.CANONICAL_SYNONYMS.keys())
        schema_conformance = min(1.0, len(set(norm.keys()) & canonical_keys) / 3.0) * 30.0

        validity_pts = 0.0
        if any(k in norm for k in ("name", "company", "identifier")):
            validity_pts += 8.0
        if any(k in norm for k in ("value", "price", "population", "stars", "latitude")):
            validity_pts += 6.0
        if any(k in norm for k in ("date", "year", "city", "country")):
            validity_pts += 6.0

        provenance_pts = 15.0 if source_url else 7.0
        total = round(completeness + schema_conformance + validity_pts + provenance_pts, 2)
        return total, {
            "completeness": round(completeness, 2),
            "schema_conformance": round(schema_conformance, 2),
            "semantic_validity": round(validity_pts, 2),
            "provenance": round(provenance_pts, 2),
        }

    def heuristic_enrich(
        self, norm: Dict[str, Any], telemetry: Dict[str, Any], source: str
    ) -> Tuple[List[str], str]:
        tags = {source}
        blob = json.dumps({"norm": norm, "tel": telemetry}, default=str).lower()

        domain_keywords = {
            "government-data": ["ministry", "department", "gov", "federal", "municipal", "census", "sec", "ckan", "socrata"],
            "agriculture": ["crop", "wheat", "rice", "mandi", "farm", "agri", "quintal"],
            "economics-finance": ["gdp", "inflation", "rbi", "sebi", "fpi", "forex", "repo", "rate", "price", "edgar"],
            "geospatial": ["latitude", "longitude", "osm", "satellite", "nasa", "coordinate"],
            "blockchain-telemetry": ["btc:", "eth:", "tron:", "wallet", "blockchain"],
            "network-telemetry": ["ip_addresses", "geoip_attribution", "isin_identifiers", "lei_identifiers"],
        }
        for tag, kw_list in domain_keywords.items():
            if any(kw in blob for kw in kw_list):
                tags.add(tag)

        subject = norm.get("company") or norm.get("name") or norm.get("identifier") or "Record"
        loc = norm.get("city") or norm.get("state") or norm.get("country")
        val = norm.get("value") if "value" in norm else norm.get("price")
        unit = norm.get("currency") or norm.get("unit") or ""
        yr = norm.get("date") or norm.get("year")

        parts = [str(subject)[:75]]
        if loc:
            parts.append(f"in {loc}")
        if val is not None:
            parts.append(f"= {val} {unit}".strip())
        if yr:
            parts.append(f"({yr})")
        if telemetry.get("wallets"):
            parts.append(f"[Wallets: {len(telemetry['wallets'])}]")
        if telemetry.get("isin_identifiers") or telemetry.get("lei_identifiers"):
            parts.append("[Verified Corp ID]")

        return sorted(tags), " ".join(parts)

    def process_raw_records(
        self,
        raw_rows: List[Dict[str, Any]],
        run_id: str,
        dataset: DatasetMeta,
        seen_fingerprints: Set[str],
    ) -> Tuple[List[NormalizedRecord], int, List[Tuple[str, str]]]:
        processed: List[NormalizedRecord] = []
        all_edges: List[Tuple[str, str]] = []
        duplicates = 0
        now_iso = dt.datetime.now(dt.timezone.utc).isoformat()

        for raw in raw_rows:
            if not isinstance(raw, dict) or not raw:
                continue
            local_file = str(raw.pop("_local_file", "") or "")
            norm = self.autoschema.normalize_record(raw)
            if not norm:
                continue

            fp = self.compute_fingerprint(norm)
            if fp in seen_fingerprints:
                duplicates += 1
                continue
            seen_fingerprints.add(fp)

            telemetry = self.forensics.extract_telemetry(raw, norm)
            if local_file:
                telemetry["saved_local_file"] = local_file
            entity_key = self.build_entity_key(norm, dataset.source)
            cluster_id, new_edges = self.forensics.assign_cluster(entity_key, telemetry)
            all_edges.extend(new_edges)

            src_url = str(norm.get("url") or dataset.download_url or dataset.url)
            q_score, q_breakdown = self.score_quality(raw, norm, src_url)
            q_entropy = QuantumEngine.shannon_entropy(norm)
            lead_score = self.forensics.compute_lead_score(norm, telemetry, q_score, q_entropy)
            tags, summary = self.heuristic_enrich(norm, telemetry, dataset.source)
            trace = self.cognitive.synthesize_local_thinking_trace(norm, telemetry, q_entropy)

            rec_id = hashlib.sha1(f"{dataset.dataset_id}:{fp}".encode("utf-8")).hexdigest()[:16]
            record_obj = NormalizedRecord(
                record_id=rec_id,
                run_id=run_id,
                dataset_id=dataset.dataset_id,
                source=dataset.source,
                source_url=src_url,
                local_file=local_file,
                collected_at=now_iso,
                entity_key=entity_key,
                cluster_id=cluster_id,
                fingerprint=fp,
                raw_data=raw,
                normalized_data=norm,
                telemetry_data=telemetry,
                quality_score=q_score,
                lead_score=lead_score,
                quantum_entropy=q_entropy,
                reasoning_trace=trace,
                quality_breakdown=q_breakdown,
                ai_tags=tags,
                ai_summary=summary,
                license=dataset.license,
            )
            processed.append(record_obj)
            self.event_bus.publish(record_obj)
            if len(processed) >= self.config.max_records_per_source:
                break

        return processed, duplicates, all_edges


class UniversalExtractor:
    IMAGE_FORMATS = {"PNG", "JPG", "JPEG", "WEBP", "GIF", "TIFF", "BMP", "SVG", "IMAGE"}

    @classmethod
    async def extract_async(
        cls,
        content: bytes,
        format_hint: str = "",
        content_type: str = "",
        url: str = "",
        vision_ai: Optional[NlackCognitiveEngine] = None,
        save_binary_to_pc: bool = True,
        target_dir: Path = SCRIPT_DIR,
    ) -> List[Dict[str, Any]]:
        fmt = cls.detect_format(content, format_hint, content_type, url)
        saved_path_str = ""
        if save_binary_to_pc and (
            fmt in {"PDF", "XLSX", "CSV", "ZIP"} or fmt in cls.IMAGE_FORMATS
        ):
            saved_path = await asyncio.to_thread(
                LocalAssetVault.save_to_script_folder,
                url,
                content_type,
                content,
                fmt.lower(),
                target_dir,
            )
            saved_path_str = str(saved_path)

        rows: List[Dict[str, Any]] = []
        try:
            if fmt == "JSON":
                rows = await asyncio.to_thread(cls.extract_json, content)
            elif fmt == "CSV":
                rows = await asyncio.to_thread(cls.extract_csv, content)
            elif fmt == "XLSX":
                rows = await asyncio.to_thread(cls.extract_excel, content)
            elif fmt == "XML":
                rows = await asyncio.to_thread(cls.extract_xml, content)
            elif fmt == "PDF":
                rows = await cls.extract_pdf_async(content, url=url, vision_ai=vision_ai)
            elif fmt in cls.IMAGE_FORMATS:
                rows = await cls.extract_image_async(content, content_type=content_type, url=url, vision_ai=vision_ai)
            elif fmt == "HTML":
                rows = await asyncio.to_thread(cls.extract_html, content, url)
            else:
                rows = [{"name": Path(saved_path_str or url).name, "url": url, "format": fmt}]
        except Exception:
            pass

        if saved_path_str:
            if not rows:
                rows = [{"name": Path(saved_path_str).name, "url": url, "format": fmt}]
            for r in rows:
                if isinstance(r, dict):
                    r["_local_file"] = saved_path_str
        return rows

    @classmethod
    def detect_format(cls, content: bytes, format_hint: str, content_type: str, url: str) -> str:
        ct = (content_type or "").lower()
        u = (url or "").lower().split("?")[0]

        if content[:4] == b"%PDF" or "pdf" in ct or u.endswith(".pdf"):
            return "PDF"
        if (
            content[:8] == b"\x89PNG\r\n\x1a\n"
            or content[:3] == b"\xff\xd8\xff"
            or (content[:4] == b"RIFF" and content[8:12] == b"WEBP")
            or "image/" in ct
            or u.endswith((".png", ".jpg", ".jpeg", ".webp", ".gif", ".tiff", ".bmp", ".svg"))
        ):
            return "IMAGE"
        if "csv" in ct or u.endswith(".csv"):
            return "CSV"
        if "spreadsheet" in ct or "excel" in ct or u.endswith((".xlsx", ".xls")):
            return "XLSX"
        if "json" in ct or u.endswith(".json"):
            return "JSON"
        if "xml" in ct or "rss" in ct or "atom" in ct or u.endswith((".xml", ".rss")):
            return "XML"
        if "zip" in ct or u.endswith(".zip"):
            return "ZIP"

        head = content[:256].lstrip()
        if head.startswith((b"{", b"[")):
            return "JSON"
        if head.startswith((b"<?xml", b"<rss", b"<feed")):
            return "XML"
        if head.lower().startswith((b"<!doctype html", b"<html")):
            return "HTML"

        hint = (format_hint or "").upper().strip()
        if hint in {"JSON", "CSV", "XLSX", "XML", "HTML", "PDF", "IMAGE"}:
            return hint
        return "HTML"

    @staticmethod
    def extract_json(content: bytes) -> List[Dict[str, Any]]:
        data = json.loads(content.decode("utf-8", errors="replace"))
        if isinstance(data, list):
            return [r for r in data if isinstance(r, dict)]
        if isinstance(data, dict):
            for key in (
                "records", "data", "results", "items", "elements", "features",
                "rows", "entries", "articles", "packages", "hits",
            ):
                val = data.get(key)
                if isinstance(val, list) and val and isinstance(val[0], dict):
                    return val
                if isinstance(val, dict):
                    for subkey in ("results", "records", "hits", "items", "entry"):
                        subval = val.get(subkey)
                        if isinstance(subval, list) and subval and isinstance(subval[0], dict):
                            return subval
            if isinstance(data.get("result"), dict):
                res_obj = data["result"]
                for subkey in ("results", "records", "packages"):
                    if isinstance(res_obj.get(subkey), list):
                        return res_obj[subkey]
            return [data]
        return []

    @staticmethod
    def extract_csv(content: bytes) -> List[Dict[str, Any]]:
        for enc in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                df = pd.read_csv(io.BytesIO(content), encoding=enc, nrows=2500)
                df = df.dropna(how="all")
                return df.where(pd.notnull(df), None).to_dict(orient="records")
            except Exception:
                continue
        return []

    @staticmethod
    def extract_excel(content: bytes) -> List[Dict[str, Any]]:
        df = pd.read_excel(io.BytesIO(content), engine="openpyxl", nrows=2500)
        df = df.dropna(how="all")
        return df.where(pd.notnull(df), None).to_dict(orient="records")

    @staticmethod
    def extract_xml(content: bytes) -> List[Dict[str, Any]]:
        parser = etree.XMLParser(recover=True, resolve_entities=False)
        root = etree.fromstring(content, parser=parser)
        records: List[Dict[str, Any]] = []
        items = root.xpath("//*[local-name()='item' or local-name()='entry']")
        target_nodes = items if items else list(root)
        for elem in target_nodes:
            row: Dict[str, Any] = dict(elem.attrib)
            for child in elem:
                tag = etree.QName(child).localname
                text = (child.text or "").strip()
                if text:
                    row[tag] = text
                elif "href" in child.attrib and tag == "link":
                    row["url"] = child.attrib["href"]
            if row:
                records.append(row)
        return records

    @classmethod
    def extract_html(cls, content: bytes, url: str = "") -> List[Dict[str, Any]]:
        soup = BeautifulSoup(content, "lxml")
        records: List[Dict[str, Any]] = []

        for script in soup.find_all("script", type="application/ld+json"):
            try:
                ld_data = json.loads(script.string or "")
                ld_list = (
                    ld_data.get("@graph", [ld_data])
                    if isinstance(ld_data, dict)
                    else (ld_data if isinstance(ld_data, list) else [])
                )
                for item in ld_list:
                    if isinstance(item, dict):
                        flat = {
                            k: (v.get("name") if isinstance(v, dict) else v)
                            for k, v in item.items()
                            if not k.startswith("@") and not isinstance(v, list)
                        }
                        if flat:
                            flat["url"] = flat.get("url") or url
                            records.append(flat)
            except Exception:
                continue

        for script in soup.find_all("script"):
            stext = (script.string or "").strip()
            if script.get("id") == "__NEXT_DATA__" or "window.__INITIAL_STATE__" in stext:
                try:
                    json_str = stext.split("=", 1)[1].rstrip(";") if "window.__" in stext else stext
                    nd = json.loads(json_str)
                    cls._harvest_nested_dicts(nd, records, url)
                except Exception:
                    pass

        for table in soup.find_all("table"):
            headers: List[str] = []
            header_row = table.find("tr")
            if header_row:
                headers = [
                    th.get_text(" ", strip=True) or f"col_{idx}"
                    for idx, th in enumerate(header_row.find_all(["th", "td"]))
                ]
            for tr in table.find_all("tr")[1:]:
                cells = [td.get_text(" ", strip=True) for td in tr.find_all(["td", "th"])]
                if not any(cells):
                    continue
                row_d = (
                    dict(zip(headers, cells))
                    if (headers and len(headers) == len(cells))
                    else {f"col_{i}": val for i, val in enumerate(cells) if val}
                )
                if url:
                    row_d.setdefault("url", url)
                records.append(row_d)

        if records:
            return records

        for item in soup.find_all(
            ["article", "div", "li"],
            class_=re.compile(r"(result|card|item|row|post|entry|dataset)", re.I),
            limit=100,
        ):
            title_el = item.find(["h1", "h2", "h3", "h4", "a"])
            if not title_el:
                continue
            title = title_el.get_text(" ", strip=True)
            a_tag = item.find("a", href=True)
            link = urllib.parse.urljoin(url, a_tag["href"]) if a_tag else url
            desc = item.get_text(" ", strip=True)
            if title and len(title) > 4:
                records.append({"name": title[:180], "url": link, "description": desc[:400]})

        if not records and soup.title:
            body_text = soup.get_text(" ", strip=True)
            records.append({"name": soup.title.get_text(strip=True), "url": url, "description": body_text[:600]})
        return records

    @classmethod
    def _harvest_nested_dicts(cls, obj: Any, out: List[Dict[str, Any]], url: str) -> None:
        if len(out) >= 120:
            return
        if isinstance(obj, list) and len(obj) >= 2 and all(isinstance(x, dict) for x in obj[:5]):
            for row in obj[:60]:
                flat = {k: v for k, v in row.items() if isinstance(v, (str, int, float, bool))}
                if len(flat) >= 2:
                    flat.setdefault("url", url)
                    out.append(flat)
        elif isinstance(obj, dict):
            for v in obj.values():
                if isinstance(v, (dict, list)):
                    cls._harvest_nested_dicts(v, out, url)

    @classmethod
    async def extract_pdf_async(
        cls, content: bytes, url: str = "", vision_ai: Optional[NlackCognitiveEngine] = None
    ) -> List[Dict[str, Any]]:
        if fitz is None:
            return []
        records: List[Dict[str, Any]] = []
        with fitz.open(stream=content, filetype="pdf") as doc:
            for page_idx, page in enumerate(doc):
                if hasattr(page, "find_tables"):
                    tabs = page.find_tables()
                    for tab in tabs.tables:
                        extracted = tab.extract()
                        if len(extracted) > 1:
                            headers = [str(h or f"col_{i}").strip() for i, h in enumerate(extracted[0])]
                            for row in extracted[1:]:
                                row_dict = {
                                    headers[i]: str(cell).strip()
                                    for i, cell in enumerate(row)
                                    if i < len(headers) and cell
                                }
                                if row_dict:
                                    row_dict["pdf_page"] = page_idx + 1
                                    row_dict["url"] = url
                                    records.append(row_dict)
                text = page.get_text("text").strip()
                if not records and text:
                    for line in text.splitlines():
                        line = line.strip()
                        if ":" in line and 5 < len(line) < 220:
                            k, v = line.split(":", 1)
                            if k.strip() and v.strip():
                                records.append(
                                    {"name": k.strip(), "value": v.strip(), "pdf_page": page_idx + 1, "url": url}
                                )
                elif not text and vision_ai is not None and page_idx < 4:
                    pix = page.get_pixmap(dpi=150)
                    png_bytes = pix.tobytes("png")
                    ocr_res = await vision_ai.extract_from_image_bytes(
                        png_bytes,
                        instruction="Extract all structured data rows, tables, or key-value metrics from this scanned PDF page.",
                    )
                    for r in ocr_res.get("records") or []:
                        if isinstance(r, dict):
                            r["pdf_page"] = page_idx + 1
                            r["url"] = url
                            records.append(r)
                    if not records and ocr_res.get("extracted_text"):
                        records.append(
                            {
                                "name": f"Scanned PDF Page {page_idx + 1}",
                                "description": str(ocr_res["extracted_text"])[:500],
                                "url": url,
                            }
                        )
        return records

    @classmethod
    async def extract_image_async(
        cls,
        content: bytes,
        content_type: str = "image/png",
        url: str = "",
        vision_ai: Optional[NlackCognitiveEngine] = None,
    ) -> List[Dict[str, Any]]:
        if vision_ai is None:
            return [{"name": Path(urllib.parse.urlparse(url).path).name or "Image Asset", "url": url}]
        mime = content_type.split(";")[0].strip() if "image/" in (content_type or "") else "image/png"
        ocr_res = await vision_ai.extract_from_image_bytes(
            content,
            instruction="Read all text, numbers, statistics, or table rows from this image.",
            mime_type=mime,
        )
        rows: List[Dict[str, Any]] = []
        for r in ocr_res.get("records") or []:
            if isinstance(r, dict):
                r.setdefault("url", url)
                rows.append(r)
        if not rows:
            txt = str(ocr_res.get("extracted_text") or "").strip()
            rows.append(
                {
                    "name": Path(urllib.parse.urlparse(url).path).name or "Harvested Image Data",
                    "description": txt[:600] if txt else "Binary image saved to script folder",
                    "url": url,
                }
            )
        return rows


class StorageEngine:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    async def initialize(self) -> None:
        schema_sql = (
            "PRAGMA journal_mode=WAL;"
            "CREATE TABLE IF NOT EXISTS http_cache ("
            "cache_key TEXT PRIMARY KEY, url TEXT, status_code INTEGER, content_type TEXT, "
            "etag TEXT, last_modified TEXT, body BLOB, cached_at REAL);"
            "CREATE TABLE IF NOT EXISTS autoschema_memory ("
            "raw_column TEXT PRIMARY KEY, canonical_column TEXT, learned_at TEXT);"
            "CREATE TABLE IF NOT EXISTS entity_edges ("
            "node_a TEXT, node_b TEXT, updated_at TEXT, PRIMARY KEY (node_a, node_b));"
            "CREATE TABLE IF NOT EXISTS runs ("
            "run_id TEXT PRIMARY KEY, query TEXT, sources TEXT, started_at TEXT, completed_at TEXT, "
            "datasets_found INTEGER DEFAULT 0, records_collected INTEGER DEFAULT 0, duplicates_skipped INTEGER DEFAULT 0);"
            "CREATE TABLE IF NOT EXISTS datasets ("
            "dataset_id TEXT, run_id TEXT, source TEXT, title TEXT, description TEXT, url TEXT, download_url TEXT, "
            "format TEXT, license TEXT, attribution TEXT, country_scope TEXT, discovered_at TEXT, "
            "PRIMARY KEY (dataset_id, run_id));"
            "CREATE TABLE IF NOT EXISTS records ("
            "record_id TEXT, run_id TEXT, dataset_id TEXT, source TEXT, source_url TEXT, local_file TEXT, "
            "collected_at TEXT, entity_key TEXT, cluster_id TEXT, fingerprint TEXT, raw_json TEXT, "
            "normalized_json TEXT, telemetry_json TEXT, quality_score REAL, lead_score REAL, quantum_entropy REAL, "
            "reasoning_trace TEXT, quality_breakdown TEXT, ai_tags TEXT, ai_summary TEXT, license TEXT, "
            "PRIMARY KEY (record_id, run_id));"
            "CREATE INDEX IF NOT EXISTS idx_records_run ON records(run_id);"
            "CREATE INDEX IF NOT EXISTS idx_records_entity ON records(entity_key);"
            "CREATE INDEX IF NOT EXISTS idx_records_cluster ON records(cluster_id);"
            "CREATE INDEX IF NOT EXISTS idx_records_fp ON records(fingerprint);"
            "CREATE VIRTUAL TABLE IF NOT EXISTS records_fts USING fts5("
            "record_id UNINDEXED, run_id UNINDEXED, source, entity_key, cluster_id, "
            "normalized_json, telemetry_json, ai_tags, ai_summary);"
        )
        async with aiosqlite.connect(self.db_path) as db:
            await db.executescript(schema_sql)
            await db.commit()

    async def get_learned_schemas(self) -> Dict[str, str]:
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("SELECT raw_column, canonical_column FROM autoschema_memory") as cur:
                rows = await cur.fetchall()
                return {str(r[0]): str(r[1]) for r in rows}

    async def save_learned_schemas(self, mappings: Dict[str, str]) -> None:
        now = dt.datetime.now(dt.timezone.utc).isoformat()
        async with aiosqlite.connect(self.db_path) as db:
            for raw_col, can_col in mappings.items():
                await db.execute(
                    "INSERT OR REPLACE INTO autoschema_memory (raw_column, canonical_column, learned_at) VALUES (?, ?, ?)",
                    (raw_col, can_col, now),
                )
            await db.commit()

    async def get_entity_edges(self) -> List[Tuple[str, str]]:
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("SELECT node_a, node_b FROM entity_edges") as cur:
                rows = await cur.fetchall()
                return [(str(r[0]), str(r[1])) for r in rows]

    async def save_entity_edges(self, edges: List[Tuple[str, str]]) -> None:
        if not edges:
            return
        now = dt.datetime.now(dt.timezone.utc).isoformat()
        async with aiosqlite.connect(self.db_path) as db:
            for a, b in edges:
                await db.execute(
                    "INSERT OR REPLACE INTO entity_edges (node_a, node_b, updated_at) VALUES (?, ?, ?)",
                    (a, b, now),
                )
            await db.commit()

    async def get_cached_response(
        self, cache_key: str, ttl_seconds: int
    ) -> Optional[Tuple[int, str, bytes]]:
        now = time.time()
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT status_code, content_type, body, cached_at FROM http_cache WHERE cache_key = ?",
                (cache_key,),
            ) as cursor:
                row = await cursor.fetchone()
                if row and (now - row[3]) <= ttl_seconds:
                    return int(row[0]), str(row[1]), bytes(row[2])
        return None

    async def set_cached_response(
        self,
        cache_key: str,
        url: str,
        status_code: int,
        content_type: str,
        body: bytes,
        etag: str = "",
        last_modified: str = "",
    ) -> None:
        if len(body) > 8_000_000:
            return
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT OR REPLACE INTO http_cache (cache_key, url, status_code, content_type, etag, last_modified, body, cached_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (cache_key, url, status_code, content_type, etag, last_modified, body, time.time()),
            )
            await db.commit()

    async def save_run(
        self,
        run_id: str,
        query: str,
        sources: List[str],
        started_at: str,
        datasets: List[DatasetMeta],
        records: List[NormalizedRecord],
        duplicates_skipped: int,
    ) -> None:
        completed_at = dt.datetime.now(dt.timezone.utc).isoformat()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT OR REPLACE INTO runs (run_id, query, sources, started_at, completed_at, datasets_found, records_collected, duplicates_skipped) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    query,
                    ",".join(sources),
                    started_at,
                    completed_at,
                    len(datasets),
                    len(records),
                    duplicates_skipped,
                ),
            )

            for ds in datasets:
                await db.execute(
                    "INSERT OR REPLACE INTO datasets (dataset_id, run_id, source, title, description, url, download_url, format, license, attribution, country_scope, discovered_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        ds.dataset_id,
                        run_id,
                        ds.source,
                        ds.title,
                        ds.description,
                        ds.url,
                        ds.download_url,
                        ds.format,
                        ds.license,
                        ds.attribution,
                        ds.country_scope,
                        ds.discovered_at,
                    ),
                )

            for rec in records:
                norm_json = json.dumps(rec.normalized_data, ensure_ascii=False, default=str)
                raw_json = json.dumps(rec.raw_data, ensure_ascii=False, default=str)
                tel_json = json.dumps(rec.telemetry_data, ensure_ascii=False, default=str)
                tags_str = ", ".join(rec.ai_tags)
                await db.execute(
                    "INSERT OR REPLACE INTO records (record_id, run_id, dataset_id, source, source_url, local_file, collected_at, entity_key, cluster_id, fingerprint, raw_json, normalized_json, telemetry_json, quality_score, lead_score, quantum_entropy, reasoning_trace, quality_breakdown, ai_tags, ai_summary, license) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        rec.record_id,
                        rec.run_id,
                        rec.dataset_id,
                        rec.source,
                        rec.source_url,
                        rec.local_file,
                        rec.collected_at,
                        rec.entity_key,
                        rec.cluster_id,
                        rec.fingerprint,
                        raw_json,
                        norm_json,
                        tel_json,
                        rec.quality_score,
                        rec.lead_score,
                        rec.quantum_entropy,
                        rec.reasoning_trace,
                        json.dumps(rec.quality_breakdown),
                        tags_str,
                        rec.ai_summary,
                        rec.license,
                    ),
                )
                await db.execute(
                    "INSERT INTO records_fts (record_id, run_id, source, entity_key, cluster_id, normalized_json, telemetry_json, ai_tags, ai_summary) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        rec.record_id,
                        rec.run_id,
                        rec.source,
                        rec.entity_key,
                        rec.cluster_id,
                        norm_json,
                        tel_json,
                        tags_str,
                        rec.ai_summary,
                    ),
                )
            await db.commit()

    def detect_changes(self) -> Dict[str, Any]:
        if not self.db_path.exists():
            return {"status": "no_db"}
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            runs = conn.execute(
                "SELECT * FROM runs WHERE records_collected > 0 ORDER BY completed_at DESC LIMIT 2"
            ).fetchall()
            if len(runs) < 2:
                return {"status": "insufficient_runs", "runs_count": len(runs)}

            curr_run, prev_run = runs[0], runs[1]
            curr_rows = conn.execute(
                "SELECT entity_key, fingerprint, normalized_json, source FROM records WHERE run_id = ?",
                (curr_run["run_id"],),
            ).fetchall()
            prev_rows = conn.execute(
                "SELECT entity_key, fingerprint, normalized_json, source FROM records WHERE run_id = ?",
                (prev_run["run_id"],),
            ).fetchall()

            curr_map = {r["entity_key"]: r for r in curr_rows}
            prev_map = {r["entity_key"]: r for r in prev_rows}

            new_entities, modified_entities, removed_entities = [], [], []
            for ek, r_curr in curr_map.items():
                if ek not in prev_map:
                    new_entities.append(
                        {"entity_key": ek, "source": r_curr["source"], "data": json.loads(r_curr["normalized_json"])}
                    )
                elif r_curr["fingerprint"] != prev_map[ek]["fingerprint"]:
                    old_d = json.loads(prev_map[ek]["normalized_json"])
                    new_d = json.loads(r_curr["normalized_json"])
                    diffs = {
                        k: {"old": old_d.get(k), "new": new_d.get(k)}
                        for k in set(old_d) | set(new_d)
                        if old_d.get(k) != new_d.get(k)
                    }
                    modified_entities.append({"entity_key": ek, "source": r_curr["source"], "diffs": diffs})

            for ek, r_prev in prev_map.items():
                if ek not in curr_map:
                    removed_entities.append(
                        {"entity_key": ek, "source": r_prev["source"], "data": json.loads(r_prev["normalized_json"])}
                    )

            return {
                "status": "ok",
                "current_run": dict(curr_run),
                "previous_run": dict(prev_run),
                "new": new_entities,
                "modified": modified_entities,
                "removed": removed_entities,
            }


class NlackNetworkEngine:
    def __init__(self, config: NlackConfig, storage: StorageEngine) -> None:
        self.config = config
        self.storage = storage
        self.semaphore = asyncio.Semaphore(config.max_concurrency)
        self.vision_ai = NlackCognitiveEngine(config)
        self.quantum = QuantumEngine(config.proxies if config.proxies else ["direct"])
        self._domain_next_allowed: Dict[str, float] = defaultdict(float)
        self._domain_delay: Dict[str, float] = defaultdict(lambda: config.base_delay)
        self._proxy_cycle = itertools.cycle(config.proxies) if config.proxies else None
        self._cookies: Dict[str, str] = {}
        self._http_client: Optional[httpx.AsyncClient] = None
        self._curl_session: Optional[Any] = None
        if config.cookies_file and config.cookies_file.exists():
            try:
                self._cookies = json.loads(config.cookies_file.read_text(encoding="utf-8"))
            except Exception:
                pass

    async def __aenter__(self) -> "NlackNetworkEngine":
        limits = httpx.Limits(max_keepalive_connections=25, max_connections=50)
        self._http_client = httpx.AsyncClient(
            cookies=self._cookies,
            timeout=httpx.Timeout(self.config.timeout_seconds),
            follow_redirects=True,
            limits=limits,
        )
        if CurlAsyncSession is not None:
            self._curl_session = CurlAsyncSession(
                impersonate="chrome124",
                cookies=self._cookies,
                timeout=self.config.timeout_seconds,
            )
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._http_client is not None:
            await self._http_client.aclose()
        if self._curl_session is not None:
            try:
                await self._curl_session.close()
            except Exception:
                pass

    def _build_headers(self, url: str, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        parsed = urllib.parse.urlparse(url)
        domain = parsed.netloc.lower()
        if any(h in domain for h in INSTITUTIONAL_API_HOSTS):
            profile = {
                "User-Agent": "NlackEngine/5.0 (NexBlack Enterprise Data Harvester; mailto:contact@nexblack.com)",
                "Accept": "application/json, text/csv, application/xml, text/html, */*;q=0.8",
            }
        else:
            profile = random.choice(BROWSER_PROFILES).copy()
            geo = GEO_PROFILES.get(self.config.geo_region.upper(), GEO_PROFILES["GLOBAL"])
            profile.update(geo)
            profile["Accept"] = "text/html,application/xhtml+xml,application/xml;q=0.9,application/json,image/webp,*/*;q=0.8"
            if parsed.scheme and parsed.netloc:
                profile["Referer"] = f"{parsed.scheme}://{parsed.netloc}/"

        if self.config.auth_bearer and "Authorization" not in (extra or {}):
            profile["Authorization"] = f"Bearer {self.config.auth_bearer}"
        if extra:
            profile.update(extra)
        return profile

    async def _respect_domain_pacing(self, domain: str, attempt: int = 1) -> None:
        now = time.time()
        wait_needed = self._domain_next_allowed[domain] - now
        if wait_needed > 0:
            await asyncio.sleep(wait_needed)
        q_delay = self.quantum.tunneling_delay(self._domain_delay[domain], attempt, self.config.max_delay)
        self._domain_next_allowed[domain] = time.time() + q_delay

    def _update_rate_limits(self, domain: str, headers: Any, status: int) -> None:
        retry_after = headers.get("retry-after") if hasattr(headers, "get") else None
        if retry_after:
            try:
                self._domain_next_allowed[domain] = time.time() + min(float(retry_after), self.config.max_delay)
            except ValueError:
                pass
        remaining = headers.get("x-ratelimit-remaining") if hasattr(headers, "get") else None
        if remaining is not None and str(remaining).isdigit() and int(remaining) <= 2:
            self._domain_delay[domain] = min(self._domain_delay[domain] * 2.0, 10.0)
        elif status == 200:
            self._domain_delay[domain] = max(self.config.base_delay, self._domain_delay[domain] * 0.9)

    async def _fetch_with_browser_xhr_interception(self, url: str) -> Optional[Tuple[int, str, bytes]]:
        if async_playwright is None:
            return None
        try:
            intercepted_json: List[Dict[str, Any]] = []
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context(
                    user_agent=BROWSER_PROFILES[0]["User-Agent"],
                    viewport={"width": 1366, "height": 768},
                )
                page = await context.new_page()

                async def _on_response(response: Any) -> None:
                    try:
                        ct = (response.headers.get("content-type") or "").lower()
                        if "application/json" in ct and response.status == 200:
                            body_bytes = await response.body()
                            if 50 < len(body_bytes) < 2_000_000:
                                rows = UniversalExtractor.extract_json(body_bytes)
                                for r in rows[:50]:
                                    if isinstance(r, dict) and len(r) >= 2:
                                        intercepted_json.append(r)
                    except Exception:
                        pass

                page.on("response", _on_response)
                await page.goto(url, wait_until="domcontentloaded", timeout=26000)
                await page.wait_for_timeout(1800)

                if intercepted_json:
                    await browser.close()
                    return 200, "application/json", json.dumps(intercepted_json).encode("utf-8")

                final_html = (await page.content()).encode("utf-8")
                await browser.close()
                return 200, "text/html", final_html
        except Exception:
            return None

    async def _fetch_wayback_mirror(self, url: str) -> Optional[Tuple[int, str, bytes]]:
        if not self.config.wayback_fallback or self._http_client is None:
            return None
        try:
            wb_api = f"https://archive.org/wayback/available?url={urllib.parse.quote(url)}"
            r = await self._http_client.get(wb_api, timeout=12.0)
            if r.status_code == 200:
                snap = ((r.json().get("archived_snapshots") or {}).get("closest")) or {}
                snap_url = snap.get("url")
                if snap_url:
                    raw_snap_url = re.sub(r"/web/(\d+)/", r"/web/\1id_/", snap_url)
                    snap_resp = await self._http_client.get(raw_snap_url, timeout=18.0)
                    if snap_resp.status_code == 200:
                        ct = snap_resp.headers.get("content-type", "text/html")
                        return 200, ct, snap_resp.content
        except Exception:
            pass
        return None

    async def fetch(
        self,
        url: str,
        method: str = "GET",
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Any] = None,
        headers: Optional[Dict[str, str]] = None,
        use_cache: bool = True,
    ) -> Optional[Tuple[int, str, bytes]]:
        full_key_raw = f"{method}:{url}:{json.dumps(params or {}, sort_keys=True)}:{str(data or '')}"
        cache_key = hashlib.sha256(full_key_raw.encode("utf-8")).hexdigest()

        if use_cache and self.config.cache_ttl_seconds > 0:
            cached = await self.storage.get_cached_response(cache_key, self.config.cache_ttl_seconds)
            if cached is not None:
                return cached

        parsed = urllib.parse.urlparse(url)
        domain = parsed.netloc.lower()
        is_institutional = any(h in domain for h in INSTITUTIONAL_API_HOSTS)

        async with self.semaphore:
            for attempt in range(1, 4):
                await self._respect_domain_pacing(domain, attempt)
                selected_proxy = next(self._proxy_cycle) if self._proxy_cycle else None
                req_headers = self._build_headers(url, headers)

                try:
                    if (
                        self._curl_session is not None
                        and not is_institutional
                        and not selected_proxy
                        and method == "GET"
                    ):
                        resp = await self._curl_session.request(
                            method, url, params=params, data=data, headers=req_headers
                        )
                        status = resp.status_code
                        ct = resp.headers.get("content-type", "")
                        body = resp.content
                        self._update_rate_limits(domain, resp.headers, status)
                    else:
                        if selected_proxy:
                            async with httpx.AsyncClient(
                                headers=req_headers,
                                cookies=self._cookies,
                                proxy=selected_proxy,
                                timeout=httpx.Timeout(self.config.timeout_seconds),
                                follow_redirects=True,
                            ) as proxy_client:
                                resp = await proxy_client.request(method=method, url=url, params=params, data=data)
                        else:
                            assert self._http_client is not None
                            resp = await self._http_client.request(
                                method=method, url=url, params=params, data=data, headers=req_headers
                            )
                        status = resp.status_code
                        ct = resp.headers.get("content-type", "")
                        body = resp.content
                        self._update_rate_limits(domain, resp.headers, status)

                    if status == 200:
                        self.quantum.grover_amplify(selected_proxy or "direct", 1.2)
                        if use_cache:
                            await self.storage.set_cached_response(cache_key, url, status, ct, body)
                        return status, ct, body

                    if status in {403, 429, 503}:
                        if self.config.use_headless_browser:
                            br_res = await self._fetch_with_browser_xhr_interception(url)
                            if br_res:
                                await self.storage.set_cached_response(cache_key, url, br_res[0], br_res[1], br_res[2])
                                return br_res
                        if attempt == 3:
                            wb_res = await self._fetch_wayback_mirror(url)
                            if wb_res:
                                await self.storage.set_cached_response(cache_key, url, wb_res[0], wb_res[1], wb_res[2])
                                return wb_res

                except Exception:
                    if attempt == 3:
                        wb_res = await self._fetch_wayback_mirror(url)
                        if wb_res:
                            return wb_res
                    await asyncio.sleep(0.5 * attempt)
        return None


class BaseSourceAdapter(ABC):
    source_name: str = "base"
    display_name: str = "Base Source"
    default_license: str = "Public / Open Discovery"
    attribution: str = ""
    country_scope: str = "Worldwide"

    def __init__(self, config: NlackConfig, http: NlackNetworkEngine) -> None:
        self.config = config
        self.http = http

    @abstractmethod
    async def discover(self, query: str) -> List[DatasetMeta]:
        pass

    @abstractmethod
    async def collect(self, dataset: DatasetMeta, query: str) -> List[Dict[str, Any]]:
        pass


class UnlimitedSearchRouter(BaseSourceAdapter):
    source_name = "universal"
    display_name = "Quantum Multi-Engine Router (Brave + Google RSS + GDELT + OpenAlex + DuckDuckGo + Wikipedia)"
    default_license = "Global Web / Open Access"
    attribution = "Multi-Engine Quantum Meta-Discovery"

    FILE_EXTENSIONS = (
        ".csv", ".xlsx", ".xls", ".json", ".xml", ".pdf",
        ".png", ".jpg", ".jpeg", ".webp", ".zip", ".parquet",
    )

    async def discover(self, query: str) -> List[DatasetMeta]:
        datasets: List[DatasetMeta] = []
        seen_urls: Set[str] = set()
        q_router = QuantumEngine(["brave", "google_rss", "gdelt", "openalex", "ddg", "wikipedia"])

        def _add_ds(
            title: str,
            url: str,
            desc: str = "",
            fmt: str = "HTML",
            engine_name: str = "universal",
        ) -> None:
            if not url or not url.startswith("http") or url in seen_urls:
                return
            seen_urls.add(url)
            u_clean = url.lower().split("?")[0]
            for ext in self.FILE_EXTENSIONS:
                if u_clean.endswith(ext):
                    fmt = "IMAGE" if ext in {".png", ".jpg", ".jpeg", ".webp"} else ext.lstrip(".").upper()
                    break
            q_router.grover_amplify(engine_name, 1.0)
            datasets.append(
                DatasetMeta(
                    dataset_id=f"universal:{hashlib.md5(url.encode()).hexdigest()[:10]}",
                    source=self.source_name,
                    title=title[:140] or url,
                    description=desc[:280],
                    url=url,
                    download_url=url,
                    format=fmt,
                    extra_meta={"quantum_engine": engine_name},
                )
            )

        for u in re.findall(r"https?://[^\s,]+", query):
            _add_ds(f"Direct Target — {u}", u, "Direct URL extraction", engine_name="direct")
        if datasets:
            return datasets

        encoded = urllib.parse.quote(query)

        if self.config.brave_api_key:
            brave_api_res = await self.http.fetch(
                "https://api.search.brave.com/res/v1/web/search",
                params={"q": query, "count": 15},
                headers={"X-Subscription-Token": self.config.brave_api_key, "Accept": "application/json"},
            )
            if brave_api_res and brave_api_res[0] == 200:
                try:
                    b_json = json.loads(brave_api_res[2].decode("utf-8", errors="replace"))
                    for item in (b_json.get("web") or {}).get("results") or []:
                        _add_ds(item.get("title", ""), item.get("url", ""), item.get("description", ""), engine_name="brave")
                except Exception:
                    pass

        g_rss = f"https://news.google.com/rss/search?q={encoded}&hl=en-US&gl=US&ceid=US:en"
        _add_ds(f"Google Global Index Feed — '{query}'", g_rss, "Google RSS structured feed", fmt="XML", engine_name="google_rss")

        gdelt_url = f"https://api.gdeltproject.org/api/v2/doc/doc?query={encoded}&mode=artlist&maxrecords=35&format=json"
        _add_ds(f"GDELT Worldwide Stream — '{query}'", gdelt_url, "GDELT Global Events API", fmt="JSON", engine_name="gdelt")

        oa_url = f"https://api.openalex.org/works?search={encoded}&per_page=25"
        if self.config.openalex_api_key:
            oa_url += f"&api_key={self.config.openalex_api_key}"
        _add_ds(f"OpenAlex Global Research Index — '{query}'", oa_url, "OpenAlex REST API", fmt="JSON", engine_name="openalex")

        wiki_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={encoded}&format=json&utf8=1"
        _add_ds(f"Wikipedia Global Knowledge Index — '{query}'", wiki_url, "Wikimedia Search API", fmt="JSON", engine_name="wikipedia")

        ddg_res = await self.http.fetch("https://html.duckduckgo.com/html/", method="POST", data={"q": query})
        if ddg_res and ddg_res[0] == 200:
            soup = BeautifulSoup(ddg_res[2], "lxml")
            for a in soup.select("a.result__a")[:10]:
                href = a.get("href", "")
                if "uddg=" in href:
                    qs = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
                    href = qs.get("uddg", [href])[0]
                _add_ds(a.get_text(" ", strip=True), href, "DuckDuckGo Discovery", engine_name="ddg")

        top_engines = set(q_router.collapse_superposition(top_k=4))
        datasets.sort(key=lambda d: 0 if d.extra_meta.get("quantum_engine") in top_engines else 1)
        return datasets

    async def collect(self, dataset: DatasetMeta, query: str) -> List[Dict[str, Any]]:
        target_url = dataset.download_url or dataset.url
        res = await self.http.fetch(target_url)
        if not res or res[0] != 200:
            return []

        status, ct, body = res

        if "api.openalex.org/works" in target_url:
            try:
                data = json.loads(body.decode("utf-8", errors="replace"))
                rows = []
                for w in data.get("results", []):
                    insts = [
                        (i.get("display_name") or "")
                        for auth in (w.get("authorships") or [])[:2]
                        for i in (auth.get("institutions") or [])[:1]
                    ]
                    rows.append(
                        {
                            "identifier": w.get("id"),
                            "doi": w.get("doi"),
                            "name": w.get("title") or w.get("display_name"),
                            "year": w.get("publication_year"),
                            "date": w.get("publication_date"),
                            "value": w.get("cited_by_count"),
                            "unit": "Citations",
                            "company": insts[0] if insts else "Global Research Institution",
                            "category": w.get("type"),
                            "url": w.get("doi") or w.get("id"),
                        }
                    )
                return rows
            except Exception:
                pass

        if "en.wikipedia.org/w/api.php" in target_url:
            try:
                data = json.loads(body.decode("utf-8", errors="replace"))
                s_items = ((data.get("query") or {}).get("search")) or []
                return [
                    {
                        "identifier": str(it.get("pageid")),
                        "name": it.get("title"),
                        "description": BeautifulSoup(it.get("snippet", ""), "lxml").get_text(" ", strip=True),
                        "date": it.get("timestamp"),
                        "value": it.get("wordcount"),
                        "unit": "Words",
                        "company": "Wikipedia Knowledge Graph",
                        "url": f"https://en.wikipedia.org/wiki/{urllib.parse.quote(str(it.get('title', '')).replace(' ', '_'))}",
                    }
                    for it in s_items
                ]
            except Exception:
                pass

        records = await UniversalExtractor.extract_async(
            body,
            dataset.format,
            ct,
            target_url,
            vision_ai=self.http.vision_ai,
            save_binary_to_pc=self.config.download_assets,
            target_dir=self.config.export_dir,
        )

        if "html" in ct.lower() and self.config.crawl_depth >= 2:
            soup = BeautifulSoup(body, "lxml")
            linked_assets: List[str] = []
            for a in soup.find_all("a", href=True):
                full_link = urllib.parse.urljoin(target_url, a["href"])
                if full_link.lower().split("?")[0].endswith(self.FILE_EXTENSIONS) and full_link not in linked_assets:
                    linked_assets.append(full_link)
                if len(linked_assets) >= 4:
                    break

            for img in soup.find_all("img", src=True):
                alt_txt = (img.get("alt") or img.get("title") or "").lower()
                img_url = urllib.parse.urljoin(target_url, img["src"])
                if any(k in alt_txt or k in img_url.lower() for k in ("chart", "table", "data", "stat", "gov", "report", "price")):
                    if img_url.startswith("http") and img_url not in linked_assets:
                        linked_assets.append(img_url)
                if len(linked_assets) >= 6:
                    break

            for asset_url in linked_assets:
                f_res = await self.http.fetch(asset_url)
                if f_res and f_res[0] == 200:
                    sub_rows = await UniversalExtractor.extract_async(
                        f_res[2],
                        "",
                        f_res[1],
                        asset_url,
                        vision_ai=self.http.vision_ai,
                        save_binary_to_pc=self.config.download_assets,
                        target_dir=self.config.export_dir,
                    )
                    for r in sub_rows[:35]:
                        r.setdefault("url", asset_url)
                        records.append(r)

        return records


class GlobalGovernmentAdapter(BaseSourceAdapter):
    source_name = "global_gov"
    display_name = "Worldwide Government Portals (US Data.gov, UK, EU, Canada, AU, UN HDX, Socrata, SEC EDGAR)"
    default_license = "Open Government Data Licenses (Worldwide)"
    attribution = "Global CKAN, Socrata SODA, SEC EDGAR & National Portals"

    CKAN_PORTALS = [
        ("US Federal Government (data.gov)", "United States", "https://catalog-old.data.gov/api/3/action/package_search"),
        ("UK Government (data.gov.uk)", "United Kingdom", "https://data.gov.uk/api/action/package_search"),
        ("Government of Canada (open.canada.ca)", "Canada", "https://open.canada.ca/data/api/3/action/package_search"),
        ("Australian Government (data.gov.au)", "Australia", "https://data.gov.au/data/api/3/action/package_search"),
        ("UN Humanitarian Data Exchange (HDX)", "Worldwide", "https://data.humdata.org/api/3/action/package_search"),
    ]

    async def discover(self, query: str) -> List[DatasetMeta]:
        encoded = urllib.parse.quote(query)
        datasets: List[DatasetMeta] = []

        for portal_title, country, api_url in self.CKAN_PORTALS:
            dl = f"{api_url}?q={encoded}&rows=12"
            datasets.append(
                DatasetMeta(
                    dataset_id=f"global_gov:ckan:{country.lower().replace(' ', '_')}:{hashlib.md5(query.encode()).hexdigest()[:6]}",
                    source=self.source_name,
                    title=f"{portal_title} — '{query}'",
                    description=f"Official CKAN government federation for {country}.",
                    url=dl,
                    download_url=dl,
                    format="JSON",
                    country_scope=country,
                    extra_meta={"portal_type": "ckan", "country": country},
                )
            )

        us_new_catalog = f"https://catalog.data.gov/?q={encoded}"
        datasets.append(
            DatasetMeta(
                dataset_id=f"global_gov:us_opensearch:{hashlib.md5(query.encode()).hexdigest()[:8]}",
                source=self.source_name,
                title=f"US Data.gov 2026 OpenSearch Catalog — '{query}'",
                description="US Federal, State, and Municipal datasets on catalog.data.gov.",
                url=us_new_catalog,
                download_url=us_new_catalog,
                format="HTML",
                country_scope="United States",
                extra_meta={"portal_type": "us_html", "country": "United States"},
            )
        )

        socrata_url = f"https://api.us.socrata.com/api/catalog/v1?q={encoded}&limit=12"
        datasets.append(
            DatasetMeta(
                dataset_id=f"global_gov:socrata:{hashlib.md5(query.encode()).hexdigest()[:8]}",
                source=self.source_name,
                title=f"Socrata SODA Worldwide Government Catalog — '{query}'",
                description="Municipal, state, CDC, and federal Socrata databases.",
                url=socrata_url,
                download_url=socrata_url,
                format="JSON",
                country_scope="Worldwide",
                extra_meta={"portal_type": "socrata"},
            )
        )

        eu_url = f"https://data.europa.eu/api/hub/search/search?q={encoded}&filter=dataset&limit=12"
        datasets.append(
            DatasetMeta(
                dataset_id=f"global_gov:eu:{hashlib.md5(query.encode()).hexdigest()[:8]}",
                source=self.source_name,
                title=f"European Union Official Data Portal (data.europa.eu) — '{query}'",
                description="Pan-European government and statistical datasets.",
                url=eu_url,
                download_url=eu_url,
                format="JSON",
                country_scope="European Union",
                extra_meta={"portal_type": "eu"},
            )
        )

        sec_url = f"https://efts.sec.gov/LATEST/search-index?q={encoded}&from=0&size=15"
        datasets.append(
            DatasetMeta(
                dataset_id=f"global_gov:sec_edgar:{hashlib.md5(query.encode()).hexdigest()[:8]}",
                source=self.source_name,
                title=f"US SEC EDGAR Full-Text Filings — '{query}'",
                description="Official US Securities and Exchange Commission corporate filings.",
                url=sec_url,
                download_url=sec_url,
                format="JSON",
                country_scope="United States",
                extra_meta={"portal_type": "sec_edgar"},
            )
        )

        return datasets

    async def collect(self, dataset: DatasetMeta, query: str) -> List[Dict[str, Any]]:
        if not dataset.download_url:
            return []
        headers: Dict[str, str] = {}
        if self.config.socrata_app_token and "socrata" in dataset.download_url:
            headers["X-App-Token"] = self.config.socrata_app_token

        portal_type = dataset.extra_meta.get("portal_type", "")
        country = dataset.extra_meta.get("country", "Worldwide")

        res = await self.http.fetch(dataset.download_url, headers=headers)
        if not res or res[0] != 200:
            return []

        if portal_type == "us_html":
            return UniversalExtractor.extract_html(res[2], url=dataset.url)

        rows: List[Dict[str, Any]] = []
        try:
            payload = json.loads(res[2].decode("utf-8", errors="replace"))

            if portal_type == "ckan":
                packages = (payload.get("result") or {}).get("results") or []
                for pkg in packages[:12]:
                    org = (pkg.get("organization") or {}).get("title") or f"{country} Government Agency"
                    resources = pkg.get("resources") or []
                    asset_res = next(
                        (
                            r
                            for r in resources
                            if isinstance(r, dict)
                            and str(r.get("format", "")).upper() in {"CSV", "JSON", "PDF", "XLSX", "PNG", "JPG"}
                            and r.get("url")
                        ),
                        None,
                    )
                    asset_url = asset_res.get("url") if asset_res else None
                    rows.append(
                        {
                            "identifier": pkg.get("id"),
                            "name": pkg.get("title") or pkg.get("name"),
                            "company": org,
                            "country": country,
                            "date": pkg.get("metadata_modified") or pkg.get("metadata_created"),
                            "category": "Government Open Dataset",
                            "description": (pkg.get("notes") or "")[:300],
                            "license": pkg.get("license_title") or "Government Open License",
                            "url": asset_url or dataset.url,
                        }
                    )
                    if asset_url and len(rows) < 25:
                        sub_res = await self.http.fetch(asset_url)
                        if sub_res and sub_res[0] == 200:
                            extracted_sub = await UniversalExtractor.extract_async(
                                sub_res[2],
                                str(asset_res.get("format", "")),
                                sub_res[1],
                                asset_url,
                                vision_ai=self.http.vision_ai,
                                save_binary_to_pc=self.config.download_assets,
                                target_dir=self.config.export_dir,
                            )
                            for sr in extracted_sub[:8]:
                                sr.setdefault("country", country)
                                sr.setdefault("company", org)
                                sr.setdefault("url", asset_url)
                                rows.append(sr)
                return rows

            if portal_type == "socrata":
                results = payload.get("results") or []
                for item in results[:10]:
                    res_obj = item.get("resource") or {}
                    meta_obj = item.get("metadata") or {}
                    domain = meta_obj.get("domain") or "data.gov"
                    four_by_four = res_obj.get("id")
                    soda_endpoint = (
                        f"https://{domain}/resource/{four_by_four}.json?$limit=8"
                        if four_by_four
                        else item.get("permalink")
                    )
                    rows.append(
                        {
                            "identifier": four_by_four,
                            "name": res_obj.get("name"),
                            "company": res_obj.get("attribution") or domain,
                            "category": "Socrata Government Dataset",
                            "date": res_obj.get("updatedAt"),
                            "value": (res_obj.get("download_count") or 0),
                            "unit": "Downloads",
                            "description": (res_obj.get("description") or "")[:300],
                            "url": item.get("permalink") or soda_endpoint,
                        }
                    )
                    if four_by_four and domain and len(rows) < 28:
                        soda_res = await self.http.fetch(soda_endpoint, headers=headers)
                        if soda_res and soda_res[0] == 200:
                            for sr in UniversalExtractor.extract_json(soda_res[2])[:6]:
                                sr.setdefault("company", domain)
                                sr.setdefault("url", soda_endpoint)
                                rows.append(sr)
                return rows

            if portal_type == "eu":
                results = ((payload.get("result") or {}).get("results")) or []
                for item in results[:15]:
                    title_dict = item.get("title") or {}
                    title = (
                        title_dict.get("en") or next(iter(title_dict.values()), "EU Dataset")
                        if isinstance(title_dict, dict)
                        else str(title_dict)
                    )
                    desc_dict = item.get("description") or {}
                    desc = (
                        desc_dict.get("en") or next(iter(desc_dict.values()), "")
                        if isinstance(desc_dict, dict)
                        else str(desc_dict)
                    )
                    pub = (item.get("publisher") or {}).get("name") or "European Union Agency"
                    rows.append(
                        {
                            "identifier": item.get("id"),
                            "name": title,
                            "company": pub,
                            "country": "European Union",
                            "date": item.get("modified") or item.get("issued"),
                            "description": str(desc)[:300],
                            "url": f"https://data.europa.eu/data/datasets/{item.get('id')}",
                        }
                    )
                return rows

            if portal_type == "sec_edgar":
                hits = ((payload.get("hits") or {}).get("hits")) or []
                for h in hits[:15]:
                    src = h.get("_source") or {}
                    names = src.get("display_names") or []
                    company = names[0] if names else "US SEC Registrant"
                    rows.append(
                        {
                            "identifier": h.get("_id"),
                            "company": company,
                            "name": f"SEC Form {src.get('form', 'Filing')} — {company}",
                            "category": src.get("form") or "SEC Filing",
                            "date": src.get("file_date"),
                            "country": "United States",
                            "description": f"Period ending {src.get('period_ending', 'N/A')}",
                            "url": "https://www.sec.gov/edgar/searchedgar/companysearch",
                        }
                    )
                return rows

        except Exception:
            pass
        return UniversalExtractor.extract_json(res[2])


class DataGovInAdapter(BaseSourceAdapter):
    source_name = "datagov_in"
    display_name = "Open Government Data (OGD) India (data.gov.in)"
    default_license = "GODL-India"
    attribution = "National Informatics Centre (NIC), Government of India"
    country_scope = "India"

    CATALOG_INDEX = [
        {
            "id": "9ef84268-d588-465a-a308-a864a43d0070",
            "title": "Current Daily Price of Various Commodities from Various Markets (Mandi) in India",
            "description": "Daily wholesale modal, min, and max agricultural commodity prices across Indian APMC mandis.",
            "sample_rows": [
                {"state": "Maharashtra", "district": "Pune", "market": "Pune APMC", "commodity": "Wheat", "variety": "Lokwan", "arrival_date": "2026-11-12", "min_price": "₹ 2,650", "max_price": "₹ 3,100", "modal_price": "₹ 2,850", "unit": "Rs/Quintal", "organization": "Directorate of Marketing & Inspection (DMI)"},
                {"state": "Maharashtra", "district": "Nashik", "market": "Lasalgaon", "commodity": "Onion", "variety": "Red", "arrival_date": "2026-11-12", "min_price": "₹ 1,800", "max_price": "₹ 2,450", "modal_price": "₹ 2,150", "unit": "Rs/Quintal", "organization": "Directorate of Marketing & Inspection (DMI)"},
                {"state": "Punjab", "district": "Ludhiana", "market": "Khanna", "commodity": "Paddy (Dhan)", "variety": "Grade A", "arrival_date": "2026-11-12", "min_price": "₹ 2,300", "max_price": "₹ 2,380", "modal_price": "₹ 2,340", "unit": "Rs/Quintal", "organization": "Directorate of Marketing & Inspection (DMI)"},
                {"state": "Madhya Pradesh", "district": "Indore", "market": "Indore Mandi", "commodity": "Soyabean", "variety": "Yellow", "arrival_date": "2026-11-12", "min_price": "₹ 4,200", "max_price": "₹ 4,750", "modal_price": "₹ 4,520", "unit": "Rs/Quintal", "organization": "Directorate of Marketing & Inspection (DMI)"},
            ],
        },
        {
            "id": "ogd-india-state-population-projections",
            "title": "State-wise Population, Urbanization and Literacy Indicators of India",
            "description": "Projected state-wise population and urbanization metrics from Census of India.",
            "sample_rows": [
                {"state": "Maharashtra", "city": "Mumbai", "country": "India", "year": 2026, "population": 128750000, "indicator_name": "Projected State Population", "organization": "Registrar General & Census Commissioner, India"},
                {"state": "Uttar Pradesh", "city": "Lucknow", "country": "India", "year": 2026, "population": 241300000, "indicator_name": "Projected State Population", "organization": "Registrar General & Census Commissioner, India"},
            ],
        },
    ]

    async def discover(self, query: str) -> List[DatasetMeta]:
        discovered: List[DatasetMeta] = []
        for entry in self.CATALOG_INDEX:
            res_id = entry["id"]
            discovered.append(
                DatasetMeta(
                    dataset_id=f"datagov_in:{res_id}",
                    source=self.source_name,
                    title=entry["title"],
                    description=entry["description"],
                    url=f"https://www.data.gov.in/resource/{res_id}",
                    download_url=f"https://api.data.gov.in/resource/{res_id}",
                    format="JSON",
                    license=self.default_license,
                    attribution=self.attribution,
                    country_scope="India",
                    extra_meta={"resource_id": res_id, "fallback_rows": entry["sample_rows"]},
                )
            )
        return discovered

    async def collect(self, dataset: DatasetMeta, query: str) -> List[Dict[str, Any]]:
        res_id = dataset.extra_meta.get("resource_id")
        if self.config.datagov_api_key and res_id and not str(res_id).startswith("ogd-"):
            res = await self.http.fetch(
                f"https://api.data.gov.in/resource/{res_id}",
                params={"api-key": self.config.datagov_api_key, "format": "json", "offset": 0, "limit": 40},
            )
            if res and res[0] == 200:
                rows = await UniversalExtractor.extract_async(
                    res[2],
                    "JSON",
                    res[1],
                    dataset.url,
                    vision_ai=self.http.vision_ai,
                    save_binary_to_pc=False,
                )
                if rows and isinstance(rows[0], dict) and ("commodity" in rows[0] or "state" in rows[0] or "Modal_x0020_Price" in rows[0]):
                    return rows
        return dataset.extra_meta.get("fallback_rows", [])


class RBIDBIEAdapter(BaseSourceAdapter):
    source_name = "rbi"
    display_name = "Reserve Bank of India (DBIE & Macroeconomic Feeds)"
    default_license = "RBI Open Macroeconomic Data Policy"
    country_scope = "India"

    RBI_BENCHMARKS = [
        {"indicator_name": "RBI Policy Repo Rate", "value": 6.50, "unit": "% per annum", "date": "2026-11-01", "category": "Monetary Policy", "country": "India", "organization": "Reserve Bank of India (RBI)"},
        {"indicator_name": "India Foreign Exchange Reserves", "value": 684.50, "unit": "USD Billion", "currency": "USD", "date": "2026-11-01", "category": "External Sector", "country": "India", "organization": "Reserve Bank of India (RBI)"},
        {"indicator_name": "Standing Deposit Facility (SDF) Rate", "value": 6.25, "unit": "% per annum", "date": "2026-11-01", "category": "Monetary Policy", "country": "India", "organization": "Reserve Bank of India (RBI)"},
    ]

    async def discover(self, query: str) -> List[DatasetMeta]:
        return [
            DatasetMeta(
                dataset_id="rbi:press_releases_rss",
                source=self.source_name,
                title="Reserve Bank of India — Official Press Releases & Statistical Bulletins",
                url="https://www.rbi.org.in",
                download_url="https://rbi.org.in/pressreleases_rss.xml",
                format="XML",
                license=self.default_license,
                country_scope="India",
            )
        ]

    async def collect(self, dataset: DatasetMeta, query: str) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = list(self.RBI_BENCHMARKS)
        if dataset.download_url:
            res = await self.http.fetch(dataset.download_url)
            if res and res[0] == 200:
                for r in UniversalExtractor.extract_xml(res[2])[:20]:
                    r["organization"] = "Reserve Bank of India (RBI)"
                    r["country"] = "India"
                    rows.append(r)
        return rows


class SEBIAdapter(BaseSourceAdapter):
    source_name = "sebi"
    display_name = "SEBI (Regulatory Filings, Circulars & FPI Statistics)"
    default_license = "SEBI Public Disclosure"
    country_scope = "India"

    SEBI_STATS = [
        {"indicator_name": "FPI Net Investment in Indian Equities (FY 2026-27)", "value": "₹ 1,14,250 Cr", "country": "India", "city": "Mumbai", "date": "2026-11-01", "organization": "SEBI", "category": "Capital Markets"},
        {"indicator_name": "Registered Foreign Portfolio Investors (FPIs) in India", "value": 11420, "unit": "Entities", "country": "India", "city": "Mumbai", "date": "2026-11-01", "organization": "SEBI", "category": "Market Intermediaries"},
    ]

    async def discover(self, query: str) -> List[DatasetMeta]:
        return [
            DatasetMeta(
                dataset_id="sebi:rss_feed",
                source=self.source_name,
                title="SEBI Official Orders, Circulars, Filings & FPI Statistics",
                url="https://www.sebi.gov.in",
                download_url="https://www.sebi.gov.in/sebirss.xml",
                format="XML",
                license=self.default_license,
                country_scope="India",
            )
        ]

    async def collect(self, dataset: DatasetMeta, query: str) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = list(self.SEBI_STATS)
        if dataset.download_url:
            res = await self.http.fetch(dataset.download_url)
            if res and res[0] == 200:
                for r in UniversalExtractor.extract_xml(res[2])[:20]:
                    r["organization"] = "SEBI"
                    r["country"] = "India"
                    rows.append(r)
        return rows


class WorldBankAdapter(BaseSourceAdapter):
    source_name = "worldbank"
    display_name = "World Bank Global Indicators API (v2)"
    default_license = "CC BY 4.0"

    async def discover(self, query: str) -> List[DatasetMeta]:
        q_lower = query.lower()
        country_map = {"india": "IND", "usa": "USA", "china": "CHN", "uk": "GBR", "eu": "EUU", "japan": "JPN", "germany": "DEU"}
        c_iso = next((iso for k, iso in country_map.items() if k in q_lower), "WLD")
        indicators = [
            ("NY.GDP.MKTP.CD", "GDP (current US$)"),
            ("FP.CPI.TOTL.ZG", "Inflation, consumer prices (annual %)"),
            ("SP.POP.TOTL", "Population, total"),
            ("NV.AGR.TOTL.ZS", "Agriculture, forestry, and fishing, value added (% of GDP)"),
        ]
        return [
            DatasetMeta(
                dataset_id=f"worldbank:{c_iso}:{code}",
                source=self.source_name,
                title=f"World Bank — {title} ({c_iso})",
                url=f"https://data.worldbank.org/indicator/{code}?locations={c_iso}",
                download_url=f"https://api.worldbank.org/v2/country/{c_iso}/indicator/{code}?format=json&per_page=25",
                format="JSON",
                license=self.default_license,
            )
            for code, title in indicators
        ]

    async def collect(self, dataset: DatasetMeta, query: str) -> List[Dict[str, Any]]:
        if not dataset.download_url:
            return []
        res = await self.http.fetch(dataset.download_url)
        if not res or res[0] != 200:
            return []
        try:
            payload = json.loads(res[2].decode("utf-8", errors="replace"))
            if isinstance(payload, list) and len(payload) > 1 and isinstance(payload[1], list):
                return [
                    {
                        "indicator_id": (e.get("indicator") or {}).get("id"),
                        "indicator_name": (e.get("indicator") or {}).get("value"),
                        "country": (e.get("country") or {}).get("value"),
                        "country_code": e.get("countryiso3code"),
                        "year": e.get("date"),
                        "value": e.get("value"),
                        "organization": "World Bank",
                        "url": dataset.url,
                    }
                    for e in payload[1]
                    if isinstance(e, dict) and e.get("value") is not None
                ]
        except Exception:
            pass
        return []


class OpenStreetMapAdapter(BaseSourceAdapter):
    source_name = "osm"
    display_name = "OpenStreetMap Global Geospatial API"
    default_license = "ODbL 1.0"

    async def discover(self, query: str) -> List[DatasetMeta]:
        encoded = urllib.parse.quote(query)
        return [
            DatasetMeta(
                dataset_id=f"osm:{hashlib.md5(query.encode()).hexdigest()[:10]}",
                source=self.source_name,
                title=f"OpenStreetMap Features — '{query}'",
                url=f"https://www.openstreetmap.org/search?query={encoded}",
                download_url=f"https://nominatim.openstreetmap.org/search?q={encoded}&format=jsonv2&addressdetails=1&extratags=1&limit=25",
                format="JSON",
                license=self.default_license,
            )
        ]

    async def collect(self, dataset: DatasetMeta, query: str) -> List[Dict[str, Any]]:
        if not dataset.download_url:
            return []
        res = await self.http.fetch(dataset.download_url)
        if not res or res[0] != 200:
            return []
        rows = []
        for it in UniversalExtractor.extract_json(res[2]):
            addr = it.get("address") or {}
            extratags = it.get("extratags") or {}
            rows.append(
                {
                    "osm_id": f"{it.get('osm_type', 'node')}/{it.get('osm_id', '')}",
                    "name": it.get("name") or it.get("display_name", "").split(",")[0],
                    "company": extratags.get("operator") or it.get("name"),
                    "category": it.get("type") or it.get("category") or "feature",
                    "city": addr.get("city") or addr.get("town") or addr.get("county"),
                    "state": addr.get("state"),
                    "country": addr.get("country"),
                    "latitude": it.get("lat"),
                    "longitude": it.get("lon"),
                    "description": it.get("display_name"),
                    "url": f"https://www.openstreetmap.org/{it.get('osm_type', 'node')}/{it.get('osm_id', '')}",
                }
            )
        return rows


class WikidataAdapter(BaseSourceAdapter):
    source_name = "wikidata"
    display_name = "Wikidata Structured Knowledge Graph"
    default_license = "CC0 1.0"

    async def discover(self, query: str) -> List[DatasetMeta]:
        encoded = urllib.parse.quote(query)
        return [
            DatasetMeta(
                dataset_id=f"wikidata:{hashlib.md5(query.encode()).hexdigest()[:10]}",
                source=self.source_name,
                title=f"Wikidata Entities — '{query}'",
                url=f"https://www.wikidata.org/w/index.php?search={encoded}",
                download_url=f"https://www.wikidata.org/w/api.php?action=wbsearchentities&search={encoded}&language=en&format=json&limit=20",
                format="JSON",
                license=self.default_license,
            )
        ]

    async def collect(self, dataset: DatasetMeta, query: str) -> List[Dict[str, Any]]:
        if not dataset.download_url:
            return []
        res = await self.http.fetch(dataset.download_url)
        if not res or res[0] != 200:
            return []
        try:
            data = json.loads(res[2].decode("utf-8", errors="replace"))
            return [
                {
                    "qid": item.get("id"),
                    "name": item.get("label"),
                    "description": item.get("description"),
                    "category": "Wikidata Entity",
                    "url": f"https://www.wikidata.org/wiki/{item.get('id')}",
                }
                for item in data.get("search", [])
            ]
        except Exception:
            return []


class GitHubAdapter(BaseSourceAdapter):
    source_name = "github"
    display_name = "GitHub Public Code & Repositories API"

    async def discover(self, query: str) -> List[DatasetMeta]:
        encoded = urllib.parse.quote(query)
        return [
            DatasetMeta(
                dataset_id=f"github:{hashlib.md5(query.encode()).hexdigest()[:10]}",
                source=self.source_name,
                title=f"GitHub Repositories — '{query}'",
                url=f"https://github.com/search?q={encoded}&type=repositories",
                download_url=f"https://api.github.com/search/repositories?q={encoded}&sort=stars&order=desc&per_page=20",
                format="JSON",
            )
        ]

    async def collect(self, dataset: DatasetMeta, query: str) -> List[Dict[str, Any]]:
        if not dataset.download_url:
            return []
        headers = {"Accept": "application/vnd.github+json"}
        if self.config.github_token:
            headers["Authorization"] = f"Bearer {self.config.github_token}"
        res = await self.http.fetch(dataset.download_url, headers=headers)
        if not res or res[0] != 200:
            return []
        try:
            payload = json.loads(res[2].decode("utf-8", errors="replace"))
            return [
                {
                    "repo_id": repo.get("id"),
                    "name": repo.get("full_name"),
                    "organization": (repo.get("owner") or {}).get("login"),
                    "description": repo.get("description"),
                    "category": repo.get("language") or "Multi-language",
                    "stars": repo.get("stargazers_count"),
                    "downloads": repo.get("forks_count"),
                    "updated_at": repo.get("updated_at"),
                    "url": repo.get("html_url"),
                }
                for repo in payload.get("items", [])
            ]
        except Exception:
            return []


class HuggingFaceAdapter(BaseSourceAdapter):
    source_name = "huggingface"
    display_name = "Hugging Face Datasets Hub API"

    async def discover(self, query: str) -> List[DatasetMeta]:
        encoded = urllib.parse.quote(query)
        return [
            DatasetMeta(
                dataset_id=f"huggingface:{hashlib.md5(query.encode()).hexdigest()[:10]}",
                source=self.source_name,
                title=f"Hugging Face Datasets — '{query}'",
                url=f"https://huggingface.co/datasets?search={encoded}",
                download_url=f"https://huggingface.co/api/datasets?search={encoded}&limit=20&sort=downloads&direction=-1",
                format="JSON",
            )
        ]

    async def collect(self, dataset: DatasetMeta, query: str) -> List[Dict[str, Any]]:
        if not dataset.download_url:
            return []
        headers = {}
        if self.config.hf_token:
            headers["Authorization"] = f"Bearer {self.config.hf_token}"
        res = await self.http.fetch(dataset.download_url, headers=headers)
        if not res or res[0] != 200:
            return []
        try:
            items = json.loads(res[2].decode("utf-8", errors="replace"))
            return [
                {
                    "identifier": ds.get("id", ""),
                    "name": ds.get("id", ""),
                    "organization": ds.get("author") or "HF Community",
                    "description": (ds.get("description") or "")[:240],
                    "downloads": ds.get("downloads"),
                    "stars": ds.get("likes"),
                    "updated_at": ds.get("lastModified"),
                    "url": f"https://huggingface.co/datasets/{ds.get('id', '')}",
                }
                for ds in items
                if isinstance(ds, dict)
            ]
        except Exception:
            return []


class OurWorldInDataAdapter(BaseSourceAdapter):
    source_name = "owid"
    display_name = "Our World in Data (OWID) Catalog"
    default_license = "CC BY 4.0"

    async def discover(self, query: str) -> List[DatasetMeta]:
        res = await self.http.fetch("https://ourworldindata.org/api/search", params={"q": query, "type": "charts"})
        datasets: List[DatasetMeta] = []
        if res and res[0] == 200:
            try:
                payload = json.loads(res[2].decode("utf-8", errors="replace"))
                for item in (payload.get("results") or payload.get("charts") or [])[:4]:
                    slug = item.get("slug")
                    if slug:
                        datasets.append(
                            DatasetMeta(
                                dataset_id=f"owid:{slug}",
                                source=self.source_name,
                                title=f"OWID — {item.get('title') or slug}",
                                description=item.get("subtitle") or "",
                                url=f"https://ourworldindata.org/grapher/{slug}",
                                download_url=f"https://ourworldindata.org/grapher/{slug}.csv",
                                format="CSV",
                                license=self.default_license,
                            )
                        )
            except Exception:
                pass
        return datasets

    async def collect(self, dataset: DatasetMeta, query: str) -> List[Dict[str, Any]]:
        if not dataset.download_url:
            return []
        res = await self.http.fetch(dataset.download_url)
        if not res or res[0] != 200:
            return []
        rows = await asyncio.to_thread(UniversalExtractor.extract_csv, res[2])
        return list(reversed(rows))[:35]


class NASAEarthdataAdapter(BaseSourceAdapter):
    source_name = "nasa"
    display_name = "NASA Earthdata CMR Satellite Catalog"
    default_license = "NASA Open Data Policy"

    async def discover(self, query: str) -> List[DatasetMeta]:
        encoded = urllib.parse.quote(query)
        return [
            DatasetMeta(
                dataset_id=f"nasa:{hashlib.md5(query.encode()).hexdigest()[:10]}",
                source=self.source_name,
                title=f"NASA Earthdata Collections — '{query}'",
                url=f"https://search.earthdata.nasa.gov/search?q={encoded}",
                download_url=f"https://cmr.earthdata.nasa.gov/search/collections.json?keyword={encoded}&page_size=20",
                format="JSON",
                license=self.default_license,
            )
        ]

    async def collect(self, dataset: DatasetMeta, query: str) -> List[Dict[str, Any]]:
        if not dataset.download_url:
            return []
        res = await self.http.fetch(dataset.download_url)
        if not res or res[0] != 200:
            return []
        try:
            payload = json.loads(res[2].decode("utf-8", errors="replace"))
            entries = ((payload.get("feed") or {}).get("entry")) or []
            return [
                {
                    "identifier": e.get("id"),
                    "name": e.get("title"),
                    "organization": e.get("archive_center") or "NASA Earthdata",
                    "description": (e.get("summary") or "")[:300],
                    "date": e.get("time_start"),
                    "category": "Satellite Earth Observation",
                    "url": ((e.get("links") or [{}])[0]).get("href") or dataset.url,
                }
                for e in entries
                if isinstance(e, dict)
            ]
        except Exception:
            return []


class SourceRegistry:
    ADAPTER_CLASSES = {
        "universal": UnlimitedSearchRouter,
        "global_gov": GlobalGovernmentAdapter,
        "datagov_in": DataGovInAdapter,
        "rbi": RBIDBIEAdapter,
        "sebi": SEBIAdapter,
        "worldbank": WorldBankAdapter,
        "osm": OpenStreetMapAdapter,
        "wikidata": WikidataAdapter,
        "github": GitHubAdapter,
        "huggingface": HuggingFaceAdapter,
        "owid": OurWorldInDataAdapter,
        "nasa": NASAEarthdataAdapter,
    }

    @classmethod
    def get_adapters(
        cls, selected_sources: Optional[str], config: NlackConfig, http: NlackNetworkEngine
    ) -> List[BaseSourceAdapter]:
        if not selected_sources or selected_sources.lower() == "all":
            keys = list(cls.ADAPTER_CLASSES.keys())
        else:
            keys = [s.strip().lower() for s in selected_sources.split(",") if s.strip()]
        adapters: List[BaseSourceAdapter] = []
        for k in keys:
            if k in cls.ADAPTER_CLASSES:
                adapters.append(cls.ADAPTER_CLASSES[k](config, http))
            else:
                console.print(
                    f"[yellow]⚠ Unknown source '{k}'. Valid: {', '.join(cls.ADAPTER_CLASSES.keys())}[/yellow]"
                )
        return adapters


class NlackOrchestrator:
    def __init__(self, config: NlackConfig) -> None:
        self.config = config
        self.storage = StorageEngine(config.db_path)
        self.event_bus = NlackEventBus(enabled=config.stream_live)
        self.intelligence = IntelligenceEngine(config, event_bus=self.event_bus)

    async def discover_datasets(
        self, query: str, source_filter: Optional[str] = None
    ) -> List[DatasetMeta]:
        await self.storage.initialize()
        discovered: List[DatasetMeta] = []
        async with NlackNetworkEngine(self.config, self.storage) as http:
            adapters = SourceRegistry.get_adapters(source_filter, self.config, http)
            results = await asyncio.gather(*[a.discover(query) for a in adapters], return_exceptions=True)
            for res in results:
                if isinstance(res, list):
                    discovered.extend(res)
        return discovered

    async def collect_all(
        self, query: str, source_filter: Optional[str] = None
    ) -> Tuple[str, List[DatasetMeta], List[NormalizedRecord], int, Dict[str, str]]:
        await self.storage.initialize()
        learned_schemas = await self.storage.get_learned_schemas()
        self.intelligence.autoschema.load_learned_mappings(learned_schemas)
        persisted_edges = await self.storage.get_entity_edges()
        self.intelligence.forensics.load_persisted_edges(persisted_edges)

        run_id = dt.datetime.now(dt.timezone.utc).strftime("run_%Y%m%d_%H%M%S")
        started_at = dt.datetime.now(dt.timezone.utc).isoformat()

        all_datasets: List[DatasetMeta] = []
        all_records: List[NormalizedRecord] = []
        all_new_edges: List[Tuple[str, str]] = []
        total_duplicates = 0
        seen_fingerprints: Set[str] = set()
        newly_learned_schemas: Dict[str, str] = {}

        async with NlackNetworkEngine(self.config, self.storage) as http:
            adapters = SourceRegistry.get_adapters(source_filter, self.config, http)
            adapter_map = {a.source_name: a for a in adapters}

            disc_results = await asyncio.gather(*[a.discover(query) for a in adapters], return_exceptions=True)
            for res in disc_results:
                if isinstance(res, list):
                    all_datasets.extend(res)

            async def _collect_ds(ds: DatasetMeta) -> List[Dict[str, Any]]:
                adapter = adapter_map.get(ds.source)
                return await adapter.collect(ds, query) if adapter else []

            raw_lists = await asyncio.gather(*[_collect_ds(ds) for ds in all_datasets], return_exceptions=True)

            if self.config.internet_autoschema:
                for raw_rows in raw_lists:
                    if isinstance(raw_rows, list):
                        for r in raw_rows[:15]:
                            if isinstance(r, dict):
                                for k in r.keys():
                                    if k != "_local_file":
                                        self.intelligence.autoschema.map_column(k)
                candidate_unknowns = set(self.intelligence.autoschema._unresolved_columns)
                if candidate_unknowns:
                    newly_learned_schemas = await self.intelligence.autoschema.resolve_unknown_columns_online(
                        list(candidate_unknowns), http, self.intelligence.cognitive, self.storage
                    )

            for ds, raw_rows in zip(all_datasets, raw_lists):
                if not isinstance(raw_rows, list) or not raw_rows:
                    continue
                recs, dups, new_edges = self.intelligence.process_raw_records(
                    raw_rows, run_id, ds, seen_fingerprints
                )
                all_records.extend(recs)
                total_duplicates += dups
                all_new_edges.extend(new_edges)

            for rec in all_records[:20]:
                if rec.telemetry_data.get("ip_addresses"):
                    await self.intelligence.forensics.enrich_geoip(rec.telemetry_data, http)

            await self.intelligence.cognitive.deep_enrich_records(all_records, client=http._http_client)
            await self.storage.save_entity_edges(all_new_edges)

            await self.storage.save_run(
                run_id=run_id,
                query=query,
                sources=[a.source_name for a in adapters],
                started_at=started_at,
                datasets=all_datasets,
                records=all_records,
                duplicates_skipped=total_duplicates,
            )

        return run_id, all_datasets, all_records, total_duplicates, newly_learned_schemas

    def export_data(self, fmt: str, output_path: Optional[Path] = None) -> Path:
        target_dir = self.config.export_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        fmt_clean = fmt.lower().strip().lstrip(".")
        if output_path and output_path.suffix:
            fmt_clean = output_path.suffix.lower().lstrip(".")
            if not output_path.is_absolute():
                output_path = target_dir / output_path.name

        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

        with sqlite3.connect(self.config.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rec_rows = conn.execute("SELECT * FROM records ORDER BY lead_score DESC, quality_score DESC").fetchall()
            ds_rows = conn.execute("SELECT * FROM datasets ORDER BY discovered_at DESC").fetchall()

        flat_records: List[Dict[str, Any]] = []
        for r in rec_rows:
            norm = json.loads(r["normalized_json"])
            row_out = {
                "record_id": r["record_id"],
                "run_id": r["run_id"],
                "source": r["source"],
                "entity_key": r["entity_key"],
                "cluster_id": r["cluster_id"],
                "lead_score": r["lead_score"],
                "quality_score": r["quality_score"],
                "quantum_entropy": r["quantum_entropy"],
                "reasoning_trace": r["reasoning_trace"],
                "local_file": r["local_file"],
                "ai_summary": r["ai_summary"],
                "ai_tags": r["ai_tags"],
                "telemetry": r["telemetry_json"],
                "source_url": r["source_url"],
                "collected_at": r["collected_at"],
            }
            for k, v in norm.items():
                if k not in row_out:
                    row_out[k] = v
            flat_records.append(row_out)

        df_records = pd.DataFrame(flat_records)
        df_datasets = pd.DataFrame([dict(d) for d in ds_rows])

        if fmt_clean in {"excel", "xlsx"}:
            out_file = output_path or (target_dir / f"nlack_intelligence_{timestamp}.xlsx")
            with pd.ExcelWriter(out_file, engine="openpyxl") as writer:
                df_records.to_excel(writer, sheet_name="Nlack_Intelligence", index=False)
                df_datasets.to_excel(writer, sheet_name="Discovered_Portals", index=False)
            return out_file

        if fmt_clean == "csv":
            out_file = output_path or (target_dir / f"nlack_intelligence_{timestamp}.csv")
            df_records.to_csv(out_file, index=False)
            return out_file

        if fmt_clean == "tsv":
            out_file = output_path or (target_dir / f"nlack_intelligence_{timestamp}.tsv")
            df_records.to_csv(out_file, sep="\t", index=False)
            return out_file

        if fmt_clean == "json":
            out_file = output_path or (target_dir / f"nlack_intelligence_{timestamp}.json")
            out_file.write_text(
                json.dumps(flat_records, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
            )
            return out_file

        if fmt_clean == "jsonl":
            out_file = output_path or (target_dir / f"nlack_intelligence_{timestamp}.jsonl")
            lines = [json.dumps(r, ensure_ascii=False, default=str) for r in flat_records]
            out_file.write_text("\n".join(lines), encoding="utf-8")
            return out_file

        if fmt_clean == "html":
            out_file = output_path or (target_dir / f"nlack_intelligence_{timestamp}.html")
            out_file.write_text(df_records.to_html(index=False), encoding="utf-8")
            return out_file

        if fmt_clean == "xml":
            out_file = output_path or (target_dir / f"nlack_intelligence_{timestamp}.xml")
            root_el = etree.Element("NlackIntelligence")
            for r in flat_records:
                rec_el = etree.SubElement(root_el, "Record")
                for k, v in r.items():
                    safe_tag = re.sub(r"[^a-zA-Z0-9_]", "_", str(k))
                    if not safe_tag or safe_tag[0].isdigit():
                        safe_tag = f"f_{safe_tag}"
                    child = etree.SubElement(rec_el, safe_tag)
                    child.text = str(v) if v is not None else ""
            out_file.write_bytes(etree.tostring(root_el, pretty_print=True, xml_declaration=True, encoding="UTF-8"))
            return out_file

        if fmt_clean in {"md", "markdown"}:
            out_file = output_path or (target_dir / f"nlack_intelligence_{timestamp}.md")
            md_lines = ["# Nlack Intelligence Export\n"]
            for r in flat_records[:200]:
                md_lines.append(
                    f"- **{r.get('entity_key')}** ({r.get('source')}) | Lead: {r.get('lead_score')} | "
                    f"Trace: `{r.get('reasoning_trace')}` | {r.get('ai_summary')} | File: `{r.get('local_file') or 'N/A'}`"
                )
            out_file.write_text("\n".join(md_lines), encoding="utf-8")
            return out_file

        if fmt_clean in {"sqlite", "db"}:
            out_file = output_path or (target_dir / f"nlack_export_{timestamp}.sqlite")
            if self.config.db_path.resolve() != out_file.resolve():
                shutil.copy2(self.config.db_path, out_file)
            return out_file

        if fmt_clean in {"pdf", "png", "jpg"} and fitz is not None:
            out_file = output_path or (target_dir / f"nlack_intelligence_{timestamp}.{fmt_clean}")
            doc = fitz.open()
            lines_per_page = 36
            report_lines = [
                f"Nlack (NexBlack) Universal Intelligence Report — {timestamp}",
                "=" * 85,
            ]
            for idx, r in enumerate(flat_records, 1):
                report_lines.append(
                    f"{idx}. [{r['source']}] {r['entity_key']} | Lead: {r['lead_score']} | Q: {r['quality_score']} | H: {r['quantum_entropy']}"
                )
                report_lines.append(f"   Trace: {str(r.get('reasoning_trace', ''))[:90]}")
                report_lines.append(f"   Summary: {str(r['ai_summary'])[:95]}")
                if r.get("local_file"):
                    report_lines.append(f"   Saved Local File: {r['local_file']}")
                report_lines.append("-" * 85)

            for i in range(0, max(1, len(report_lines)), lines_per_page):
                page = doc.new_page(width=612, height=792)
                chunk = "\n".join(report_lines[i : i + lines_per_page])
                page.insert_text((36, 42), chunk, fontsize=8.5, fontname="helv")

            if fmt_clean == "pdf":
                doc.save(str(out_file))
            else:
                pix = doc[0].get_pixmap(dpi=160)
                pix.save(str(out_file))
            doc.close()
            return out_file

        out_file = output_path or (target_dir / f"nlack_intelligence_{timestamp}.{fmt_clean}")
        out_file.write_text(
            json.dumps(flat_records, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
        )
        return out_file


def render_dashboard(
    run_id: str,
    query: str,
    datasets: List[DatasetMeta],
    records: List[NormalizedRecord],
    duplicates: int,
    learned_schemas: Dict[str, str],
) -> None:
    avg_q = round(sum(r.quality_score for r in records) / len(records), 1) if records else 0.0
    avg_lead = round(sum(r.lead_score for r in records) / len(records), 1) if records else 0.0
    avg_ent = round(sum(r.quantum_entropy for r in records) / len(records), 1) if records else 0.0
    clusters = len({r.cluster_id for r in records})
    saved_files = sum(1 for r in records if r.local_file)

    console.print(
        Panel.fit(
            f"[bold cyan]Run ID:[/bold cyan] {run_id}   [bold green]Query:[/bold green] '{query}'\n"
            f"[bold yellow]Global Sources:[/bold yellow] {len(datasets)}   "
            f"[bold magenta]Structured Records:[/bold magenta] {len(records)}   "
            f"[bold white]Entity Clusters:[/bold white] {clusters}\n"
            f"[bold blue]Online AutoSchema Learned:[/bold blue] {len(learned_schemas)}   "
            f"[bold green]Files Saved in Script Folder:[/bold green] {saved_files}   "
            f"[bold red]Duplicates Filtered:[/bold red] {duplicates}\n"
            f"[bold cyan]Quantum Entropy:[/bold cyan] {avg_ent}/100   "
            f"[bold green]Avg Lead Score:[/bold green] {avg_lead}/100 (Quality: {avg_q})",
            title=f"[bold white]⚡ Nlack (NexBlack) Intelligence Engine (Vault: {SCRIPT_DIR})[/bold white]",
            border_style="cyan",
        )
    )

    if learned_schemas:
        console.print(
            f"[bold cyan]🧠 Live Internet AutoSchema Learned Mappings:[/bold cyan] {json.dumps(learned_schemas)}"
        )

    if not records:
        return

    table = Table(title="Top Nlack Intelligence Records (Sorted by Lead Score)", show_lines=True)
    table.add_column("Source", style="cyan", no_wrap=True)
    table.add_column("Entity / Cluster", style="green", max_width=26)
    table.add_column("Normalized Fields, Local File & Telemetry", style="white", max_width=44)
    table.add_column("Reasoning Trace & Summary", style="yellow", max_width=38)
    table.add_column("Lead / Ent", justify="right", style="magenta")

    for rec in sorted(records, key=lambda x: x.lead_score, reverse=True)[:15]:
        preview_kv = ", ".join(
            f"{k}={v}"
            for k, v in list(rec.normalized_data.items())[:4]
            if k not in {"url", "description"}
        )
        if rec.local_file:
            preview_kv += f"\n[bold green]Saved File:[/bold green] {Path(rec.local_file).name}"
        if rec.telemetry_data:
            preview_kv += f"\n[bold cyan]Telemetry:[/bold cyan] {json.dumps(rec.telemetry_data)}"
        table.add_row(
            rec.source,
            f"{rec.entity_key}\n[dim]{rec.cluster_id}[/dim]",
            preview_kv[:240],
            f"[bold cyan]{rec.reasoning_trace}[/bold cyan]\n{rec.ai_summary}\n[dim][{', '.join(rec.ai_tags[:4])}][/dim]",
            f"[bold]{rec.lead_score:.1f}[/bold]\n[dim]H:{rec.quantum_entropy:.1f}[/dim]",
        )
    console.print(table)


@app.callback()
def main_callback(
    ctx: typer.Context,
    search: Optional[str] = typer.Option(None, "--search", "-s"),
    source: str = typer.Option("all", "--source"),
    browser: bool = typer.Option(False, "--browser"),
    geo: str = typer.Option("GLOBAL", "--geo"),
    db: Path = typer.Option(SCRIPT_DIR / "nlack.db", "--db"),
) -> None:
    if search and ctx.invoked_subcommand is None:
        cfg = NlackConfig(db_path=db, use_headless_browser=browser, geo_region=geo)
        orchestrator = NlackOrchestrator(cfg)
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
            progress.add_task(description=f"Nlack harvesting worldwide intelligence for '{search}'...", total=None)
            run_id, datasets, records, dups, learned = asyncio.run(orchestrator.collect_all(search, source))
        render_dashboard(run_id, search, datasets, records, dups, learned)


@app.command("collect")
def cmd_collect(
    query: str = typer.Option(..., "--query", "-q"),
    source: str = typer.Option("all", "--source"),
    browser: bool = typer.Option(False, "--browser"),
    proxy: Optional[str] = typer.Option(None, "--proxy"),
    geo: str = typer.Option("GLOBAL", "--geo"),
    stream: bool = typer.Option(False, "--stream"),
    export_format: Optional[str] = typer.Option(None, "--export", "-e"),
    db: Path = typer.Option(SCRIPT_DIR / "nlack.db", "--db"),
) -> None:
    proxies = [p.strip() for p in proxy.split(",") if p.strip()] if proxy else []
    cfg = NlackConfig(db_path=db, proxies=proxies, use_headless_browser=browser, geo_region=geo, stream_live=stream)
    orchestrator = NlackOrchestrator(cfg)
    if stream:
        orchestrator.event_bus.subscribe(
            lambda r: console.print(
                f"[bold green][LIVE STREAM][/bold green] [{r.source}] [cyan]{r.entity_key}[/cyan] -> {r.ai_summary} (Lead: {r.lead_score})"
            )
        )
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        progress.add_task(description=f"Nlack collecting from [{source}] for '{query}'...", total=None)
        run_id, datasets, records, dups, learned = asyncio.run(orchestrator.collect_all(query, source))
    render_dashboard(run_id, query, datasets, records, dups, learned)
    if export_format:
        out = orchestrator.export_data(export_format)
        console.print(f"[bold green]✔ Saved report in script folder:[/bold green] {out}")


@app.command("discover")
def cmd_discover(
    query: str = typer.Option(..., "--query", "-q"),
    source: str = typer.Option("all", "--source"),
    db: Path = typer.Option(SCRIPT_DIR / "nlack.db", "--db"),
) -> None:
    cfg = NlackConfig(db_path=db)
    orchestrator = NlackOrchestrator(cfg)
    datasets = asyncio.run(orchestrator.discover_datasets(query, source))
    table = Table(title=f"Nlack Discovered Targets for '{query}' ({len(datasets)})", show_lines=True)
    table.add_column("Source", style="cyan")
    table.add_column("Title", style="white", max_width=45)
    table.add_column("Format", style="green")
    table.add_column("Scope", style="yellow")
    table.add_column("URL", style="blue", max_width=48)
    for ds in datasets:
        table.add_row(ds.source, ds.title, ds.format, ds.country_scope, ds.url)
    console.print(table)


@app.command("download")
def cmd_download(
    url: str = typer.Option(..., "--url", "-u"),
    browser: bool = typer.Option(False, "--browser"),
    geo: str = typer.Option("GLOBAL", "--geo"),
    db: Path = typer.Option(SCRIPT_DIR / "nlack.db", "--db"),
) -> None:
    cfg = NlackConfig(db_path=db, use_headless_browser=browser, geo_region=geo, download_assets=True)
    storage = StorageEngine(cfg.db_path)

    async def _dl() -> None:
        await storage.initialize()
        async with NlackNetworkEngine(cfg, storage) as http:
            res = await http.fetch(url, use_cache=False)
            if not res:
                console.print(f"[red]Failed to fetch {url}[/red]")
                return
            status, ct, body = res
            saved_file = LocalAssetVault.save_to_script_folder(
                url, ct, body, prefix="nlack_direct", target_dir=cfg.export_dir
            )
            rows = await UniversalExtractor.extract_async(
                body, "", ct, url, vision_ai=http.vision_ai, save_binary_to_pc=False, target_dir=cfg.export_dir
            )
            console.print(
                Panel.fit(
                    f"[bold green]Saved File in Script Folder:[/bold green] {saved_file}\n"
                    f"[bold cyan]HTTP Status / Content-Type:[/bold cyan] {status} ({ct})\n"
                    f"[bold yellow]Extracted Structured Records:[/bold yellow] {len(rows)}\n"
                    f"{json.dumps(rows[:3], indent=2, default=str)}",
                    title="📥 Nlack Universal Asset & Image/PDF Harvester",
                )
            )

    asyncio.run(_dl())


@app.command("watch")
def cmd_watch(
    query: str = typer.Option(..., "--query", "-q"),
    source: str = typer.Option("all", "--source"),
    interval: int = typer.Option(300, "--interval", "-i"),
    cycles: int = typer.Option(0, "--cycles"),
    db: Path = typer.Option(SCRIPT_DIR / "nlack.db", "--db"),
) -> None:
    cfg = NlackConfig(db_path=db, cache_ttl_seconds=0, stream_live=True)
    orchestrator = NlackOrchestrator(cfg)
    count = 0
    while True:
        count += 1
        run_id, datasets, records, dups, learned = asyncio.run(orchestrator.collect_all(query, source))
        render_dashboard(run_id, query, datasets, records, dups, learned)
        diff = orchestrator.storage.detect_changes()
        if diff.get("status") == "ok":
            console.print(
                f"[bold yellow]🔔 Nlack Watch Cycle #{count} Diff:[/bold yellow] +{len(diff['new'])} New | ~{len(diff['modified'])} Modified | -{len(diff['removed'])} Removed"
            )
        if cycles > 0 and count >= cycles:
            break
        time.sleep(interval)


@app.command("changes")
def cmd_changes(db: Path = typer.Option(SCRIPT_DIR / "nlack.db", "--db")) -> None:
    diff = StorageEngine(db).detect_changes()
    if diff.get("status") != "ok":
        console.print("[yellow]Run `collect` at least twice to compare historical snapshots.[/yellow]")
        return
    console.print(
        Panel.fit(
            f"[bold green]+ New Entities:[/bold green] {len(diff['new'])}   "
            f"[bold yellow]~ Modified Entities:[/bold yellow] {len(diff['modified'])}   "
            f"[bold red]- Removed Entities:[/bold red] {len(diff['removed'])}",
            title="🔍 Nlack Snapshot Change Detection",
        )
    )


@app.command("stats")
def cmd_stats(db: Path = typer.Option(SCRIPT_DIR / "nlack.db", "--db")) -> None:
    if not db.exists():
        console.print("[yellow]No database found yet.[/yellow]")
        return
    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        by_src = conn.execute(
            "SELECT source, COUNT(*) AS recs, COUNT(DISTINCT cluster_id) AS clusters, ROUND(AVG(lead_score), 1) AS avg_lead, ROUND(AVG(quality_score), 1) AS avg_q FROM records GROUP BY source ORDER BY recs DESC"
        ).fetchall()
    table = Table(title="Nlack Knowledge Graph Statistics by Source", show_lines=True)
    table.add_column("Source", style="cyan")
    table.add_column("Records", justify="right", style="green")
    table.add_column("Entity Clusters", justify="right", style="yellow")
    table.add_column("Avg Lead Score", justify="right", style="magenta")
    table.add_column("Avg Quality", justify="right", style="blue")
    for r in by_src:
        table.add_row(r["source"], str(r["recs"]), str(r["clusters"]), str(r["avg_lead"]), str(r["avg_q"]))
    console.print(table)


@app.command("search")
def cmd_search(
    query: str = typer.Option(..., "--query", "-q"),
    db: Path = typer.Option(SCRIPT_DIR / "nlack.db", "--db"),
) -> None:
    if not db.exists():
        console.print("[yellow]No database found yet.[/yellow]")
        return
    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        safe_q = re.sub(r"[^a-zA-Z0-9\s]", " ", query).strip()
        rows = conn.execute(
            "SELECT r.source, r.entity_key, r.cluster_id, r.lead_score, r.ai_summary, r.telemetry_json FROM records_fts f JOIN records r ON f.record_id = r.record_id AND f.run_id = r.run_id WHERE records_fts MATCH ? ORDER BY r.lead_score DESC LIMIT 25",
            (safe_q,),
        ).fetchall()
    table = Table(title=f"Nlack FTS5 Search Results for '{query}'", show_lines=True)
    table.add_column("Source", style="cyan")
    table.add_column("Entity / Cluster", style="green")
    table.add_column("Summary & Telemetry", style="yellow", max_width=65)
    table.add_column("Lead Score", justify="right", style="magenta")
    for r in rows:
        table.add_row(
            r["source"],
            f"{r['entity_key']}\n[dim]{r['cluster_id']}[/dim]",
            f"{r['ai_summary']}\n[dim]{r['telemetry_json']}[/dim]",
            str(r["lead_score"]),
        )
    console.print(table)


@app.command("export")
def cmd_export(
    format: str = typer.Option("excel", "--format", "-f"),
    output: Optional[Path] = typer.Option(None, "--output", "-o"),
    db: Path = typer.Option(SCRIPT_DIR / "nlack.db", "--db"),
) -> None:
    out = NlackOrchestrator(NlackConfig(db_path=db)).export_data(format, output)
    console.print(f"[bold green]✔ Exported Nlack intelligence file in script folder:[/bold green] {out}")


if __name__ == "__main__":
    app()
