# ⚡ Nlack — Universal Data Harvester & Intelligence Engine

**Nlack** is a single-file, self-evolving public, institutional, and government data harvesting and forensic intelligence engine engineered by **Adarsh Satyajit Adhikary**.

Designed as both a standalone CLI tool and an importable Python SDK (`nlack.py`), Nlack replaces fragile website-specific scrapers with an autonomous 8-layer pipeline. It discovers datasets across global search engines and government portals, extracts structured records from any file format (including scanned PDFs and statistical images via Vision OCR), unifies schemas online via **AutoSchema**, verifies forensic and corporate telemetry with mathematical checksums, links entities across runs in a persistent knowledge graph (`nlack.db`), and exports intelligence into any file extension.

---

## 🏗️ System Architecture

Nlack separates its permanent execution engine (`nlack.py`) from its continuously learning local database (`nlack.db`):

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│                        nlack.py (Static Execution Core)                      │
├────────────────────────┬──────────────────────────┬──────────────────────────┤
│ 1. Quantum Search      │ 2. 4-Tier Network Engine │ 3. Multi-Core Extractor  │
│ • Brave / Google RSS   │ • Persistent TCP/TLS Pool│ • CSV / XLSX / JSON / XML│
│ • GDELT 2.0 / OpenAlex │ • Chrome JA3 Impersonate │ • PDF Table & Text Parser│
│ • DuckDuckGo / Wiki    │ • Headless XHR Sniffer   │ • Multimodal Vision OCR  │
│ • 12 Native Adapters   │ • Wayback Mirror Fallback│ • Direct PC Asset Vault  │
├────────────────────────┼──────────────────────────┼──────────────────────────┤
│ 4. Online AutoSchema   │ 5. Forensic Verifier     │ 6. Real-Time Event Bus   │
│ • Hilbert Fidelity     │ • Base58Check (BTC/TRON) │ • Live Terminal Stream   │
│ • Datamuse & Wikidata  │ • Luhn (ISIN) & Mod97 LEI│ • Delta Watch & Diffing  │
│ • LLM Ontology Mapper  │ • GeoIP & ASN Attribution│ • Any-Extension Exporter │
└───────────┬────────────┴─────────────┬────────────┴─────────────┬────────────┘
            │                          │                          │
            ▼                          ▼                          ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                     nlack.db (Self-Evolving SQLite WAL Brain)                │
│ • autoschema_memory (Learned Column Mappings)                                │
│ • entity_edges (Persistent Cross-Run Knowledge Graph)                        │
│ • records & records_fts (Normalized Intelligence + Millisecond FTS5 Search)  │
│ • http_cache (ETag / Last-Modified & Response Cache)                         │
└──────────────────────────────────────────────────────────────────────────────┘
