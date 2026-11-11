# 🤖 CrewAI Document Pipeline

Pipeline **autônomo e resiliente** de leitura de **Cadeias de Custódia (COC)** e formulários tabulares, construído sobre [CrewAI](https://docs.crewai.com/) + [Google Gemini Vision](https://ai.google.dev/) com engenharia de confiabilidade de nível de produção.

O fluxo adquire documentos via Google Drive (ou diretório local `--local-dir`), rasteriza páginas PDF a **300 DPI** em memória sem modificar os originais, corrige a orientação por página via Visão Computacional, extrai dados sob **JSON Schema estrito** com comprovação de cobertura de páginas (`covered_pages` / `source_pages`), persiste estado em **SQLite WAL** recuperável de falhas e exporta o resultado consolidado em CSV atômico.

---

## 🏗️ Arquitetura de Confiabilidade

Este repositório evoluiu de uma orquestração CrewAI sequencial simples para incorporar mecanismos completos de resiliência operacional:

1. **Journal Transacional SQLite WAL (`runtime/checkpoint.py`)**: Registro durável com `PRAGMA synchronous=FULL`, identidade determinística por `(source_id, revision)`, detecção de conflitos e recuperação automática de estados em trânsito (`SENDING -> UNCERTAIN`, `EXTRACTING -> DEFERRED`).
2. **Rasterização 300 DPI + Orientação por Página (`runtime/extraction.py`, `tools/orientation_tool.py`, `tools/pdf_processor.py`)**: Conversão lossless via PyMuPDF em diretório temporário/isolado preservando o PDF original intacto, detecção visual em lote `[0, 90, 180, 270]`, cache SHA-256 com TTL de 47h na Files API (`uploads.json`) e validação obrigatória de `finish_reason == "STOP"`, streaming limitado e cobertura completa `1..N`.
3. **Orçamento Monotônico de Tempo (`runtime/deadline.py`, `runtime/config.py`)**: Propagação de `Deadline` baseada em `time.monotonic()`, clamping explícito de timeouts HTTP em milissegundos (`HttpOptions` com `attempts=1`) e validação estrita de invariantes de configuração.
4. **Contenção de Subprocessos por SO (`runtime/supervisor.py`, `runtime/ocr_worker.py`)**: Execução isolada de cada documento em subprocesso `spawn`, associado a **Windows Job Objects** (`JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`) ou sessões POSIX com guarda contra reuso de PID, além de bounded logging (2 MB) e modo watchdog `--mode supervise`.
5. **I/O Atômico e Bloqueio de Concorrência (`runtime/files.py`)**: Escrita atômica (`mkstemp` + `fsync` + `os.replace`) para JSON/CSV (`output/resultado_crew.csv`) e lock exclusivo `InstanceLock` (`msvcrt`/`fcntl`) para evitar execuções simultâneas.
6. **Camada CrewAI Integrada (`crew.py`, `tools/crew_wrappers.py`, `tools/_crew_compat.py`)**: Decorators `@CrewBase` / `@tool` com adaptadores compatíveis para execução tanto via CrewAI quanto no modo pipeline resiliente e suíte de testes offline.
7. **Testes Determinísticos Offline (`tests/`)**: Suíte automatizada com clientes simulados (`FakeClient`, `FakeDrive`), PDFs sintéticos PyMuPDF e validação em CI (`.github/workflows/ci.yml`).

---

## 📂 Estrutura de Pastas

```text
crewai-document-pipeline/
├── main.py                      # CLI unificada (--mode crew | pipeline | supervise)
├── crew.py                      # Orquestrador CrewAI sequencial (ProtocrewCrew)
├── requirements.txt             # Dependências Python
├── .env.example                 # Modelo completo de variáveis de ambiente
├── credentials.example.json     # Modelo OAuth2 Desktop do Google Drive
├── .github/workflows/ci.yml     # CI automatizado com pytest
│
├── config/
│   ├── agents.yaml              # Definição dos 3 agentes CrewAI
│   ├── tasks.yaml               # Definição das 3 tarefas sequenciais
│   └── template_registry.json   # Registry de layouts (LAB_ALPHA, LAB_BETA, rascunhos)
│
├── runtime/
│   ├── config.py                # RuntimeConfig congelado e validado via ambiente
│   ├── deadline.py              # Deadline monotônico e clamping de timeout HTTP
│   ├── files.py                 # atomic_write_text, atomic_json e InstanceLock
│   ├── checkpoint.py            # Journal SQLite WAL (Checkpoint, PublicationConflict)
│   ├── supervisor.py            # WindowsJob / killpg, LimitedLog, run_worker, watchdog
│   ├── extraction.py            # Rasterização 300 DPI, upload cache, schema e stream guard
│   ├── ocr_worker.py            # Worker isolado com verificação MD5 e progress markers
│   └── pipeline.py              # DocumentPipeline com circuit breaker e exportação CSV atômica
│
├── tools/
│   ├── _crew_compat.py          # Compatibilidade opcional com crewai para modo standalone/CI
│   ├── crew_wrappers.py         # Wrappers @tool (Download, Extração 300 DPI, Validação/Export)
│   ├── drive_tools.py           # GoogleDriveDownloadTool com download .partial + move idempotente
│   ├── pattern_identifier.py    # Match por âncoras via PyMuPDF + fallback Gemini
│   ├── filler_agent.py          # Construção de prompt parametrizado + pipeline 300 DPI
│   ├── orientation_tool.py      # Detecção visual de rotação [0, 90, 180, 270]
│   ├── pdf_processor.py         # Rotação em memória via pypdf
│   ├── reviewer.py              # Validação estrutural automática das amostras
│   ├── feedback_store.py        # Persistência atômica de métricas por template
│   └── utils.py                 # Normalização de código, hora e prefixos configuráveis
│
└── tests/
    ├── test_extraction.py
    ├── test_checkpoint_pipeline.py
    ├── test_supervisor.py
    ├── test_tools_and_utils.py
    └── workers.py
```

---

## ⚙️ Modos de Execução (`main.py`)

```bash
# 1. Modo CrewAI (padrão): executa os 3 agentes sequenciais com ferramentas hardened
python main.py --mode crew

# 2. Modo Pipeline Resiliente: SQLite WAL + Workers isolados + CSV atômico
python main.py --mode pipeline --dry-run

# 3. Processamento local sem Google Drive (pasta com PDFs/imagens)
python main.py --mode pipeline --local-dir ./input_files --dry-run

# 4. Modo Watchdog Supervisor: encapsula o lote inteiro com hard-kill por orçamento
python main.py --mode supervise
```

---

## 🧪 Setup e Testes Offline

```bash
# 1. Criar ambiente virtual e instalar dependências
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac
pip install -r requirements.txt

# 2. Configurar variáveis de ambiente
cp .env.example .env
cp credentials.example.json credentials.json

# 3. Executar suíte de testes unitários (100% offline, sem APIs externas)
python -m pytest -v
```

---

## 🔑 Variáveis de Ambiente Principais

| Variável | Padrão | Descrição |
|---|---|---|
| `GEMINI_API_KEY` | — | Chave da API Google Gemini |
| `DRIVE_FOLDER_ID` | — | ID da pasta de entrada no Google Drive |
| `SAMPLE_CODE_PREFIXES` | `""` | Prefixos opcionais separados por vírgula para normalização |
| `GEMINI_MODEL_PRO` | `gemini-2.5-pro` | Modelo principal de extração |
| `GEMINI_MODEL_FAST` | `gemini-2.5-flash` | Modelo rápido para orientação e inferência |
| `PIPELINE_MODE` | `crew` | Modo padrão (`crew`, `pipeline` ou `supervise`) |
| `PIPELINE_STATE_DIR` | `state` | Diretório do SQLite WAL (`journal.sqlite3`), locks e logs de workers |
| `PIPELINE_OUTPUT_CSV` | `output/resultado_crew.csv` | Caminho de exportação consolidada atômica |
| `BATCH_BUDGET_SECONDS` | `1500` | Orçamento total de tempo do lote |
| `FILE_BUDGET_SECONDS` | `600` | Orçamento máximo por documento |
| `ORIENTATION_ENABLED` | `true` | Ativa detecção automática de páginas rotacionadas |
| `STREAMING_ENABLED` | `true` | Ativa streaming com verificação obrigatória de `STOP` |
| `DRY_RUN` | `false` | Executa análise/CSV sem mover arquivos no Drive |
| `MOVE_DRIVE_FILES` | `true` | Move arquivos no Drive para `processados`/`rejeitados` |

---

## 📊 Saída Consolidada (`output/resultado_crew.csv`)

```csv
Arquivo;Codigo;Identificacao;DataHora;Template;Status;Motivo
doc1.pdf;0001234;2026/001 - Ponto Norte;10/02/2026 09:45;LAB_ALPHA;APROVADO;
doc2.pdf;12345;PONTO-B2;15/03/2026 14:30;LAB_BETA;REJEITADO;Código '12345' não tem 7 dígitos numéricos
```

---

## 📝 Licença

Este projeto é distribuído sob a **MIT License**.
