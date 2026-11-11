# 🤖 CrewAI Document Pipeline (PROTOCREW)

Pipeline **100% autônomo** de leitura de **Cadeias de Custódia (COC)** laboratoriais, construído com [CrewAI](https://docs.crewai.com/) + [Google Gemini](https://ai.google.dev/).

O robô baixa PDFs do Google Drive, identifica o template do laboratório, extrai dados via OCR com IA (código da amostra, identificação, data, hora), valida os resultados e exporta tudo em CSV — sem intervenção humana.

---

## 📂 Estrutura de Pastas

```text
crewai-document-pipeline/
├── main.py                      # Ponto de entrada do programa
├── crew.py                      # Orquestrador CrewAI (agentes + tarefas)
├── requirements.txt             # Dependências Python
├── .env.example                 # Modelo de variáveis de ambiente
├── credentials.example.json     # Modelo de credenciais OAuth2 do Google Drive
│
├── config/                      # Configuração dos agentes, tarefas e templates
│   ├── agents.yaml              # Definição dos agentes CrewAI
│   ├── tasks.yaml               # Definição das tarefas CrewAI
│   └── template_registry.json   # Registro de templates de COC
│
└── tools/                       # Ferramentas usadas pelos agentes
    ├── __init__.py
    ├── crew_wrappers.py         # Wrappers @tool para integração com o CrewAI
    ├── drive_tools.py           # Download/listagem/movimentação no Google Drive
    ├── pattern_identifier.py    # Identificação automática de template
    ├── filler_agent.py          # Extração de dados parametrizada
    ├── reviewer.py              # Validação automática dos dados
    ├── feedback_store.py        # Persistência de feedback para evolução
    └── utils.py                 # Funções utilitárias (normalização)
```

---

## 📄 Descrição dos Arquivos

### Raiz

| Arquivo | Descrição | Obrigatório? |
|---|---|---|
| `main.py` | Ponto de entrada. Carrega `.env`, valida `DRIVE_FOLDER_ID`, configura o modelo Gemini e dispara a `crew`. | ✅ Sim |
| `crew.py` | Define a `ProtocrewCrew` usando decorators do CrewAI. Conecta agentes a tarefas em fluxo sequencial. | ✅ Sim |
| `requirements.txt` | Lista de dependências (`crewai`, `google-genai`, `google-api-python-client`, etc.). | ✅ Sim |
| `.env.example` | Modelo de variáveis de ambiente (`GEMINI_API_KEY`, `DRIVE_FOLDER_ID`). Copiar para `.env`. | ✅ Sim |
| `credentials.example.json` | Modelo de credenciais OAuth2 Desktop do Google Drive. Copiar/substituir por `credentials.json`. | ✅ Sim, para acesso ao Drive |

### `config/`

| Arquivo | Descrição | Obrigatório? |
|---|---|---|
| `agents.yaml` | Define os 3 agentes (`drive_manager`, `ocr_specialist`, `data_analyst`) com role, goal e backstory. | ✅ Sim |
| `tasks.yaml` | Define as 3 tarefas sequenciais: `download_files`, `extract_data`, `validate_and_export`. | ✅ Sim |
| `template_registry.json` | Registro de templates de COC (ex.: `LAB_ALPHA`, `LAB_BETA`). Contém âncoras de detecção, mapeamento de campos e regras especiais por template. | ✅ Sim |

### `tools/`

| Arquivo | Descrição | Obrigatório? | Uso em outros projetos |
|---|---|---|---|
| `crew_wrappers.py` | Adapta as ferramentas para o formato `@tool` do CrewAI (`Baixar_Arquivos_Drive`, `Extrair_Dados_Cadeia_Custodia`, `Validar_e_Exportar_Dados`). | ✅ Sim | 🔒 Específico deste projeto |
| `drive_tools.py` | Classe `GoogleDriveDownloadTool`: autenticação OAuth2, listagem, download e movimentação de arquivos no Google Drive. | ✅ Sim | ♻️ **Reutilizável** em qualquer projeto que precise acessar Google Drive |
| `pattern_identifier.py` | Classe `PatternIdentifierTool`: identificação do template de uma COC em **2 camadas** — (1) match rápido por âncoras textuais via regex e (2) fallback por inferência via Gemini. Templates novos são salvos como rascunho no registry. | ✅ Sim | ♻️ Parcialmente reutilizável (lógica de match por âncoras) |
| `filler_agent.py` | Classe `FillerAgentTool`: extração parametrizada de dados. Recebe o `field_map` do `PatternIdentifierTool` e monta um prompt Gemini dinâmico para extrair código, identificação, data e hora. | ✅ Sim | 🔒 Específico (depende do `field_map`) |
| `reviewer.py` | Classe `Reviewer`: validação automática dos dados extraídos. Verifica formato do código (7 dígitos), data válida (`DD/MM/AAAA`), hora (`HH:MM`) e identificação. Classifica amostras como `APROVADO` ou `REJEITADO`. | ✅ Sim | ♻️ **Reutilizável** em projetos de validação de dados laboratoriais |
| `feedback_store.py` | Classe `FeedbackStore`: registra sucesso/falha das extrações por template em `output/feedback/`. Mantém estatísticas para promoção automática de templates (rascunho → validado) e detecção de degradação. | ⚠️ Opcional | ♻️ **Reutilizável** como sistema genérico de feedback e evolução |
| `utils.py` | Funções auxiliares `normalize_time()`, `normalize_code()` e `get_configured_prefixes()`. Corrige formatos comuns de OCR (ex.: `10h35` → `10:35`) com prefixos opcionais configuráveis. | ✅ Sim | ♻️ **Reutilizável** |

---

## ⚙️ Como Funciona (Pipeline)

O pipeline é **sequencial** e opera em 3 etapas:

```text
1. DOWNLOAD          2. EXTRAÇÃO (OCR)         3. VALIDAÇÃO + EXPORT
┌──────────────┐    ┌──────────────────────┐   ┌────────────────────┐
│ drive_manager│───>│  ocr_specialist      │──>│  data_analyst      │
│              │    │                      │   │                    │
│ Baixa PDFs   │    │ 1. Identifica layout │   │ Revisa cada amostra│
│ do Drive     │    │ 2. Extrai dados      │   │ Exporta para CSV   │
│ → input_files│    │    com Gemini OCR    │   │ Remove PDFs locais │
└──────────────┘    └──────────────────────┘   └────────────────────┘
```

### Detalhamento

1. **Download** (`drive_manager`): Conecta ao Google Drive via OAuth2, lista e baixa todos os PDFs da pasta configurada em `DRIVE_FOLDER_ID` para `input_files/`.
2. **Extração** (`ocr_specialist`): Para cada PDF:
   - O `PatternIdentifierTool` identifica o template (`LAB_ALPHA`, `LAB_BETA`, etc.) usando âncoras textuais ou inferência via Gemini.
   - O `FillerAgentTool` monta um prompt parametrizado pelo `field_map` do template e envia ao Gemini para extração dos campos.
3. **Validação e Exportação** (`data_analyst`): O `Reviewer` valida cada amostra (formato do código, data, hora) e o resultado é exportado para `output/resultado_crew.csv`.

---

## 🧩 Templates de Exemplo (`template_registry.json`)

| Template | Status | Descrição |
|---|---|---|
| `LAB_ALPHA` | ✅ Validado | Layout tabular padrão com campos impressos e código manuscrito |
| `LAB_BETA` | ✅ Validado | Layout com identificação composta (`{protocolo} - {ponto_coleta}`) e coleta em célula combinada |

> Templates novos são detectados automaticamente via Gemini e salvos como rascunho no `template_registry.json`. Após 10+ extrações com 90%+ de aprovação, podem ser promovidos para "validado".

---

## 🚀 Instalação e Execução

### Pré-requisitos
- Python 3.10+
- Projeto no Google Cloud Console com a API Google Drive habilitada
- Chave de API do Google Gemini

### Setup

```bash
# 1. Criar e ativar ambiente virtual
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# 2. Instalar dependências
pip install -r requirements.txt

# 3. Configurar variáveis de ambiente a partir do modelo
cp .env.example .env
# Edite o .env preenchendo GEMINI_API_KEY e DRIVE_FOLDER_ID

# 4. Configurar credenciais OAuth2 do Google Drive
# Baixe suas credenciais OAuth2 Desktop no Google Cloud Console e salve como credentials.json
# (use credentials.example.json como referência de estrutura)
cp credentials.example.json credentials.json

# 5. Executar
python main.py
```

Na primeira execução, o fluxo OAuth abrirá o navegador para autorizar o acesso ao Google Drive. O `token.json` será gerado automaticamente (e permanece ignorado pelo `.gitignore`).

---

## 📊 Saída

O resultado é exportado em `output/resultado_crew.csv` com a estrutura:

```csv
Arquivo;Codigo;Identificacao;DataHora;Template;Status;Motivo
doc1.pdf;0001234;2026/001 - Ponto Norte;10/02/2026 09:45;LAB_ALPHA;APROVADO;
doc2.pdf;12345;PONTO-B2;15/03/2026 14:30;LAB_BETA;REJEITADO;Código '12345' não tem 7 dígitos numéricos
```

---

## 🔑 Variáveis de Ambiente

| Variável | Obrigatória? | Descrição |
|---|---|---|
| `GEMINI_API_KEY` | ✅ Sim | Chave da API do Google Gemini para OCR e inferência de layout |
| `DRIVE_FOLDER_ID` | ✅ Sim | ID da pasta do Google Drive contendo os PDFs de entrada |
| `SAMPLE_CODE_PREFIXES` | ❌ Opcional | Lista separada por vírgulas de prefixos esperados para normalização de códigos de amostra |

---

## 📝 Licença

Este projeto é distribuído sob a **MIT License**.
