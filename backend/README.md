# Backend HAR (FastAPI)

Servico para receber dados do app e exibir dashboard web simples.

## O que esse backend faz

- `POST /v1/coletas`: recebe 1 janela com 200 amostras do acelerometro
- `GET /v1/coletas`: lista registros armazenados
- `GET /v1/dashboard-summary`: agrega dados para graficos
- `GET /`: pagina web com tabela e graficos de pizza

## Estrutura de payload esperada

```json
{
  "device": "iPhone 13, iOS 18",
  "user_name": "Marcos",
  "samples": [
    {"t": "2026-05-20T12:00:00Z", "x": 0.11, "y": -9.8, "z": 0.31}
  ],
  "confidence": 0.87,
  "top_class_probability": 0.91,
  "predicted_class": "Walking",
  "real_class": "Walking",
  "is_correct": true
}
```

Observacao: `samples` deve ter exatamente 200 itens.

## Como rodar

1. Criar ambiente virtual e instalar dependencias:

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Subir API:

```bash
uvicorn app.main:app --reload --port 8000
```

3. Abrir dashboard:

- http://localhost:8000

## Banco

- SQLite local em `backend/data/har.db`

## Classes aceitas

- Downstairs
- Jogging
- Sitting
- Standing
- Upstairs
- Walking
