

# 🎯 Sistema Inteligente de Monitoramento com YOLOv8

[![Streamlit App](https://img.shields.io/badge/🟢%20Acesse%20o%20App-Streamlit-green)](https://deteccao.streamlit.app/)

Sistema de monitoramento em vídeo baseado em **Visão Computacional** e **YOLOv8**, capaz de detectar pessoas, animais e outros objetos em tempo real. Ideal para aplicações em ambientes comerciais, industriais ou de vigilância sanitária.

> 🔗 **Acesse a aplicação ao vivo**: [https://deteccao.streamlit.app/](https://deteccao.streamlit.app/)

---

## 🚀 Funcionalidades

- 🔍 **Detecção em Tempo Quase Real** utilizando o modelo **YOLOv8n** (Ultralytics).
- 🎯 **Filtro por classe de objeto** (ex: pessoa, cavalo, cachorro, etc).
- 📊 **Gráfico de detecções** por tipo de objeto identificado.
- 📋 **Relatório detalhado** com timestamp de início/fim e número de ocorrências por classe.
- 💾 **Exportação** do relatório em **CSV**.
- 📥 **Download do vídeo processado** com as detecções destacadas.
- ⏱️ **Barra de progresso** visual para feedback em tempo real do processamento.

---

## 🧠 Tecnologias Utilizadas

- [YOLOv8 (Ultralytics)](https://docs.ultralytics.com)
- [Streamlit](https://streamlit.io)
- OpenCV
- Pandas
- Python 3.9+

---

## 🛠️ Como Executar Localmente

### 1. Clone o repositório
```bash
git clone https://github.com/opablodantas/yolov8-video-analyzer.git
cd yolov8-video-analyzer
````

### 2. Crie o ambiente e instale dependências

```bash
python -m venv venv
source venv/bin/activate  # ou venv\Scripts\activate no Windows
pip install -r requirements.txt
```

### 3. Rode o aplicativo

```bash
streamlit run deteccao.py
```

---

## 📁 Estrutura Principal

```plaintext
├── deteccao.py             # Aplicação principal Streamlit
├── requirements.txt        # Dependências
├── README.md               # Este arquivo
```

---

## 📌 Exemplos de Uso

* **Restaurantes e supermercados**: monitoramento de pragas, presença de pessoas em áreas restritas.
* **Fábricas e armazéns**: vigilância e contagem de fluxo operacional.
* **Lojas e shoppings**: análise de movimentação e comportamento do consumidor.

---

## 👨‍💻 Desenvolvido por

**\[Pablo Dantas]**
Engenheiro de Inteligência Artificial
[LinkedIn](https://www.linkedin.com/in/pablodantasevangelista/)

---

## 🛰️ Acesse agora:

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://deteccao.streamlit.app/)

