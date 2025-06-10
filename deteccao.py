import streamlit as st
import cv2
import tempfile
import os
from ultralytics import YOLO
from collections import defaultdict
import pandas as pd

# ------------------------------
# Menu de Navegação
# ------------------------------
st.set_page_config(page_title="Sistema de Monitoramento", layout="wide")
menu = st.sidebar.radio("Navegação", ["🏠 HOME", "ℹ️ SOBRE"])

# ------------------------------
# Página HOME
# ------------------------------
if menu == "🏠 HOME":
    st.title("📹 Sistema de Monitoramento com Visão Computacional")
    st.markdown("Envie um vídeo MP4 para detectar pessoas, cavalos e outros animais em tempo real utilizando o modelo YOLOv8n.")

    # Upload do vídeo
    uploaded_file = st.file_uploader("Selecione um vídeo .mp4", type=["mp4"])

    if uploaded_file:
        # Salvar vídeo temporariamente
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_file.read())
            video_path = tmp.name

        # Exibir o vídeo original enviado pelo usuário
        st.subheader("🎞️ Vídeo Original")
        st.video(video_path)

        # Carregar o modelo YOLOv8n pré-treinado
        model = YOLO("yolov8n.pt")

        # Abrir o vídeo com OpenCV
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        # Definir caminho de saída para o vídeo processado
        output_path = os.path.join(tempfile.gettempdir(), "output_detectado.mp4")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Dicionário para armazenar as detecções por classe e tempo
        detections_por_classe = defaultdict(list)
        frame_count = 0

        # Informar ao usuário que o vídeo está sendo processado
        st.info("🔎 Processando vídeo, aguarde...")

        # Loop por todos os frames do vídeo
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Calcular timestamp do frame atual
            timestamp = frame_count / fps

            # Executar detecção com YOLO
            results = model(frame, verbose=False)[0]

            # Registrar detecções por classe
            for box in results.boxes:
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
                detections_por_classe[cls_name].append(timestamp)

            # Desenhar bounding boxes nos frames
            frame_com_box = results.plot()
            out.write(frame_com_box)

            frame_count += 1

        # Finalizar os objetos de vídeo
        cap.release()
        out.release()

        # Exibir vídeo com detecções
        st.subheader("🧠 Vídeo com Detecções")
        st.video(output_path)

        # Gráfico de barras com contagem de detecções por classe
        st.subheader("📊 Gráfico de Detecções por Classe")
        contagens = {classe: len(tempos) for classe, tempos in detections_por_classe.items()}
        chart_data = pd.DataFrame.from_dict(contagens, orient="index", columns=["Detecções"])
        chart_data = chart_data.sort_values("Detecções", ascending=False)
        st.bar_chart(chart_data)

        # Gerar relatório com intervalo de tempo por classe
        st.subheader("📋 Relatório de Detecções por Intervalo de Tempo")
        relatorio = []
        for classe, tempos in detections_por_classe.items():
            if tempos:
                tempos.sort()
                relatorio.append({
                    "Classe": classe,
                    "Início (s)": round(min(tempos), 2),
                    "Fim (s)": round(max(tempos), 2),
                    "Total de Detecções": len(tempos)
                })

        df_relatorio = pd.DataFrame(relatorio).sort_values("Total de Detecções", ascending=False)
        st.dataframe(df_relatorio)

        # Botão para download do vídeo com as detecções
        with open(output_path, "rb") as f:
            st.download_button("📥 Baixar vídeo com detecções", f, "video_detectado.mp4", mime="video/mp4")

# ------------------------------
# Página SOBRE
# ------------------------------
elif menu == "ℹ️ SOBRE":
    st.title("ℹ️ Sobre o Projeto")

    st.markdown("""
    Em um mercado cada vez mais competitivo, a reputação de um negócio pode ser impactada por detalhes que muitas vezes passam despercebidos. Pensando nisso, nosso projeto de **Bounding Box com detecção de movimento** oferece uma solução inovadora e inteligente para estabelecimentos como restaurantes, lojas, supermercados e outros ambientes comerciais.

    Por meio de algoritmos de visão computacional, o sistema identifica com precisão movimentações em áreas críticas, permitindo monitorar em tempo real tanto atividades humanas quanto possíveis ameaças à higiene, como a presença de pragas ou comportamentos fora do padrão. Essa tecnologia proporciona benefícios diretos e concretos:

    - **Prevenção de problemas sanitários:** Ao detectar movimentos suspeitos, como pequenos animais ou insetos, é possível agir de forma proativa antes que eles afetem a operação ou imagem do local.
    - **Melhoria na vigilância e segurança:** O sistema reforça o monitoramento de áreas sensíveis, evitando furtos, entradas não autorizadas e contribuindo para um ambiente mais seguro.
    - **Apoio na gestão operacional:** A análise de movimento pode fornecer insights sobre fluxo de pessoas, horários de pico e eficiência da equipe, otimizando processos e decisões de vendas.
    - **Proteção da reputação da marca:** Ao garantir ambientes limpos, controlados e seguros, o projeto ajuda a construir uma imagem sólida e confiável perante os clientes.

    Mais do que um sistema de detecção, este projeto representa uma ferramenta estratégica de gestão, que une tecnologia, prevenção e inteligência de dados para transformar desafios operacionais em vantagens competitivas.
    """)
