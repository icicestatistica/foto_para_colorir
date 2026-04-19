"""
app_colorir.py — App interativo para converter imagens em páginas de colorir

Dependências:
    pip install gradio opencv-python pillow

Uso:
    python app_colorir.py
    → abre automaticamente no navegador
"""

import cv2
import gradio as gr
import numpy as np
from pathlib import Path
from PIL import Image

# ── Estado da sessão ────────────────────────────────────────
state = {
    "pasta":    None,
    "arquivos": [],
    "indice":   0,
}

EXTENSOES = (".png", ".jpg", ".jpeg")


def carregar_pasta(pasta_str):
    pasta = Path(pasta_str.strip())
    if not pasta.exists():
        return None, None, "❌ Pasta não encontrada.", gr.update(interactive=False), gr.update(interactive=False)

    arquivos = sorted([f for f in pasta.iterdir() if f.suffix.lower() in EXTENSOES])
    if not arquivos:
        return None, None, "❌ Nenhuma imagem encontrada.", gr.update(interactive=False), gr.update(interactive=False)

    state["pasta"]    = pasta
    state["arquivos"] = arquivos
    state["indice"]   = 0

    img_orig, img_prev = preview(0, 50, 150, 3, 3)
    status = _status()
    return img_orig, img_prev, status, gr.update(interactive=True), gr.update(interactive=True)


def _status():
    i   = state["indice"]
    tot = len(state["arquivos"])
    nome = state["arquivos"][i].name if state["arquivos"] else ""
    return f"📄 {i+1} / {tot}  —  {nome}"


def processar(img_bgr, t1, t2, blur, dilatacao):
    blur = int(blur) | 1          # garante ímpar
    cinza     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    suavizado = cv2.GaussianBlur(cinza, (blur, blur), 0)
    bordas    = cv2.Canny(suavizado, int(t1), int(t2))
    if dilatacao > 0:
        k      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(dilatacao), int(dilatacao)))
        bordas = cv2.dilate(bordas, k, iterations=1)
    return cv2.bitwise_not(bordas)


def preview(indice, t1, t2, blur, dilatacao):
    if not state["arquivos"]:
        return None, None
    f       = state["arquivos"][int(indice)]
    img_bgr = cv2.imread(str(f))
    resultado = processar(img_bgr, t1, t2, blur, dilatacao)
    orig_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(orig_rgb), Image.fromarray(resultado)


def atualizar_preview(t1, t2, blur, dilatacao):
    orig, prev = preview(state["indice"], t1, t2, blur, dilatacao)
    return orig, prev


def salvar_e_avancar(t1, t2, blur, dilatacao):
    if not state["arquivos"]:
        return None, None, "Carregue uma pasta primeiro."

    # salva a imagem atual
    f       = state["arquivos"][state["indice"]]
    img_bgr = cv2.imread(str(f))
    resultado = processar(img_bgr, t1, t2, blur, dilatacao)

    pasta_saida = state["pasta"] / "colorir"
    pasta_saida.mkdir(exist_ok=True)
    cv2.imwrite(str(pasta_saida / f.name), resultado)

    # avança para a próxima
    state["indice"] += 1
    if state["indice"] >= len(state["arquivos"]):
        return None, None, f"✅ Todas as {len(state['arquivos'])} imagens exportadas para '{pasta_saida}'!"

    orig, prev = preview(state["indice"], t1, t2, blur, dilatacao)
    return orig, prev, _status()


def pular():
    if not state["arquivos"]:
        return None, None, "Carregue uma pasta primeiro."
    state["indice"] += 1
    if state["indice"] >= len(state["arquivos"]):
        return None, None, "✅ Fim das imagens."
    orig, prev = preview(state["indice"], 50, 150, 3, 3)
    return orig, prev, _status()


# ── Interface ───────────────────────────────────────────────
with gr.Blocks(title="Páginas de Colorir", theme=gr.themes.Soft()) as app:
    gr.Markdown("# 🎨 Conversor de Imagens para Colorir")

    with gr.Row():
        txt_pasta = gr.Textbox(label="📁 Pasta com as imagens", placeholder="C:/Users/voce/imagens", scale=4)
        btn_carregar = gr.Button("Carregar", variant="primary", scale=1)

    status = gr.Markdown("_Informe uma pasta e clique em Carregar._")

    with gr.Row():
        img_original = gr.Image(label="Original", interactive=False)
        img_preview  = gr.Image(label="Preview — para colorir", interactive=False)

    with gr.Row():
        sl_t1        = gr.Slider(0,   300, value=50,  step=5,   label="t1 — limiar mínimo (menos bordas ↑)")
        sl_t2        = gr.Slider(0,   500, value=150, step=5,   label="t2 — limiar máximo (mais limpo ↑)")
    with gr.Row():
        sl_blur      = gr.Slider(1,   11,  value=3,   step=2,   label="blur — suavização (remove ruído ↑)")
        sl_dilatacao = gr.Slider(0,   8,   value=3,   step=1,   label="dilatação — funde linhas duplas ↑")

    with gr.Row():
        btn_ok   = gr.Button("✅ Salvar e avançar", variant="primary",   interactive=False)
        btn_skip = gr.Button("⏭️ Pular",            variant="secondary", interactive=False)

    # eventos
    sliders = [sl_t1, sl_t2, sl_blur, sl_dilatacao]

    btn_carregar.click(
        carregar_pasta,
        inputs=[txt_pasta],
        outputs=[img_original, img_preview, status, btn_ok, btn_skip]
    )

    for sl in sliders:
        sl.change(atualizar_preview, inputs=sliders, outputs=[img_original, img_preview])

    btn_ok.click(
        salvar_e_avancar,
        inputs=sliders,
        outputs=[img_original, img_preview, status]
    )

    btn_skip.click(
        pular,
        outputs=[img_original, img_preview, status]
    )

if __name__ == "__main__":
    app.launch(inbrowser=True)
