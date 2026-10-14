"""Tareas de I/O que corren en hilos (workers).

La GPU/modelo NO se toca aquí: solo hay un modelo, usado desde el hilo
principal. Los workers se encargan de lo que no necesita GPU:
leer/decodificar imágenes y escribir resultados a disco.
"""
import json
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np


def cargar_imagen(ruta: Path) -> np.ndarray:
    """Lee y decodifica una imagen (BGR). Corre en un worker de carga."""
    datos = np.fromfile(str(ruta), dtype=np.uint8)
    imagen = cv2.imdecode(datos, cv2.IMREAD_COLOR)
    if imagen is None:
        raise ValueError(f"No se pudo decodificar: {ruta.name}")
    return imagen


def enviar_carga_lote(pool: ThreadPoolExecutor, rutas: list[Path]) -> list[Future]:
    """Encola la carga de un lote; devuelve un Future por imagen (mismo orden)."""
    return [pool.submit(cargar_imagen, ruta) for ruta in rutas]


def construir_resultado(ruta: Path, res, tiempo_segundos: float, incluir_coordenadas: bool) -> dict:
    """Convierte el resultado de PaddleOCR al formato JSON deseado."""
    textos = []
    for texto, conf, poly in zip(res["rec_texts"], res["rec_scores"], res["rec_polys"]):
        item = {"texto": texto, "confianza": round(float(conf), 4)}
        if incluir_coordenadas:
            item["coordenadas"] = [[int(x), int(y)] for x, y in poly]
        textos.append(item)
    return {"archivo": ruta.stem, "tiempo_segundos": tiempo_segundos, "textos": textos}


def guardar_json(salida: Path, resultado: dict) -> None:
    destino = salida / f"{resultado['archivo']}.json"
    destino.write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")


def guardar_imagen_anotada(salida: Path, ruta: Path, res) -> None:
    """Guarda la imagen con las cajas dibujadas, con nombre propio por archivo."""
    res.img["ocr_res_img"].save(salida / f"{ruta.stem}_ocr.png")


def guardar_resultado(salida: Path, ruta: Path, res, tiempo_segundos: float, incluir_coordenadas: bool) -> str:
    """Tarea completa de escritura (JSON + imagen). Corre en un worker de guardado."""
    resultado = construir_resultado(ruta, res, tiempo_segundos, incluir_coordenadas)
    guardar_json(salida, resultado)
    guardar_imagen_anotada(salida, ruta, res)
    return ruta.name
