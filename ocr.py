import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import torch  # noqa: F401  debe importarse antes que paddle (conflicto de DLLs en Windows)
import paddle
from paddleocr import PaddleOCR

from workers import enviar_carga_lote, guardar_resultado

print(f"Paddle compilado con CUDA: {paddle.device.is_compiled_with_cuda()}")
print(f"GPUs detectadas: {paddle.device.cuda.device_count() if paddle.device.is_compiled_with_cuda() else 0}")

RUTA_ENTRADA = "img_ocr"  # carpeta con imágenes (o un archivo)
RUTA_SALIDA = "output"
TAMANO_LOTE = 8  # imágenes por llamada al modelo
NUM_WORKERS_CARGA = 4
NUM_WORKERS_GUARDADO = 2
INCLUIR_COORDENADAS = False

EXTENSIONES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

# Create the pipeline once and reuse it — loading models is the slow part
ocr = PaddleOCR(
    lang="es",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    device="gpu:0",
)
print(f"Dispositivo en uso: {paddle.device.get_device()}")


def listar_imagenes(ruta: Path) -> list[Path]:
    if ruta.is_file():
        return [ruta]
    if ruta.is_dir():
        return sorted(p for p in ruta.iterdir() if p.suffix.lower() in EXTENSIONES)
    raise FileNotFoundError(f"No existe la ruta: {ruta}")


def dividir_en_lotes(rutas: list[Path], tamano: int) -> list[list[Path]]:
    return [rutas[i:i + tamano] for i in range(0, len(rutas), tamano)]


def recolectar_lote(rutas: list[Path], futuros: list[Future]) -> tuple[list[Path], list]:
    """Espera las cargas del lote; descarta (y reporta) las imágenes que fallaron."""
    rutas_ok, imagenes = [], []
    for ruta, futuro in zip(rutas, futuros):
        try:
            imagenes.append(futuro.result())
            rutas_ok.append(ruta)
        except Exception as e:
            print(f"  Error cargando {ruta.name}: {e}")
    return rutas_ok, imagenes


def inferir_lote(imagenes: list) -> tuple[list, float]:
    """Una sola llamada al modelo (único, en GPU) para todo el lote."""
    inicio = time.perf_counter()
    resultados = list(ocr.predict(imagenes))
    total = time.perf_counter() - inicio
    return resultados, total / len(imagenes)  # tiempo promedio por imagen


def procesar_lotes(rutas: list[Path], salida: Path) -> None:
    """Pipeline: mientras la GPU infiere el lote N, los workers cargan el N+1
    y guardan los resultados del N-1."""
    lotes = dividir_en_lotes(rutas, TAMANO_LOTE)
    guardados: list[Future] = []

    with ThreadPoolExecutor(NUM_WORKERS_CARGA, "carga") as pool_carga, \
            ThreadPoolExecutor(NUM_WORKERS_GUARDADO, "guardado") as pool_guardado:

        pendiente = enviar_carga_lote(pool_carga, lotes[0])
        for n, lote in enumerate(lotes):
            rutas_ok, imagenes = recolectar_lote(lote, pendiente)

            # Prefetch del siguiente lote antes de ocupar la GPU
            if n + 1 < len(lotes):
                pendiente = enviar_carga_lote(pool_carga, lotes[n + 1])

            if not imagenes:
                continue
            print(f"Lote {n + 1}/{len(lotes)}: {len(imagenes)} imágenes")
            resultados, tiempo = inferir_lote(imagenes)

            for ruta, res in zip(rutas_ok, resultados):
                guardados.append(pool_guardado.submit(
                    guardar_resultado, salida, ruta, res, round(tiempo, 2), INCLUIR_COORDENADAS))

        for futuro in guardados:
            try:
                futuro.result()
            except Exception as e:
                print(f"  Error guardando: {e}")


def main() -> None:
    imagenes = listar_imagenes(Path(RUTA_ENTRADA))
    salida = Path(RUTA_SALIDA)
    if not imagenes:
        print("No se encontraron imágenes.")
        return

    salida.mkdir(parents=True, exist_ok=True)
    inicio = time.perf_counter()
    procesar_lotes(imagenes, salida)
    print(f"Listo: {len(imagenes)} imágenes en {time.perf_counter() - inicio:.2f}s")


if __name__ == "__main__":
    main()
