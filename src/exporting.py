import io
import re
import zipfile


_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


def _safe_image_stem(image_name: str) -> str:
    filename = image_name.replace("\\", "/").rsplit("/", 1)[-1]
    stem = filename.rsplit(".", 1)[0] if "." in filename else filename
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    return safe_stem or "imagen"


def build_export_zip(
    image_name: str,
    overlay_png: bytes,
    mask_png: bytes,
    heatmap_png: bytes,
    csv_data: bytes,
    json_data: bytes,
) -> tuple[bytes, str]:
    """Build an in-memory ZIP containing all GretaVision exports."""
    image_stem = _safe_image_stem(image_name)
    files = (
        (f"{image_stem}_overlay.png", overlay_png),
        (f"{image_stem}_mascara.png", mask_png),
        (f"{image_stem}_heatmap.png", heatmap_png),
        (f"{image_stem}_metricas.csv", csv_data),
        (f"{image_stem}_reporte.json", json_data),
    )

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for filename, content in files:
            member = zipfile.ZipInfo(filename, date_time=_ZIP_TIMESTAMP)
            member.compress_type = zipfile.ZIP_DEFLATED
            member.external_attr = 0o600 << 16
            archive.writestr(member, content)

    zip_name = f"{image_stem}_resultados_gretavision.zip"
    return buffer.getvalue(), zip_name


build_results_zip = build_export_zip
