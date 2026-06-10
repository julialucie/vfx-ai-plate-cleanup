"""
Foundry Nuke — AI Plate Cleanup Gizmo Button Script.

Drop this into your Nuke menu.py or attach it to a Python Button knob on a
Group/Gizmo node. When clicked it:
  1. Reads the selected Read node's file path and the cursor position knob.
  2. POSTs to the plate-cleanup FastAPI backend.
  3. Writes the conformed EXR path back into a connected Write node (or shows it).

Installation
------------
Add to ~/.nuke/menu.py:

    import plate_cleanup_button
    toolbar = nuke.menu("Nodes")
    m = toolbar.addMenu("VFX AI")
    m.addCommand("Plate Cleanup", "plate_cleanup_button.run()")

Or paste the run() body into a Python Button knob's script field.
"""

import json
import urllib.request
import urllib.error

API_URL = "http://localhost:8000/cleanup"  # Override via Nuke environment var if needed


def _get_api_url() -> str:
    import os
    return os.environ.get("PLATE_CLEANUP_API_URL", API_URL)


def run():
    """
    Main entry point called by the Nuke Python Button.
    Reads node context, calls API, injects result path.
    """
    try:
        import nuke
    except ImportError:
        raise RuntimeError("This script must run inside Foundry Nuke.")

    # --- Resolve the selected Read node ---
    node = nuke.selectedNode()
    if node.Class() not in ("Read", "ReadGeo"):
        nuke.message("[Plate Cleanup] Select a Read node first.")
        return

    exr_path = node["file"].evaluate()
    if not exr_path:
        nuke.message("[Plate Cleanup] Read node has no file path set.")
        return

    # --- Read cleanup coordinates from knobs (add these to your Group node) ---
    try:
        coord_x = int(node.knob("cleanup_x").value()) if node.knob("cleanup_x") else 0
        coord_y = int(node.knob("cleanup_y").value()) if node.knob("cleanup_y") else 0
    except Exception:
        coord_x, coord_y = 0, 0

    prompt = "film plate background, seamless, matching grain, photorealistic"
    if node.knob("cleanup_prompt"):
        prompt = node.knob("cleanup_prompt").value() or prompt

    # --- Call the API ---
    payload = json.dumps({
        "exr_path": exr_path,
        "coord_x": coord_x,
        "coord_y": coord_y,
        "prompt": prompt,
    }).encode("utf-8")

    req = urllib.request.Request(
        _get_api_url(),
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        nuke.message(f"[Plate Cleanup] API connection failed:\n{exc.reason}\n\nIs the server running at {_get_api_url()}?")
        return
    except Exception as exc:
        nuke.message(f"[Plate Cleanup] Unexpected error: {exc}")
        return

    output_path = result.get("output_path", "")
    job_id = result.get("job_id", "unknown")

    # --- Wire result into a new Read node downstream ---
    clean_read = nuke.createNode("Read", inpanel=False)
    clean_read["file"].setValue(output_path)
    clean_read["label"].setValue(f"AI Cleanup\nJob: {job_id[:8]}")
    nuke.message(
        f"[Plate Cleanup] Done!\n\nJob: {job_id}\nConformed plate: {output_path}\n\nA new Read node has been created."
    )
