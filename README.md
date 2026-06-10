# High-Dynamic-Range AI Plate Cleanup Pipeline

An automated engineering pipeline that bridges Hollywood-standard image files with frontier computer vision models to isolate and strip out on-set tracking artifacts dynamically.

## The Engineering Problem
Traditional AI models process standard 8-bit images (JPEGs/PNGs) bounded tightly between values of `0` and `255`. Professional film pipelines utilize 32-bit floating-point linear OpenEXR files to store uncompressed light values captured by camera sensors. 

Passing linear float values directly into conventional neural network encoders crashes or creates severe numerical clipping errors. Furthermore, generative AI tools frequently introduce random hallucinations, changing pixels outside the artist's targeted work zone and breaking visual continuity between consecutive film frames.

## Pipeline Solution Architecture

This system implements a production-grade data loop across three functional modules:

### 1. Ingestion & Dynamic Data Interleaving
Reads uncompressed `.exr` file metadata headers to resolve dynamic image resolutions. The ingestion loop converts the file's native planar layer layout structure into an interleaved multi-channel floating-point data tensor:
$$\mathbf{I} \in \mathbb{R}^{H \times W \times 4}$$

### 2. Neural Context Mapping & Dynamic Stenciling
To prevent network encoding failure due to extreme high-exposure highlights, the pipeline feeds the data through a localized Reinhard tonemapping viewing curve, scaling infinite float parameters safely down to bounded 8-bit integer formats. The data maps directly to a specified pixel vector sequence context $(X,Y)$ passed to a localized foundation model (Segment Anything) to isolate and calculate structural artifact boundaries, exporting a tight mathematical 2D binary stencil layout file:
$$\mathbf{M} \in \{0, 1\}^{H \times W}$$

### 3. Absolute Matrix Composition & Asset Serialization
To guarantee that background pixel matrices remain completely safe from model distortion or texture flickering, the script executes a strict masking composition matrix multiplication equation:
$$\mathbf{I}_{\text{final}} = (1 - \mathbf{M}) \odot \mathbf{I}_{\text{original}} + \mathbf{M} \odot \mathbf{I}_{\text{generated}}$$

The modified matrix packs color data layers back into planar byte streams, casting parameters down to half-precision (`FP16`) format strings to output a conformed studio asset file containing a pixel-perfect alpha transparency opening.

## Quick Start Configuration

Ensure your local or server environment matches the specifications within `requirements.txt`. Execute the headless command-line interface module like this:

```bash
python conformed_pipeline.py --input shot_010.exr --output conformed_shot_010.exr --x 500 --y 300
```
