# Flappy Bird behavior-cloned browser demo

This is a separate Pygbag version of the desktop demo. It uses the same
4-element observation, normalization constants, `4 -> 128 -> 128 -> 2`
network, and Flappy Bird physics, scoring, and collision rules. It does not
use a teacher, random actions, replay buffer, or training.

## Why the browser model is exported

Pygbag's browser runtime does not provide desktop PyTorch. Keep the original
`models/flappy_bc.pth` unchanged and in this folder; `export_model.py` reads
that checkpoint on the desktop and writes `models/flappy_bc.json`, containing
the same six float32 tensors. In the browser, the policy applies those weights
with a pure-Python dense-network forward pass and chooses the maximum output.
Because this forward pass has no autograd engine, no gradient tracking is
created. The desktop demo and checkpoint are not affected.

## Prepare and build

From the repository root, using the project's existing virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r .\web_demo\requirements.txt
Copy-Item .\models\flappy_bc.pth .\web_demo\models\flappy_bc.pth -Force
.\.venv\Scripts\python.exe .\web_demo\export_model.py
.\.venv\Scripts\python.exe -m pygbag --build .\web_demo
```

The build output is `web_demo/build/web/`.

## Test locally

Start Pygbag's local server from the repository root:

```powershell
.\.venv\Scripts\python.exe -m pygbag .\web_demo
```

Open `http://localhost:8000/` in a current browser. The Pygbag runtime may
need to download its WebAssembly runtime on the first visit. The canvas keeps
the game's 9:16 aspect ratio; the neural policy controls every in-game action.
After a collision, press **R**, **Space**, **Enter**, or click the canvas to
start a new episode. Each episode gets a distinct seed; a fresh base seed is
generated for every page load.

For reproducible local browser testing, Pygbag can pass environment variables
in the query string; for example:

```text
http://localhost:8000/?FLAPPY_BC_SEED=2026
```

## GitHub Pages

The repository's `.github/workflows/deploy-pages.yml` workflow builds this
folder and publishes `web_demo/build/web/` through GitHub Pages. Enable Pages
in the repository settings and select **GitHub Actions** as the deployment
source. The workflow runs on pushes to `main` and can also be started manually
from the Actions tab.

For manual deployment, publish the contents of `web_demo/build/web/` as the
Pages artifact. Do not publish the source folder in place of the build output:
the generated `index.html`, runtime assets, and packaged game files are
required by the browser.
