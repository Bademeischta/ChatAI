# ChatAI from Scratch

Dieses Projekt demonstriert einen kleinen Transformer-Chatbot. Der Trainingscode nutzt nun PyTorch, um echte Gradientenberechnungen und Optimierung zu ermöglichen. Die Module unter `src/` dienen als einfache Lernbasis.

## Struktur

```
src/
  tokenizer.py      # Tokenisierung und Vokabular
  transformer.py    # Transformer-Bausteine (Attention, FFN, LayerNorm)
  model.py          # Seq2SeqTransformer
  optimizer.py      # Einfacher Adam-Optimizer
  train.py          # Training und einfache CLI
  inference.py      # Greedy-Decoder
  utils.py          # Hilfsfunktionen und Config-Klasse
```

Unter `tests/` befinden sich ein paar Unit-Tests für Tokenizer, Transformer und Optimizer.

## Installation

```bash
pip install -r requirements.txt
# falls keine GPU vorhanden ist
pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu
```

## Beispiel-Daten

Die Daten liegen im JSONL-Format vor. Eine Vorlage befindet sich in `data/example_template.jsonl`:

```jsonl
{"input": "Hier steht eine Beispiel-Eingabe.", "response": "Hier steht eine Beispiel-Antwort."}
```

Beim ersten Training baut der Tokenizer sein Vokabular automatisch auf. Anschließend kann es mit

```bash
python - <<'EOF'
from src.tokenizer import Tokenizer
tok = Tokenizer()
tok.build_vocab_from_texts(["Hallo", "Welt"])
tok.save_vocab('data/vocab.json')
EOF
```

gespeichert und über `Tokenizer.load_vocab()` wieder geladen werden.

## Erstes Training

```bash
python src/train.py --mode train --config_file config.json
```

Nach dem Training entsteht ein Checkpoint unter `checkpoints/` (Datei `.pt`). Zum einfachen Chatten:

```bash
python src/train.py --mode chat --config_file config.json
```

Weitere Einstellungen können über `config.json` vorgenommen werden (z.B. `beam_width`, `top_k`, `top_p`).

## Roadmap

- Ausarbeitung von Beam-Search und Sampling
- GPU-Unterstützung via CuPy
- Bessere Logging-Optionen
- Unit-Tests erweitern und automatische Gradientenberechnung
