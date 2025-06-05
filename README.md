# ChatAI from Scratch

Dieses Projekt demonstriert einen einfachen Transformer-Chatbot, implementiert ausschließlich mit NumPy. Der Code ist in mehrere Module unter `src/` aufgeteilt und soll zum Lernen der Kernkonzepte eines Seq2Seq-Transformers dienen.

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
```

## Beispiel-Daten

Die Daten liegen im JSONL-Format vor. Eine Vorlage befindet sich in `data/example_template.jsonl`:

```jsonl
{"input": "Hier steht eine Beispiel-Eingabe.", "response": "Hier steht eine Beispiel-Antwort."}
```

## Erstes Training

```bash
python src/train.py --mode train --config_file config.json
```

Danach wird ein Checkpoint unter `checkpoints/` erzeugt. Zum Chatten (rudimentär):

```bash
python src/train.py --mode chat --config_file config.json
```

Weitere Einstellungen können über `config.json` vorgenommen werden (z.B. `beam_width`, `top_k`, `top_p`).

## Roadmap

- Ausarbeitung von Beam-Search und Sampling
- GPU-Unterstützung via CuPy
- Bessere Logging-Optionen
- Unit-Tests erweitern und automatische Gradientenberechnung
