# ChatAI from Scratch

Dieses Repository demonstriert, wie man einen einfachen Transformer-Chatbot **komplett ohne Deep-Learning-Frameworks** nur mit NumPy implementiert. Zielgruppe sind Studierende und Entwickler, die die Konzepte eines Seq2Seq-Transformers ohne "Magie" verstehen möchten.

Das Skript `src/chatbot_from_scratch.py` zeigt Schritt für Schritt:

1. **Vokabular-Erstellung** mit einer eigenen `Tokenizer`-Klasse.
2. **Transformer-Blöcke** (Multi-Head Attention, Feed-Forward, LayerNorm).
3. **Seq2Seq-Modell** aus Encoder und Decoder.
4. **Trainingsloop** inklusive eines einfachen Adam-Optimizers.

## Eingabedaten

Die Trainingsdaten werden im JSONL-Format erwartet. Jede Zeile enthält ein JSON-Objekt mit den Feldern `input` und `response`:

```jsonl
{"input": "Hallo, wie geht's?", "response": "Mir geht's gut."}
```

Eine Vorlage findest du unter `data/example_template.jsonl`.

Beim ersten Start erzeugt das Skript automatisch Dateien wie `vocab_token2id.npy`, `vocab_id2token.npy` und legt trainierte Gewichte in `checkpoints/` ab.

## Installation

Python 3.8 oder höher wird benötigt. Installiere die Abhängigkeiten mit:

```bash
pip install -r requirements.txt
```

`numpy` ist zwingend erforderlich, `tqdm` wird optional für Fortschrittsbalken genutzt.

## Verzeichnisstruktur (Beispiel)

```
.
├── src/
│   └── chatbot_from_scratch.py
├── data/
│   └── example_template.jsonl
├── checkpoints/
├── requirements.txt
└── README.md
```

Während des Trainings werden automatisch `vocab_token2id.npy`, `vocab_id2token.npy` und Dateien wie `checkpoints/epoch_1.npz` erzeugt.

## Kurzer Einstieg

Trainieren des Modells:

```bash
python src/chatbot_from_scratch.py --mode train --train_file data/train.jsonl --valid_file data/valid.jsonl --epochs 10 --batch_size 32 --lr 1e-4
```

Nach dem Training kann der Chat-Modus gestartet werden:

```bash
python src/chatbot_from_scratch.py --mode chat --checkpoint_dir checkpoints/
```

## Was fehlt noch und mögliche Erweiterungen

- Dropout und Label-Smoothing sind bewusst minimal gehalten und könnten verbessert werden.
- Logging und Visualisierung (z.B. TensorBoard) fehlen.
- Beam-Search-Parameter lassen sich erweitern, ebenso Top-k und Top-p Sampling.

Viel Spaß beim Experimentieren! 🚀
