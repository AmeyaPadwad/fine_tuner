# Humanitarians AI — Volunteer Work

Code, experiments and documentation for my volunteer work with [Humanitarians AI](https://www.humanitarians.ai),
under the Iacon Optimus program (Nik Bear Brown, Northeastern University).

Maintained by Pia Adlakha ([priyaadlakha.net](https://priyaadlakha.net)).

## Contents

- [`projects/iacon-optimus-tool-selection/`](projects/iacon-optimus-tool-selection/) — Fine-tune FunctionGemma 270M for tool selection (current project)
- [`logs/`](logs/) — Weekly hour logs
- [`docs/`](docs/) — Notes, decisions and reading list

## Weekly logs

Logged in [`logs/`](logs/), one file per week. Each entry lists hours, what was done, and links to
sources or commits.

## Current project

**Fine-tune FunctionGemma 270M for tool selection** — research question: can a fine-tuned
~270M-parameter model select tools reliably enough to remove tool-routing work from the main agent?
See [`projects/iacon-optimus-tool-selection/README.md`](projects/iacon-optimus-tool-selection/README.md)
for the full scope, pipeline and results.

## Setup

```bash
git clone https://github.com/adlakhapriya95/humanitarians-ai.git
cd humanitarians-ai
python -m venv .venv
source .venv/bin/activate
pip install -r projects/iacon-optimus-tool-selection/requirements.txt
```

## License

Code in this repo is licensed under MIT (see [LICENSE](LICENSE)). Model weights (FunctionGemma)
remain under Google's Gemma Terms of Use — see the project README for details.
