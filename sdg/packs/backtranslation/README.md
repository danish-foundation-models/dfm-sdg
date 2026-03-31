# backtranslation

`backtranslation` is a small pack for instruction backtranslation.

It starts from finished articles, filters them, and asks one model to write the user prompt that would have produced each article.

## Current shape

- source articles can come from a local JSONL file or a Hugging Face dataset
- filtering is intentionally minimal and currently supports `min_article_chars`
- one model role, `instruction_writer`, generates the prompt for each kept article
- outputs are standard `prompt` / `target` rows, where `target` is the source article text

## Base config

The starter config points at `oliverkinch/danish_wikipedia`.

I checked the dataset with `datasets`: it has one `train` split and the rows expose `url`, `title`, and `text`.

If you switch to a different source, update `text_field`, `title_field`, and `url_field` in the config.

## Usage

```bash
uv run sdg build sdg/packs/backtranslation/configs/base.yaml
uv run sdg verify <run-id>
uv run sdg publish <run-id>
```
