# SpendSnap Student Edition

SpendSnap turns a receipt photo into transaction and line-item data, then gives
students a simple view of where their everyday spending goes.

## What the MVP does

- Upload and preview JPG, JPEG, PNG, or HEIC (iPhone) receipts.
- Send the image to a configurable Google Gemini API model for multimodal
  extraction of merchant, date, currency, reported total, tax, tip, and items.
- Validate the returned JSON, review and edit details, and flag mismatches
  between the known line-item sum and the reported total.
- Categorize items with straightforward Python keyword rules.
- Show known receipt totals, category spending, transaction details, and a CSV
  download.
- Estimate a hypothetical monthly amount using a purchase and a frequency you
  choose (4.33 weeks per month). The estimate is not presented as actual
  recurring spending or guaranteed savings.

Saved receipts are stored on your computer in `data/receipts.json` (ignored by
git, so they are never pushed); this MVP has no database or account system. The image is sent to Google for AI processing when
you choose **Analyze receipt**. Avoid uploading sensitive receipts unless you
are comfortable sharing them with the configured provider.

## Model support and Gemma 4 status

The default model is `gemma-4-26b-a4b-it`. The configured Google API key
successfully looked up this model and processed a synthetic receipt image with
the Gemini `generateContent` API. This Gemma model exposes `generateContent`,
so the app routes Gemma requests through that API; other configured Gemini
models use the Interactions API.

The model is configurable through `GEMINI_MODEL`. Before entering a Gemma model
name, verify that the exact identifier is available for your Google AI Studio
API key, accepts image input, and supports the request format used here; then
set it in `.env`. Do not assume a Gemma-branded model identifier is supported
by the Gemini API.

The API integration uses Google's maintained
[Google Gen AI Python SDK](https://ai.google.dev/gemini-api/docs/libraries).

## Run locally

Requires Python 3.10 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Add your Gemini API key to `.env`:

```dotenv
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemma-4-26b-a4b-it
```

Then launch:

```bash
streamlit run app.py
```

Get a key from [Google AI Studio](https://aistudio.google.com/apikey). Keep it
in `.env`; `.gitignore` excludes the file. Never paste keys into the app or
commit them.

## Try a sample receipt

1. Open the app and select **Scan a receipt**.
2. Choose a clear JPG, JPEG, PNG, or HEIC receipt image and inspect the preview.
3. Read the privacy note, then select **Analyze receipt**.
4. Review the extracted fields and line items. Correct names, quantity, prices,
   categories, and receipt total as needed.
5. Compare the line-item warning with the printed receipt. The app never
   overwrites the reported total to make the numbers match.
6. Select **Save transaction** and view the dashboard. Saved receipts stay
   on the dashboard after refreshing or restarting the app.

If you do not yet have an API key, the dashboard and local tests still work,
but live receipt analysis requires a valid key and model access.

## Test

```bash
python -m unittest
```

The tests cover response validation, unknown values, categorization, and
line-item arithmetic. Live Gemini calls require a key and are not run by the
unit tests.

## License

This project is provided under the MIT License. A license file does not imply
eligibility for any particular event prize; check the full hackathon rules.
