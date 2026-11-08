# Streamlit UI/UX Improvements

**Date:** 2026-11-08

## Design

- Forced **light** theme (`.streamlit/config.toml`) + high-contrast CSS in `app.py`
- Consistent brand green, Manrope/Source Sans typography
- Navigation as clear primary/secondary **buttons** (stable on Streamlit 1.67)

## Analyser

- Simpler French intro; formats JPG/JPEG/PNG/WebP; max 8 Mo stated
- Upload tip card; example image; preview + size; Analyse / Nouvelle image
- Results in tabs; friendly errors; FAQ; privacy note
- Welcome dialog first visit only

## Jouer

- Replaced freehand drawing with **Google-style 3×3 tile selection**
- Removed countdown entirely
- Quit anytime + thank-you message after contribution / exit
- Clear labels: your tiles vs AI vs ground truth

## Revue

- Fail-closed auth messaging
- Button filters (no radio/selectbox for status)
- Confirmation checkboxes before review decision and export
- Eligibility shown separately from approval

## Remaining UX limits

- Tile grid is 3×3 only (coarser than freehand masks)
- No Playwright visual regression suite
- Mobile layout verified by CSS rules, not device lab
