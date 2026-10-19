# Frontend Changes: Dark/Light Theme Toggle

## index.html
- Added a fixed top-right icon button (`#themeToggle`) with sun and moon SVGs. It is a native `<button>`, so Tab/Enter/Space work, and `aria-label` updates with the current theme.
- Added an inline head script that reads `localStorage.theme` and sets `data-theme` on `<html>` before first paint (no flash).
- Bumped cache-busting versions to `?v=10`.

## style.css
- Added `:root[data-theme="light"]` overriding all color variables (light background and surfaces, dark text `#0f172a`/`#475569`, `#cbd5e1` borders, same blue primary). Dark remains the default.
- Added new variables (`--code-bg`, `--error-text`, `--success-text`) and replaced hardcoded colors for code blocks, error/success messages and the welcome shadow so they work in both themes. Error/success text is darker in light mode for contrast.
- Added `.theme-toggle` styles: circular button, hover scale, `:focus-visible` outline, and a rotate/fade crossfade between sun (dark theme) and moon (light theme).
- Added a `.theme-transition` class that applies a 0.3s color/border/shadow transition to all elements, only while toggling, so normal interactions are unaffected.

## script.js
- Added `setupThemeToggle()`: toggles `data-theme` on `<html>` between `dark` and `light`, persists the choice in `localStorage`, and updates the button's `aria-label`.
