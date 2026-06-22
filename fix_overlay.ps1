$path = "app.py"
$text = (Get-Content $path -Raw) -replace "`r`n", "`n"
$ok = $true

$anchor1 = @"
        .stButton > button:hover {
            background-color: #25496F;
            color: #FFFFFF;
        }
"@ -replace "`r`n", "`n"
$replace1 = @"
        .stButton > button:hover {
            background-color: #25496F;
            color: #FFFFFF;
        }

        /* Full-screen processing overlay: shown while a Groq call runs so the
           user sees the app is busy. The synchronous call blocks interaction
           anyway; this makes that visible. */
        .ugegbe-overlay {
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(251, 250, 247, 0.82);
            backdrop-filter: blur(1.5px);
            z-index: 9999;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .ugegbe-overlay-box {
            background: #FFFFFF;
            border: 1px solid var(--border);
            border-left: 5px solid var(--accent);
            border-radius: 6px;
            padding: 1.1rem 1.6rem;
            font-family: 'Source Sans 3', sans-serif;
            font-size: 1rem;
            color: var(--text);
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        }
        .ugegbe-overlay-box strong { color: var(--accent); }
"@ -replace "`r`n", "`n"
if ($text.Contains($anchor1)) { $text = $text.Replace($anchor1, $replace1); Write-Host "Edit 1 OK (CSS)" } else { Write-Host "Edit 1 FAIL"; $ok = $false }

$anchor2 = @"
def skeleton_pause(seconds: float = 0.5) -> None:
    """A short, deliberate pause so results resolve smoothly rather than snapping in."""
    time.sleep(seconds)
"@ -replace "`r`n", "`n"
$replace2 = @"
def skeleton_pause(seconds: float = 0.5) -> None:
    """A short, deliberate pause so results resolve smoothly rather than snapping in."""
    time.sleep(seconds)


def show_processing_overlay(placeholder, message: str) -> None:
    """Paint a full-screen 'please wait' overlay into the given placeholder."""
    placeholder.markdown(
        f'<div class="ugegbe-overlay"><div class="ugegbe-overlay-box">'
        f'<strong>Working.</strong> {message}</div></div>',
        unsafe_allow_html=True,
    )
"@ -replace "`r`n", "`n"
if ($text.Contains($anchor2)) { $text = $text.Replace($anchor2, $replace2); Write-Host "Edit 2 OK (helper)" } else { Write-Host "Edit 2 FAIL"; $ok = $false }

$anchor3 = @"
        classify_system = get_classifier()
        with st.spinner("Reading the Act and classifying the system..."):
"@ -replace "`r`n", "`n"
$replace3 = @"
        classify_system = get_classifier()
        overlay = st.empty()
        show_processing_overlay(overlay, "Reading the Act and classifying the system. This can take a moment.")
        with st.spinner("Reading the Act and classifying the system..."):
"@ -replace "`r`n", "`n"
if ($text.Contains($anchor3)) { $text = $text.Replace($anchor3, $replace3); Write-Host "Edit 3 OK (classify)" } else { Write-Host "Edit 3 FAIL"; $ok = $false }

$anchor4 = @"
        generate_report = get_report_generator()
        with st.spinner("Mapping obligations and retrieving source passages..."):
"@ -replace "`r`n", "`n"
$replace4 = @"
        generate_report = get_report_generator()
        overlay = st.empty()
        show_processing_overlay(overlay, "Mapping obligations and retrieving source passages. This can take a moment.")
        with st.spinner("Mapping obligations and retrieving source passages..."):
"@ -replace "`r`n", "`n"
if ($text.Contains($anchor4)) { $text = $text.Replace($anchor4, $replace4); Write-Host "Edit 4 OK (obligations)" } else { Write-Host "Edit 4 FAIL"; $ok = $false }

if ($ok) {
    Set-Content -Path $path -Value $text -NoNewline
    Write-Host "ALL EDITS APPLIED."
} else {
    Write-Host "ONE OR MORE FAILED. Nothing written."
}
