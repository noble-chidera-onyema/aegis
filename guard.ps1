$path = "app.py"
$text = (Get-Content $path -Raw) -replace "`r`n", "`n"
$ok = $true

# Edit 1: classify guard condition (line 672)
$a1 = "    if classify_clicked:`n"
$r1 = "    if classify_clicked and not st.session_state.get(`"classify_running`", False):`n"
if ($text.Contains($a1)) { $text = $text.Replace($a1, $r1); Write-Host "Edit 1 OK" } else { Write-Host "Edit 1 FAIL"; $ok = $false }

# Edit 2: set classify flag right before get_classifier (line 679)
$a2 = "        classify_system = get_classifier()`n"
$r2 = "        st.session_state.classify_running = True`n        classify_system = get_classifier()`n"
if ($text.Contains($a2)) { $text = $text.Replace($a2, $r2); Write-Host "Edit 2 OK" } else { Write-Host "Edit 2 FAIL"; $ok = $false }

# Edit 3: obligations guard + set flag (line 824-ish)
$a3 = "    if st.button(`"View obligations`"):`n        generate_report = get_report_generator()`n"
$r3 = "    if st.button(`"View obligations`") and not st.session_state.get(`"obligations_running`", False):`n        st.session_state.obligations_running = True`n        generate_report = get_report_generator()`n"
if ($text.Contains($a3)) { $text = $text.Replace($a3, $r3); Write-Host "Edit 3 OK" } else { Write-Host "Edit 3 FAIL"; $ok = $false }

# Edit 4: clear classify flag after its success (line 696)
$a4 = "            st.success(`"Classification complete. Open the Classification tab to see the result.`")`n"
$r4 = "            st.success(`"Classification complete. Open the Classification tab to see the result.`")`n        st.session_state.classify_running = False`n"
if ($text.Contains($a4)) { $text = $text.Replace($a4, $r4); Write-Host "Edit 4 OK" } else { Write-Host "Edit 4 FAIL"; $ok = $false }

# Edit 5: clear obligations flag after its success (line 833)
$a5 = "            st.success(`"Obligations report ready. Open the Obligations tab.`")`n"
$r5 = "            st.success(`"Obligations report ready. Open the Obligations tab.`")`n        st.session_state.obligations_running = False`n"
if ($text.Contains($a5)) { $text = $text.Replace($a5, $r5); Write-Host "Edit 5 OK" } else { Write-Host "Edit 5 FAIL"; $ok = $false }

if ($ok) {
    Set-Content -Path $path -Value $text -NoNewline
    Write-Host "ALL GUARDS APPLIED."
} else {
    Write-Host "ONE OR MORE FAILED. Nothing written."
}
