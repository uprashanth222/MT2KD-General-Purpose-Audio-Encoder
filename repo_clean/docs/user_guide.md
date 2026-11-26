# SvaraMitra – user guide

**What it does.** Upload or record one audio clip, then: transcribe English speech (ASR), name the sounds in it (AT), or check whether two recordings are the same speaker (SV).

**Open the app.** Use the link shared by the team (`https://….trycloudflare.com`) on a phone or computer, or run it locally (`src/app/run_windows.bat` → http://localhost:8000). The badge shows **Live** when the model is running; **Demo** shows labelled sample results only.

**Steps**
1. *Audio*: drop a file (WAV, MP3, FLAC, OGG; up to 50 MB / 10 min) or press **Record with your microphone**. The clip can be played back and is reused for all three tasks.
2. *ASR*: choose **Transcribe** → Run. Output: transcript with the language model, the plain model output underneath, Copy / Download .txt.
3. *AT*: choose **Tag sounds** → Run. Output: the most likely sound classes with scores (e.g. a guitar clip: Music, Guitar, Plucked string instrument).
4. *SV*: choose **Verify speaker**, then either add a reference recording (upload or record) or pick a registered speaker → Run. Output: cosine similarity, the threshold (0.428) and Same / Different speaker.

**Expected inputs.** Clear English speech works best for ASR (the model learned from read audiobooks); any sound works for AT; 5–15 s of speech per recording works best for SV. Clips under 1 s give unreliable results.

**Privacy.** Audio is processed in memory on the server and is deleted when the server restarts; nothing is saved.
