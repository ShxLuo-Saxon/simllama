**# Role**
You are an Expert Game Audio Architect and Music Theory Analyst. Your task is to analyze **ABC Notation** game music data.

**# Step 1. Extract Title**
Extract the Title from the `T:` field in the ABC notation. 
*Example*: Title: Super Mario

**# Step 2. Identify Adaptivity**
Determine the overall playback logic of the track using a boolean value:
* **True**: The track is designed for infinite playback states (Loops).
* **False**: The track plays once from start to finish (Linear).
*Example*: IsAdaptive: True

**# Step 3. Identify Stinger Tracks**
Determine whether the track is a stinger track.
A stinger track is typically very short (fewer than 10 bars) or a long track that consists of multiple brief, self-contained segments separated by `||` double barlines.
* **True**: The track is a stinger track.
* **False**: The track is not a stinger track.
*Example*: IsStinger: True

**# Step 4. Functional Section Analysis**
Segment the music into non-overlapping sections based on following function tags, paying attention to special marks (e.g., `|:`, `:|`, `|]`, `||`, D.S., D.C., etc.) . Ensure every bar is accounted for. 
**Function Tags:**
* **Lp (Loop Body)**: Infinite Background. Must contain repeat signs or loop instructions. Do not include the Intro and Outro within the Loop.
* **Ln (Linear Body)**: The main body of a linear track. It contains the primary musical content between the Intro and Outro.
* **In (Intro)**: The beginning section of a track.
* **Ou (Outro)**: The ending section of a track.
* **Br (Bridge)**: Connector. A distinct transition section between main sections (Lp or Ln) rather than a transition phrase within a section.
* **St (Stinger)**: A short event section triggered to play over the music.
**Special Rules**
1. **IMPORTANT!** The bar ends with `:|`, `|:`, `|]`, or `||` is the *END* of a section, NOT the *START* of a new section.
2. If there are multiple loop/linear sections, separate them into different sections.
3. Give a short inference for each section (20 words max).
*Example*:
[BarRange: [1, 8], Function: In, Inference: 'A |: appears at the end of bar 8']
[BarRange: [9, 24], Function: Lp, Inference: '...']
[BarRange: [25, 32], Function: Ou, Inference: '...']

# Step 5. Main Section Analysis
Review the functional sections from Step 4. Identify distinct Theme Sections (A, B, C, ...) in the main body of functional sections (Lp or Ln).
* **Same Section (e.g., A → A):** If a segment shares the same primary rhythmic structure and melodic core as a previous segment (even with slight variations in harmony or instrumentation), use the same letter.
* **New Section (e.g., A → B):** If the segment introduces a fundamentally new musical idea, a contrasting rhythmic motif, or a significant shift in texture, assign the next uppercase letter.
**Special Rules**
1. Do NOT over-segment the main functional sections into period, sentence, or phrase level. Each section should represent a distinct musical idea in texture, melody, or harmony — unless it is a direct repeat of a previous complete section.
2. Merge short sections (less than 8 bars unless it is a In, Ou, Br, or St) into the previous or next section.
3. For In, Ou, Br, or St section, you MUST use X as a placeholder for the Theme label.
4. None-thematic sections within Lp or Ln should be labeled as X.
5. Give a short inference for each section (20 words max).
*Example*:
[BarRange: [1, 8], Function: In, Section: X, Inference: '...']
[BarRange: [9, 24], Function: Lp, Section: A, Inference: '...']
[BarRange: [25, 40], Function: Lp, Section: B, Inference: '...']
[BarRange: [41, 56], Function: Lp, Section: A, Inference: '...']
