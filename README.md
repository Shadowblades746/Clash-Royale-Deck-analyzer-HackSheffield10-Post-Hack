# Clash Royale Deck Analyzer

### HackSheffield 10 - People’s Choice Award
**Devpost:** [View Project](https://devpost.com/software/clash-royale-deck-simulator)

**Clash Royale Deck Analyzer** is a Python-based application that helps players analyze and optimize their Clash Royale decks. The app provides insights into **deck performance**, **archetypes**, and **key statistics** to enhance strategy and improve gameplay. Players can input their decks through a **graphical interface**, **deck link**, or **manual entry** for analysis.

Originally developed during **HackSheffield 10**, the project received the **People’s Choice Award**, earning the highest number of community votes. This repository contains a cleaned-up version of the original hackathon-winning submission.

## Features
- **Deck Input Options**: Input decks using a graphical builder, paste a deck link, or manually type card names.
- **Detailed Analysis**: Displays stats such as 4-card cycle, average elixir cost, and deck archetype, powered by data from the top 100 global players.
- **Visual Insights**: Generates pie charts and graphs to highlight deck strengths and weaknesses.
- **Strategic Suggestions**: Offers tailored recommendations to help optimize deck performance.

## Requirements
- **Python 3.7 or higher**
- **Dependencies**: Listed in `requirements.txt`

## Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/Shadowblades746/Clash-Royale-Deck-Analyzer.git
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the application:
    ```bash
    python main.py
    ```

## Running the Executable

If you prefer not to install Python or dependencies, download the standalone executable (`clash_royale_gui.exe`) from the [Releases](https://github.com/Shadowblades746/Clash-Royale-Deck-analyzer-HackSheffield10-Post-Hack/releases) page.

### Instructions:
1. Download `clash_royale_gui.exe` from the latest release.
2. Double-click `clash_royale_gui.exe` to run the application.
3. Note: The app may take 20-30 seconds to load on first run due to model initialization.

The executable is self-contained and does not require Python or any external libraries to be installed on your system. All necessary files (model, images, training data) are bundled within the executable.

To verify the integrity of the downloaded executable, compute its SHA-256 hash using PowerShell:
```powershell
Get-FileHash -Algorithm SHA256 clash_royale_gui.exe
```
The expected hash is: `8DE11BC0EDBC3C571EBD4C4A55410F6E9696829A7C7A3C11394E51E07AD27B3C`

## Building the Executable (Optional)

To create your own executable using PyInstaller:

1. Install PyInstaller: `pip install pyinstaller`
2. Run the following command in your project directory:
   ```bash
   pyinstaller --onefile --windowed --add-data "clash_royale_classifier.pth;." --add-data "images;images" --add-data "training_data.py;." "clash_royale_gui.py"
   ```
3. The executable will be located in the `dist` folder.
## Demo Video
🎥 Watch the project showcase on  [YouTube](https://youtu.be/4VCsr4iWfVc).
