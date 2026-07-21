MOF Adsorption Prediction Platform
An intelligent materials informatics dashboard built with Streamlit and Python. This tool predicts the equilibrium adsorption capacity (Qe) of Metal-Organic Frameworks (MOFs) and their composites for target adsorbates (Methylene Blue & Methyl Orange) using two frozen ML models (Catboost + Gradient Boosting Regressor) for MB and (ExtraTrees) for MO.
Installation & Setup
This application requires Python 3.8 or higher.

For macOS Users
Click the green "Code" button at the top of this GitHub page and select "Download ZIP".
Double-click the downloaded ZIP file to extract the folder.
Open your Mac's Terminal, type cd  (with a space after it), drag the extracted folder from your Finder and drop it directly into the Terminal window, then hit Enter.
Install the required libraries by typing: pip3 install -r requirements.txt and hit Enter.
Launch the app by typing: streamlit run app.py and hit Enter.

For Windows Users
Click the green "Code" button at the top of this GitHub page and select "Download ZIP".
Right-click the downloaded ZIP file and select "Extract All".
Open Command Prompt, type cd  (with a space after it), drag the extracted folder from your File Explorer and drop it directly into the Command Prompt window, then hit Enter.
Install the required libraries by typing: pip install -r requirements.txt and hit Enter.
Launch the app by typing: streamlit run app.py and hit Enter.

Using the Batch Uploader
When utilizing the High-Throughput Batch Processing feature, ensure your uploaded .csv or .xlsx file contains exactly the following 8 column headers in this exact order:
BET_Surface_area, Pore_volume, Pore_size, Dosage, Temp, Time, Initial_Concentration, pH