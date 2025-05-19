# Text Analysis Agents Project

This project utilizes several Python libraries for text analysis tasks. This README provides instructions on how to set up the necessary environment and run the code.

## Prerequisites
This project utilizes several Python libraries for text analysis tasks. This README provides instructions on how to set up the necessary environment, access the code from the Git repository, and run the project.

* **Git:** Ensure you have Git installed on your system. You can check if Git is installed by running `git --version` in your terminal. If not, you can download and install it from [https://git-scm.com/downloads](https://git-scm.com/downloads).
* **Python 3.7 or higher:** Ensure you have Python 3.7 or a later version installed on your system. You can check your Python version by running `python --version` or `python3 --version` in your terminal.
* **pip:** Python's package installer, pip, should be installed by default with your Python installation. You can check if pip is installed by running `pip --version` in your terminal.

## Accessing the Code from the Git Repository

You can access the project code by cloning the Git repository. Open your terminal or command prompt and navigate to the directory where you want to store the project, then run the following command:

```bash
git clone [https://github.com/Priyanka1423/Assignment_14.git](https://github.com/Priyanka1423/Assignment_14.git)
* **Python 3.7 or higher:** Ensure you have Python 3.7 or a later version installed on your system. You can check your Python version by running `python --version` or `python3 --version` in your terminal.
* **pip:** Python's package installer, pip, should be installed by default with your Python installation. You can check if pip is installed by running `pip --version` in your terminal.

## Setup Instructions

Follow these steps to create a virtual environment and install the required dependencies:

**1.  Create a Virtual Environment:**

    It's highly recommended to create a virtual environment to isolate the project dependencies. This prevents conflicts with other Python projects on your system.

    Open your terminal or command prompt and navigate to the project directory (the directory containing this README file and your `requirements.txt` file). Then, run the following command:
    
    ```bash
    cd Assignment_14
    python -m venv venv
    ```

    * On some older systems, you might need to use `python3 -m venv venv`.

    This command creates a new directory named `venv` (short for virtual environment) that contains a copy of the Python interpreter and related files.

**2.  Activate the Virtual Environment:**

    You need to activate the virtual environment before installing any packages. The activation command depends on your operating system:

    * **On macOS and Linux:**

        ```bash
        source venv/bin/activate
        ```

    * **On Windows:**

        ```bash
        .\\venv\\Scripts\\activate
        ```

    Once the virtual environment is activated, you should see the name of the environment (`(venv)`) at the beginning of your terminal prompt.

**3.  Install Dependencies from `requirements.txt`:**

    The `requirements.txt` file lists all the necessary Python libraries for this project. To install them, navigate to the project directory in your activated virtual environment and run the following command:

    ```bash
    pip install -r requirements.txt
    ```

    This command will read the `requirements.txt` file and install each listed package along with its dependencies. Based on the `requirements.txt` you provided, the following packages will be installed:

    ```
    nltk
    spacy
    gensim
    scikit-learn
    transformers
    langchain
    duckduckgo-search
    beautifulsoup4
    pandas
    seaborn
    matplotlib
    networkx
    torch
    tiktoken
    langchain-community
    wikipedia-api
    langchain-google-genai
    ```

**4.  Run the Code:**

    After successfully installing the dependencies, you can now run your Python code. Based on the filename you provided (`Text_Agents_Teksystem.ipynb`), it appears you have a Jupyter Notebook.

    To run the Jupyter Notebook:

    1.  Ensure your virtual environment is still activated.
    2.  Navigate to the project directory in your terminal.
    3.  Run the following command to start the Jupyter Notebook server:

        ```bash
        jupyter notebook
        ```

    This will open a new tab in your web browser displaying the Jupyter Notebook interface. You can then navigate to and open the `Text_Agents_Teksystem.ipynb` file and run the cells within the notebook.

    Open  "Text_Agent_main.ipynb" file
## Next Steps

Once you have the environment set up and the code running, you can start exploring and using the text analysis agents implemented in this project. Refer to the comments and documentation within the code for more specific instructions on how to use the different functionalities.

## Deactivating the Virtual Environment

When you are finished working on the project, you can deactivate the virtual environment by simply running the following command in your terminal:

```bash
deactivate
