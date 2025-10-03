## Prerequisites

A modern web browser (e.g., Chrome, Firefox, Safari, Edge).

Download & Install min Python 3.x.

```sh
https://www.python.org/downloads/
```

`git` installed on your system to clone the repository.

---

## Clone the Repository

Open your terminal or command prompt and run the following command:
CMD:

```sh
git clone https://github.com/DineshNallaiyaCannyfore/data_Analysis.git
```

---

## Getting Started

Follow these simple steps to get the tool up and running on your local machine.

---

## Run the Server

**Step 1:**
Open the project folder in VS Code.

**Step 2:**
Open a terminal (VS Code Terminal or Command Prompt).

**Step 3:**
Navigate to the server directory:
CMD:

```sh
D:\Projects\DataChat\dataserverapp>
```

**Step 4:**
Verify Python is installed:
CMD: `sh  python -V `
Next:
If you see a version (e.g., Python 3.12.x), continue.

If not installed -----> Download & Install min Python 3.x.

```sh
https://www.python.org/downloads/
```

**Step 5:**

#### Install dependencies:

```sh
cd D:\Projects\DataChat\dataserverapp

pip install -r requirement.txt

pip install duckdb
```

**Step 6:**

#### Run the server:

CMD:

```sh
D:\Projects\DataChat\dataserverapp> run.bat
```

**run.bat** else just type **run** only, it start the server

If the server environment is not working properly, follow the steps in the readme file.

```
D:\Projects\DataChat\dataserverapp\readme
```

If everything is fine, the terminal will show the server running URL & port
**Step 7:**

#### Run the Client

Navigate to the client app directory:
CMD:

```sh
D:\Projects\DataChat\dataclientapp\app\html
```

**Step 8:**
Install VS Code extension ----> Search: Live Server Preview.

Or

To view the file, open `index.html` in a web browser or double-click it.

**Step 9:**
Right-click on index.html -----> select Open with Live Server Preview.

**Step 10:**
Upload a file and start using the client.

---

**Step 11:**
ngrok http 8000 --host-header=8000
