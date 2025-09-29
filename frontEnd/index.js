let tableStructure = [];
var selectedFile = null;
var resultSqlQuery = null;
var resultBinaryData = null;
let resultQuestions = [];
let lastQueryDetails = { context: "", question: "" };
let chartInstance = null;
let selectedLLMModel = "gemini-2.5-flash";
let uploadedFile = null;
document
  .getElementById("barChartBtn")
  .addEventListener("click", () => fetchChart("bar"));
document
  .getElementById("lineChartBtn")
  .addEventListener("click", () => fetchChart("line"));
document
  .getElementById("pieChartBtn")
  .addEventListener("click", () => fetchChart("pie"));

async function uploadFile(event) {
    let fileNames = [];
    const formData = new FormData();
    for (let i = 0; i < event.target.files.length; i++) {
        const file = event.target.files[i];
        fileNames.push(file.name)
        formData.append("files", file, file.name);
    }
    const outputDiv = document.getElementById("selectedFile");
    outputDiv.innerHTML = fileNames.join(", ");;
    uploadedFiles = formData;
}

async function submitQuestion() {
  console.log(uploadedFile)
  showLoader();
   try {
    const res = await fetch(
      "http://localhost:8000/api/getData",
      {
        method: "POST",
        body: uploadedFiles
      }
    );

    const result = await res.json();
    if (res.ok) {
      console.log(result)
      const keys = Object.keys(resultQuestions)
      if(!keys.includes('error'))
      {
        resultQuestions = result?.generated_answer;
        displayQuestions(resultQuestions);
      }
      else{
        alert(resultQuestions?.error)
      }
    } else {
      console.error("Upload failed:", result.detail || "Error");
    }
  } catch (error) {
    console.error("Error  uploading file:", error);
  } finally{
      hideLoader();
  }


}
function displayQuestions(resultQuestions) {
  const outputDiv = document.getElementById("answereContainer");
  const displayOutput  =  document.getElementById("resultSection");
   displayOutput.style.display = "block";
  outputDiv.innerHTML = "";
    const p = document.createElement("p");
    p.textContent = resultQuestions;
    outputDiv.appendChild(p);

}

function showLoader() {

  const loader = document.getElementById("loader");
    console.log("loader wordes",loader)
  loader.style.display = "flex";
}

function hideLoader() {
  const loader = document.getElementById("loader");
  loader.style.display = "none";
}

function downloadQueryResult() {
  if (!resultBinaryData) {
    alert("No data available to download.");
    return;
  }
  const byteCharacters = atob(resultBinaryData);
  const byteNumbers = Array.from(byteCharacters, (char) => char.charCodeAt(0));
  const byteArray = new Uint8Array(byteNumbers);
  const blob = new Blob([byteArray], { type: "text/csv" });
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "query_results.csv";
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  window.URL.revokeObjectURL(url);
}

// function uploadFile(event) {
//   document.getElementById("divQuestions").style.display = "block";
//   document.getElementById("resultSection").style.display = "none";
//   document.getElementById("result-table").innerHTML = "";
//   document.getElementById("chartImage").style.display = "none";
//   document.getElementById("chartIcons").style.display = "none";

//   tableStructure = [];
//   selectedFile = null;
//   resultSqlQuery = null;
//   resultBinaryData = null;
//   resultQuestions = [];
//   lastQueryDetails = { context: "", question: "" };

//   document.getElementById("tableOutput").innerHTML = "";
//   document.getElementById("result-table").innerHTML = "";
//   document.getElementById("result-query").innerHTML = "";
//   document.getElementById("resultSection").style.display = "none";
//   document.getElementById("divQuestions").innerHTML = "";

//   if (chartInstance) {
//     chartInstance.destroy();
//     chartInstance = null;
//   }

//   for (let i = 0; i < event.target.files.length; i++) {
//     const file = event.target.files[i];
//     selectedFile = file;
//     const fileNameExtacter = file.name.split(".");
//     const fileExt = fileNameExtacter[fileNameExtacter.length - 1].toLowerCase();

//     switch (fileExt) {
//       case "db":
//         extractDBfile(file);
//         break;
//       case "xlsx":
//       case "csv":
//       case "xls":
//         extractXlsxFile(file);
//         break;
//       default:
//         alert("Unsupported file type.");
//     }
//   }
// }

function llmModelChange() {
  selectedLLMModel = document.getElementById("llm-model").value;
  if (!selectedLLMModel) {
    selectedLLMModel = "gemini-2.5-flash";
  }
}
