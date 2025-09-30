let uploadedFile = null;
async function uploadFile(event) {
  let fileNames = [];
  const formData = new FormData();
  for (let i = 0; i < event.target.files.length; i++) {
    const file = event.target.files[i];
    fileNames.push(file.name);
    formData.append("files", file, file.name);
  }
  const outputDiv = document.getElementById("selectedFile");
  outputDiv.innerHTML = fileNames.join(", ");
  uploadedFiles = formData;
}

async function submitQuestion() {
  console.log(uploadedFile);
  showLoader();
  try {
    const res = await fetch("http://localhost:8000/api/getData", {
      method: "POST",
      body: uploadedFiles,
    });

    const result = await res.json();
    if (res.ok) {
 
      const keys = Object.keys(result);
      console.log(result);
      if (!keys.includes("error")) {
        resultQuestions = result?.generated_answer;
        displayQuestions(resultQuestions);
      } else {
        alert(result?.error);
      }
    } else {
      console.error("Upload failed:", result.detail || "Error");
    }
  } catch (error) {
    console.error("Error  uploading file:", error);
  } finally {
    hideLoader();
  }
}
function displayQuestions(resultQuestions) {
  const outputDiv = document.getElementById("answereContainer");
  const displayOutput = document.getElementById("resultSection");
  displayOutput.style.display = "block";
  outputDiv.innerHTML = "";
  const p = document.createElement("p");
  p.textContent = resultQuestions;
  outputDiv.appendChild(p);
}

function showLoader() {
  const loader = document.getElementById("loader");
  console.log("loader wordes", loader);
  loader.style.display = "flex";
}

function hideLoader() {
  const loader = document.getElementById("loader");
  loader.style.display = "none";
}
