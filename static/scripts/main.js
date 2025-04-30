// main.js

let lastInputData = null;
let speedChart = null;

window.onload = function () {
  fetchStoredPredictions();
  updateSpeedChart(0);
  setTimeout(() => {
    showNotification("Welcome to QoS Prediction Tool!", "success");
  }, 500);
};

document.addEventListener("DOMContentLoaded", () => {
  document.getElementById("csvFile").addEventListener("change", handleFileUpload);
});

async function handleFileUpload(event) {
  const fileInput = document.getElementById("csvFile");
  const file = fileInput.files[0];
  if (!file) return;

  document.getElementById("uploadIcon").classList.add("animate-spin");
  document.getElementById("uploadText").innerText = "Processing...";
  document.getElementById("result").innerText = "Analyzing network data...";

  const reader = new FileReader();
  reader.onload = async function (e) {
    const csvData = e.target.result;
    const jsonData = csvToJson(csvData);

    if (jsonData.length === 0) {
      showNotification("Invalid CSV file or format.", "error");
      resetUploadState();
      return;
    }

    lastInputData = jsonData[0];

    try {
      const response = await sendToFastAPI(jsonData[0]);
      const predictionValue = parseFloat(response.prediction_mbps).toFixed(2);
      document.getElementById("selectedFileName").innerText = file.name;
      document.getElementById("result").innerHTML = `<span class="text-primary-700 font-bold text-3xl">${predictionValue}</span> <span class="text-gray-600 text-xl">Mbit/s</span>`;

      document.getElementById("explainBtn").classList.remove("hidden");
      document.getElementById("explainBtn").classList.add("flex");

      showNotification("Prediction completed successfully!", "success");
      updateSpeedChart(predictionValue);
      fetchStoredPredictions();
    } catch (error) {
      showNotification("Failed to get prediction.", "error");
    }

    resetUploadState();
  };

  reader.readAsText(file);
}

function resetUploadState() {
  document.getElementById("uploadIcon").classList.remove("animate-spin");
  document.getElementById("uploadText").innerText = "Choose File";
}

function showNotification(message, type) {
  const notification = document.getElementById("notification");
  notification.innerText = message;
  notification.className = "fixed top-4 right-4 py-2 px-4 rounded-lg shadow-lg transition-opacity duration-500";

  if (type === "error") {
    notification.classList.add("bg-red-500", "text-white");
  } else if (type === "success") {
    notification.classList.add("bg-green-500", "text-white");
  }

  notification.classList.remove("opacity-0");
  setTimeout(() => {
    notification.classList.add("opacity-0");
  }, 3000);
}

function csvToJson(csv) {
  const lines = csv.split("\n").map(line => line.trim()).filter(line => line.length > 0);
  const headers = lines[0].split(",");
  const jsonData = [];

  for (let i = 1; i < lines.length; i++) {
    const values = lines[i].split(",");
    if (values.length !== headers.length) continue;
    let obj = {};
    for (let j = 0; j < headers.length; j++) {
      obj[headers[j].trim()] = isNaN(values[j]) ? values[j].trim() : parseFloat(values[j]);
    }
    jsonData.push(obj);
  }
  return jsonData;
}

async function sendToFastAPI(data) {
  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data)
    });
    if (!response.ok) throw new Error(`Server error: ${response.statusText}`);
    return await response.json();
  } catch (error) {
    console.error("Error:", error);
    showNotification("Failed to get prediction.", "error");
    return { prediction_mbps: "Error" };
  }
}

function handleExplain() {
  if (!lastInputData) {
    showNotification("No input data available for explanation.", "error");
    return;
  }
  localStorage.setItem("lastInputData", JSON.stringify(lastInputData));
  const popup = window.open("/explain-page", "ExplainPopup", "width=800,height=700,scrollbars=yes,resizable=yes");
  if (!popup || popup.closed || typeof popup.closed === "undefined") {
    showNotification("Popup blocked! Please allow popups for this website.", "error");
  }
}

async function fetchStoredPredictions() {
  try {
    const response = await fetch("/predictions");
    if (!response.ok) throw new Error("Failed to fetch stored predictions.");
    const data = await response.json();
    updatePredictionsTable(data);
  } catch (error) {
    console.error("Error fetching predictions:", error);
  }
}

function updatePredictionsTable(predictions) {
  const tableBody = document.getElementById("predictions-table-body");
  tableBody.innerHTML = "";
  if (predictions.length === 0) {
    const row = document.createElement("tr");
    row.innerHTML = `<td colspan="2" class="border px-4 py-6 text-center text-gray-500">No predictions available yet</td>`;
    tableBody.appendChild(row);
    return;
  }
  predictions.slice(-10).reverse().forEach(prediction => {
    const row = document.createElement("tr");
    const predValue = parseFloat(prediction.prediction_mbps).toFixed(2);
    let speedClass = '';
    if (predValue > 40) speedClass = 'bg-green-50';
    else if (predValue > 20) speedClass = 'bg-yellow-50';
    else speedClass = 'bg-red-50';

    row.className = `${speedClass} hover:bg-gray-100 transition-colors duration-150`;
    row.innerHTML = `
      <td class="border-b border-gray-200 px-4 py-3">${new Date(prediction.timestamp * 1000).toLocaleString()}</td>
      <td class="border-b border-gray-200 px-4 py-3 font-medium">${predValue} Mbit/s</td>
    `;
    tableBody.appendChild(row);
  });
}

function updateSpeedChart(speed) {
  const ctx = document.getElementById("speedChart").getContext("2d");
  if (speedChart) speedChart.destroy();

  const poorSpeed = 10, fairSpeed = 30, goodSpeed = 50, maxSpeed = 100;
  const percentage = Math.min(100, (speed / maxSpeed) * 100);
  let color = '#EF4444';
  if (speed > fairSpeed) color = '#10B981';
  else if (speed > poorSpeed) color = '#F59E0B';

  speedChart = new Chart(ctx, {
    type: 'doughnut',
    data: {
      datasets: [{
        data: [percentage, 100 - percentage],
        backgroundColor: [color, '#E5E7EB'],
        borderWidth: 0,
        circumference: 180,
        rotation: 270
      }]
    },
    options: {
      responsive: true,
      cutout: '70%',
      plugins: {
        tooltip: { enabled: false },
        legend: { display: false }
      },
      animation: {
        animateRotate: true,
        animateScale: true
      }
    }
  });

  document.getElementById("speedLabel").textContent = "Network Speed";
  const speedClass = document.getElementById("speedClass");
  if (speed > fairSpeed) {
    speedClass.textContent = "GOOD";
    speedClass.className = "text-green-600 font-bold text-sm";
  } else if (speed > poorSpeed) {
    speedClass.textContent = "FAIR";
    speedClass.className = "text-yellow-600 font-bold text-sm";
  } else {
    speedClass.textContent = "POOR";
    speedClass.className = "text-red-600 font-bold text-sm";
  }
}
