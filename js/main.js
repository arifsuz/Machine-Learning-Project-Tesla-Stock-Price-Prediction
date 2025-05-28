const ctx = document.getElementById("barChart").getContext("2d");
const data = {
  labels: ["Decision Tree", "Naive Bayes", "KNN"],
  datasets: [
    {
      label: "Accuracy (%)",
      data: [61, 52.8, 58.3],
      backgroundColor: ["#576cbc", "#9ba4b5", "#f7c873"],
      borderRadius: 8,
      borderWidth: 1,
    },
    {
      label: "AUC",
      data: [0.6451, 0.6049, 0.6217],
      backgroundColor: ["#394867", "#a5d7e8", "#f9d923"],
      borderRadius: 8,
      borderWidth: 1,
      yAxisID: "y1",
    },
  ],
};
const config = {
  type: "bar",
  data: data,
  options: {
    responsive: true,
    plugins: {
      legend: { position: "top" },
      title: {
        display: false,
      },
      tooltip: {
        callbacks: {
          label: function (context) {
            if (context.dataset.label === "Accuracy (%)") {
              return `Accuracy: ${context.parsed.y}%`;
            } else {
              return `AUC: ${context.parsed.y}`;
            }
          },
        },
      },
    },
    scales: {
      y: {
        beginAtZero: true,
        max: 100,
        title: { display: true, text: "Accuracy (%)" },
      },
      y1: {
        beginAtZero: true,
        max: 1,
        position: "right",
        grid: { drawOnChartArea: false },
        title: { display: true, text: "AUC" },
      },
    },
  },
};
new Chart(ctx, config);
