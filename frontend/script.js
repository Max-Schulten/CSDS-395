const resumeInput = document.getElementById("resumeInput");
const uploadBtn = document.getElementById("uploadBtn");
const fileNameDisplay = document.getElementById("fileName");
const matchBtn = document.getElementById("matchBtn");
const jobResults = document.getElementById("jobResults");

// Open hidden file picker when custom button is clicked
uploadBtn.addEventListener("click", () => {
    resumeInput.click();
});

// Validate and display file name
resumeInput.addEventListener("change", () => {
    const file = resumeInput.files[0];

    const allowedTypes = [
        "application/pdf",
        "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ];

    if (file && !allowedTypes.includes(file.type)) {
        alert("Please upload a PDF or Word document.");
        resumeInput.value = "";
        fileNameDisplay.textContent = "";
        return;
    }

    if (file) {
        fileNameDisplay.textContent = "Uploaded: " + file.name;
    }
});

// Simulated job matching
matchBtn.addEventListener("click", () => {
    if (!resumeInput.files.length) {
        alert("Please upload your resume first.");
        return;
    }

    jobResults.innerHTML = `
        <div class="job-card">
            <h3>Data Analyst</h3>
            <p><strong>Company:</strong> TechCorp</p>
            <p><strong>Location:</strong> Remote</p>
        </div>

        <div class="job-card">
            <h3>Machine Learning Engineer</h3>
            <p><strong>Company:</strong> AI Solutions</p>
            <p><strong>Location:</strong> San Francisco, CA</p>
        </div>

        <div class="job-card">
            <h3>Cybersecurity Analyst</h3>
            <p><strong>Company:</strong> SecureNet</p>
            <p><strong>Location:</strong> New York, NY</p>
        </div>
    `;
});
