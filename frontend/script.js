const resumeInput = document.getElementById("resumeInput");
const uploadBtn = document.getElementById("uploadBtn");
const fileNameDisplay = document.getElementById("fileName");
const matchBtn = document.getElementById("matchBtn");
const jobResults = document.getElementById("jobResults");

// Open hidden file picker
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

matchBtn.addEventListener("click", async () => {
    if (!resumeInput.files.length) {
        alert("Please upload your resume first.");
        return;
    }

    const file = resumeInput.files[0];
    
    // Show loading state
    jobResults.innerHTML = '<p class="placeholder">Parsing resume...</p>';

    try {
        // Extract plain text from the file
        const plainText = await parseFileToText(file);
        
        console.log("Parsed Text:", plainText);

        const response = await fetch("/match", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ resume_text: plainText })
        });

        const results = await response.json();
        console.log(results)
        // Future: Connect to backend
        // await sendToPythonBackend(plainText);

        // Display mock results (simulating a success)
        displayMockResults(plainText);

    } catch (error) {
        console.error(error);
        jobResults.innerHTML = `<p class="error">Error parsing file: ${error.message}</p>`;
    }
});

/**
 * Helper: Parses PDF or DOCX file to plain text
 */
async function parseFileToText(file) {
    const arrayBuffer = await file.arrayBuffer();

    // PDF
    if (file.type === "application/pdf") {
        const pdf = await pdfjsLib.getDocument(arrayBuffer).promise;
        let fullText = "";
        
        // Loop through all pages
        for (let i = 1; i <= pdf.numPages; i++) {
            const page = await pdf.getPage(i);
            const textContent = await page.getTextContent();
            const pageText = textContent.items.map(item => item.str).join(" ");
            fullText += pageText + "\n";
        }
        return fullText;
    } 
    
    // CASE 2: DOCX
    else if (file.type === "application/vnd.openxmlformats-officedocument.wordprocessingml.document") {
        const result = await mammoth.extractRawText({ arrayBuffer: arrayBuffer });
        return result.value;
    }

    throw new Error("Unsupported file type for client-side parsing.");
}

function displayMockResults(textSnippet) {
    // Just showing a snippet of the parsed text to prove it worked
    const snippet = textSnippet.substring(0, 100) + "...";
    
    jobResults.innerHTML = `
        <div style="padding: 15px; background: #e0f2fe; border-radius: 8px; margin-bottom: 20px;">
            <strong>Success! Parsed Content Preview:</strong><br>
            <em>"${snippet}"</em>
        </div>

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
    `;
}