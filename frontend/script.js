const resumeInput = document.getElementById("resumeInput");
const uploadBtn = document.getElementById("uploadBtn");
const fileNameDisplay = document.getElementById("fileName");
const matchBtn = document.getElementById("matchBtn");
const jobResults = document.getElementById("jobResults");
const jobDescInput = document.getElementById("jobDescInput");

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

    const jobDescText = jobDescInput.value.trim();
    const file = resumeInput.files[0];
    
    // Show loading state
    jobResults.innerHTML = '<p class="placeholder">Analyzing resume and finding matches...</p>';

    try {
        // Extract plain text from the file
        const plainText = await parseFileToText(file);
        console.log("Parsed Text:", plainText);

        let htmlOutput = "";

        // 1. If a job description was pasted, compute and add the score
        if (jobDescText) {
            const matchScore = calculateMatchScore(plainText, jobDescText);
            htmlOutput += `
                <div class="score-container">
                    <div class="score-number">${matchScore}%</div>
                    <div class="score-label">Keyword Match for Target Role</div>
                </div>
                <p style="text-align: center; color: #4b5563; font-size: 0.95rem; margin-bottom: 25px;">
                    ${matchScore >= 70 ? 'Great match! Your resume aligns well with this role.' : 'Consider adding more keywords from the job description to your resume.'}
                </p>
                <hr style="border: 0; border-top: 1px solid #e5e7eb; margin-bottom: 25px;">
            `;
        }

        // 2. Always add the recommended job cards
        htmlOutput += `
            <h3 style="margin-bottom: 15px; text-align: left; color: #1f2937;">Recommended Opportunities</h3>
            
            <div class="job-card">
                <h3 style="color: #1d4ed8;">Item Planner</h3>
                <p><strong>Company:</strong> RetailCorp Logistics</p>
                <p><strong>Location:</strong> Hybrid</p>
            </div>

            <div class="job-card">
                <h3 style="color: #1d4ed8;">Appeals Panel Member</h3>
                <p><strong>Company:</strong> FDIC</p>
                <p><strong>Location:</strong> Remote</p>
            </div>

            <div class="job-card">
                <h3 style="color: #1d4ed8;">Data Analyst</h3>
                <p><strong>Company:</strong> Tech Solutions</p>
                <p><strong>Location:</strong> San Francisco, CA</p>
            </div>
        `;

        // Render everything to the screen
        jobResults.innerHTML = htmlOutput;

    } catch (error) {
        console.error(error);
        jobResults.innerHTML = `<p class="error" style="color: red;">Error processing file: ${error.message}</p>`;
    }
});

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

function calculateMatchScore(resumeText, jobDescText) {
    const stopWords = new Set(["the", "and", "a", "to", "of", "in", "for", "is", "on", "that", "by", "this", "with", "i", "you", "it", "not", "or", "be", "are", "from", "at", "as", "your", "all", "have", "new", "more", "an", "was", "we", "will", "can", "us", "about", "if", "my", "has", "but", "our", "one", "other", "do", "no", "they", "he", "up", "may", "what", "which", "their", "out", "use", "any", "there", "see", "only", "so", "his", "when", "who", "also", "now", "get"]);

    const tokenize = (text) => {
        return text
            .toLowerCase()
            .replace(/[^a-z0-9\s]/g, '')
            .split(/\s+/)
            .filter(word => word.length > 2 && !stopWords.has(word));
    };

    const jobKeywords = new Set(tokenize(jobDescText));
    const resumeWords = new Set(tokenize(resumeText));

    if (jobKeywords.size === 0) return 0;

    let matchCount = 0;
    
    jobKeywords.forEach(word => {
        if (resumeWords.has(word)) {
            matchCount++;
        }
    });

    return Math.round((matchCount / jobKeywords.size) * 100);
}