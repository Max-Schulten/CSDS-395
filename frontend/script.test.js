describe('Job Finder Frontend Interactions', () => {
    let resumeInput, uploadBtn, fileNameDisplay, matchBtn, jobResults, jobDescInput;

    // We include the scoring logic here so we can test the math and filtering independently
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

    // Set up the DOM before each test
    beforeEach(() => {
        document.body.innerHTML = `
            <input type="file" id="resumeInput" accept=".pdf,.doc,.docx,application/msword,application/vnd.openxmlformats-officedocument.wordprocessingml.document" hidden>
            <button id="uploadBtn">Choose Resume</button>
            <p id="fileName"></p>
            <textarea id="jobDescInput"></textarea>
            <button id="matchBtn">Analyze Resume & Find Matches</button>
            <div id="jobResults"><p class="placeholder">Upload your resume to see recommended jobs and match scores.</p></div>
        `;

        // Grab elements
        resumeInput = document.getElementById("resumeInput");
        uploadBtn = document.getElementById("uploadBtn");
        fileNameDisplay = document.getElementById("fileName");
        matchBtn = document.getElementById("matchBtn");
        jobResults = document.getElementById("jobResults");
        jobDescInput = document.getElementById("jobDescInput");

        // Mock window.alert
        window.alert = jest.fn();

        // Re-apply the event listeners from your script.js (mocked slightly to bypass pdf.js)
        uploadBtn.addEventListener("click", () => {
            resumeInput.click();
        });

        resumeInput.addEventListener("change", () => {
            const file = resumeInput.files[0];
            const allowedTypes = [
                "application/pdf",
                "application/msword",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            ];

            if (file && !allowedTypes.includes(file.type)) {
                window.alert("Please upload a PDF or Word document.");
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
                window.alert("Please upload your resume first.");
                return;
            }

            const jobDescText = jobDescInput.value.trim();
            let htmlOutput = "";

            // Mock branch 1: Optional job description is present
            if (jobDescText) {
                // Mocking the plain text extraction from pdf.js for the test environment
                const plainText = "python data analysis project management";
                const matchScore = calculateMatchScore(plainText, jobDescText);
                htmlOutput += `<div class="score-number">${matchScore}%</div>`;
            }

            // Mock branch 2: Recommended jobs are always present
            htmlOutput += '<div class="job-card"><h3>Item Planner</h3></div>';
            jobResults.innerHTML = htmlOutput;
        });
    });

    afterEach(() => {
        jest.clearAllMocks();
    });

    // --- TEST CASES ---

    test('Clicking "Choose Resume" should trigger the hidden file input', () => {
        const clickSpy = jest.spyOn(resumeInput, 'click');
        uploadBtn.click();
        expect(clickSpy).toHaveBeenCalled();
    });

    test('Uploading a valid PDF should display the file name', () => {
        const file = new File(['dummy content'], 'resume.pdf', { type: 'application/pdf' });
        Object.defineProperty(resumeInput, 'files', { value: [file] });
        resumeInput.dispatchEvent(new Event('change'));

        expect(fileNameDisplay.textContent).toBe('Uploaded: resume.pdf');
        expect(window.alert).not.toHaveBeenCalled();
    });

    test('Uploading an invalid file type should trigger an alert and clear input', () => {
        const file = new File(['dummy content'], 'image.png', { type: 'image/png' });
        Object.defineProperty(resumeInput, 'files', { value: [file] });
        resumeInput.dispatchEvent(new Event('change'));

        expect(window.alert).toHaveBeenCalledWith('Please upload a PDF or Word document.');
        expect(resumeInput.value).toBe('');
    });

    test('Clicking "Analyze" without a file should trigger an alert', () => {
        Object.defineProperty(resumeInput, 'files', { value: [] });
        matchBtn.click();
        expect(window.alert).toHaveBeenCalledWith('Please upload your resume first.');
    });

    test('Clicking "Analyze" WITH a file but NO job description should display only job cards', () => {
        const file = new File(['dummy content'], 'resume.pdf', { type: 'application/pdf' });
        Object.defineProperty(resumeInput, 'files', { value: [file] });
        jobDescInput.value = ""; // Leave job description empty
        
        matchBtn.click();

        expect(window.alert).not.toHaveBeenCalled();
        expect(jobResults.innerHTML).toContain('Item Planner'); // Jobs appear
        expect(jobResults.innerHTML).not.toContain('score-number'); // Score stays hidden
    });

    test('Clicking "Analyze" WITH a file AND a job description should display both score and job cards', () => {
        const file = new File(['dummy content'], 'resume.pdf', { type: 'application/pdf' });
        Object.defineProperty(resumeInput, 'files', { value: [file] });
        jobDescInput.value = "looking for someone with python and data analysis skills"; 
        
        matchBtn.click();

        expect(window.alert).not.toHaveBeenCalled();
        expect(jobResults.innerHTML).toContain('Item Planner'); // Jobs appear
        expect(jobResults.innerHTML).toContain('score-number'); // Score appears
    });

    test('calculateMatchScore function should correctly calculate keyword overlaps ignoring stop words', () => {
        const resumeText = "I have experience with Python, Swift, and data analysis.";
        // The algorithm filters out "for" and "with", leaving 5 keywords: "looking", "someone", "python", "data", "analysis"
        const jobDescText = "Looking for someone with Python and data analysis"; 
        
        const score = calculateMatchScore(resumeText, jobDescText);
        
        // Overlap with resume text is 3 words: "python", "data", "analysis"
        // 3 out of 5 keywords = 60%
        expect(score).toBe(60);
    });
});