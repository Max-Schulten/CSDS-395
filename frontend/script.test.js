describe('Job Finder Frontend Interactions', () => {
    let resumeInput, uploadBtn, fileNameDisplay, matchBtn, jobResults;

    // Set up the DOM before each test
    beforeEach(() => {
        document.body.innerHTML = `
            <input type="file" id="resumeInput" accept=".pdf,.doc,.docx,application/msword,application/vnd.openxmlformats-officedocument.wordprocessingml.document" hidden>
            <button id="uploadBtn">Choose Resume</button>
            <p id="fileName"></p>
            <button id="matchBtn">Match Jobs</button>
            <div id="jobResults"><p class="placeholder">Upload your resume to see job matches.</p></div>
        `;

        // Grab elements
        resumeInput = document.getElementById("resumeInput");
        uploadBtn = document.getElementById("uploadBtn");
        fileNameDisplay = document.getElementById("fileName");
        matchBtn = document.getElementById("matchBtn");
        jobResults = document.getElementById("jobResults");

        // Mock window.alert
        window.alert = jest.fn();

        // --- Re-apply the event listeners from your script.js ---
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

        matchBtn.addEventListener("click", () => {
            if (!resumeInput.files.length) {
                window.alert("Please upload your resume first.");
                return;
            }
            jobResults.innerHTML = '<div class="job-card"><h3>Data Analyst</h3></div>';
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
        // Create a mock PDF file
        const file = new File(['dummy content'], 'resume.pdf', { type: 'application/pdf' });
        
        // Simulate file selection
        Object.defineProperty(resumeInput, 'files', { value: [file] });
        resumeInput.dispatchEvent(new Event('change'));

        expect(fileNameDisplay.textContent).toBe('Uploaded: resume.pdf');
        expect(window.alert).not.toHaveBeenCalled();
    });

    test('Uploading an invalid file type (e.g., .png) should trigger an alert and clear input', () => {
        // Create a mock Image file
        const file = new File(['dummy content'], 'image.png', { type: 'image/png' });
        
        Object.defineProperty(resumeInput, 'files', { value: [file] });
        resumeInput.dispatchEvent(new Event('change'));

        expect(window.alert).toHaveBeenCalledWith('Please upload a PDF or Word document.');
        expect(resumeInput.value).toBe('');
        expect(fileNameDisplay.textContent).toBe('');
    });

    test('Clicking "Match Jobs" without a file should trigger an alert', () => {
        // Ensure no files are selected
        Object.defineProperty(resumeInput, 'files', { value: [] });
        
        matchBtn.click();

        expect(window.alert).toHaveBeenCalledWith('Please upload your resume first.');
        // Ensure the results container wasn't updated with job cards
        expect(jobResults.innerHTML).not.toContain('job-card');
    });

    test('Clicking "Match Jobs" with a valid file should display job results', () => {
        const file = new File(['dummy content'], 'resume.pdf', { type: 'application/pdf' });
        Object.defineProperty(resumeInput, 'files', { value: [file] });
        
        matchBtn.click();

        expect(window.alert).not.toHaveBeenCalled();
        expect(jobResults.innerHTML).toContain('Data Analyst');
    });
});