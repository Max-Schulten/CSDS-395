const resumeInput = document.getElementById("resumeInput");
const uploadBtn = document.getElementById("uploadBtn");
const fileNameDisplay = document.getElementById("fileName");
const matchBtn = document.getElementById("matchBtn");
const jobResults = document.getElementById("jobResults");
const jobDescInput = document.getElementById("jobDescInput");
const jobDescSection = document.getElementById("jobDescSection");
const findJobsToggle = document.getElementById("findJobsToggle");
const checkMatchToggle = document.getElementById("checkMatchToggle");

let mode = "find-jobs"; // "find-jobs" | "check-match"

// --- Toggle ---
findJobsToggle.addEventListener("click", () => setMode("find-jobs"));
checkMatchToggle.addEventListener("click", () => setMode("check-match"));

function setMode(newMode) {
    mode = newMode;
    if (mode === "find-jobs") {
        findJobsToggle.classList.add("active");
        checkMatchToggle.classList.remove("active");
        jobDescSection.style.display = "none";
        matchBtn.textContent = "Find Matching Jobs";
    } else {
        checkMatchToggle.classList.add("active");
        findJobsToggle.classList.remove("active");
        jobDescSection.style.display = "block";
        matchBtn.textContent = "Check My Match";
    }
    jobResults.innerHTML = '<p class="placeholder">Upload your resume to see recommended jobs and match scores.</p>';
}

// --- File upload ---
uploadBtn.addEventListener("click", () => resumeInput.click());

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
    if (file) fileNameDisplay.textContent = "Uploaded: " + file.name;
});

// --- Main action ---
matchBtn.addEventListener("click", async () => {
    if (!resumeInput.files.length) {
        alert("Please upload your resume first.");
        return;
    }

    jobResults.innerHTML = '<div class="spinner"></div><p class="spinner-label">Analyzing your resume…</p>';

    try {
        const resumeText = await parseFileToText(resumeInput.files[0]);

        if (mode === "check-match") {
            const jobDescText = jobDescInput.value.trim();
            if (!jobDescText) {
                alert("Please paste a job description to check your match.");
                jobResults.innerHTML = '<p class="placeholder">Upload your resume to see recommended jobs and match scores.</p>';
                return;
            }
            const data = await postJSON("/score-job", { resume_text: resumeText, job_text: jobDescText });
            renderCheckMatch(data);
        } else {
            const data = await postJSON("/find-jobs", { resume_text: resumeText });
            renderFindJobs(data);
        }
        jobResults.scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (err) {
        console.error(err);
        jobResults.innerHTML = `<p style="color:red;">Error: ${err.message}</p>`;
    }
});

// --- Fetch helper ---
async function postJSON(url, body) {
    const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body)
    });
    if (!res.ok) throw new Error(`Server error: ${res.status}`);
    return res.json();
}

// --- Renderers ---
function renderCheckMatch(data) {
    const score = data.scores[0];
    const skillCov = data.skill_coverages[0];
    const semScore = data.semantic_scores[0];
    const eduCov = data.education_coverages[0];
    const coveredSkills = data.covered_skills[0] || [];
    const unmatchedSkills = data.unmatched_skills[0] || [];

    const pct = Math.round(score * 100);
    const skillPct = Math.round(skillCov * 100);
    const semPct = Math.round(semScore * 100);
    const eduPct = Math.round(eduCov * 100);

    const matchMsg = pct >= 70
        ? "Strong match — your resume aligns well with this role."
        : pct >= 45
        ? "Moderate match — consider highlighting more relevant skills."
        : "Low match — your resume may need significant tailoring for this role.";

    const skillsHtml = buildSkillTags(coveredSkills, unmatchedSkills);

    jobResults.innerHTML = renderResumeStats(data.resume) + `
        <div class="score-container">
            <div class="score-number">${pct}%</div>
            <div class="score-label">Overall Match Score</div>
        </div>
        <p style="text-align:center; color:#4b5563; font-size:0.95rem; margin-bottom:20px;">${matchMsg}</p>
        <div class="sub-scores">
            <div class="sub-score-item">
                <span class="sub-score-label">Skill Coverage</span>
                <span class="sub-score-value">${skillPct}%</span>
            </div>
            <div class="sub-score-item">
                <span class="sub-score-label">Semantic Alignment</span>
                <span class="sub-score-value">${semPct}%</span>
            </div>
            <div class="sub-score-item">
                <span class="sub-score-label">Education Match</span>
                <span class="sub-score-value">${eduPct}%</span>
            </div>
        </div>
        <div class="covered-skills-section">
            <h4>Job Skills</h4>
            <div class="skill-tags">${skillsHtml}</div>
        </div>
    `;
}

function renderFindJobs(data) {
    if (!data.jobs || !data.jobs.length) {
        jobResults.innerHTML = '<p class="placeholder">No matching jobs found.</p>';
        return;
    }

    const indices = data.scores
        .map((s, i) => ({ s, i }))
        .sort((a, b) => b.s - a.s)
        .map(x => x.i);

    const cards = indices.map(i => {
        const job = data.jobs[i];
        const pct = Math.round(data.scores[i] * 100);
        const coveredSkills = (data.covered_skills[i] || []).slice(0, 5);
        const unmatchedSkills = (data.unmatched_skills[i] || []).slice(0, 3);
        const skillsHtml = buildSkillTags(coveredSkills, unmatchedSkills);
        const snippet = job.job_desc ? job.job_desc.slice(0, 150) + "…" : "";

        return `
            <div class="job-card">
                <div class="job-card-header">
                    <span class="job-score">${pct}% Match</span>
                </div>
                <p class="job-snippet">${snippet}</p>
                ${skillsHtml ? `<div class="skill-tags" style="margin-top:8px;">${skillsHtml}</div>` : ""}
            </div>
        `;
    }).join("");

    jobResults.innerHTML = renderResumeStats(data.resume) + `
        <h3 style="margin-bottom:15px; text-align:left; color:#1f2937;">Recommended Opportunities</h3>
        ${cards}
    `;
}

// --- Resume stats panel ---
function renderResumeStats(resume) {
    const skills = Object.keys(resume.skills || {});
    const education = resume.education || {};
    const categories = resume.categories || [];

    const skillsHtml = skills.length
        ? skills.map(s => `<span class="skill-tag">${s}</span>`).join("")
        : "<em style='color:#9ca3af;'>None detected</em>";

    const degree = education.degree
        ? education.degree.charAt(0).toUpperCase() + education.degree.slice(1)
        : null;
    const majors = (education.majors || []).join(", ");
    const eduLine = [degree, majors].filter(Boolean).join(" — ") || "Not detected";

    const categoriesHtml = categories.length
        ? categories.map(c => `<span class="category-tag">${c}</span>`).join("")
        : "<em style='color:#9ca3af;'>Unknown</em>";

    return `
        <div class="resume-stats">
            <h3 class="resume-stats-title">Your Resume at a Glance</h3>
            <div class="resume-stats-grid">
                <div class="resume-stat-block">
                    <span class="stat-label">Categories</span>
                    <div class="skill-tags" style="margin-top:6px;">${categoriesHtml}</div>
                </div>
                <div class="resume-stat-block">
                    <span class="stat-label">Education</span>
                    <p class="stat-value">${eduLine}</p>
                </div>
                <div class="resume-stat-block resume-stat-block--full">
                    <details class="skills-details">
                        <summary>Skills Detected (${skills.length})</summary>
                        <div class="skill-tags">${skillsHtml}</div>
                    </details>
                </div>
            </div>
        </div>
        <hr class="results-divider">
    `;
}

// --- Skill tag builder ---
function buildSkillTags(covered, unmatched) {
    if (!covered.length && !unmatched.length) return "<em style='color:#9ca3af;'>None detected</em>";
    const coveredHtml = covered.map(s => `<span class="skill-tag">${s}</span>`).join("");
    const unmatchedHtml = unmatched.map(s => `<span class="skill-tag skill-tag--missing">${s}</span>`).join("");
    return coveredHtml + unmatchedHtml;
}

// --- File parsing ---
async function parseFileToText(file) {
    const arrayBuffer = await file.arrayBuffer();

    if (file.type === "application/pdf") {
        const pdf = await pdfjsLib.getDocument(arrayBuffer).promise;
        let fullText = "";
        for (let i = 1; i <= pdf.numPages; i++) {
            const page = await pdf.getPage(i);
            const textContent = await page.getTextContent();
            fullText += textContent.items.map(item => item.str).join(" ") + "\n";
        }
        return fullText;
    }

    if (file.type === "application/vnd.openxmlformats-officedocument.wordprocessingml.document") {
        const result = await mammoth.extractRawText({ arrayBuffer });
        return result.value;
    }

    throw new Error("Unsupported file type.");
}
