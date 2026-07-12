"use strict";

(function () {
    // ==========================================================================
    // DOM Element References Cache
    // ==========================================================================
    const DOM = {
        // Global / Navigation
        healthStatus: document.getElementById("health-status"),

        // Composer Form
        agentForm: document.getElementById("agent-form"),
        queryInput: document.getElementById("query-input"),
        charCounter: document.getElementById("char-counter"),
        dropZone: document.getElementById("drop-zone"),
        fileInput: document.getElementById("file-input"),
        filesContainer: document.getElementById("files-container"),
        fileList: document.getElementById("file-list"),
        btnClearAll: document.getElementById("btn-clear-all"),
        btnSubmit: document.getElementById("btn-submit"),
        submitSpinner: document.getElementById("submit-spinner"),

        // Progress Overlay
        progressCard: document.getElementById("progress-card"),
        progressMessage: document.getElementById("progress-message"),

        // Clarification Widget
        clarificationCard: document.getElementById("clarification-card"),
        clarificationQuestionText: document.getElementById("clarification-question-text"),
        clarificationForm: document.getElementById("clarification-form"),
        clarificationAnswerInput: document.getElementById("clarification-answer-input"),
        btnContinueClarify: document.getElementById("btn-continue-clarify"),
        continueSpinner: document.getElementById("continue-spinner"),
        btnCancelClarify: document.getElementById("btn-cancel-clarify"),

        // Final Answer Panel
        answerCard: document.getElementById("answer-card"),
        answerEmpty: document.getElementById("answer-empty"),
        answerContent: document.getElementById("answer-content"),
        answerStatusBadge: document.getElementById("answer-status-badge"),
        btnCopyAnswer: document.getElementById("btn-copy-answer"),
        answerText: document.getElementById("answer-text"),
        answerRequestId: document.getElementById("answer-request-id"),

        // Scenario Cards
        scenariosGrid: document.getElementById("scenarios-grid"),
        scenarioCards: document.querySelectorAll(".scenario-card"),

        // Alerts
        alertsPanel: document.getElementById("alerts-panel"),
        warningSection: document.getElementById("warning-section"),
        warningList: document.getElementById("warning-list"),
        errorSection: document.getElementById("error-section"),
        errorList: document.getElementById("error-list"),

        // Inspector Overview Metrics
        metricsCard: document.getElementById("metrics-card"),
        metricTotal: document.getElementById("metric-total"),
        metricExecuted: document.getElementById("metric-executed"),
        metricSuccess: document.getElementById("metric-success"),
        metricFailed: document.getElementById("metric-failed"),
        metricSkipped: document.getElementById("metric-skipped"),

        // Inspector Panels
        extractedInputsCard: document.getElementById("extracted-inputs-card"),
        inputsEmpty: document.getElementById("inputs-empty"),
        inputsContainer: document.getElementById("inputs-container"),

        executionPlanCard: document.getElementById("execution-plan-card"),
        planEmpty: document.getElementById("plan-empty"),
        planDetails: document.getElementById("plan-details"),
        planGoalText: document.getElementById("plan-goal-text"),
        planConstraints: document.getElementById("plan-constraints"),
        planConstraintsList: document.getElementById("plan-constraints-list"),
        planTimelineList: document.getElementById("plan-timeline-list"),

        executionTraceCard: document.getElementById("execution-trace-card"),
        traceEmpty: document.getElementById("trace-empty"),
        planTraceContainer: document.getElementById("plan-trace-container"),
        planTraceList: document.getElementById("plan-trace-list"),
        execTraceContainer: document.getElementById("exec-trace-container"),
        execTraceList: document.getElementById("exec-trace-list"),
        tabPlanTrace: document.getElementById("tab-plan-trace"),
        tabExecTrace: document.getElementById("tab-exec-trace"),

        // Parent Tabs
        technicalTabsCard: document.getElementById("technical-tabs-card"),
        tabPlan: document.getElementById("tab-plan"),
        tabInputs: document.getElementById("tab-inputs"),
        tabTrace: document.getElementById("tab-trace"),

        // Inspector State Containers
        inspectorIdleState: document.getElementById("inspector-idle-state"),
        inspectorStatusBadge: document.getElementById("inspector-status-badge"),
        statusBar: document.getElementById("status-bar"),
        primaryResultArea: document.getElementById("primary-result-area"),
        btnNewRequest: document.getElementById("btn-new-request")
    };

    // ==========================================================================
    // Application State
    // ==========================================================================
    const STATE = {
        selectedFiles: [],
        originalQuery: "",
        isProcessing: false,
        clarificationRequired: false,
        progressIntervalId: null,
        activeTab: "plan-trace" // "plan-trace" or "exec-trace"
    };

    // Mandatory Assessment Scenarios mapping
    const SCENARIOS = {
        "1": {
            query: "Summarize this audio.",
            hint: "Please select an audio file (.mp3, .wav, or .m4a)"
        },
        "2": {
            query: "Extract only the action items from this PDF.",
            hint: "Please select a PDF document (.pdf)"
        },
        "3": {
            query: "Explain this code. Identify the programming language, explain what it does, identify bugs or issues, and provide the time complexity.",
            hint: "Please select an image of code (.png, .jpg, or .jpeg)"
        },
        "4": {
            query: "Summarize the YouTube video linked in this PDF.",
            hint: "Please select a PDF document (.pdf) containing a YouTube link"
        },
        "5": {
            query: "Compare these two inputs. Identify their similarities and differences.",
            hint: "Please select both a PDF document (.pdf) and an audio file (.mp3, .wav, or .m4a)"
        }
    };

    const PROGRESS_STEPS = [
        "Securing uploads...",
        "Extracting multimodal content...",
        "Building normalized context...",
        "Planning tool execution...",
        "Executing constrained tools...",
        "Composing final response..."
    ];

    // ==========================================================================
    // Helper Utilities
    // ==========================================================================
    function formatBytes(bytes) {
        if (bytes === 0) return "0 Bytes";
        const k = 1024;
        const sizes = ["Bytes", "KB", "MB", "GB"];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
    }

    function getFileCategory(filename) {
        const ext = filename.split(".").pop().toLowerCase();
        if (ext === "pdf") return "PDF";
        if (["png", "jpg", "jpeg"].includes(ext)) return "Image";
        if (["mp3", "wav", "m4a"].includes(ext)) return "Audio";
        return "File";
    }

    function getFileIcon(category) {
        if (category === "PDF") return "📄";
        if (category === "Image") return "🖼️";
        if (category === "Audio") return "🔊";
        return "📁";
    }

    function setUIState(state) {
        if (state === "idle") {
            DOM.inspectorIdleState.classList.remove("hidden");
            DOM.progressCard.classList.add("hidden");
            DOM.statusBar.classList.add("hidden");
            DOM.alertsPanel.classList.add("hidden");
            DOM.answerCard.classList.add("hidden");
            DOM.technicalTabsCard.classList.add("hidden");
            DOM.clarificationCard.classList.add("hidden");
            DOM.primaryResultArea.classList.add("centered");
            DOM.inspectorStatusBadge.textContent = "IDLE";
            DOM.inspectorStatusBadge.className = "status-badge idle";
        } else if (state === "running") {
            DOM.inspectorIdleState.classList.add("hidden");
            DOM.progressCard.classList.remove("hidden");
            DOM.statusBar.classList.add("hidden");
            DOM.alertsPanel.classList.add("hidden");
            DOM.answerCard.classList.add("hidden");
            DOM.technicalTabsCard.classList.add("hidden");
            DOM.clarificationCard.classList.add("hidden");
            DOM.primaryResultArea.classList.add("centered");
            DOM.inspectorStatusBadge.textContent = "RUNNING";
            DOM.inspectorStatusBadge.className = "status-badge running";
        } else if (state === "results") {
            DOM.inspectorIdleState.classList.add("hidden");
            DOM.progressCard.classList.add("hidden");
            DOM.statusBar.classList.remove("hidden");
            DOM.technicalTabsCard.classList.remove("hidden");
            DOM.primaryResultArea.classList.remove("centered");
        }
    }

    function switchParentTab(tabName) {
        DOM.tabPlan.classList.remove("active");
        DOM.tabInputs.classList.remove("active");
        DOM.tabTrace.classList.remove("active");
        DOM.tabPlan.setAttribute("aria-selected", "false");
        DOM.tabInputs.setAttribute("aria-selected", "false");
        DOM.tabTrace.setAttribute("aria-selected", "false");

        DOM.executionPlanCard.classList.add("hidden");
        DOM.extractedInputsCard.classList.add("hidden");
        DOM.executionTraceCard.classList.add("hidden");

        if (tabName === "plan") {
            DOM.tabPlan.classList.add("active");
            DOM.tabPlan.setAttribute("aria-selected", "true");
            DOM.executionPlanCard.classList.remove("hidden");
        } else if (tabName === "inputs") {
            DOM.tabInputs.classList.add("active");
            DOM.tabInputs.setAttribute("aria-selected", "true");
            DOM.extractedInputsCard.classList.remove("hidden");
        } else if (tabName === "trace") {
            DOM.tabTrace.classList.add("active");
            DOM.tabTrace.setAttribute("aria-selected", "true");
            DOM.executionTraceCard.classList.remove("hidden");
        }
    }

    // ==========================================================================
    // Initialization & Health Checks
    // ==========================================================================
    async function checkAPIHealth() {
        try {
            const response = await fetch("/health");
            if (response.ok) {
                const data = await response.json();
                if (data.status === "healthy") {
                    updateHealthIndicator("online", "API Online");
                    return;
                }
            }
            updateHealthIndicator("unavailable", "API Healthy? No");
        } catch (error) {
            console.error("Health check failed:", error);
            updateHealthIndicator("unavailable", "Connection Failed");
        }
    }

    function updateHealthIndicator(status, message) {
        const dot = DOM.healthStatus.querySelector(".status-dot");
        const text = DOM.healthStatus.querySelector(".status-text");

        dot.className = "status-dot";
        dot.classList.add(`status-${status}`);
        text.textContent = message;
    }

    // ==========================================================================
    // Composer & File Attachment State Handlers
    // ==========================================================================
    function handleFileSelection(files) {
        if (!files || files.length === 0) return;

        const maxFilesLimit = 5;
        const totalPendingFiles = STATE.selectedFiles.length + files.length;

        if (totalPendingFiles > maxFilesLimit) {
            alert(`You can upload a maximum of ${maxFilesLimit} files per request.`);
            return;
        }

        Array.from(files).forEach(file => {
            // Avoid duplicate filenames in current select list
            const alreadySelected = STATE.selectedFiles.some(f => f.name === file.name && f.size === file.size);
            if (!alreadySelected) {
                STATE.selectedFiles.push(file);
            }
        });

        renderSelectedFiles();
    }

    function removeFile(index) {
        STATE.selectedFiles.splice(index, 1);
        renderSelectedFiles();
    }

    function clearAllFiles() {
        STATE.selectedFiles = [];
        renderSelectedFiles();
    }

    function renderSelectedFiles() {
        if (STATE.selectedFiles.length === 0) {
            DOM.filesContainer.classList.add("hidden");
            DOM.fileList.innerHTML = "";
            return;
        }

        DOM.filesContainer.classList.remove("hidden");
        DOM.fileList.innerHTML = "";

        STATE.selectedFiles.forEach((file, index) => {
            const category = getFileCategory(file.name);
            const icon = getFileIcon(category);

            const li = document.createElement("li");
            li.className = "file-item";

            const infoGroup = document.createElement("div");
            infoGroup.className = "file-info-group";

            const iconSpan = document.createElement("span");
            iconSpan.className = "file-type-icon";
            iconSpan.textContent = icon;

            const detailsSpan = document.createElement("span");
            detailsSpan.className = "file-details";

            const nameSpan = document.createElement("span");
            nameSpan.className = "file-name";
            nameSpan.textContent = file.name;

            const sizeSpan = document.createElement("span");
            sizeSpan.className = "file-size";
            sizeSpan.textContent = ` (${formatBytes(file.size)})`;

            detailsSpan.appendChild(nameSpan);
            detailsSpan.appendChild(sizeSpan);
            infoGroup.appendChild(iconSpan);
            infoGroup.appendChild(detailsSpan);

            const btnRemove = document.createElement("button");
            btnRemove.type = "button";
            btnRemove.className = "btn-remove-file";
            btnRemove.innerHTML = "&times;";
            btnRemove.setAttribute("aria-label", `Remove ${file.name}`);
            btnRemove.addEventListener("click", () => removeFile(index));

            li.appendChild(infoGroup);
            li.appendChild(btnRemove);
            DOM.fileList.appendChild(li);
        });
    }

    // ==========================================================================
    // UI Visual Locking & Progress Timers
    // ==========================================================================
    function setProcessingState(processing, statusMessage = "Processing...") {
        STATE.isProcessing = processing;

        if (processing) {
            // Lock form inputs
            DOM.queryInput.disabled = true;
            DOM.fileInput.disabled = true;
            DOM.btnSubmit.disabled = true;
            DOM.btnClearAll.disabled = true;
            DOM.btnCancelClarify.disabled = true;
            DOM.btnContinueClarify.disabled = true;
            DOM.clarificationAnswerInput.disabled = true;

            DOM.submitSpinner.classList.remove("hidden");
            DOM.continueSpinner.classList.remove("hidden");

            // Scenario controls locking
            DOM.scenariosGrid.classList.add("disabled-element");
            DOM.dropZone.classList.add("disabled-element");

            // Display loading overlay
            setUIState("running");
            DOM.progressMessage.textContent = statusMessage;
            startProgressSequence();
        } else {
            // Unlock form inputs
            DOM.queryInput.disabled = false;
            DOM.fileInput.disabled = false;
            DOM.btnSubmit.disabled = false;
            DOM.btnClearAll.disabled = false;
            DOM.btnCancelClarify.disabled = false;
            DOM.btnContinueClarify.disabled = false;
            DOM.clarificationAnswerInput.disabled = false;

            DOM.submitSpinner.classList.add("hidden");
            DOM.continueSpinner.classList.add("hidden");

            DOM.scenariosGrid.classList.remove("disabled-element");
            DOM.dropZone.classList.remove("disabled-element");

            // Hide loading overlay
            DOM.progressCard.classList.add("hidden");
            stopProgressSequence();
        }
    }

    function startProgressSequence() {
        stopProgressSequence();
        let stepIdx = 0;
        DOM.progressMessage.textContent = PROGRESS_STEPS[stepIdx];

        STATE.progressIntervalId = setInterval(() => {
            stepIdx = (stepIdx + 1) % PROGRESS_STEPS.length;
            DOM.progressMessage.textContent = PROGRESS_STEPS[stepIdx];
        }, 2500);
    }

    function stopProgressSequence() {
        if (STATE.progressIntervalId) {
            clearInterval(STATE.progressIntervalId);
            STATE.progressIntervalId = null;
        }
    }

    // ==========================================================================
    // API Request Submission & Orchestration
    // ==========================================================================
    async function submitAgentRun(clarificationAnswer = null) {
        if (STATE.isProcessing) return;

        const query = (clarificationAnswer !== null) ? STATE.originalQuery : DOM.queryInput.value.trim();

        if (!query) {
            alert("Prompt query cannot be blank.");
            return;
        }

        if (query.length > 10000) {
            alert("Query exceeds character limit of 10,000.");
            return;
        }

        STATE.originalQuery = query;
        clearDashboardState();
        setProcessingState(true, clarificationAnswer !== null ? "Resubmitting plan context..." : "Securing uploads...");

        const formData = new FormData();
        formData.append("query", query);

        STATE.selectedFiles.forEach(file => {
            formData.append("files", file);
        });

        if (clarificationAnswer !== null) {
            formData.append("clarification_answer", clarificationAnswer);
        }

        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 120000); // 120 seconds timeout

        try {
            const response = await fetch("/agent/run", {
                method: "POST",
                body: formData,
                signal: controller.signal
            });


            if (!response.ok) {
                let errorDetails = "Server returned an error status.";
                try {
                    const errData = await response.json();
                    if (errData && errData.error && errData.error.message) {
                        errorDetails = errData.error.message;
                    }
                } catch (_) { }
                throw new Error(`HTTP ${response.status}: ${errorDetails}`);
            }

            const responseData = await response.json();
            renderResponse(responseData);

        } catch (error) {
            console.error("Agent execution failed:", error);
            renderFailure(error.name === "AbortError" ? "Agent execution timed out (limit: 120s)." : error.message);
        } finally {
            clearTimeout(timeoutId);
            setProcessingState(false);
        }
    }

    // ==========================================================================
    // Response Rendering Engine (Safe textContent & DOM nodes manipulation)
    // ==========================================================================
    function clearDashboardState() {
        setUIState("idle");

        // Reset panels
        DOM.answerContent.classList.add("hidden");
        DOM.answerEmpty.classList.add("hidden");
        DOM.answerText.textContent = "";
        DOM.answerRequestId.textContent = "";

        DOM.alertsPanel.classList.add("hidden");
        DOM.warningSection.classList.add("hidden");
        DOM.warningList.innerHTML = "";
        DOM.errorSection.classList.add("hidden");
        DOM.errorList.innerHTML = "";

        DOM.metricTotal.textContent = "0";
        DOM.metricExecuted.textContent = "0";
        DOM.metricSuccess.textContent = "0";
        DOM.metricFailed.textContent = "0";
        DOM.metricSkipped.textContent = "0";

        DOM.inputsEmpty.classList.remove("hidden");
        DOM.inputsContainer.classList.add("hidden");
        DOM.inputsContainer.innerHTML = "";

        DOM.planEmpty.classList.remove("hidden");
        DOM.planDetails.classList.add("hidden");
        DOM.planGoalText.textContent = "";
        DOM.planConstraints.classList.add("hidden");
        DOM.planConstraintsList.innerHTML = "";
        DOM.planTimelineList.innerHTML = "";

        DOM.traceEmpty.classList.remove("hidden");
        DOM.planTraceContainer.classList.add("hidden");
        DOM.planTraceList.innerHTML = "";
        DOM.execTraceContainer.classList.add("hidden");
        DOM.execTraceList.innerHTML = "";

        // Reset parent tab
        switchParentTab("plan");
    }

    function renderResponse(res) {
        clearDashboardState();
        STATE.lastResponse = res;

        const status = res.status || "failed";
        const requestId = res.request_id || "N/A";

        // Update inspector status badge in compact status bar
        DOM.inspectorStatusBadge.textContent = status.replace("_", " ").toUpperCase();
        DOM.inspectorStatusBadge.className = `status-badge ${status}`;

        // 1. Clarification Required Check
        if (status === "clarification_required") {
            STATE.clarificationRequired = true;
            DOM.clarificationQuestionText.textContent = res.clarification_question || "Clarification requested, but no question provided.";
            DOM.clarificationAnswerInput.value = "";
            DOM.clarificationCard.classList.remove("hidden");
            DOM.agentForm.classList.add("hidden");

            setUIState("results");
            DOM.answerCard.classList.add("hidden");

            // Fill partial technical details
            renderMetadata(res.metadata);
            renderExtractedInputs(res.extracted_inputs);
            renderExecutionPlan(res.plan, res.plan_trace);
            renderTraceTimeline(res.trace, res.plan_trace);
            renderWarningsAndErrors(res.warnings, res.errors);
            return;
        }

        // Standard Completed/Failed rendering
        STATE.clarificationRequired = false;
        DOM.clarificationCard.classList.add("hidden");
        DOM.agentForm.classList.remove("hidden");

        setUIState("results");

        // 2. Render Final Answer Card
        let answerTextVal = res.final_answer || res.answer || "";
        if (status === "completed" && !answerTextVal) {
            answerTextVal = "Execution completed, but no final answer was returned.";
        }

        DOM.answerStatusBadge.className = `status-badge ${status}`;
        DOM.answerStatusBadge.textContent = status;
        DOM.answerRequestId.textContent = requestId;

        if (status === "completed" || status === "partial" || answerTextVal) {
            DOM.answerCard.classList.remove("hidden");
            DOM.answerContent.classList.remove("hidden");
            DOM.answerText.textContent = answerTextVal || "Execution succeeded partially, but encountered errors. Check logs.";
        } else {
            DOM.answerCard.classList.add("hidden");
            DOM.answerContent.classList.add("hidden");
        }

        // Render Inspector components
        renderMetadata(res.metadata);
        renderExtractedInputs(res.extracted_inputs);
        renderExecutionPlan(res.plan, res.plan_trace);
        renderTraceTimeline(res.trace, res.plan_trace);
        renderWarningsAndErrors(res.warnings, res.errors);
    }

    function renderFailure(errorMessage) {
        clearDashboardState();

        setUIState("results");

        // Update status bar badge
        DOM.inspectorStatusBadge.textContent = "FAILED";
        DOM.inspectorStatusBadge.className = "status-badge failed";

        // Show Failure details in main workspace answer card
        DOM.answerCard.classList.remove("hidden");
        DOM.answerStatusBadge.className = "status-badge failed";
        DOM.answerStatusBadge.textContent = "failed";
        DOM.answerText.textContent = "Failed to run agent. Network request failed or returned invalid response.";
        DOM.answerRequestId.textContent = "N/A";

        // Show detailed error log in alerts section
        DOM.alertsPanel.classList.remove("hidden");
        DOM.errorSection.classList.remove("hidden");

        const li = document.createElement("li");
        li.className = "error-text";

        const codeSpan = document.createElement("span");
        codeSpan.className = "alert-list-error-code";
        codeSpan.textContent = "NETWORK_ERROR";

        const textSpan = document.createElement("span");
        textSpan.textContent = errorMessage;

        li.appendChild(codeSpan);
        li.appendChild(textSpan);
        DOM.errorList.appendChild(li);
    }

    function renderMetadata(meta) {
        if (!meta) return;

        DOM.metricsCard.classList.remove("hidden");
        DOM.metricTotal.textContent = meta.total_plan_steps ?? 0;
        DOM.metricExecuted.textContent = meta.executed_steps ?? 0;
        DOM.metricSuccess.textContent = meta.successful_steps ?? 0;
        DOM.metricFailed.textContent = meta.failed_steps ?? 0;
        DOM.metricSkipped.textContent = meta.skipped_steps ?? 0;
    }

    function renderExtractedInputs(inputs) {
        if (!inputs || inputs.length === 0) {
            DOM.inputsEmpty.classList.remove("hidden");
            DOM.inputsContainer.classList.add("hidden");
            return;
        }

        DOM.inputsEmpty.classList.add("hidden");
        DOM.inputsContainer.classList.remove("hidden");
        DOM.inputsContainer.innerHTML = "";

        inputs.forEach((input, index) => {
            const card = document.createElement("div");
            card.className = "collapsible-card";

            const trigger = document.createElement("button");
            trigger.type = "button";
            trigger.className = "collapsible-trigger";
            trigger.setAttribute("aria-expanded", "false");

            const leftGroup = document.createElement("div");
            leftGroup.className = "trigger-left";

            const iconSpan = document.createElement("span");
            iconSpan.textContent = getFileIcon(getFileCategory(input.filename));

            const titleSpan = document.createElement("span");
            titleSpan.className = "trigger-title";
            titleSpan.textContent = input.filename;

            const tagSpan = document.createElement("span");
            tagSpan.className = "trigger-tag";
            tagSpan.textContent = input.input_type || "text";

            leftGroup.appendChild(iconSpan);
            leftGroup.appendChild(titleSpan);
            leftGroup.appendChild(tagSpan);

            // Simple SVG chevron indicator
            const chevron = document.createElement("span");
            chevron.className = "chevron-icon";
            chevron.innerHTML = `
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <polyline points="6 9 12 15 18 9"/>
                </svg>
            `;

            trigger.appendChild(leftGroup);
            trigger.appendChild(chevron);
            card.appendChild(trigger);

            const contentDiv = document.createElement("div");
            contentDiv.className = "collapsible-content";

            const table = document.createElement("table");
            table.className = "content-table";
            const tbody = document.createElement("tbody");

            // Build meta metadata table defensively
            const metaMap = {
                "Source ID": input.source_id,
                "MIME Type": input.metadata?.mime_type,
                "File Size": input.metadata?.size_bytes ? formatBytes(input.metadata.size_bytes) : null,
                "Page Count": input.metadata?.page_count,
                "Extraction Mode": input.metadata?.extraction_method,
                "OCR Confidence":
                    input.metadata?.ocr_confidence !== null &&
                        input.metadata?.ocr_confidence !== undefined
                        ? `${Number(input.metadata.ocr_confidence).toFixed(1)}%`
                        : null,
                "Resolution": (input.metadata?.width && input.metadata?.height) ? `${input.metadata.width} x ${input.metadata.height} px` : null,
                "Duration": input.metadata?.duration_seconds ? `${input.metadata.duration_seconds.toFixed(1)}s` : null,
                "Detected Language": input.metadata?.language
            };

            for (const [key, value] of Object.entries(metaMap)) {
                if (value !== null && value !== undefined) {
                    const tr = document.createElement("tr");
                    const tdKey = document.createElement("td");
                    tdKey.textContent = key;
                    const tdVal = document.createElement("td");
                    tdVal.textContent = value;
                    tr.appendChild(tdKey);
                    tr.appendChild(tdVal);
                    tbody.appendChild(tr);
                }
            }

            table.appendChild(tbody);
            contentDiv.appendChild(table);

            // Warnings nested in input context
            if (input.warnings && input.warnings.length > 0) {
                const warnDiv = document.createElement("div");
                warnDiv.className = "extracted-warnings warning-text";
                const heading = document.createElement("strong");
                heading.textContent = "Warnings:";
                warnDiv.appendChild(heading);

                const ul = document.createElement("ul");
                ul.style.listStyle = "square inside";
                ul.style.marginTop = "4px";
                input.warnings.forEach(warn => {
                    const li = document.createElement("li");
                    li.textContent = warn;
                    ul.appendChild(li);
                });
                warnDiv.appendChild(ul);
                contentDiv.appendChild(warnDiv);
            }

            card.appendChild(contentDiv);
            DOM.inputsContainer.appendChild(card);

            // Toggle expansion listener
            trigger.addEventListener("click", () => {
                const isExpanded = card.classList.contains("expanded");
                if (isExpanded) {
                    card.classList.remove("expanded");
                    trigger.setAttribute("aria-expanded", "false");
                } else {
                    card.classList.add("expanded");
                    trigger.setAttribute("aria-expanded", "true");
                }
            });
        });
    }

    function renderExecutionPlan(plan, planTrace) {
        if (!plan) {
            DOM.planEmpty.classList.remove("hidden");
            DOM.planDetails.classList.add("hidden");
            return;
        }

        DOM.planEmpty.classList.add("hidden");
        DOM.planDetails.classList.remove("hidden");

        DOM.planGoalText.textContent = plan.goal || "Goal not stated.";

        // Render constraints
        if (plan.constraints && plan.constraints.length > 0) {
            DOM.planConstraints.classList.remove("hidden");
            DOM.planConstraintsList.innerHTML = "";
            plan.constraints.forEach(constraint => {
                const li = document.createElement("li");
                li.textContent = constraint;
                DOM.planConstraintsList.appendChild(li);
            });
        } else {
            DOM.planConstraints.classList.add("hidden");
        }

        // Render Steps List Timeline
        DOM.planTimelineList.innerHTML = "";

        // Map trace step statuses
        const planTraceStatuses = {};
        if (planTrace && planTrace.length > 0) {
            planTrace.forEach(event => {
                if (event.step_id && event.status) {
                    planTraceStatuses[event.step_id] = event.status;
                }
            });
        }

        if (plan.steps && plan.steps.length > 0) {
            plan.steps.forEach(step => {
                const statusVal = step.status || planTraceStatuses[step.id] || "pending";

                const li = document.createElement("li");
                li.className = `timeline-step status-${statusVal}`;

                const header = document.createElement("div");
                header.className = "timeline-step-header";

                const idSpan = document.createElement("span");
                idSpan.className = "timeline-step-id";
                idSpan.textContent = step.id;

                const toolSpan = document.createElement("span");
                toolSpan.className = "timeline-step-tool";
                toolSpan.textContent = step.tool || "generic";

                const statusSpan = document.createElement("span");
                statusSpan.className = `timeline-step-status ${statusVal}`;
                statusSpan.textContent = statusVal;

                header.appendChild(idSpan);
                header.appendChild(toolSpan);
                header.appendChild(statusSpan);
                li.appendChild(header);

                const reason = document.createElement("p");
                reason.className = "timeline-step-reason";
                reason.textContent = step.reason || "";
                li.appendChild(reason);

                const meta = document.createElement("div");
                meta.className = "timeline-step-meta";

                if (step.input_reference && step.input_reference.type) {
                    const inputSpan = document.createElement("span");
                    let refDetail = `Input: ${step.input_reference.type}`;
                    if (step.input_reference.source_id) refDetail += ` (${step.input_reference.source_id})`;
                    if (step.input_reference.step_id) refDetail += ` (${step.input_reference.step_id})`;
                    inputSpan.textContent = refDetail;
                    meta.appendChild(inputSpan);
                }

                if (step.depends_on && step.depends_on.length > 0) {
                    const depSpan = document.createElement("span");
                    depSpan.textContent = `Depends on: ${step.depends_on.join(", ")}`;
                    meta.appendChild(depSpan);
                }

                li.appendChild(meta);
                DOM.planTimelineList.appendChild(li);
            });
        }
    }

    function renderTraceTimeline(trace, planTrace) {
        const hasTrace = trace && trace.length > 0;
        const hasPlanTrace = planTrace && planTrace.length > 0;

        if (!hasTrace && !hasPlanTrace) {
            DOM.traceEmpty.classList.remove("hidden");
            DOM.planTraceContainer.classList.add("hidden");
            DOM.execTraceContainer.classList.add("hidden");
            return;
        }

        DOM.traceEmpty.classList.add("hidden");
        toggleTraceTabs(STATE.activeTab);

        // 1. Render Plan Trace List
        DOM.planTraceList.innerHTML = "";
        if (hasPlanTrace) {
            planTrace.forEach(event => {
                const li = document.createElement("li");
                li.className = "trace-item";
                if (event.status) li.classList.add(event.status);

                const header = document.createElement("div");
                header.className = "trace-item-header";

                const stageSpan = document.createElement("span");
                stageSpan.className = "trace-item-stage";
                stageSpan.textContent = event.stage || "planner";

                header.appendChild(stageSpan);

                if (event.tool_name) {
                    const toolSpan = document.createElement("span");
                    toolSpan.className = "trace-item-tool";
                    toolSpan.textContent = event.tool_name;
                    header.appendChild(toolSpan);
                }

                if (event.status) {
                    const statusSpan = document.createElement("span");
                    statusSpan.className = `trace-item-status ${event.status}`;
                    statusSpan.textContent = event.status;
                    header.appendChild(statusSpan);
                }

                li.appendChild(header);

                const msg = document.createElement("p");
                msg.className = "trace-item-message";
                msg.textContent = event.message || "";
                li.appendChild(msg);

                const meta = document.createElement("div");
                meta.className = "trace-item-meta";

                if (event.step_id) {
                    const stepSpan = document.createElement("span");
                    stepSpan.textContent = `Step ID: ${event.step_id}`;
                    meta.appendChild(stepSpan);
                }

                if (event.sequence !== undefined) {
                    const seqSpan = document.createElement("span");
                    seqSpan.textContent = `Seq: ${event.sequence}`;
                    meta.appendChild(seqSpan);
                }

                li.appendChild(meta);
                DOM.planTraceList.appendChild(li);
            });
        } else {
            const empty = document.createElement("li");
            empty.textContent = "No plan trace entries recorded.";
            empty.style.color = "var(--text-muted)";
            DOM.planTraceList.appendChild(empty);
        }

        // 2. Render Execution Trace List
        DOM.execTraceList.innerHTML = "";
        if (hasTrace) {
            trace.forEach(event => {
                const li = document.createElement("li");
                li.className = "trace-item";
                if (event.status) li.classList.add(event.status);

                const header = document.createElement("div");
                header.className = "trace-item-header";

                const stageSpan = document.createElement("span");
                stageSpan.className = "trace-item-stage";
                stageSpan.textContent = event.stage || "executor";

                header.appendChild(stageSpan);

                if (event.tool_name) {
                    const toolSpan = document.createElement("span");
                    toolSpan.className = "trace-item-tool";
                    toolSpan.textContent = event.tool_name;
                    header.appendChild(toolSpan);
                }

                if (event.status) {
                    const statusSpan = document.createElement("span");
                    statusSpan.className = `trace-item-status ${event.status}`;
                    statusSpan.textContent = event.status;
                    header.appendChild(statusSpan);
                }

                li.appendChild(header);

                const msg = document.createElement("p");
                msg.className = "trace-item-message";
                msg.textContent = event.message || "";
                li.appendChild(msg);

                const meta = document.createElement("div");
                meta.className = "trace-item-meta";

                if (event.step_id) {
                    const stepSpan = document.createElement("span");
                    stepSpan.textContent = `Step ID: ${event.step_id}`;
                    meta.appendChild(stepSpan);
                }

                if (event.duration_ms !== null && event.duration_ms !== undefined) {
                    const durSpan = document.createElement("span");
                    durSpan.textContent = `Duration: ${event.duration_ms}ms`;
                    meta.appendChild(durSpan);
                }

                if (event.error_code) {
                    const errSpan = document.createElement("span");
                    errSpan.className = "error-text";
                    errSpan.textContent = `Error: ${event.error_code}`;
                    meta.appendChild(errSpan);
                }

                li.appendChild(meta);
                DOM.execTraceList.appendChild(li);
            });
        } else {
            const empty = document.createElement("li");
            empty.textContent = "No execution trace logs recorded.";
            empty.style.color = "var(--text-muted)";
            DOM.execTraceList.appendChild(empty);
        }
    }

    function toggleTraceTabs(tabId) {
        STATE.activeTab = tabId;
        if (tabId === "plan-trace") {
            DOM.tabPlanTrace.classList.add("active");
            DOM.tabExecTrace.classList.remove("active");
            DOM.planTraceContainer.classList.remove("hidden");
            DOM.execTraceContainer.classList.add("hidden");
        } else {
            DOM.tabExecTrace.classList.add("active");
            DOM.tabPlanTrace.classList.remove("active");
            DOM.execTraceContainer.classList.remove("hidden");
            DOM.planTraceContainer.classList.add("hidden");
        }
    }

    function renderWarningsAndErrors(warnings, errors) {
        const hasWarnings = warnings && warnings.length > 0;
        const hasErrors = errors && errors.length > 0;

        if (!hasWarnings && !hasErrors) {
            DOM.alertsPanel.classList.add("hidden");
            return;
        }

        DOM.alertsPanel.classList.remove("hidden");

        // Warnings display
        if (hasWarnings) {
            DOM.warningSection.classList.remove("hidden");
            DOM.warningList.innerHTML = "";
            warnings.forEach(warn => {
                const li = document.createElement("li");

                // Handle warning strings or structured warnings
                if (typeof warn === "string") {
                    li.textContent = warn;
                } else if (warn && typeof warn === "object") {
                    li.textContent = warn.message || JSON.stringify(warn);
                } else {
                    li.textContent = "Unspecified warning details.";
                }
                DOM.warningList.appendChild(li);
            });
        } else {
            DOM.warningSection.classList.add("hidden");
        }

        // Errors display
        if (hasErrors) {
            DOM.errorSection.classList.remove("hidden");
            DOM.errorList.innerHTML = "";
            errors.forEach(err => {
                const li = document.createElement("li");

                const codeSpan = document.createElement("span");
                codeSpan.className = "alert-list-error-code";
                codeSpan.textContent = err.code || "EXEC_ERROR";

                const textSpan = document.createElement("span");
                textSpan.textContent = err.message || "An execution error occurred.";

                li.appendChild(codeSpan);
                li.appendChild(textSpan);

                let metaDetails = "";
                if (err.step_id) metaDetails += `Step: ${err.step_id}`;
                if (err.tool_name) metaDetails += (metaDetails ? " | " : "") + `Tool: ${err.tool_name}`;

                if (metaDetails) {
                    const metaDiv = document.createElement("div");
                    metaDiv.className = "alert-list-meta";
                    metaDiv.textContent = metaDetails;
                    li.appendChild(metaDiv);
                }

                DOM.errorList.appendChild(li);
            });
        } else {
            DOM.errorSection.classList.add("hidden");
        }
    }

    // ==========================================================================
    // Event Handlers & Element Binding Setup
    // ==========================================================================
    function setupEventListeners() {
        // Character counter handler
        DOM.queryInput.addEventListener("input", () => {
            const count = DOM.queryInput.value.length;
            DOM.charCounter.textContent = `${count.toLocaleString()} / 10,000`;
            if (count > 9500) {
                DOM.charCounter.style.color = "var(--warning)";
            } else {
                DOM.charCounter.style.color = "var(--text-muted)";
            }
        });

        // Scenario Card shortcut click triggers
        DOM.scenariosGrid.addEventListener("click", (event) => {
            if (STATE.isProcessing) return;

            const card = event.target.closest(".scenario-card");
            if (!card) return;

            const scenarioId = card.getAttribute("data-scenario");
            const conf = SCENARIOS[scenarioId];
            if (!conf) return;

            clearDashboardState();

            // Remove active style from cards
            DOM.scenarioCards.forEach(c => c.classList.remove("active"));
            card.classList.add("active");

            // Set composer state
            DOM.queryInput.value = conf.query;
            DOM.queryInput.dispatchEvent(new Event("input"));

            alert(`Shortcut Loaded!\n${conf.hint}\nPlease upload required files and click Run Agent.`);
        });

        // Copy button trigger
        DOM.btnCopyAnswer.addEventListener("click", () => {
            const textToCopy = DOM.answerText.textContent;
            if (!textToCopy) return;

            navigator.clipboard.writeText(textToCopy).then(() => {
                const btnText = DOM.btnCopyAnswer.querySelector("span");
                const oldText = btnText.textContent;
                btnText.textContent = "Copied!";
                setTimeout(() => {
                    btnText.textContent = oldText;
                }, 2000);
            }).catch(err => {
                console.error("Clipboard copy failed:", err);
            });
        });

        // Form Submit boundary
        DOM.agentForm.addEventListener("submit", (event) => {
            event.preventDefault();
            submitAgentRun();
        });

        // Textarea Submission mapping (Ctrl/Cmd + Enter)
        DOM.queryInput.addEventListener("keydown", (event) => {
            if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
                event.preventDefault();
                submitAgentRun();
            }
        });

        // Drop Zone actions
        DOM.dropZone.addEventListener("click", () => {
            if (STATE.isProcessing) return;
            DOM.fileInput.click();
        });

        DOM.fileInput.addEventListener("change", () => {
            handleFileSelection(DOM.fileInput.files);
            DOM.fileInput.value = ""; // Reset input so same file can be re-selected if removed
        });

        // Drag and Drop triggers
        ["dragenter", "dragover"].forEach(eventName => {
            DOM.dropZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                if (STATE.isProcessing) return;
                DOM.dropZone.classList.add("drag-active");
            }, false);
        });

        ["dragleave", "drop"].forEach(eventName => {
            DOM.dropZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                if (STATE.isProcessing) return;
                DOM.dropZone.classList.remove("drag-active");
            }, false);
        });

        DOM.dropZone.addEventListener("drop", (e) => {
            if (STATE.isProcessing) return;
            const dt = e.dataTransfer;
            const files = dt.files;
            handleFileSelection(files);
        }, false);

        DOM.btnClearAll.addEventListener("click", clearAllFiles);

        // Parent Tabs buttons
        DOM.tabPlan.addEventListener("click", () => switchParentTab("plan"));
        DOM.tabInputs.addEventListener("click", () => switchParentTab("inputs"));
        DOM.tabTrace.addEventListener("click", () => switchParentTab("trace"));

        // Technical Trace Tab buttons
        DOM.tabPlanTrace.addEventListener("click", () => toggleTraceTabs("plan-trace"));
        DOM.tabExecTrace.addEventListener("click", () => toggleTraceTabs("exec-trace"));

        // Clarification continue submit action
        DOM.clarificationForm.addEventListener("submit", (event) => {
            event.preventDefault();
            const answer = DOM.clarificationAnswerInput.value.trim();
            if (!answer) {
                alert("Please provide an answer to continue.");
                return;
            }
            submitAgentRun(answer);
        });

        // Clarification cancel click action
        DOM.btnCancelClarify.addEventListener("click", () => {
            STATE.clarificationRequired = false;
            DOM.clarificationCard.classList.add("hidden");
            DOM.agentForm.classList.remove("hidden");
            clearAllFiles();
            DOM.queryInput.value = "";
            DOM.queryInput.dispatchEvent(new Event("input"));
            clearDashboardState();
        });

        // New Run button listener (resets UI back to composer state)
        if (DOM.btnNewRequest) {
            DOM.btnNewRequest.addEventListener("click", clearDashboardState);
        }
    }

    // ==========================================================================
    // Lifecycle Init Hook
    // ==========================================================================
    document.addEventListener("DOMContentLoaded", () => {
        setupEventListeners();
        checkAPIHealth();
    });
})();
