// Build the APA 7 interim report (.docx) from report_data.json and the analysis figures.
// Usage (from analysis/report):  node build_report.js
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, WidthType, AlignmentType, HeadingLevel,
  ImageRun, Header, PageNumber, PageBreak, TableOfContents, BorderStyle, LevelFormat, TabStopType,
} = require("docx");

const D = JSON.parse(fs.readFileSync(path.join(__dirname, "report_data.json"), "utf8"));
const FIG = path.join(__dirname, "..", "outputs", "figures");
const OUT = path.join(__dirname, "..", "docs", "Group4_LRajoria_QM640_Final_Report.docx");
const FONT = "Times New Roman";
const DOUBLE = { line: 480, before: 0, after: 0 };
const SINGLE = { line: 240, before: 0, after: 0 };
const CONTENT_W = 9360; // 6.5 in at 1440 DXA/in

// ---------- inline markup: **bold**, *italic* ----------
function runs(text, size = 24) {
  const parts = text.split(/(\*\*[^*]+\*\*|\*[^*]+\*)/).filter(Boolean);
  return parts.map((p) => {
    if (p.startsWith("**")) return new TextRun({ text: p.slice(2, -2), bold: true, font: FONT, size });
    if (p.startsWith("*")) return new TextRun({ text: p.slice(1, -1), italics: true, font: FONT, size });
    return new TextRun({ text: p, font: FONT, size });
  });
}
const P = (t) => new Paragraph({ children: runs(t), spacing: DOUBLE, indent: { firstLine: 720 } });
const PN = (t) => new Paragraph({ children: runs(t), spacing: DOUBLE }); // no indent
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, alignment: AlignmentType.CENTER, spacing: DOUBLE,
  children: [new TextRun({ text: t, bold: true, font: FONT, size: 24 })] });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, spacing: DOUBLE,
  children: [new TextRun({ text: t, bold: true, font: FONT, size: 24 })] });
const H3 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_3, spacing: DOUBLE,
  children: [new TextRun({ text: t, bold: true, italics: true, font: FONT, size: 24 })] });
const BUL = (t) => new Paragraph({ children: runs(t), spacing: DOUBLE, numbering: { reference: "bullets", level: 0 } });
const NUM = (t, ref = "steps") => new Paragraph({ children: runs(t), spacing: DOUBLE, numbering: { reference: ref, level: 0 } });
const BREAK = () => new Paragraph({ children: [new PageBreak()] });

// ---------- APA tables and figures ----------
let tableNo = 0, figNo = 0;
const none = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
const line = { style: BorderStyle.SINGLE, size: 6, color: "000000" };
function apaTable(title, header, rows, widths, note) {
  tableNo += 1;
  const total = widths.reduce((a, b) => a + b, 0);
  const w = widths.map((x) => Math.round((x / total) * CONTENT_W));
  w[w.length - 1] += CONTENT_W - w.reduce((a, b) => a + b, 0);
  const mk = (txt, i, isHead, isLast) => new TableCell({
    width: { size: w[i], type: WidthType.DXA },
    borders: { left: none, right: none, top: isHead ? line : none, bottom: isHead || isLast ? line : none },
    margins: { top: 40, bottom: 40, left: 80, right: 80 },
    children: [new Paragraph({ spacing: SINGLE, children: runs(String(txt), 20).map((r) => r) })],
  });
  const headRow = new TableRow({ tableHeader: true, children: header.map((h, i) => mk(`**${h}**`, i, true, false)) });
  const body = rows.map((r, ri) => new TableRow({ children: r.map((c, i) => mk(c, i, false, ri === rows.length - 1)) }));
  const out = [
    new Paragraph({ spacing: { ...DOUBLE, before: 240 }, keepNext: true, children: [new TextRun({ text: `Table ${tableNo}`, bold: true, font: FONT, size: 24 })] }),
    new Paragraph({ spacing: DOUBLE, keepNext: true, children: [new TextRun({ text: title, italics: true, font: FONT, size: 24 })] }),
    new Table({ width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: w, rows: [headRow, ...body] }),
  ];
  if (note) out.push(new Paragraph({ spacing: { ...DOUBLE, after: 240 }, children: runs(`*Note.* ${note}`, 20) }));
  else out.push(new Paragraph({ spacing: DOUBLE, children: [] }));
  return out;
}
function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20), data: b };
}
function apaFigure(title, file, note, widthIn = 6.0) {
  figNo += 1;
  const img = pngSize(file);
  const wpx = Math.round(widthIn * 96), hpx = Math.round((wpx * img.h) / img.w);
  return [
    new Paragraph({ spacing: { ...DOUBLE, before: 240 }, keepNext: true, children: [new TextRun({ text: `Figure ${figNo}`, bold: true, font: FONT, size: 24 })] }),
    new Paragraph({ spacing: DOUBLE, keepNext: true, children: [new TextRun({ text: title, italics: true, font: FONT, size: 24 })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, children: [new ImageRun({ type: "png", data: img.data, transformation: { width: wpx, height: hpx } })] }),
    new Paragraph({ spacing: { ...DOUBLE, after: 240 }, children: runs(`*Note.* ${note}`, 20) }),
  ];
}
const fig = (f) => path.join(FIG, f);

// ---------- content ----------
const ABSTRACT = "Video players and mobile networks react to throughput drops only after they happen, which causes stalls and quality switches, especially for users on the move. This study asked whether machine learning can predict 5G downlink throughput five seconds ahead more accurately than persistence, the baseline that assumes current throughput continues, and which inputs matter. The data were a public 5G dataset recorded on a commercial Irish network with one smartphone between November 20, 2019, and February 27, 2020: 83 recording sessions (traces, the unit of analysis) with 180,743 cleaned one-second records of throughput, four radio measurements, speed, network mode, and application type. Ridge regression, random forest, and gradient boosting were tuned and tested with nested cross-validation that held out whole traces, and two stricter schemes held out whole recording days. The analysis was implemented in Python 3.13 with scikit-learn 1.7, stored in a SQLite database and Parquet files, and run on a laptop central processing unit (Intel Core i7-1165G7, 24 GB of memory) without a graphics processing unit; training under all three schemes took about 43 minutes. Radio measurements were not associated with throughput within traces (mean Spearman correlations −.01 to .03; all Holm-adjusted *p* ≥ .05). Gradient boosting with an absolute-error loss was the best model: it reduced mean trace absolute error from 9.49 to 7.73 Mbps, an improvement of 1.76 Mbps (95% CI [1.10, 2.54], *p* < .001), beat persistence on all 83 traces, showed little overfitting (training 7.58 vs. test 7.70 Mbps), and kept its advantage when whole days were held out. Context features added 0.16 Mbps in the primary analysis but nothing on new recording days, and the gain did not differ between stationary and driving conditions. Throughput history, not radio quality, drove the predictions. The implementation is a reproducible Python pipeline with a SQLite database and a Colab notebook, and a lightweight prediction service for video players and network test benches is proposed, with the limits of one-second, single-device data stated.";
const S = [];
const add = (...xs) => xs.flat().forEach((x) => S.push(x));

// Tables and figures are numbered in the order listed here, so the text can refer to them before they appear.
const TAB_ORDER = ["lit", "dict", "db", "precision", "cleaning", "overview", "descriptive", "edaSummary", "models", "features",
  "metrics", "rq1", "rq4", "perf", "traintest", "sens", "hyp"];
const FIG_ORDER = ["repo", "preview", "pipeline", "coverage", "eda1", "eda3", "eda2", "eda4", "importance", "rq1", "rq2", "example",
  "rq3", "rq4", "predobs", "sens"];
const T = (id) => { const i = TAB_ORDER.indexOf(id); if (i < 0) throw new Error(`unknown table ${id}`); return `Table ${i + 1}`; };
const F = (id) => { const i = FIG_ORDER.indexOf(id); if (i < 0) throw new Error(`unknown figure ${id}`); return `Figure ${i + 1}`; };
function tbl(id, ...args) {
  if (TAB_ORDER[tableNo] !== id) throw new Error(`table ${id} out of order (expected ${TAB_ORDER[tableNo]})`);
  return apaTable(...args);
}
function figure(id, ...args) {
  if (FIG_ORDER[figNo] !== id) throw new Error(`figure ${id} out of order (expected ${FIG_ORDER[figNo]})`);
  return apaFigure(...args);
}
const desc = Object.fromEntries(D.descriptive.map((r) => [r[0], r]));
const range = (name, unit) => `${desc[name][5]} to ${desc[name][7]}${unit ? " " + unit : ""}`;
const Q = D.quality, s2 = D.rq2_sel, s3 = D.rq3, E = D.err, PV = D.pvals, TT = D.traintest;

// Title page (template fields; APA 7 student title page)
const center = (t, bold = false, before = 0) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { ...DOUBLE, before },
  children: [new TextRun({ text: t, bold, font: FONT, size: 24 })] });
add(center("AI/ML-Based Prediction of 5G Downlink Throughput", true, 2880),
  center("Using Channel and Context Metrics", true),
  center("Final Report", false), new Paragraph({ spacing: DOUBLE, children: [] }),
  center("Lokesh Rajoria"), center("Walsh College"), center("QM640: Data Analytics Capstone"),
  center("Ms. Keya Choudhury Ganguli"), center("Term 3"), center("November 8, 2026"), BREAK());

// Repository link (template: directly after the title page), then the table of contents
add(center("Project Repository", true),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: DOUBLE, children: runs("GitHub repository with the data files and code: https://github.com/lokeshrajoria-design/Capstone_AIML_Throughput_Prediction") }),
  new Paragraph({ spacing: DOUBLE, children: [] }),
  center("Abstract", true),
  PN(ABSTRACT),
  new Paragraph({ spacing: DOUBLE, indent: { firstLine: 720 }, children: runs("*Keywords:* 5G, throughput prediction, machine learning, gradient boosting, persistence baseline, adaptive streaming") }),
  BREAK(),
  center("Table of Contents", true),
  new TableOfContents("Table of Contents", { hyperlink: true, headingStyleRange: "1-2" }), BREAK());

// ===== Introduction =====
add(center("AI/ML-Based Prediction of 5G Downlink Throughput Using Channel and Context Metrics", true),
  H1("Introduction"),
  H2("Background and Context"),
  P("Video streaming through platforms such as Netflix, YouTube, and Amazon Prime Video has become part of everyday life, and users expect smooth playback whether they are at home or traveling in a car. Maintaining consistent throughput is difficult when radio conditions change, particularly during movement and handovers between cells. The stakeholders are mobile users, who notice stalls and quality drops; network operators and equipment vendors, who tune how networks adapt to changing conditions; and application developers, whose video players choose bitrates from throughput estimates."),
  P("Mobile networks use channel state information (CSI), including the channel quality indicator (CQI), precoding matrix indicator (PMI), and rank indicator (RI), to choose transmission settings. Radio measurements such as reference signal received power (RSRP) and reference signal received quality (RSRQ) describe signal conditions. Over more than 18 years of 4G Long Term Evolution (LTE) and 5G throughput testing, I have observed that throughput under mobility depends strongly on how current this feedback is, which motivated an artificial intelligence and machine learning (AI/ML) proof of concept for this capstone."),
  P("Because company confidentiality rules prevent the use of internal data, the study uses a public dataset recorded on a commercial mobile network (Raca, Leahy, et al., 2020). It compares short-term throughput predictions from AI/ML models with a baseline that assumes current throughput continues, and examines whether mobility and application context improve those predictions."),
  H2("Stakeholder Value"),
  P("Each stakeholder gains something different from a reliable short-term forecast. Mobile users would see fewer stalls and fewer sudden quality drops if their video player could lower its bitrate a few seconds before throughput falls. Application developers would get a better input for bitrate adaptation than the simple averages players use today. Network operators, equipment vendors, and test engineers would get an early warning of throughput loss under mobility and, just as important, evidence about which measurements are worth collecting for prediction: throughput history, radio quality, or context."),
  H2("Problem Statement"),
  P("Networks and video players adapt to throughput changes only after they happen, so a drop under mobility shows up as a stall or a quality switch before anything reacts. It is not known how much of the next few seconds of throughput can be predicted from what a phone already measures, or which of those measurements matter. The study therefore asks whether AI/ML models can predict short-term downlink throughput more accurately than a persistence baseline, and whether radio measurements, recent throughput history, and contextual information (speed, network mode, and application type) contribute to that prediction under stationary and moving conditions. The results bear on the potential of AI/ML to support channel-feedback enhancement, while remaining limited to prediction rather than changes to CSI feedback or network performance."),
  H2("Purpose of the Study"),
  P("The study is a **prediction** study with an **explanatory** component. It predicts downlink throughput five seconds ahead as a supervised regression problem, and it explains which inputs drive the predictions through feature importance and a direct comparison of models with and without context features."),
  H2("Scope and Objectives"),
  P("The scope is an offline proof of concept on public data: AI/ML models predict throughput five seconds ahead and are compared with persistence. CQI is an input, not a target, and the study does not modify CSI feedback or demonstrate a throughput gain on a network."),
  P("The objectives are to (a) measure the association between radio measurements and throughput, (b) compare AI/ML prediction with persistence, (c) test the added value of context, and (d) compare performance across mobility and application conditions. The **dependent variable** is downlink throughput five seconds ahead (Mbps). The **independent variables** are RSRP, RSRQ, signal-to-noise ratio (SNR), CQI, recent throughput history, and the context variables speed, network mode, and application type. The study is **feasible** because the data are public, small enough for a single computer (about 190,000 records), and suited to standard tree-based and linear models."),
  P("**Dependent variable (Y):** downlink throughput in megabits per second (Mbps) exactly five seconds after the prediction origin. **Independent variables (X):** RSRP, RSRQ, SNR, CQI, throughput history (current value, lags of 1, 2, and 5 seconds, and the 10-second mean and standard deviation), and the context variables speed, network mode, and application type. **Unit and timeframe:** the trace is the unit of analysis and the one-second record the unit of prediction; the records span November 20, 2019, to February 27, 2020. **Success metrics:** the study counts as successful if (a) the selected model’s mean trace mean absolute error (MAE) is lower than persistence with a one-sided *p* < .05 and a 95% confidence interval above zero on held-out traces, (b) the improvement remains significant when whole recording days are held out, and (c) the model beats persistence on most traces; root mean squared error (RMSE) is reported alongside."),
  H2("Research Questions"),
  P("The study addresses four research questions (RQs), each with a null hypothesis (*H*₀) and an alternative hypothesis (*H*ₐ). Every test uses a 5% significance level (α = .05)."),
  H3("Research Question 1 (RQ1)"),
  P("What relationships exist between radio measurements (RSRP, RSRQ, SNR, and CQI) and observed downlink throughput? *H*₀₁: the mean within-trace Spearman correlation with throughput is zero for each of the four measurements (ρ̄ = 0); *H*ₐ₁: it is nonzero for at least one (ρ̄ ≠ 0). Test: one-sample *t* test across traces for each measurement with Holm adjustment, α = .05."),
  H3("Research Question 2 (RQ2)"),
  P("How accurately can AI/ML models using radio measurements and recent throughput history predict throughput five seconds ahead, compared with persistence? With improvement Δ defined per trace as persistence MAE minus AI/ML MAE, *H*₀₂: the mean improvement is at most zero (μΔ ≤ 0); *H*ₐ₂: it is greater than zero (μΔ > 0). Test: one-sided bootstrap over traces, α = .05."),
  H3("Research Question 3 (RQ3)"),
  P("Does adding speed, network mode, and application type improve prediction? *H*₀₃: the mean context benefit (base MAE minus context MAE) is at most zero (μΔ ≤ 0); *H*ₐ₃: it is greater than zero (μΔ > 0). Test: one-sided bootstrap over traces, α = .05."),
  H3("Research Question 4 (RQ4)"),
  P("How does the improvement over persistence vary between stationary and moving conditions, and across application types? *H*₀₄ₐ: the mean improvement is equal for stationary and driving traces; *H*ₐ₄ₐ: it differs (primary; two-sided 95% bootstrap interval, α = .05). *H*₀₄b: the mean improvement is equal across Amazon Prime, Netflix, and download; *H*ₐ₄b: at least one pair differs (exploratory; Bonferroni-adjusted 98.3% intervals, α = .05 overall)."),
);

// ===== Literature survey =====
add(H1("Literature Survey"),
  H2("Literature Review Approach"),
  P("Sources were located through keyword searches of scholarly indexes (Google Scholar, Crossref, OpenAlex, and Semantic Scholar) and publisher libraries (IEEE Xplore, the ACM Digital Library, Springer, MDPI, and arXiv), and by following the references of relevant papers. Keywords combined “throughput prediction” or “bandwidth prediction” with “5G,” “LTE,” “cellular,” “CQI,” “machine learning,” “deep learning,” “large language model,” “agent,” and “adaptive bitrate streaming.” Studies were included if they were peer-reviewed papers or preprints from 2015 onward, addressed cellular or streaming throughput, and either reported quantitative prediction results or published data used here. Six methodological sources on cross-validation, forecast metrics, sample size, and the learning algorithms were added for the methods. Every bibliographic record was checked against its DOI. Seventeen domain sources are reviewed below; each is linked to the research questions in " + T("lit") + "."),
  H2("Summary of Key Literature"),
  P("Research on mobile throughput prediction falls into two broad families: methods that forecast throughput from its own recent history, and methods that predict it from radio, network, and context measurements. A third body of work, on adaptive video streaming, shows why the forecasts matter, because a player’s bitrate decisions are only as good as the throughput estimate behind them. Public datasets that record throughput together with radio measurements make a direct comparison possible."),
  H3("Prediction From Throughput History"),
  P("Sun et al. (2016) analyzed more than 20 million video sessions and found that sessions sharing features such as network provider and region had similar initial throughput, and that throughput within a session moved between persistent states. Their CS2P system clusters similar sessions and uses a hidden Markov model for midstream prediction; it reduced the median prediction error by 40% for initial throughput and by 50% for midstream throughput compared with existing predictors. Mei et al. (2019) predicted mobile bandwidth one step ahead with a long short-term memory (LSTM) network on traces recorded on subways and buses, and reported significant accuracy gains over recursive least squares in individual scenarios. Biernacki (2024) grouped fragments of 5G throughput traces into clusters by their mean and variance and fitted a separate LSTM model for each cluster; on fixed and mobile 5G traffic, the clustered models predicted better than a single global model."),
  P("Biswas et al. (2023) moved in the opposite direction from deep learning. They modeled future throughput as a multiple linear regression of present throughput and present network parameters, then corrected the estimate with a Kalman filter that accounts for measurement error. Across seven public 5G datasets and several prediction windows, this low-complexity method was more accurate than the machine-learning and deep-learning baselines and was faster to run and retrain. Together these studies support a common premise: over horizons of a few seconds, recent throughput is the strongest single signal, and simple methods built on it are hard to beat. That premise is the reason this study judges every model against persistence and a 10-second moving average."),
  H3("Prediction From Radio, Network, and Context Measurements"),
  P("Yue et al. (2018) measured two commercial LTE networks in the United States and identified five types of lower-layer information that were correlated with link bandwidth. Their LinkForecast framework combined these with upper-layer features and predicted bandwidth at one-second granularity with an average error of 3.9% to 17.0% across the scenarios studied; features read through standard operating-system interfaces gave accuracy similar to those from specialized logging tools. Minovski et al. (2021) trained machine-learning models on lower-layer radio metrics collected while driving in urban, suburban, and rural areas and in crowded venues. They reported *R*² of .93 in LTE and .84 in non-standalone 5G, with mean squared errors of 0.06 and 0.17, respectively."),
  P("Raca, Zahran, et al. (2020) compared random forest, support vector machine, and LSTM predictors on cellular traces. Summarizing the input history as quantiles markedly improved the two classical models, and the LSTM reached similar accuracy from raw inputs with the lowest 90th-percentile error. Two further results bear directly on this study. Sampling every 100 ms instead of every second reduced the absolute relative error to 20%–36% of its one-second value, and adding network-side measurements, such as cell load and the CQI of competing users, improved accuracy for every combination of history length and horizon tested. Narayanan et al. (2020) showed in the Lumos5G study that commercial millimeter-wave 5G throughput is predictable from user-side features such as location, mobility, and connection state, and that combining feature groups reduced prediction error by a factor of 1.37 to 4.84 compared with existing models. Kaviani et al. (2026) addressed prediction of CQI itself, using a residual-based adaptive Huber loss to cope with heavy-tailed errors, which is directly relevant to the absolute-error loss selected in this study."),
  H3("Throughput Prediction in Adaptive Streaming"),
  P("In Hypertext Transfer Protocol (HTTP) adaptive streaming the player chooses a bitrate for each video segment, so prediction errors turn directly into playback stalls or needlessly low quality. Yin et al. (2015) formulated bitrate adaptation as model predictive control (MPC), which combines a throughput forecast with the current buffer level; that controller uses a harmonic mean of past throughput as its forecast (Sun et al., 2016). Replacing the harmonic mean with the CS2P predictor raised overall quality of experience (QoE) by 3.2% and average bitrate by 10.9% (Sun et al., 2016). Mao et al. (2017) took a different route: their Pensieve system learns a bitrate policy by reinforcement learning from observed throughput and buffer states, and it improved average QoE by 12% to 25% over the best existing scheme. Raca, Zahran, et al. (2020) estimated the upper bound of the benefit: with perfect knowledge of the next 20 seconds of throughput, stall events disappeared while average bitrate fell by only about 6%. These results show that the value of a forecast lies in how much it improves on the simple estimators players already use, which is why this study reports improvement over persistence and a moving average rather than absolute error alone."),
  H3("Deep Learning and Large Language Model Approaches"),
  P("Deep learning (DL) has moved from single recurrent networks toward models built around network-specific structure. Ye et al. (2024) measured carrier aggregation in commercial 5G networks and showed that the combination of radio channels in use changes throughput in ways that generic predictors miss; their carrier-aggregation-aware DL framework, Prism5G, improved throughput prediction accuracy by more than 14% on average and up to 22% over existing algorithms. Large language models (LLMs) are now being adapted to the same problems. Wu et al. (2024) adapted a pretrained LLM to networking tasks with a multimodal input encoder, a task-specific output head, and low-cost fine-tuning, and reported gains of 14.5% to 36.6% over state-of-the-art adaptive bitrate algorithms. Jiang et al. (2024) moved toward agentic artificial intelligence: their CommLLM system combines several LLM agents for knowledge retrieval, collaborative planning, and evaluation and reflection to solve communication tasks stated in natural language. These approaches rely on large training corpora or pretrained models and substantial computing power, which is one reason this study, with 83 traces and a laptop processor, uses tree models and treats DL and LLM methods as future work."),
  H3("Datasets"),
  P("Raca et al. (2018) published a 4G LTE dataset of 135 traces from two Irish operators, averaging 15 minutes each, with throughput from 0 to 173 Mbps and channel and context metrics recorded once per second by the G-NetTrack Pro application under static, pedestrian, car, bus, and train mobility. Raca, Leahy, et al. (2020) extended the approach to 5G with the dataset used here. Both record application-level throughput alongside RSRP, RSRQ, SNR, and CQI, which makes it possible to test history-based and measurement-based prediction on the same traces."));
add(tbl("lit", "Literature Relevance Matrix",
  ["Author (Year)", "Domain/Context", "Dataset/Setting", "Method(s)", "Key findings", "How it supports the RQs"],
  [["Sun et al. (2016)", "Video streaming", "20M+ video sessions", "Session clustering, hidden Markov model", "Median error 40% lower (initial), 50% lower (midstream)", "RQ2: history-based prediction; value over simple predictors"],
   ["Mei et al. (2019)", "Mobile bandwidth", "Subway and bus traces", "LSTM vs. RLS", "Significant gains over RLS", "RQ2: history as main signal; LSTM considered, not used"],
   ["Biernacki (2024)", "5G video", "Fixed and mobile 5G traces", "Clustering, per-cluster LSTM", "Clustered models better than one global model", "RQ3, RQ4: condition-specific behavior"],
   ["Biswas et al. (2023)", "5G throughput", "Seven public 5G datasets", "Regression with Kalman filter", "More accurate and faster than ML and DL baselines", "RQ2: simple methods as strong benchmarks"],
   ["Yue et al. (2018)", "LTE link bandwidth", "Two U.S. LTE networks", "LinkForecast (ML)", "Average error 3.9%–17.0% at 1 s", "RQ1, RQ2: lower-layer radio features"],
   ["Minovski et al. (2021)", "LTE and 5G", "LTE and NSA 5G drive tests", "ML regression", "*R*² = .93 (LTE), .84 (5G)", "RQ1, RQ2: radio metrics as predictors"],
   ["Raca, Zahran, et al. (2020)", "Cellular, video", "Cellular traces", "RF, SVM, LSTM", "100 ms sampling cut error to 20%–36% of 1 s value", "RQ1, RQ3: limits of 1 s device data"],
   ["Narayanan et al. (2020)", "mmWave 5G", "Commercial mmWave 5G", "Lumos5G ML framework", "Error 1.37 to 4.84 times lower than existing models", "RQ3: context features"],
   ["Kaviani et al. (2026)", "5G CQI", "5G CQI data", "Adaptive Huber loss", "Robust loss for heavy-tailed errors", "RQ2: absolute-error loss"],
   ["Yin et al. (2015)", "Video streaming", "Trace-driven emulation", "MPC bitrate control", "Prediction-based bitrate control framework", "Motivation: how forecasts are used"],
   ["Mao et al. (2017)", "Video streaming", "Trace-driven and real networks", "Reinforcement learning (Pensieve)", "Average QoE 12%–25% higher", "Motivation: value of good estimates"],
   ["Ye et al. (2024)", "5G carrier aggregation", "Commercial 5G measurements", "CA-aware deep learning (Prism5G)", "Prediction accuracy 14% better on average, up to 22%", "Future scope: DL with richer radio data"],
   ["Wu et al. (2024)", "Networking with LLMs", "ABR and other networking tasks", "Adapted LLM (NetLLM)", "14.5%–36.6% better ABR performance", "Future scope: LLM-based prediction and control"],
   ["Jiang et al. (2024)", "6G communications", "Communication-task case study", "LLM multi-agent system (CommLLM)", "Agents for retrieval, planning, evaluation, and reflection", "Proposed deployment: agentic integration"],
   ["Raca et al. (2018); Raca, Leahy, et al. (2020)", "4G and 5G data", "4G (135 traces) and 5G datasets", "Dataset papers", "Public throughput, channel, and context data", "Data source for all RQs"]],
  [1.5, 1.1, 1.2, 1.3, 1.5, 1.4],
  "Results are as stated in each publication; they are not directly comparable because datasets, horizons, and metrics differ. LSTM = long short-term memory; RLS = recursive least squares; ML = machine learning; DL = deep learning; NSA = non-standalone; RF = random forest; SVM = support vector machine; MPC = model predictive control; QoE = quality of experience; CA = carrier aggregation; ABR = adaptive bitrate; LLM = large language model."));
add(H2("Synthesis and Gap"),
  P(T("lit") + " compares the studies. Four points emerge. First, history-based and measurement-based approaches are usually evaluated separately. Studies that add radio or network features report gains, but mostly against other learned models rather than against persistence, so it remains unclear how much radio and context measurements add once throughput history is known. RQ2 and RQ3 test exactly that increment, and RQ1 checks the precondition that radio measurements move with throughput within a trace."),
  P("Second, the reported accuracy figures cannot be compared with one another. The studies use different horizons, from one step to 20 seconds, different metrics, from relative error and *R*² to QoE, and different datasets. Results also depend heavily on how data are split: consecutive one-second records are strongly correlated, so a random record-level split lets a model see near-duplicates of its test data (Roberts et al., 2017). Holding out whole traces or recording days gives a more realistic estimate of performance under new conditions. This study therefore evaluates on complete held-out traces, repeats the analysis with whole recording days held out, and reports MAE and RMSE in megabits per second with bootstrap confidence intervals computed over traces."),
  P("Third, the newest work shifts effort from feature engineering to model capacity. DL models such as Prism5G and LLM-based systems such as NetLLM and CommLLM report gains when large datasets, pretrained models, and accelerators are available (Jiang et al., 2024; Wu et al., 2024; Ye et al., 2024). For a small public dataset, the more useful question is whether a simple, well-validated model already captures most of the predictable signal, because the answer sets the bar that any DL or LLM approach must clear."),
  P("Fourth, the largest radio-side gains in the literature come from information that a handset log does not contain: network-side measurements such as cell load, and sub-second sampling (Raca, Zahran, et al., 2020; Yue et al., 2018). The public 5G dataset used here records device-side radio measurements once per second. A small or null contribution from radio features in this study would therefore be consistent with the literature rather than contradict it, and RQ1 and RQ3 are interpreted with that limit in view. The gap this study fills is a transparent, trace-level comparison of history-only and context-augmented models against simple baselines on a public 5G dataset, with uncertainty reported at the level of independent traces."));

// ===== Data description =====
add(H1("Data Description"),
  P("The study uses the production portion of a public 5G dataset recorded on a commercial Irish network (Raca, Leahy, et al., 2020). Each file is one recording session, or trace, with one record per second of throughput, radio, and context measurements."),
  H2("Data Source and Access"),
  P("The data are the production dataset of “Beyond Throughput, the Next Generation: A 5G Dataset with Channel and Context Metrics” (Raca, Leahy, et al., 2020), recorded with the G-NetTrack Pro application on a Samsung Galaxy S10 5G during stationary and driving sessions with Amazon Prime Video, Netflix, and file downloads. The archive 5G-production-dataset.zip is available from the authors’ repository at https://github.com/uccmisl/5Gdataset. Its folders separate application and mobility condition, and each comma-separated values (CSV) file is one trace (" + F("repo") + "). The copy used was verified with a Secure Hash Algorithm 256 (SHA-256) checksum."),
  H2("Dataset Overview"),
  BUL("**Number of records (rows):** 188,711 raw one-second records; 180,743 after repeated timestamps were merged."),
  BUL("**Number of variables (columns):** 26 per file, defined in " + T("dict") + "."),
  BUL("**Time period:** November 20, 2019, to February 27, 2020, on 27 recording dates."),
  BUL("**Unit of analysis:** the one-second record is the unit of prediction; the trace (83 in total: 60 driving, 23 stationary) is the independent sampling unit for inference."),
  BUL("**Target variable:** downlink throughput in megabits per second exactly five seconds after the prediction origin, within the same trace."),
  H2("Inclusion and Exclusion Criteria"),
  BUL("**Included traces:** all 83 traces of the production dataset (Amazon Prime, Netflix, and download; stationary and driving)."),
  BUL("**Included records:** every cleaned one-second record (180,743) for exploratory data analysis (EDA)."),
  BUL("**Included prediction origins:** records with current throughput, the exact five-second target in the same trace, and at least 7 of the 10 preceding seconds observed: " + D.eligible_total.toLocaleString("en-US") + " origins (88.9%)."),
  BUL("**Excluded variables:** coordinates and cell or node identifiers (privacy), ping fields (98.3% missing), neighbor-cell fields (meaning unverified), and UL_bitrate, received signal strength indicator (RSSI), and State (descriptive only)."),
  BUL("**Excluded records:** none were deleted; repeated timestamps were merged and sentinel codes set to missing (see Data Cleaning)."),
  H2("Data Dictionary"),
  P(T("dict") + " defines every variable in the source files, its data type, the observed range, how missing values are handled, and whether it is used. Variables derived for modeling are listed with the features in the Modeling section."));
add(tbl("dict", "Data Dictionary",
  ["Variable", "Definition", "Data type", "Allowed values / observed range", "Missing values handling", "Notes"],
  [["Timestamp", "Date and time of the record", "Date/time", "2019-11-20 to 2020-02-27", "None missing; repeated stamps merged", "Orders records; defines lags and the 5 s target"],
   ["DL_bitrate", "Downlink application throughput", "Numeric (kbps; converted to Mbps)", range("Throughput (Mbps)", "Mbps"), "None missing", "History features and target"],
   ["UL_bitrate", "Uplink throughput", "Numeric (kbps)", "0 to 6,862 kbps", "None missing", "Descriptive only"],
   ["RSRP", "Reference signal received power", "Numeric, continuous (dBm)", range("RSRP (dBm)", "dBm") + "; −200 is a sentinel", `${D.missing.RSRP}% missing; training-fold median plus flag`, "Radio predictor (RQ1)"],
   ["RSRQ", "Reference signal received quality", "Numeric, continuous (dB)", range("RSRQ (dB)", "dB"), `${D.missing.RSRQ}% missing; median plus flag`, "Values above −3 dB occur only in 3G/2G modes"],
   ["SNR", "Signal-to-noise ratio", "Numeric, continuous (dB)", range("SNR (dB)", "dB"), `${D.missing.SNR}% missing; median plus flag`, "Radio predictor (RQ1)"],
   ["CQI", "Channel quality indicator", "Ordinal integer", "0 to 15", `${D.missing.CQI}% missing; median plus flag`, "Input, not a target"],
   ["RSSI", "Received signal strength indicator", "Numeric (dBm)", "−113 to −31 dBm", `${D.missing.RSSI}% missing`, "Descriptive only"],
   ["Speed", "Speed from the phone’s location service", "Numeric (km/h)", range("Speed (km/h)", "km/h"), "None missing", "Context predictor (RQ3)"],
   ["NetworkMode", "Radio access technology in use", "Categorical", "5G, LTE, HSPA+, HSDPA, HSUPA, UMTS, EDGE, GPRS", "None missing; one-hot encoded", "Context predictor; grouped as 3G/2G for description"],
   ["State", "Traffic state code from G-NetTrack Pro", "Categorical", "D, I, V, VD", "None missing", "Descriptive only"],
   ["Operatorname", "Masked operator code", "Categorical", "One operator code", "Not applicable", "Not modeled"],
   ["Longitude, Latitude", "Location of the phone", "Numeric (degrees)", "Irish locations", "Not applicable", "Excluded for privacy"],
   ["CellID, RAWCELLID, CELLHEX, NODEHEX, LACHEX", "Cell, node, and area identifiers", "Categorical identifiers", "Network codes", "Not applicable", "Auditing only; not modeled or published"],
   ["PINGAVG, PINGMIN, PINGMAX, PINGSTDEV, PINGLOSS", "Ping round-trip statistics", "Numeric (ms, %)", "64 to 889 ms (average)", `${D.missing.PINGAVG}% missing`, "Excluded"],
   ["NRxRSRP, NRxRSRQ", "Neighbor-cell RSRP and RSRQ", "Numeric (dBm, dB)", "−205 sentinel present", `${D.missing.NRxRSRP}% and ${D.missing.NRxRSRQ}% missing`, "Excluded; meaning unverified"]],
  [1.5, 1.4, 1.1, 1.3, 1.4, 1.3],
  "A literal hyphen in the source files denotes a missing value. Ranges are observed values: after cleaning for modeled variables, in the raw files otherwise. kbps = kilobits per second; Mbps = megabits per second; HSPA+, HSDPA, HSUPA, and UMTS are 3G technologies; EDGE and GPRS are 2G."));
add(H2("Database Setup and Data Access"),
  P("The cleaned data are stored in a SQLite database, a single-file relational database that needs no server (data/capstone.sqlite, 87.9 MB). It is built by build_database.py from the outputs of the cleaning pipeline and holds the four tables in " + T("db") + ", each indexed on the trace identifier, plus a view that summarizes traces by group. Coordinates and cell identifiers are not stored."));
add(tbl("db", "Database Tables", ["Table", "Rows", "Columns", "Content"],
  [["traces", "83", "8", "One row per recording session: application, mobility, content, recording date, start time, duration"],
   ["records", "180,743", "25", "Cleaned one-second records: radio measurements, throughput, speed, network mode"],
   ["features", "180,743", "28", "Model inputs, missing-value flags, eligibility, and the five-second target per origin"],
   ["splits", "83", "7", "Fold assignment of each trace under the three evaluation schemes"],
   ["v_group_summary (view)", "6", "5", "Traces, seconds, and mean throughput by application and mobility"]],
  [1.8, 0.9, 0.9, 4.6], "Every table is indexed on the trace identifier (unique in traces and splits); in records and features, the trace identifier and timestamp together identify one row."));
add(P("Data are accessed in two ways. Ad hoc questions and checks use Structured Query Language (SQL) against the database through Python’s sqlite3 module or any SQLite client; the view v_group_summary, for example, returns the same trace counts and mean throughput per group as the summary in the EDA section, which confirms that the database and the analysis files agree. The modeling scripts read the same tables from Parquet files with pandas, so every analysis can be rerun from either source."));
add(H2("Sample Size"),
  P("Traces, not individual records, are the independent sampling units, because consecutive readings within a trace are repeated observations. Following a confidence-interval precision approach (Lakens, 2022), the required number of traces is *N* = ceiling[(*z* × σ / *e*)²], where σ is the between-trace standard deviation of the estimated quantity and *e* the desired interval half-width. Targets were *e* = 0.25σ for means and *e* = 0.50σ for group differences. " + T("precision") + " shows the calculation for each RQ and the precision the 83 available traces actually achieve."));
add(tbl("precision", "Required and Achieved Precision by Research Question",
  ["RQ", "Required", "Available", "Achieved half-width", "Target", "Target met?"],
  D.precision, [0.6, 1.5, 2.0, 1.0, 0.7, 1.5],
  "Required sizes follow the synopsis: RQ1 uses a Bonferroni *z* of 2.4977 across four measurements as a conservative planning bound (the analysis applies Holm adjustment); RQ2 and RQ3 use *z* = 1.96; RQ4a uses *z* = 1.96 with a 60:23 driving-to-static allocation; RQ4b uses a Bonferroni *z* of 2.394 across three pairs and reports the least precise pair. The overall requirement was 138 traces."));
add(P("The dataset meets the precision targets for RQ2, RQ3, and the primary mobility comparison (RQ4a), narrowly misses the RQ1 target, and falls clearly short for the application comparison (RQ4b), which is therefore treated as exploratory. A “No” in " + T("precision") + " means that the confidence interval is wider than planned; it does not mean that the research question could not be answered."),
  H2("GitHub Data Availability Statement"),
  P("The raw and processed datasets and all code are in the project’s GitHub repository (https://github.com/lokeshrajoria-design/Capstone_AIML_Throughput_Prediction), which mirrors the project folder:"),
  BUL("**Raw data:** data/raw/5G-production-dataset.zip (the original archive, verified by checksum)."),
  BUL("**Processed and clean data:** data/interim/clean_1s.parquet (cleaned one-second records) and data/processed/ (features.parquet, splits.csv, and trace_manifest.csv)."),
  BUL("**Database:** data/capstone.sqlite (built by build_database.py)."),
  BUL("**Code and notebooks:** the tput/ package, the run and figure scripts listed in Appendix A, and Capstone_5G_Throughput_Colab.ipynb."),
  H2("Screenshots and Evidence"),
  P(F("repo") + " shows the authors’ repository and the folder tree of the archive, and " + F("preview") + " shows the first records of one trace file."));
add(figure("repo", "Repository Structure of the 5G Production Dataset", path.join(__dirname, "synopsis_fig1_0.png"),
  "Left: root of the public repository. Right: contents of 5G-production-dataset.zip with the number of CSV traces per folder (83 in total). Dataset from Raca, Leahy, et al. (2020).", 5.6));
add(figure("preview", "Preview of a Raw Trace File", fig("fig0_dataset_preview.png"),
  "Records 101–112 of Download/Driving/B_2019.12.14_10.16.30.csv. Coordinates, identifiers, ping, and neighbor-cell columns are omitted. DL_bitrate and UL_bitrate are in kbps."));
add(P("The preview already shows three of the data-quality issues treated below: the timestamp 10.18.20 appears twice, the second 10.18.21 is missing, and RSSI changes to the hyphen that marks a missing value. It also shows throughput falling from about 21 Mbps to zero within two seconds while CQI stays at 11."),
  H2("Ethics, Completeness, and Limitations"),
  P("The data are public measurements with no personal information collected for this study. Coordinates and cell identifiers could reveal routes, so they are excluded from models and no locations are published; the original authors are cited as requested. The data come from one device on one operator in 2019–2020, the traces are unbalanced (60 driving, 23 stationary), about 10% of seconds are missing from the logs, and application throughput reflects application demand as well as network conditions. These limitations are quantified in the Analysis section."));

// ===== Analysis =====
add(H1("Project Pipeline"),
  P(F("pipeline") + " shows the six stages of the project. The first five (data, preprocessing, EDA, feature engineering, and modeling) were implemented and evaluated; the sixth, deployment, is a proposed design described in the Deployment and Integration section."));
add(figure("pipeline", "Project Pipeline From Data to Deployment", fig("fig9_pipeline.png"),
  "Solid blue boxes are implemented and evaluated; the dashed orange box is proposed and not implemented. RF = random forest; GB = gradient boosting."));
add(H2("Data Ingestion"),
  P("Ingestion starts from the downloaded archive. The pipeline script checks the archive’s SHA-256 checksum, extracts the 83 CSV files, reads each one, parses the timestamp, converts the hyphen used for missing values, and adds the trace identifier, application, and mobility condition taken from the folder path. The cleaned records, features, and fold assignments are written to Parquet files and then loaded into the SQLite database."));
add(H1("Analysis"),
  P("This section describes how the data were cleaned and what the exploratory analysis showed, with each table and figure followed by its interpretation."),
  H2("Data Cleaning"),
  P(T("cleaning") + " logs each data-quality issue, the variables affected, how it was detected, how it was treated, and why. Missing values were the most common problem, repeated timestamps the main source of duplicates, and the apparent outliers turned out to be genuine."));
add(tbl("cleaning", "Data Cleaning Log",
  ["Issue", "Variables affected", "Detection method", "Treatment applied", "Rationale"],
  [["Missing values", `SNR ${D.missing.SNR}%, CQI ${D.missing.CQI}%, RSRQ ${D.missing.RSRQ}%, RSSI ${D.missing.RSSI}%, ping ${D.missing.PINGAVG}%`, "Count of the literal “-” per column", "Converted to missing; radio values imputed with the training-fold median plus indicator flags; RQ1 uses pairwise deletion; ping and neighbor fields excluded", "Imputing inside folds avoids leakage; flags let the models use missingness"],
   ["Duplicates (repeated timestamps)", "All variables", `${Q.rows_with_repeated_timestamp.toLocaleString("en-US")} rows share a timestamp with another row`, "Merged to one row per second: numeric values averaged, categorical values take the last value", "Exact lags and the 5 s target need one record per second"],
   ["Missing seconds", "Timestamp", `Gaps between consecutive records: ${Q.gaps_2s.toLocaleString("en-US")} of 2 s, ${Q.gaps_3s_plus} of 3 s or more`, "No interpolation; lags and targets looked up by exact time", "Interpolation would invent throughput values"],
   ["Sentinel codes", "RSRP, NRxRSRP", `RSRP −200 dBm (${Q.rsrp_sentinel_minus200} rows); neighbor −205 dBm (${Q.neighbour_rsrp_sentinel_minus205} rows)`, "Set to missing", "Below the 3rd Generation Partnership Project (3GPP) reporting floor of −156 dBm"],
   ["Outliers: extreme values", "DL_bitrate, Speed, SNR, CQI", `Range checks against physical and 5G limits: maximum ${Q.max_dl_mbps} Mbps, speed up to ${Q.max_speed_kmh} km/h, ${Q.snr_below_minus23} SNR values below −23 dB, ${Q.cqi_zero} CQI = 0`, "Kept", "Plausible values; removing them would bias results toward easy conditions"],
   ["Outliers: zero throughput", "DL_bitrate", "Share of zero seconds by group (0%–77%)", "Kept", "Genuine idle periods of the video player, not errors"],
   ["Anomalous RSRQ", "RSRQ, NetworkMode", "Values above −3 dB cross-tabulated with network mode: 12,567 seconds, all 3G/2G", "Kept; network mode offered to the context model", "The field holds a different quantity in 3G modes"]],
  [1.2, 1.4, 1.8, 2.0, 1.6]));
add(P("Prediction origins required the exact five-second target and current throughput. The synopsis also required all ten history seconds and every lag, but that rule kept only 18.8% of origins and removed groups unevenly. The rule used instead requires at least 7 of the 10 window seconds and imputes missing lags inside training folds, keeping " + D.eligible_total.toLocaleString("en-US") + " origins (88.9%)."));
add(figure("coverage", "Prediction Origins Retained Under Two Eligibility Rules", fig("fig5_coverage_rules.png"),
  "Share of logged seconds usable as prediction origins under the strict synopsis rule and the rule used."));
add(P(F("coverage") + " shows why the rule was changed: the strict rule kept as little as 4% of origins in some driving and download groups, while the rule used keeps 82%–92% in every group, so no condition is underrepresented. The change was fixed before any model was fitted, and the strict rule remains available as a sensitivity setting."),
  H2("Reproducible Workflow"),
  P("The workflow is a Python package with fixed random seed 640 and an input checksum. Running the scripts listed in Appendix A in order regenerates every table and figure in this report from the raw archive, and the Colab notebook reproduces the same steps in a browser."),
  H2("EDA Results"),
  H3("Statistical Summaries"),
  P(T("overview") + " summarizes each application and mobility group, and " + T("descriptive") + " the distribution of the modeling variables."));
add(tbl("overview", "Trace and Throughput Summary by Group",
  ["Application", "Mobility", "Traces", "Seconds", "Median trace (min)", "Mean (Mbps)", "Median (Mbps)", "P90 (Mbps)", "Zero (%)", "5G (%)", "Speed (km/h)"],
  D.overview, [1.3, 0.9, 0.7, 0.9, 0.9, 0.8, 0.8, 0.8, 0.7, 0.6, 0.8],
  "Statistics over cleaned one-second records. P90 = 90th percentile. Mean speed above zero in stationary groups reflects location noise."));
add(P("The groups differ by an order of magnitude: stationary downloads average 70.9 Mbps, while Netflix averages under 1 Mbps and carries no data in 71%–77% of seconds. Driving traces spend far less time on 5G than stationary ones."));
add(tbl("descriptive", "Descriptive Statistics of the Modeling Variables",
  ["Variable", "*N*", "Missing (%)", "*M*", "*SD*", "Min", "Median", "Max"], D.descriptive, [1.8, 1.1, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9],
  "Computed over 180,743 cleaned one-second records."));
add(P("Throughput is extremely skewed: the median second carries no data, while the mean is 10.9 Mbps and the maximum 533 Mbps. A squared-error loss would chase these rare peaks, which is why an absolute-error loss was included in the tuning grid."),
  H3("Visual Analysis"));
add(figure("eda1", "Throughput by Application and Mobility", fig("eda1_throughput_by_group.png"),
  "Left: distribution of non-zero throughput (log scale; whiskers at 1.5 interquartile ranges, outliers not drawn). Right: share of seconds with zero throughput. Orange = stationary, blue = driving."));
add(P(F("eda1") + " shows that the three applications behave very differently. Downloads transfer continuously at tens to hundreds of megabits per second. Netflix is idle in 71%–77% of seconds and, when active, mostly sends small amounts; Amazon Prime sits between the two. Video throughput is therefore governed largely by the player’s buffer, not by the radio link."));
add(figure("eda3", "Network Mode and Logging Gaps", fig("eda3_network_mode_and_gaps.png"),
  "Left: share of seconds by recorded network mode. Right: time between consecutive records after merging repeated timestamps."));
add(P(F("eda3") + " shows that driving traces spent only about 57% of their time on 5G, against 92% for stationary traces, with the rest on LTE or 3G/2G, and that about 10% of consecutive records are two or more seconds apart. Both points matter for modeling: network mode changes the meaning of the radio fields, and gaps limit exact five-second targets."));
add(figure("eda2", "Distribution of Radio Measurements by Mobility", fig("eda2_radio_distributions.png"),
  "Normalized histograms of cleaned one-second values; central 99% shown for RSRP, RSRQ, and SNR."));
add(P(F("eda2") + " compares the radio measurements between conditions. Stationary traces cluster at a few RSRP levels because each session stays at one location, while driving spreads RSRP and SNR across a wide range; CQI is concentrated at 9–15 in both. The cluster of RSRQ near and above −3 dB in driving traces consists entirely of 3G-mode records."));
add(figure("eda4", "Throughput and CQI in Two Example Traces", fig("eda4_example_traces.png"),
  "Each panel shows 300 consecutive records of one trace of median length in its group."));
add(P(F("eda4") + " places throughput beside CQI for two typical traces. In the Netflix trace, throughput arrives in short bursts separated by idle periods while CQI stays high, so the two are almost unrelated. In the download trace, throughput follows the link more closely but still shows sudden jumps that CQI does not anticipate."),
  H3("EDA Insight Summary"),
  P(T("edaSummary") + " collects the main insight from each EDA table and figure, why it matters for the RQs, and the decision it led to."));
add(tbl("edaSummary", "EDA Insight Summary",
  ["Figure/Table", "What it shows", "Key insight", "Why it matters (RQ)", "Decision / next step"],
  [[T("overview"), "Traces, throughput, idle share, and 5G share by group", "Throughput differs by an order of magnitude between applications", "RQ4: absolute errors are not comparable across groups", "Report relative as well as absolute improvement"],
   [T("descriptive"), "Distribution and missingness of modeling variables", "Throughput is heavily skewed; SNR and CQI about 9% missing", "RQ2: squared-error losses chase rare peaks", "Tune an absolute-error loss; impute with flags"],
   [F("eda1"), "Throughput and idle seconds by application and mobility", "Video is idle 36%–77% of the time", "RQ1: weak radio association expected", "Test RQ1 within traces; check downloads separately"],
   [F("eda3"), "Network mode share and gaps between records", "Driving is on 5G only 57% of the time; 10% of records have gaps", "RQ3: network mode as context; exact targets", "Offer network mode; look up lags by exact time"],
   [F("eda2"), "Radio distributions by mobility", "Driving widens RSRP and SNR; RSRQ anomaly in 3G", "RQ1, RQ4: radio meaning varies by condition", "Keep the anomaly; add network mode to context"],
   [F("eda4"), "Throughput and CQI in two traces", "Throughput bursts that CQI does not anticipate", "RQ1, RQ2: history should dominate", "Compare with persistence and a moving average"]],
  [1.0, 1.6, 1.8, 1.6, 1.8]));
add(H3("Insights Linked to the Research Questions"),
  BUL("**RQ1:** because video throughput is set mainly by the player’s on/off pattern (" + F("eda1") + " and " + F("eda4") + "), within-trace correlations between radio measurements and throughput are expected to be weak, and stronger for downloads."),
  BUL("**RQ2:** strong persistence over seconds and long idle runs suggest that throughput history will dominate, and that any learned model must be compared with simple smoothing as well as persistence."),
  BUL("**RQ3:** application type and network mode separate very different throughput regimes (" + T("overview") + ", " + F("eda3") + "), so they may add predictive value, but their effect may be specific to the recorded sessions."),
  BUL("**RQ4:** absolute throughput, and therefore absolute error, differs by an order of magnitude between applications, so group comparisons need relative as well as absolute measures."));

// ===== Modeling =====
add(H1("Modeling"),
  H2("Problem Type and Baseline"),
  P("Predicting throughput five seconds ahead is a **supervised regression** problem on time-ordered data. The baseline is **persistence**, which predicts that throughput five seconds ahead equals current throughput; it is the standard reference in forecasting and is hard to beat over short horizons (Hyndman & Koehler, 2006). A 10-second moving average and an always-zero predictor were added after the primary results as reference checks."),
  H2("Choice of Models With Justification"),
  P("Three learned models were compared with the baseline, chosen to span increasing flexibility. " + T("models") + " summarizes them with the settings searched."),
  H3("Model 1: Ridge Regression"),
  P("Ridge regression is a regularized linear model. It is the transparent benchmark: if a weighted sum of the radio and history features were enough, the tree models would not be needed. Inputs are standardized and the penalty strength is tuned."),
  H3("Model 2: Random Forest"),
  P("A random forest averages many decision trees fitted on bootstrap samples (Breiman, 2001). It captures non-linear effects and interactions, such as a CQI drop that matters only at speed, needs little tuning, and handles numeric and one-hot inputs together."),
  H3("Model 3: Gradient Boosting"),
  P("Gradient boosting builds trees in sequence, each correcting the errors of the previous ones (Friedman, 2001). It is usually the strongest learner on tabular data, and its histogram-based implementation supports an absolute-error loss, which suits a heavy-tailed, zero-inflated target (Kaviani et al., 2026)."));
add(tbl("models", "Candidate Models, Rationale, and Tuning Grid", ["Model", "Why it was chosen", "Settings searched"],
  [["Persistence", "Required baseline; no training", "–"],
   ["Ridge regression", "Regularized linear benchmark; shows how much a linear model captures", "α ∈ {0.1, 1, 10, 100}"],
   ["Random forest (Breiman, 2001)", "Captures non-linear effects and interactions; robust with little tuning", "Minimum leaf size {5, 20} × maximum features {0.5, 1.0}; 200 trees"],
   ["Gradient boosting (Friedman, 2001)", "Usually strongest on tabular data; supports an absolute-error loss", "Loss {squared, absolute error} × learning rate {0.05, 0.1}; 300 iterations"]],
  [2.0, 3.6, 2.6], "All models were fitted with scikit-learn (Pedregosa et al., 2011). Gradient boosting uses the histogram-based implementation."));
add(P("Deep sequence models such as LSTM (Mei et al., 2019) were not used: with 83 traces the data are small for recurrent networks, tree models give directly interpretable feature importance, and the research questions concern which inputs matter rather than maximum accuracy. The models assume that the relationship between features and future throughput is stable across traces, that observations at or before the origin are sufficient, and that traces are independent; the Limitations section discusses where these assumptions are weak."),
  H2("Features Included and Feature Engineering"),
  P(T("features") + " lists every feature, whether it is original or engineered, its type, why it was included, and which models use it. The base feature set contains the radio measurements and throughput history; the context set adds speed, network mode, and application type. Throughput history was engineered because the literature and the EDA both point to it as the strongest short-term signal."));
add(tbl("features", "Feature Set and Usage",
  ["Feature", "Original/Engineered", "Type", "Reason for inclusion", "Used in model(s)"],
  [["RSRP, RSRQ, SNR, CQI", "Original", "Numeric (CQI ordinal)", "Radio conditions; RQ1 and radio-based literature", "Ridge, random forest, gradient boosting (base and context)"],
   ["throughput_lag_0", "Engineered from DL_bitrate", "Numeric (Mbps)", "Current throughput", "Persistence and all learned models"],
   ["throughput_lag_1, 2, 5", "Engineered", "Numeric (Mbps)", "Short-term dynamics", "All learned models"],
   ["throughput_mean_10s, throughput_sd_10s", "Engineered", "Numeric (Mbps)", "Level and volatility over 10 s (at least 7 observed seconds)", "All learned models"],
   ["Missing-value flags", "Engineered", "Binary", "Missingness itself can be informative", "All learned models"],
   ["Speed", "Original", "Numeric (km/h)", "Mobility; RQ3", "Context models"],
   ["NetworkMode", "Original", "Categorical (one-hot)", "Radio technology changes what the radio fields mean", "Context models"],
   ["application_type", "Engineered from folder names", "Categorical (one-hot, 3 levels)", "Traffic pattern differs by application", "Context models"],
   ["target_throughput_5s", "Engineered", "Numeric (Mbps)", "Throughput 5 s after the origin, same trace", "Prediction target"]],
  [2.0, 1.4, 1.3, 2.0, 1.7]));
add(P("Missing numeric values are imputed with the training-fold median, numeric inputs are standardized for ridge regression, and categorical inputs are one-hot encoded; all of these steps sit inside a single pipeline fitted on training rows only. Leakage is prevented by construction: every feature uses data at or before the prediction origin, targets are looked up within the same trace, coordinates and cell identifiers are excluded, and unit tests check timing and fold separation."),
  H2("Feature Importance"),
  P("Permutation importance measures how much MAE increases when one input is shuffled, using the tuned model of each fold on that fold’s held-out traces (" + F("importance") + ")."));
add(figure("importance", "Permutation Feature Importance of the Context Model", fig("fig8_feature_importance.png"),
  "Mean increase in MAE across the five outer folds when each input is randomly permuted on held-out traces; error bars show the standard deviation across folds. Missing-value flags are grouped."));
add(P("Throughput history dominates: shuffling the 10-second mean raises MAE by " + D.importance[0][1] + " Mbps and shuffling application type by " + D.importance[1][1] + " Mbps, while each radio measurement changes MAE by less than 0.002 Mbps."),
  H2("Evaluation Metrics and Formulae"),
  P(T("metrics") + " gives the formula, meaning, and use of each metric. In the formulae, *y*ᵢⱼ is observed throughput at origin *i* of trace *j*, *ŷ*ᵢⱼ the prediction, *n*ⱼ the number of origins in trace *j*, and *J* the number of traces."));
add(tbl("metrics", "Evaluation Metrics and Formulae", ["Metric", "Formula", "Interpretation", "Used for"],
  [["MAE of a trace", "MAEⱼ = (1/*n*ⱼ) Σᵢ |*y*ᵢⱼ − *ŷ*ᵢⱼ|", "Average absolute miss in Mbps; robust to rare peaks", "Primary metric"],
   ["Mean trace MAE", "MAE = (1/*J*) Σⱼ MAEⱼ", "Each trace counts equally, so long traces do not dominate", "Model comparison"],
   ["RMSE of a trace", "RMSEⱼ = √[(1/*n*ⱼ) Σᵢ (*y*ᵢⱼ − *ŷ*ᵢⱼ)²]", "Weights large misses more heavily", "Secondary metric"],
   ["Improvement", "Δⱼ = MAEⱼ(persistence) − MAEⱼ(model)", "Positive values mean the model is better", "RQ2; RQ3 with base minus context"],
   ["Relative improvement", "100 × Δⱼ / MAEⱼ(persistence)", "Comparable across groups with different throughput", "RQ4"],
   ["Spearman correlation", "ρⱼ = Pearson correlation of the ranks of a radio measure and throughput in trace *j*", "−1 to 1; 0 means no monotone association", "RQ1"],
   ["Bootstrap 95% CI", "2.5th and 97.5th percentiles of the mean of Δⱼ over 5,000 resamples of traces", "Uncertainty of the mean improvement", "RQ2–RQ4"]],
  [1.6, 3.0, 2.4, 1.4]));
const pers = parseFloat(E.persistence[0]), gbm = parseFloat(E.base_selected[0]);
add(P(`As a worked example, persistence has a mean trace MAE of ${E.persistence[0]} Mbps and the selected gradient-boosting model ${E.base_selected[0]} Mbps. The mean improvement is therefore ${E.persistence[0]} − ${E.base_selected[0]} = ${(pers - gbm).toFixed(2)} Mbps, a reduction of ${(pers - gbm).toFixed(2)} / ${E.persistence[0]} = ${(100 * (pers - gbm) / pers).toFixed(1)}% of the persistence error. The same model’s mean trace RMSE is ${E.base_selected[1]} Mbps against ${E.persistence[1]} Mbps for persistence.`),
  H2("Data Splitting and Validation"),
  P("Data were split by trace, never by record, so all seconds of a trace are either in training or in testing. Models are evaluated with **nested grouped cross-validation (CV)** (Roberts et al., 2017): five outer folds hold out complete traces, stratified by application and mobility, and three inner folds on the training traces select the model settings. The algorithm family with the lowest inner error is the selected AI/ML approach, and the context model uses the same family and search budget. Two sensitivity schemes repeat the procedure with whole recording dates held out and with the latest five recording dates held out. Uncertainty is expressed as percentile bootstrap intervals over traces, labeled approximate because they do not include retraining variability (Bengio & Grandvalet, 2004). RQ1 uses within-trace Spearman correlations with one-sample *t* tests across traces and Holm adjustment."),
  H2("Model Development"),
  P("Model development followed the same steps in every outer fold. The three model families were tuned on the inner folds of the training traces with the grids in " + T("models") + ", the family with the lowest inner MAE was refitted on all training traces, and its predictions on the held-out traces were saved; all five folds selected gradient boosting. The context model was then tuned with the same family and budget. The code is a Python 3.13 package built on pandas 2.2, NumPy 2.3, and scikit-learn 1.7. It ran on the central processing unit (CPU) of a laptop with an Intel Core i7-1165G7 processor and 24 GB of memory; no graphics processing unit (GPU) was used, because scikit-learn’s tree models run on the CPU. Training and evaluation took 18 minutes for the primary scheme, 20 minutes with recording dates held out, and 5 minutes for the chronological scheme."));

// ===== Preliminary results =====
add(H1("Results"),
  P("This section reports the result for each RQ with its hypothesis test at α = .05, then compares all models on training and test data."),
  H2("Findings by Research Question"),
  H3("RQ1: Radio Measurements and Throughput"),
  P("No radio measurement was associated with contemporaneous throughput at α = .05 after Holm adjustment (smallest adjusted *p* = .050, for SNR), so *H*₀₁ was not rejected (" + T("rq1") + ", " + F("rq1") + "). Mean within-trace correlations were between −.01 and .03; SNR came closest to significance. Correlations were somewhat higher in download traces (.11 to .18) than in video traces, consistent with the EDA."));
add(tbl("rq1", "Mean Within-Trace Spearman Correlation With Throughput",
  ["Measurement", "Traces", "Mean ρ", "95% CI", "Holm *p*", "Date-level Holm *p*"], D.rq1, [1.4, 0.9, 0.9, 1.6, 0.9, 1.3],
  "Confidence intervals are *t* intervals across traces. The date-level test first averages traces within each of 27 recording dates."));
add(figure("rq1", "Within-Trace Correlation Between Radio Measurements and Throughput", fig("fig1_rq1_correlations.png"),
  "Each dot is one trace (blue = driving, orange = stationary); diamonds show the mean with a 95% *t* interval."));
add(H3("RQ2: AI/ML Compared With Persistence"),
  P(`Gradient boosting with an absolute-error loss had the lowest inner error in all five folds. It reduced mean trace MAE from ${s2.pers} to ${s2.model} Mbps, an improvement of ${s2.delta} Mbps, 95% CI ${s2.ci}, *p* ${PV.outer_fold.rq2} (one-sided bootstrap), and was more accurate than persistence on all 83 traces, so *H*₀₂ was rejected (${F("rq2")}; all models are compared in the Model Performance Comparison section). Resampling whole recording dates gave ${s2.date_ci}. Against the 10-second moving average the selected model still improved MAE by ${D.rq2_vs_mean.delta} Mbps ${D.rq2_vs_mean.ci} and was better on ${D.rq2_vs_mean.share}% of traces. Predicting zero for every Netflix origin was slightly more accurate than the base model, showing that part of the gain reflects the player’s idle periods (${F("example")}).`));
add(figure("rq2", "Reduction in Prediction Error Relative to Persistence", fig("fig2_rq2_improvement.png"),
  "Mean reduction in trace-level MAE with approximate 95% bootstrap intervals; gray rows are reference baselines."));
add(figure("example", "Observed and Predicted Throughput on a Held-Out Driving Download Trace", fig("fig6_example_trace.png"),
  "Persistence repeats the value from 5 s earlier and lags every change; the absolute-error model predicts between burst and idle levels, lowering average error but underestimating peaks."));
add(H3("RQ3: Added Value of Context"),
  P(`Adding speed, network mode, and application type reduced MAE by a further ${s3.delta} Mbps, 95% CI ${s3.ci}, *p* ${PV.outer_fold.rq3}, about 2% of the base error, and improved ${s3.share}% of traces, so *H*₀₃ was rejected in the primary analysis (${F("rq3")}). The benefit did not survive the sensitivity analyses (*p* ${PV.date_fold.rq3} with whole recording days held out; see Sensitivity Analyses).`));
add(figure("rq3", "Per-Trace Benefit of Adding Context Features", fig("fig3_rq3_context_benefit.png"),
  "Base-model MAE minus context-model MAE per trace; the dashed line marks the mean."));
add(H3("RQ4: Mobility and Application Conditions"),
  P(`The improvement over persistence did not differ clearly between stationary and driving traces (difference ${D.rq4_diff["Driving - Static"]} Mbps; the 95% interval includes zero, so *p* > .05), so *H*₀₄ₐ was not rejected; relative improvements were about 30% in both groups (${T("rq4")}, ${F("rq4")}). In the exploratory application comparison, downloads gained more in absolute terms (Amazon Prime minus Download ${D.rq4_diff["Amazon Prime - Download"]} Mbps, 98.3% Bonferroni interval), but the ordering reverses in relative terms.`));
add(tbl("rq4", "Improvement of the Context Model Over Persistence by Group",
  ["Group", "Traces", "Persistence MAE", "Model MAE", "Improvement (Mbps) [95% CI]", "Median relative improvement"],
  D.rq4, [1.4, 0.8, 1.1, 1.0, 2.0, 1.5]));
add(figure("rq4", "Relative Error Reduction by Mobility and Application", fig("fig4_rq4_groups.png"),
  "Per-trace MAE reduction relative to persistence; horizontal lines mark group medians."));
const imp = Object.fromEntries(D.rq2.map((r) => [r[0], r[3]]));
add(H2("Model Performance Comparison"),
  P(T("perf") + " compares every model on the same held-out traces. **The best model is gradient boosting with an absolute-error loss:** it had the lowest MAE of the planned models and beat persistence on every trace; the random forest had the lowest RMSE because it was tuned with a squared-error loss, which favors fitting the peaks. Models tuned on squared error beat persistence on only 25%–37% of traces, so the loss function mattered as much as the algorithm."));
add(tbl("perf", "Model Performance Comparison",
  ["Model", "Feature set", "Validation approach", "Mean trace MAE (Mbps)", "Mean trace RMSE (Mbps)", "Improvement over persistence (Mbps) [95% CI]", "Key takeaway"],
  [["Persistence", "Current throughput", "Held-out traces", E.persistence[0], E.persistence[1], "–", "Baseline"],
   ["Ridge regression", "Base", "Nested grouped CV", E.base_ridge[0], E.base_ridge[1], imp["Ridge regression"], "Linear model gains little"],
   ["Random forest", "Base", "Nested grouped CV", E.base_rf[0], E.base_rf[1], imp["Random forest"], "Lowest RMSE; better on 37% of traces"],
   ["**Gradient boosting (selected, best)**", "Base", "Nested grouped CV", E.base_selected[0], E.base_selected[1], imp["Gradient boosting (selected)"], "Best model: lowest base MAE; better on all traces"],
   ["Gradient boosting", "Context", "Nested grouped CV", E.context_selected[0], E.context_selected[1], `${D.rq2_ctx.delta} ${D.rq2_ctx.ci}`, `Small further gain; not robust (${T("sens")})`],
   ["Reference: 10 s mean", "Throughput history", "Held-out traces", E.ref_mean_10s[0], E.ref_mean_10s[1], imp["Reference: 10 s mean"], "Simple smoothing captures part of the gain"],
   ["Reference: always 0", "None", "Held-out traces", E.ref_zero[0], E.ref_zero[1], imp["Reference: always 0"], "Worse than persistence overall"]],
  [1.4, 1.0, 1.1, 0.9, 0.9, 1.4, 1.6],
  "*N* = 83 held-out traces. Nested grouped CV = five outer folds of whole traces and three inner folds for tuning. Reference rows were added after the primary results, are not part of the planned tests, and need no training."));
add(H2("Training and Test Results"),
  P(T("traintest") + " separates the error each model makes on the training traces it was fitted on from its error on the held-out test traces of the same fold. Persistence and ridge regression show no gap, gradient boosting a small gap of " + TT.base_hgb[4] + " Mbps, and the random forest the largest gap (" + TT.base_rf[4] + " Mbps), which shows that it fits the training traces more closely than it generalizes. The selected model therefore does not overfit to a degree that would change its ranking."));
add(tbl("traintest", "Training and Test Error by Model",
  ["Model", "Training MAE (Mbps)", "Test MAE (Mbps)", "Training RMSE (Mbps)", "Test RMSE (Mbps)", "Test − training MAE"],
  [["Persistence", ...TT.persistence],
   ["Ridge regression", ...TT.base_ridge],
   ["Random forest", ...TT.base_rf],
   ["Gradient boosting (selected)", ...TT.base_hgb],
   ["Gradient boosting + context", ...TT.context_selected]],
  [2.2, 1.1, 1.1, 1.1, 1.1, 1.2],
  "Means over the five outer folds of the trace-averaged error on the training traces of each fold and on its held-out test traces. Because these are fold averages, test MAE differs slightly from the overall trace mean in " + T("perf") + ". Persistence needs no training; its two columns compare the same rule on the two sets of traces."));
add(H2("Visual Evaluation"),
  P("Because the task is regression, the equivalent of a receiver operating characteristic curve or confusion matrix is a plot of predicted against observed values. " + F("predobs") + " shows this for all " + E.rows + " held-out prediction origins. Persistence scatters widely around the diagonal: it predicts a burst when the player has already gone idle and zero when a burst starts, which forms the bands along both axes at zero. Gradient boosting pulls mid-range predictions toward the diagonal and rarely predicts above about 300 Mbps, so it underestimates the largest peaks, and it still misses sudden starts and stops. Its row-level MAE is " + E.row_sel + " Mbps against " + E.row_pers + " Mbps for persistence."));
add(figure("predobs", "Predicted Versus Observed Throughput on Held-Out Traces", fig("fig10_predicted_vs_observed.png"),
  "Each hexagon counts held-out prediction origins (log color scale); axes use a log(1 + x) scale. The dashed line marks perfect prediction."));
add(H2("Sensitivity Analyses"),
  P(T("sens") + " and " + F("sens") + " repeat the key results under two stricter evaluation schemes. The RQ2 improvement held in every scheme; the RQ3 benefit vanished when whole days were held out and reversed in the chronological split, where the test set contained 14 download traces and no Netflix traces; RQ4a remained inconclusive."));
add(tbl("sens", "Key Results Under Three Evaluation Schemes",
  ["Scheme", "Test traces", "RQ2 improvement (Mbps)", "RQ3 context benefit (Mbps)", "RQ4a driving − static (Mbps)"],
  D.sensitivity, [2.0, 1.0, 1.8, 1.8, 1.8], "Values in brackets are approximate 95% bootstrap intervals over traces."));
add(figure("sens", "Key Results Under Three Evaluation Schemes", fig("fig7_sensitivity.png"),
  "Blue marks the primary analysis. Absolute values in the chronological scheme are not comparable because its test set is mostly downloads."));
add(H2("Summary of Hypothesis Tests"), P(T("hyp") + " summarizes the decision for each hypothesis and the evidence behind it."));
add(tbl("hyp", "Hypothesis Decisions at α = .05", ["Hypothesis", "Decision", "Evidence"],
  [["*H*₀₁: no radio–throughput association", "Not rejected", "Mean ρ −.01 to .03; Holm *p* = .050 to .853, none below .05"],
   ["*H*₀₂: AI/ML no better than persistence", "Rejected", `*M*Δ = ${s2.delta} Mbps, 95% CI ${s2.ci}, *p* ${PV.outer_fold.rq2}; *p* ${PV.date_fold.rq2} with days held out`],
   ["*H*₀₃: context adds no benefit", "Rejected (primary only)", `*M*Δ = ${s3.delta} Mbps, 95% CI ${s3.ci}, *p* ${PV.outer_fold.rq3}; *p* ${PV.date_fold.rq3} with days held out`],
   ["*H*₀₄ₐ: equal improvement, static vs. driving", "Not rejected", `Difference ${D.rq4_diff["Driving - Static"]} Mbps; 95% interval includes 0`],
   ["*H*₀₄b: equal improvement across applications", "Rejected (exploratory)", "Prime − Download and Download − Netflix intervals exclude 0; reverses in relative terms"]],
  [3.0, 1.8, 3.4]));
// ===== Deployment (proposed) =====
add(H1("Deployment and Integration"),
  P("The model has not been deployed. This section describes a proposed deployment so that the results can be judged against a realistic use; none of it has been built or tested."),
  H2("Deployment Approach"),
  P("The selected model would be saved together with its preprocessing pipeline as a single file and served by a small prediction service. Every second the service receives the latest radio and throughput measurements of a session, keeps the last 10 seconds in memory, builds the same features as in training, and returns the predicted throughput five seconds ahead. On the laptop used here, one prediction took a median of 7.5 ms and refitting the model on about 137,000 origins took 9 s, so the service needs no GPU or cloud computing. The model would be retrained periodically on new logs, and its rolling MAE against persistence would be monitored so that the service falls back to persistence whenever it stops winning, which the RQ3 sensitivity results show can happen on new recording days."),
  H2("System Integration"),
  P("The service sits between a data source and a consumer. On the source side it reads the fields this dataset records, namely device-side radio measurements and application throughput, from a phone logging application, a network monitoring feed, or a test bench, through a simple interface such as an HTTP endpoint or a local socket. On the consumer side, a video player’s bitrate logic (Yin et al., 2015), a monitoring dashboard, or a test controller calls it once per second. New logs are stored in the SQLite database through the same ingestion code, so retraining reuses the pipeline in " + F("pipeline") + ". Later versions could place an LLM agent in front of the service to answer engineers’ questions in natural language, in the spirit of CommLLM (Jiang et al., 2024), but this study does not test that."),
  H2("Benefits to Users"),
  P("The study measured prediction accuracy, not user experience, so the benefits below are expected rather than demonstrated. A player that sees a drop five seconds ahead can lower its bitrate before its buffer empties, which reduces stalls (Raca, Zahran, et al., 2020), so viewers would see fewer pauses and fewer sudden quality changes. Network and test engineers would get an early warning of throughput loss and a measured baseline to beat; because history dominates and radio inputs add little at one-second resolution, they can direct data collection toward throughput history and toward sub-second or network-side measurements."),
  H2("Example Use Case"),
  P("Consider a passenger streaming a series in a car, as in the driving Netflix traces. Every second, the phone’s player sends the last 10 seconds of throughput and radio values to the service. When throughput has been falling for several seconds, the model predicts a lower level five seconds ahead while persistence still assumes the current level, and the player switches to a lower bitrate in time instead of stalling when its buffer runs out. On the held-out driving download trace in " + F("example") + ", the model followed changes in the throughput level more closely than persistence, which by definition lags every change by five seconds."));

// ===== Limitations =====
add(H1("Limitations"),
  H2("Data Limitations"),
  BUL("**Sample:** one device, one operator, and one country in 2019–2020; 83 traces, below the 138 planned, which slightly widens the RQ1 confidence intervals and limits the application comparison (RQ4b) to exploratory status (" + T("precision") + ")."),
  BUL("**Measurement:** the 5G service was most likely non-standalone, so radio fields may describe the LTE anchor; values are device-side and recorded once per second; in 3G modes the RSRQ field carries a different quantity."),
  BUL("**Target:** application throughput reflects application demand, such as a video player’s idle periods, as well as link capacity."),
  H2("Model Limitations"),
  BUL("**Tuning and inputs:** limited tuning grids; application type is known exactly from folder names, so its value is an upper bound."),
  BUL("**Stability:** relationships may change across recording days, as the RQ3 sensitivity results show."),
  BUL("**Uncertainty:** bootstrap intervals do not include retraining variability and are likely somewhat narrow; the reference baselines and the relaxed eligibility rule were decided after the synopsis."),
  H2("Impact on Findings and Generalizability"),
  P("Together these limits mean the findings describe short-term prediction from one-second, device-side logs of one phone on one non-standalone 5G network. The main result, that a model built on throughput history beats persistence by about 18.5%, is likely to hold in similar settings because it held under every evaluation scheme. The null radio result (RQ1) and the weak context result (RQ3) should not be generalized to sub-second, network-side, or standalone 5G data, where the literature reports larger radio-side gains. Improved prediction accuracy also does not by itself demonstrate improved CSI feedback, network throughput, or user experience."));

// ===== Conclusions and future scope =====
add(H1("Conclusions and Future Scope"),
  P("Short-term 5G throughput is predictable mainly from its own recent history. Radio measurements were not associated with throughput within traces (RQ1). Gradient boosting with an absolute-error loss beat persistence by " + s2.delta + " Mbps on all 83 held-out traces, also with whole recording days held out (RQ2). Context features helped only when test traces shared recording days with training traces (RQ3), and the gain did not differ between stationary and driving conditions (RQ4). The practical message is that a simple, well-validated model on throughput history already captures most of the predictable signal in one-second device logs, and that the choice of loss function mattered as much as the choice of algorithm."),
  H2("Future Scope"),
  BUL("Collect sub-second and network-side measurements, such as cell load and competing users, which the literature links to the largest radio-side gains."),
  BUL("Test generalization on standalone 5G and on more devices, operators, and countries."),
  BUL("Compare DL sequence models and LLM-based or agentic approaches (Jiang et al., 2024; Wu et al., 2024) once a larger dataset is available."),
  BUL("Evaluate other horizons (1 to 20 seconds) and probabilistic forecasts with prediction intervals."),
  BUL("Run a live trial of the proposed service with a video player and measure stalls and bitrate, to test the expected user benefit."),
  BUL("Add online retraining and drift monitoring across recording days, given the RQ3 result."));

// References
const REF = [
  "Bengio, Y., & Grandvalet, Y. (2004). No unbiased estimator of the variance of k-fold cross-validation. *Journal of Machine Learning Research, 5*, 1089–1105. https://www.jmlr.org/papers/v5/grandvalet04a.html",
  "Biernacki, A. (2024). Throughput prediction of 5G network based on trace similarity for adaptive video. *Applied Sciences, 14*(5), Article 1962. https://doi.org/10.3390/app14051962",
  "Biswas, M., Chakraborty, A., & Palit, B. (2023). *A Kalman filter based low complexity throughput prediction algorithm for 5G cellular networks* [Preprint]. arXiv. https://doi.org/10.48550/arXiv.2307.04819",
  "Breiman, L. (2001). Random forests. *Machine Learning, 45*, 5–32. https://doi.org/10.1023/A:1010933404324",
  "Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *The Annals of Statistics, 29*(5), 1189–1232. https://doi.org/10.1214/aos/1013203451",
  "Hyndman, R. J., & Koehler, A. B. (2006). Another look at measures of forecast accuracy. *International Journal of Forecasting, 22*(4), 679–688. https://doi.org/10.1016/j.ijforecast.2006.03.001",
  "Jiang, F., Peng, Y., Dong, L., Wang, K., Yang, K., Pan, C., Niyato, D., & Dobre, O. A. (2024). Large language model enhanced multi-agent systems for 6G communications. *IEEE Wireless Communications, 31*(6), 48–55. https://doi.org/10.1109/MWC.016.2300600",
  "Kaviani, M., Almeida, J., & Verdi, F. L. (2026). RAHL: A residual-based adaptive Huber loss for robust CQI prediction in 5G networks. *Journal of Network and Systems Management, 34*(1), Article 28. https://doi.org/10.1007/s10922-025-10002-9",
  "Lakens, D. (2022). Sample size justification. *Collabra: Psychology, 8*(1), Article 33267. https://doi.org/10.1525/collabra.33267",
  "Mao, H., Netravali, R., & Alizadeh, M. (2017). Neural adaptive video streaming with Pensieve. In *Proceedings of the Conference of the ACM Special Interest Group on Data Communication* (pp. 197–210). Association for Computing Machinery. https://doi.org/10.1145/3098822.3098843",
  "Mei, L., Hu, R., Cao, H., Liu, Y., Han, Z., Li, F., & Li, J. (2019). Realtime mobile bandwidth prediction using LSTM neural network. In *Passive and active measurement* (Lecture Notes in Computer Science, Vol. 11419, pp. 34–47). Springer. https://doi.org/10.1007/978-3-030-15986-3_3",
  "Minovski, D., Ögren, N., Åhlund, C., & Mitra, K. (2021). Throughput prediction using machine learning in LTE and 5G networks. *IEEE Transactions on Mobile Computing*. Advance online publication. https://doi.org/10.1109/TMC.2021.3099397",
  "Narayanan, A., Ramadan, E., Mehta, R., Hu, X., Liu, Q., Fezeu, R. A. K., Dayalan, U. K., Verma, S., Ji, P., Li, T., Qian, F., & Zhang, Z.-L. (2020). Lumos5G: Mapping and predicting commercial mmWave 5G throughput. In *Proceedings of the ACM Internet Measurement Conference* (pp. 176–193). Association for Computing Machinery. https://doi.org/10.1145/3419394.3423629",
  "Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830. https://www.jmlr.org/papers/v12/pedregosa11a.html",
  "Raca, D., Leahy, D., Sreenan, C. J., & Quinlan, J. J. (2020). Beyond throughput, the next generation: A 5G dataset with channel and context metrics. In *Proceedings of the 11th ACM Multimedia Systems Conference* (pp. 303–308). Association for Computing Machinery. https://doi.org/10.1145/3339825.3394938",
  "Raca, D., Quinlan, J. J., Zahran, A. H., & Sreenan, C. J. (2018). Beyond throughput: A 4G LTE dataset with channel and context metrics. In *Proceedings of the 9th ACM Multimedia Systems Conference* (pp. 460–465). Association for Computing Machinery. https://doi.org/10.1145/3204949.3208123",
  "Raca, D., Zahran, A. H., Sreenan, C. J., Sinha, R. K., Halepovic, E., Jana, R., & Gopalakrishnan, V. (2020). On leveraging machine and deep learning for throughput prediction in cellular networks: Design, performance, and challenges. *IEEE Communications Magazine, 58*(3), 11–17. https://doi.org/10.1109/MCOM.001.1900394",
  "Roberts, D. R., Bahn, V., Ciuti, S., Boyce, M. S., Elith, J., Guillera-Arroita, G., Hauenstein, S., Lahoz-Monfort, J. J., Schröder, B., Thuiller, W., Warton, D. I., Wintle, B. A., Hartig, F., & Dormann, C. F. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography, 40*(8), 913–929. https://doi.org/10.1111/ecog.02881",
  "Sun, Y., Yin, X., Jiang, J., Sekar, V., Lin, F., Wang, N., Liu, T., & Sinopoli, B. (2016). CS2P: Improving video bitrate selection and adaptation with data-driven throughput prediction. In *Proceedings of the 2016 ACM SIGCOMM Conference* (pp. 272–285). Association for Computing Machinery. https://doi.org/10.1145/2934872.2934898",
  "Wu, D., Wang, X., Qiao, Y., Wang, Z., Jiang, J., Cui, S., & Wang, F. (2024). NetLLM: Adapting large language models for networking. In *Proceedings of the ACM SIGCOMM 2024 Conference* (pp. 661–678). Association for Computing Machinery. https://doi.org/10.1145/3651890.3672268",
  "Ye, W., Hu, X., Sleder, S., Zhang, A., Dayalan, U. K., Hassan, A., Fezeu, R. A. K., Jajoo, A., Lee, M., Ramadan, E., Qian, F., & Zhang, Z.-L. (2024). Dissecting carrier aggregation in 5G networks: Measurement, QoE implications and prediction. In *Proceedings of the ACM SIGCOMM 2024 Conference* (pp. 340–357). Association for Computing Machinery. https://doi.org/10.1145/3651890.3672250",
  "Yin, X., Jindal, A., Sekar, V., & Sinopoli, B. (2015). A control-theoretic approach for dynamic adaptive video streaming over HTTP. In *Proceedings of the 2015 ACM Conference on Special Interest Group on Data Communication* (pp. 325–338). Association for Computing Machinery. https://doi.org/10.1145/2785956.2787486",
  "Yue, C., Jin, R., Suh, K., Qin, Y., Wang, B., & Wei, W. (2018). LinkForecast: Cellular link bandwidth prediction in LTE networks. *IEEE Transactions on Mobile Computing, 17*(7), 1582–1594. https://doi.org/10.1109/TMC.2017.2756937",
];
add(BREAK(), H1("References"),
  ...REF.map((r) => new Paragraph({ children: runs(r), spacing: DOUBLE, indent: { left: 720, hanging: 720 } })));

// Appendix (APA: new page, label and title centered and bold)
add(BREAK(), H1("Appendix A"), center("Code and Workflow", true),
  PN("The scripts below are run in order from the analysis folder; each regenerates the outputs named. The random seed is 640 throughout."),
  BUL("**run_pipeline.py:** verifies the archive checksum, cleans the raw files, and builds features and folds (data/interim/, data/processed/)."),
  BUL("**run_rq1.py:** within-trace correlations and tests for RQ1 (" + T("rq1") + ", " + F("rq1") + ")."),
  BUL("**run_models.py:** nested grouped cross-validation for one evaluation scheme (outputs/models/); run once per scheme."),
  BUL("**analyze_models.py:** errors, improvements, and bootstrap intervals for RQ2–RQ4 (" + T("perf") + ", " + T("rq4") + ", " + T("sens") + ")."),
  BUL("**run_importance.py:** permutation feature importance (" + F("importance") + ")."),
  BUL("**run_train_errors.py:** training-fold versus test-fold error of every model (" + T("traintest") + ")."),
  BUL("**build_database.py:** loads the cleaned data into the SQLite database (" + T("db") + ")."),
  BUL("**make_eda.py**, **make_figures.py**, **make_preview.py**, and **make_final_figures.py:** EDA tables and all figures."),
  BUL("**report/build_report_data.py** and **report/build_report.js:** assemble this report from the saved outputs."),
  BUL("**Capstone_5G_Throughput_Colab.ipynb:** reproduces the full workflow in Google Colab."));
if (tableNo !== TAB_ORDER.length || figNo !== FIG_ORDER.length) throw new Error(`placed ${tableNo} tables and ${figNo} figures`);

// ---------- document ----------
const doc = new Document({
  creator: "Lokesh Rajoria", title: "AI/ML-Based Prediction of 5G Downlink Throughput - Final Report",
  features: { updateFields: true },
  styles: {
    default: { document: { run: { font: FONT, size: 24 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 24, bold: true, color: "000000" }, paragraph: { alignment: AlignmentType.CENTER, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 24, bold: true, color: "000000" }, paragraph: { outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 24, bold: true, italics: true, color: "000000" }, paragraph: { outlineLevel: 2 } },
      { id: "TOC1", name: "toc 1", basedOn: "Normal", next: "Normal",
        run: { font: FONT, size: 24 }, paragraph: { spacing: { before: 0, after: 0, line: 260 } } },
      { id: "TOC2", name: "toc 2", basedOn: "Normal", next: "Normal",
        run: { font: FONT, size: 24 }, paragraph: { spacing: { before: 0, after: 0, line: 260 }, indent: { left: 360 } } },
    ],
  },
  numbering: { config: [
    { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
    { reference: "steps", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
  ] },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
    headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT,
      children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 24 })] })] }) },
    children: S,
  }],
});
Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(OUT, buf); console.log("wrote", OUT, `(${tableNo} tables, ${figNo} figures)`); });
