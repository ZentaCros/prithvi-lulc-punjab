/* ═══════════════════════════════════════════════════════════════
   LULC DASHBOARD — APPLICATION LOGIC
   v2: Image preloading, progress bar, auto-retry, memory cleanup
   ═══════════════════════════════════════════════════════════════ */

// ─── DATA ──────────────────────────────────────────────────────

const BOUNDS = {
    lahore: {
        south: 31.197574693647738,  west: 74.09511449763578,
        north: 31.702420353581598, east: 74.55242942457797,
    },
    gujranwala: {
        south: 31.907053619803197,  west: 73.94921941181867,
        north: 32.41299282185118,  east: 74.4083481382861,
    },
};

// Paths to the colored prediction PNGs (Prithvi = best model)
const OVERLAY_PATHS = {
    lahore: {
        "2018": "../results/pipeline/lahore/web_lahore_2018.png",
        "2021": "../results/pipeline/lahore/web_lahore_2021.png",
        "2026": "../results/pipeline/lahore/web_lahore_2026.png",
    },
    gujranwala: {
        "2018": "../results/pipeline/gujranwala/web_gujranwala_2018.png",
        "2021": "../results/pipeline/gujranwala/web_gujranwala_2021.png",
        "2026": "../results/pipeline/gujranwala/web_gujranwala_2026.png",
    },
};

const BASEMAP_PATHS = {
    lahore: {
        "2018": "../results/pipeline/lahore/web_basemap_lahore_2018.webp",
        "2021": "../results/pipeline/lahore/web_basemap_lahore_2018.webp", // Fallback to 2018 for intermediate
        "2026": "../results/pipeline/lahore/web_basemap_lahore_2026.webp",
    },
    gujranwala: {
        "2018": "../results/pipeline/gujranwala/web_basemap_gujranwala_2018.webp",
        "2021": "../results/pipeline/gujranwala/web_basemap_gujranwala_2018.webp",
        "2026": "../results/pipeline/gujranwala/web_basemap_gujranwala_2026.webp",
    },
};

// Change detection (Ensemble results from inference pipeline)
const CHANGE_DATA = {
    lahore: {
        "Vegetation":   { area_2018: 793.23, area_2021: 771.0,  area_2026: 751.59, net: -41.64 },
        "Built-up":     { area_2018: 500.35, area_2021: 553.0,  area_2026: 606.97, net: 106.62 },
        "Cropland":     { area_2018: 889.74, area_2021: 910.0,  area_2026: 928.38, net:  38.64 },
        "Water":        { area_2018:  21.21, area_2021:  22.7,   area_2026:  24.20, net:   2.99 },
        "Barren/Other": { area_2018: 196.27, area_2021: 144.0,  area_2026:  89.66, net:-106.61 },
    },
    gujranwala: {
        "Vegetation":   { area_2018: 616.27, area_2021: 607.0,  area_2026: 598.13, net: -18.14 },
        "Built-up":     { area_2018: 188.84, area_2021: 207.0,  area_2026: 225.66, net:  36.82 },
        "Cropland":     { area_2018:1540.84, area_2021:1540.0,  area_2026:1539.08, net:  -1.76 },
        "Water":        { area_2018:   7.95, area_2021:   7.3,   area_2026:   6.73, net:  -1.22 },
        "Barren/Other": { area_2018:  36.54, area_2021:  28.7,   area_2026:  20.83, net: -15.70 },
    },
};

// Full-city evaluation metrics (from pipeline evaluation JSONs, best seed)
const MODEL_METRICS = {
    lahore: {
        "U-Net (B4)":    { acc: 68.92, f1: 64.84, miou: 49.68 },
        "SegFormer":     { acc: 68.83, f1: 64.83, miou: 49.72 },
        "DeepLabV3+":    { acc: 69.05, f1: 65.24, miou: 50.14 },
        "Prithvi v2":    { acc: 70.66, f1: 66.71, miou: 51.66 },
        "Ensemble":      { acc: 70.37, f1: 66.51, miou: 51.47 },
    },
    gujranwala: {
        "U-Net (B4)":    { acc: 74.78, f1: 58.90, miou: 44.75 },
        "SegFormer":     { acc: 74.84, f1: 59.26, miou: 45.14 },
        "DeepLabV3+":    { acc: 74.82, f1: 58.18, miou: 44.34 },
        "Prithvi v2":    { acc: 75.45, f1: 60.69, miou: 46.60 },
        "Ensemble":      { acc: 75.48, f1: 60.65, miou: 46.52 },
    },
};

// Patch-level metrics (mean ± std over 3 seeds: 42, 123, 7)
const PATCH_METRICS = {
    "U-Net (B4)":    { miou: 59.16, std: 0.07, oa: 81.18, oa_std: 0.09 },
    "MiT-B2+FPN":    { miou: 59.07, std: 0.07, oa: 80.61, oa_std: 0.12 },
    "DeepLabV3+":    { miou: 59.05, std: 0.17, oa: 80.81, oa_std: 0.06 },
    "Prithvi v2":    { miou: 61.19, std: 0.25, oa: 82.75, oa_std: 0.24 },
};

// LoRA rank ablation (seed 42, 8 unfrozen blocks)
const LORA_ABLATION = [
    { rank:  8, trainable: "106.7M", pct: 34.6, miou: 60.40, oa: 82.07 },
    { rank: 16, trainable: "109.6M", pct: 35.2, miou: 60.67, oa: 82.63 },
    { rank: 32, trainable: "115.4M", pct: 36.4, miou: 61.12, oa: 82.62 },
    { rank: 64, trainable: "121.4M", pct: 37.6, miou: 61.19, oa: 82.70 },
];

const CLASS_COLORS = {
    "Vegetation":   "#22c55e",
    "Built-up":     "#ef4444",
    "Cropland":     "#eab308",
    "Water":        "#3b82f6",
    "Barren/Other": "#d2b48c",
};


// ─── STATE ─────────────────────────────────────────────────────

let currentCity = "lahore";
let currentYear = "2026";
let map, overlay, baseTile;
let donutChart = null;
let barChart   = null;


// ─── IMAGE CACHE & PRELOADER ───────────────────────────────────

const imageCache = {};  // url -> HTMLImageElement (persists across city/year switches)

/**
 * Load a single image with auto-retry on failure.
 * Returns a Promise that resolves with the cached Image element.
 * Retries up to maxRetries times with exponential backoff.
 */
function loadImageWithRetry(url, maxRetries = 5) {
    // Return cached image immediately if already loaded
    if (imageCache[url] && imageCache[url].complete && imageCache[url].naturalWidth > 0) {
        return Promise.resolve(imageCache[url]);
    }

    return new Promise((resolve) => {
        let attempt = 0;

        function tryLoad() {
            attempt++;
            const img = new Image();

            img.onload = () => {
                imageCache[url] = img;
                resolve(img);
            };

            img.onerror = () => {
                if (attempt < maxRetries) {
                    const delay = Math.min(1000 * Math.pow(2, attempt - 1), 8000); // 1s, 2s, 4s, 8s, 8s
                    updateLoaderStatus(`Retrying ${getFileName(url)}... (attempt ${attempt + 1})`);
                    setTimeout(tryLoad, delay);
                } else {
                    // After all retries, resolve anyway to not block the app
                    console.warn(`Failed to load after ${maxRetries} attempts: ${url}`);
                    imageCache[url] = img; // store even failed so we don't re-block
                    resolve(img);
                }
            };

            // Cache-bust on retries to avoid stale cached errors
            img.src = attempt > 1 ? `${url}?retry=${attempt}` : url;
        }

        tryLoad();
    });
}

/**
 * Extract file name from URL for status display
 */
function getFileName(url) {
    return url.split("/").pop().split("?")[0];
}

/**
 * Update the loading overlay progress bar and status text
 */
function updateLoaderProgress(loaded, total) {
    const pct = Math.round((loaded / total) * 100);
    const fill = document.getElementById("progress-fill");
    const status = document.getElementById("loader-status");
    if (fill) fill.style.width = pct + "%";
    if (status) status.textContent = `Loading imagery... ${loaded} of ${total} (${pct}%)`;
}

function updateLoaderStatus(text) {
    const status = document.getElementById("loader-status");
    if (status) status.textContent = text;
}

/**
 * Collect all unique image URLs that the dashboard needs
 */
function getAllImageUrls() {
    const urls = new Set();
    for (const city of Object.keys(OVERLAY_PATHS)) {
        for (const year of Object.keys(OVERLAY_PATHS[city])) {
            urls.add(OVERLAY_PATHS[city][year]);
        }
    }
    for (const city of Object.keys(BASEMAP_PATHS)) {
        for (const year of Object.keys(BASEMAP_PATHS[city])) {
            urls.add(BASEMAP_PATHS[city][year]);
        }
    }
    return Array.from(urls);
}

/**
 * Preload all images with real progress tracking.
 * Returns a Promise that resolves when all images are loaded.
 */
async function preloadAllImages() {
    const urls = getAllImageUrls();
    const total = urls.length;
    let loaded = 0;

    updateLoaderProgress(0, total);

    const promises = urls.map(url =>
        loadImageWithRetry(url).then(() => {
            loaded++;
            updateLoaderProgress(loaded, total);
        })
    );

    await Promise.all(promises);

    // Final status
    updateLoaderStatus("Ready!");
}

/**
 * Dismiss the loading overlay with a smooth fade
 */
function hideLoader() {
    const overlay = document.getElementById("loading-overlay");
    if (overlay) {
        overlay.classList.add("hidden");
        // Remove from DOM after animation completes to free memory
        setTimeout(() => overlay.remove(), 700);
    }
}


// ─── MAP INIT ──────────────────────────────────────────────────

function initMap() {
    const b = BOUNDS[currentCity];
    const center = [(b.south + b.north) / 2, (b.west + b.east) / 2];

    map = L.map("map", {
        center,
        zoom: 11,
        zoomControl: true,
        attributionControl: false,
    });

    // Add attribution
    L.control.attribution({ prefix: false, position: "bottomleft" })
        .addAttribution('Imagery &copy; Sentinel-2 | LULC &copy; Prithvi v2')
        .addTo(map);

    updateOverlay();
}


function updateOverlay() {
    const b = BOUNDS[currentCity];
    const bounds = [[b.south, b.west], [b.north, b.east]];
    const imgPath = OVERLAY_PATHS[currentCity][currentYear];
    const baseImgPath = BASEMAP_PATHS[currentCity][currentYear] || BASEMAP_PATHS[currentCity]["2018"];
    const opacity = document.getElementById("opacity-slider").value / 100;

    // Properly remove old layers to free memory
    if (overlay) {
        map.removeLayer(overlay);
        overlay = null;
    }
    if (baseTile) {
        map.removeLayer(baseTile);
        baseTile = null;
    }

    baseTile = L.imageOverlay(baseImgPath, bounds, {
        opacity: 1.0,
        interactive: false,
    }).addTo(map);

    overlay = L.imageOverlay(imgPath, bounds, {
        opacity,
        interactive: false,
    }).addTo(map);

    map.fitBounds(bounds, { padding: [20, 20] });
}


// ─── STATS ─────────────────────────────────────────────────────

function updateStats() {
    const data = CHANGE_DATA[currentCity];
    const grid = document.getElementById("stats-grid");

    const items = [
        {
            label: "Vegetation",
            value: data["Vegetation"].net,
            sub: `${data["Vegetation"].area_2018} → ${data["Vegetation"].area_2026} km²`,
        },
        {
            label: "Built-up",
            value: data["Built-up"].net,
            sub: `${data["Built-up"].area_2018} → ${data["Built-up"].area_2026} km²`,
        },
        {
            label: "Cropland",
            value: data["Cropland"].net,
            sub: `${data["Cropland"].area_2018} → ${data["Cropland"].area_2026} km²`,
        },
        {
            label: "Barren",
            value: data["Barren/Other"].net,
            sub: `${data["Barren/Other"].area_2018} → ${data["Barren/Other"].area_2026} km²`,
        },
    ];

    grid.innerHTML = items.map(item => {
        const sign = item.value >= 0 ? "+" : "";
        const cls  = item.value >= 0 ? "positive" : "negative";
        return `
            <div class="stat-item">
                <div class="stat-value ${cls}">${sign}${item.value.toFixed(1)}</div>
                <div class="stat-label">${item.label} (km²)</div>
                <div class="stat-sub">${item.sub}</div>
            </div>
        `;
    }).join("");
}


// ─── DONUT CHART ───────────────────────────────────────────────

function updateDonutChart() {
    const data = CHANGE_DATA[currentCity];
    const yearKey = `area_${currentYear}`;
    const values = Object.keys(data).map(k => data[k][yearKey]);
    const labels = Object.keys(data);
    const colors = labels.map(l => CLASS_COLORS[l]);

    document.getElementById("donut-year-label").textContent = currentYear;

    if (donutChart) donutChart.destroy();
    Chart.register(ChartDataLabels);

    donutChart = new Chart(document.getElementById("donut-chart"), {
        type: "doughnut",
        data: {
            labels,
            datasets: [{
                data: values,
                backgroundColor: colors,
                borderColor: "rgba(6, 11, 24, 0.8)",
                borderWidth: 2,
                hoverOffset: 8,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: "60%",
            plugins: {
                datalabels: {
                    color: '#ffffff',
                    font: { family: 'Inter', weight: 'bold', size: 12 },
                    formatter: (value, ctx) => {
                        const total = ctx.dataset.data.reduce((a, b) => a + b, 0);
                        const percentage = (value * 100 / total).toFixed(1);
                        return percentage >= 5 ? percentage + "%" : ""; // Only show if >= 5% to avoid clutter
                    },
                    textStrokeColor: 'rgba(0,0,0,0.8)',
                    textStrokeWidth: 3
                },
                legend: {
                    position: "bottom",
                    labels: {
                        color: "#8b95ab",
                        font: { family: "Inter", size: 11 },
                        padding: 12,
                        usePointStyle: true,
                        pointStyleWidth: 8,
                    },
                },
                tooltip: {
                    backgroundColor: "rgba(12, 18, 37, 0.95)",
                    titleColor: "#e8ecf4",
                    bodyColor: "#8b95ab",
                    borderColor: "rgba(59, 130, 246, 0.3)",
                    borderWidth: 1,
                    cornerRadius: 8,
                    padding: 10,
                    bodyFont: { family: "Inter" },
                    callbacks: {
                        label: ctx => {
                            const total = ctx.dataset.data.reduce((a, b) => a + b, 0);
                            const percentage = (ctx.parsed * 100 / total).toFixed(1);
                            return ` ${ctx.label}: ${ctx.parsed.toFixed(1)} km² (${percentage}%)`;
                        },
                    },
                },
            },
        },
    });
}


// ─── BAR CHART ─────────────────────────────────────────────────

function updateBarChart() {
    const data = CHANGE_DATA[currentCity];
    const classes = Object.keys(data);
    const years = ["2018", "2021", "2026"];

    const datasets = years.map((yr, i) => ({
        label: yr,
        data: classes.map(c => data[c][`area_${yr}`]),
        backgroundColor: [
            "rgba(59, 130, 246, 0.5)",
            "rgba(99, 102, 241, 0.5)",
            "rgba(34, 197, 94, 0.5)",
        ][i],
        borderColor: [
            "rgba(59, 130, 246, 1)",
            "rgba(99, 102, 241, 1)",
            "rgba(34, 197, 94, 1)",
        ][i],
        borderWidth: 1,
        borderRadius: 4,
    }));

    if (barChart) barChart.destroy();

    barChart = new Chart(document.getElementById("bar-chart"), {
        type: "bar",
        data: { labels: classes, datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    ticks: { color: "#8b95ab", font: { family: "Inter", size: 10 } },
                    grid: { color: "rgba(59, 130, 246, 0.06)" },
                },
                y: {
                    ticks: {
                        color: "#8b95ab",
                        font: { family: "Inter", size: 10 },
                        callback: v => v >= 1000 ? `${(v/1000).toFixed(1)}k` : v,
                    },
                    grid: { color: "rgba(59, 130, 246, 0.06)" },
                },
            },
            plugins: {
                legend: {
                    labels: {
                        color: "#8b95ab",
                        font: { family: "Inter", size: 11 },
                        usePointStyle: true,
                        pointStyleWidth: 8,
                        padding: 12,
                    },
                },
                tooltip: {
                    backgroundColor: "rgba(12, 18, 37, 0.95)",
                    titleColor: "#e8ecf4",
                    bodyColor: "#8b95ab",
                    borderColor: "rgba(59, 130, 246, 0.3)",
                    borderWidth: 1,
                    cornerRadius: 8,
                    bodyFont: { family: "Inter" },
                    callbacks: {
                        label: ctx => ` ${ctx.dataset.label}: ${ctx.parsed.y.toFixed(1)} km²`,
                    },
                },
            },
        },
    });
}


// ─── MODEL INFO ────────────────────────────────────────────────

function updateModelInfo() {
    const metrics = MODEL_METRICS[currentCity];
    const container = document.getElementById("model-info");

    // Full-city metrics
    let bestMiou = 0;
    let bestName = "";
    for (const [name, m] of Object.entries(metrics)) {
        if (m.miou > bestMiou) { bestMiou = m.miou; bestName = name; }
    }

    let html = `<div class="model-section-label">Full-City (${currentCity === "lahore" ? "Lahore" : "Gujranwala"})</div>`;
    html += Object.entries(metrics).map(([name, m]) => {
        const isBest = name === bestName;
        return `
            <div class="model-row ${isBest ? 'best' : ''}">
                <span class="model-name">${name}</span>
                <span class="model-metric ${isBest ? 'best-val' : ''}">${m.miou.toFixed(2)}% mIoU</span>
            </div>
        `;
    }).join("");

    // Patch-level metrics (mean ± std)
    html += `<div class="model-section-label" style="margin-top:12px;">Patch-Level (mean ± std, 3 seeds)</div>`;
    let bestPatch = 0;
    let bestPatchName = "";
    for (const [name, m] of Object.entries(PATCH_METRICS)) {
        if (m.miou > bestPatch) { bestPatch = m.miou; bestPatchName = name; }
    }
    html += Object.entries(PATCH_METRICS).map(([name, m]) => {
        const isBest = name === bestPatchName;
        return `
            <div class="model-row ${isBest ? 'best' : ''}">
                <span class="model-name">${name}</span>
                <span class="model-metric ${isBest ? 'best-val' : ''}">${m.miou.toFixed(2)}±${m.std.toFixed(2)}%</span>
            </div>
        `;
    }).join("");

    // LoRA ablation
    html += `<div class="model-section-label" style="margin-top:12px;">LoRA Rank Ablation</div>`;
    LORA_ABLATION.forEach(r => {
        const isBest = r.rank === 64;
        html += `
            <div class="model-row ${isBest ? 'best' : ''}">
                <span class="model-name">r=${r.rank} <span style="opacity:0.5;font-size:10px">(${r.trainable})</span></span>
                <span class="model-metric ${isBest ? 'best-val' : ''}">${r.miou.toFixed(2)}% mIoU</span>
            </div>
        `;
    });

    container.innerHTML = html;
}


// ─── EVENT LISTENERS ───────────────────────────────────────────

function setupListeners() {
    // City toggle
    document.querySelectorAll("#city-toggle .toggle-btn").forEach(btn => {
        btn.addEventListener("click", () => {
            document.querySelectorAll("#city-toggle .toggle-btn").forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            currentCity = btn.dataset.city;
            refreshAll();
        });
    });

    // Year toggle
    document.querySelectorAll("#year-toggle .toggle-btn").forEach(btn => {
        btn.addEventListener("click", () => {
            document.querySelectorAll("#year-toggle .toggle-btn").forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            currentYear = btn.dataset.year;
            updateOverlay();
            updateDonutChart();
        });
    });

    // Opacity slider
    document.getElementById("opacity-slider").addEventListener("input", e => {
        if (overlay) overlay.setOpacity(e.target.value / 100);
    });
}


// ─── REFRESH ALL ───────────────────────────────────────────────

function refreshAll() {
    updateOverlay();
    updateStats();
    updateDonutChart();
    updateBarChart();
    updateModelInfo();
}


// ─── INIT ──────────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", async () => {
    // Phase 1: Preload all images with progress
    await preloadAllImages();

    // Phase 2: Initialize the dashboard (instant since images are cached)
    initMap();
    setupListeners();
    updateStats();
    updateDonutChart();
    updateBarChart();
    updateModelInfo();

    // Phase 3: Fade out the loader
    hideLoader();
});
