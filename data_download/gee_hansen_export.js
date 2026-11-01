// ============================================================================
// HANSEN GLOBAL FOREST CHANGE — LAHORE DISTRICT (FOR VALIDATION)
// ============================================================================
// HOW TO USE:
//   1. Go to https://code.earthengine.google.com
//   2. Create a new script, paste this entire code
//   3. Click "Run"
//   4. Go to the "Tasks" tab and click "RUN"
//   5. Find the file in Google Drive → LULC_Lahore/
// ============================================================================

// --- 1. DEFINE STUDY AREA ---
var lahore = ee.Geometry.Polygon([
  [74.10, 31.70],
  [74.55, 31.70],
  [74.55, 31.20],
  [74.10, 31.20],
  [74.10, 31.70]
]);

Map.centerObject(lahore, 11);

// --- 2. LOAD HANSEN GLOBAL FOREST CHANGE ---
// Dataset: University of Maryland (GLAD Lab)
// Tree cover defined as canopy closure for vegetation > 5m height
var hansen = ee.Image('UMD/hansen/global_forest_change_2023_v1_11');

// Key layers:
// treecover2000 — Tree canopy cover in year 2000 (%)
// loss          — Forest loss during 2001-2023 (binary)
// gain          — Forest gain during 2001-2012 (binary)
// lossyear      — Year of loss event (1-23, meaning 2001-2023)

var treeCover2000 = hansen.select('treecover2000').clip(lahore);
var loss = hansen.select('loss').clip(lahore);
var gain = hansen.select('gain').clip(lahore);
var lossYear = hansen.select('lossyear').clip(lahore);

// --- 3. DERIVED LAYERS ---
// Forest mask (>25% canopy cover threshold — common for South Asian landscapes)
var forestMask2000 = treeCover2000.gte(25);

// Forest loss between 2018-2023 (to match our study period)
var lossInStudyPeriod = lossYear.gte(18).and(lossYear.lte(23)).and(forestMask2000);

// Approximate forest cover in 2018 (2000 cover minus losses 2001-2017)
var lossBy2017 = lossYear.gte(1).and(lossYear.lte(17)).and(forestMask2000);
var forestMask2018 = forestMask2000.and(lossBy2017.not());

// --- 4. VISUALIZE ---
Map.addLayer(lahore, {color: 'white'}, 'Lahore Boundary');
Map.addLayer(treeCover2000, {min: 0, max: 100, palette: ['black', 'green']}, 'Tree Cover 2000');
Map.addLayer(forestMask2000, {min: 0, max: 1, palette: ['black', '00FF00']}, 'Forest Mask 2000 (>25%)');
Map.addLayer(loss, {min: 0, max: 1, palette: ['black', 'FF0000']}, 'Forest Loss (all years)');
Map.addLayer(lossInStudyPeriod, {min: 0, max: 1, palette: ['black', 'FF6600']}, 'Forest Loss 2018-2023');
Map.addLayer(gain, {min: 0, max: 1, palette: ['black', '00FFFF']}, 'Forest Gain 2001-2012');

// --- 5. STATISTICS ---
// Calculate area of forest loss in study period
var lossArea = lossInStudyPeriod.multiply(ee.Image.pixelArea()).divide(1e6);  // km²
var lossStats = lossArea.reduceRegion({
  reducer: ee.Reducer.sum(),
  geometry: lahore,
  scale: 30,
  maxPixels: 1e10
});
print('Forest loss 2018-2023 area (km²):', lossStats);

var coverStats = forestMask2000.multiply(ee.Image.pixelArea()).divide(1e6);
var totalForest = coverStats.reduceRegion({
  reducer: ee.Reducer.sum(),
  geometry: lahore,
  scale: 30,
  maxPixels: 1e10
});
print('Forest cover 2000 area (km²):', totalForest);

// --- 6. EXPORT ---
// Export a multi-band validation image
var validationStack = treeCover2000
  .addBands(loss.rename('loss'))
  .addBands(gain.rename('gain'))
  .addBands(lossYear.rename('lossyear'))
  .addBands(forestMask2018.rename('forest_2018'))
  .addBands(lossInStudyPeriod.rename('loss_2018_2023'));

Export.image.toDrive({
  image: validationStack,
  description: 'hansen_forest_change_lahore',
  folder: 'LULC_Lahore',
  fileNamePrefix: 'hansen_forest_change',
  region: lahore,
  scale: 30,  // Hansen native resolution is 30m
  crs: 'EPSG:32643',
  maxPixels: 1e10,
  fileFormat: 'GeoTIFF'
});

print('✅ Script ready! Go to Tasks tab and click RUN.');
print('📁 File will appear in Google Drive → LULC_Lahore/');
print('');
print('BANDS IN EXPORT:');
print('  1. treecover2000 — Canopy cover % in 2000');
print('  2. loss — Binary forest loss (any year)');
print('  3. gain — Binary forest gain 2001-2012');
print('  4. lossyear — Year of loss (1=2001, 23=2023)');
print('  5. forest_2018 — Estimated forest mask in 2018');
print('  6. loss_2018_2023 — Forest loss during our study period');
