// ============================================================================
// SENTINEL-2 CLOUD-FREE COMPOSITES FOR LAHORE DISTRICT
// Exports 10-band median composites for 2018 and 2024
// ============================================================================
// HOW TO USE:
//   1. Go to https://code.earthengine.google.com
//   2. Create a new script, paste this entire code
//   3. Click "Run"
//   4. Go to the "Tasks" tab (right panel)
//   5. Click "RUN" next to each export task (there will be 2)
//   6. Accept default settings, click "Run" in the dialog
//   7. Wait for exports to finish (check Tasks tab for progress)
//   8. Find the files in your Google Drive under "LULC_Lahore" folder
// ============================================================================

// --- 1. DEFINE STUDY AREA: LAHORE DISTRICT ---
// Manual bounding polygon for Lahore District (~1,772 km²)
// Covers the full district including cantonment, suburbs, and peri-urban areas
var lahore = ee.Geometry.Polygon([
  [74.10, 31.70],  // NW corner
  [74.55, 31.70],  // NE corner
  [74.55, 31.20],  // SE corner
  [74.10, 31.20],  // SW corner
  [74.10, 31.70]   // close polygon
]);

// Verify
print('Lahore study area (km²):', lahore.area().divide(1e6));

// Center the map on Lahore
Map.centerObject(lahore, 11);
Map.addLayer(lahore, {color: 'red'}, 'Lahore Study Area');

// --- DEBUG: Uncomment these lines to find the correct GAUL name ---
// var pakistan_districts = ee.FeatureCollection('FAO/GAUL/2015/level2')
//   .filter(ee.Filter.eq('ADM0_NAME', 'Pakistan'));
// print('All Pakistan districts:', pakistan_districts.aggregate_array('ADM2_NAME'));

// --- 2. BAND SELECTION ---
// 10 bands: 4 at native 10m + 6 at native 20m (resampled to 10m on export)
var BANDS = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12'];
var BAND_NAMES = ['Blue', 'Green', 'Red', 'RE1', 'RE2', 'RE3', 'NIR', 'NIR_Narrow', 'SWIR1', 'SWIR2'];

// --- 3. CLOUD MASKING FUNCTION ---
// Uses the Scene Classification Layer (SCL) to mask clouds, shadows, and cirrus
function maskS2clouds(image) {
  var scl = image.select('SCL');
  // Mask out: cloud shadows (3), clouds med/high (8,9), cirrus (10)
  var cloudMask = scl.neq(3)   // Cloud shadows
    .and(scl.neq(8))           // Cloud medium probability
    .and(scl.neq(9))           // Cloud high probability
    .and(scl.neq(10));         // Thin cirrus
  
  // Also mask saturated/defective pixels (0) and dark area (1 - optional, keep for now)
  var qualityMask = scl.neq(0);
  
  return image
    .updateMask(cloudMask.and(qualityMask))
    .select(BANDS)
    .copyProperties(image, ['system:time_start']);
}

// --- 4. CREATE COMPOSITES ---
// Using S2_SR_HARMONIZED for consistent Surface Reflectance across years
// Note: SR data available from 2017-03-28 onward, so we use 2018 (not 2016)
//       for better data availability (full S2A+S2B constellation)

// 2018 Composite (dry season: Oct-Mar for less cloud cover in Punjab)
var s2_2018 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(lahore)
  .filterDate('2017-10-01', '2018-03-31')  // Dry season composite
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
  .map(maskS2clouds)
  .select(BANDS);

var composite_2018 = s2_2018
  .median()
  .clip(lahore)
  .toFloat();

print('2018 composite band count:', composite_2018.bandNames().size());
print('2018 images used:', s2_2018.size());

// 2024 Composite (same season for fair comparison)
var s2_2024 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(lahore)
  .filterDate('2023-10-01', '2024-03-31')  // Dry season composite
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
  .map(maskS2clouds)
  .select(BANDS);

var composite_2024 = s2_2024
  .median()
  .clip(lahore)
  .toFloat();

print('2024 composite band count:', composite_2024.bandNames().size());
print('2024 images used:', s2_2024.size());

// --- 5. VISUALIZE ---
var visParams = {bands: ['B4', 'B3', 'B2'], min: 0, max: 3000};
Map.addLayer(composite_2018, visParams, 'Sentinel-2 Composite 2018');
Map.addLayer(composite_2024, visParams, 'Sentinel-2 Composite 2024');

// NDVI for quick sanity check
var ndvi_2018 = composite_2018.normalizedDifference(['B8', 'B4']).rename('NDVI_2018');
var ndvi_2024 = composite_2024.normalizedDifference(['B8', 'B4']).rename('NDVI_2024');
Map.addLayer(ndvi_2018, {min: -0.2, max: 0.8, palette: ['red', 'yellow', 'green']}, 'NDVI 2018');
Map.addLayer(ndvi_2024, {min: -0.2, max: 0.8, palette: ['red', 'yellow', 'green']}, 'NDVI 2024');

// --- 6. EXPORT TO GOOGLE DRIVE ---
// Both exports go to a "LULC_Lahore" folder in your Google Drive
// Scale: 10m | CRS: UTM Zone 43N (covers Lahore)

Export.image.toDrive({
  image: composite_2018,
  description: 'sentinel2_2018_lahore',
  folder: 'LULC_Lahore',
  fileNamePrefix: 'sentinel2_2018_10bands',
  region: lahore,
  scale: 10,
  crs: 'EPSG:32643',  // UTM Zone 43N
  maxPixels: 1e10,
  fileFormat: 'GeoTIFF'
});

Export.image.toDrive({
  image: composite_2024,
  description: 'sentinel2_2024_lahore',
  folder: 'LULC_Lahore',
  fileNamePrefix: 'sentinel2_2024_10bands',
  region: lahore,
  scale: 10,
  crs: 'EPSG:32643',
  maxPixels: 1e10,
  fileFormat: 'GeoTIFF'
});

print('✅ Script ready! Go to Tasks tab and click RUN on both exports.');
print('📁 Files will appear in Google Drive → LULC_Lahore/');
