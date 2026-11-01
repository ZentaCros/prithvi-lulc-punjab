// ============================================================================
// SENTINEL-2 CLOUD-FREE COMPOSITE FOR LAHORE DISTRICT (2021)
// Exports 10-band median composite for 2021 to perfectly align with ESA WorldCover
// ============================================================================
// HOW TO USE:
//   1. Go to https://code.earthengine.google.com
//   2. Create a new script, paste this entire code
//   3. Click "Run"
//   4. Go to the "Tasks" tab (right panel)
//   5. Click "RUN" next to the export task
//   6. Accept default settings, click "Run" in the dialog
//   7. Wait for exports to finish (check Tasks tab for progress)
//   8. Find the file in your Google Drive under "LULC_Lahore" folder
//   9. Move the file to your project: data/sentinel2/sentinel2_2021_10bands.tif
// ============================================================================

// --- 1. DEFINE STUDY AREA: LAHORE DISTRICT ---
var lahore = ee.Geometry.Polygon([
  [74.10, 31.70],  // NW corner
  [74.55, 31.70],  // NE corner
  [74.55, 31.20],  // SE corner
  [74.10, 31.20],  // SW corner
  [74.10, 31.70]   // close polygon
]);

Map.centerObject(lahore, 11);
Map.addLayer(lahore, {color: 'red'}, 'Lahore Study Area');

// --- 2. BAND SELECTION ---
var BANDS = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12'];

// --- 3. CLOUD MASKING FUNCTION ---
function maskS2clouds(image) {
  var scl = image.select('SCL');
  var cloudMask = scl.neq(3).and(scl.neq(8)).and(scl.neq(9)).and(scl.neq(10));
  var qualityMask = scl.neq(0);
  return image.updateMask(cloudMask.and(qualityMask)).select(BANDS);
}

// --- 4. CREATE 2021 COMPOSITE ---
// Using same season: Oct 2020 - Mar 2021 to match the 2021 ESA labels perfectly
var s2_2021 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(lahore)
  .filterDate('2020-10-01', '2021-03-31') 
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
  .map(maskS2clouds)
  .select(BANDS);

var composite_2021 = s2_2021
  .median()
  .clip(lahore)
  .toFloat();

print('2021 composite band count:', composite_2021.bandNames().size());
print('2021 images used:', s2_2021.size());

// --- 5. EXPORT TO GOOGLE DRIVE ---
Export.image.toDrive({
  image: composite_2021,
  description: 'sentinel2_2021_lahore',
  folder: 'LULC_Lahore',
  fileNamePrefix: 'sentinel2_2021_10bands',
  region: lahore,
  scale: 10,
  crs: 'EPSG:32643',  // UTM Zone 43N
  maxPixels: 1e10,
  fileFormat: 'GeoTIFF'
});

print('✅ Script ready! Go to Tasks tab and click RUN.');
print('📁 File will appear in Google Drive → LULC_Lahore/');
