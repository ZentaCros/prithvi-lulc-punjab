// ============================================================================
// ESA WORLDCOVER 2021 — REMAPPED TO 5 CLASSES FOR LAHORE DISTRICT
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
Map.addLayer(lahore, {color: 'red'}, 'Lahore District Boundary');

// --- 2. LOAD ESA WORLDCOVER 2021 (v200) ---
// Original 11 classes at 10m resolution
var worldcover = ee.ImageCollection('ESA/WorldCover/v200')
  .first()
  .clip(lahore);

// --- 3. ORIGINAL CLASS MAPPING ---
// ESA WorldCover classes:
//  10 = Tree cover
//  20 = Shrubland
//  30 = Grassland
//  40 = Cropland
//  50 = Built-up
//  60 = Bare / sparse vegetation
//  70 = Snow and ice
//  80 = Permanent water bodies
//  90 = Herbaceous wetland
//  95 = Mangroves
// 100 = Moss and lichen

// --- 4. REMAP TO 5 CLASSES ---
// Our 5-class scheme for Punjab/Lahore:
//  0 = Vegetation    (Tree cover + Shrubland + Grassland)
//  1 = Built-up      (Built-up areas)
//  2 = Cropland      (Cropland)
//  3 = Water         (Permanent water bodies)
//  4 = Barren/Other  (Bare/sparse veg + Wetland + Snow + Mangroves + Moss)

var remapped = worldcover.remap(
  [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100],  // From (ESA originals)
  [ 0,  0,  0,  2,  1,  4,  4,  3,  4,  4,   4],  // To (our 5 classes)
  4  // Default value for any unmapped pixels
).rename('lulc').toByte();

// --- 5. VISUALIZE ---
// Original WorldCover
var wcVis = {
  min: 10, max: 100,
  palette: [
    '006400', // 10 Tree cover - dark green
    'ffbb22', // 20 Shrubland - orange
    'ffff4c', // 30 Grassland - yellow
    'f096ff', // 40 Cropland - pink
    'fa0000', // 50 Built-up - red
    'b4b4b4', // 60 Bare - grey
    'f0f0f0', // 70 Snow - white
    '0064c8', // 80 Water - blue
    '0096a0', // 90 Wetland - teal
    '00cf75', // 95 Mangroves - green
    'fae6a0', // 100 Moss - beige
  ]
};
Map.addLayer(worldcover, wcVis, 'WorldCover Original (11 classes)');

// Our remapped 5-class version
var remapVis = {
  min: 0, max: 4,
  palette: [
    '2d7d46',  // 0 Vegetation - green
    'e63946',  // 1 Built-up - red
    'f4a261',  // 2 Cropland - yellow/orange
    '457b9d',  // 3 Water - blue
    'a8a8a8',  // 4 Barren/Other - grey
  ]
};
Map.addLayer(remapped, remapVis, 'Remapped 5-Class LULC');

// --- 6. CLASS STATISTICS ---
// Count pixels per class to understand class distribution
var classNames = ['Vegetation', 'Built-up', 'Cropland', 'Water', 'Barren/Other'];
var histogram = remapped.reduceRegion({
  reducer: ee.Reducer.frequencyHistogram(),
  geometry: lahore,
  scale: 10,
  maxPixels: 1e10
});
print('Class Distribution (pixel counts):', histogram);

// --- 7. EXPORT ---
Export.image.toDrive({
  image: remapped,
  description: 'worldcover_2021_5class_lahore',
  folder: 'LULC_Lahore',
  fileNamePrefix: 'worldcover_2021_5class',
  region: lahore,
  scale: 10,
  crs: 'EPSG:32643',  // Same CRS as Sentinel-2 exports!
  maxPixels: 1e10,
  fileFormat: 'GeoTIFF'
});

// Also export the ORIGINAL (11 class) version for reference
Export.image.toDrive({
  image: worldcover.toByte(),
  description: 'worldcover_2021_original_lahore',
  folder: 'LULC_Lahore',
  fileNamePrefix: 'worldcover_2021_original',
  region: lahore,
  scale: 10,
  crs: 'EPSG:32643',
  maxPixels: 1e10,
  fileFormat: 'GeoTIFF'
});

print('✅ Script ready! Go to Tasks tab and click RUN on both exports.');
print('📁 Files will appear in Google Drive → LULC_Lahore/');
print('');
print('CLASS MAPPING:');
print('  0 = Vegetation (Tree + Shrub + Grass)');
print('  1 = Built-up');
print('  2 = Cropland');
print('  3 = Water');
print('  4 = Barren/Other');
