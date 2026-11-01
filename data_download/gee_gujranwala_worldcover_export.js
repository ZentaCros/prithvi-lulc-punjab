// ============================================================================
// ESA WORLDCOVER 2021 — REMAPPED TO 5 CLASSES FOR GUJRANWALA
// ============================================================================

// --- 1. DEFINE STUDY AREA ---
// Same 0.45 x 0.50 degree scale used for the Sentinel-2 exports
var roi = ee.Geometry.Polygon([
  [
    [73.955, 31.91], // SW
    [74.405, 31.91], // SE
    [74.405, 32.41], // NE
    [73.955, 32.41], // NW
    [73.955, 31.91]  // SW (Close Polygon)
  ]
]);

Map.centerObject(roi, 11);
Map.addLayer(roi, {color: 'red'}, 'Gujranwala Boundary');

// --- 2. LOAD ESA WORLDCOVER 2021 (v200) ---
var worldcover = ee.ImageCollection('ESA/WorldCover/v200')
  .first()
  .clip(roi);

// --- 3. REMAP TO 5 CLASSES ---
// Our 5-class scheme:
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

// --- 4. EXPORT ---
Export.image.toDrive({
  image: remapped,
  description: 'worldcover_2021_5class_gujranwala',
  folder: 'LULC_Gujranwala',
  fileNamePrefix: 'worldcover_2021_5class_gujranwala',
  region: roi,
  scale: 10,
  crs: 'EPSG:32643',  // Same CRS as Sentinel-2 exports
  maxPixels: 1e10,
  fileFormat: 'GeoTIFF'
});

print('✅ Gujranwala WorldCover Script ready! Go to Tasks tab and click RUN.');
