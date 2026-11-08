// ==============================================================================
// GEE Script: Gujranwala DRY-SEASON Sentinel-2 Composite (Oct-Mar)
// Fix #5: Same temporal window as Lahore for fair cross-city comparison
// ==============================================================================
// HOW TO USE:
//   1. Go to https://code.earthengine.google.com
//   2. Create a new script, paste this entire code
//   3. Click "Run"
//   4. Go to the "Tasks" tab (right panel)
//   5. Click "RUN" next to the export task
//   6. Wait for export to finish (~10-15 min)
//   7. Find the file in Google Drive under "LULC_Gujranwala" folder
//   8. Download and place in: data/inference/gujranwala/
//      Rename to: gujranwala_sentinel2_2021_dryseason_10bands.tif
// ==============================================================================

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

var BANDS = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12'];

function maskS2clouds(image) {
  var scl = image.select('SCL');
  var cloudMask = scl.neq(3)
    .and(scl.neq(8))
    .and(scl.neq(9))
    .and(scl.neq(10));
  var qualityMask = scl.neq(0);

  return image
    .updateMask(cloudMask.and(qualityMask))
    .select(BANDS)
    .copyProperties(image, ['system:time_start']);
}

// Dry-season composite: Oct 2020 - Mar 2021
// Matches Lahore's temporal window (Oct-Mar) for fair comparison
var s2_2021_dry = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(roi)
  .filterDate('2020-10-01', '2021-03-31')
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
  .map(maskS2clouds);

var composite = s2_2021_dry.median().clip(roi).toFloat();

print('Gujranwala dry-season 2021 composite');
print('Band count:', composite.bandNames().size());
print('Images used:', s2_2021_dry.size());

// Visualize
var visParams = {bands: ['B4', 'B3', 'B2'], min: 0, max: 3000};
Map.addLayer(composite, visParams, 'Gujranwala Dry-Season 2021');
Map.addLayer(roi, {color: 'red'}, 'Study Area');

// NDVI check
var ndvi = composite.normalizedDifference(['B8', 'B4']).rename('NDVI');
Map.addLayer(ndvi, {min: -0.2, max: 0.8, palette: ['red', 'yellow', 'green']}, 'NDVI');

// Export
Export.image.toDrive({
  image: composite,
  description: 'gujranwala_sentinel2_2021_dryseason_10bands',
  folder: 'LULC_Gujranwala',
  scale: 10,
  region: roi,
  maxPixels: 1e13,
  crs: 'EPSG:32643'
});

print('Ready! Go to Tasks tab and click RUN.');
