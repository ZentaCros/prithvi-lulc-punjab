// ==============================================================================
// GEE Script: Export Lahore Sentinel-2 Median Composite for 2026 (Absolute Latest)
// ==============================================================================

var roi = ee.Geometry.Polygon([
  [74.10, 31.70],
  [74.55, 31.70],
  [74.55, 31.20],
  [74.10, 31.20],
  [74.10, 31.70]
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

// Get 2026 Jan to May Composite
var img2026 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(roi)
  .filterDate('2026-01-01', '2026-05-01')
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
  .map(maskS2clouds)
  .median()
  .clip(roi)
  .toFloat();

Export.image.toDrive({
  image: img2026,
  description: 'sentinel2_2026_10bands',
  folder: 'LULC_Lahore_2026',
  scale: 10,
  region: roi,
  maxPixels: 1e13,
  crs: 'EPSG:32643'
});
