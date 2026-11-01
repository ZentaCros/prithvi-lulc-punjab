// ==============================================================================
// GEE Script: Export Gujranwala Sentinel-2 Median Composites (2018, 2021, 2024, 2026)
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

function getYearlyComposite(year) {
  // For 2026, we only use Jan to May since we are currently in May 2026
  var endDate = (year === 2026) ? '2026-05-01' : year + '-12-31';
  var collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(roi)
    .filterDate(year + '-01-01', endDate)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
    .map(maskS2clouds);
  return collection.median().clip(roi).toFloat();
}

var img2018 = getYearlyComposite(2018);
var img2021 = getYearlyComposite(2021);
var img2024 = getYearlyComposite(2024);
var img2026 = getYearlyComposite(2026);

// Export Functions
var exportImg = function(img, year) {
  Export.image.toDrive({
    image: img,
    description: 'gujranwala_sentinel2_' + year + '_10bands',
    folder: 'LULC_Gujranwala',
    scale: 10,
    region: roi,
    maxPixels: 1e13,
    crs: 'EPSG:32643'
  });
};

exportImg(img2018, '2018');
exportImg(img2021, '2021');
exportImg(img2024, '2024');
exportImg(img2026, '2026');
