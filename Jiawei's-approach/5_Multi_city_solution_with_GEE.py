# In this file, we use Google Earth Enginee to automatically download satellite images and automatically crop images into chips with proper size
# We also try to create corresponding mask chips to match each satellite image chip
# To smooth the process and make use of google drive, this file can be run in Google Colab

# import packages
import numpy as np
import matplotlib.pyplot as plt
import ee
import geemap
import logging
import multiprocessing
import os
import requests
import shutil
from retry import retry
import rasterio
import gdal
from osgeo import gdal
from gdalconst import GA_ReadOnly

#################################################################
# download satellite image

# initialize ee and map
ee.Authenticate()
ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
Map = geemap.Map()

# define region of interest (San Francisco here)
roi = ee.Geometry.Rectangle([-122.5136606, 37.8320222, -122.3497964, 37.707959])

# set the parameters of image
image = ee.ImageCollection('COPERNICUS/S2_SR') \
            .filterBounds(roi) \
            .filterDate('2020-06-01', '2020-09-30') \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10)) \
            .select('B2', 'B3', 'B4') \
            .median() \
            .visualize(min=0, max=4000) \
            .clip(roi)

# visualize map
Map.addLayer(image, {}, "Image")
Map.addLayer(roi, {}, "ROI", False)
Map.setCenter((-122.5136606 + -122.3497964)/2, (37.8320222 + 37.707959)/2, 12)
Map

# download satellite image
task = ee.batch.Export.image.toDrive(image=image,
                                     description='satellite_image_SF',
                                     scale=10,
                                     region=roi,
                                     fileNamePrefix='satellite_image_SF',
                                     crs='EPSG:4326',
                                     fileFormat='GeoTIFF')
task.start()

# visualize
image_SF = gdal.Open('file_path_to_saved_satellite_image')
image_array_SF = image_SF.ReadAsArray()
image_array_SF = np.transpose(image_array_SF, [1, 2, 0])  # transpose to fit the shape for visualization
plt.imshow(image_array_SF)

# get meta data of satellite image
geo_transform = image_SF.GetGeoTransform()
x_min = geo_transform[0]
y_max = geo_transform[3]
x_max = x_min + geo_transform[1] * image_SF.RasterXSize
y_min = y_max + geo_transform[5] * image_SF.RasterYSize
x_res = image_SF.RasterXSize
y_res = image_SF.RasterYSize
pixel_width = geo_transform[1]

print("x_min", x_min)
print("x_max", x_max)
print("y_min", y_min)
print("y_max", y_max)
print("x_res", x_res)
print("y_res", y_res)
print("pixel_width", pixel_width)

########################################################################
# download image chips for satellite image

# set the parameters for downloading chips
params = {
    'count': 10,               # How many image chips to export
    'buffer': 1500,            # The buffer distance (m) around each point
    'scale': 100,              # The scale to do stratified sampling
    'seed': 1,                 # A randomization seed to use for subsampling.
    'dimensions': '256x256',   # The dimension of each image chip
    'format': "GEO_TIFF",      # The output image format, can be png, jpg, ZIPPED_GEO_TIFF, GEO_TIFF, NPY
    'prefix': 'tile_',         # The filename prefix
    'processes': 25,           # How many processes to used for parallel processing
    'out_dir': '.',            # The output directory. Default to the current working directly
}

# define functions for downloading chips
def getRequests():
    img = ee.Image(1).rename("Class").addBands(image)
    points = img.stratifiedSample(
        numPoints=params['count'],
        region=roi,
        scale=params['scale'],
        seed=params['seed'],
        geometries=True,
    )
    Map.data = points
    return points.aggregate_array('.geo').getInfo()

@retry(tries=10, delay=1, backoff=2)
def getResult(index, point):
    point = ee.Geometry.Point(point['coordinates'])
    region = point.buffer(params['buffer']).bounds()

    if params['format'] in ['png', 'jpg']:
        url = image.getThumbURL(
            {
                'region': region,
                'dimensions': params['dimensions'],
                'format': params['format'],
            }
        )
    else:
        url = image.getDownloadURL(
            {
                'region': region,
                'dimensions': params['dimensions'],
                'format': params['format'],
            }
        )

    if params['format'] == "GEO_TIFF":
        ext = 'tif'
    else:
        ext = params['format']

    r = requests.get(url, stream=True)
    if r.status_code != 200:
        r.raise_for_status()

    out_dir = os.path.abspath(params['out_dir'])
    basename = str(index).zfill(len(str(params['count'])))
    filename = f"{out_dir}/{params['prefix']}{basename}.{ext}"
    with open(filename, 'wb') as out_file:
        shutil.copyfileobj(r.raw, out_file)
    print("Done: ", basename)

# download chips
%%time
logging.basicConfig()
items = getRequests()

pool = multiprocessing.Pool(params['processes'])
pool.starmap(getResult, enumerate(items))
pool.close()

#################################################################################
# create corresponding mask chips

# rasterize vector data of park (mask)
# read parks polygon
import geopandas as gpd
parks = gpd.read_file('file_path_to_park_polygon')

# Rasterize parks polygon
from rasterio import features
rst_base = rasterio.open('file_path_to_base_image')  # Base image to rasterize the *.shp
meta = rst_base.meta.copy()  # Copy metadata from the base image
meta.update(compress='lzw')

# Burn the AOIs *.shp file into raster and save it
out_rst = 'path_path_to_rasterized_park'
with rasterio.open(out_rst, 'w+', **meta) as out:
    out_arr = out.read(1)
    # Create a generator of geom, value pairs to use in rasterizing
    shapes = (geom for geom in parks.geometry)
    burned = features.rasterize(shapes=shapes, fill=0, out=out_arr, transform=out.transform)
    out.write_band(1, burned)

# get meta data of rasterized park
park_img = gdal.Open('path_path_to_rasterized_park')
geo_transform = park_img.GetGeoTransform()
x_min = geo_transform[0]
y_max = geo_transform[3]
x_max = x_min + geo_transform[1] * park_img.RasterXSize
y_min = y_max + geo_transform[5] * park_img.RasterYSize
x_res = park_img.RasterXSize
y_res = park_img.RasterYSize
pixel_width = geo_transform[1]

print("x_min", x_min)
print("x_max", x_max)
print("y_min", y_min)
print("y_max", y_max)
print("x_res", x_res)
print("y_res", y_res)
print("pixel_width", pixel_width)

# clip rasterized_park to let its extent match satellite image chip
# a for loop can be created for each satellite image chip, here is only an example
maskDs = gdal.Open('file_path_to_one_satellite_image_chip', GA_ReadOnly)
projection = maskDs.GetProjectionRef()
geoTransform = maskDs.GetGeoTransform()
minx = geoTransform[0]
maxy = geoTransform[3]
maxx = minx + geoTransform[1] * maskDs.RasterXSize
miny = maxy + geoTransform[5] * maskDs.RasterYSize

data = gdal.Open('file_path_to_rasterized_park')
output = 'file_path_to_park_chip'
data = gdal.Translate(output,data,format='GTiff', projWin=[minx, maxy, maxx, miny], outputSRS=projection)
data = None