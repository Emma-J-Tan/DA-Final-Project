import numpy as np
import os
import requests
import shapefile

from io import BytesIO
from matplotlib.pyplot import imshow
from PIL import Image
from numpy.core.defchararray import strip

URL = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi?SERVICE=WMS&REQUEST=GetMap&layers" \
      "=MODIS_Aqua_CorrectedReflectance_TrueColor&version=1.3.0&crs=EPSG:4326&transparent=false&width={}&height={" \
      "}&bbox={}&format=image/tiff&time={} "
KM_PER_DEG_AT_EQ = 111.


def calculate_width_height(extent, resolution):
    """
    extent: [lower_latitude, left_longitude, higher_latitude, right_longitude], EG: [51.46162974683544,
    -22.94768591772153,53.03698575949367,-20.952234968354432] resolution: represents the pixel resolution,
    i.e. km/pixel. Should be a value from this list: [0.03, 0.06, 0.125, 0.25, 0.5, 1, 5, 10]
    """
    lats = extent[::2]
    lons = extent[1::2]
    km_per_deg_at_lat = KM_PER_DEG_AT_EQ * np.cos(np.pi * np.mean(lats) / 180.)
    width = int((lons[1] - lons[0]) * km_per_deg_at_lat / resolution)
    height = int((lats[1] - lats[0]) * KM_PER_DEG_AT_EQ / resolution)
    print(width, height)
    return width, height


def modis_url(time, extent, resolution):
    """
    time: utc time in iso format EG: 2020-02-19T00:00:00Z extent: [lower_latitude, left_longitude, higher_latitude,
    right_longitude], EG: [51.46162974683544,-22.94768591772153,53.03698575949367,-20.952234968354432] resolution:
    represents the pixel resolution, i.e. km/pixel. Should be a value from this list: [0.03, 0.06, 0.125, 0.25, 0.5,
    1, 5, 10]
    """
    width, height = calculate_width_height(extent, resolution)
    extent = ','.join(map(lambda x: str(x), extent))
    return width, height, URL.format(width, height, extent, time)


invalid_dates = ['2002-01-28', '2002-05-11', '2002-05-12', '2005-10-31']
directory = '/Users/yaoxiaoyi/Desktop/HLD'
save_dir = '/Users/yaoxiaoyi/Desktop/HLD_Images'
for filename in os.listdir(directory):
    shpfile = directory + '/' + filename + '/' + filename
    sh = shapefile.Reader(shpfile)
    date = filename[19:29]
    #if date not in invalid_dates:
    time = date + 'T00:00:00Z'
    width, height, url = modis_url(time,
                                      [sh.bbox[1], sh.bbox[0], sh.bbox[3], sh.bbox[2]], .25)
    response = requests.get(strip(url))
    img = BytesIO(response.content)
    im = Image.open(img)
    im.save(save_dir + '/' + filename + '.jpg')
