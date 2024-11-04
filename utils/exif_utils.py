import os
# import ffmpeg
from exiftool import ExifToolHelper
from PIL import Image, ExifTags
from pillow_heif import register_heif_opener
register_heif_opener()

from geopy.geocoders import Nominatim
from datetime import datetime

def convert_gps_to_degree(gps):
    gps = eval(gps)
    d = gps[0]
    m = gps[1]
    s = gps[2]
    return d + (m / 60.0) + (s / 3600.0)

# def read_date_time_from_image(image):
#     exif = image._getexif()
#     exif = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}
#     date_time = exif.get('DateTimeOriginal')
#     return date_time

def get_time_of_the_day(hour):
    if 5 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 18:
        return "Afternoon"
    elif 18 <= hour < 23:
        return "Evening"
    else:
        return "Night"

def parse_date_time(exif=None, date_time_string=""):
    if date_time_string == "":
        date_time = exif.get('DateTime')
        if date_time is None:
            date_time = exif.get('DateTimeOriginal')
            if date_time is None:
                return None
    else:
        date_time = date_time_string
    
    if 'PM' in date_time:
        date_time = date_time.replace('PM', '')
    date_time_object = datetime.strptime(date_time, '%Y:%m:%d %H:%M:%S')
    day_of_week = date_time_object.strftime('%A')
    time_of_the_day = get_time_of_the_day(date_time_object.hour)

    date_info = {
        'date_string': date_time,
        'day_of_week': day_of_week,
        'time_of_the_day': time_of_the_day
    }
    return date_info

def extract_date_time_modified(filepath=""):
    modification_time = os.path.getmtime(filepath)
    modification_time_readable = datetime.fromtimestamp(modification_time).strftime('%Y:%m:%d %H:%M:%S')
    return modification_time_readable

def read_GPS_from_image(image):
    image_info = image.getexif().get_ifd(0x8825)

    geo_tagging_info = {}
    gps_keys = ['GPSVersionID', 'GPSLatitudeRef', 'GPSLatitude', 'GPSLongitudeRef', 'GPSLongitude',
            'GPSAltitudeRef', 'GPSAltitude', 'GPSTimeStamp', 'GPSSatellites', 'GPSStatus', 'GPSMeasureMode',
            'GPSDOP', 'GPSSpeedRef', 'GPSSpeed', 'GPSTrackRef', 'GPSTrack', 'GPSImgDirectionRef',
            'GPSImgDirection', 'GPSMapDatum', 'GPSDestLatitudeRef', 'GPSDestLatitude', 'GPSDestLongitudeRef',
            'GPSDestLongitude', 'GPSDestBearingRef', 'GPSDestBearing', 'GPSDestDistanceRef', 'GPSDestDistance',
            'GPSProcessingMethod', 'GPSAreaInformation', 'GPSDateStamp', 'GPSDifferential']
    for k, v in image_info.items():
        try:
            geo_tagging_info[gps_keys[k]] = str(v)
        except IndexError:
            pass

    if geo_tagging_info == {}:
        return {}

    latitude_ref = geo_tagging_info['GPSLatitudeRef']
    latitude = geo_tagging_info['GPSLatitude']
    latitude = convert_gps_to_degree(latitude)
    if latitude_ref == 'S':
        latitude = -latitude
    longitude_ref = geo_tagging_info['GPSLongitudeRef']
    longitude = geo_tagging_info['GPSLongitude']
    longitude = convert_gps_to_degree(longitude)
    if longitude_ref == 'W':
        longitude = -longitude
    
    gps = (latitude, longitude)
    geolocator = Nominatim(user_agent="omniquery")
    location = geolocator.reverse(f"{latitude}, {longitude}")
    address = location.address
    address_split = address.split(', ')
    address_split.reverse()

    location_info = {}
    location_info['gps'] = gps
    location_info['address'] = address

    label_list = ['country', 'zip', 'state', 'county', 'city']
    for i, label in enumerate(label_list):
        if i >= len(address_split):
            break
        location_info[label] = address_split[i]

    return location_info

def read_metadata_from_image(image, filepath=""):
    try:
        exif_info = image._getexif()
    except:
        exif_info = image.getexif()
    exif = {ExifTags.TAGS.get(k, k): v for k, v in exif_info.items()}
    # print(exif)
    
    if 'GPSInfo' in exif:
        gps_data = read_GPS_from_image(image)
        capture_method = 'photo'
    else:
        gps_data = {}

        if 'UserComment' in exif:
            user_comment = exif['UserComment']
            if 'Screenshot' in user_comment.decode('utf-8'):
                capture_method = 'screenshot'
        else:
            capture_method = 'unknown'
        

    if 'DateTimeOriginal' in exif or 'DateTime' in exif:
        temporal_data = parse_date_time(exif)
    else:
        date_time = extract_date_time_modified(filepath)
        temporal_data = parse_date_time(exif=None, date_time_string=date_time)
    
    metadata = {
        'temporal_info': temporal_data,
        'location': gps_data,
        'capture_method': capture_method,
    }
    
    return metadata

def parse_date_time_exiftool(date_time):
    if 'PM' in date_time:
        date_time = date_time.replace('PM', '')
    if 'AM' in date_time:
        date_time = date_time.replace('AM', '')
    if '-' in date_time:
        date_time = date_time.split('-')[0]
    if '+' in date_time:
        date_time = date_time.split('+')[0]

    if '上午' in date_time:
        date_time = date_time.split('上午')[0]
    if '下午' in date_time:
        date_time = date_time.split('下午')[0]

    year = date_time.split(':')[0]
    month = date_time.split(':')[1]
    day_hour = date_time.split(':')[2]
    day = day_hour.split(' ')[0]
    hour = day_hour.split(' ')[1]
    minute = date_time.split(':')[3]
    second = date_time.split(':')[4]
    second = second[:2]

    date_time = f"{year}:{month}:{day} {hour}:{minute}:{second}"


        
    date_time_object = datetime.strptime(date_time, '%Y:%m:%d %H:%M:%S')
    day_of_week = date_time_object.strftime('%A')
    time_of_the_day = get_time_of_the_day(date_time_object.hour)

    date_info = {
        'date_string': date_time,
        'day_of_week': day_of_week,
        'time_of_the_day': time_of_the_day
    }
    return date_info

def read_gps_from_metadata_exiftool(metadata):
    gps = metadata[0].get('Composite:GPSPosition', None)
    if gps is None:
        return {}
    gps_latitude = metadata[0].get('Composite:GPSLatitude', None)
    gps_longitude = metadata[0].get('Composite:GPSLongitude', None)
    gps = (gps_latitude, gps_longitude)
    geolocator = Nominatim(user_agent="omniquery")
    location = geolocator.reverse(f"{gps_latitude}, {gps_longitude}")
    address = location.address
    address_split = address.split(', ')
    address_split.reverse()

    location_info = {}
    location_info['gps'] = gps
    location_info['address'] = address

    label_list = ['country', 'zip', 'state', 'county', 'city']
    for i, label in enumerate(label_list):
        if i >= len(address_split):
            break
        location_info[label] = address_split[i]
    return location_info

def read_capture_method_from_metadata_exiftool(metadata):
    meta = metadata[0]
    if 'image' in meta.get('File:MIMEType', ''):
        if 'EXIF:Model' in meta:
            model = meta['EXIF:Model']
            return 'photo'
        elif 'EXIF:UserComment' in meta:
            user_comment = meta['EXIF:UserComment']
            if 'Screenshot' in user_comment:
                return 'screenshot'
            else:
                return 'unknown'
        return 'unknown'
    else:
        return 'video'


def read_metadata_from_image_exiftool(image_path):
    with ExifToolHelper() as et:
        metadata = et.get_metadata(image_path)
        date = metadata[0].get('EXIF:DateTimeOriginal', None)
        if date is None:
            date = metadata[0].get('EXIF:DateTime', None)
        if date is None:
            date = metadata[0].get('File:FileModifyDate', None)

        date_info = parse_date_time_exiftool(date)

        try:
            location_info = read_gps_from_metadata_exiftool(metadata)
        except Exception as e:
            print(e)
            location_info = {}
        capture_method = read_capture_method_from_metadata_exiftool(metadata)
        
        metadata_result = {
            'temporal_info': date_info,
            'location': location_info,
            'capture_method': capture_method
        }
        return metadata_result

def read_metadata_from_video(video_path):
    with ExifToolHelper() as et:
        metadata = et.get_metadata(video_path)
        date = metadata[0].get('QuickTime:CreationDate', None)
        
        if date is None:
            date = metadata[0].get('File:FileModifyDate', None)

        date_info = parse_date_time_exiftool(date)
        location_info = read_gps_from_metadata_exiftool(metadata)
        capture_method = read_capture_method_from_metadata_exiftool(metadata)

        duration = metadata[0].get('QuickTime:Duration', None)
        fps = metadata[0].get('QuickTime:VideoFrameRate', None)

        metadata_result = {
            'temporal_info': date_info,
            'location': location_info,
            'capture_method': capture_method,
            'duration': duration,
            'fps': fps,
        }
        return metadata_result
