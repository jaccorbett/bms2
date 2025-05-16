# Import packages

import numpy as np
import matplotlib.pyplot as plt
import pystac_client
from odc.geo.geom import BoundingBox
import datetime
import matplotlib.pyplot as plt
from matplotlib import gridspec
import odc.stac
from datetime import datetime, timedelta
import joblib
import streamlit as st
import geopandas as gpd
import pandas as pd
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import rioxarray


from pyproj import CRS
print(CRS.from_epsg(4326))

# Title
st.title("Blue Magic - Sentinel-2")
st.header("Water Clarity Prediction") 

# Define variables
# Define AOI
arc_point_lat = -28.27 #tweed
arc_point_lon = 153.51

point_lat = -28.176304005228644
point_lon = 153.5671747765546

buffer = 0.1

x = (point_lat - buffer, point_lat + buffer)
y = (point_lon - buffer, point_lon + buffer)

bbox = BoundingBox.from_xy(y, x)


# Date input
date_value = datetime.today()
date = st.date_input("Select date", value=date_value, min_value=datetime(2020, 1, 1), max_value=datetime.today())

date_range = f"{date.strftime('%Y-%m-%d')}/{date.strftime('%Y-%m-%d')}"
print(f"Date range: {date_range}")

dea_catalog = pystac_client.Client.open("https://explorer.dea.ga.gov.au/stac")

odc.stac.configure_rio(
    cloud_defaults=True,
    aws={"aws_unsigned": True},
)

s2a_collection = ["ga_s2am_ard_3"]
s2b_collection = ['ga_s2bm_ard_3']
s2c_collection = ['ga_s2cm_ard_3']

all_bands = [
    "nbart_coastal_aerosol",
    "nbart_red",
    "nbart_green",
    "nbart_blue",
    "nbart_red_edge_1",
    "nbart_red_edge_2",
    "nbart_red_edge_3",
    "nbart_nir_1",
    "nbart_nir_2",
    "nbart_swir_2",
    "nbart_swir_3",
    "oa_fmask",
    "oa_s2cloudless_mask",
    "oa_s2cloudless_prob",
]

s2_query = dea_catalog.search(
    bbox=bbox,
    collections=s2a_collection + s2b_collection + s2c_collection,
    datetime=date_range,
    filter="eo:cloud_cover < 100"
)

s2_items = list(s2_query.items())
print(f"Found: {len(s2_items):d} datasets")

if len(s2_items) == 0:
    print("No Sentinel-2 imagery found for the selected date range.")
    st.error("No Sentinel-2 imagery found for the selected date.")
    
    # Expand the date range and find the closest date
    exp_date_range = f"{(date - timedelta(days=5)).strftime('%Y-%m-%d')}/{(date + timedelta(days=5)).strftime('%Y-%m-%d')}"
    
    s2_query = dea_catalog.search(
        bbox=bbox,
        collections=s2a_collection + s2b_collection + s2c_collection,
        datetime=exp_date_range,
        filter="eo:cloud_cover < 100"
    )
    s2_items = list(s2_query.items())
    st.write(f"Found {len(s2_items)} Sentinel-2 images in the expanded date range.")
    print(f"Found: {len(s2_items):d} datasets")
    for item in s2_items:
        print(item.datetime)
        st.write(item.datetime)
    
    st.stop()  # Stop the script execution
    raise SystemExit("Exiting script.")



# Print all dates of the items
for item in s2_items:
    print(item.datetime)
    item = s2_items[-1]


s2_date_str = item.datetime.strftime('%d %b %Y')
s2_time_str = (item.datetime + timedelta(hours=10)).strftime('%I:%M%p').lower()

st.write(f"**Image Date and Time:** {s2_date_str} ({s2_time_str})")
st.write(f"Cloud Cover: {item.properties['eo:cloud_cover']:.2f}%")

s2 = odc.stac.load(
    [item],
    bands=all_bands, # Change to run model
    crs="utm",
    resolution=30,
    #groupby="solar_day",
    bbox=bbox,
)


s2_img = s2.isel(time=0)
s2_img = s2_img.where(s2_img.oa_fmask == 5)
s2_img_rgb = s2_img[['nbart_red', 'nbart_green', 'nbart_blue']].to_array(dim="color").compute()


mlp = joblib.load('./inputs/zsd_model.pkl')
scaler = joblib.load('./inputs/scaler.pkl')


# Extract the data variables and reshape
data_vars = [
    'nbart_coastal_aerosol', 'nbart_red', 'nbart_green', 'nbart_blue',
    'nbart_red_edge_1', 'nbart_red_edge_2', 'nbart_red_edge_3',
    'nbart_nir_1', 'nbart_nir_2', 'nbart_swir_2', 'nbart_swir_3'
]

# Stack the data variables into a 2D array (samples, features)
data = np.stack([s2_img[var].values.flatten() for var in data_vars], axis=-1)

# Fill all NaN values with 0
data = np.nan_to_num(data)

# Apply MLP model 
data = scaler.transform(data) # Only use for MLP model

predictions = mlp.predict(data)
predictions_reshaped = predictions.reshape(s2.dims['time'], s2.dims['y'], s2.dims['x'])

s2['zsd_predicted'] = (('time', 'y', 'x'), predictions_reshaped)



# Plot S2 ZSD image
# Load the image
img = s2.isel(time=0)
img = img.where(img.oa_fmask == 5)

# Create a figure with higher resolution
fig = plt.figure(figsize=(16, 8), dpi=150)
gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.05])

# Plot the RGB image
ax1 = fig.add_subplot(gs[0])
s2_img_rgb.plot.imshow(ax=ax1, robust=True)
ax1.set_title('')  # Remove title
ax1.set_xlabel('')  # Remove x-axis label
ax1.set_ylabel('')  # Remove y-axis label
ax1.set_xticks([])  # Remove x-axis ticks
ax1.set_yticks([])  # Remove y-axis ticks
for spine in ax1.spines.values():  # Remove plot borders
    spine.set_visible(False)

# Plot the ZSD map
ax2 = fig.add_subplot(gs[1])
zsd_plot = img.zsd_predicted.plot.imshow(ax=ax2, robust=True, cmap='jet_r', vmin=0, vmax=20, add_colorbar=False)
ax2.set_title('')  # Remove title
ax2.set_xlabel('')  # Remove x-axis label
ax2.set_ylabel('')  # Remove y-axis label
ax2.set_xticks([])  # Remove x-axis ticks
ax2.set_yticks([])  # Remove y-axis ticks
for spine in ax2.spines.values():  # Remove plot borders
    spine.set_visible(False)

# Customize colorbar for ZSD map
cbar_ax = fig.add_subplot(gs[2])
cbar = plt.colorbar(zsd_plot, cax=cbar_ax)
cbar.ax.tick_params(labelsize=10)  # Keep colorbar ticks
cbar.set_label('Zsd predicted (m)', fontsize=10)
cbar.set_ticks([0, 3, 6, 9, 12, 15, 18])

plt.tight_layout()

# Display the plot in Streamlit
st.pyplot(fig)



###############     SUMMARISE DIVE SITES   ######################

st.subheader("Dive Sites")


dive_locations_pts = gpd.read_file('./inputs/dive_sites/dive_sites_pts.shp')
dive_locations_pts = dive_locations_pts.to_crs(epsg=32756)

dive_locations_poly = gpd.read_file('./inputs/dive_sites/dive_sites_updated.shp')
dive_locations_poly = dive_locations_poly.to_crs(epsg=32756)

#Sort by fid
dive_locations_poly = dive_locations_poly.sort_values(by='fid')
dive_locations_poly.set_index('fid', inplace=True)

# Filter dive_locations_poly to only include: Kingscliff Reef, Cook Island, Nine Mile Reef, Kirra Reef, Palm Beach Reef
dive_locations_poly = dive_locations_poly[dive_locations_poly['reef_name'].isin(['Kingscliff Reef', 'Cook Island', 'Nine Mile Reef', 'Kirra Reef', 'Palm Beach Reef'])]


# Calculate the mean secchi depth for each dive_locations_poly
secchi_depths = []

for i in range(len(dive_locations_poly)):
    # Get the geometry of the dive site
    geom = dive_locations_poly.iloc[i]['geometry']
    minx, miny, maxx, maxy = geom.bounds
    #site_bbox = [[minx, miny], [maxx, maxy]]
    site = dive_locations_poly.iloc[i]
    st.markdown(f"**{site['reef_name']}**")
    date = str(img.time.values)[:10]
    print(f"Processing {site['reef_name']} on {date}")
    
    # Crop the RGB image using the geometry of the polygon
    img_rgb = img[['nbart_red', 'nbart_green', 'nbart_blue']].to_array(dim="color").compute()    
    #cropped_rgb = img_rgb.rio.clip([geom], from_disk=True)
    #cropped_zsd = img.secchi_depth_predicted.rio.clip([geom])
    
    cropped_rgb = img_rgb.rio.clip_box(minx=minx, miny=miny, maxx=maxx, maxy=maxy)
    cropped_zsd = img.zsd_predicted.rio.clip_box(minx=minx, miny=miny, maxx=maxx, maxy=maxy)
    
    # Calculate the mean, median, min, and max secchi depth
    mean_secchi_depth = cropped_zsd.mean().values.item()
    median_secchi_depth = cropped_zsd.median().values.item()
    min_secchi_depth = cropped_zsd.min().values.item()
    max_secchi_depth = cropped_zsd.max().values.item()
    
    
    # Calculate the perc pixels that are NaN
    total_pixels = cropped_zsd.size
    nan_pixels = np.isnan(cropped_zsd).sum().values.item()
    nan_percentage = (nan_pixels / total_pixels) * 100
    
    
    # Filter the points that are within the bounding box of the current dive site 
    points_within_bounds = dive_locations_pts.cx[minx:maxx, miny:maxy]
    
    x_coords = points_within_bounds.geometry.x
    y_coords = points_within_bounds.geometry.y
    
    # Sample the secchi depth values at the points
    point_secchi_depth = [cropped_zsd.sel(x=x, y=y, method='nearest').values.item() for x, y in zip(x_coords, y_coords)]
    point_secchi_depth = point_secchi_depth[0]

    # Append the secchi depth statistics to the dictionary
    secchi_depths.append({
        'location': site['reef_name'],
        'date': date,
        'point_zsd': point_secchi_depth,
        'mean_zsd': mean_secchi_depth,
        'median_zsd': median_secchi_depth,
        'min_zsd': min_secchi_depth,
        'max_zsd': max_secchi_depth,
        'nan_perc': nan_percentage,
    })
    
    
    # Print the secchi depth statistics
    print(f"Point Zsd: {point_secchi_depth:.2f} m")
    print(f"Mean Zsd: {mean_secchi_depth:.2f} m")
    print(f"Median Zsd: {median_secchi_depth:.2f} m")
    print(f"Min Zsd: {min_secchi_depth:.2f} m")
    print(f"Max Zsd: {max_secchi_depth:.2f} m")
    print(f"NaN percentage: {nan_percentage:.2f}%")
    

    # Create a figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
    
    # Plot the RGB image
    cropped_rgb.plot.imshow(ax=ax1, robust=True)
    ax1.set_title(f"Sentinel-2 RGB: {date}", fontsize=22)
    points_within_bounds.plot(ax=ax1, marker='o', facecolor='none', edgecolor='black', markersize=40, linewidth=1, label='Site')
    ax1.set_aspect('auto')
    ax1.set_xlabel('')
    ax1.set_ylabel('')
    ax1.tick_params(axis='both', which='major', labelsize=12)
    
    
    
    # Plot the Zsd predicted raster data
    im = cropped_zsd.plot.imshow(ax=ax2, robust=True, cmap='jet_r', vmin=0, vmax=20, add_colorbar=False)
    ax2.set_title(f"Secchi Depth: {site['reef_name']}", fontsize=22)
    points_within_bounds.plot(ax=ax2, marker='o', facecolor='none', edgecolor='black', markersize=40, linewidth=1, label='Site')
    ax2.set_aspect('auto')
    ax2.set_xlabel('')
    ax2.set_ylabel('')
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])

 
    cbar = fig.colorbar(im, ax=ax2, orientation='vertical')
    cbar.set_label('Secchi Depth (m)', fontsize=20)
    cbar.set_ticks([0, 5, 10, 15, 20])
    
  # Plot the histogram of secchi depth values if not all NaN
    if not np.isnan(cropped_zsd.values).all():
        ax3.hist(cropped_zsd.values.flatten(), bins=30, color='lightblue')
        ax3.set_title(f"Secchi Depth Histogram", fontsize=22)
        ax3.set_xlabel("Secchi Depth (m)", fontsize=20)
        ax3.set_ylabel("Frequency", fontsize=20)
        
        # Add mean and median lines to histogram
        ax3.axvline(mean_secchi_depth, color='red', linewidth=1, label='Zonal Mean')
        ax3.axvline(median_secchi_depth, color='green', linewidth=1, label='Zonal Median')
        ax3.axvline(point_secchi_depth, color='black', linewidth=1, label='Point Zsd')
        ax3.legend(fontsize=20)
        ax3.legend()
        
        # Add text annotations for the secchi depth statistics
        text_x_position = ax3.get_xlim()[1] * 0.97
        text_y_position = ax3.get_ylim()[1] * 0.55  
        vertical_spacing = (ax3.get_ylim()[1] - ax3.get_ylim()[0]) * 0.04
        
        ax3.text(text_x_position, text_y_position, f'Mean: {mean_secchi_depth:.2f} m', color='red', ha='right', fontsize=16)
        ax3.text(text_x_position, text_y_position - vertical_spacing, f'Median: {median_secchi_depth:.2f} m', color='green', ha='right', fontsize=16)
        ax3.text(text_x_position, text_y_position - 2 * vertical_spacing, f'Point: {point_secchi_depth:.2f} m', color='black', ha='right', fontsize=16)
        ax3.text(text_x_position, text_y_position - 3 * vertical_spacing, f'Min: {min_secchi_depth:.2f} m', color='blue', ha='right', fontsize=16)
        ax3.text(text_x_position, text_y_position - 4 * vertical_spacing, f'Max: {max_secchi_depth:.2f} m', color='purple', ha='right', fontsize=16)
        ax3.text(text_x_position, text_y_position - 5 * vertical_spacing, f'NaN %: {nan_percentage:.2f}%', color='orange', ha='right', fontsize=16)
        
        ax3.tick_params(axis='both', which='major', labelsize=12)
    else:
        ax3.set_title(f"No valid Secchi Depth data", fontsize=22)
        ax3.set_xlabel("")
        ax3.set_ylabel("")
    
    
    # Show the plot
    plt.tight_layout()
    st.pyplot(fig)


st.subheader("Data Summary")

secchi_depth_df = pd.DataFrame(secchi_depths)
secchi_depth_df


# Filter out locations with nan_perc > 50
filtered_secchi_depth_df = secchi_depth_df[secchi_depth_df['nan_perc'] <= 50]

# Normalize the median_zsd values to the range [0, 15]
norm = Normalize(vmin=0, vmax=15)
cmap = cm.get_cmap('jet_r')

# Map each location's median_zsd to a color
colors = filtered_secchi_depth_df['median_zsd'].apply(lambda zsd: cmap(norm(zsd)))

# Plot secchi depth predicted at dive sites - bar plot
fig, ax = plt.subplots()
filtered_secchi_depth_df.plot.bar(
    x='location', 
    y='median_zsd', 
    color=colors, 
    legend=False, 
    ax=ax
)
ax.set_title('')
ax.set_ylabel('Zsd predicted (m)', fontsize=8)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
ax.set_xlabel('')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=8)
st.pyplot(fig)