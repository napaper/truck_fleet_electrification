from matplotlib.colors import  ListedColormap
import colorsys
# set textwidth for plots
# 6.32 = FTM Thesis
textwidth = 6.32 
h_169 = textwidth * 9 / 16
h_43 = textwidth * 3 / 4

# Define alpha values for grid lines
alpha_major = .8
alpha_minor = .4

# Colors definition
colors = {
    'TUMBlack': '#000000',
    'TUMWhite': '#FFFFFF',
    'TUMBlue1': '#005293',
    'TUMOrange': '#E37222', # 2
    'TUMBlue2': '#3070b3',  # 3
    'TUMGreen3': '#A2AD00', # 4 (light green)
    'TUMBlue3': '#64A0C8',
    'TUMGreen2': '#515600', # 6 (dark green)
    'TUMBlue4': '#98C6EA',  # light blue
    'TUMGray1': '#333333',  # dark 
    'TUMBlue5': '#B0D0F0',
    'TUMGray2': '#7F7F7F',  # 10 medium
    'TUMBlue6': '#2A4D70',
    'TUMGray3': '#CCCCCC',  # light
    'TUMBlueDark': '#0B1340',
    'TUMIvory': '#DAD7CB',
    'TUMGreen1': '#292B00',
    'Purple': '#B3679B',    # 15
    'LightPurple': "#020202",
    'DarkPurple': '#31081F',
    'LightYellow': '#FFE548', #18
    'Rust': '#885053'         #19
}

# Remove white and light gray colors from the palette
line_colors = {k: v for k, v in colors.items() if v not in ['#FFFFFF', '#CCCCCC']}

# Function to calculate brightness
def brightness(hex_color):
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (1, 3, 5))
    hls = colorsys.rgb_to_hls(rgb[0]/255, rgb[1]/255, rgb[2]/255)
    return hls[1]  # Luminance value

# Sort color_list by brightness
color_list = list(line_colors.values())
color_list_sorted = sorted(color_list, key=brightness, reverse=True)

colors_plot = list(color_list[i] for i in [10, 2, 3, 4, 15, 18, 19, 1, 6])

cmap = ListedColormap(color_list_sorted[:97])
