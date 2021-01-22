import seeed_mlx9064x
#mlx = seeed_mlx9064x.grove_mxl90640()
mlx = seeed_mlx9064x.grove_mxl90641()
mlx.refresh_rate = seeed_mlx9064x.RefreshRate.REFRESH_8_HZ  # The fastest for raspberry 4 
# REFRESH_0_5_HZ = 0b000  # 0.5Hz
# REFRESH_1_HZ = 0b001  # 1Hz
# REFRESH_2_HZ = 0b010  # 2Hz
# REFRESH_4_HZ = 0b011  # 4Hz
# REFRESH_8_HZ = 0b100  # 8Hz
# REFRESH_16_HZ = 0b101  # 16Hz
# REFRESH_32_HZ = 0b110  # 32Hz
# REFRESH_64_HZ = 0b111  # 64Hz

while True:
    try:
        #frame = [0]*768
        frame = [0]*192
        mlx.getFrame(frame)
        print("Drawing frame..")
    except ValueError:
        print("Fail")
        continue
