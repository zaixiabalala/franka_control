import pyrealsense2 as rs

L515_ID = 'f0172289'
L515_STREAMS = [
    (rs.stream.depth, 640, 480, rs.format.z16, 30), 
    (rs.stream.color, 960, 540, rs.format.bgr8, 30)
]

T265_ID = '230222110234'
T265_STREAMS = [
    (rs.stream.pose, rs.format.six_dof, 200), 
    (rs.stream.fisheye, 1, 848, 800, rs.format.y8, 30), 
    (rs.stream.fisheye, 2, 848, 800, rs.format.y8, 30)
]

D415_ID = '104122063633'
D415_STREAMS = [
    (rs.stream.depth, 640, 480, rs.format.z16, 15), 
    (rs.stream.color, 640, 480, rs.format.bgr8, 15)
]
