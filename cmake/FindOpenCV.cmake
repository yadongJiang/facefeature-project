
#opencv

set(OpenCV_FOUND 1)
set(OpenCV_DIR /usr/local/opencv-3.3.0/share/OpenCV)
# set(OpenCV_DIR /home/oyrq/opencv-3.3.1/build)

find_package(OpenCV REQUIRED)

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")