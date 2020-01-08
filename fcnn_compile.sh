third_party_library=/home/yipeng/thirdlib

g++ -g -O3 -fopenmp test_fcnn.cpp src/*.cpp -o fcnn \
-I ./include -I $third_party_library/glog/include/ -I $third_party_library/gflags/include/ \
-I $third_party_library/opencv/include/ -L $third_party_library/glog/lib/ \
-L $third_party_library/gflags/lib/ -L $third_party_library/opencv/lib/ \
-lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -lopencv_objdetect \
-lopencv_videoio -lglog -lgflags -std=gnu++11 -lpthread \
&& ./fcnn -flagfile=fcnn_flagfile_configure  && rm fcnn
