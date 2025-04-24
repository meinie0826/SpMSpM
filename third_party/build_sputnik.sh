

cd ${SpInfer_HOME}/third_party/sputnik  && mkdir build && cd build
cmake .. -DGLOG_INCLUDE_DIR=/usr/include/glog -DGLOG_LIBRARY=/usr/lib/x86_64-linux-gnu/libglog.so -DCMAKE_BUILD_TYPE=Release -DBUILD_TEST=OFF -DBUILD_BENCHMARK=OFF -DCUDA_ARCHS="89;86;80"
make -j12 

