cmake_minimum_required(VERSION 3.16)
project(rt_denoiser CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-Wall -Wextra -O2)

# DSP headers (fft/stft/wav/ring_buffer) come from the DSP-Fundamentals-Toolkit
# submodule under this folder's lib/. Fetch it once with:
#   git submodule update --init --recursive
set(DSP_INC "${CMAKE_SOURCE_DIR}/lib/dsp-fundamentals-toolkit/include")
if(NOT EXISTS "${DSP_INC}/dsp/fft.hpp")
  message(FATAL_ERROR
    "DSP toolkit not found. Run: git submodule update --init --recursive")
endif()

# ONNX Runtime: download prebuilt package and point ORT_DIR at it, e.g.
#   wget https://github.com/microsoft/onnxruntime/releases/download/v1.18.0/onnxruntime-linux-x64-1.18.0.tgz
#   tar xzf onnxruntime-linux-x64-1.18.0.tgz
#   cmake -DORT_DIR=$PWD/../onnxruntime-linux-x64-1.18.0 ..
set(ORT_DIR "" CACHE PATH "Path to onnxruntime prebuilt package")
include_directories(${ORT_DIR}/include ${DSP_INC} ${CMAKE_SOURCE_DIR}/include)
link_directories(${ORT_DIR}/lib)

add_executable(offline_denoise src/offline_main.cpp)
target_link_libraries(offline_denoise onnxruntime)

# Real-time demo needs PortAudio: sudo apt install portaudio19-dev
find_library(PORTAUDIO portaudio)
if(PORTAUDIO)
  add_executable(realtime_denoise src/realtime_main.cpp)
  target_link_libraries(realtime_denoise onnxruntime ${PORTAUDIO} pthread)
endif()
