cmake_minimum_required(VERSION 3.18)
project(fast_elec LANGUAGES CXX)

find_package(pybind11 CONFIG REQUIRED)

# fast_elec.cpp はプロジェクトルートに置く前提
pybind11_add_module(fast_elec src/fast_elec.cpp)

target_compile_features(fast_elec PRIVATE cxx_std_17)

install(TARGETS fast_elec
  LIBRARY DESTINATION .
  RUNTIME DESTINATION .
)

# 例: NumPy のヘッダが必要なら
# find_package(Python3 COMPONENTS Development NumPy REQUIRED)
# target_include_directories(fast_elec PRIVATE ${Python3_NumPy_INCLUDE_DIRS})
