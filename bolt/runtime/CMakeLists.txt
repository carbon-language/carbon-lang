cmake_minimum_required(VERSION 3.1.0)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(libbolt_rt_project)

add_library(bolt_rt STATIC
  instr.cpp
  )

install(TARGETS bolt_rt DESTINATION lib)
