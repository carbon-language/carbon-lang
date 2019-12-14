cmake_minimum_required(VERSION 3.1.0)

include(CheckIncludeFiles)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(libbolt_rt_project)

check_include_files(elf.h HAVE_ELF_H)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/config.h.in
               ${CMAKE_CURRENT_BINARY_DIR}/config.h)

add_library(bolt_rt STATIC
  instr.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/config.h
  )

# Don't let the compiler think it can create calls to standard libs
target_compile_options(bolt_rt PRIVATE -ffreestanding -fno-exceptions -fno-rtti)
target_include_directories(bolt_rt PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

install(TARGETS bolt_rt DESTINATION lib)
