# Copyright 2017 Igalia S.L. All Rights Reserved.
#
# Distributed under MIT license.
# See file LICENSE for detail or copy at https://opensource.org/licenses/MIT

# Ubuntu 12.04 LTS has CMake 2.8.7, and is an important target since
# several CI services, such as Travis and Drone, use it.  Solaris 11
# has 2.8.6, and it's not difficult to support if you already have to
# support 2.8.7.
cmake_minimum_required(VERSION 2.8.6)

project(woff2)

include(GNUInstallDirs)

# Build options
option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(CANONICAL_PREFIXES "Canonical prefixes" OFF)
option(NOISY_LOGGING "Noisy logging" ON)

# Version information
set(WOFF2_VERSION 1.0.2)

# When building shared libraries it is important to set the correct rpath
# See https://cmake.org/Wiki/CMake_RPATH_handling#Always_full_RPATH
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
list(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_LIBDIR}" isSystemDir)
if ("${isSystemDir}" STREQUAL "-1")
  set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_LIBDIR}")
endif()

# Find Brotli dependencies
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(BrotliDec)
if (NOT BROTLIDEC_FOUND)
    message(FATAL_ERROR "librotlidec is needed to build woff2.")
endif ()
find_package(BrotliEnc)
if (NOT BROTLIENC_FOUND)
    message(FATAL_ERROR "librotlienc is needed to build woff2.")
endif ()

# Set compiler flags
if (NOT CANONICAL_PREFIXES)
    add_definitions(-no-canonical-prefixes)
  endif ()
if (NOISY_LOGGING)
    add_definitions(-DFONT_COMPRESSION_BIN)
endif ()
add_definitions(-D__STDC_FORMAT_MACROS)
set(COMMON_FLAGS -fno-omit-frame-pointer)

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    add_definitions(-DOS_MACOSX)
else ()
    set(COMMON_FLAGS "${COMMON_FLAG} -fno-omit-frame-pointer")
endif()

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${COMMON_FLAG}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${COMMON_FLAG}")
set(CMAKE_CXX_STANDARD 11)

# Set search path for our private/public headers as well as Brotli headers
include_directories("src" "include"
                    "${BROTLIDEC_INCLUDE_DIRS}" "${BROTLIENC_INCLUDE_DIRS}")

# Common part used by decoder and encoder
add_library(woff2common
            src/table_tags.cc
            src/variable_length.cc
            src/woff2_common.cc)

# WOFF2 Decoder
add_library(woff2dec
            src/woff2_dec.cc
            src/woff2_out.cc)
target_link_libraries(woff2dec woff2common "${BROTLIDEC_LIBRARIES}")
add_executable(woff2_decompress src/woff2_decompress.cc)
target_link_libraries(woff2_decompress woff2dec)

# WOFF2 Encoder
add_library(woff2enc
            src/font.cc
            src/glyph.cc
            src/normalize.cc
            src/transform.cc
            src/woff2_enc.cc)
target_link_libraries(woff2enc woff2common "${BROTLIENC_LIBRARIES}")
add_executable(woff2_compress src/woff2_compress.cc)
target_link_libraries(woff2_compress woff2enc)

# WOFF2 info
add_executable(woff2_info src/woff2_info.cc)
target_link_libraries(woff2_info woff2common)

foreach(lib woff2common woff2dec woff2enc)
  set_target_properties(${lib} PROPERTIES
    SOVERSION ${WOFF2_VERSION}
    VERSION ${WOFF2_VERSION}
    POSITION_INDEPENDENT_CODE TRUE)
endforeach()

# Fuzzer libraries
add_library(convert_woff2ttf_fuzzer STATIC src/convert_woff2ttf_fuzzer.cc)
target_link_libraries(convert_woff2ttf_fuzzer woff2dec)
add_library(convert_woff2ttf_fuzzer_new_entry STATIC src/convert_woff2ttf_fuzzer_new_entry.cc)
target_link_libraries(convert_woff2ttf_fuzzer_new_entry woff2dec)

# PC files
include(CMakeParseArguments)

function(generate_pkg_config_path outvar path)
  string(LENGTH "${path}" path_length)

  set(path_args ${ARGV})
  list(REMOVE_AT path_args 0 1)
  list(LENGTH path_args path_args_remaining)

  set("${outvar}" "${path}")

  while(path_args_remaining GREATER 1)
    list(GET path_args 0 name)
    list(GET path_args 1 value)

    get_filename_component(value_full "${value}" ABSOLUTE)
    string(LENGTH "${value}" value_length)

    if(path_length EQUAL value_length AND path STREQUAL value)
      set("${outvar}" "\${${name}}")
      break()
    elseif(path_length GREATER value_length)
      # We might be in a subdirectory of the value, but we have to be
      # careful about a prefix matching but not being a subdirectory
      # (for example, /usr/lib64 is not a subdirectory of /usr/lib).
      # We'll do this by making sure the next character is a directory
      # separator.
      string(SUBSTRING "${path}" ${value_length} 1 sep)
      if(sep STREQUAL "/")
        string(SUBSTRING "${path}" 0 ${value_length} s)
        if(s STREQUAL value)
          string(SUBSTRING "${path}" "${value_length}" -1 suffix)
          set("${outvar}" "\${${name}}${suffix}")
          break()
        endif()
      endif()
    endif()

    list(REMOVE_AT path_args 0 1)
    list(LENGTH path_args path_args_remaining)
  endwhile()

  set("${outvar}" "${${outvar}}" PARENT_SCOPE)
endfunction(generate_pkg_config_path)

function(generate_pkg_config output_file)
  set (options)
  set (oneValueArgs NAME DESCRIPTION URL VERSION PREFIX LIBDIR INCLUDEDIR)
  set (multiValueArgs DEPENDS DEPENDS_PRIVATE CFLAGS LIBRARIES)
  cmake_parse_arguments(GEN_PKG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  unset (options)
  unset (oneValueArgs)
  unset (multiValueArgs)

  if(NOT GEN_PKG_PREFIX)
    set(GEN_PKG_PREFIX "${CMAKE_INSTALL_PREFIX}")
  endif()

  if(NOT GEN_PKG_LIBDIR)
    set(GEN_PKG_LIBDIR "${CMAKE_INSTALL_FULL_LIBDIR}")
  endif()
  generate_pkg_config_path(GEN_PKG_LIBDIR "${GEN_PKG_LIBDIR}"
    prefix "${GEN_PKG_PREFIX}")

  if(NOT GEN_PKG_INCLUDEDIR)
    set(GEN_PKG_INCLUDEDIR "${CMAKE_INSTALL_FULL_INCLUDEDIR}")
  endif()
  generate_pkg_config_path(GEN_PKG_INCLUDEDIR "${GEN_PKG_INCLUDEDIR}"
    prefix "${GEN_PKG_PREFIX}")

  file(WRITE  "${output_file}" "prefix=${GEN_PKG_PREFIX}\n")
  file(APPEND "${output_file}" "libdir=${GEN_PKG_LIBDIR}\n")
  file(APPEND "${output_file}" "includedir=${GEN_PKG_INCLUDEDIR}\n")
  file(APPEND "${output_file}" "\n")

  if(GEN_PKG_NAME)
    file(APPEND "${output_file}" "Name: ${GEN_PKG_NAME}\n")
  else()
    file(APPEND "${output_file}" "Name: ${CMAKE_PROJECT_NAME}\n")
  endif()

  if(GEN_PKG_DESCRIPTION)
    file(APPEND "${output_file}" "Description: ${GEN_PKG_DESCRIPTION}\n")
  endif()

  if(GEN_PKG_URL)
    file(APPEND "${output_file}" "URL: ${GEN_PKG_URL}\n")
  endif()

  if(GEN_PKG_VERSION)
    file(APPEND "${output_file}" "Version: ${GEN_PKG_VERSION}\n")
  endif()

  if(GEN_PKG_DEPENDS)
    file(APPEND "${output_file}" "Requires: ${GEN_PKG_DEPENDS}\n")
  endif()

  if(GEN_PKG_DEPENDS_PRIVATE)
    file(APPEND "${output_file}" "Requires.private:")
    foreach(lib ${GEN_PKG_DEPENDS_PRIVATE})
      file(APPEND "${output_file}" " ${lib}")
    endforeach()
    file(APPEND "${output_file}" "\n")
  endif()

  if(GEN_PKG_LIBRARIES)
    set(libs)

    file(APPEND "${output_file}" "Libs: -L\${libdir}")
    foreach(lib ${GEN_PKG_LIBRARIES})
      file(APPEND "${output_file}" " -l${lib}")
    endforeach()
    file(APPEND "${output_file}" "\n")
  endif()

  file(APPEND "${output_file}" "Cflags: -I\${includedir}")
  if(GEN_PKG_CFLAGS)
    foreach(cflag ${GEN_PKG_CFLAGS})
      file(APPEND "${output_file}" " ${cflag}")
    endforeach()
  endif()
  file(APPEND "${output_file}" "\n")
endfunction(generate_pkg_config)

generate_pkg_config ("${CMAKE_CURRENT_BINARY_DIR}/libwoff2common.pc"
  NAME libwoff2common
  DESCRIPTION "Shared data used by libwoff2 and libwoff2dec libraries"
  URL "https://github.com/google/woff2"
  VERSION "${WOFF2_VERSION}"
  LIBRARIES woff2common)

generate_pkg_config ("${CMAKE_CURRENT_BINARY_DIR}/libwoff2dec.pc"
  NAME libwoff2dec
  DESCRIPTION "WOFF2 decoder library"
  URL "https://github.com/google/woff2"
  VERSION "${WOFF2_VERSION}"
  DEPENDS libbrotlidec
  DEPENDS_PRIVATE libwoff2common
  LIBRARIES woff2dec)

generate_pkg_config ("${CMAKE_CURRENT_BINARY_DIR}/libwoff2enc.pc"
  NAME libwoff2enc
  DESCRIPTION "WOFF2 encoder library"
  URL "https://github.com/google/woff2"
  VERSION "${WOFF2_VERSION}"
  DEPENDS libbrotlienc
  DEPENDS_PRIVATE libwoff2common
  LIBRARIES woff2enc)

# Installation
if (NOT BUILD_SHARED_LIBS)
  install(
    TARGETS woff2_decompress woff2_compress woff2_info
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
  )
endif()

install(
  TARGETS woff2common woff2dec woff2enc
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
)
install(
  DIRECTORY include/woff2
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/libwoff2common.pc"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/pkgconfig")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/libwoff2dec.pc"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/pkgconfig")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/libwoff2enc.pc"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/pkgconfig")
