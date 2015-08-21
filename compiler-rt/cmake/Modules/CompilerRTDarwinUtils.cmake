# On OS X SDKs can be installed anywhere on the base system and xcode-select can
# set the default Xcode to use. This function finds the SDKs that are present in
# the current Xcode.
function(find_darwin_sdk_dir var sdk_name)
  # Let's first try the internal SDK, otherwise use the public SDK.
  execute_process(
    COMMAND xcodebuild -version -sdk ${sdk_name}.internal Path
    OUTPUT_VARIABLE var_internal
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_FILE /dev/null
  )
  if("" STREQUAL "${var_internal}")
    execute_process(
      COMMAND xcodebuild -version -sdk ${sdk_name} Path
      OUTPUT_VARIABLE var_internal
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_FILE /dev/null
    )
  endif()
  set(${var} ${var_internal} PARENT_SCOPE)
endfunction()

# There isn't a clear mapping of what architectures are supported with a given
# target platform, but ld's version output does list the architectures it can
# link for.
function(darwin_get_toolchain_supported_archs output_var)
  execute_process(
    COMMAND ld -v
    ERROR_VARIABLE LINKER_VERSION)

  string(REGEX MATCH "configured to support archs: ([^\n]+)"
         ARCHES_MATCHED "${LINKER_VERSION}")
  if(ARCHES_MATCHED)
    set(ARCHES "${CMAKE_MATCH_1}")
    message(STATUS "Got ld supported ARCHES: ${ARCHES}")
    string(REPLACE " " ";" ARCHES ${ARCHES})
  else()
    # If auto-detecting fails, fall back to a default set
    message(WARNING "Detecting supported architectures from 'ld -v' failed. Returning default set.")
    set(ARCHES "i386;x86_64;armv7;armv7s;arm64")
  endif()
  
  set(${output_var} ${ARCHES} PARENT_SCOPE)
endfunction()

# This function takes an OS and a list of architectures and identifies the
# subset of the architectures list that the installed toolchain can target.
function(darwin_test_archs os valid_archs)
  set(archs ${ARGN})
  message(STATUS "Finding valid architectures for ${os}...")
  set(SIMPLE_CPP ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/src.cpp)
  file(WRITE ${SIMPLE_CPP} "#include <iostream>\nint main() { return 0; }\n")

  set(os_linker_flags)
  foreach(flag ${DARWIN_${os}_LINKFLAGS})
    set(os_linker_flags "${os_linker_flags} ${flag}")
  endforeach()

  # The simple program will build for x86_64h on the simulator because it is 
  # compatible with x86_64 libraries (mostly), but since x86_64h isn't actually
  # a valid or useful architecture for the iOS simulator we should drop it.
  if(${os} STREQUAL "iossim")
    list(REMOVE_ITEM archs "x86_64h")
  endif()

  set(working_archs)
  foreach(arch ${archs})
    
    set(arch_linker_flags "-arch ${arch} ${os_linker_flags}")
    try_compile(CAN_TARGET_${os}_${arch} ${CMAKE_BINARY_DIR} ${SIMPLE_CPP}
                COMPILE_DEFINITIONS "-v -arch ${arch}" ${DARWIN_${os}_CFLAGS}
                CMAKE_FLAGS "-DCMAKE_EXE_LINKER_FLAGS=${arch_linker_flags}"
                OUTPUT_VARIABLE TEST_OUTPUT)
    if(${CAN_TARGET_${os}_${arch}})
      list(APPEND working_archs ${arch})
    endif()
  endforeach()
  set(${valid_archs} ${working_archs} PARENT_SCOPE)
endfunction()

# This function checks the host cpusubtype to see if it is post-haswell. Haswell
# and later machines can run x86_64h binaries. Haswell is cpusubtype 8.
function(darwin_filter_host_archs input output)
  list_union(tmp_var DARWIN_osx_ARCHS ${input})
  execute_process(
    COMMAND sysctl hw.cpusubtype
    OUTPUT_VARIABLE SUBTYPE)

  string(REGEX MATCH "hw.cpusubtype: ([0-9]*)"
         SUBTYPE_MATCHED "${SUBTYPE}")
  set(HASWELL_SUPPORTED Off)
  if(SUBTYPE_MATCHED)
    if(${CMAKE_MATCH_1} GREATER 7)
      set(HASWELL_SUPPORTED On)
    endif()
  endif()
  if(NOT HASWELL_SUPPORTED)
    list(REMOVE_ITEM tmp_var x86_64h)
  endif()
  set(${output} ${tmp_var} PARENT_SCOPE)
endfunction()
