# This tool creates a shared library from the LLVM libraries. Generating this
# library is enabled by setting LLVM_BUILD_LLVM_DYLIB=yes on the CMake
# commandline. By default the shared library only exports the LLVM C API.

add_definitions( -DLLVM_VERSION_INFO=\"${PACKAGE_VERSION}\" )

set(SOURCES
  libllvm.cpp
  )

llvm_map_components_to_libnames(LIB_NAMES ${LLVM_DYLIB_COMPONENTS})

if(LLVM_LINK_LLVM_DYLIB)
  if(NOT LLVM_DYLIB_EXPORT_ALL)
    message(FATAL_ERROR "LLVM_DYLIB_EXPORT_ALL must be ON when LLVM_LINK_LLVM_DYLIB is ON")
  endif()

  # libLLVM.so should not have any dependencies on any other LLVM
  # shared libraries. When using the "all" pseudo-component,
  # LLVM_AVAILABLE_LIBS is added to the dependencies, which may
  # contain shared libraries (e.g. libLTO).
  #
  # Also exclude libLLVMTableGen for the following reasons:
  #  - it is only used by internal *-tblgen utilities;
  #  - it pollutes the global options space.
  foreach(lib ${LIB_NAMES})
    get_target_property(t ${lib} TYPE)
    if("${lib}" STREQUAL "LLVMTableGen")
    elseif("x${t}" STREQUAL "xSTATIC_LIBRARY")
      list(APPEND FILTERED_LIB_NAMES ${lib})
    endif()
  endforeach()
  set(LIB_NAMES ${FILTERED_LIB_NAMES})
endif()

if(NOT DEFINED LLVM_DYLIB_EXPORTED_SYMBOL_FILE)
  # To get the export list for a single llvm library:
  # nm ${LIB_PATH} | awk "/T _LLVM/ { print $3 }" | sort -u | sed -e "s/^_//g" > ${LIB_PATH}.exports

  if (NOT LLVM_DYLIB_EXPORT_ALL)
    if( WIN32 AND NOT CYGWIN )
      message(FATAL_ERROR "Auto-generation not implemented for Win32 without GNU utils. Please specify LLVM_EXPORTED_SYMBOL_FILE.")
    endif()

    set(LLVM_EXPORTED_SYMBOL_FILE ${CMAKE_BINARY_DIR}/libllvm.exports)

    foreach (lib ${LIB_NAMES})
      set(LIB_DIR ${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/lib${LLVM_LIBDIR_SUFFIX})
      set(LIB_NAME ${LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}${lib})
      set(LIB_PATH ${LIB_NAME}${CMAKE_STATIC_LIBRARY_SUFFIX})
      set(LIB_EXPORTS_PATH ${LIB_NAME}.exports)
      list(APPEND LLVM_DYLIB_REQUIRED_EXPORTS ${LIB_EXPORTS_PATH})

      
      add_custom_command(OUTPUT ${LIB_EXPORTS_PATH}
        COMMAND nm ${LIB_PATH} | awk "/T _LLVM/ || /T LLVM/ { print $3 }" | sort -u | sed -e "s/^_//g" > ${LIB_EXPORTS_PATH}
        WORKING_DIRECTORY ${LIB_DIR}
        DEPENDS ${lib}
        COMMENT "Generating Export list for ${lib}..."
        VERBATIM )
    endforeach ()

    add_custom_command(OUTPUT ${LLVM_EXPORTED_SYMBOL_FILE}
      COMMAND cat ${LLVM_DYLIB_REQUIRED_EXPORTS} > ${LLVM_EXPORTED_SYMBOL_FILE}
      WORKING_DIRECTORY ${LIB_DIR}
      DEPENDS ${LLVM_DYLIB_REQUIRED_EXPORTS}
      COMMENT "Generating combined export list...")
    add_custom_target(libLLVMExports DEPENDS ${LLVM_EXPORTED_SYMBOL_FILE})
  endif()
else()
  set(LLVM_EXPORTED_SYMBOL_FILE ${LLVM_DYLIB_EXPORTED_SYMBOL_FILE})
  add_custom_target(libLLVMExports DEPENDS ${LLVM_EXPORTED_SYMBOL_FILE})
endif()

add_llvm_library(LLVM SHARED DISABLE_LLVM_LINK_LLVM_DYLIB ${SOURCES})

list(REMOVE_DUPLICATES LIB_NAMES)
if("${CMAKE_SYSTEM_NAME}" STREQUAL "Linux") # FIXME: It should be "GNU ld for elf"
  # GNU ld doesn't resolve symbols in the version script.
  set(LIB_NAMES -Wl,--whole-archive ${LIB_NAMES} -Wl,--no-whole-archive)
elseif("${CMAKE_SYSTEM_NAME}" STREQUAL "Darwin")
  set(LIB_NAMES -Wl,-all_load ${LIB_NAMES})
endif()

target_link_libraries(LLVM PRIVATE ${LIB_NAMES})

if(TARGET libLLVMExports)
  add_dependencies(LLVM libLLVMExports)
endif()

if (APPLE)
  set_property(TARGET LLVM APPEND_STRING PROPERTY
              LINK_FLAGS
              " -compatibility_version 1 -current_version ${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}")
endif()

