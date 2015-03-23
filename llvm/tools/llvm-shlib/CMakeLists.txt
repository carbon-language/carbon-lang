# This tool creates a shared library from the LLVM libraries. Generating this
# library is enabled by setting LLVM_BUILD_LLVM_DYLIB=yes on the CMake
# commandline. By default the shared library only exports the LLVM C API.


# You can configure which libraries from LLVM you want to include in the shared
# library by setting LLVM_DYLIB_COMPONENTS to a semi-colon delimited list of
# LLVM components. All compoenent names handled by llvm-config are valid.

if(NOT DEFINED LLVM_DYLIB_COMPONENTS)
  set(LLVM_DYLIB_COMPONENTS
    ${LLVM_TARGETS_TO_BUILD}
    Analysis
    BitReader
    BitWriter
    CodeGen
    Core
    ExecutionEngine
    IPA
    IPO
    IRReader
    InstCombine
    Instrumentation
    Interpreter
    Linker
    MCDisassembler
    MCJIT
    ObjCARCOpts
    Object
    ScalarOpts
    Support
    Target
    TransformUtils
    Vectorize
    native
    )
endif()

add_definitions( -DLLVM_VERSION_INFO=\"${PACKAGE_VERSION}\" )

set(SOURCES
  libllvm.cpp
  )

if(NOT DEFINED LLVM_EXPORTED_SYMBOL_FILE)

  if( WIN32 AND NOT CYGWIN )
    message(FATAL_ERROR "Auto-generation not implemented for Win32 without GNU utils. Please specify LLVM_EXPORTED_SYMBOL_FILE.")
  endif()

  # To get the export list for a single llvm library:
  # nm ${LIB_PATH} | awk "/T _LLVM/ { print $3 }" | sort -u | sed -e "s/^_//g" > ${LIB_PATH}.exports

  set(LLVM_EXPORTED_SYMBOL_FILE ${CMAKE_BINARY_DIR}/libllvm.exports)

  llvm_map_components_to_libnames(LIB_NAMES ${LLVM_DYLIB_COMPONENTS})

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

add_llvm_library(LLVM SHARED ${SOURCES})

if("${CMAKE_SYSTEM_NAME}" STREQUAL "Linux") # FIXME: It should be "GNU ld for elf"
  # GNU ld doesn't resolve symbols in the version script.
  list(REMOVE_DUPLICATES LIB_NAMES)
  set(LIB_NAMES -Wl,--whole-archive ${LIB_NAMES} -Wl,--no-whole-archive)
endif()

target_link_libraries(LLVM PRIVATE ${LIB_NAMES})

add_dependencies(LLVM libLLVMExports)

if (APPLE)
  set_property(TARGET LLVM APPEND_STRING PROPERTY
              LINK_FLAGS
              " -compatibility_version ${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR} -current_version ${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}")
endif()

