# Get sources

set(LIBUNWIND_CXX_SOURCES
    libunwind.cpp
    Unwind-EHABI.cpp)
append_if(LIBUNWIND_CXX_SOURCES APPLE Unwind_AppleExtras.cpp)

set(LIBUNWIND_C_SOURCES
    UnwindLevel1.c
    UnwindLevel1-gcc-ext.c
    Unwind-sjlj.c)
set_source_files_properties(${LIBUNWIND_C_SOURCES}
                            PROPERTIES
                              COMPILE_FLAGS "-std=c99")

set(LIBUNWIND_ASM_SOURCES
    UnwindRegistersRestore.S
    UnwindRegistersSave.S)
set_source_files_properties(${LIBUNWIND_ASM_SOURCES}
                            PROPERTIES
                              LANGUAGE C)

set(LIBUNWIND_HEADERS
  AddressSpace.hpp
  assembly.h
  CompactUnwinder.hpp
  config.h
  dwarf2.h
  DwarfInstructions.hpp
  DwarfParser.hpp
  libunwind_ext.h
  Registers.hpp
  UnwindCursor.hpp
  unwind_ext.h
  ../../include/libunwind.h
  ../../include/unwind.h
)

append_if(LIBCXXABI_HEADERS APPLE ../../include/mach-o/compact_unwind_encoding.h)

if (MSVC_IDE)
  # Force them all into the headers dir on MSVC, otherwise they end up at
  # project scope because they don't have extensions.
  source_group("Header Files" FILES ${LIBUNWIND_HEADERS})
endif()

set(LIBUNWIND_SOURCES
    ${LIBUNWIND_CXX_SOURCES}
    ${LIBUNWIND_C_SOURCES}
    ${LIBUNWIND_ASM_SOURCES})

if (LIBUNWIND_ENABLE_SHARED)
  add_library(unwind SHARED ${LIBUNWIND_SOURCES} ${LIBUNWIND_HEADERS})
else()
  add_library(unwind STATIC ${LIBUNWIND_SOURCES} ${LIBUNWIND_HEADERS})
endif()

include_directories("${LIBCXXABI_LIBCXX_INCLUDES}")

# Generate library list.
set(libraries ${LIBCXXABI_CXX_ABI_LIBRARIES})
append_if(libraries LIBCXXABI_HAS_C_LIB c)
append_if(libraries LIBCXXABI_HAS_DL_LIB dl)
append_if(libraries LIBCXXABI_HAS_PTHREAD_LIB pthread)

target_link_libraries(unwind ${libraries})

# Setup flags.
append_if(LIBCXXABI_COMPILE_FLAGS LIBCXXABI_HAS_FPIC_FLAG -fPIC)
append_if(LIBCXXABI_LINK_FLAGS LIBCXXABI_HAS_NODEFAULTLIBS_FLAG -nodefaultlibs)

set(LIBUNWIND_COMPILE_FLAGS)
append_if(LIBUNWIND_COMPILE_FLAGS LIBCXXABI_HAS_NO_RTTI_FLAG -fno-rtti)
if ( LIBCXXABI_HAS_NO_EXCEPTIONS_FLAG AND LIBCXXABI_HAS_FUNWIND_TABLES )
  list(APPEND LIBUNWIND_COMPILE_FLAGS -fno-exceptions)
  list(APPEND LIBUNWIND_COMPILE_FLAGS -funwind-tables)
elseif( LIBUNWIND_ENABLE_SHARED )
  message(FATAL_ERROR "Compiler doesn't support generation of unwind tables "
                      "if exception support is disabled.  Building libunwind "
                      "DSO with runtime dependency on libcxxabi is not "
                      "supported.")
endif()

set(LIBCXXABI_UNWINDER_NAME "unwind")

if ( APPLE )
  if ( CMAKE_OSX_DEPLOYMENT_TARGET STREQUAL "10.6" )
    list(APPEND LIBCXXABI_COMPILE_FLAGS "-U__STRICT_ANSI__")
    list(APPEND LIBCXXABI_LINK_FLAGS
      "-compatibility_version 1"
      "-current_version ${LIBCXXABI_VERSION}"
      "-install_name /usr/lib/lib${LIBCXXABI_UNWINDER_NAME}.1.dylib"
      "/usr/lib/libSystem.B.dylib")
  else()
    list(APPEND LIBCXXABI_LINK_FLAGS
      "-compatibility_version 1"
      "-install_name /usr/lib/lib${LIBCXXABI_UNWINDER_NAME}.1.dylib")
  endif()
endif()

string(REPLACE ";" " " LIBCXXABI_COMPILE_FLAGS "${LIBCXXABI_COMPILE_FLAGS}")
string(REPLACE ";" " " LIBUNWIND_COMPILE_FLAGS "${LIBUNWIND_COMPILE_FLAGS}")
string(REPLACE ";" " " LIBCXXABI_LINK_FLAGS "${LIBCXXABI_LINK_FLAGS}")

set_target_properties(unwind
  PROPERTIES
    COMPILE_FLAGS "${LIBCXXABI_COMPILE_FLAGS} ${LIBUNWIND_COMPILE_FLAGS}"
    LINK_FLAGS    "${LIBCXXABI_LINK_FLAGS}"
    OUTPUT_NAME   "${LIBCXXABI_UNWINDER_NAME}"
    VERSION       "1.0"
    SOVERSION     "1"
  )

install(TARGETS unwind
  LIBRARY DESTINATION lib${LLVM_LIBDIR_SUFFIX}
  ARCHIVE DESTINATION lib${LLVM_LIBDIR_SUFFIX}
  )
