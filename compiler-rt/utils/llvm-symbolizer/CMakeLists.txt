# Build llvm-symbolizer binary, which will be used as an external symbolizer
# by sanitizer tools. Later it can probably become another LLVM tool.

set(SYMBOLIZER_SOURCES
  llvm-symbolizer.cc
  )

# Append sources of LLVM libs we use.
append_llvm_sources(SYMBOLIZER_SOURCES
  LLVMDebugInfo
  LLVMSupport
  LLVMObject
  )

set(SYMBOLIZER_CFLAGS
  -I${LLVM_MAIN_SRC_DIR}/include
  -I${LLVM_BINARY_DIR}/include
  -fPIC
  -fomit-frame-pointer
  -O3
  )

set(SYMBOLIZER_DEFINITIONS
#  __STDC_CONSTANT_MACROS=1
#  __STDC_LIMIT_MACROS=1
  )

# FIXME: Generalize the following ugly piece in sanitizer_common, symbolizer
#        and asan folders.
set(SYMBOLIZER_BINARIES)
if (APPLE)
  # Build universal binary on Apple.
  add_executable(llvm-symbolizer.osx ${SYMBOLIZER_SOURCES})
  set_target_compile_flags(llvm-symbolizer.osx ${SYMBOLIZER_CFLAGS})
  filter_available_targets(SYMBOLIZER_TARGETS x86_64 i386)
  set_target_properties(llvm-symbolizer.osx PROPERTIES
    OSX_ARCHITECTURES "${SYMBOLIZER_TARGETS}")
  list(APPEND SYMBOLIZER_BINARIES llvm-symbolizer.osx)
elseif(UNIX)
  # Assume Linux
  if(CAN_TARGET_X86_64)
    add_executable(llvm-symbolizer-x86_64 ${SYMBOLIZER_SOURCES})
    set_target_compile_flags(llvm-symbolizer-x86_64
      ${SYMBOLIZER_CFLAGS} ${TARGET_X86_64_CFLAGS})
    set_property(TARGET llvm-symbolizer-x86_64 APPEND_STRING PROPERTY
      LINK_FLAGS " ${TARGET_X86_64_CFLAGS}")
    list(APPEND SYMBOLIZER_BINARIES llvm-symbolizer-x86_64)
  endif()
  if(CAN_TARGET_I386)
    add_executable(llvm-symbolizer-i386 ${SYMBOLIZER_SOURCES})
    set_target_compile_flags(llvm-symbolizer-i386
      ${SYMBOLIZER_CFLAGS} ${TARGET_I386_CFLAGS})
    set_property(TARGET llvm-symbolizer-i386 APPEND_STRING PROPERTY
      LINK_FLAGS " ${TARGET_I386_CFLAGS}")
    list(APPEND SYMBOLIZER_BINARIES llvm-symbolizer-i386)
  endif()
  # Linux-specific linker flags.
  set_property(TARGET ${SYMBOLIZER_BINARIES} APPEND_STRING PROPERTY
    LINK_FLAGS " -lpthread -ldl")
endif()

set_property(TARGET ${SYMBOLIZER_BINARIES} APPEND PROPERTY
  COMPILE_DEFINITIONS ${SYMBOLIZER_DEFINITIONS}
  )
# FIXME: Setup sane output directories for ${SYMBOLIZER_BINARIES}
set_target_properties(${SYMBOLIZER_BINARIES} PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
