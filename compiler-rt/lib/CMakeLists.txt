# The top-level lib directory contains a large amount of C code which provides
# generic implementations of the core runtime library along with optimized
# architecture-specific code in various subdirectories.

file(GLOB GENERIC_SOURCES . "*.c")

# FIXME: We don't currently support building an atomic library, and as it must
# be a separate library from the runtime library, we need to remove its source
# code from the glob.
file(GLOB ATOMIC . "atomic.c")
list(REMOVE_ITEM GENERIC_SOURCES ${ATOMIC})

if(CAN_TARGET_X86_64)
  file(GLOB X86_64_SOURCES . "*.c" "*.S")
  add_library(clang_rt.x86_64 STATIC ${X86_64_SOURCES} ${GENERIC_SOURCES})
  set_target_properties(clang_rt.x86_64 PROPERTIES COMPILE_FLAGS "${TARGET_X86_64_CFLAGS}")
endif()
if(CAN_TARGET_I386)
  file(GLOB I386_SOURCES . "*.c" "*.S")
  add_library(clang_rt.i386 STATIC ${I386_SOURCES} ${GENERIC_SOURCES})
  set_target_properties(clang_rt.i386 PROPERTIES COMPILE_FLAGS "${TARGET_I386_CFLAGS}")
endif()

# Also support building feature-based runtime libraries in their various
# subdircetories.
add_subdirectory(asan)

# FIXME: Add support for the profile library.
