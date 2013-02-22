# First, add the subdirectories which contain feature-based runtime libraries
# and several convenience helper libraries.
if(CMAKE_SYSTEM_NAME MATCHES "Darwin|Linux")
  # AddressSanitizer is supported on Linux and Mac OS X.
  # Windows support is work in progress.
  add_subdirectory(asan)
  add_subdirectory(interception)
  add_subdirectory(sanitizer_common)
  if(NOT ANDROID)
    add_subdirectory(ubsan)
  endif()
endif()
if("${CMAKE_SYSTEM_NAME}" STREQUAL "Linux" AND NOT ANDROID)
  # ThreadSanitizer and MemorySanitizer are supported on Linux only.
  add_subdirectory(tsan)
  add_subdirectory(msan)
  add_subdirectory(msandr)
endif()

# FIXME: Add support for the profile library.

# The top-level lib directory contains a large amount of C code which provides
# generic implementations of the core runtime library along with optimized
# architecture-specific code in various subdirectories.

set(GENERIC_SOURCES
  absvdi2.c
  absvsi2.c
  absvti2.c
  adddf3.c
  addsf3.c
  addvdi3.c
  addvsi3.c
  addvti3.c
  apple_versioning.c
  ashldi3.c
  ashlti3.c
  ashrdi3.c
  ashrti3.c
  clear_cache.c
  clzdi2.c
  clzsi2.c
  clzti2.c
  cmpdi2.c
  cmpti2.c
  comparedf2.c
  comparesf2.c
  ctzdi2.c
  ctzsi2.c
  ctzti2.c
  divdc3.c
  divdf3.c
  divdi3.c
  divmoddi4.c
  divmodsi4.c
  divsc3.c
  divsf3.c
  divsi3.c
  divti3.c
  divxc3.c
  enable_execute_stack.c
  eprintf.c
  extendsfdf2.c
  ffsdi2.c
  ffsti2.c
  fixdfdi.c
  fixdfsi.c
  fixdfti.c
  fixsfdi.c
  fixsfsi.c
  fixsfti.c
  fixunsdfdi.c
  fixunsdfsi.c
  fixunsdfti.c
  fixunssfdi.c
  fixunssfsi.c
  fixunssfti.c
  fixunsxfdi.c
  fixunsxfsi.c
  fixunsxfti.c
  fixxfdi.c
  fixxfti.c
  floatdidf.c
  floatdisf.c
  floatdixf.c
  floatsidf.c
  floatsisf.c
  floattidf.c
  floattisf.c
  floattixf.c
  floatundidf.c
  floatundisf.c
  floatundixf.c
  floatunsidf.c
  floatunsisf.c
  floatuntidf.c
  floatuntisf.c
  floatuntixf.c
  gcc_personality_v0.c
  int_util.c
  lshrdi3.c
  lshrti3.c
  moddi3.c
  modsi3.c
  modti3.c
  muldc3.c
  muldf3.c
  muldi3.c
  mulodi4.c
  mulosi4.c
  muloti4.c
  mulsc3.c
  mulsf3.c
  multi3.c
  mulvdi3.c
  mulvsi3.c
  mulvti3.c
  mulxc3.c
  negdf2.c
  negdi2.c
  negsf2.c
  negti2.c
  negvdi2.c
  negvsi2.c
  negvti2.c
  paritydi2.c
  paritysi2.c
  parityti2.c
  popcountdi2.c
  popcountsi2.c
  popcountti2.c
  powidf2.c
  powisf2.c
  powitf2.c
  powixf2.c
  subdf3.c
  subsf3.c
  subvdi3.c
  subvsi3.c
  subvti3.c
  trampoline_setup.c
  truncdfsf2.c
  ucmpdi2.c
  ucmpti2.c
  udivdi3.c
  udivmoddi4.c
  udivmodsi4.c
  udivmodti4.c
  udivsi3.c
  udivti3.c
  umoddi3.c
  umodsi3.c
  umodti3.c
  )

set(x86_64_SOURCES
  x86_64/floatdidf.c
  x86_64/floatdisf.c
  x86_64/floatdixf.c
  x86_64/floatundidf.S
  x86_64/floatundisf.S
  x86_64/floatundixf.S
  ${GENERIC_SOURCES})

set(i386_SOURCES
  i386/ashldi3.S
  i386/ashrdi3.S
  i386/divdi3.S
  i386/floatdidf.S
  i386/floatdisf.S
  i386/floatdixf.S
  i386/floatundidf.S
  i386/floatundisf.S
  i386/floatundixf.S
  i386/lshrdi3.S
  i386/moddi3.S
  i386/muldi3.S
  i386/udivdi3.S
  i386/umoddi3.S
  ${GENERIC_SOURCES})

foreach(arch x86_64 i386)
  if(CAN_TARGET_${arch})
    add_compiler_rt_static_runtime(clang_rt.${arch} ${arch}
      SOURCES ${${arch}_SOURCES}
      CFLAGS "-std=c99")
  endif()
endforeach()
