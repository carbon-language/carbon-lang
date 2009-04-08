// RUN: clang-cc -Eonly %s -DOPT_O2 -O2 -verify &&
#ifdef OPT_O2
  #ifndef __OPTIMIZE__
    #error "__OPTIMIZE__ not defined"
  #endif
  #ifdef __OPTIMIZE_SIZE__
    #error "__OPTIMIZE_SIZE__ defined"
  #endif
#endif

// RUN: clang-cc -Eonly %s -DOPT_O0 -O0 -verify &&
#ifdef OPT_O0
  #ifdef __OPTIMIZE__
    #error "__OPTIMIZE__ defined"
  #endif
  #ifdef __OPTIMIZE_SIZE__
    #error "__OPTIMIZE_SIZE__ defined"
  #endif
#endif

// RUN: clang-cc -Eonly %s -DOPT_OS -Os -verify
#ifdef OPT_OS
  #ifndef __OPTIMIZE__
    #error "__OPTIMIZE__ not defined"
  #endif
  #ifdef __OPTIMIZE_SIZE__
    #error "__OPTIMIZE_SIZE__ not defined"
  #endif
#endif
