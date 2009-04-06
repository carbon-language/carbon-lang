// RUN: clang-cc -Eonly optimize.c -DOPT_O2 -O2 -verify &&
#ifdef OPT_O2
  #ifndef __OPTIMIZE__
    #error "__OPTIMIZE__ not defined"
  #endif
  #ifdef __OPTIMIZE_SIZE
    #error "__OPTIMIZE_SIZE__ defined"
  #endif
#endif

// RUN: clang-cc -Eonly optimize.c -DOPT_O0 -O0 -verify &&
#ifdef OPT_O0
  #ifdef __OPTIMIZE__
    #error "__OPTIMIZE__ defined"
  #endif
  #ifdef __OPTIMIZE_SIZE
    #error "__OPTIMIZE_SIZE__ defined"
  #endif
#endif

// RUN: clang-cc -Eonly optimize.c -DOPT_OS -Os -verify
#ifdef OPT_OS
  #ifndef __OPTIMIZE__
    #error "__OPTIMIZE__ not defined"
  #endif
  #ifndef __OPTIMIZE_SIZE
    #error "__OPTIMIZE_SIZE__ not defined"
  #endif
#endif
