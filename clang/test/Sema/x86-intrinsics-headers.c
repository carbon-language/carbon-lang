// RUN: clang-cc -triple x86_64-apple-darwin10 -fsyntax-only --mcpu=core2 %s &&
// RUN: clang-cc -triple x86_64-apple-darwin10 -fsyntax-only --mcpu=core2 -fno-lax-vector-conversions %s &&
// RUN: clang-cc -triple x86_64-apple-darwin10 -fsyntax-only --mcpu=core2 -x c++ %s

#include <emmintrin.h>
#include <mm_malloc.h>
#include <pmmintrin.h>
#include <tmmintrin.h>
#include <xmmintrin.h>
