// Test needle overflow in strcasestr function
// RUN: %clang_asan %s -o %t && ASAN_OPTIONS=$ASAN_OPTIONS:strict_string_checks=true not %run %t 2>&1 | FileCheck %s

// Test intercept_strstr asan option
// Disable other interceptors because strlen may be called inside strcasestr
// RUN: env ASAN_OPTIONS=$ASAN_OPTIONS:intercept_strstr=false:replace_str=false %run %t 2>&1

// There's no interceptor for strcasestr on Windows
// XFAIL: win32

#define _GNU_SOURCE
#include <assert.h>
#include <string.h>

int main(int argc, char **argv) {
  char *r = 0;
  char s1[] = "ab";
  char s2[] = {'c'};
  char s3 = 0;
  r = strcasestr(s1, s2);
  assert(r == 0);
  // CHECK:'s{{[2|3]}}' <== Memory access at offset {{[0-9]+ .*}}flows this variable
  return 0;
}
