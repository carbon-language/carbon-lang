// Test needle overflow in strcasestr function
// RUN: %clang_asan %s -o %t && %env_asan_opts=strict_string_checks=true not %run %t 2>&1 | FileCheck %s

// Test intercept_strstr asan option
// Disable other interceptors because strlen may be called inside strcasestr
// RUN: %env_asan_opts=intercept_strstr=false:replace_str=false:intercept_strlen=false %run %t 2>&1

// There's no interceptor for strcasestr on Windows
// XFAIL: windows-msvc

#define _GNU_SOURCE
#include <assert.h>
#include <string.h>
#include <sanitizer/asan_interface.h>

int main(int argc, char **argv) {
  char *r = 0;
  char s1[] = "ab";
  char s2[4] = "cba";
  __asan_poison_memory_region ((char *)&s2[2], 2);
  r = strcasestr(s1, s2);
  assert(r == 0);
  // CHECK:'s2'{{.*}} <== Memory access at offset {{[0-9]+}} partially overflows this variable
  return 0;
}
