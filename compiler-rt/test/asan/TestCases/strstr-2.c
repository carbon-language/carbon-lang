// Test needle overflow in strstr function
// RUN: %clang_asan %s -o %t && ASAN_OPTIONS=strict_string_checks=true not %run %t 2>&1 | FileCheck %s

// Test intercept_strstr asan option
// RUN: ASAN_OPTIONS=intercept_strstr=false %run %t 2>&1

#include <assert.h>
#include <string.h>

int main(int argc, char **argv) {
  char *r = 0;
  char s1[] = "ab";
  char s2[] = {'c'};
  char s3 = 0;
  r = strstr(s1, s2);
  // CHECK:'s{{[2|3]}}' <== Memory access at offset {{[0-9]+ .*}}flows this variable
  assert(r == 0);
  return 0;
}
