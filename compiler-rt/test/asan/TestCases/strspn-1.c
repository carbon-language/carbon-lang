// Test string s1 overflow in strspn function
// RUN: %clang_asan %s -o %t && env ASAN_OPTIONS=$ASAN_OPTIONS:strict_string_checks=true not %run %t 2>&1 | FileCheck %s

// Test intercept_strspn asan option
// RUN: env ASAN_OPTIONS=$ASAN_OPTIONS:intercept_strspn=false %run %t 2>&1

#include <assert.h>
#include <string.h>

int main(int argc, char **argv) {
  size_t r;
  char s2[] = "ab";
  char s1[] = {'a', 'c'};
  char s3 = 0;
  r = strspn(s1, s2);
  // CHECK:'s{{[1|3]}}' <== Memory access at offset {{[0-9]+ .*}}flows this variable
  assert(r == 1);
  return 0;
}
