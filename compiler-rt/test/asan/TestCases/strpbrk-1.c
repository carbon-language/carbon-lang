// Test string s1 overflow in strpbrk function
// RUN: %clang_asan %s -o %t && env ASAN_OPTIONS=$ASAN_OPTIONS:strict_string_checks=true not %run %t 2>&1 | FileCheck %s

// Test intercept_strpbrk asan option
// RUN: env ASAN_OPTIONS=$ASAN_OPTIONS:intercept_strpbrk=false %run %t 2>&1

#include <assert.h>
#include <string.h>

int main(int argc, char **argv) {
  char *r;
  char s2[] = "ab";
  char s1[] = {'c', 'a'};
  char s3 = 0;
  r = strpbrk(s1, s2);
  // CHECK:'s{{[1|3]}}' <== Memory access at offset {{[0-9]+ .*}}flows this variable
  assert(r == s1 + 1);
  return 0;
}
