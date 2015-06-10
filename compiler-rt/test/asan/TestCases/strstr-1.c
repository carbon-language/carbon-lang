// Test haystack overflow in strstr function
// RUN: %clang_asan %s -o %t && ASAN_OPTIONS=strict_string_checks=true not %run %t 2>&1 | FileCheck %s

// Test intercept_strstr asan option
// Disable other interceptors because strlen may be called inside strstr
// RUN: ASAN_OPTIONS=intercept_strstr=false:replace_str=false %run %t 2>&1

#include <assert.h>
#include <string.h>

int main(int argc, char **argv) {
  char *r = 0;
  char s2[] = "c";
  char s1[] = {'a', 'c'};
  char s3 = 0;
  r = strstr(s1, s2);
  // CHECK:'s{{[1|3]}}' <== Memory access at offset {{[0-9]+ .*}}flows this variable
  assert(r == s1 + 1);
  return 0;
}
