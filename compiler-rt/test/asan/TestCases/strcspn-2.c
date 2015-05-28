// Test stopset overflow in strcspn function
// RUN: %clang_asan %s -o %t && ASAN_OPTIONS=strict_string_checks=true not %run %t 2>&1 | FileCheck %s

// Test intercept_strcspn asan option
// RUN: ASAN_OPTIONS=intercept_strspn=false %run %t 2>&1

#include <assert.h>
#include <string.h>

int main(int argc, char **argv) {
  size_t r;
  char s1[] = "ab";
  char s2[] = {'c', 'd'};
  char s3 = 0;
  r = strcspn(s1, s2);
  // CHECK:'s{{[2|3]}}' <== Memory access at offset {{[0-9]+ .*}}flows this variable
  assert(r == sizeof(s1) - 1);
  return 0;
}
