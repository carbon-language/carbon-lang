// Test strict_string_checks option in strtol function
// RUN: %clang_asan -DTEST1 %s -o %t
// RUN: %run %t test1 2>&1
// RUN: ASAN_OPTIONS=strict_string_checks=false %run %t test1 2>&1
// RUN: ASAN_OPTIONS=strict_string_checks=true not %run %t test1 2>&1 | FileCheck %s --check-prefix=CHECK1
// RUN: %run %t test2 2>&1
// RUN: ASAN_OPTIONS=strict_string_checks=false %run %t test2 2>&1
// RUN: ASAN_OPTIONS=strict_string_checks=true not %run %t test2 2>&1 | FileCheck %s --check-prefix=CHECK2
// RUN: %run %t test3 2>&1
// RUN: ASAN_OPTIONS=strict_string_checks=false %run %t test3 2>&1
// RUN: ASAN_OPTIONS=strict_string_checks=true not %run %t test3 2>&1 | FileCheck %s --check-prefix=CHECK3
// RUN: %run %t test4 2>&1
// RUN: ASAN_OPTIONS=strict_string_checks=false %run %t test4 2>&1
// RUN: ASAN_OPTIONS=strict_string_checks=true not %run %t test4 2>&1 | FileCheck %s --check-prefix=CHECK4
// RUN: %run %t test5 2>&1
// RUN: ASAN_OPTIONS=strict_string_checks=false %run %t test5 2>&1
// RUN: ASAN_OPTIONS=strict_string_checks=true not %run %t test5 2>&1 | FileCheck %s --check-prefix=CHECK5
// RUN: %run %t test6 2>&1
// RUN: ASAN_OPTIONS=strict_string_checks=false %run %t test6 2>&1
// RUN: ASAN_OPTIONS=strict_string_checks=true not %run %t test6 2>&1 | FileCheck %s --check-prefix=CHECK6
// RUN: %run %t test7 2>&1
// RUN: ASAN_OPTIONS=strict_string_checks=false %run %t test7 2>&1
// RUN: ASAN_OPTIONS=strict_string_checks=true not %run %t test7 2>&1 | FileCheck %s --check-prefix=CHECK7

#include <assert.h>
#include <stdlib.h>
#include <string.h>

void test1(char *array, char *endptr) {
  // Buffer overflow if there is no terminating null (depends on base)
  long r = strtol(array, &endptr, 3);
  assert(array + 2 == endptr);
  assert(r == 5);
}

void test2(char *array, char *endptr) {
  // Buffer overflow if there is no terminating null (depends on base)
  array[2] = 'z';
  long r = strtol(array, &endptr, 35);
  assert(array + 2 == endptr);
  assert(r == 37);
}

void test3(char *array, char *endptr) {
  // Buffer overflow if base is invalid.
  long r = strtol(array - 1, NULL, -1);
  assert(r == 0);
}

void test4(char *array, char *endptr) {
  // Buffer overflow if base is invalid.
  long r = strtol(array + 3, NULL, 1);
  assert(r == 0);
}

void test5(char *array, char *endptr) {
  // Overflow if no digits are found.
  array[0] = ' ';
  array[1] = '+';
  array[2] = '-';
  long r = strtol(array, NULL, 0);
  assert(r == 0);
}

void test6(char *array, char *endptr) {
  // Overflow if no digits are found.
  array[0] = ' ';
  array[1] = array[2] = 'z';
  long r = strtol(array, &endptr, 0);
  assert(array == endptr);
  assert(r == 0);
}

void test7(char *array, char *endptr) {
  // Overflow if no digits are found.
  array[2] = 'z';
  long r = strtol(array + 2, NULL, 0);
  assert(r == 0);
}

int main(int argc, char **argv) {
  char *array = (char*)malloc(3);
  char *endptr = NULL;
  array[0] = '1';
  array[1] = '2';
  array[2] = '3';
  if (argc != 2) return 1;
  if (!strcmp(argv[1], "test1")) test1(array, endptr);
  // CHECK1: {{.*ERROR: AddressSanitizer: heap-buffer-overflow on address}}
  // CHECK1: READ of size 4
  if (!strcmp(argv[1], "test2")) test2(array, endptr);
  // CHECK2: {{.*ERROR: AddressSanitizer: heap-buffer-overflow on address}}
  // CHECK2: READ of size 4
  if (!strcmp(argv[1], "test3")) test3(array, endptr);
  // CHECK3: {{.*ERROR: AddressSanitizer: heap-buffer-overflow on address}}
  // CHECK3: READ of size 5
  if (!strcmp(argv[1], "test4")) test4(array, endptr);
  // CHECK4: {{.*ERROR: AddressSanitizer: heap-buffer-overflow on address}}
  // CHECK4: READ of size 1
  if (!strcmp(argv[1], "test5")) test5(array, endptr);
  // CHECK5: {{.*ERROR: AddressSanitizer: heap-buffer-overflow on address}}
  // CHECK5: READ of size 4
  if (!strcmp(argv[1], "test6")) test6(array, endptr);
  // CHECK6: {{.*ERROR: AddressSanitizer: heap-buffer-overflow on address}}
  // CHECK6: READ of size 4
  if (!strcmp(argv[1], "test7")) test7(array, endptr);
  // CHECK7: {{.*ERROR: AddressSanitizer: heap-buffer-overflow on address}}
  // CHECK7: READ of size 2
  free(array);
  return 0;
}
