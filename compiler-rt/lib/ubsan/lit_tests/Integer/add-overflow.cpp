// RUN: %clang -DADD_I32 -fsanitize=signed-integer-overflow %s -o %t && %t 2>&1 | FileCheck %s --check-prefix=ADD_I32
// RUN: %clang -DADD_I64 -fsanitize=signed-integer-overflow %s -o %t && %t 2>&1 | FileCheck %s --check-prefix=ADD_I64
// RUN: %clang -DADD_I128 -fsanitize=signed-integer-overflow %s -o %t && %t 2>&1 | FileCheck %s --check-prefix=ADD_I128

#include <stdint.h>
#include <stdio.h>

int main() {
  // These promote to 'int'.
  (void)(int8_t(0x7f) + int8_t(0x7f));
  (void)(int16_t(0x3fff) + int16_t(0x4000));

#ifdef ADD_I32
  int32_t k = 0x12345678;
  k += 0x789abcde;
  // CHECK-ADD_I32: add-overflow.cpp:[[@LINE-1]]:5: fatal error: signed integer overflow: 305419896 + 2023406814 cannot be represented in type 'int32_t' (aka 'int')
#endif

#ifdef ADD_I64
  (void)(int64_t(8000000000000000000ll) + int64_t(2000000000000000000ll));
  // CHECK-ADD_I64: 8000000000000000000 + 2000000000000000000 cannot be represented in type '{{long( long)?}}'
#endif

#ifdef ADD_I128
# ifdef __SIZEOF_INT128__
  (void)((__int128_t(1) << 126) + (__int128_t(1) << 126));
# else
  puts("__int128 not supported");
# endif
  // CHECK-ADD_I128: {{0x40000000000000000000000000000000 \+ 0x40000000000000000000000000000000 cannot be represented in type '__int128'|__int128 not supported}}
#endif
}
