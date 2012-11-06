// RUN: %clang -DSUB_I32 -fsanitize=signed-integer-overflow %s -o %t && %t 2>&1 | FileCheck %s --check-prefix=SUB_I32
// RUN: %clang -DSUB_I64 -fsanitize=signed-integer-overflow %s -o %t && %t 2>&1 | FileCheck %s --check-prefix=SUB_I64
// RUN: %clang -DSUB_I128 -fsanitize=signed-integer-overflow %s -o %t && %t 2>&1 | FileCheck %s --check-prefix=SUB_I128

#include <stdint.h>

int main() {
  // These promote to 'int'.
  (void)(int8_t(-2) - int8_t(0x7f));
  (void)(int16_t(-2) - int16_t(0x7fff));

#ifdef SUB_I32
  (void)(int32_t(-2) - int32_t(0x7fffffff));
  // CHECK-SUB_I32: sub-overflow.cpp:13:22: fatal error: signed integer overflow: -2 - 2147483647 cannot be represented in type 'int'
#endif

#ifdef SUB_I64
  (void)(int64_t(-8000000000000000000ll) - int64_t(2000000000000000000ll));
  // CHECK-SUB_I64: -8000000000000000000 - 2000000000000000000 cannot be represented in type 'long'
#endif

#ifdef SUB_I128
  (void)(-(__int128_t(1) << 126) - (__int128_t(1) << 126) - 1);
  // CHECK-SUB_I128: 0x80000000000000000000000000000000 - 1 cannot be represented in type '__int128'
#endif
}
