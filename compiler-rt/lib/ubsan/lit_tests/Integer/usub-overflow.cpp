// RUN: %clang -DSUB_I32 -fsanitize=unsigned-integer-overflow %s -o %t && %t 2>&1 | FileCheck %s --check-prefix=SUB_I32
// RUN: %clang -DSUB_I64 -fsanitize=unsigned-integer-overflow %s -o %t && %t 2>&1 | FileCheck %s --check-prefix=SUB_I64
// RUN: %clang -DSUB_I128 -fsanitize=unsigned-integer-overflow %s -o %t && %t 2>&1 | FileCheck %s --check-prefix=SUB_I128

#include <stdint.h>

int main() {
  // These promote to 'int'.
  (void)(uint8_t(0) - uint8_t(0x7f));
  (void)(uint16_t(0) - uint16_t(0x7fff));

#ifdef SUB_I32
  (void)(uint32_t(1) - uint32_t(2));
  // CHECK-SUB_I32: usub-overflow.cpp:13:22: fatal error: unsigned integer overflow: 1 - 2 cannot be represented in type 'unsigned int'
#endif

#ifdef SUB_I64
  (void)(uint64_t(8000000000000000000ll) - uint64_t(9000000000000000000ll));
  // CHECK-SUB_I64: 8000000000000000000 - 9000000000000000000 cannot be represented in type 'unsigned long'
#endif

#ifdef SUB_I128
  (void)((__uint128_t(1) << 126) - (__uint128_t(1) << 127));
  // CHECK-SUB_I128: 0x40000000000000000000000000000000 - 0x80000000000000000000000000000000 cannot be represented in type 'unsigned __int128'
#endif
}
