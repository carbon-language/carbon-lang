// RUN: %clangxx -fsanitize=unsigned-integer-overflow %s -o %t && %run %t 2>&1 | FileCheck %s --check-prefix=RECOVER
// RUN: %clangxx -fsanitize=unsigned-integer-overflow -fno-sanitize-recover=all -fsanitize-recover=unsigned-integer-overflow %s -o %t && %run %t 2>&1 | FileCheck %s --check-prefix=RECOVER
// RUN: %env_ubsan_opts=silence_unsigned_overflow=1 %run %t 2>&1 | FileCheck %s --check-prefix=SILENT-RECOVER --allow-empty
// RUN: %clangxx -fsanitize=unsigned-integer-overflow -fno-sanitize-recover=unsigned-integer-overflow %s -o %t
// RUN: not %run %t 2>&1 | FileCheck %s --check-prefix=ABORT
// RUN: %env_ubsan_opts=silence_unsigned_overflow=1 not %run %t 2>&1 | FileCheck %s --check-prefix=ABORT

#include <stdint.h>

int main() {
  // These promote to 'int'.
  (void)(uint8_t(0xff) + uint8_t(0xff));
  (void)(uint16_t(0xf0fff) + uint16_t(0x0fff));
  // RECOVER-NOT: runtime error
  // ABORT-NOT: runtime error

  uint32_t k = 0x87654321;
  k += 0xedcba987;
  // RECOVER: no-recover.cpp:[[@LINE-1]]:5: runtime error: unsigned integer overflow: 2271560481 + 3989547399 cannot be represented in type 'unsigned int'
  // ABORT: no-recover.cpp:[[@LINE-2]]:5: runtime error: unsigned integer overflow: 2271560481 + 3989547399 cannot be represented in type 'unsigned int'

  (void)(uint64_t(10000000000000000000ull) + uint64_t(9000000000000000000ull));
  // RECOVER: 10000000000000000000 + 9000000000000000000 cannot be represented in type 'unsigned {{long( long)?}}'
  // SILENT-RECOVER-NOT: runtime error
  // ABORT-NOT: runtime error
}
