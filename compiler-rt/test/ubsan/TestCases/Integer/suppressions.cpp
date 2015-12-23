// XFAIL: win32
// On Windows, %t starts with c:\. lit's ShLexer helpfully strips the
// quotes in the suppressions="%t..." lines below, so the UBSAN_OPTIONS
// env var that ubsan effectively sees is halt_on_error=1:suppressions=c:\...
// without any quotes.  Since : is ubsan's UBSAN_OPTIONS separator, this
// confuses sanitizer_flag_parser.
// FIXME: Figure out how to make this test go on Windows.

// RUN: %clangxx -fsanitize=integer -g0 %s -o %t

// Fails without any suppression.
// RUN: %env_ubsan_opts=halt_on_error=1 not %run %t 2>&1 | FileCheck %s

// RUN: echo "signed-integer-overflow:%t" > %t.wrong-supp
// RUN: %env_ubsan_opts=halt_on_error=1:suppressions="%t.wrong-supp" not %run %t 2>&1 | FileCheck %s

// RUN: echo "unsigned-integer-overflow:do_overflow" > %t.func-supp
// RUN: %env_ubsan_opts=halt_on_error=1:suppressions="%t.func-supp" %run %t
// RUN: echo "unsigned-integer-overflow:%t" > %t.module-supp
// RUN: %env_ubsan_opts=halt_on_error=1:suppressions="%t.module-supp" %run %t

// Note: file-level suppressions should work even without debug info.
// RUN: echo "unsigned-integer-overflow:%s" > %t.file-supp
// RUN: %env_ubsan_opts=halt_on_error=1:suppressions="%t.file-supp" %run %t

// Suppressions don't work for unrecoverable kinds.
// RUN: %clangxx -fsanitize=integer -fno-sanitize-recover=integer %s -o %t-norecover
// RUN: %env_ubsan_opts=halt_on_error=1:suppressions="%t.module-supp" not %run %t-norecover 2>&1 | FileCheck %s

#include <stdint.h>

extern "C" void do_overflow() {
  (void)(uint64_t(10000000000000000000ull) + uint64_t(9000000000000000000ull));
  // CHECK: runtime error: unsigned integer overflow
}

int main() {
  do_overflow();
  return 0;
}

