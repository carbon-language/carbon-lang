// RUN: %clang_cc1 -Wover-aligned %s -isystem %S/Inputs -verify

// This test ensures that we still get the warning even if we #include <new>
// where the header here simulates <new>.
#include <warn-new-overaligned-3.h>

struct Test {
  template <typename T>
  struct SeparateCacheLines {
    T data;
  } __attribute__((aligned(256)));

  SeparateCacheLines<int> high_contention_data[10];
};

void helper() {
  Test t;
  new Test;  // expected-warning {{type 'Test' requires 256 bytes of alignment and the default allocator only guarantees}}
  new Test[10];  // expected-warning {{type 'Test' requires 256 bytes of alignment and the default allocator only guarantees}}
}
