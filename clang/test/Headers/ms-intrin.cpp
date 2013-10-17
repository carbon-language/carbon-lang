// RUN: %clang_cc1 -triple i386-pc-win32 -target-cpu pentium4 \
// RUN:     -fms-extensions -fms-compatibility -fmsc-version=1700 \
// RUN:     -ffreestanding -verify %s

// Intrin.h needs size_t, but -ffreestanding prevents us from getting it from
// stddef.h.  Work around it with this typedef.
typedef __SIZE_TYPE__ size_t;

#include <Intrin.h>

// Use some C++ to make sure we closed the extern "C" brackets.
template <typename T>
void foo(T V) {}

void bar() {
  _ReadWriteBarrier();  // expected-warning {{is deprecated: use other intrinsics or C++11 atomics instead}}
  _ReadBarrier();       // expected-warning {{is deprecated: use other intrinsics or C++11 atomics instead}}
  _WriteBarrier();      // expected-warning {{is deprecated: use other intrinsics or C++11 atomics instead}}
  // FIXME: It'd be handy if we didn't have to hardcode the line number in
  // intrin.h.
  // expected-note@Intrin.h:754 {{declared here}}
  // expected-note@Intrin.h:759 {{declared here}}
  // expected-note@Intrin.h:764 {{declared here}}
}
