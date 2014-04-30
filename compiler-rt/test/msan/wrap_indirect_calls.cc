// Test indirect call wrapping in MemorySanitizer.

// RUN: %clangxx_msan -O0 %p/wrap_indirect_calls/two.cc -fPIC -shared -o %t-two-so.so
// RUN: %clangxx_msan -O0 %p/wrap_indirect_calls/wrapper.cc -fPIC -shared -o %t-wrapper-so.so

// Disable fast path.

// RUN: %clangxx_msan -O0 %p/wrap_indirect_calls/caller.cc %p/wrap_indirect_calls/one.cc %s \
// RUN:     %t-two-so.so %t-wrapper-so.so \
// RUN:     -mllvm -msan-wrap-indirect-calls=wrapper \
// RUN:     -mllvm -msan-wrap-indirect-calls-fast=0 \
// RUN:     -DSLOW=1 \
// RUN:     -Wl,--defsym=__executable_start=0 -o %t
// RUN: %run %t

// Enable fast path, call from executable, -O0.

// RUN: %clangxx_msan -O0 %p/wrap_indirect_calls/caller.cc %p/wrap_indirect_calls/one.cc %s \
// RUN:     %t-two-so.so %t-wrapper-so.so \
// RUN:     -mllvm -msan-wrap-indirect-calls=wrapper \
// RUN:     -mllvm -msan-wrap-indirect-calls-fast=1 \
// RUN:     -DSLOW=0 \
// RUN:     -Wl,--defsym=__executable_start=0 -o %t
// RUN: %run %t

// Enable fast path, call from executable, -O3.

// RUN: %clangxx_msan -O3 %p/wrap_indirect_calls/caller.cc %p/wrap_indirect_calls/one.cc %s \
// RUN:     %t-two-so.so %t-wrapper-so.so \
// RUN:     -mllvm -msan-wrap-indirect-calls=wrapper \
// RUN:     -mllvm -msan-wrap-indirect-calls-fast=1 \
// RUN:     -DSLOW=0 \
// RUN:     -Wl,--defsym=__executable_start=0 -o %t
// RUN: %run %t

// Enable fast path, call from DSO, -O0.

// RUN: %clangxx_msan -O0 %p/wrap_indirect_calls/caller.cc %p/wrap_indirect_calls/one.cc -shared \
// RUN:     %t-two-so.so %t-wrapper-so.so \
// RUN:     -mllvm -msan-wrap-indirect-calls=wrapper \
// RUN:     -mllvm -msan-wrap-indirect-calls-fast=1 \
// RUN:     -DSLOW=0 \
// RUN:     -Wl,--defsym=__executable_start=0 -o %t-caller-so.so
// RUN: %clangxx_msan -O0 %s %t-caller-so.so %t-two-so.so %t-wrapper-so.so -o %t
// RUN: %run %t

// Enable fast path, call from DSO, -O3.

// RUN: %clangxx_msan -O3 %p/wrap_indirect_calls/caller.cc %p/wrap_indirect_calls/one.cc -shared \
// RUN:     %t-two-so.so %t-wrapper-so.so \
// RUN:     -mllvm -msan-wrap-indirect-calls=wrapper \
// RUN:     -mllvm -msan-wrap-indirect-calls-fast=1 \
// RUN:     -DSLOW=0 \
// RUN:     -Wl,--defsym=__executable_start=0 -o %t-caller-so.so
// RUN: %clangxx_msan -O3 %s %t-caller-so.so %t-two-so.so %t-wrapper-so.so -o %t
// RUN: %run %t

// The actual test is in multiple files in wrap_indirect_calls/ directory.
void run_test();

int main() {
  run_test();
  return 0;
}
