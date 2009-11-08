// RUN: clang-cc %s -emit-llvm -o %t -fobjc-gc -fblocks -triple i386-apple-darwin10
// RUN: grep "objc_assign_strongCast" %t | count 2
// RUN: true

// This should generate a strong cast.

id test3(id x) {
  __block id result;
  ^{ result = x; }();
  return result;
}
