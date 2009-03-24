// RUN: clang-cc -triple i386-unknown-unknown --emit-llvm-bc -o - %s | opt -std-compile-opts | llvm-dis > %t &&
// RUN: grep "ret i32 %" %t

// Make sure return is not constant (if empty range is skipped or miscompiled)

int f0(unsigned x) {
  switch(x) {
  case 2:
    // fallthrough empty range
  case 10 ... 9:
    return 10;
  default:
    return 0;
  }
}
