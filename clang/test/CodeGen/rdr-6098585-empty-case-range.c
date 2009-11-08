// RUN: clang-cc -triple i386-unknown-unknown --emit-llvm-bc -o - %s | opt -std-compile-opts | llvm-dis > %t
// RUN: grep "ret i32" %t | count 2
// RUN: grep "ret i32 3" %t | count 2

// This generated incorrect code because of poor switch chaining.
int f1(int x) {
  switch(x) {
  default:
    return 3;
  case 10 ... 0xFFFFFFFF:
    return 0;
  }
}

// This just asserted because of the way case ranges were calculated.
int f2(int x) {
  switch (x) {
  default:
    return 3;
  case 10 ... -1: 
    return 0;
  }
}
