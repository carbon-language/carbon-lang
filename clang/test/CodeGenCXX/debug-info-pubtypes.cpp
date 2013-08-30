// REQUIRES: x86-64-registered-target
// RUN: %clang_cc1 -triple x86_64-apple-darwin10  -g -fno-limit-debug-info -S -mllvm -generate-dwarf-pub-sections=Enable %s -o - | FileCheck %s

// FIXME: This testcase shouldn't rely on assembly emission.
//CHECK: Lpubtypes_begin[[SECNUM:[0-9]:]]
//CHECK:         .asciz   "G"
//CHECK-NEXT:    .long   0
//CHECK-NEXT: Lpubtypes_end[[SECNUM]]

class G {
public:
  void foo();
};

void G::foo() {
}
