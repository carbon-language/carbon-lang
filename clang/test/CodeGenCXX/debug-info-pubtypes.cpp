// REQUIRES: x86-64-registered-target
// RUN: %clang_cc1 -triple x86_64-apple-darwin10  -g -fno-limit-debug-info -S %s -o %t
// RUN: FileCheck %s < %t

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
