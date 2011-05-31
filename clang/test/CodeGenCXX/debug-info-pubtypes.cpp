// RUN: %clang -cc1 -triple x86_64-apple-darwin10  -g -S %s -o %t
// RUN: FileCheck %s < %t

//CHECK:         .asciz   "G"
//CHECK-NEXT:    .long   0
//CHECK-NEXT: Lpubtypes_end1:

class G {
public:
	void foo();
};

void G::foo() {
}
