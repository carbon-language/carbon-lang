; RUN: llvm-as %s -o - | llvm-dis > %t1.ll
; RUN: llvm-as %t1.ll -o - | llvm-dis > %t2.ll
; RUN: diff %t1.ll %t2.ll


implementation

declare void %bar()

int %foo() {  ;; Calling this function has undefined behavior
	unreachable
}

double %xyz() {
	call void %bar()
	unreachable          ;; Bar must not return.
}
