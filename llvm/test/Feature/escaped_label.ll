; RUN: llvm-as %s -o - | llvm-dis > %t1.ll
; RUN: llvm-as %t1.ll -o - | llvm-dis > %t2.ll
; RUN: diff %t1.ll %t2.ll


int %foo() {
	br label "foo`~!@#$%^&*()-_=+{}[]\\|;:',<.>/?"
"foo`~!@#$%^&*()-_=+{}[]\\|;:',<.>/?":
	ret int 17
}
