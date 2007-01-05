; RUN: llvm-upgrade < %s | llvm-as | llvm-dis > %t1.ll
; RUN: llvm-as %t1.ll -o - | llvm-dis > %t2.ll
; RUN: diff %t1.ll %t2.ll

; Test forward references and redefinitions of globals

%A = global int* %B
%B = global int 7

declare void %X()

declare void %X()

void %X() {
  ret void
}

declare void %X()
