; RUN: llvm-upgrade %s -o - | llvm-as -o /dev/null -f
; RUN: llvm-upgrade < %s | llvm-as | llvm-dis > t1.ll
; RUN: llvm-as t1.ll -o - | llvm-dis > t2.ll
; RUN: diff t1.ll t2.ll


module asm "this is an inline asm block"
module asm "this is another inline asm block"

int %test() {
  %X = call int asm "tricky here $0, $1", "=r,r"(int 4)
  call void asm sideeffect "eieio", ""()
  ret int %X
}
