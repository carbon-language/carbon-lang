; RUN: llc < %s -march=mips -relocation-model=static | FileCheck %s
; RUN: llc < %s -march=mips -relocation-model=static -regalloc=basic | FileCheck %s
; Fix PR7473

define i32 @main() nounwind readnone {
entry:
  %a = alloca i32, align 4                        ; <i32*> [#uses=2]
  %c = alloca i32, align 4                        ; <i32*> [#uses=2]
  volatile store i32 1, i32* %a, align 4
  volatile store i32 0, i32* %c, align 4
  %0 = volatile load i32* %a, align 4             ; <i32> [#uses=1]
  %1 = icmp eq i32 %0, 0                          ; <i1> [#uses=1]
; CHECK: addiu $[[R1:[0-9]+]], $zero, 0
  %iftmp.0.0 = select i1 %1, i32 3, i32 0         ; <i32> [#uses=1]
  %2 = volatile load i32* %c, align 4             ; <i32> [#uses=1]
  %3 = icmp eq i32 %2, 0                          ; <i1> [#uses=1]
; CHECK: addiu $[[R1]], $zero, 3
; CHECK: addu $2, ${{.}}, $[[R1]]
  %iftmp.2.0 = select i1 %3, i32 0, i32 5         ; <i32> [#uses=1]
  %4 = add nsw i32 %iftmp.2.0, %iftmp.0.0         ; <i32> [#uses=1]
  ret i32 %4
}
