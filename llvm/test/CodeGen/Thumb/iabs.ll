; RUN: llc < %s -march=thumb -filetype=obj -o %t.o
; RUN: llvm-objdump -disassemble -arch=thumb %t.o | FileCheck %s

define i32 @test(i32 %a) {
        %tmp1neg = sub i32 0, %a
        %b = icmp sgt i32 %a, -1
        %abs = select i1 %b, i32 %a, i32 %tmp1neg
        ret i32 %abs

; This test just checks that 4 instructions were emitted

; CHECK:      {{text}}
; CHECK:      0:
; CHECK-NEXT: 2:
; CHECK-NEXT: 4:
; CHECK-NEXT: 6:

; CHECK-NOT: 8:
}

