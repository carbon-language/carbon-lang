; RUN: llc < %s -march=bpf | FileCheck %s

define i32 @test0(i32 %X) {
  %tmp.1 = add i32 %X, 1
  ret i32 %tmp.1
; CHECK-LABEL: test0:
; CHECK: addi r1, 1
}

; CHECK-LABEL: store_imm:
; CHECK: stw  0(r1), r0
; CHECK: stw  4(r2), r0
define i32 @store_imm(i32* %a, i32* %b) {
entry:
  store i32 0, i32* %a, align 4
  %0 = getelementptr inbounds i32, i32* %b, i32 1
  store i32 0, i32* %0, align 4
  ret i32 0
}

@G = external global i8
define zeroext i8 @loadG() {
  %tmp = load i8* @G
  ret i8 %tmp
; CHECK-LABEL: loadG:
; CHECK: ld_64 r1
; CHECK: ldb  r0, 0(r1)
}
