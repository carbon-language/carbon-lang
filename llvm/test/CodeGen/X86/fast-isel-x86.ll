; RUN: llc -march=x86 -relocation-model=pic < %s

; This should use flds to set the return value.
; CHECK: test0:
; CHECK: flds
; CHECK: ret
@G = external global float
define float @test0() nounwind {
  %t = load float* @G
  ret float %t
}

; This should pop 4 bytes on return.
; CHECK: test1:
; CHECK: ret $4
define void @test1({i32, i32, i32, i32}* sret %p) nounwind {
  store {i32, i32, i32, i32} zeroinitializer, {i32, i32, i32, i32}* %p
  ret void
}

; Properly initialize the pic base.
; CHECK: test2:
; CHECK-NOT: HHH
; CHECK: call{{.*}}L2$pb
; CHECK-NEXT: L2$pb:
; CHECK-NEXT: pop
; CHECK: HHH
; CHECK: ret
@HHH = external global i32
define i32 @test2() nounwind {
  %t = load i32* @HHH
  ret i32 %t
}
