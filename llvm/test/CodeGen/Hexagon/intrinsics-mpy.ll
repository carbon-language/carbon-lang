; RUN: llc -march=hexagon -mcpu=hexagonv5 < %s | FileCheck %s

; Verify that the mpy intrinsics are lowered into the right instructions.

@c = external global i32

; CHECK: r{{[0-9]+}}{{ *}}={{ *}}mpy(r{{[0-9]+}}.l{{ *}},{{ *}}r{{[0-9]+}}.l)

define void @test1(i32 %a1, i32 %b1) #0 {
entry:
  %0 = tail call i32 @llvm.hexagon.M2.mpy.ll.s0(i32 %a1, i32 %b1)
  store i32 %0, i32* @c, align 4, !tbaa !1
  ret void
}

declare i32 @llvm.hexagon.M2.mpy.ll.s0(i32, i32) #1

; CHECK: r{{[0-9]+}}{{ *}}={{ *}}mpy(r{{[0-9]+}}.l{{ *}},{{ *}}r{{[0-9]+}}.h)

define void @test2(i32 %a2, i32 %b2) #0 {
entry:
  %0 = tail call i32 @llvm.hexagon.M2.mpy.lh.s0(i32 %a2, i32 %b2)
  store i32 %0, i32* @c, align 4, !tbaa !1
  ret void
}

declare i32 @llvm.hexagon.M2.mpy.lh.s0(i32, i32) #1

; CHECK: r{{[0-9]+}}{{ *}}={{ *}}mpy(r{{[0-9]+}}.h{{ *}},{{ *}}r{{[0-9]+}}.l)

define void @test3(i32 %a3, i32 %b3) #0 {
entry:
  %0 = tail call i32 @llvm.hexagon.M2.mpy.hl.s0(i32 %a3, i32 %b3)
  store i32 %0, i32* @c, align 4, !tbaa !1
  ret void
}

declare i32 @llvm.hexagon.M2.mpy.hl.s0(i32, i32) #1

; CHECK: r{{[0-9]+}}{{ *}}={{ *}}mpy(r{{[0-9]+}}.h{{ *}},{{ *}}r{{[0-9]+}}.h)

define void @test4(i32 %a4, i32 %b4) #0 {
entry:
  %0 = tail call i32 @llvm.hexagon.M2.mpy.hh.s0(i32 %a4, i32 %b4)
  store i32 %0, i32* @c, align 4, !tbaa !1
  ret void
}

declare i32 @llvm.hexagon.M2.mpy.hh.s0(i32, i32) #1

; CHECK: r{{[0-9]+}}{{ *}}={{ *}}mpy(r{{[0-9]+}}.l{{ *}},{{ *}}r{{[0-9]+}}.l):sat

define void @test5(i32 %a5, i32 %b5) #0 {
entry:
  %0 = tail call i32 @llvm.hexagon.M2.mpy.sat.ll.s0(i32 %a5, i32 %b5)
  store i32 %0, i32* @c, align 4, !tbaa !1
  ret void
}

declare i32 @llvm.hexagon.M2.mpy.sat.ll.s0(i32, i32) #1

; CHECK: r{{[0-9]+}}{{ *}}={{ *}}mpy(r{{[0-9]+}}.l{{ *}},{{ *}}r{{[0-9]+}}.h):sat

define void @test6(i32 %a6, i32 %b6) #0 {
entry:
  %0 = tail call i32 @llvm.hexagon.M2.mpy.sat.lh.s0(i32 %a6, i32 %b6)
  store i32 %0, i32* @c, align 4, !tbaa !1
  ret void
}

declare i32 @llvm.hexagon.M2.mpy.sat.lh.s0(i32, i32) #1

; CHECK: r{{[0-9]+}}{{ *}}={{ *}}mpy(r{{[0-9]+}}.h{{ *}},{{ *}}r{{[0-9]+}}.l):sat

define void @test7(i32 %a7, i32 %b7) #0 {
entry:
  %0 = tail call i32 @llvm.hexagon.M2.mpy.sat.hl.s0(i32 %a7, i32 %b7)
  store i32 %0, i32* @c, align 4, !tbaa !1
  ret void
}

declare i32 @llvm.hexagon.M2.mpy.sat.hl.s0(i32, i32) #1

; CHECK: r{{[0-9]+}}{{ *}}={{ *}}mpy(r{{[0-9]+}}.h{{ *}},{{ *}}r{{[0-9]+}}.h):sat

define void @test8(i32 %a8, i32 %b8) #0 {
entry:
  %0 = tail call i32 @llvm.hexagon.M2.mpy.sat.hh.s0(i32 %a8, i32 %b8)
  store i32 %0, i32* @c, align 4, !tbaa !1
  ret void
}

declare i32 @llvm.hexagon.M2.mpy.sat.hh.s0(i32, i32) #1

; CHECK: r{{[0-9]+}}{{ *}}={{ *}}mpy(r{{[0-9]+}}.l{{ *}},{{ *}}r{{[0-9]+}}.l):rnd

define void @test9(i32 %a9, i32 %b9) #0 {
entry:
  %0 = tail call i32 @llvm.hexagon.M2.mpy.rnd.ll.s0(i32 %a9, i32 %b9)
  store i32 %0, i32* @c, align 4, !tbaa !1
  ret void
}

declare i32 @llvm.hexagon.M2.mpy.rnd.ll.s0(i32, i32) #1

; CHECK: r{{[0-9]+}}{{ *}}={{ *}}mpy(r{{[0-9]+}}.l{{ *}},{{ *}}r{{[0-9]+}}.h):rnd

define void @test10(i32 %a10, i32 %b10) #0 {
entry:
  %0 = tail call i32 @llvm.hexagon.M2.mpy.rnd.lh.s0(i32 %a10, i32 %b10)
  store i32 %0, i32* @c, align 4, !tbaa !1
  ret void
}

declare i32 @llvm.hexagon.M2.mpy.rnd.lh.s0(i32, i32) #1

; CHECK: r{{[0-9]+}}{{ *}}={{ *}}mpy(r{{[0-9]+}}.h{{ *}},{{ *}}r{{[0-9]+}}.l):rnd

define void @test11(i32 %a11, i32 %b11) #0 {
entry:
  %0 = tail call i32 @llvm.hexagon.M2.mpy.rnd.hl.s0(i32 %a11, i32 %b11)
  store i32 %0, i32* @c, align 4, !tbaa !1
  ret void
}

declare i32 @llvm.hexagon.M2.mpy.rnd.hl.s0(i32, i32) #1

; CHECK: r{{[0-9]+}}{{ *}}={{ *}}mpy(r{{[0-9]+}}.h{{ *}},{{ *}}r{{[0-9]+}}.h):rnd

define void @test12(i32 %a12, i32 %b12) #0 {
entry:
  %0 = tail call i32 @llvm.hexagon.M2.mpy.rnd.hh.s0(i32 %a12, i32 %b12)
  store i32 %0, i32* @c, align 4, !tbaa !1
  ret void
}

declare i32 @llvm.hexagon.M2.mpy.rnd.hh.s0(i32, i32) #1

; CHECK: r{{[0-9]+}}{{ *}}={{ *}}mpy(r{{[0-9]+}}.l{{ *}},{{ *}}r{{[0-9]+}}.l):rnd:sat

define void @test13(i32 %a13, i32 %b13) #0 {
entry:
  %0 = tail call i32 @llvm.hexagon.M2.mpy.sat.rnd.ll.s0(i32 %a13, i32 %b13)
  store i32 %0, i32* @c, align 4, !tbaa !1
  ret void
}

declare i32 @llvm.hexagon.M2.mpy.sat.rnd.ll.s0(i32, i32) #1

; CHECK: r{{[0-9]+}}{{ *}}={{ *}}mpy(r{{[0-9]+}}.l{{ *}},{{ *}}r{{[0-9]+}}.h):rnd:sat

define void @test14(i32 %a14, i32 %b14) #0 {
entry:
  %0 = tail call i32 @llvm.hexagon.M2.mpy.sat.rnd.lh.s0(i32 %a14, i32 %b14)
  store i32 %0, i32* @c, align 4, !tbaa !1
  ret void
}

declare i32 @llvm.hexagon.M2.mpy.sat.rnd.lh.s0(i32, i32) #1

; CHECK: r{{[0-9]+}}{{ *}}={{ *}}mpy(r{{[0-9]+}}.h{{ *}},{{ *}}r{{[0-9]+}}.l):rnd:sat

define void @test15(i32 %a15, i32 %b15) #0 {
entry:
  %0 = tail call i32 @llvm.hexagon.M2.mpy.sat.rnd.hl.s0(i32 %a15, i32 %b15)
  store i32 %0, i32* @c, align 4, !tbaa !1
  ret void
}

declare i32 @llvm.hexagon.M2.mpy.sat.rnd.hl.s0(i32, i32) #1

; CHECK: r{{[0-9]+}}{{ *}}={{ *}}mpy(r{{[0-9]+}}.h{{ *}},{{ *}}r{{[0-9]+}}.h):rnd:sat

define void @test16(i32 %a16, i32 %b16) #0 {
entry:
  %0 = tail call i32 @llvm.hexagon.M2.mpy.sat.rnd.hh.s0(i32 %a16, i32 %b16)
  store i32 %0, i32* @c, align 4, !tbaa !1
  ret void
}

declare i32 @llvm.hexagon.M2.mpy.sat.rnd.hh.s0(i32, i32) #1

; CHECK: r{{[0-9]+}}{{ *}}={{ *}}mpyu(r{{[0-9]+}}.l{{ *}},{{ *}}r{{[0-9]+}}.l)

define void @test17(i32 %a17, i32 %b17) #0 {
entry:
  %0 = tail call i32 @llvm.hexagon.M2.mpyu.ll.s0(i32 %a17, i32 %b17)
  store i32 %0, i32* @c, align 4, !tbaa !1
  ret void
}

declare i32 @llvm.hexagon.M2.mpyu.ll.s0(i32, i32) #1

; CHECK: r{{[0-9]+}}{{ *}}={{ *}}mpyu(r{{[0-9]+}}.l{{ *}},{{ *}}r{{[0-9]+}}.h)

define void @test18(i32 %a18, i32 %b18) #0 {
entry:
  %0 = tail call i32 @llvm.hexagon.M2.mpyu.lh.s0(i32 %a18, i32 %b18)
  store i32 %0, i32* @c, align 4, !tbaa !1
  ret void
}

declare i32 @llvm.hexagon.M2.mpyu.lh.s0(i32, i32) #1

; CHECK: r{{[0-9]+}}{{ *}}={{ *}}mpyu(r{{[0-9]+}}.h{{ *}},{{ *}}r{{[0-9]+}}.l)

define void @test19(i32 %a19, i32 %b19) #0 {
entry:
  %0 = tail call i32 @llvm.hexagon.M2.mpyu.hl.s0(i32 %a19, i32 %b19)
  store i32 %0, i32* @c, align 4, !tbaa !1
  ret void
}

declare i32 @llvm.hexagon.M2.mpyu.hl.s0(i32, i32) #1

; CHECK: r{{[0-9]+}}{{ *}}={{ *}}mpyu(r{{[0-9]+}}.h{{ *}},{{ *}}r{{[0-9]+}}.h)

define void @test20(i32 %a20, i32 %b20) #0 {
entry:
  %0 = tail call i32 @llvm.hexagon.M2.mpyu.hh.s0(i32 %a20, i32 %b20)
  store i32 %0, i32* @c, align 4, !tbaa !1
  ret void
}

declare i32 @llvm.hexagon.M2.mpyu.hh.s0(i32, i32) #1

; CHECK: r{{[0-9]+}}{{ *}}={{ *}}mpy(r{{[0-9]+}}.l{{ *}},{{ *}}r{{[0-9]+}}.l):<<1

define void @test21(i32 %a21, i32 %b21) #0 {
entry:
  %0 = tail call i32 @llvm.hexagon.M2.mpy.ll.s1(i32 %a21, i32 %b21)
  store i32 %0, i32* @c, align 4, !tbaa !1
  ret void
}

declare i32 @llvm.hexagon.M2.mpy.ll.s1(i32, i32) #1

; CHECK: r{{[0-9]+}}{{ *}}={{ *}}mpy(r{{[0-9]+}}.l{{ *}},{{ *}}r{{[0-9]+}}.h):<<1

define void @test22(i32 %a22, i32 %b22) #0 {
entry:
  %0 = tail call i32 @llvm.hexagon.M2.mpy.lh.s1(i32 %a22, i32 %b22)
  store i32 %0, i32* @c, align 4, !tbaa !1
  ret void
}

declare i32 @llvm.hexagon.M2.mpy.lh.s1(i32, i32) #1

; CHECK: r{{[0-9]+}}{{ *}}={{ *}}mpy(r{{[0-9]+}}.h{{ *}},{{ *}}r{{[0-9]+}}.l):<<1

define void @test23(i32 %a23, i32 %b23) #0 {
entry:
  %0 = tail call i32 @llvm.hexagon.M2.mpy.hl.s1(i32 %a23, i32 %b23)
  store i32 %0, i32* @c, align 4, !tbaa !1
  ret void
}

declare i32 @llvm.hexagon.M2.mpy.hl.s1(i32, i32) #1

; CHECK: r{{[0-9]+}}{{ *}}={{ *}}mpy(r{{[0-9]+}}.h{{ *}},{{ *}}r{{[0-9]+}}.h):<<1

define void @test24(i32 %a24, i32 %b24) #0 {
entry:
  %0 = tail call i32 @llvm.hexagon.M2.mpy.hh.s1(i32 %a24, i32 %b24)
  store i32 %0, i32* @c, align 4, !tbaa !1
  ret void
}

declare i32 @llvm.hexagon.M2.mpy.hh.s1(i32, i32) #1

; CHECK: r{{[0-9]+}}{{ *}}={{ *}}mpy(r{{[0-9]+}}.l{{ *}},{{ *}}r{{[0-9]+}}.l):<<1:sat

define void @test25(i32 %a25, i32 %b25) #0 {
entry:
  %0 = tail call i32 @llvm.hexagon.M2.mpy.sat.ll.s1(i32 %a25, i32 %b25)
  store i32 %0, i32* @c, align 4, !tbaa !1
  ret void
}

declare i32 @llvm.hexagon.M2.mpy.sat.ll.s1(i32, i32) #1

; CHECK: r{{[0-9]+}}{{ *}}={{ *}}mpy(r{{[0-9]+}}.l{{ *}},{{ *}}r{{[0-9]+}}.h):<<1:sat

define void @test26(i32 %a26, i32 %b26) #0 {
entry:
  %0 = tail call i32 @llvm.hexagon.M2.mpy.sat.lh.s1(i32 %a26, i32 %b26)
  store i32 %0, i32* @c, align 4, !tbaa !1
  ret void
}

declare i32 @llvm.hexagon.M2.mpy.sat.lh.s1(i32, i32) #1

; CHECK: r{{[0-9]+}}{{ *}}={{ *}}mpy(r{{[0-9]+}}.h{{ *}},{{ *}}r{{[0-9]+}}.l):<<1:sat

define void @test27(i32 %a27, i32 %b27) #0 {
entry:
  %0 = tail call i32 @llvm.hexagon.M2.mpy.sat.hl.s1(i32 %a27, i32 %b27)
  store i32 %0, i32* @c, align 4, !tbaa !1
  ret void
}

declare i32 @llvm.hexagon.M2.mpy.sat.hl.s1(i32, i32) #1

; CHECK: r{{[0-9]+}}{{ *}}={{ *}}mpy(r{{[0-9]+}}.h{{ *}},{{ *}}r{{[0-9]+}}.h):<<1:sat

define void @test28(i32 %a28, i32 %b28) #0 {
entry:
  %0 = tail call i32 @llvm.hexagon.M2.mpy.sat.hh.s1(i32 %a28, i32 %b28)
  store i32 %0, i32* @c, align 4, !tbaa !1
  ret void
}

declare i32 @llvm.hexagon.M2.mpy.sat.hh.s1(i32, i32) #1

; CHECK: r{{[0-9]+}}{{ *}}={{ *}}mpy(r{{[0-9]+}}.l{{ *}},{{ *}}r{{[0-9]+}}.l):<<1:rnd

define void @test29(i32 %a29, i32 %b29) #0 {
entry:
  %0 = tail call i32 @llvm.hexagon.M2.mpy.rnd.ll.s1(i32 %a29, i32 %b29)
  store i32 %0, i32* @c, align 4, !tbaa !1
  ret void
}

declare i32 @llvm.hexagon.M2.mpy.rnd.ll.s1(i32, i32) #1

; CHECK: r{{[0-9]+}}{{ *}}={{ *}}mpy(r{{[0-9]+}}.l{{ *}},{{ *}}r{{[0-9]+}}.h):<<1:rnd

define void @test30(i32 %a30, i32 %b30) #0 {
entry:
  %0 = tail call i32 @llvm.hexagon.M2.mpy.rnd.lh.s1(i32 %a30, i32 %b30)
  store i32 %0, i32* @c, align 4, !tbaa !1
  ret void
}

declare i32 @llvm.hexagon.M2.mpy.rnd.lh.s1(i32, i32) #1

; CHECK: r{{[0-9]+}}{{ *}}={{ *}}mpy(r{{[0-9]+}}.h{{ *}},{{ *}}r{{[0-9]+}}.l):<<1:rnd

define void @test31(i32 %a31, i32 %b31) #0 {
entry:
  %0 = tail call i32 @llvm.hexagon.M2.mpy.rnd.hl.s1(i32 %a31, i32 %b31)
  store i32 %0, i32* @c, align 4, !tbaa !1
  ret void
}

declare i32 @llvm.hexagon.M2.mpy.rnd.hl.s1(i32, i32) #1

; CHECK: r{{[0-9]+}}{{ *}}={{ *}}mpy(r{{[0-9]+}}.h{{ *}},{{ *}}r{{[0-9]+}}.h):<<1:rnd

define void @test32(i32 %a32, i32 %b32) #0 {
entry:
  %0 = tail call i32 @llvm.hexagon.M2.mpy.rnd.hh.s1(i32 %a32, i32 %b32)
  store i32 %0, i32* @c, align 4, !tbaa !1
  ret void
}

declare i32 @llvm.hexagon.M2.mpy.rnd.hh.s1(i32, i32) #1

; CHECK: r{{[0-9]+}}{{ *}}={{ *}}mpy(r{{[0-9]+}}.l{{ *}},{{ *}}r{{[0-9]+}}.l):<<1:rnd:sat

define void @test33(i32 %a33, i32 %b33) #0 {
entry:
  %0 = tail call i32 @llvm.hexagon.M2.mpy.sat.rnd.ll.s1(i32 %a33, i32 %b33)
  store i32 %0, i32* @c, align 4, !tbaa !1
  ret void
}

declare i32 @llvm.hexagon.M2.mpy.sat.rnd.ll.s1(i32, i32) #1

; CHECK: r{{[0-9]+}}{{ *}}={{ *}}mpy(r{{[0-9]+}}.l{{ *}},{{ *}}r{{[0-9]+}}.h):<<1:rnd:sat

define void @test34(i32 %a34, i32 %b34) #0 {
entry:
  %0 = tail call i32 @llvm.hexagon.M2.mpy.sat.rnd.lh.s1(i32 %a34, i32 %b34)
  store i32 %0, i32* @c, align 4, !tbaa !1
  ret void
}

declare i32 @llvm.hexagon.M2.mpy.sat.rnd.lh.s1(i32, i32) #1

; CHECK: r{{[0-9]+}}{{ *}}={{ *}}mpy(r{{[0-9]+}}.h{{ *}},{{ *}}r{{[0-9]+}}.l):<<1:rnd:sat

define void @test35(i32 %a35, i32 %b35) #0 {
entry:
  %0 = tail call i32 @llvm.hexagon.M2.mpy.sat.rnd.hl.s1(i32 %a35, i32 %b35)
  store i32 %0, i32* @c, align 4, !tbaa !1
  ret void
}

declare i32 @llvm.hexagon.M2.mpy.sat.rnd.hl.s1(i32, i32) #1

; CHECK: r{{[0-9]+}}{{ *}}={{ *}}mpy(r{{[0-9]+}}.h{{ *}},{{ *}}r{{[0-9]+}}.h):<<1:rnd:sat

define void @test36(i32 %a36, i32 %b36) #0 {
entry:
  %0 = tail call i32 @llvm.hexagon.M2.mpy.sat.rnd.hh.s1(i32 %a36, i32 %b36)
  store i32 %0, i32* @c, align 4, !tbaa !1
  ret void
}

declare i32 @llvm.hexagon.M2.mpy.sat.rnd.hh.s1(i32, i32) #1

; CHECK: r{{[0-9]+}}{{ *}}={{ *}}mpyu(r{{[0-9]+}}.l{{ *}},{{ *}}r{{[0-9]+}}.l):<<1

define void @test37(i32 %a37, i32 %b37) #0 {
entry:
  %0 = tail call i32 @llvm.hexagon.M2.mpyu.ll.s1(i32 %a37, i32 %b37)
  store i32 %0, i32* @c, align 4, !tbaa !1
  ret void
}

declare i32 @llvm.hexagon.M2.mpyu.ll.s1(i32, i32) #1

; CHECK: r{{[0-9]+}}{{ *}}={{ *}}mpyu(r{{[0-9]+}}.l{{ *}},{{ *}}r{{[0-9]+}}.h):<<1

define void @test38(i32 %a38, i32 %b38) #0 {
entry:
  %0 = tail call i32 @llvm.hexagon.M2.mpyu.lh.s1(i32 %a38, i32 %b38)
  store i32 %0, i32* @c, align 4, !tbaa !1
  ret void
}

declare i32 @llvm.hexagon.M2.mpyu.lh.s1(i32, i32) #1

; CHECK: r{{[0-9]+}}{{ *}}={{ *}}mpyu(r{{[0-9]+}}.h{{ *}},{{ *}}r{{[0-9]+}}.l):<<1

define void @test39(i32 %a39, i32 %b39) #0 {
entry:
  %0 = tail call i32 @llvm.hexagon.M2.mpyu.hl.s1(i32 %a39, i32 %b39)
  store i32 %0, i32* @c, align 4, !tbaa !1
  ret void
}

declare i32 @llvm.hexagon.M2.mpyu.hl.s1(i32, i32) #1

; CHECK: r{{[0-9]+}}{{ *}}={{ *}}mpyu(r{{[0-9]+}}.h{{ *}},{{ *}}r{{[0-9]+}}.h):<<1

define void @test40(i32 %a40, i32 %b40) #0 {
entry:
  %0 = tail call i32 @llvm.hexagon.M2.mpyu.hh.s1(i32 %a40, i32 %b40)
  store i32 %0, i32* @c, align 4, !tbaa !1
  ret void
}

declare i32 @llvm.hexagon.M2.mpyu.hh.s1(i32, i32) #1

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.ident = !{!0}

!0 = metadata !{metadata !"QuIC LLVM Hexagon Clang version 7.1-internal"}
!1 = metadata !{metadata !2, metadata !2, i64 0}
!2 = metadata !{metadata !"int", metadata !3, i64 0}
!3 = metadata !{metadata !"omnipotent char", metadata !4, i64 0}
!4 = metadata !{metadata !"Simple C/C++ TBAA"}
