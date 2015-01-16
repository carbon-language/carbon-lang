; RUN: llc -march=hexagon -mcpu=hexagonv5 < %s | FileCheck %s

; Verify that the mpy intrinsics are lowered into the right instructions.
; These instructions have a 64-bit destination register.

@c = external global i64

; CHECK: r{{[0-9]+}}:{{[0-9]+}}{{ *}}={{ *}}mpy(r{{[0-9]+}}.h{{ *}},{{ *}}r{{[0-9]+}}.h)

define void @test1(i32 %a1, i32 %b1) #0 {
entry:
  %0 = tail call i64 @llvm.hexagon.M2.mpyd.hh.s0(i32 %a1, i32 %b1)
  store i64 %0, i64* @c, align 8
  ret void
}

declare i64 @llvm.hexagon.M2.mpyd.hh.s0(i32, i32) #1

; CHECK: r{{[0-9]+}}:{{[0-9]+}}{{ *}}={{ *}}mpy(r{{[0-9]+}}.h{{ *}},{{ *}}r{{[0-9]+}}.l)

define void @test2(i32 %a2, i32 %b2) #0 {
entry:
  %0 = tail call i64 @llvm.hexagon.M2.mpyd.hl.s0(i32 %a2, i32 %b2)
  store i64 %0, i64* @c, align 8
  ret void
}

declare i64 @llvm.hexagon.M2.mpyd.hl.s0(i32, i32) #1

; CHECK: r{{[0-9]+}}:{{[0-9]+}}{{ *}}={{ *}}mpy(r{{[0-9]+}}.l{{ *}},{{ *}}r{{[0-9]+}}.h)

define void @test3(i32 %a3, i32 %b3) #0 {
entry:
  %0 = tail call i64 @llvm.hexagon.M2.mpyd.lh.s0(i32 %a3, i32 %b3)
  store i64 %0, i64* @c, align 8
  ret void
}

declare i64 @llvm.hexagon.M2.mpyd.lh.s0(i32, i32) #1

; CHECK: r{{[0-9]+}}:{{[0-9]+}}{{ *}}={{ *}}mpy(r{{[0-9]+}}.l{{ *}},{{ *}}r{{[0-9]+}}.l)

define void @test4(i32 %a4, i32 %b4) #0 {
entry:
  %0 = tail call i64 @llvm.hexagon.M2.mpyd.ll.s0(i32 %a4, i32 %b4)
  store i64 %0, i64* @c, align 8
  ret void
}

declare i64 @llvm.hexagon.M2.mpyd.ll.s0(i32, i32) #1

; CHECK: r{{[0-9]+}}:{{[0-9]+}}{{ *}}={{ *}}mpy(r{{[0-9]+}}.h{{ *}},{{ *}}r{{[0-9]+}}.h):<<1

define void @test5(i32 %a5, i32 %b5) #0 {
entry:
  %0 = tail call i64 @llvm.hexagon.M2.mpyd.hh.s1(i32 %a5, i32 %b5)
  store i64 %0, i64* @c, align 8
  ret void
}

declare i64 @llvm.hexagon.M2.mpyd.hh.s1(i32, i32) #1

; CHECK: r{{[0-9]+}}:{{[0-9]+}}{{ *}}={{ *}}mpy(r{{[0-9]+}}.h{{ *}},{{ *}}r{{[0-9]+}}.l):<<1

define void @test6(i32 %a6, i32 %b6) #0 {
entry:
  %0 = tail call i64 @llvm.hexagon.M2.mpyd.hl.s1(i32 %a6, i32 %b6)
  store i64 %0, i64* @c, align 8
  ret void
}

declare i64 @llvm.hexagon.M2.mpyd.hl.s1(i32, i32) #1

; CHECK: r{{[0-9]+}}:{{[0-9]+}}{{ *}}={{ *}}mpy(r{{[0-9]+}}.l{{ *}},{{ *}}r{{[0-9]+}}.h):<<1

define void @test7(i32 %a7, i32 %b7) #0 {
entry:
  %0 = tail call i64 @llvm.hexagon.M2.mpyd.lh.s1(i32 %a7, i32 %b7)
  store i64 %0, i64* @c, align 8
  ret void
}

declare i64 @llvm.hexagon.M2.mpyd.lh.s1(i32, i32) #1

; CHECK: r{{[0-9]+}}:{{[0-9]+}}{{ *}}={{ *}}mpy(r{{[0-9]+}}.l{{ *}},{{ *}}r{{[0-9]+}}.l):<<1

define void @test8(i32 %a8, i32 %b8) #0 {
entry:
  %0 = tail call i64 @llvm.hexagon.M2.mpyd.ll.s1(i32 %a8, i32 %b8)
  store i64 %0, i64* @c, align 8
  ret void
}

declare i64 @llvm.hexagon.M2.mpyd.ll.s1(i32, i32) #1

; CHECK: r{{[0-9]+}}:{{[0-9]+}}{{ *}}={{ *}}mpy(r{{[0-9]+}}.h{{ *}},{{ *}}r{{[0-9]+}}.h):rnd

define void @test9(i32 %a9, i32 %b9) #0 {
entry:
  %0 = tail call i64 @llvm.hexagon.M2.mpyd.rnd.hh.s0(i32 %a9, i32 %b9)
  store i64 %0, i64* @c, align 8
  ret void
}

declare i64 @llvm.hexagon.M2.mpyd.rnd.hh.s0(i32, i32) #1

; CHECK: r{{[0-9]+}}:{{[0-9]+}}{{ *}}={{ *}}mpy(r{{[0-9]+}}.h{{ *}},{{ *}}r{{[0-9]+}}.l):rnd

define void @test10(i32 %a10, i32 %b10) #0 {
entry:
  %0 = tail call i64 @llvm.hexagon.M2.mpyd.rnd.hl.s0(i32 %a10, i32 %b10)
  store i64 %0, i64* @c, align 8
  ret void
}

declare i64 @llvm.hexagon.M2.mpyd.rnd.hl.s0(i32, i32) #1

; CHECK: r{{[0-9]+}}:{{[0-9]+}}{{ *}}={{ *}}mpy(r{{[0-9]+}}.l{{ *}},{{ *}}r{{[0-9]+}}.h):rnd

define void @test11(i32 %a11, i32 %b11) #0 {
entry:
  %0 = tail call i64 @llvm.hexagon.M2.mpyd.rnd.lh.s0(i32 %a11, i32 %b11)
  store i64 %0, i64* @c, align 8
  ret void
}

declare i64 @llvm.hexagon.M2.mpyd.rnd.lh.s0(i32, i32) #1

; CHECK: r{{[0-9]+}}:{{[0-9]+}}{{ *}}={{ *}}mpy(r{{[0-9]+}}.l{{ *}},{{ *}}r{{[0-9]+}}.l):rnd

define void @test12(i32 %a12, i32 %b12) #0 {
entry:
  %0 = tail call i64 @llvm.hexagon.M2.mpyd.rnd.ll.s0(i32 %a12, i32 %b12)
  store i64 %0, i64* @c, align 8
  ret void
}

declare i64 @llvm.hexagon.M2.mpyd.rnd.ll.s0(i32, i32) #1

; CHECK: r{{[0-9]+}}:{{[0-9]+}}{{ *}}={{ *}}mpy(r{{[0-9]+}}.h{{ *}},{{ *}}r{{[0-9]+}}.h):<<1:rnd

define void @test13(i32 %a13, i32 %b13) #0 {
entry:
  %0 = tail call i64 @llvm.hexagon.M2.mpyd.rnd.hh.s1(i32 %a13, i32 %b13)
  store i64 %0, i64* @c, align 8
  ret void
}

declare i64 @llvm.hexagon.M2.mpyd.rnd.hh.s1(i32, i32) #1

; CHECK: r{{[0-9]+}}:{{[0-9]+}}{{ *}}={{ *}}mpy(r{{[0-9]+}}.h{{ *}},{{ *}}r{{[0-9]+}}.l):<<1:rnd

define void @test14(i32 %a14, i32 %b14) #0 {
entry:
  %0 = tail call i64 @llvm.hexagon.M2.mpyd.rnd.hl.s1(i32 %a14, i32 %b14)
  store i64 %0, i64* @c, align 8
  ret void
}

declare i64 @llvm.hexagon.M2.mpyd.rnd.hl.s1(i32, i32) #1

; CHECK: r{{[0-9]+}}:{{[0-9]+}}{{ *}}={{ *}}mpy(r{{[0-9]+}}.l{{ *}},{{ *}}r{{[0-9]+}}.h):<<1:rnd

define void @test15(i32 %a15, i32 %b15) #0 {
entry:
  %0 = tail call i64 @llvm.hexagon.M2.mpyd.rnd.lh.s1(i32 %a15, i32 %b15)
  store i64 %0, i64* @c, align 8
  ret void
}

declare i64 @llvm.hexagon.M2.mpyd.rnd.lh.s1(i32, i32) #1

; CHECK: r{{[0-9]+}}:{{[0-9]+}}{{ *}}={{ *}}mpy(r{{[0-9]+}}.l{{ *}},{{ *}}r{{[0-9]+}}.l):<<1:rnd

define void @test16(i32 %a16, i32 %b16) #0 {
entry:
  %0 = tail call i64 @llvm.hexagon.M2.mpyd.rnd.ll.s1(i32 %a16, i32 %b16)
  store i64 %0, i64* @c, align 8
  ret void
}

declare i64 @llvm.hexagon.M2.mpyd.rnd.ll.s1(i32, i32) #1

; CHECK: r{{[0-9]+}}:{{[0-9]+}}{{ *}}={{ *}}mpyu(r{{[0-9]+}}.h{{ *}},{{ *}}r{{[0-9]+}}.h)

define void @test17(i32 %a17, i32 %b17) #0 {
entry:
  %0 = tail call i64 @llvm.hexagon.M2.mpyud.hh.s0(i32 %a17, i32 %b17)
  store i64 %0, i64* @c, align 8
  ret void
}

declare i64 @llvm.hexagon.M2.mpyud.hh.s0(i32, i32) #1

; CHECK: r{{[0-9]+}}:{{[0-9]+}}{{ *}}={{ *}}mpyu(r{{[0-9]+}}.h{{ *}},{{ *}}r{{[0-9]+}}.l)

define void @test18(i32 %a18, i32 %b18) #0 {
entry:
  %0 = tail call i64 @llvm.hexagon.M2.mpyud.hl.s0(i32 %a18, i32 %b18)
  store i64 %0, i64* @c, align 8
  ret void
}

declare i64 @llvm.hexagon.M2.mpyud.hl.s0(i32, i32) #1

; CHECK: r{{[0-9]+}}:{{[0-9]+}}{{ *}}={{ *}}mpyu(r{{[0-9]+}}.l{{ *}},{{ *}}r{{[0-9]+}}.h)

define void @test19(i32 %a19, i32 %b19) #0 {
entry:
  %0 = tail call i64 @llvm.hexagon.M2.mpyud.lh.s0(i32 %a19, i32 %b19)
  store i64 %0, i64* @c, align 8
  ret void
}

declare i64 @llvm.hexagon.M2.mpyud.lh.s0(i32, i32) #1

; CHECK: r{{[0-9]+}}:{{[0-9]+}}{{ *}}={{ *}}mpyu(r{{[0-9]+}}.l{{ *}},{{ *}}r{{[0-9]+}}.l)

define void @test20(i32 %a20, i32 %b20) #0 {
entry:
  %0 = tail call i64 @llvm.hexagon.M2.mpyud.ll.s0(i32 %a20, i32 %b20)
  store i64 %0, i64* @c, align 8
  ret void
}

declare i64 @llvm.hexagon.M2.mpyud.ll.s0(i32, i32) #1

; CHECK: r{{[0-9]+}}:{{[0-9]+}}{{ *}}={{ *}}mpyu(r{{[0-9]+}}.h{{ *}},{{ *}}r{{[0-9]+}}.h):<<1

define void @test21(i32 %a21, i32 %b21) #0 {
entry:
  %0 = tail call i64 @llvm.hexagon.M2.mpyud.hh.s1(i32 %a21, i32 %b21)
  store i64 %0, i64* @c, align 8
  ret void
}

declare i64 @llvm.hexagon.M2.mpyud.hh.s1(i32, i32) #1

; CHECK: r{{[0-9]+}}:{{[0-9]+}}{{ *}}={{ *}}mpyu(r{{[0-9]+}}.h{{ *}},{{ *}}r{{[0-9]+}}.l):<<1

define void @test22(i32 %a22, i32 %b22) #0 {
entry:
  %0 = tail call i64 @llvm.hexagon.M2.mpyud.hl.s1(i32 %a22, i32 %b22)
  store i64 %0, i64* @c, align 8
  ret void
}

declare i64 @llvm.hexagon.M2.mpyud.hl.s1(i32, i32) #1

; CHECK: r{{[0-9]+}}:{{[0-9]+}}{{ *}}={{ *}}mpyu(r{{[0-9]+}}.l{{ *}},{{ *}}r{{[0-9]+}}.h):<<1

define void @test23(i32 %a23, i32 %b23) #0 {
entry:
  %0 = tail call i64 @llvm.hexagon.M2.mpyud.lh.s1(i32 %a23, i32 %b23)
  store i64 %0, i64* @c, align 8
  ret void
}

declare i64 @llvm.hexagon.M2.mpyud.lh.s1(i32, i32) #1

; CHECK: r{{[0-9]+}}:{{[0-9]+}}{{ *}}={{ *}}mpyu(r{{[0-9]+}}.l{{ *}},{{ *}}r{{[0-9]+}}.l):<<1

define void @test24(i32 %a24, i32 %b24) #0 {
entry:
  %0 = tail call i64 @llvm.hexagon.M2.mpyud.ll.s1(i32 %a24, i32 %b24)
  store i64 %0, i64* @c, align 8
  ret void
}

declare i64 @llvm.hexagon.M2.mpyud.ll.s1(i32, i32) #1
