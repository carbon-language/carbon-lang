; RUN: llc -march=hexagon -mcpu=hexagonv5 < %s | FileCheck %s

; Verify that the mpy intrinsics with add/subtract are being lowered to the right instruction.

@c = external global i64

; CHECK: r{{[0-9]+}}{{ *}}+={{ *}}mpy(r{{[0-9]+}}.l{{ *}},{{ *}}r{{[0-9]+}}.l)

define void @test1(i64 %a1, i64 %b1) #0 {
entry:
  %0 = load i64* @c, align 8
  %conv = trunc i64 %0 to i32
  %conv1 = trunc i64 %a1 to i32
  %conv2 = trunc i64 %b1 to i32
  %1 = tail call i32 @llvm.hexagon.M2.mpy.acc.ll.s0(i32 %conv, i32 %conv1, i32 %conv2)
  %conv3 = sext i32 %1 to i64
  store i64 %conv3, i64* @c, align 8
  ret void
}

declare i32 @llvm.hexagon.M2.mpy.acc.ll.s0(i32, i32, i32) #1

; CHECK: r{{[0-9]+}}{{ *}}+={{ *}}mpy(r{{[0-9]+}}.l{{ *}},{{ *}}r{{[0-9]+}}.h)

define void @test2(i64 %a2, i64 %b2) #0 {
entry:
  %0 = load i64* @c, align 8
  %conv = trunc i64 %0 to i32
  %conv1 = trunc i64 %a2 to i32
  %conv2 = trunc i64 %b2 to i32
  %1 = tail call i32 @llvm.hexagon.M2.mpy.acc.lh.s0(i32 %conv, i32 %conv1, i32 %conv2)
  %conv3 = sext i32 %1 to i64
  store i64 %conv3, i64* @c, align 8
  ret void
}

declare i32 @llvm.hexagon.M2.mpy.acc.lh.s0(i32, i32, i32) #1

; CHECK: r{{[0-9]+}}{{ *}}+={{ *}}mpy(r{{[0-9]+}}.h{{ *}},{{ *}}r{{[0-9]+}}.l)

define void @test3(i64 %a3, i64 %b3) #0 {
entry:
  %0 = load i64* @c, align 8
  %conv = trunc i64 %0 to i32
  %conv1 = trunc i64 %a3 to i32
  %conv2 = trunc i64 %b3 to i32
  %1 = tail call i32 @llvm.hexagon.M2.mpy.acc.hl.s0(i32 %conv, i32 %conv1, i32 %conv2)
  %conv3 = sext i32 %1 to i64
  store i64 %conv3, i64* @c, align 8
  ret void
}

declare i32 @llvm.hexagon.M2.mpy.acc.hl.s0(i32, i32, i32) #1

; CHECK: r{{[0-9]+}}{{ *}}+={{ *}}mpy(r{{[0-9]+}}.h{{ *}},{{ *}}r{{[0-9]+}}.h)

define void @test4(i64 %a4, i64 %b4) #0 {
entry:
  %0 = load i64* @c, align 8
  %conv = trunc i64 %0 to i32
  %conv1 = trunc i64 %a4 to i32
  %conv2 = trunc i64 %b4 to i32
  %1 = tail call i32 @llvm.hexagon.M2.mpy.acc.hh.s0(i32 %conv, i32 %conv1, i32 %conv2)
  %conv3 = sext i32 %1 to i64
  store i64 %conv3, i64* @c, align 8
  ret void
}

declare i32 @llvm.hexagon.M2.mpy.acc.hh.s0(i32, i32, i32) #1

; CHECK: r{{[0-9]+}}{{ *}}+={{ *}}mpy(r{{[0-9]+}}.l{{ *}},{{ *}}r{{[0-9]+}}.l):sat

define void @test5(i64 %a5, i64 %b5) #0 {
entry:
  %0 = load i64* @c, align 8
  %conv = trunc i64 %0 to i32
  %conv1 = trunc i64 %a5 to i32
  %conv2 = trunc i64 %b5 to i32
  %1 = tail call i32 @llvm.hexagon.M2.mpy.acc.sat.ll.s0(i32 %conv, i32 %conv1, i32 %conv2)
  %conv3 = sext i32 %1 to i64
  store i64 %conv3, i64* @c, align 8
  ret void
}

declare i32 @llvm.hexagon.M2.mpy.acc.sat.ll.s0(i32, i32, i32) #1

; CHECK: r{{[0-9]+}}{{ *}}+={{ *}}mpy(r{{[0-9]+}}.l{{ *}},{{ *}}r{{[0-9]+}}.h):sat

define void @test6(i64 %a6, i64 %b6) #0 {
entry:
  %0 = load i64* @c, align 8
  %conv = trunc i64 %0 to i32
  %conv1 = trunc i64 %a6 to i32
  %conv2 = trunc i64 %b6 to i32
  %1 = tail call i32 @llvm.hexagon.M2.mpy.acc.sat.lh.s0(i32 %conv, i32 %conv1, i32 %conv2)
  %conv3 = sext i32 %1 to i64
  store i64 %conv3, i64* @c, align 8
  ret void
}

declare i32 @llvm.hexagon.M2.mpy.acc.sat.lh.s0(i32, i32, i32) #1

; CHECK: r{{[0-9]+}}{{ *}}+={{ *}}mpy(r{{[0-9]+}}.h{{ *}},{{ *}}r{{[0-9]+}}.l):sat

define void @test7(i64 %a7, i64 %b7) #0 {
entry:
  %0 = load i64* @c, align 8
  %conv = trunc i64 %0 to i32
  %conv1 = trunc i64 %a7 to i32
  %conv2 = trunc i64 %b7 to i32
  %1 = tail call i32 @llvm.hexagon.M2.mpy.acc.sat.hl.s0(i32 %conv, i32 %conv1, i32 %conv2)
  %conv3 = sext i32 %1 to i64
  store i64 %conv3, i64* @c, align 8
  ret void
}

declare i32 @llvm.hexagon.M2.mpy.acc.sat.hl.s0(i32, i32, i32) #1

; CHECK: r{{[0-9]+}}{{ *}}+={{ *}}mpy(r{{[0-9]+}}.h{{ *}},{{ *}}r{{[0-9]+}}.h):sat

define void @test8(i64 %a8, i64 %b8) #0 {
entry:
  %0 = load i64* @c, align 8
  %conv = trunc i64 %0 to i32
  %conv1 = trunc i64 %a8 to i32
  %conv2 = trunc i64 %b8 to i32
  %1 = tail call i32 @llvm.hexagon.M2.mpy.acc.sat.hh.s0(i32 %conv, i32 %conv1, i32 %conv2)
  %conv3 = sext i32 %1 to i64
  store i64 %conv3, i64* @c, align 8
  ret void
}

declare i32 @llvm.hexagon.M2.mpy.acc.sat.hh.s0(i32, i32, i32) #1

; CHECK: r{{[0-9]+}}{{ *}}-={{ *}}mpy(r{{[0-9]+}}.l{{ *}},{{ *}}r{{[0-9]+}}.l)

define void @test9(i64 %a9, i64 %b9) #0 {
entry:
  %0 = load i64* @c, align 8
  %conv = trunc i64 %0 to i32
  %conv1 = trunc i64 %a9 to i32
  %conv2 = trunc i64 %b9 to i32
  %1 = tail call i32 @llvm.hexagon.M2.mpy.nac.ll.s0(i32 %conv, i32 %conv1, i32 %conv2)
  %conv3 = sext i32 %1 to i64
  store i64 %conv3, i64* @c, align 8
  ret void
}

declare i32 @llvm.hexagon.M2.mpy.nac.ll.s0(i32, i32, i32) #1

; CHECK: r{{[0-9]+}}{{ *}}-={{ *}}mpy(r{{[0-9]+}}.l{{ *}},{{ *}}r{{[0-9]+}}.h)

define void @test10(i64 %a10, i64 %b10) #0 {
entry:
  %0 = load i64* @c, align 8
  %conv = trunc i64 %0 to i32
  %conv1 = trunc i64 %a10 to i32
  %conv2 = trunc i64 %b10 to i32
  %1 = tail call i32 @llvm.hexagon.M2.mpy.nac.lh.s0(i32 %conv, i32 %conv1, i32 %conv2)
  %conv3 = sext i32 %1 to i64
  store i64 %conv3, i64* @c, align 8
  ret void
}

declare i32 @llvm.hexagon.M2.mpy.nac.lh.s0(i32, i32, i32) #1

; CHECK: r{{[0-9]+}}{{ *}}-={{ *}}mpy(r{{[0-9]+}}.h{{ *}},{{ *}}r{{[0-9]+}}.l)

define void @test11(i64 %a11, i64 %b11) #0 {
entry:
  %0 = load i64* @c, align 8
  %conv = trunc i64 %0 to i32
  %conv1 = trunc i64 %a11 to i32
  %conv2 = trunc i64 %b11 to i32
  %1 = tail call i32 @llvm.hexagon.M2.mpy.nac.hl.s0(i32 %conv, i32 %conv1, i32 %conv2)
  %conv3 = sext i32 %1 to i64
  store i64 %conv3, i64* @c, align 8
  ret void
}

declare i32 @llvm.hexagon.M2.mpy.nac.hl.s0(i32, i32, i32) #1

; CHECK: r{{[0-9]+}}{{ *}}-={{ *}}mpy(r{{[0-9]+}}.h{{ *}},{{ *}}r{{[0-9]+}}.h)

define void @test12(i64 %a12, i64 %b12) #0 {
entry:
  %0 = load i64* @c, align 8
  %conv = trunc i64 %0 to i32
  %conv1 = trunc i64 %a12 to i32
  %conv2 = trunc i64 %b12 to i32
  %1 = tail call i32 @llvm.hexagon.M2.mpy.nac.hh.s0(i32 %conv, i32 %conv1, i32 %conv2)
  %conv3 = sext i32 %1 to i64
  store i64 %conv3, i64* @c, align 8
  ret void
}

declare i32 @llvm.hexagon.M2.mpy.nac.hh.s0(i32, i32, i32) #1

; CHECK: r{{[0-9]+}}{{ *}}-={{ *}}mpy(r{{[0-9]+}}.l{{ *}},{{ *}}r{{[0-9]+}}.l):sat

define void @test13(i64 %a13, i64 %b13) #0 {
entry:
  %0 = load i64* @c, align 8
  %conv = trunc i64 %0 to i32
  %conv1 = trunc i64 %a13 to i32
  %conv2 = trunc i64 %b13 to i32
  %1 = tail call i32 @llvm.hexagon.M2.mpy.nac.sat.ll.s0(i32 %conv, i32 %conv1, i32 %conv2)
  %conv3 = sext i32 %1 to i64
  store i64 %conv3, i64* @c, align 8
  ret void
}

declare i32 @llvm.hexagon.M2.mpy.nac.sat.ll.s0(i32, i32, i32) #1

; CHECK: r{{[0-9]+}}{{ *}}-={{ *}}mpy(r{{[0-9]+}}.l{{ *}},{{ *}}r{{[0-9]+}}.h):sat

define void @test14(i64 %a14, i64 %b14) #0 {
entry:
  %0 = load i64* @c, align 8
  %conv = trunc i64 %0 to i32
  %conv1 = trunc i64 %a14 to i32
  %conv2 = trunc i64 %b14 to i32
  %1 = tail call i32 @llvm.hexagon.M2.mpy.nac.sat.lh.s0(i32 %conv, i32 %conv1, i32 %conv2)
  %conv3 = sext i32 %1 to i64
  store i64 %conv3, i64* @c, align 8
  ret void
}

declare i32 @llvm.hexagon.M2.mpy.nac.sat.lh.s0(i32, i32, i32) #1

; CHECK: r{{[0-9]+}}{{ *}}-={{ *}}mpy(r{{[0-9]+}}.h{{ *}},{{ *}}r{{[0-9]+}}.l):sat

define void @test15(i64 %a15, i64 %b15) #0 {
entry:
  %0 = load i64* @c, align 8
  %conv = trunc i64 %0 to i32
  %conv1 = trunc i64 %a15 to i32
  %conv2 = trunc i64 %b15 to i32
  %1 = tail call i32 @llvm.hexagon.M2.mpy.nac.sat.hl.s0(i32 %conv, i32 %conv1, i32 %conv2)
  %conv3 = sext i32 %1 to i64
  store i64 %conv3, i64* @c, align 8
  ret void
}

declare i32 @llvm.hexagon.M2.mpy.nac.sat.hl.s0(i32, i32, i32) #1

; CHECK: r{{[0-9]+}}{{ *}}-={{ *}}mpy(r{{[0-9]+}}.h{{ *}},{{ *}}r{{[0-9]+}}.h):sat

define void @test16(i64 %a16, i64 %b16) #0 {
entry:
  %0 = load i64* @c, align 8
  %conv = trunc i64 %0 to i32
  %conv1 = trunc i64 %a16 to i32
  %conv2 = trunc i64 %b16 to i32
  %1 = tail call i32 @llvm.hexagon.M2.mpy.nac.sat.hh.s0(i32 %conv, i32 %conv1, i32 %conv2)
  %conv3 = sext i32 %1 to i64
  store i64 %conv3, i64* @c, align 8
  ret void
}

declare i32 @llvm.hexagon.M2.mpy.nac.sat.hh.s0(i32, i32, i32) #1

; CHECK: r{{[0-9]+}}{{ *}}+={{ *}}mpyu(r{{[0-9]+}}.l{{ *}},{{ *}}r{{[0-9]+}}.l)

define void @test17(i64 %a17, i64 %b17) #0 {
entry:
  %0 = load i64* @c, align 8
  %conv = trunc i64 %0 to i32
  %conv1 = trunc i64 %a17 to i32
  %conv2 = trunc i64 %b17 to i32
  %1 = tail call i32 @llvm.hexagon.M2.mpyu.acc.ll.s0(i32 %conv, i32 %conv1, i32 %conv2)
  %conv3 = sext i32 %1 to i64
  store i64 %conv3, i64* @c, align 8
  ret void
}

declare i32 @llvm.hexagon.M2.mpyu.acc.ll.s0(i32, i32, i32) #1

; CHECK: r{{[0-9]+}}{{ *}}+={{ *}}mpyu(r{{[0-9]+}}.l{{ *}},{{ *}}r{{[0-9]+}}.h)

define void @test18(i64 %a18, i64 %b18) #0 {
entry:
  %0 = load i64* @c, align 8
  %conv = trunc i64 %0 to i32
  %conv1 = trunc i64 %a18 to i32
  %conv2 = trunc i64 %b18 to i32
  %1 = tail call i32 @llvm.hexagon.M2.mpyu.acc.lh.s0(i32 %conv, i32 %conv1, i32 %conv2)
  %conv3 = sext i32 %1 to i64
  store i64 %conv3, i64* @c, align 8
  ret void
}

declare i32 @llvm.hexagon.M2.mpyu.acc.lh.s0(i32, i32, i32) #1

; CHECK: r{{[0-9]+}}{{ *}}+={{ *}}mpyu(r{{[0-9]+}}.h{{ *}},{{ *}}r{{[0-9]+}}.l)

define void @test19(i64 %a19, i64 %b19) #0 {
entry:
  %0 = load i64* @c, align 8
  %conv = trunc i64 %0 to i32
  %conv1 = trunc i64 %a19 to i32
  %conv2 = trunc i64 %b19 to i32
  %1 = tail call i32 @llvm.hexagon.M2.mpyu.acc.hl.s0(i32 %conv, i32 %conv1, i32 %conv2)
  %conv3 = sext i32 %1 to i64
  store i64 %conv3, i64* @c, align 8
  ret void
}

declare i32 @llvm.hexagon.M2.mpyu.acc.hl.s0(i32, i32, i32) #1

; CHECK: r{{[0-9]+}}{{ *}}+={{ *}}mpyu(r{{[0-9]+}}.h{{ *}},{{ *}}r{{[0-9]+}}.h)

define void @test20(i64 %a20, i64 %b20) #0 {
entry:
  %0 = load i64* @c, align 8
  %conv = trunc i64 %0 to i32
  %conv1 = trunc i64 %a20 to i32
  %conv2 = trunc i64 %b20 to i32
  %1 = tail call i32 @llvm.hexagon.M2.mpyu.acc.hh.s0(i32 %conv, i32 %conv1, i32 %conv2)
  %conv3 = sext i32 %1 to i64
  store i64 %conv3, i64* @c, align 8
  ret void
}

declare i32 @llvm.hexagon.M2.mpyu.acc.hh.s0(i32, i32, i32) #1

; CHECK: r{{[0-9]+}}{{ *}}-={{ *}}mpyu(r{{[0-9]+}}.l{{ *}},{{ *}}r{{[0-9]+}}.l)

define void @test21(i64 %a21, i64 %b21) #0 {
entry:
  %0 = load i64* @c, align 8
  %conv = trunc i64 %0 to i32
  %conv1 = trunc i64 %a21 to i32
  %conv2 = trunc i64 %b21 to i32
  %1 = tail call i32 @llvm.hexagon.M2.mpyu.nac.ll.s0(i32 %conv, i32 %conv1, i32 %conv2)
  %conv3 = sext i32 %1 to i64
  store i64 %conv3, i64* @c, align 8
  ret void
}

declare i32 @llvm.hexagon.M2.mpyu.nac.ll.s0(i32, i32, i32) #1

; CHECK: r{{[0-9]+}}{{ *}}-={{ *}}mpyu(r{{[0-9]+}}.l{{ *}},{{ *}}r{{[0-9]+}}.h)

define void @test22(i64 %a22, i64 %b22) #0 {
entry:
  %0 = load i64* @c, align 8
  %conv = trunc i64 %0 to i32
  %conv1 = trunc i64 %a22 to i32
  %conv2 = trunc i64 %b22 to i32
  %1 = tail call i32 @llvm.hexagon.M2.mpyu.nac.lh.s0(i32 %conv, i32 %conv1, i32 %conv2)
  %conv3 = sext i32 %1 to i64
  store i64 %conv3, i64* @c, align 8
  ret void
}

declare i32 @llvm.hexagon.M2.mpyu.nac.lh.s0(i32, i32, i32) #1

; CHECK: r{{[0-9]+}}{{ *}}-={{ *}}mpyu(r{{[0-9]+}}.h{{ *}},{{ *}}r{{[0-9]+}}.l)

define void @test23(i64 %a23, i64 %b23) #0 {
entry:
  %0 = load i64* @c, align 8
  %conv = trunc i64 %0 to i32
  %conv1 = trunc i64 %a23 to i32
  %conv2 = trunc i64 %b23 to i32
  %1 = tail call i32 @llvm.hexagon.M2.mpyu.nac.hl.s0(i32 %conv, i32 %conv1, i32 %conv2)
  %conv3 = sext i32 %1 to i64
  store i64 %conv3, i64* @c, align 8
  ret void
}

declare i32 @llvm.hexagon.M2.mpyu.nac.hl.s0(i32, i32, i32) #1

; CHECK: r{{[0-9]+}}{{ *}}-={{ *}}mpyu(r{{[0-9]+}}.h{{ *}},{{ *}}r{{[0-9]+}}.h)

define void @test24(i64 %a24, i64 %b24) #0 {
entry:
  %0 = load i64* @c, align 8
  %conv = trunc i64 %0 to i32
  %conv1 = trunc i64 %a24 to i32
  %conv2 = trunc i64 %b24 to i32
  %1 = tail call i32 @llvm.hexagon.M2.mpyu.nac.hh.s0(i32 %conv, i32 %conv1, i32 %conv2)
  %conv3 = sext i32 %1 to i64
  store i64 %conv3, i64* @c, align 8
  ret void
}

declare i32 @llvm.hexagon.M2.mpyu.nac.hh.s0(i32, i32, i32) #1

; CHECK: r{{[0-9]+}}{{ *}}+={{ *}}mpy(r{{[0-9]+}}.l{{ *}},{{ *}}r{{[0-9]+}}.l):<<1

define void @test25(i64 %a25, i64 %b25) #0 {
entry:
  %0 = load i64* @c, align 8
  %conv = trunc i64 %0 to i32
  %conv1 = trunc i64 %a25 to i32
  %conv2 = trunc i64 %b25 to i32
  %1 = tail call i32 @llvm.hexagon.M2.mpy.acc.ll.s1(i32 %conv, i32 %conv1, i32 %conv2)
  %conv3 = sext i32 %1 to i64
  store i64 %conv3, i64* @c, align 8
  ret void
}

declare i32 @llvm.hexagon.M2.mpy.acc.ll.s1(i32, i32, i32) #1

; CHECK: r{{[0-9]+}}{{ *}}+={{ *}}mpy(r{{[0-9]+}}.l{{ *}},{{ *}}r{{[0-9]+}}.h):<<1

define void @test26(i64 %a26, i64 %b26) #0 {
entry:
  %0 = load i64* @c, align 8
  %conv = trunc i64 %0 to i32
  %conv1 = trunc i64 %a26 to i32
  %conv2 = trunc i64 %b26 to i32
  %1 = tail call i32 @llvm.hexagon.M2.mpy.acc.lh.s1(i32 %conv, i32 %conv1, i32 %conv2)
  %conv3 = sext i32 %1 to i64
  store i64 %conv3, i64* @c, align 8
  ret void
}

declare i32 @llvm.hexagon.M2.mpy.acc.lh.s1(i32, i32, i32) #1

; CHECK: r{{[0-9]+}}{{ *}}+={{ *}}mpy(r{{[0-9]+}}.h{{ *}},{{ *}}r{{[0-9]+}}.l):<<1

define void @test27(i64 %a27, i64 %b27) #0 {
entry:
  %0 = load i64* @c, align 8
  %conv = trunc i64 %0 to i32
  %conv1 = trunc i64 %a27 to i32
  %conv2 = trunc i64 %b27 to i32
  %1 = tail call i32 @llvm.hexagon.M2.mpy.acc.hl.s1(i32 %conv, i32 %conv1, i32 %conv2)
  %conv3 = sext i32 %1 to i64
  store i64 %conv3, i64* @c, align 8
  ret void
}

declare i32 @llvm.hexagon.M2.mpy.acc.hl.s1(i32, i32, i32) #1

; CHECK: r{{[0-9]+}}{{ *}}+={{ *}}mpy(r{{[0-9]+}}.h{{ *}},{{ *}}r{{[0-9]+}}.h):<<1

define void @test28(i64 %a28, i64 %b28) #0 {
entry:
  %0 = load i64* @c, align 8
  %conv = trunc i64 %0 to i32
  %conv1 = trunc i64 %a28 to i32
  %conv2 = trunc i64 %b28 to i32
  %1 = tail call i32 @llvm.hexagon.M2.mpy.acc.hh.s1(i32 %conv, i32 %conv1, i32 %conv2)
  %conv3 = sext i32 %1 to i64
  store i64 %conv3, i64* @c, align 8
  ret void
}

declare i32 @llvm.hexagon.M2.mpy.acc.hh.s1(i32, i32, i32) #1

; CHECK: r{{[0-9]+}}{{ *}}+={{ *}}mpy(r{{[0-9]+}}.l{{ *}},{{ *}}r{{[0-9]+}}.l):<<1:sat

define void @test29(i64 %a29, i64 %b29) #0 {
entry:
  %0 = load i64* @c, align 8
  %conv = trunc i64 %0 to i32
  %conv1 = trunc i64 %a29 to i32
  %conv2 = trunc i64 %b29 to i32
  %1 = tail call i32 @llvm.hexagon.M2.mpy.acc.sat.ll.s1(i32 %conv, i32 %conv1, i32 %conv2)
  %conv3 = sext i32 %1 to i64
  store i64 %conv3, i64* @c, align 8
  ret void
}

declare i32 @llvm.hexagon.M2.mpy.acc.sat.ll.s1(i32, i32, i32) #1

; CHECK: r{{[0-9]+}}{{ *}}+={{ *}}mpy(r{{[0-9]+}}.l{{ *}},{{ *}}r{{[0-9]+}}.h):<<1:sat

define void @test30(i64 %a30, i64 %b30) #0 {
entry:
  %0 = load i64* @c, align 8
  %conv = trunc i64 %0 to i32
  %conv1 = trunc i64 %a30 to i32
  %conv2 = trunc i64 %b30 to i32
  %1 = tail call i32 @llvm.hexagon.M2.mpy.acc.sat.lh.s1(i32 %conv, i32 %conv1, i32 %conv2)
  %conv3 = sext i32 %1 to i64
  store i64 %conv3, i64* @c, align 8
  ret void
}

declare i32 @llvm.hexagon.M2.mpy.acc.sat.lh.s1(i32, i32, i32) #1

; CHECK: r{{[0-9]+}}{{ *}}+={{ *}}mpy(r{{[0-9]+}}.h{{ *}},{{ *}}r{{[0-9]+}}.l):<<1:sat

define void @test31(i64 %a31, i64 %b31) #0 {
entry:
  %0 = load i64* @c, align 8
  %conv = trunc i64 %0 to i32
  %conv1 = trunc i64 %a31 to i32
  %conv2 = trunc i64 %b31 to i32
  %1 = tail call i32 @llvm.hexagon.M2.mpy.acc.sat.hl.s1(i32 %conv, i32 %conv1, i32 %conv2)
  %conv3 = sext i32 %1 to i64
  store i64 %conv3, i64* @c, align 8
  ret void
}

declare i32 @llvm.hexagon.M2.mpy.acc.sat.hl.s1(i32, i32, i32) #1

; CHECK: r{{[0-9]+}}{{ *}}+={{ *}}mpy(r{{[0-9]+}}.h{{ *}},{{ *}}r{{[0-9]+}}.h):<<1:sat

define void @test32(i64 %a32, i64 %b32) #0 {
entry:
  %0 = load i64* @c, align 8
  %conv = trunc i64 %0 to i32
  %conv1 = trunc i64 %a32 to i32
  %conv2 = trunc i64 %b32 to i32
  %1 = tail call i32 @llvm.hexagon.M2.mpy.acc.sat.hh.s1(i32 %conv, i32 %conv1, i32 %conv2)
  %conv3 = sext i32 %1 to i64
  store i64 %conv3, i64* @c, align 8
  ret void
}

declare i32 @llvm.hexagon.M2.mpy.acc.sat.hh.s1(i32, i32, i32) #1

; CHECK: r{{[0-9]+}}{{ *}}-={{ *}}mpy(r{{[0-9]+}}.l{{ *}},{{ *}}r{{[0-9]+}}.l):<<1

define void @test33(i64 %a33, i64 %b33) #0 {
entry:
  %0 = load i64* @c, align 8
  %conv = trunc i64 %0 to i32
  %conv1 = trunc i64 %a33 to i32
  %conv2 = trunc i64 %b33 to i32
  %1 = tail call i32 @llvm.hexagon.M2.mpy.nac.ll.s1(i32 %conv, i32 %conv1, i32 %conv2)
  %conv3 = sext i32 %1 to i64
  store i64 %conv3, i64* @c, align 8
  ret void
}

declare i32 @llvm.hexagon.M2.mpy.nac.ll.s1(i32, i32, i32) #1

; CHECK: r{{[0-9]+}}{{ *}}-={{ *}}mpy(r{{[0-9]+}}.l{{ *}},{{ *}}r{{[0-9]+}}.h):<<1

define void @test34(i64 %a34, i64 %b34) #0 {
entry:
  %0 = load i64* @c, align 8
  %conv = trunc i64 %0 to i32
  %conv1 = trunc i64 %a34 to i32
  %conv2 = trunc i64 %b34 to i32
  %1 = tail call i32 @llvm.hexagon.M2.mpy.nac.lh.s1(i32 %conv, i32 %conv1, i32 %conv2)
  %conv3 = sext i32 %1 to i64
  store i64 %conv3, i64* @c, align 8
  ret void
}

declare i32 @llvm.hexagon.M2.mpy.nac.lh.s1(i32, i32, i32) #1

; CHECK: r{{[0-9]+}}{{ *}}-={{ *}}mpy(r{{[0-9]+}}.h{{ *}},{{ *}}r{{[0-9]+}}.l):<<1

define void @test35(i64 %a35, i64 %b35) #0 {
entry:
  %0 = load i64* @c, align 8
  %conv = trunc i64 %0 to i32
  %conv1 = trunc i64 %a35 to i32
  %conv2 = trunc i64 %b35 to i32
  %1 = tail call i32 @llvm.hexagon.M2.mpy.nac.hl.s1(i32 %conv, i32 %conv1, i32 %conv2)
  %conv3 = sext i32 %1 to i64
  store i64 %conv3, i64* @c, align 8
  ret void
}

declare i32 @llvm.hexagon.M2.mpy.nac.hl.s1(i32, i32, i32) #1

; CHECK: r{{[0-9]+}}{{ *}}-={{ *}}mpy(r{{[0-9]+}}.h{{ *}},{{ *}}r{{[0-9]+}}.h):<<1

define void @test36(i64 %a36, i64 %b36) #0 {
entry:
  %0 = load i64* @c, align 8
  %conv = trunc i64 %0 to i32
  %conv1 = trunc i64 %a36 to i32
  %conv2 = trunc i64 %b36 to i32
  %1 = tail call i32 @llvm.hexagon.M2.mpy.nac.hh.s1(i32 %conv, i32 %conv1, i32 %conv2)
  %conv3 = sext i32 %1 to i64
  store i64 %conv3, i64* @c, align 8
  ret void
}

declare i32 @llvm.hexagon.M2.mpy.nac.hh.s1(i32, i32, i32) #1

; CHECK: r{{[0-9]+}}{{ *}}-={{ *}}mpy(r{{[0-9]+}}.l{{ *}},{{ *}}r{{[0-9]+}}.l):<<1:sat

define void @test37(i64 %a37, i64 %b37) #0 {
entry:
  %0 = load i64* @c, align 8
  %conv = trunc i64 %0 to i32
  %conv1 = trunc i64 %a37 to i32
  %conv2 = trunc i64 %b37 to i32
  %1 = tail call i32 @llvm.hexagon.M2.mpy.nac.sat.ll.s1(i32 %conv, i32 %conv1, i32 %conv2)
  %conv3 = sext i32 %1 to i64
  store i64 %conv3, i64* @c, align 8
  ret void
}

declare i32 @llvm.hexagon.M2.mpy.nac.sat.ll.s1(i32, i32, i32) #1

; CHECK: r{{[0-9]+}}{{ *}}-={{ *}}mpy(r{{[0-9]+}}.l{{ *}},{{ *}}r{{[0-9]+}}.h):<<1:sat

define void @test38(i64 %a38, i64 %b38) #0 {
entry:
  %0 = load i64* @c, align 8
  %conv = trunc i64 %0 to i32
  %conv1 = trunc i64 %a38 to i32
  %conv2 = trunc i64 %b38 to i32
  %1 = tail call i32 @llvm.hexagon.M2.mpy.nac.sat.lh.s1(i32 %conv, i32 %conv1, i32 %conv2)
  %conv3 = sext i32 %1 to i64
  store i64 %conv3, i64* @c, align 8
  ret void
}

declare i32 @llvm.hexagon.M2.mpy.nac.sat.lh.s1(i32, i32, i32) #1

; CHECK: r{{[0-9]+}}{{ *}}-={{ *}}mpy(r{{[0-9]+}}.h{{ *}},{{ *}}r{{[0-9]+}}.l):<<1:sat

define void @test39(i64 %a39, i64 %b39) #0 {
entry:
  %0 = load i64* @c, align 8
  %conv = trunc i64 %0 to i32
  %conv1 = trunc i64 %a39 to i32
  %conv2 = trunc i64 %b39 to i32
  %1 = tail call i32 @llvm.hexagon.M2.mpy.nac.sat.hl.s1(i32 %conv, i32 %conv1, i32 %conv2)
  %conv3 = sext i32 %1 to i64
  store i64 %conv3, i64* @c, align 8
  ret void
}

declare i32 @llvm.hexagon.M2.mpy.nac.sat.hl.s1(i32, i32, i32) #1

; CHECK: r{{[0-9]+}}{{ *}}-={{ *}}mpy(r{{[0-9]+}}.h{{ *}},{{ *}}r{{[0-9]+}}.h):<<1:sat

define void @test40(i64 %a40, i64 %b40) #0 {
entry:
  %0 = load i64* @c, align 8
  %conv = trunc i64 %0 to i32
  %conv1 = trunc i64 %a40 to i32
  %conv2 = trunc i64 %b40 to i32
  %1 = tail call i32 @llvm.hexagon.M2.mpy.nac.sat.hh.s1(i32 %conv, i32 %conv1, i32 %conv2)
  %conv3 = sext i32 %1 to i64
  store i64 %conv3, i64* @c, align 8
  ret void
}

declare i32 @llvm.hexagon.M2.mpy.nac.sat.hh.s1(i32, i32, i32) #1

; CHECK: r{{[0-9]+}}{{ *}}+={{ *}}mpyu(r{{[0-9]+}}.l{{ *}},{{ *}}r{{[0-9]+}}.l):<<1

define void @test41(i64 %a41, i64 %b41) #0 {
entry:
  %0 = load i64* @c, align 8
  %conv = trunc i64 %0 to i32
  %conv1 = trunc i64 %a41 to i32
  %conv2 = trunc i64 %b41 to i32
  %1 = tail call i32 @llvm.hexagon.M2.mpyu.acc.ll.s1(i32 %conv, i32 %conv1, i32 %conv2)
  %conv3 = sext i32 %1 to i64
  store i64 %conv3, i64* @c, align 8
  ret void
}

declare i32 @llvm.hexagon.M2.mpyu.acc.ll.s1(i32, i32, i32) #1

; CHECK: r{{[0-9]+}}{{ *}}+={{ *}}mpyu(r{{[0-9]+}}.l{{ *}},{{ *}}r{{[0-9]+}}.h):<<1

define void @test42(i64 %a42, i64 %b42) #0 {
entry:
  %0 = load i64* @c, align 8
  %conv = trunc i64 %0 to i32
  %conv1 = trunc i64 %a42 to i32
  %conv2 = trunc i64 %b42 to i32
  %1 = tail call i32 @llvm.hexagon.M2.mpyu.acc.lh.s1(i32 %conv, i32 %conv1, i32 %conv2)
  %conv3 = sext i32 %1 to i64
  store i64 %conv3, i64* @c, align 8
  ret void
}

declare i32 @llvm.hexagon.M2.mpyu.acc.lh.s1(i32, i32, i32) #1

; CHECK: r{{[0-9]+}}{{ *}}+={{ *}}mpyu(r{{[0-9]+}}.h{{ *}},{{ *}}r{{[0-9]+}}.l):<<1

define void @test43(i64 %a43, i64 %b43) #0 {
entry:
  %0 = load i64* @c, align 8
  %conv = trunc i64 %0 to i32
  %conv1 = trunc i64 %a43 to i32
  %conv2 = trunc i64 %b43 to i32
  %1 = tail call i32 @llvm.hexagon.M2.mpyu.acc.hl.s1(i32 %conv, i32 %conv1, i32 %conv2)
  %conv3 = sext i32 %1 to i64
  store i64 %conv3, i64* @c, align 8
  ret void
}

declare i32 @llvm.hexagon.M2.mpyu.acc.hl.s1(i32, i32, i32) #1

; CHECK: r{{[0-9]+}}{{ *}}+={{ *}}mpyu(r{{[0-9]+}}.h{{ *}},{{ *}}r{{[0-9]+}}.h):<<1

define void @test44(i64 %a44, i64 %b44) #0 {
entry:
  %0 = load i64* @c, align 8
  %conv = trunc i64 %0 to i32
  %conv1 = trunc i64 %a44 to i32
  %conv2 = trunc i64 %b44 to i32
  %1 = tail call i32 @llvm.hexagon.M2.mpyu.acc.hh.s1(i32 %conv, i32 %conv1, i32 %conv2)
  %conv3 = sext i32 %1 to i64
  store i64 %conv3, i64* @c, align 8
  ret void
}

declare i32 @llvm.hexagon.M2.mpyu.acc.hh.s1(i32, i32, i32) #1

; CHECK: r{{[0-9]+}}{{ *}}-={{ *}}mpyu(r{{[0-9]+}}.l{{ *}},{{ *}}r{{[0-9]+}}.l):<<1

define void @test45(i64 %a45, i64 %b45) #0 {
entry:
  %0 = load i64* @c, align 8
  %conv = trunc i64 %0 to i32
  %conv1 = trunc i64 %a45 to i32
  %conv2 = trunc i64 %b45 to i32
  %1 = tail call i32 @llvm.hexagon.M2.mpyu.nac.ll.s1(i32 %conv, i32 %conv1, i32 %conv2)
  %conv3 = sext i32 %1 to i64
  store i64 %conv3, i64* @c, align 8
  ret void
}

declare i32 @llvm.hexagon.M2.mpyu.nac.ll.s1(i32, i32, i32) #1

; CHECK: r{{[0-9]+}}{{ *}}-={{ *}}mpyu(r{{[0-9]+}}.l{{ *}},{{ *}}r{{[0-9]+}}.h):<<1

define void @test46(i64 %a46, i64 %b46) #0 {
entry:
  %0 = load i64* @c, align 8
  %conv = trunc i64 %0 to i32
  %conv1 = trunc i64 %a46 to i32
  %conv2 = trunc i64 %b46 to i32
  %1 = tail call i32 @llvm.hexagon.M2.mpyu.nac.lh.s1(i32 %conv, i32 %conv1, i32 %conv2)
  %conv3 = sext i32 %1 to i64
  store i64 %conv3, i64* @c, align 8
  ret void
}

declare i32 @llvm.hexagon.M2.mpyu.nac.lh.s1(i32, i32, i32) #1

; CHECK: r{{[0-9]+}}{{ *}}-={{ *}}mpyu(r{{[0-9]+}}.h{{ *}},{{ *}}r{{[0-9]+}}.l):<<1

define void @test47(i64 %a47, i64 %b47) #0 {
entry:
  %0 = load i64* @c, align 8
  %conv = trunc i64 %0 to i32
  %conv1 = trunc i64 %a47 to i32
  %conv2 = trunc i64 %b47 to i32
  %1 = tail call i32 @llvm.hexagon.M2.mpyu.nac.hl.s1(i32 %conv, i32 %conv1, i32 %conv2)
  %conv3 = sext i32 %1 to i64
  store i64 %conv3, i64* @c, align 8
  ret void
}

declare i32 @llvm.hexagon.M2.mpyu.nac.hl.s1(i32, i32, i32) #1

; CHECK: r{{[0-9]+}}{{ *}}-={{ *}}mpyu(r{{[0-9]+}}.h{{ *}},{{ *}}r{{[0-9]+}}.h):<<1

define void @test48(i64 %a48, i64 %b48) #0 {
entry:
  %0 = load i64* @c, align 8
  %conv = trunc i64 %0 to i32
  %conv1 = trunc i64 %a48 to i32
  %conv2 = trunc i64 %b48 to i32
  %1 = tail call i32 @llvm.hexagon.M2.mpyu.nac.hh.s1(i32 %conv, i32 %conv1, i32 %conv2)
  %conv3 = sext i32 %1 to i64
  store i64 %conv3, i64* @c, align 8
  ret void
}

declare i32 @llvm.hexagon.M2.mpyu.nac.hh.s1(i32, i32, i32) #1
