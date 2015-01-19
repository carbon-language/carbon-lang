; RUN: llc -march=hexagon -mcpu=hexagonv5 < %s | FileCheck %s

; Verify that the mpy intrinsics with accumulation are lowered into
; the right instructions. These instructions have a 64-bit destination register.

@c = external global i64

; CHECK: r{{[0-9]+}}:{{[0-9]+}}{{ *}}+={{ *}}mpy(r{{[0-9]+}}.l{{ *}},{{ *}}r{{[0-9]+}}.l)

define void @test1(i32 %a1, i32 %b1) #0 {
entry:
  %0 = load i64* @c, align 8
  %1 = tail call i64 @llvm.hexagon.M2.mpyd.acc.ll.s0(i64 %0, i32 %a1, i32 %b1)
  store i64 %1, i64* @c, align 8
  ret void
}

declare i64 @llvm.hexagon.M2.mpyd.acc.ll.s0(i64, i32, i32) #1

; CHECK: r{{[0-9]+}}:{{[0-9]+}}{{ *}}+={{ *}}mpy(r{{[0-9]+}}.l{{ *}},{{ *}}r{{[0-9]+}}.h)

define void @test2(i32 %a2, i32 %b2) #0 {
entry:
  %0 = load i64* @c, align 8
  %1 = tail call i64 @llvm.hexagon.M2.mpyd.acc.lh.s0(i64 %0, i32 %a2, i32 %b2)
  store i64 %1, i64* @c, align 8
  ret void
}

declare i64 @llvm.hexagon.M2.mpyd.acc.lh.s0(i64, i32, i32) #1

; CHECK: r{{[0-9]+}}:{{[0-9]+}}{{ *}}+={{ *}}mpy(r{{[0-9]+}}.h{{ *}},{{ *}}r{{[0-9]+}}.l)

define void @test3(i32 %a3, i32 %b3) #0 {
entry:
  %0 = load i64* @c, align 8
  %1 = tail call i64 @llvm.hexagon.M2.mpyd.acc.hl.s0(i64 %0, i32 %a3, i32 %b3)
  store i64 %1, i64* @c, align 8
  ret void
}

declare i64 @llvm.hexagon.M2.mpyd.acc.hl.s0(i64, i32, i32) #1

; CHECK: r{{[0-9]+}}:{{[0-9]+}}{{ *}}+={{ *}}mpy(r{{[0-9]+}}.h{{ *}},{{ *}}r{{[0-9]+}}.h)

define void @test4(i32 %a4, i32 %b4) #0 {
entry:
  %0 = load i64* @c, align 8
  %1 = tail call i64 @llvm.hexagon.M2.mpyd.acc.hh.s0(i64 %0, i32 %a4, i32 %b4)
  store i64 %1, i64* @c, align 8
  ret void
}

declare i64 @llvm.hexagon.M2.mpyd.acc.hh.s0(i64, i32, i32) #1

; CHECK: r{{[0-9]+}}:{{[0-9]+}}{{ *}}-={{ *}}mpy(r{{[0-9]+}}.l{{ *}},{{ *}}r{{[0-9]+}}.l)

define void @test5(i32 %a5, i32 %b5) #0 {
entry:
  %0 = load i64* @c, align 8
  %1 = tail call i64 @llvm.hexagon.M2.mpyd.nac.ll.s0(i64 %0, i32 %a5, i32 %b5)
  store i64 %1, i64* @c, align 8
  ret void
}

declare i64 @llvm.hexagon.M2.mpyd.nac.ll.s0(i64, i32, i32) #1

; CHECK: r{{[0-9]+}}:{{[0-9]+}}{{ *}}-={{ *}}mpy(r{{[0-9]+}}.l{{ *}},{{ *}}r{{[0-9]+}}.h)

define void @test6(i32 %a6, i32 %b6) #0 {
entry:
  %0 = load i64* @c, align 8
  %1 = tail call i64 @llvm.hexagon.M2.mpyd.nac.lh.s0(i64 %0, i32 %a6, i32 %b6)
  store i64 %1, i64* @c, align 8
  ret void
}

declare i64 @llvm.hexagon.M2.mpyd.nac.lh.s0(i64, i32, i32) #1

; CHECK: r{{[0-9]+}}:{{[0-9]+}}{{ *}}-={{ *}}mpy(r{{[0-9]+}}.h{{ *}},{{ *}}r{{[0-9]+}}.l)

define void @test7(i32 %a7, i32 %b7) #0 {
entry:
  %0 = load i64* @c, align 8
  %1 = tail call i64 @llvm.hexagon.M2.mpyd.nac.hl.s0(i64 %0, i32 %a7, i32 %b7)
  store i64 %1, i64* @c, align 8
  ret void
}

declare i64 @llvm.hexagon.M2.mpyd.nac.hl.s0(i64, i32, i32) #1

; CHECK: r{{[0-9]+}}:{{[0-9]+}}{{ *}}-={{ *}}mpy(r{{[0-9]+}}.h{{ *}},{{ *}}r{{[0-9]+}}.h)

define void @test8(i32 %a8, i32 %b8) #0 {
entry:
  %0 = load i64* @c, align 8
  %1 = tail call i64 @llvm.hexagon.M2.mpyd.nac.hh.s0(i64 %0, i32 %a8, i32 %b8)
  store i64 %1, i64* @c, align 8
  ret void
}

declare i64 @llvm.hexagon.M2.mpyd.nac.hh.s0(i64, i32, i32) #1

; CHECK: r{{[0-9]+}}:{{[0-9]+}}{{ *}}+={{ *}}mpyu(r{{[0-9]+}}.l{{ *}},{{ *}}r{{[0-9]+}}.l)

define void @test9(i32 %a9, i32 %b9) #0 {
entry:
  %0 = load i64* @c, align 8
  %1 = tail call i64 @llvm.hexagon.M2.mpyud.acc.ll.s0(i64 %0, i32 %a9, i32 %b9)
  store i64 %1, i64* @c, align 8
  ret void
}

declare i64 @llvm.hexagon.M2.mpyud.acc.ll.s0(i64, i32, i32) #1

; CHECK: r{{[0-9]+}}:{{[0-9]+}}{{ *}}+={{ *}}mpyu(r{{[0-9]+}}.l{{ *}},{{ *}}r{{[0-9]+}}.h)

define void @test10(i32 %a10, i32 %b10) #0 {
entry:
  %0 = load i64* @c, align 8
  %1 = tail call i64 @llvm.hexagon.M2.mpyud.acc.lh.s0(i64 %0, i32 %a10, i32 %b10)
  store i64 %1, i64* @c, align 8
  ret void
}

declare i64 @llvm.hexagon.M2.mpyud.acc.lh.s0(i64, i32, i32) #1

; CHECK: r{{[0-9]+}}:{{[0-9]+}}{{ *}}+={{ *}}mpyu(r{{[0-9]+}}.h{{ *}},{{ *}}r{{[0-9]+}}.l)

define void @test11(i32 %a11, i32 %b11) #0 {
entry:
  %0 = load i64* @c, align 8
  %1 = tail call i64 @llvm.hexagon.M2.mpyud.acc.hl.s0(i64 %0, i32 %a11, i32 %b11)
  store i64 %1, i64* @c, align 8
  ret void
}

declare i64 @llvm.hexagon.M2.mpyud.acc.hl.s0(i64, i32, i32) #1

; CHECK: r{{[0-9]+}}:{{[0-9]+}}{{ *}}+={{ *}}mpyu(r{{[0-9]+}}.h{{ *}},{{ *}}r{{[0-9]+}}.h)

define void @test12(i32 %a12, i32 %b12) #0 {
entry:
  %0 = load i64* @c, align 8
  %1 = tail call i64 @llvm.hexagon.M2.mpyud.acc.hh.s0(i64 %0, i32 %a12, i32 %b12)
  store i64 %1, i64* @c, align 8
  ret void
}

declare i64 @llvm.hexagon.M2.mpyud.acc.hh.s0(i64, i32, i32) #1

; CHECK: r{{[0-9]+}}:{{[0-9]+}}{{ *}}-={{ *}}mpyu(r{{[0-9]+}}.l{{ *}},{{ *}}r{{[0-9]+}}.l)

define void @test13(i32 %a13, i32 %b13) #0 {
entry:
  %0 = load i64* @c, align 8
  %1 = tail call i64 @llvm.hexagon.M2.mpyud.nac.ll.s0(i64 %0, i32 %a13, i32 %b13)
  store i64 %1, i64* @c, align 8
  ret void
}

declare i64 @llvm.hexagon.M2.mpyud.nac.ll.s0(i64, i32, i32) #1

; CHECK: r{{[0-9]+}}:{{[0-9]+}}{{ *}}-={{ *}}mpyu(r{{[0-9]+}}.l{{ *}},{{ *}}r{{[0-9]+}}.h)

define void @test14(i32 %a14, i32 %b14) #0 {
entry:
  %0 = load i64* @c, align 8
  %1 = tail call i64 @llvm.hexagon.M2.mpyud.nac.lh.s0(i64 %0, i32 %a14, i32 %b14)
  store i64 %1, i64* @c, align 8
  ret void
}

declare i64 @llvm.hexagon.M2.mpyud.nac.lh.s0(i64, i32, i32) #1

; CHECK: r{{[0-9]+}}:{{[0-9]+}}{{ *}}-={{ *}}mpyu(r{{[0-9]+}}.h{{ *}},{{ *}}r{{[0-9]+}}.l)

define void @test15(i32 %a15, i32 %b15) #0 {
entry:
  %0 = load i64* @c, align 8
  %1 = tail call i64 @llvm.hexagon.M2.mpyud.nac.hl.s0(i64 %0, i32 %a15, i32 %b15)
  store i64 %1, i64* @c, align 8
  ret void
}

declare i64 @llvm.hexagon.M2.mpyud.nac.hl.s0(i64, i32, i32) #1

; CHECK: r{{[0-9]+}}:{{[0-9]+}}{{ *}}-={{ *}}mpyu(r{{[0-9]+}}.h{{ *}},{{ *}}r{{[0-9]+}}.h)

define void @test16(i32 %a16, i32 %b16) #0 {
entry:
  %0 = load i64* @c, align 8
  %1 = tail call i64 @llvm.hexagon.M2.mpyud.nac.hh.s0(i64 %0, i32 %a16, i32 %b16)
  store i64 %1, i64* @c, align 8
  ret void
}

declare i64 @llvm.hexagon.M2.mpyud.nac.hh.s0(i64, i32, i32) #1

; CHECK: r{{[0-9]+}}:{{[0-9]+}}{{ *}}+={{ *}}mpy(r{{[0-9]+}}.l{{ *}},{{ *}}r{{[0-9]+}}.l):<<1

define void @test17(i32 %a17, i32 %b17) #0 {
entry:
  %0 = load i64* @c, align 8
  %1 = tail call i64 @llvm.hexagon.M2.mpyd.acc.ll.s1(i64 %0, i32 %a17, i32 %b17)
  store i64 %1, i64* @c, align 8
  ret void
}

declare i64 @llvm.hexagon.M2.mpyd.acc.ll.s1(i64, i32, i32) #1

; CHECK: r{{[0-9]+}}:{{[0-9]+}}{{ *}}+={{ *}}mpy(r{{[0-9]+}}.l{{ *}},{{ *}}r{{[0-9]+}}.h):<<1

define void @test18(i32 %a18, i32 %b18) #0 {
entry:
  %0 = load i64* @c, align 8
  %1 = tail call i64 @llvm.hexagon.M2.mpyd.acc.lh.s1(i64 %0, i32 %a18, i32 %b18)
  store i64 %1, i64* @c, align 8
  ret void
}

declare i64 @llvm.hexagon.M2.mpyd.acc.lh.s1(i64, i32, i32) #1

; CHECK: r{{[0-9]+}}:{{[0-9]+}}{{ *}}+={{ *}}mpy(r{{[0-9]+}}.h{{ *}},{{ *}}r{{[0-9]+}}.l):<<1

define void @test19(i32 %a19, i32 %b19) #0 {
entry:
  %0 = load i64* @c, align 8
  %1 = tail call i64 @llvm.hexagon.M2.mpyd.acc.hl.s1(i64 %0, i32 %a19, i32 %b19)
  store i64 %1, i64* @c, align 8
  ret void
}

declare i64 @llvm.hexagon.M2.mpyd.acc.hl.s1(i64, i32, i32) #1

; CHECK: r{{[0-9]+}}:{{[0-9]+}}{{ *}}+={{ *}}mpy(r{{[0-9]+}}.h{{ *}},{{ *}}r{{[0-9]+}}.h):<<1

define void @test20(i32 %a20, i32 %b20) #0 {
entry:
  %0 = load i64* @c, align 8
  %1 = tail call i64 @llvm.hexagon.M2.mpyd.acc.hh.s1(i64 %0, i32 %a20, i32 %b20)
  store i64 %1, i64* @c, align 8
  ret void
}

declare i64 @llvm.hexagon.M2.mpyd.acc.hh.s1(i64, i32, i32) #1

; CHECK: r{{[0-9]+}}:{{[0-9]+}}{{ *}}-={{ *}}mpy(r{{[0-9]+}}.l{{ *}},{{ *}}r{{[0-9]+}}.l):<<1

define void @test21(i32 %a21, i32 %b21) #0 {
entry:
  %0 = load i64* @c, align 8
  %1 = tail call i64 @llvm.hexagon.M2.mpyd.nac.ll.s1(i64 %0, i32 %a21, i32 %b21)
  store i64 %1, i64* @c, align 8
  ret void
}

declare i64 @llvm.hexagon.M2.mpyd.nac.ll.s1(i64, i32, i32) #1

; CHECK: r{{[0-9]+}}:{{[0-9]+}}{{ *}}-={{ *}}mpy(r{{[0-9]+}}.l{{ *}},{{ *}}r{{[0-9]+}}.h):<<1

define void @test22(i32 %a22, i32 %b22) #0 {
entry:
  %0 = load i64* @c, align 8
  %1 = tail call i64 @llvm.hexagon.M2.mpyd.nac.lh.s1(i64 %0, i32 %a22, i32 %b22)
  store i64 %1, i64* @c, align 8
  ret void
}

declare i64 @llvm.hexagon.M2.mpyd.nac.lh.s1(i64, i32, i32) #1

; CHECK: r{{[0-9]+}}:{{[0-9]+}}{{ *}}-={{ *}}mpy(r{{[0-9]+}}.h{{ *}},{{ *}}r{{[0-9]+}}.l):<<1

define void @test23(i32 %a23, i32 %b23) #0 {
entry:
  %0 = load i64* @c, align 8
  %1 = tail call i64 @llvm.hexagon.M2.mpyd.nac.hl.s1(i64 %0, i32 %a23, i32 %b23)
  store i64 %1, i64* @c, align 8
  ret void
}

declare i64 @llvm.hexagon.M2.mpyd.nac.hl.s1(i64, i32, i32) #1

; CHECK: r{{[0-9]+}}:{{[0-9]+}}{{ *}}-={{ *}}mpy(r{{[0-9]+}}.h{{ *}},{{ *}}r{{[0-9]+}}.h):<<1

define void @test24(i32 %a24, i32 %b24) #0 {
entry:
  %0 = load i64* @c, align 8
  %1 = tail call i64 @llvm.hexagon.M2.mpyd.nac.hh.s1(i64 %0, i32 %a24, i32 %b24)
  store i64 %1, i64* @c, align 8
  ret void
}

declare i64 @llvm.hexagon.M2.mpyd.nac.hh.s1(i64, i32, i32) #1

; CHECK: r{{[0-9]+}}:{{[0-9]+}}{{ *}}+={{ *}}mpyu(r{{[0-9]+}}.l{{ *}},{{ *}}r{{[0-9]+}}.l):<<1

define void @test25(i32 %a25, i32 %b25) #0 {
entry:
  %0 = load i64* @c, align 8
  %1 = tail call i64 @llvm.hexagon.M2.mpyud.acc.ll.s1(i64 %0, i32 %a25, i32 %b25)
  store i64 %1, i64* @c, align 8
  ret void
}

declare i64 @llvm.hexagon.M2.mpyud.acc.ll.s1(i64, i32, i32) #1

; CHECK: r{{[0-9]+}}:{{[0-9]+}}{{ *}}+={{ *}}mpyu(r{{[0-9]+}}.l{{ *}},{{ *}}r{{[0-9]+}}.h):<<1

define void @test26(i32 %a26, i32 %b26) #0 {
entry:
  %0 = load i64* @c, align 8
  %1 = tail call i64 @llvm.hexagon.M2.mpyud.acc.lh.s1(i64 %0, i32 %a26, i32 %b26)
  store i64 %1, i64* @c, align 8
  ret void
}

declare i64 @llvm.hexagon.M2.mpyud.acc.lh.s1(i64, i32, i32) #1

; CHECK: r{{[0-9]+}}:{{[0-9]+}}{{ *}}+={{ *}}mpyu(r{{[0-9]+}}.h{{ *}},{{ *}}r{{[0-9]+}}.l):<<1

define void @test27(i32 %a27, i32 %b27) #0 {
entry:
  %0 = load i64* @c, align 8
  %1 = tail call i64 @llvm.hexagon.M2.mpyud.acc.hl.s1(i64 %0, i32 %a27, i32 %b27)
  store i64 %1, i64* @c, align 8
  ret void
}

declare i64 @llvm.hexagon.M2.mpyud.acc.hl.s1(i64, i32, i32) #1

; CHECK: r{{[0-9]+}}:{{[0-9]+}}{{ *}}+={{ *}}mpyu(r{{[0-9]+}}.h{{ *}},{{ *}}r{{[0-9]+}}.h):<<1

define void @test28(i32 %a28, i32 %b28) #0 {
entry:
  %0 = load i64* @c, align 8
  %1 = tail call i64 @llvm.hexagon.M2.mpyud.acc.hh.s1(i64 %0, i32 %a28, i32 %b28)
  store i64 %1, i64* @c, align 8
  ret void
}

declare i64 @llvm.hexagon.M2.mpyud.acc.hh.s1(i64, i32, i32) #1

; CHECK: r{{[0-9]+}}:{{[0-9]+}}{{ *}}-={{ *}}mpyu(r{{[0-9]+}}.l{{ *}},{{ *}}r{{[0-9]+}}.l):<<1

define void @test29(i32 %a29, i32 %b29) #0 {
entry:
  %0 = load i64* @c, align 8
  %1 = tail call i64 @llvm.hexagon.M2.mpyud.nac.ll.s1(i64 %0, i32 %a29, i32 %b29)
  store i64 %1, i64* @c, align 8
  ret void
}

declare i64 @llvm.hexagon.M2.mpyud.nac.ll.s1(i64, i32, i32) #1

; CHECK: r{{[0-9]+}}:{{[0-9]+}}{{ *}}-={{ *}}mpyu(r{{[0-9]+}}.l{{ *}},{{ *}}r{{[0-9]+}}.h):<<1

define void @test30(i32 %a30, i32 %b30) #0 {
entry:
  %0 = load i64* @c, align 8
  %1 = tail call i64 @llvm.hexagon.M2.mpyud.nac.lh.s1(i64 %0, i32 %a30, i32 %b30)
  store i64 %1, i64* @c, align 8
  ret void
}

declare i64 @llvm.hexagon.M2.mpyud.nac.lh.s1(i64, i32, i32) #1

; CHECK: r{{[0-9]+}}:{{[0-9]+}}{{ *}}-={{ *}}mpyu(r{{[0-9]+}}.h{{ *}},{{ *}}r{{[0-9]+}}.l):<<1

define void @test31(i32 %a31, i32 %b31) #0 {
entry:
  %0 = load i64* @c, align 8
  %1 = tail call i64 @llvm.hexagon.M2.mpyud.nac.hl.s1(i64 %0, i32 %a31, i32 %b31)
  store i64 %1, i64* @c, align 8
  ret void
}

declare i64 @llvm.hexagon.M2.mpyud.nac.hl.s1(i64, i32, i32) #1

; CHECK: r{{[0-9]+}}:{{[0-9]+}}{{ *}}-={{ *}}mpyu(r{{[0-9]+}}.h{{ *}},{{ *}}r{{[0-9]+}}.h):<<1

define void @test32(i32 %a32, i32 %b32) #0 {
entry:
  %0 = load i64* @c, align 8
  %1 = tail call i64 @llvm.hexagon.M2.mpyud.nac.hh.s1(i64 %0, i32 %a32, i32 %b32)
  store i64 %1, i64* @c, align 8
  ret void
}

declare i64 @llvm.hexagon.M2.mpyud.nac.hh.s1(i64, i32, i32) #1
