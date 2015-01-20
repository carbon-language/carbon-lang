; RUN: llc -march=hexagon < %s | FileCheck %s

; Verify that ALU32 - add, or, and, sub, combine intrinsics
; are lowered to the right instructions.

@e = external global i1
@b = external global i8
@d = external global i32
@c = external global i64

; CHECK: r{{[0-9]+}}{{ *}}={{ *}}add(r{{[0-9]+}}{{ *}},{{ *}}r{{[0-9]+}})

define void @test1(i32 %a, i32 %b) #0 {
entry:
  %0 = tail call i32 @llvm.hexagon.A2.add(i32 %a, i32 %b)
  store i32 %0, i32* @d, align 4
  ret void
}

; CHECK: r{{[0-9]+}}{{ *}}={{ *}}sub(r{{[0-9]+}}{{ *}},{{ *}}r{{[0-9]+}})

define void @test2(i32 %a, i32 %b) #0 {
entry:
  %0 = tail call i32 @llvm.hexagon.A2.sub(i32 %a, i32 %b)
  store i32 %0, i32* @d, align 4
  ret void
}

; CHECK: r{{[0-9]+}}{{ *}}={{ *}}and(r{{[0-9]+}}{{ *}},{{ *}}r{{[0-9]+}})

define void @test3(i32 %a, i32 %b) #0 {
entry:
  %0 = tail call i32 @llvm.hexagon.A2.and(i32 %a, i32 %b)
  store i32 %0, i32* @d, align 4
  ret void
}

; CHECK: r{{[0-9]+}}{{ *}}={{ *}}or(r{{[0-9]+}}{{ *}},{{ *}}r{{[0-9]+}})

define void @test4(i32 %a, i32 %b) #0 {
entry:
  %0 = tail call i32 @llvm.hexagon.A2.or(i32 %a, i32 %b)
  store i32 %0, i32* @d, align 4
  ret void
}

; CHECK: r{{[0-9]+}}{{ *}}={{ *}}xor(r{{[0-9]+}}{{ *}},{{ *}}r{{[0-9]+}})

define void @test5(i32 %a, i32 %b) #0 {
entry:
  %0 = tail call i32 @llvm.hexagon.A2.xor(i32 %a, i32 %b)
  store i32 %0, i32* @d, align 4
  ret void
}

; CHECK: r{{[0-9]+}}:{{[0-9]+}}{{ *}}={{ *}}combine(r{{[0-9]+}}{{ *}},{{ *}}r{{[0-9]+}})

define void @test6(i32 %a, i32 %b) #0 {
entry:
  %0 = tail call i64 @llvm.hexagon.A2.combinew(i32 %a, i32 %b)
  store i64 %0, i64* @c, align 8
  ret void
}

; CHECK: r{{[0-9]+}}{{ *}}={{ *}}add(r{{[0-9]+}}{{ *}},{{ *}}#-31849)

define void @test7(i32 %a) #0 {
entry:
  %0 = tail call i32 @llvm.hexagon.A2.addi(i32 %a, i32 -31849)
  store i32 %0, i32* @d, align 4
  ret void
}

; CHECK: r{{[0-9]+}}{{ *}}={{ *}}and(r{{[0-9]+}}{{ *}},{{ *}}#-512)

define void @test8(i32 %a) #0 {
entry:
  %0 = tail call i32 @llvm.hexagon.A2.andir(i32 %a, i32 -512)
  store i32 %0, i32* @d, align 4
  ret void
}

; CHECK: r{{[0-9]+}}{{ *}}={{ *}}or(r{{[0-9]+}}{{ *}},{{ *}}#511)

define void @test9(i32 %a) #0 {
entry:
  %0 = tail call i32 @llvm.hexagon.A2.orir(i32 %a, i32 511)
  store i32 %0, i32* @d, align 4
  ret void
}

; CHECK: r{{[0-9]+}}{{ *}}={{ *}}sub(#508{{ *}},{{ *}}r{{[0-9]+}})

define void @test10(i32 %a) #0 {
entry:
  %0 = tail call i32 @llvm.hexagon.A2.subri(i32 508, i32 %a)
  store i32 %0, i32* @d, align 4
  ret void
}

; CHECK: r{{[0-9]+}}.l{{ *}}={{ *}}#48242

define void @test11() #0 {
entry:
  %0 = load i32* @d, align 4
  %1 = tail call i32 @llvm.hexagon.A2.tfril(i32 %0, i32 48242)
  store i32 %1, i32* @d, align 4
  ret void
}

; CHECK: r{{[0-9]+}}.h{{ *}}={{ *}}#50826

define void @test12() #0 {
entry:
  %0 = load i32* @d, align 4
  %1 = tail call i32 @llvm.hexagon.A2.tfrih(i32 %0, i32 50826)
  store i32 %1, i32* @d, align 4
  ret void
}

; CHECK: r{{[0-9]+}}{{ *}}={{ *}}mux(p{{[0-3]+}}{{ *}},{{ *}}r{{[0-9]+}}{{ *}},{{ *}}##71230)

define void @test21(i32 %a) #0 {
entry:
  %0 = load i8* @b, align 1
  %conv = zext i8 %0 to i32
  %1 = tail call i32 @llvm.hexagon.C2.muxir(i32 %conv, i32 %a, i32 71230)
  store i32 %1, i32* @d, align 4
  ret void
}

; CHECK: r{{[0-9]+}}{{ *}}={{ *}}mux(p{{[0-3]+}}{{ *}},{{ *}}##5000{{ *}},{{ *}}r{{[0-9]+}})

define void @test23(i32 %a) #0 {
entry:
  %0 = load i8* @b, align 1
  %conv = zext i8 %0 to i32
  %1 = tail call i32 @llvm.hexagon.C2.muxri(i32 %conv, i32 5000, i32 %a)
  store i32 %1, i32* @d, align 4
  ret void
}

; CHECK: r{{[0-9]+}}{{ *}}={{ *}}mux(p{{[0-3]+}}{{ *}},{{ *}}##-4900{{ *}},{{ *}}#94)

define void @test24(i32 %a) #0 {
entry:
  %0 = load i8* @b, align 1
  %conv = zext i8 %0 to i32
  %1 = tail call i32 @llvm.hexagon.C2.muxii(i32 %conv, i32 -4900, i32 94)
  store i32 %1, i32* @d, align 4
  ret void
}

; CHECK: r{{[0-9]+}}:{{[0-9]+}}{{ *}}={{ *}}combine(##-1280{{ *}},{{ *}}#120)

define void @test25(i32 %a) #0 {
entry:
  %0 = tail call i64 @llvm.hexagon.A2.combineii(i32 -1280, i32 120)
  store i64 %0, i64* @c, align 4
  ret void
}

declare i32 @llvm.hexagon.A2.add(i32, i32) #1
declare i32 @llvm.hexagon.A2.sub(i32, i32) #1
declare i32 @llvm.hexagon.A2.and(i32, i32) #1
declare i32 @llvm.hexagon.A2.or(i32, i32) #1
declare i32 @llvm.hexagon.A2.xor(i32, i32) #1
declare i64 @llvm.hexagon.A2.combinew(i32, i32) #1
declare i32 @llvm.hexagon.A2.addi(i32, i32) #1
declare i32 @llvm.hexagon.A2.andir(i32, i32) #1
declare i32 @llvm.hexagon.A2.orir(i32, i32) #1
declare i32 @llvm.hexagon.A2.subri(i32, i32)
declare i32 @llvm.hexagon.A2.tfril(i32, i32) #1
declare i32 @llvm.hexagon.A2.tfrih(i32, i32) #1
declare i32 @llvm.hexagon.C2.muxir(i32, i32, i32) #1
declare i32 @llvm.hexagon.C2.muxri(i32, i32, i32) #1
declare i32 @llvm.hexagon.C2.muxii(i32, i32, i32) #1
declare i64 @llvm.hexagon.A2.combineii(i32, i32) #1
