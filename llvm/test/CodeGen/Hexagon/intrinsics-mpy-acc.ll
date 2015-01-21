; RUN: llc -march=hexagon -mcpu=hexagonv5 < %s | FileCheck %s

; Verify that the mpy intrinsics with add/subtract are being lowered to the right instruction.

@c = external global i64

; CHECK: r{{[0-9]+}}{{ *}}+{{ *}}={{ *}}mpyi(r{{[0-9]+}}{{ *}},{{ *}}#124)

define void @test1(i32 %a) #0 {
entry:
  %0 = load i64* @c, align 8
  %conv = trunc i64 %0 to i32
  %1 = tail call i32 @llvm.hexagon.M2.macsip(i32 %conv, i32 %a, i32 124)
  %conv1 = sext i32 %1 to i64
  store i64 %conv1, i64* @c, align 8
  ret void
}

declare i32 @llvm.hexagon.M2.macsip(i32, i32, i32) #1

; CHECK: r{{[0-9]+}}{{ *}}-{{ *}}={{ *}}mpyi(r{{[0-9]+}}{{ *}},{{ *}}#166)

define void @test2(i32 %a) #0 {
entry:
  %0 = load i64* @c, align 8
  %conv = trunc i64 %0 to i32
  %1 = tail call i32 @llvm.hexagon.M2.macsin(i32 %conv, i32 %a, i32 166)
  %conv1 = sext i32 %1 to i64
  store i64 %conv1, i64* @c, align 8
  ret void
}

declare i32 @llvm.hexagon.M2.macsin(i32, i32, i32) #1

; CHECK: r{{[0-9]+}}{{ *}}+{{ *}}={{ *}}mpyi(r{{[0-9]+}}{{ *}},{{ *}}r{{[0-9]+}})

define void @test3(i32 %a, i32 %b) #0 {
entry:
  %0 = load i64* @c, align 8
  %conv = trunc i64 %0 to i32
  %1 = tail call i32 @llvm.hexagon.M2.maci(i32 %conv, i32 %a, i32 %b)
  %conv1 = sext i32 %1 to i64
  store i64 %conv1, i64* @c, align 8
  ret void
}

declare i32 @llvm.hexagon.M2.maci(i32, i32, i32) #1

@d = external global i32

; CHECK: r{{[0-9]+}}{{ *}}+{{ *}}={{ *}}add(r{{[0-9]+}}{{ *}},{{ *}}#40)

define void @test7(i32 %a) #0 {
entry:
  %0 = load i64* @c, align 8
  %conv = trunc i64 %0 to i32
  %1 = tail call i32 @llvm.hexagon.M2.accii(i32 %conv, i32 %a, i32 40)
  %conv1 = sext i32 %1 to i64
  store i64 %conv1, i64* @c, align 8
  ret void
}

declare i32 @llvm.hexagon.M2.accii(i32, i32, i32) #1

; CHECK: r{{[0-9]+}}{{ *}}-{{ *}}={{ *}}add(r{{[0-9]+}}{{ *}},{{ *}}#100)

define void @test8(i32 %a) #0 {
entry:
  %0 = load i64* @c, align 8
  %conv = trunc i64 %0 to i32
  %1 = tail call i32 @llvm.hexagon.M2.naccii(i32 %conv, i32 %a, i32 100)
  %conv1 = sext i32 %1 to i64
  store i64 %conv1, i64* @c, align 8
  ret void
}

declare i32 @llvm.hexagon.M2.naccii(i32, i32, i32) #1


; CHECK: r{{[0-9]+}}{{ *}}+{{ *}}={{ *}}add(r{{[0-9]+}}{{ *}},{{ *}}r{{[0-9]+}})

define void @test9(i32 %a, i32 %b) #0 {
entry:
  %0 = load i64* @c, align 8
  %conv = trunc i64 %0 to i32
  %1 = tail call i32 @llvm.hexagon.M2.acci(i32 %conv, i32 %a, i32 %b)
  %conv1 = sext i32 %1 to i64
  store i64 %conv1, i64* @c, align 8
  ret void
}

declare i32 @llvm.hexagon.M2.acci(i32, i32, i32) #1

; CHECK: r{{[0-9]+}}{{ *}}+{{ *}}={{ *}}sub(r{{[0-9]+}}{{ *}},{{ *}}r{{[0-9]+}})

define void @test10(i32 %a, i32 %b) #0 {
entry:
  %0 = load i64* @c, align 8
  %conv = trunc i64 %0 to i32
  %1 = tail call i32 @llvm.hexagon.M2.subacc(i32 %conv, i32 %a, i32 %b)
  %conv1 = sext i32 %1 to i64
  store i64 %conv1, i64* @c, align 8
  ret void
}

declare i32 @llvm.hexagon.M2.subacc(i32, i32, i32) #1

; CHECK: r{{[0-9]+}}{{ *}}-{{ *}}={{ *}}add(r{{[0-9]+}}{{ *}},{{ *}}r{{[0-9]+}})

define void @test11(i32 %a, i32 %b) #0 {
entry:
  %0 = load i64* @c, align 8
  %conv = trunc i64 %0 to i32
  %1 = tail call i32 @llvm.hexagon.M2.nacci(i32 %conv, i32 %a, i32 %b)
  %conv1 = sext i32 %1 to i64
  store i64 %conv1, i64* @c, align 8
  ret void
}

declare i32 @llvm.hexagon.M2.nacci(i32, i32, i32) #1
