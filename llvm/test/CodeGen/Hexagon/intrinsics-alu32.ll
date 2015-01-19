; RUN: llc -march=hexagon < %s | FileCheck %s

; Verify that ALU32 - aslh, asrh, sxth, sxtb, zxth, zxtb  intrinsics
; are lowered to the right instructions.

@c = external global i64

; CHECK: r{{[0-9]+}}{{ *}}={{ *}}aslh({{ *}}r{{[0-9]+}}{{ *}})
define void @test1(i32 %a) #0 {
entry:
  %0 = tail call i32 @llvm.hexagon.A2.aslh(i32 %a)
  %conv = sext i32 %0 to i64
  store i64 %conv, i64* @c, align 8
  ret void
}

declare i32 @llvm.hexagon.A2.aslh(i32) #1

; CHECK: r{{[0-9]+}}{{ *}}={{ *}}asrh({{ *}}r{{[0-9]+}}{{ *}})
define void @test2(i32 %a) #0 {
entry:
  %0 = tail call i32 @llvm.hexagon.A2.asrh(i32 %a)
  %conv = sext i32 %0 to i64
  store i64 %conv, i64* @c, align 8
  ret void
}

declare i32 @llvm.hexagon.A2.asrh(i32) #1

; CHECK: r{{[0-9]+}}{{ *}}={{ *}}sxtb({{ *}}r{{[0-9]+}}{{ *}})
define void @test3(i32 %a) #0 {
entry:
  %0 = tail call i32 @llvm.hexagon.A2.sxtb(i32 %a)
  %conv = sext i32 %0 to i64
  store i64 %conv, i64* @c, align 8
  ret void
}

declare i32 @llvm.hexagon.A2.sxtb(i32) #1

; CHECK: r{{[0-9]+}}{{ *}}={{ *}}sxth({{ *}}r{{[0-9]+}}{{ *}})
define void @test4(i32 %a) #0 {
entry:
  %0 = tail call i32 @llvm.hexagon.A2.sxth(i32 %a)
  %conv = sext i32 %0 to i64
  store i64 %conv, i64* @c, align 8
  ret void
}

declare i32 @llvm.hexagon.A2.sxth(i32) #1

; CHECK: r{{[0-9]+}}{{ *}}={{ *}}zxtb({{ *}}r{{[0-9]+}}{{ *}})
define void @test6(i32 %a) #0 {
entry:
  %0 = tail call i32 @llvm.hexagon.A2.zxtb(i32 %a)
  %conv = sext i32 %0 to i64
  store i64 %conv, i64* @c, align 8
  ret void
}

declare i32 @llvm.hexagon.A2.zxtb(i32) #1

; CHECK: r{{[0-9]+}}{{ *}}={{ *}}zxth({{ *}}r{{[0-9]+}}{{ *}})
define void @test7(i32 %a) #0 {
entry:
  %0 = tail call i32 @llvm.hexagon.A2.zxth(i32 %a)
  %conv = sext i32 %0 to i64
  store i64 %conv, i64* @c, align 8
  ret void
}

declare i32 @llvm.hexagon.A2.zxth(i32) #1

; CHECK: r{{[0-9]+}}{{ *}}={{ *}}asrh({{ *}}r{{[0-9]+}}{{ *}})
define void @test8(i32 %a) #0 {
entry:
  %0 = tail call i32 @llvm.hexagon.SI.to.SXTHI.asrh(i32 %a)
  %conv = sext i32 %0 to i64
  store i64 %conv, i64* @c, align 8
  ret void
}

declare i32 @llvm.hexagon.SI.to.SXTHI.asrh(i32) #1
