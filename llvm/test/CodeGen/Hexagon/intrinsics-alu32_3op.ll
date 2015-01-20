; RUN: llc < %s | FileCheck %s

target datalayout = "e-m:e-p:32:32-i1:32-i64:64-a:0-v32:32-n16:32"
target triple = "hexagon"

; CHECK: test13:
; CHECK: r{{[0-9]+}} = add(r{{[0-9]+}}, r{{[0-9]+}}):sat
define i32 @test13(i32 %Rs, i32 %Rt) #0 {
entry:
  %0 = tail call i32 @llvm.hexagon.A2.addsat(i32 %Rs, i32 %Rt)
  ret i32 %0
}


; CHECK: test14:
; CHECK: r{{[0-9]+}} = sub(r{{[0-9]+}}, r{{[0-9]+}}):sat
define i32 @test14(i32 %Rs, i32 %Rt) #0 {
entry:
  %0 = tail call i32 @llvm.hexagon.A2.subsat(i32 %Rs, i32 %Rt)
  ret i32 %0
}


; CHECK: test61:
; CHECK: r{{[0-9]+:[0-9]+}} = packhl(r{{[0-9]+}}, r{{[0-9]+}})
define i64 @test61(i32 %Rs, i32 %Rt) #0 {
entry:
  %0 = tail call i64 @llvm.hexagon.S2.packhl(i32 %Rs, i32 %Rt)
  ret i64 %0
}

declare i32 @llvm.hexagon.A2.addsat(i32, i32) #1
declare i32 @llvm.hexagon.A2.subsat(i32, i32) #1
declare i64 @llvm.hexagon.S2.packhl(i32, i32) #1
