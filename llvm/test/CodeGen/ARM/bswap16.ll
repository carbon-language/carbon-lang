; RUN: llc -mtriple=arm-darwin  -mattr=v6 < %s | FileCheck %s
; RUN: llc -mtriple=thumb-darwin  -mattr=v6 < %s | FileCheck %s


define void @test1(i16* nocapture %data) {
entry:
  %0 = load i16, i16* %data, align 2
  %1 = tail call i16 @llvm.bswap.i16(i16 %0)
  store i16 %1, i16* %data, align 2
  ret void

  ; CHECK-LABEL: test1:
  ; CHECK: ldrh r[[R1:[0-9]+]], [r0]
  ; CHECK: rev16 r[[R1]], r[[R1]]
  ; CHECK: strh r[[R1]], [r0]
}


define void @test2(i16* nocapture %data, i16 zeroext %in) {
entry:
  %0 = tail call i16 @llvm.bswap.i16(i16 %in)
  store i16 %0, i16* %data, align 2
  ret void

  ; CHECK-LABEL: test2:
  ; CHECK: rev16 r[[R1:[0-9]+]], r1
  ; CHECK: strh r[[R1]], [r0]
}


define i16 @test3(i16* nocapture %data) {
entry:
  %0 = load i16, i16* %data, align 2
  %1 = tail call i16 @llvm.bswap.i16(i16 %0)
  ret i16 %1

  ; CHECK-LABEL: test3:
  ; CHECK: ldrh r[[R0:[0-9]+]], [r0]
  ; CHECK: rev16 r[[R0]], r0
}

declare i16 @llvm.bswap.i16(i16)
