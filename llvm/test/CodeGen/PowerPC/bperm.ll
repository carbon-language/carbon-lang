; RUN: llc -mcpu=pwr7 < %s | FileCheck %s
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

; Function Attrs: nounwind readnone
define zeroext i32 @bs4(i32 zeroext %a) #0 {
entry:
  %0 = tail call i32 @llvm.bswap.i32(i32 %a)
  ret i32 %0

; CHECK-LABEL: @bs4
; CHECK: rlwinm [[REG1:[0-9]+]], 3, 8, 0, 31
; CHECK: rlwimi [[REG1]], 3, 24, 16, 23
; CHECK: rlwimi [[REG1]], 3, 24, 0, 7
; CHECK: mr 3, [[REG1]]
; CHECK: blr
}

; Function Attrs: nounwind readnone
define zeroext i32 @test6(i32 zeroext %x) #0 {
entry:
  %and = lshr i32 %x, 16
  %shr = and i32 %and, 255
  %and1 = shl i32 %x, 16
  %shl = and i32 %and1, 16711680
  %or = or i32 %shr, %shl
  ret i32 %or

; CHECK-LABEL: @test6
; CHECK: rlwinm [[REG1:[0-9]+]], 3, 16, 24, 31
; CHECK: rlwimi [[REG1]], 3, 16, 8, 15
; CHECK: mr 3, [[REG1]]
; CHECK: blr
}

; Function Attrs: nounwind readnone
declare i32 @llvm.bswap.i32(i32) #0

attributes #0 = { nounwind readnone }

