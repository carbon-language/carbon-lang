; rdar://7860110
; RUN: llc < %s | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin10.2"

define void @test1(i32* nocapture %a0, i8 zeroext %a1) nounwind ssp {
entry:
  %A = load i32* %a0, align 4
  %B = and i32 %A, -256     ; 0xFFFFFF00
  %C = zext i8 %a1 to i32
  %D = or i32 %C, %B
  store i32 %D, i32* %a0, align 4
  ret void
  
; CHECK: test1:
; CHECK: movb	%sil, (%rdi)
}

define void @test2(i32* nocapture %a0, i8 zeroext %a1) nounwind ssp {
entry:
  %A = load i32* %a0, align 4
  %B = and i32 %A, -65281    ; 0xFFFF00FF
  %C = zext i8 %a1 to i32
  %CS = shl i32 %C, 8
  %D = or i32 %B, %CS
  store i32 %D, i32* %a0, align 4
  ret void
; CHECK: test2:
; CHECK: movb	%sil, 1(%rdi)
}

define void @test3(i32* nocapture %a0, i16 zeroext %a1) nounwind ssp {
entry:
  %A = load i32* %a0, align 4
  %B = and i32 %A, -65536    ; 0xFFFF0000
  %C = zext i16 %a1 to i32
  %D = or i32 %B, %C
  store i32 %D, i32* %a0, align 4
  ret void
; CHECK: test3:
; CHECK: movw	%si, (%rdi)
}

define void @test4(i32* nocapture %a0, i16 zeroext %a1) nounwind ssp {
entry:
  %A = load i32* %a0, align 4
  %B = and i32 %A, 65535    ; 0x0000FFFF
  %C = zext i16 %a1 to i32
  %CS = shl i32 %C, 16
  %D = or i32 %B, %CS
  store i32 %D, i32* %a0, align 4
  ret void
; CHECK: test4:
; CHECK: movw	%si, 2(%rdi)
}

define void @test5(i64* nocapture %a0, i16 zeroext %a1) nounwind ssp {
entry:
  %A = load i64* %a0, align 4
  %B = and i64 %A, -4294901761    ; 0xFFFFFFFF0000FFFF
  %C = zext i16 %a1 to i64
  %CS = shl i64 %C, 16
  %D = or i64 %B, %CS
  store i64 %D, i64* %a0, align 4
  ret void
; CHECK: test5:
; CHECK: movw	%si, 2(%rdi)
}

define void @test6(i64* nocapture %a0, i8 zeroext %a1) nounwind ssp {
entry:
  %A = load i64* %a0, align 4
  %B = and i64 %A, -280375465082881    ; 0xFFFF00FFFFFFFFFF
  %C = zext i8 %a1 to i64
  %CS = shl i64 %C, 40
  %D = or i64 %B, %CS
  store i64 %D, i64* %a0, align 4
  ret void
; CHECK: test6:
; CHECK: movb	%sil, 5(%rdi)
}
