; RUN: llc < %s -mtriple=armv7-apple-darwin   -mattr=+v7,+mp -show-mc-encoding | FileCheck %s -check-prefix=ARM
; RUN: llc < %s -mtriple=thumbv7-apple-darwin -mattr=+v7     -show-mc-encoding | FileCheck %s -check-prefix=T2
; rdar://8924681

define void @t1(i8* %ptr) nounwind  {
entry:
; ARM: t1:
; ARM: pldw [r0]                        @ encoding: [0x00,0xf0,0x90,0xf5]
; ARM: pld [r0]                         @ encoding: [0x00,0xf0,0xd0,0xf5]

; T2: t1:
; T2: pld [r0]                      @ encoding: [0x90,0xf8,0x00,0xf0]
  tail call void @llvm.prefetch( i8* %ptr, i32 1, i32 3 )
  tail call void @llvm.prefetch( i8* %ptr, i32 0, i32 3 )
  ret void
}

define void @t2(i8* %ptr) nounwind  {
entry:
; ARM: t2:
; ARM: pld [r0, #1023]                  @ encoding: [0xff,0xf3,0xd0,0xf5]

; T2: t2:
; T2: pld [r0, #1023]               @ encoding: [0x90,0xf8,0xff,0xf3]
  %tmp = getelementptr i8* %ptr, i32 1023
  tail call void @llvm.prefetch( i8* %tmp, i32 0, i32 3 )
  ret void
}

define void @t3(i32 %base, i32 %offset) nounwind  {
entry:
; ARM: t3:
; ARM: pld [r0, r1, lsr #2]             @ encoding: [0x21,0xf1,0xd0,0xf7]

; T2: t3:
; T2: pld [r0, r1]                  @ encoding: [0x10,0xf8,0x01,0xf0]
  %tmp1 = lshr i32 %offset, 2
  %tmp2 = add i32 %base, %tmp1
  %tmp3 = inttoptr i32 %tmp2 to i8*
  tail call void @llvm.prefetch( i8* %tmp3, i32 0, i32 3 )
  ret void
}

define void @t4(i32 %base, i32 %offset) nounwind  {
entry:
; ARM: t4:
; ARM: pld [r0, r1, lsl #2]             @ encoding: [0x01,0xf1,0xd0,0xf7]

; T2: t4:
; T2: pld [r0, r1, lsl #2]          @ encoding: [0x10,0xf8,0x21,0xf0]
  %tmp1 = shl i32 %offset, 2
  %tmp2 = add i32 %base, %tmp1
  %tmp3 = inttoptr i32 %tmp2 to i8*
  tail call void @llvm.prefetch( i8* %tmp3, i32 0, i32 3 )
  ret void
}

declare void @llvm.prefetch(i8*, i32, i32) nounwind 
