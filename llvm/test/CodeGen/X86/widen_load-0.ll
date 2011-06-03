; RUN: llc < %s -o - -mtriple=x86_64-linux | FileCheck %s
; RUN: llc < %s -o - -mtriple=x86_64-win32 | FileCheck %s -check-prefix=WIN64
; PR4891

; Both loads should happen before either store.

; CHECK: movl  (%rdi), %[[R1:...]]
; CHECK: movl  (%rsi), %[[R2:...]]
; CHECK: movl  %[[R2]], (%rdi)
; CHECK: movl  %[[R1]], (%rsi)

; WIN64: movl  (%rcx), %[[R1:...]]
; WIN64: movl  (%rdx), %[[R2:...]]
; WIN64: movl  %[[R2]], (%rcx)
; WIN64: movl  %[[R1]], (%rdx)

define void @short2_int_swap(<2 x i16>* nocapture %b, i32* nocapture %c) nounwind {
entry:
  %0 = load <2 x i16>* %b, align 2                ; <<2 x i16>> [#uses=1]
  %1 = load i32* %c, align 4                      ; <i32> [#uses=1]
  %tmp1 = bitcast i32 %1 to <2 x i16>             ; <<2 x i16>> [#uses=1]
  store <2 x i16> %tmp1, <2 x i16>* %b, align 2
  %tmp5 = bitcast <2 x i16> %0 to <1 x i32>       ; <<1 x i32>> [#uses=1]
  %tmp3 = extractelement <1 x i32> %tmp5, i32 0   ; <i32> [#uses=1]
  store i32 %tmp3, i32* %c, align 4
  ret void
}
