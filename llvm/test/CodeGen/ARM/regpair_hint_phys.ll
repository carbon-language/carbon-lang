; RUN: llc -o - %s
; ARM target used to fail an assertion if RegPair{Odd|Even} hint pointed to a
; physreg.
target datalayout = "e-m:o-p:32:32-f64:32:64-v64:32:64-v128:32:128-a:0:32-n32-S32"
target triple = "thumbv7-apple-tvos8.3.0"

declare i8* @llvm.frameaddress(i32) #1
declare i8* @llvm.returnaddress(i32) #1

@somevar = global [2 x i32] [i32 0, i32 0]

define void @__ubsan_handle_shift_out_of_bounds() #0 {
entry:
  %0 = tail call i8* @llvm.frameaddress(i32 0)
  %1 = ptrtoint i8* %0 to i32
  %2 = tail call i8* @llvm.returnaddress(i32 0)
  %3 = ptrtoint i8* %2 to i32
  %val0 = insertvalue [2 x i32] [i32 undef, i32 undef], i32 %3, 0
  %val1 = insertvalue [2 x i32] %val0, i32 %1, 1
  store [2 x i32] %val1, [2 x i32]* @somevar, align 8
  ret void
}
