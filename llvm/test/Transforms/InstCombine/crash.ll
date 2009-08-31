; RUN: llvm-as < %s | opt -instcombine | llvm-dis
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin10.0"

define i32 @_Z9model8bitR5Mixeri(i8 %tmp2) ssp {
entry:
  %tmp3 = zext i8 %tmp2 to i32
  %tmp8 = lshr i32 %tmp3, 6 
  %tmp9 = lshr i32 %tmp3, 7 
  %tmp10 = xor i32 %tmp9, 67108858
  %tmp11 = xor i32 %tmp10, %tmp8 
  %tmp12 = xor i32 %tmp11, 0     
  ret i32 %tmp12
}
