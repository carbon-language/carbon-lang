; RUN: opt -instcombine -S < %s | FileCheck %s
; <rdar://problem/8606771>

define i32 @main(i32 %argc) nounwind ssp {
entry:
  %tmp3151 = trunc i32 %argc to i8
  %tmp3161 = or i8 %tmp3151, -17
  %tmp3162 = and i8 %tmp3151, 122
  %tmp3163 = xor i8 %tmp3162, -17
  %tmp4114 = shl i8 %tmp3163, 6
  %tmp4115 = xor i8 %tmp4114, %tmp3163
  %tmp4120 = xor i8 %tmp3161, %tmp4115
; CHECK: lshr i8 %tmp4115, 1
; CHECK-NOT: shl i8 %tmp4126, 6
  %tmp4126 = lshr i8 %tmp4120, 7
  %tmp4127 = mul i8 %tmp4126, 64
  %tmp4086 = zext i8 %tmp4127 to i32
; CHECK: ret i32
  ret i32 %tmp4086
}
