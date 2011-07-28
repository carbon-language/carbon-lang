; RUN: llc < %s -mtriple=thumbv7-apple-darwin10

; <rdar://problem/8529919>
%struct.foo = type { i32, i32 }

define void @func() nounwind {
entry:
  %tmp = load i32* undef, align 4
  br label %bb1

bb1:
  %tmp1 = and i32 %tmp, 16
  %tmp2 = icmp eq i32 %tmp1, 0
  %invok.1.i = select i1 %tmp2, i32 undef, i32 0
  %tmp119 = add i32 %invok.1.i, 0
  br i1 undef, label %bb2, label %exit

bb2:
  %tmp120 = add i32 %tmp119, 0
  %scevgep810.i = getelementptr %struct.foo* null, i32 %tmp120, i32 1
  store i32 undef, i32* %scevgep810.i, align 4
  br i1 undef, label %bb2, label %bb3

bb3:
  br i1 %tmp2, label %bb2, label %bb2

exit:
  ret void
}

; PR10520 - REG_SEQUENCE with implicit-def operands.
define arm_aapcs_vfpcc void @foo() nounwind align 2 {
bb:
  %tmp = shufflevector <2 x i64> undef, <2 x i64> undef, <1 x i32> <i32 1>
  %tmp8 = bitcast <1 x i64> %tmp to <2 x float>
  %tmp9 = shufflevector <2 x float> %tmp8, <2 x float> %tmp8, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %tmp10 = fmul <4 x float> undef, %tmp9
  %tmp11 = fadd <4 x float> %tmp10, undef
  %tmp12 = fadd <4 x float> undef, %tmp11
  %tmp13 = bitcast <4 x float> %tmp12 to i128
  %tmp14 = bitcast i128 %tmp13 to <4 x float>
  %tmp15 = bitcast <4 x float> %tmp14 to i128
  %tmp16 = bitcast i128 %tmp15 to <4 x float>
  %tmp17 = bitcast <4 x float> %tmp16 to i128
  %tmp18 = bitcast i128 %tmp17 to <4 x float>
  %tmp19 = bitcast <4 x float> %tmp18 to i128
  %tmp20 = bitcast i128 %tmp19 to <4 x float>
  store <4 x float> %tmp20, <4 x float>* undef, align 16
  ret void
}
