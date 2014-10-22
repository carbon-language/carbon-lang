; XFAIL: *
; RUN: llc -march=r600 -mcpu=SI -verify-machineinstrs < %s

define void @init_data_cost_reduce_0(i32 %arg) #0 {
bb:
  br i1 undef, label %bb1, label %bb2

bb1:                                              ; preds = %bb
  br label %bb2

bb2:                                              ; preds = %bb1, %bb
  br i1 undef, label %bb3, label %bb4

bb3:                                              ; preds = %bb2
  %tmp = mul i32 undef, %arg
  br label %bb4

bb4:                                              ; preds = %bb3, %bb2
  unreachable
}

attributes #0 = { nounwind }
