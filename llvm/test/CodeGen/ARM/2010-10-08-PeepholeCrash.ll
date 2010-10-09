; RUN: llc < %s -mtriple=thumbv7-apple-darwin10
; <rdar://problem/8529919>

%struct.GLDLightProduct = type { %struct.GLIColor4, %struct.GLIColor4, %struct.GLIColor4 }
%struct.GLDPointLineLimits = type { float, float, float }
%struct.GLIColor4 = type { float, float, float, float }
%struct.__GLmaterial = type { %struct.GLIColor4, %struct.GLIColor4, %struct.GLIColor4, %struct.GLIColor4, float, float, float, float, [8 x %struct.GLDLightProduct], %struct.GLIColor4, %struct.__GLsum, i32, %struct.__GLmaterial*, %struct.__GLmaterial*, [4 x i32] }
%struct.__GLsum = type { %struct.GLIColor4, i8, i8, i8, i8 }
%struct.__GLvertex = type { %struct.GLIColor4, %struct.GLIColor4, %struct.GLIColor4, %struct.GLIColor4, %struct.GLIColor4, %struct.GLDPointLineLimits, float, %struct.GLIColor4, float, i8, i8, i8, i8, float, float, i32, i32, i32, [4 x i8], [4 x float], [2 x %struct.__GLmaterial*], i32, i32, [32 x %struct.GLIColor4], [2 x %struct.GLIColor4] }

define void @func() nounwind {
entry:
  %0 = load i32* undef, align 4
  br i1 undef, label %gleLLVMVecPrimMulti.exit, label %bb.i9

bb.i9:                                            ; preds = %entry
  switch i32 undef, label %bb3.i11 [
    i32 4, label %bb8.i
    i32 8, label %bb5.i
  ]

bb3.i11:                                          ; preds = %bb.i9
  br label %bb8.i

bb5.i:                                            ; preds = %bb.i9
  br label %bb8.i

bb8.i:                                            ; preds = %bb5.i, %bb3.i11, %bb.i9
  br i1 undef, label %gleLLVMVecPrimMulti.exit, label %bb12.i

bb12.i:                                           ; preds = %bb8.i
  br i1 undef, label %bb15.i, label %bb13.i

bb13.i:                                           ; preds = %bb12.i
  br i1 undef, label %bb16.i, label %bb14.i

bb14.i:                                           ; preds = %bb13.i
  unreachable

bb15.i:                                           ; preds = %bb12.i
  unreachable

bb16.i:                                           ; preds = %bb13.i
  br i1 undef, label %bb18.i, label %bb17.i

bb17.i:                                           ; preds = %bb16.i
  br label %bb18.i

bb18.i:                                           ; preds = %bb17.i, %bb16.i
  br i1 undef, label %bb19.i, label %gleLLVMVecPrimMulti.exit

bb19.i:                                           ; preds = %bb18.i
  %1 = and i32 %0, 16
  %2 = icmp eq i32 %1, 0
  %invok.1.i = select i1 %2, i32 undef, i32 0
  br i1 undef, label %bb27.i, label %bb26.i

bb26.i:                                           ; preds = %bb19.i
  unreachable

bb27.i:                                           ; preds = %bb19.i
  br i1 undef, label %bb29.i, label %bb28.i

bb28.i:                                           ; preds = %bb27.i
  unreachable

bb29.i:                                           ; preds = %bb27.i
  br i1 false, label %bb360.preheader.i, label %bb30.i

bb30.i:                                           ; preds = %bb29.i
  unreachable

bb360.preheader.i:                                ; preds = %bb29.i
  %tmp119 = add i32 %invok.1.i, 0
  br i1 undef, label %bb32.i, label %gleLLVMVecPrimMulti.exit

bb32.i:                                           ; preds = %bb360.preheader.i
  br label %bb179.i

bb179.i:                                          ; preds = %bb216.i, %bb32.i
  %tmp120 = add i32 %tmp119, 0
  br i1 undef, label %bb181.i, label %bb180.i

bb180.i:                                          ; preds = %bb179.i
  %scevgep810.i = getelementptr %struct.__GLvertex* null, i32 %tmp120, i32 15
  store i32 undef, i32* %scevgep810.i, align 4
  br label %bb181.i

bb181.i:                                          ; preds = %bb180.i, %bb179.i
  switch i32 undef, label %bb216.i [
    i32 4, label %bb187.i
    i32 6, label %bb197.i
  ]

bb187.i:                                          ; preds = %bb181.i
  br label %bb216.i

bb197.i:                                          ; preds = %bb181.i
  br i1 %2, label %bb199.i, label %bb198.i

bb198.i:                                          ; preds = %bb197.i
  br label %bb216.i

bb199.i:                                          ; preds = %bb197.i
  br label %bb216.i

bb216.i:                                          ; preds = %bb199.i, %bb198.i, %bb187.i, %bb181.i
  br label %bb179.i

gleLLVMVecPrimMulti.exit:                         ; preds = %bb360.preheader.i, %bb18.i, %bb8.i, %entry
  br i1 undef, label %gleLLVMVecCleanMaterials.exit, label %bb2

bb2:                                              ; preds = %gleLLVMVecPrimMulti.exit
  ret void

gleLLVMVecCleanMaterials.exit:                    ; preds = %gleLLVMVecPrimMulti.exit
  ret void
}
