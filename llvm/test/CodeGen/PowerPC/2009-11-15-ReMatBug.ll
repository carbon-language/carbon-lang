; RUN: llc < %s -mtriple=powerpc-apple-darwin8

%struct.FILE = type { i8*, i32, i32, i16, i16, %struct.__sbuf, i32, i8*, i32 (i8*)*, i32 (i8*, i8*, i32)*, i64 (i8*, i64, i32)*, i32 (i8*, i8*, i32)*, %struct.__sbuf, %struct.__sFILEX*, i32, [3 x i8], [1 x i8], %struct.__sbuf, i32, i64 }
%struct.__gcov_var = type { %struct.FILE*, i32, i32, i32, i32, i32, i32, [1025 x i32] }
%struct.__sFILEX = type opaque
%struct.__sbuf = type { i8*, i32 }
%struct.gcov_ctr_info = type { i32, i64*, void (i64*, i32)* }
%struct.gcov_ctr_summary = type { i32, i32, i64, i64, i64 }
%struct.gcov_fn_info = type { i32, i32, [0 x i32] }
%struct.gcov_info = type { i32, %struct.gcov_info*, i32, i8*, i32, %struct.gcov_fn_info*, i32, [0 x %struct.gcov_ctr_info] }
%struct.gcov_summary = type { i32, [1 x %struct.gcov_ctr_summary] }

@__gcov_var = external global %struct.__gcov_var  ; <%struct.__gcov_var*> [#uses=1]
@__sF = external global [0 x %struct.FILE]        ; <[0 x %struct.FILE]*> [#uses=1]
@.str = external constant [56 x i8], align 4      ; <[56 x i8]*> [#uses=1]
@gcov_list = external global %struct.gcov_info*   ; <%struct.gcov_info**> [#uses=1]
@.str7 = external constant [35 x i8], align 4     ; <[35 x i8]*> [#uses=1]
@.str8 = external constant [9 x i8], align 4      ; <[9 x i8]*> [#uses=1]
@.str9 = external constant [10 x i8], align 4     ; <[10 x i8]*> [#uses=1]
@.str10 = external constant [36 x i8], align 4    ; <[36 x i8]*> [#uses=1]

declare i32 @"\01_fprintf$LDBL128"(%struct.FILE*, i8*, ...) nounwind

define void @gcov_exit() nounwind {
entry:
  %gi_ptr.0357 = load %struct.gcov_info** @gcov_list, align 4 ; <%struct.gcov_info*> [#uses=1]
  %0 = alloca i8, i32 undef, align 1              ; <i8*> [#uses=3]
  br i1 undef, label %return, label %bb.nph341

bb.nph341:                                        ; preds = %entry
  %object27 = bitcast %struct.gcov_summary* undef to i8* ; <i8*> [#uses=1]
  br label %bb25

bb25:                                             ; preds = %read_fatal, %bb.nph341
  %gi_ptr.1329 = phi %struct.gcov_info* [ %gi_ptr.0357, %bb.nph341 ], [ undef, %read_fatal ] ; <%struct.gcov_info*> [#uses=1]
  call void @llvm.memset.i32(i8* %object27, i8 0, i32 36, i32 8)
  br i1 undef, label %bb49.1, label %bb48

bb48:                                             ; preds = %bb25
  br label %bb49.1

bb51:                                             ; preds = %bb48.4, %bb49.3
  switch i32 undef, label %bb58 [
    i32 0, label %rewrite
    i32 1734567009, label %bb59
  ]

bb58:                                             ; preds = %bb51
  %1 = call i32 (%struct.FILE*, i8*, ...)* @"\01_fprintf$LDBL128"(%struct.FILE* getelementptr inbounds ([0 x %struct.FILE]* @__sF, i32 0, i32 2), i8* getelementptr inbounds ([35 x i8]* @.str7, i32 0, i32 0), i8* %0) nounwind ; <i32> [#uses=0]
  br label %read_fatal

bb59:                                             ; preds = %bb51
  br i1 undef, label %bb60, label %bb3.i156

bb3.i156:                                         ; preds = %bb59
  store i8 52, i8* undef, align 1
  store i8 42, i8* undef, align 1
  %2 = call i32 (%struct.FILE*, i8*, ...)* @"\01_fprintf$LDBL128"(%struct.FILE* getelementptr inbounds ([0 x %struct.FILE]* @__sF, i32 0, i32 2), i8* getelementptr inbounds ([56 x i8]* @.str, i32 0, i32 0), i8* %0, i8* undef, i8* undef) nounwind ; <i32> [#uses=0]
  br label %read_fatal

bb60:                                             ; preds = %bb59
  br i1 undef, label %bb78.preheader, label %rewrite

bb78.preheader:                                   ; preds = %bb60
  br i1 undef, label %bb62, label %bb80

bb62:                                             ; preds = %bb78.preheader
  br i1 undef, label %bb64, label %read_mismatch

bb64:                                             ; preds = %bb62
  br i1 undef, label %bb65, label %read_mismatch

bb65:                                             ; preds = %bb64
  br i1 undef, label %bb75, label %read_mismatch

read_mismatch:                                    ; preds = %bb98, %bb119.preheader, %bb72, %bb71, %bb65, %bb64, %bb62
  %3 = icmp eq i32 undef, -1                      ; <i1> [#uses=1]
  %iftmp.11.0 = select i1 %3, i8* getelementptr inbounds ([10 x i8]* @.str9, i32 0, i32 0), i8* getelementptr inbounds ([9 x i8]* @.str8, i32 0, i32 0) ; <i8*> [#uses=1]
  %4 = call i32 (%struct.FILE*, i8*, ...)* @"\01_fprintf$LDBL128"(%struct.FILE* getelementptr inbounds ([0 x %struct.FILE]* @__sF, i32 0, i32 2), i8* getelementptr inbounds ([36 x i8]* @.str10, i32 0, i32 0), i8* %0, i8* %iftmp.11.0) nounwind ; <i32> [#uses=0]
  br label %read_fatal

bb71:                                             ; preds = %bb75
  %5 = load i32* undef, align 4                   ; <i32> [#uses=1]
  %6 = getelementptr inbounds %struct.gcov_info* %gi_ptr.1329, i32 0, i32 7, i32 undef, i32 2 ; <void (i64*, i32)**> [#uses=1]
  %7 = load void (i64*, i32)** %6, align 4        ; <void (i64*, i32)*> [#uses=1]
  %8 = call i32 @__gcov_read_unsigned() nounwind  ; <i32> [#uses=1]
  %9 = call i32 @__gcov_read_unsigned() nounwind  ; <i32> [#uses=1]
  %10 = icmp eq i32 %tmp386, %8                   ; <i1> [#uses=1]
  br i1 %10, label %bb72, label %read_mismatch

bb72:                                             ; preds = %bb71
  %11 = icmp eq i32 undef, %9                     ; <i1> [#uses=1]
  br i1 %11, label %bb73, label %read_mismatch

bb73:                                             ; preds = %bb72
  call void %7(i64* null, i32 %5) nounwind
  unreachable

bb74:                                             ; preds = %bb75
  %12 = add i32 %13, 1                            ; <i32> [#uses=1]
  br label %bb75

bb75:                                             ; preds = %bb74, %bb65
  %13 = phi i32 [ %12, %bb74 ], [ 0, %bb65 ]      ; <i32> [#uses=2]
  %tmp386 = add i32 0, 27328512                   ; <i32> [#uses=1]
  %14 = shl i32 1, %13                            ; <i32> [#uses=1]
  %15 = load i32* undef, align 4                  ; <i32> [#uses=1]
  %16 = and i32 %15, %14                          ; <i32> [#uses=1]
  %17 = icmp eq i32 %16, 0                        ; <i1> [#uses=1]
  br i1 %17, label %bb74, label %bb71

bb80:                                             ; preds = %bb78.preheader
  unreachable

read_fatal:                                       ; preds = %read_mismatch, %bb3.i156, %bb58
  br i1 undef, label %return, label %bb25

rewrite:                                          ; preds = %bb60, %bb51
  store i32 -1, i32* getelementptr inbounds (%struct.__gcov_var* @__gcov_var, i32 0, i32 6), align 4
  br i1 undef, label %bb94, label %bb119.preheader

bb94:                                             ; preds = %rewrite
  unreachable

bb119.preheader:                                  ; preds = %rewrite
  br i1 undef, label %read_mismatch, label %bb98

bb98:                                             ; preds = %bb119.preheader
  br label %read_mismatch

return:                                           ; preds = %read_fatal, %entry
  ret void

bb49.1:                                           ; preds = %bb48, %bb25
  br i1 undef, label %bb49.2, label %bb48.2

bb49.2:                                           ; preds = %bb48.2, %bb49.1
  br i1 undef, label %bb49.3, label %bb48.3

bb48.2:                                           ; preds = %bb49.1
  br label %bb49.2

bb49.3:                                           ; preds = %bb48.3, %bb49.2
  br i1 undef, label %bb51, label %bb48.4

bb48.3:                                           ; preds = %bb49.2
  br label %bb49.3

bb48.4:                                           ; preds = %bb49.3
  br label %bb51
}

declare i32 @__gcov_read_unsigned() nounwind

declare void @llvm.memset.i32(i8* nocapture, i8, i32, i32) nounwind
