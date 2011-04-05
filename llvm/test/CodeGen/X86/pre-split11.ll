; RUN: llc < %s -mtriple=x86_64-apple-darwin -mattr=+sse2 -pre-alloc-split -regalloc=linearscan | FileCheck %s

@.str = private constant [28 x i8] c"\0A\0ADOUBLE            D = %f\0A\00", align 1 ; <[28 x i8]*> [#uses=1]
@.str1 = private constant [37 x i8] c"double to long    l1 = %ld\09\09(0x%lx)\0A\00", align 8 ; <[37 x i8]*> [#uses=1]
@.str2 = private constant [35 x i8] c"double to uint   ui1 = %u\09\09(0x%x)\0A\00", align 8 ; <[35 x i8]*> [#uses=1]
@.str3 = private constant [37 x i8] c"double to ulong  ul1 = %lu\09\09(0x%lx)\0A\00", align 8 ; <[37 x i8]*> [#uses=1]

define i32 @main(i32 %argc, i8** nocapture %argv) nounwind ssp {
; CHECK: movsd %xmm0, (%rsp)
entry:
  %0 = icmp sgt i32 %argc, 4                      ; <i1> [#uses=1]
  br i1 %0, label %bb, label %bb2

bb:                                               ; preds = %entry
  %1 = getelementptr inbounds i8** %argv, i64 4   ; <i8**> [#uses=1]
  %2 = load i8** %1, align 8                      ; <i8*> [#uses=1]
  %3 = tail call double @atof(i8* %2) nounwind    ; <double> [#uses=1]
  br label %bb2

bb2:                                              ; preds = %bb, %entry
  %storemerge = phi double [ %3, %bb ], [ 2.000000e+00, %entry ] ; <double> [#uses=4]
  %4 = fptoui double %storemerge to i32           ; <i32> [#uses=2]
  %5 = fptoui double %storemerge to i64           ; <i64> [#uses=2]
  %6 = fptosi double %storemerge to i64           ; <i64> [#uses=2]
  %7 = tail call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([28 x i8]* @.str, i64 0, i64 0), double %storemerge) nounwind ; <i32> [#uses=0]
  %8 = tail call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([37 x i8]* @.str1, i64 0, i64 0), i64 %6, i64 %6) nounwind ; <i32> [#uses=0]
  %9 = tail call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([35 x i8]* @.str2, i64 0, i64 0), i32 %4, i32 %4) nounwind ; <i32> [#uses=0]
  %10 = tail call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([37 x i8]* @.str3, i64 0, i64 0), i64 %5, i64 %5) nounwind ; <i32> [#uses=0]
  ret i32 0
}

declare double @atof(i8* nocapture) nounwind readonly

declare i32 @printf(i8* nocapture, ...) nounwind
