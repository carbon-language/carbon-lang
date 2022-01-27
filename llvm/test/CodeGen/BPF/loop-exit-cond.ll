; RUN: opt -O2 -S -o %t1 < %s
; RUN: llc -march=bpf -mcpu=v3 %t1 -o - | FileCheck %s
;
; Source code:
;   typedef unsigned long u64;
;   void foo(char *data, int idx, u64 *);
;   int test(int len, char *data) {
;     if (len < 100) {
;       for (int i = 1; i < len; i++) {
;         u64 d[1];
;         d[0] = data[0] ?: '0';
;         foo("%c", i, d);
;       }
;     }
;   return 0;
; }
; Compilation flag:
;   clang -target bpf -O2 -S -emit-llvm -Xclang -disable-llvm-passes test.c

; ModuleID = 'test.c'
source_filename = "test.c"
target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128"
target triple = "bpf"

@.str = private unnamed_addr constant [3 x i8] c"%c\00", align 1

; Function Attrs: nounwind
define dso_local i32 @test(i32 %len, i8* %data) #0 {
entry:
  %len.addr = alloca i32, align 4
  %data.addr = alloca i8*, align 8
  %i = alloca i32, align 4
  %d = alloca [1 x i64], align 8
  store i32 %len, i32* %len.addr, align 4, !tbaa !3
  store i8* %data, i8** %data.addr, align 8, !tbaa !7
  %0 = load i32, i32* %len.addr, align 4, !tbaa !3
  %cmp = icmp slt i32 %0, 100
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %1 = bitcast i32* %i to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %1) #3
  store i32 1, i32* %i, align 4, !tbaa !3
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %if.then
  %2 = load i32, i32* %i, align 4, !tbaa !3
  %3 = load i32, i32* %len.addr, align 4, !tbaa !3
  %cmp1 = icmp slt i32 %2, %3
  br i1 %cmp1, label %for.body, label %for.cond.cleanup

; CHECK:      w[[LEN:[0-9]+]] = w1
; CHECK:      w[[IDX:[0-9]+]] += 1
; CHECK-NEXT: w[[IDX]] s< w[[LEN]] goto

for.cond.cleanup:                                 ; preds = %for.cond
  %4 = bitcast i32* %i to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %4) #3
  br label %for.end

for.body:                                         ; preds = %for.cond
  %5 = bitcast [1 x i64]* %d to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* %5) #3
  %6 = load i8*, i8** %data.addr, align 8, !tbaa !7
  %arrayidx = getelementptr inbounds i8, i8* %6, i64 0
  %7 = load i8, i8* %arrayidx, align 1, !tbaa !9
  %conv = sext i8 %7 to i32
  %tobool = icmp ne i32 %conv, 0
  br i1 %tobool, label %cond.true, label %cond.false

cond.true:                                        ; preds = %for.body
  br label %cond.end

cond.false:                                       ; preds = %for.body
  br label %cond.end

cond.end:                                         ; preds = %cond.false, %cond.true
  %cond = phi i32 [ %conv, %cond.true ], [ 48, %cond.false ]
  %conv2 = sext i32 %cond to i64
  %arrayidx3 = getelementptr inbounds [1 x i64], [1 x i64]* %d, i64 0, i64 0
  store i64 %conv2, i64* %arrayidx3, align 8, !tbaa !10
  %8 = load i32, i32* %i, align 4, !tbaa !3
  %arraydecay = getelementptr inbounds [1 x i64], [1 x i64]* %d, i64 0, i64 0
  call void @foo(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str, i64 0, i64 0), i32 %8, i64* %arraydecay)
  %9 = bitcast [1 x i64]* %d to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* %9) #3
  br label %for.inc

for.inc:                                          ; preds = %cond.end
  %10 = load i32, i32* %i, align 4, !tbaa !3
  %inc = add nsw i32 %10, 1
  store i32 %inc, i32* %i, align 4, !tbaa !3
  br label %for.cond, !llvm.loop !12

for.end:                                          ; preds = %for.cond.cleanup
  br label %if.end

if.end:                                           ; preds = %for.end, %entry
  ret i32 0
}

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #1

declare dso_local void @foo(i8*, i32, i64*) #2

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #1

attributes #0 = { nounwind "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { argmemonly nofree nosync nounwind willreturn }
attributes #2 = { "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #3 = { nounwind }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"frame-pointer", i32 2}
!2 = !{!"clang version 14.0.0 (https://github.com/llvm/llvm-project.git 8385de118443144518c9fba8b3d831d9076e746b)"}
!3 = !{!4, !4, i64 0}
!4 = !{!"int", !5, i64 0}
!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C/C++ TBAA"}
!7 = !{!8, !8, i64 0}
!8 = !{!"any pointer", !5, i64 0}
!9 = !{!5, !5, i64 0}
!10 = !{!11, !11, i64 0}
!11 = !{!"long", !5, i64 0}
!12 = distinct !{!12, !13}
!13 = !{!"llvm.loop.mustprogress"}
