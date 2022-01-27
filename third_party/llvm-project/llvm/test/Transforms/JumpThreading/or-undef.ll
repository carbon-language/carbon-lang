; RUN: opt -jump-threading -S < %s | FileCheck %s
; rdar://7620633

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin11.0"

define void @test1(i8* %args, i32 %from_tty) nounwind optsize ssp {
entry:
  %tmp = call i8* @f3(void (i8*)* null, i8* null) nounwind ; <i8*> [#uses=1]
  %tmp1 = icmp eq i8* %args, null                 ; <i1> [#uses=1]
  br i1 %tmp1, label %bb2, label %bb

; CHECK: entry:
; CHECK-NEXT: %tmp = call i8* @f3
; CHECK-NEXT: %tmp1 = icmp eq i8* %args, null
; CHECK-NEXT: br i1 %tmp1, label %bb7, label %bb

bb:                                               ; preds = %entry
  %tmp2 = call noalias i8** @buildargv(i8* %args) nounwind ; <i8**> [#uses=4]
  %tmp3 = icmp eq i8** %tmp2, null                ; <i1> [#uses=1]
  br i1 %tmp3, label %bb2, label %bb1

bb1:                                              ; preds = %bb
  call void @f2(i8** %tmp2) nounwind
  br label %bb2

bb2:                                              ; preds = %bb1, %bb, %entry
  %argv.0 = phi i8** [ %tmp2, %bb1 ], [ %tmp2, %bb ], [ undef, %entry ] ; <i8**> [#uses=4]
  %tmp5 = icmp eq i8* %args, null                 ; <i1> [#uses=1]
  %tmp6 = icmp eq i8** %argv.0, null              ; <i1> [#uses=1]
  %tmp7 = or i1 %tmp5, %tmp6                      ; <i1> [#uses=1]
  br i1 %tmp7, label %bb7, label %bb5

bb5:                                              ; preds = %bb2
  %tmp8 = load i8*, i8** %argv.0, align 8              ; <i8*> [#uses=1]
  %tmp9 = icmp eq i8* %tmp8, null                 ; <i1> [#uses=1]
  br i1 %tmp9, label %bb7, label %bb6

bb6:                                              ; preds = %bb5
  %tmp10 = load i8*, i8** %argv.0, align 8             ; <i8*> [#uses=1]
  %tmp11 = load i8, i8* %tmp10, align 1               ; <i8> [#uses=1]
  %tmp12 = icmp eq i8 %tmp11, 0                   ; <i1> [#uses=1]
  br i1 %tmp12, label %bb7, label %bb8

bb7:                                              ; preds = %bb6, %bb5, %bb2
  call void @f1() nounwind optsize ssp
  br label %bb9

bb8:                                              ; preds = %bb6
  %tmp13 = load i8*, i8** %argv.0, align 8             ; <i8*> [#uses=1]
  %tmp14 = call i64 @f5(i8* %tmp13) nounwind      ; <i64> [#uses=0]
  br label %bb9

bb9:                                              ; preds = %bb8, %bb7
  call void @f4(i8* %tmp) nounwind
  ret void
}

declare noalias i8** @buildargv(i8*)

declare void @f2(i8**)

declare void @f4(i8*)

declare i8* @f3(void (i8*)*, i8*)

declare void @f1()

declare i64 @f5(i8*)
