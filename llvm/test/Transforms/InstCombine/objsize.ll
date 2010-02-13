; Test a pile of objectsize bounds checking.
; RUN: opt < %s -instcombine -S | FileCheck %s
; We need target data to get the sizes of the arrays and structures.
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"

@a = private global [60 x i8] zeroinitializer, align 1 ; <[60 x i8]*>
@.str = private constant [8 x i8] c"abcdefg\00"   ; <[8 x i8]*>

define i32 @foo() nounwind {
; CHECK: @foo
; CHECK-NEXT: ret i32 60
  %1 = call i32 @llvm.objectsize.i32(i8* getelementptr inbounds ([60 x i8]* @a, i32 0, i32 0), i1 false)
  ret i32 %1
}

define i8* @bar() nounwind {
; CHECK: @bar
entry:
  %retval = alloca i8*
  %0 = call i32 @llvm.objectsize.i32(i8* getelementptr inbounds ([60 x i8]* @a, i32 0, i32 0), i1 false)
  %cmp = icmp ne i32 %0, -1
; CHECK: br i1 true
  br i1 %cmp, label %cond.true, label %cond.false

cond.true:
  %1 = load i8** %retval;
  ret i8* %1;

cond.false:
  %2 = load i8** %retval;
  ret i8* %2;
}

define i32 @f() nounwind {
; CHECK: @f
; CHECK-NEXT: ret i32 0
  %1 = call i32 @llvm.objectsize.i32(i8* getelementptr ([60 x i8]* @a, i32 1, i32 0), i1 false)
  ret i32 %1
}

@window = external global [0 x i8]

define i1 @baz() nounwind {
; CHECK: @baz
; CHECK-NEXT: ret i1 true
  %1 = tail call i32 @llvm.objectsize.i32(i8* getelementptr inbounds ([0 x i8]* @window, i32 0, i32 0), i1 false)
  %2 = icmp eq i32 %1, -1
  ret i1 %2
}

define void @test1(i8* %q, i32 %x) nounwind noinline {
; CHECK: @test1
; CHECK: objectsize.i32
entry:
  %0 = call i32 @llvm.objectsize.i32(i8* getelementptr inbounds ([0 x i8]* @window, i32 0, i32 10), i1 false) ; <i64> [#uses=1]
  %1 = icmp eq i32 %0, -1                         ; <i1> [#uses=1]
  br i1 %1, label %"47", label %"46"

"46":                                             ; preds = %entry
  unreachable

"47":                                             ; preds = %entry
  unreachable
}

@.str5 = private constant [9 x i32] [i32 97, i32 98, i32 99, i32 100, i32 0, i32
 101, i32 102, i32 103, i32 0], align 4 ;
define i32 @test2() nounwind {
; CHECK: @test2
; CHECK-NEXT: ret i32 34
  %1 = call i32 @llvm.objectsize.i32(i8* getelementptr (i8* bitcast ([9 x i32]* @.str5 to i8*), i32 2), i1 false)
  ret i32 %1
}

declare i32 @llvm.objectsize.i32(i8*, i1) nounwind readonly