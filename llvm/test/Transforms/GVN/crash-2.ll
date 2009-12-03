; RUN: opt -gvn -S %s
; XFAIL: *
; rdar://7438974

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin9.0"

@g = external global i64, align 8

define i32* @foo() {
do.end17.i:
  %tmp18.i = load i7** undef
  %tmp1 = bitcast i7* %tmp18.i to i8*
  br i1 undef, label %do.body36.i, label %if.then21.i

if.then21.i:
  %tmp2 = bitcast i7* %tmp18.i to i8*
  ret i32* undef

do.body36.i:
  %ivar38.i = load i64* @g 
  %tmp3 = bitcast i7* %tmp18.i to i8*
  %add.ptr39.sum.i = add i64 %ivar38.i, 8
  %tmp40.i = getelementptr inbounds i8* %tmp3, i64 %add.ptr39.sum.i
  %tmp4 = bitcast i8* %tmp40.i to i64*
  %tmp41.i = load i64* %tmp4
  br i1 undef, label %if.then48.i, label %do.body57.i

if.then48.i:
  %call54.i = call i32 @foo2()
  br label %do.body57.i

do.body57.i:
  %tmp58.i = load i7** undef
  %ivar59.i = load i64* @g
  %tmp5 = bitcast i7* %tmp58.i to i8*
  %add.ptr65.sum.i = add i64 %ivar59.i, 8
  %tmp66.i = getelementptr inbounds i8* %tmp5, i64 %add.ptr65.sum.i
  %tmp6 = bitcast i8* %tmp66.i to i64*
  %tmp67.i = load i64* %tmp6
  ret i32* undef
}

declare i32 @foo2()
