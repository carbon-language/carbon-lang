; This test makes sure that gep(gep ...) merge doesn't come into effect.
; RUN: opt < %s -instcombine -S | FileCheck %s

; Make sure there are no geps being merged.
; CHECK-LABEL: @fn3(
; CHECK: getelementptr
; CHECK: getelementptr
; CHECK: getelementptr

@_ZN2cv1aE = global i8* zeroinitializer, align 8
declare i32 @fn1() #2
declare i32 @fn2() #2

; Function Attrs: uwtable
define linkonce_odr i32 @fn3() {
entry:
  %call = call i32 @fn1()
  %call1 = call i32 @fn2()
  %0 = load i8*, i8** @_ZN2cv1aE, align 8
  %idx.ext2 = sext i32 %call1 to i64
  %add.ptr3 = getelementptr inbounds i8, i8* %0, i64 %idx.ext2
  br label %for.cond5

for.cond5:
  %total1 = phi i32 [ 0, %entry ], [ %total2, %for.body7 ]
  %x.1 = phi i32 [ 0, %entry ], [ %inc, %for.body7 ]
  %cmp6 = icmp slt i32 %x.1, %call
  br i1 %cmp6, label %for.body7, label %for.cond34

for.body7:                                        ; preds = %for.cond5
  %mul = mul nsw i32 %x.1, 2
  %idxprom = sext i32 %mul to i64
  %arrayidx = getelementptr inbounds i8, i8* %add.ptr3, i64 %idxprom
  %1 = load i8, i8* %arrayidx, align 1
  %conv = zext i8 %1 to i32
  %sub = sub nsw i32 %mul, 1
  %idxprom10 = sext i32 %sub to i64
  %arrayidx11 = getelementptr inbounds i8, i8* %add.ptr3, i64 %idxprom10
  %2 = load i8, i8* %arrayidx11, align 1
  %conv2 = zext i8 %2 to i32
  %add1 = add nsw i32 %conv, %conv2
  %total2 = add nsw i32 %total1, %add1
  %inc = add nsw i32 %x.1, 1
  br label %for.cond5

for.cond34:
  ret i32 %total1
}
