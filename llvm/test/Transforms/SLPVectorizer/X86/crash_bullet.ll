; RUN: opt < %s -basicaa -slp-vectorizer -dce -S -mtriple=x86_64-apple-macosx10.8.0 -mcpu=corei7

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

%"struct.btTypedConstraint::btConstraintInfo1.17.157.357.417.477.960" = type { i32, i32 }

define void @_ZN23btGeneric6DofConstraint8getInfo1EPN17btTypedConstraint17btConstraintInfo1E(%"struct.btTypedConstraint::btConstraintInfo1.17.157.357.417.477.960"* nocapture %info) {
entry:
  br i1 undef, label %if.else, label %if.then

if.then:                                          ; preds = %entry
  ret void

if.else:                                          ; preds = %entry
  %m_numConstraintRows4 = getelementptr inbounds %"struct.btTypedConstraint::btConstraintInfo1.17.157.357.417.477.960"* %info, i64 0, i32 0
  %nub5 = getelementptr inbounds %"struct.btTypedConstraint::btConstraintInfo1.17.157.357.417.477.960"* %info, i64 0, i32 1
  br i1 undef, label %land.lhs.true.i.1, label %if.then7.1

land.lhs.true.i.1:                                ; preds = %if.else
  br i1 undef, label %for.inc.1, label %if.then7.1

if.then7.1:                                       ; preds = %land.lhs.true.i.1, %if.else
  %inc.1 = add nsw i32 0, 1
  store i32 %inc.1, i32* %m_numConstraintRows4, align 4
  %dec.1 = add nsw i32 6, -1
  store i32 %dec.1, i32* %nub5, align 4
  br label %for.inc.1

for.inc.1:                                        ; preds = %if.then7.1, %land.lhs.true.i.1
  %0 = phi i32 [ %dec.1, %if.then7.1 ], [ 6, %land.lhs.true.i.1 ]
  %1 = phi i32 [ %inc.1, %if.then7.1 ], [ 0, %land.lhs.true.i.1 ]
  %inc.2 = add nsw i32 %1, 1
  store i32 %inc.2, i32* %m_numConstraintRows4, align 4
  %dec.2 = add nsw i32 %0, -1
  store i32 %dec.2, i32* %nub5, align 4
  unreachable
}
