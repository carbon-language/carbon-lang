; RUN: opt < %s -basicaa -slp-vectorizer -dce -S -mtriple=x86_64-apple-macosx10.8.0 -mcpu=corei7

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

%struct.GPar.0.16.26 = type { [0 x double], double }

@d = external global double, align 8

declare %struct.GPar.0.16.26* @Rf_gpptr(...)

define void @Rf_GReset() {
entry:
  %sub = fsub double -0.000000e+00, undef
  %0 = load double, double* @d, align 8
  %sub1 = fsub double -0.000000e+00, %0
  br i1 icmp eq (%struct.GPar.0.16.26* (...)* inttoptr (i64 115 to %struct.GPar.0.16.26* (...)*), %struct.GPar.0.16.26* (...)* @Rf_gpptr), label %if.then, label %if.end7

if.then:                                          ; preds = %entry
  %sub2 = fsub double %sub, undef
  %div.i = fdiv double %sub2, undef
  %sub4 = fsub double %sub1, undef
  %div.i16 = fdiv double %sub4, undef
  %cmp = fcmp ogt double %div.i, %div.i16
  br i1 %cmp, label %if.then6, label %if.end7

if.then6:                                         ; preds = %if.then
  br label %if.end7

if.end7:                                          ; preds = %if.then6, %if.then, %entry
  %g.0 = phi double [ 0.000000e+00, %if.then6 ], [ %sub, %if.then ], [ %sub, %entry ]
  ret void
}


