; RUN: llc < %s
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:16-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:16:16-f128:128:128"
target triple = "s390x-ibm-linux-gnu"

@__JCR_LIST__ = internal global [0 x i8*] zeroinitializer, section ".jcr", align 8 ; <[0 x i8*]*> [#uses=1]

define internal void @frame_dummy() nounwind {
entry:
  %asmtmp = tail call void (i8*)* (void (i8*)*)* asm "", "=r,0"(void (i8*)* @_Jv_RegisterClasses) nounwind ; <void (i8*)*> [#uses=2]
  %0 = icmp eq void (i8*)* %asmtmp, null          ; <i1> [#uses=1]
  br i1 %0, label %return, label %bb3

bb3:                                              ; preds = %entry
  tail call void %asmtmp(i8* bitcast ([0 x i8*]* @__JCR_LIST__ to i8*)) nounwind
  ret void

return:                                           ; preds = %entry
  ret void
}

declare extern_weak void @_Jv_RegisterClasses(i8*)
