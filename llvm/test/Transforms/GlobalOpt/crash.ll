; RUN: opt -globalopt -disable-output %s
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128-n8:16:32"
target triple = "i386-apple-darwin9.8"

%0 = type { i32, void ()* }
%struct.btSimdScalar = type { %"union.btSimdScalar::$_14" }
%"union.btSimdScalar::$_14" = type { <4 x float> }

@_ZL6vTwist =  global %struct.btSimdScalar zeroinitializer ; <%struct.btSimdScalar*> [#uses=1]
@llvm.global_ctors = appending global [1 x %0] [%0 { i32 65535, void ()* @_GLOBAL__I__ZN21btConeTwistConstraintC2Ev }] ; <[12 x %0]*> [#uses=0]

define internal  void @_GLOBAL__I__ZN21btConeTwistConstraintC2Ev() nounwind section "__TEXT,__StaticInit,regular,pure_instructions" {
entry:
  store float 1.0, float* getelementptr inbounds (%struct.btSimdScalar* @_ZL6vTwist, i32 0, i32 0, i32 0, i32 3), align 4
  ret void
}
