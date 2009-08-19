; RUN: llvm-as < %s | opt -instcombine
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin10.0"

%0 = type { i8*, [19 x i8] }
%1 = type { i8*, [0 x i8] }

@s = external global %0                           ; <%0*> [#uses=1]
@"\01LC8" = external constant [17 x i8]           ; <[17 x i8]*> [#uses=1]

define i32 @main() nounwind {
entry:
  %0 = call i32 (i8*, ...)* @printf(i8* getelementptr ([17 x i8]* @"\01LC8", i32 0, i32 0), i8* undef, i8* getelementptr (%1* bitcast (%0* @s to %1*), i32 0, i32 1, i32 0)) nounwind ; <i32> [#uses=0]
  ret i32 0
}

declare i32 @printf(i8*, ...) nounwind
