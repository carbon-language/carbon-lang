; RUN: llvm-as < %s | opt -instcombine | llvm-dis | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin10.0"

%0 = type { i8*, [19 x i8] }
%1 = type { i8*, [0 x i8] }

@array = external global [11 x i8]

@s = external global %0                           ; <%0*> [#uses=1]
@"\01LC8" = external constant [17 x i8]           ; <[17 x i8]*> [#uses=1]

; Instcombine should be able to fold this getelementptr.

define i32 @main() nounwind {
; CHECK: call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([17 x i8]* @"\01LC8", i32 0, i32 0), i8* undef, i8* bitcast (i8** getelementptr (%1* bitcast (%0* @s to %1*), i32 1, i32 0) to i8*)) nounwind

  call i32 (i8*, ...)* @printf(i8* getelementptr ([17 x i8]* @"\01LC8", i32 0, i32 0), i8* undef, i8* getelementptr (%1* bitcast (%0* @s to %1*), i32 0, i32 1, i32 0)) nounwind ; <i32> [#uses=0]
  ret i32 0
}

; Instcombine should constant-fold the GEP so that indices that have
; static array extents are within bounds of those array extents.
; In the below, -1 is not in the range [0,11). After the transformation,
; the same address is computed, but 3 is in the range of [0,11).

define i8* @foo() nounwind {
; CHECK: ret i8* getelementptr ([11 x i8]* @array, i32 390451572, i32 3)
  ret i8* getelementptr ([11 x i8]* @array, i32 0, i64 -1)
}

declare i32 @printf(i8*, ...) nounwind
