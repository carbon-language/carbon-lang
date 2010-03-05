; RUN: opt < %s -simplify-libcalls -S | FileCheck %s
; rdar://7719085

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"

%struct.data = type { [100 x i32], [100 x i32], [1024 x i8] }

define i32 @t() nounwind ssp {
; CHECK: @t
; CHECK: @llvm.memset.i64
entry:
  %0 = alloca %struct.data, align 8               ; <%struct.data*> [#uses=1]
  %1 = bitcast %struct.data* %0 to i8*            ; <i8*> [#uses=1]
  %2 = call i8* @__memset_chk(i8* %1, i32 0, i64 1824, i64 1824) nounwind ; <i8*> [#uses=0]
  ret i32 0
}

declare i8* @__memset_chk(i8*, i32, i64, i64) nounwind
