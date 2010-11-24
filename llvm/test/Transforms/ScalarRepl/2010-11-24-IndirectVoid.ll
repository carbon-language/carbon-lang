; RUN: opt -scalarrepl -disable-output %s
; PR8680

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"

define void @main() nounwind {
entry:
  %memtmp = alloca i32, align 4
  %0 = bitcast i32* %memtmp to void ()*
  call void %0() nounwind
  ret void
}

