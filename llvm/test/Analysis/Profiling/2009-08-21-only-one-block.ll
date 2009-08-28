; RUN: llvm-as < %s | opt -insert-edge-profiling > %t1
; RUN: lli -load %llvmlibsdir/profile_rt%shlibext %t1
; RUN: mv llvmprof.out %t2
; RUN: llvm-prof -print-all-code %t1 %t2 | tee %t3 | FileCheck %s
; CHECK:  1.     1/1 main
; CHECK:  1.   100%     1/1	main() - entry
; ModuleID = '<stdin>'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK:;;; %main called 1 times.
; CHECK:;;;
define i32 @main() nounwind readnone {
entry:
; CHECK:entry:
; CHECK:	;;; Basic block executed 1 times.
  ret i32 undef
}
