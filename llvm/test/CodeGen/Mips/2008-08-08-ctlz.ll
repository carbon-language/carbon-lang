; RUN: llc -march=mips < %s | FileCheck %s

define i32 @A0(i32 %u) nounwind  {
entry:
; CHECK: clz 
  call i32 @llvm.ctlz.i32( i32 %u )
  ret i32 %0
}

declare i32 @llvm.ctlz.i32(i32) nounwind readnone 
