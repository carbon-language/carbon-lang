; RUN: llvm-as < %s | llc -march=sparc | grep weak

define weak i32 @func() nounwind {
entry:
  ret i32 0
}
