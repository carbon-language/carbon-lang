; RUN: llvm-dis < %s.bc | not grep {i32 @llvm\\.pmulld}
; RUN: llvm-dis < %s.bc | grep mul
