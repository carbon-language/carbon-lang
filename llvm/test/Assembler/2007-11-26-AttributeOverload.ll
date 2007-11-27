; RUN: not llvm-as < %s

declare i32 @atoi(i8*) nounwind readonly
declare i32 @atoi(i8*)
