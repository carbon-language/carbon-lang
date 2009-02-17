; RUN: llvm-as < %s | opt -globaldce

@A = alias internal void ()* @F
define internal void @F() { ret void }
