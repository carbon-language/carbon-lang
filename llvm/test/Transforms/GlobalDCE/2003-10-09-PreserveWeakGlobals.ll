; Weak variables should be preserved by global DCE!

; RUN: opt %s -globaldce | llvm-dis | grep @A


@A = weak global i32 54
