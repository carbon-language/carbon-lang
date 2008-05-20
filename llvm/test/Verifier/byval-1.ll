; RUN: not llvm-as %s -o /dev/null -f
declare void @h(i32 byval %num)
