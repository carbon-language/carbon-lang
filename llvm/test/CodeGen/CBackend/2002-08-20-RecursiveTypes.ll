; RUN: llc < %s -march=c

@MyIntList = external global { \2*, i32 }
