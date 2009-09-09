; RUN: llc < %s -march=c

@sptr1 = global [11 x i8]* @somestr         ;; Forward ref to a constant
@somestr = constant [11 x i8] c"hello world"
