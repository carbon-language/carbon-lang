; RUN: llvm-as < %s | llc -march=c

%bob = external global int              ; <int*> [#uses=2]

