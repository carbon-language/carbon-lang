; RUN: llvm-upgrade < %s | llvm-as | llc -march=c

%bob = external global int              ; <int*> [#uses=2]

