; Weak variables should be preserved by global DCE!

; RUN: llvm-upgrade < %s | llvm-as | opt -globaldce | llvm-dis | grep @A


%A = weak global int 54
