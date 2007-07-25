; This test ensures that we get a bitcast constant expression in and out,
; not a sitofp constant expression. 
; RUN: llvm-upgrade < %s | llvm-as | llvm-dis | \
; RUN:   grep {bitcast (}

%G = external global int

float %tryit(int %A) {
   ret float bitcast( int ptrtoint (int* %G to int) to float)
}
