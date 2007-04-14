; RUN: llvm-upgrade < %s | llvm-as | opt -instcombine | llvm-dis | grep sub
; RUN: llvm-upgrade < %s | llvm-as | opt -instcombine | llvm-dis | grep add

<4 x float> %test(<4 x float> %tmp26, <4 x float> %tmp53) {
  ; (X+Y)-Y != X for fp vectors.
  %tmp64 = add <4 x float> %tmp26, %tmp53
  %tmp75 = sub <4 x float> %tmp64, %tmp53
  ret <4 x float> %tmp75
}
