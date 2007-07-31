; RUN: llvm-as < %s | llc -march=x86-64 | grep {shufps	\$3, %xmm0, %xmm0}

define float @foo(<8 x float> %a) {
  %c = extractelement <8 x float> %a, i32 3
  ret float %c
}
