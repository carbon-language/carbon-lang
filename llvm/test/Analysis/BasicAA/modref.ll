; A very rudimentary test on AliasAnalysis::getModRefInfo.
; RUN: llvm-upgrade < %s | llvm-as | \
; RUN:   opt -print-all-alias-modref-info -aa-eval -disable-output |& \
; RUN:   not grep NoModRef

int %callee() {
  %X = alloca { int, int }
  %Y = getelementptr { int, int }* %X, uint 0, uint 0
  %Z = load int* %Y
  ret int %Z
}

int %caller() {
  %X = call int %callee()
  ret int %X
}
