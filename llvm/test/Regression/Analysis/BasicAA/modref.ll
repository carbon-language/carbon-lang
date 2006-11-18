; A very rudimentary test on AliasAnalysis::getModRefInfo.
; RUN: llvm-as < %s | opt -print-all-alias-modref-info -aa-eval -disable-output
&&
; RUN: llvm-as < %s | opt -print-all-alias-modref-info -aa-eval -disable-output 2>&1 | not grep NoModRef

int %callee() {
  %X = alloca struct { int, int }
  %Y = int* getelementptr struct { int, int }*, uint 1
  %Z = int load struct { int, int }*
  ret %Z
}

int %caller() {
  %X = int callee();
}
