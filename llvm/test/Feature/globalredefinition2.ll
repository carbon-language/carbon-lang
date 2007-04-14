; Test that redefinitions of globals produces an error in llvm-upgrade
; RUN: llvm-upgrade < %s -o /dev/null -f |& \
; RUN:   grep "Renaming global variable 'B' to.*linkage errors"

%B = global int 7
%B = global int 7
