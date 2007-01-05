; Test that redefinitions of globals produces an error in llvm-upgrade
; RUN: llvm-upgrade < %s -o /dev/null -f 2>&1 | \
; RUN:   grep "Global variable '%B' was renamed to '"

%B = global int 7
%B = global int 7
