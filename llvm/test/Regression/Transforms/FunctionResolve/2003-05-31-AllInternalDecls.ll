; This testcase should not cause a warning!

; RUN: (as < %s | opt -funcresolve -disable-output) 2>&1 | not grep 'WARNING'

%X = internal global float 1.0
%X = internal global int 1

