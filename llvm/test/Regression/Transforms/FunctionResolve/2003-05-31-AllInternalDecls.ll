; This testcase should not cause a warning!

; RUN: if (as < %s | opt -funcresolve -disable-output) 2>&1 | grep 'WARNING'
; RUN: then exit 1
; RUN: else exit 0
; RUN: fi

%X = internal global float 1.0
%X = internal global int 1

