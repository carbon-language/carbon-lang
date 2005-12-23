; RUN: llvm-as %s -o /dev/null 2>&1 | grep "constant invalid for type"
; XFAIL: *

;; This is a testcase for PR409

; make sure that 'float' values are in range

%D1 = constant double 3.40282347e+39
%D2 = constant double -3.40282347e+39
%F1 = constant float 3.40282346e+38
%F2 = constant float -3.40282346e+38
%D1 = constant float 3.40282347e+39
%D2 = constant float -3.40282347e+39
