; This testcase is primarily used for testing that global values can be used as 
; constant pointer initializers.  This is tricky because they can be forward
; declared and involves an icky bytecode encoding.  There is no meaningful 
; optimization that can be performed on this file, it is just here to test 
; assembly and disassembly.
;


%t3 = global int * %t1           ;; Forward reference
%t1 = global int 4
%t2 = global int * %t1

global float * %0                ;; Forward numeric reference
global float * %0                ;; Duplicate forward numeric reference
global float 0.0
global float * %0                ;; Numeric reference


%fptr = global void() * %f       ;; Forward ref method defn
declare void "f"()               ;; External method

implementation

