global int* cast (float* %0 to int*)   ;; Forward numeric reference
global float* %0                       ;; Duplicate forward numeric reference
global float 0.0

%array  = constant [2 x int] [ int 12, int 52 ]
%arrayPtr = global int* getelementptr ([2 x int]* %array, long 0, long 0)    ;; int* &%array[0][0]

