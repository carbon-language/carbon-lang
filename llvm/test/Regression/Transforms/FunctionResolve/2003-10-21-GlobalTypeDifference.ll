; RUN: llvm-as < %s | opt -funcresolve -disable-output 2>&1 | grep WARNING

%X = external global int
%Z = global int* %X

%X = global float 1.0
%Y = global float* %X

implementation

