; RUN: llvm-as < %s | opt -funcresolve -disable-output 2>&1 | grep WARNING

%X = external global {long, int }
%Z = global {long, int} * %X

%X = global float 1.0
%Y = global float* %X

implementation

