; RUN: llvm-as < %s | opt -constmerge | llvm-dis | grep -C2 foo  | grep bar

%foo = constant int 6
%bar = constant int 6

implementation

