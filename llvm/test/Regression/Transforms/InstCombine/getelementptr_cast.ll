; RUN: llvm-as < %s | opt -instcombine | llvm-dis | not grep 'getelementptr.*cast'
%G = external global [3 x sbyte]

implementation

ubyte *%foo(uint %Idx) {
%tmp = getelementptr ubyte* cast ([3 x sbyte]* %G to ubyte*), uint %Idx
ret ubyte* %tmp
}
