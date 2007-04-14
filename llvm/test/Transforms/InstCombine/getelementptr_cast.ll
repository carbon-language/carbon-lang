; RUN: llvm-upgrade < %s | llvm-as | opt -instcombine | llvm-dis | \
; RUN:    notcast {} {getelementptr.*}

%G = external global [3 x sbyte]

implementation

ubyte *%foo(uint %Idx) {
    %tmp = getelementptr ubyte* cast ([3 x sbyte]* %G to ubyte*), uint %Idx
    ret ubyte* %tmp
}
