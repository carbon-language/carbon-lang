; RUN: llvm-as %s -f -o %s.bc ; llvm-link -f -o %s.linked %s.bc

target endian = big
target pointersize = 32
target triple = "powerpc-apple-darwin7.7.0"
deplibs = [ "c", "crtend" ]
%source = global <4 x int> < int 0, int 1, int 2, int 3 >

implementation   ; Functions:

int %main() {
entry:
        ret int 0
}
