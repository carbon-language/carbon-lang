
; RUN: llvm-as < %s | opt -scalarrepl | llvm-dis | not grep alloca &&
; RUN: llvm-as < %s | opt -scalarrepl | llvm-dis | grep 'ret sbyte'

target endian = little
target pointersize = 32
target triple = "i686-apple-darwin8.7.2"

implementation   ; Functions:

sbyte* %test(short* %X) {
        %X_addr = alloca short*
        store short* %X, short** %X_addr
        %X_addr = cast short** %X_addr to sbyte**
        %tmp = load sbyte** %X_addr
        ret sbyte* %tmp
}

