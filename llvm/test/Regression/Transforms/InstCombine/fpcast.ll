; Test some floating point casting cases
; RUN: llvm-upgrade %s -o - | llvm-as | opt -instcombine | llvm-dis | notcast
; RUN: llvm-upgrade %s -o - | llvm-as | opt -instcombine | llvm-dis | \
; RUN:   grep 'ret [us]byte \(-1\)\|\(255\)'

sbyte %test() {
    %x = fptoui float 255.0 to sbyte 
    ret sbyte %x
}

ubyte %test() {
    %x = fptosi float -1.0 to ubyte
    ret ubyte %x
}
