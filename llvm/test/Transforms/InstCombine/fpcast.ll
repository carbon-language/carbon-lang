; Test some floating point casting cases
; RUN: llvm-upgrade %s -o - | llvm-as | opt -instcombine | llvm-dis | notcast
; RUN: llvm-upgrade %s -o - | llvm-as | opt -instcombine | llvm-dis | \
; RUN:   egrep {ret i8 \(-1\)\|\(255\)}

sbyte %test1() {
    %x = fptoui float 255.0 to sbyte 
    ret sbyte %x
}

ubyte %test2() {
    %x = fptosi float -1.0 to ubyte
    ret ubyte %x
}
