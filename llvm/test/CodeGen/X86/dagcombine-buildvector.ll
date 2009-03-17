; RUN: llvm-as < %s | llc -march=x86 -o %t -f
; RUN: grep unpcklpd %t | count 1
; RUN: grep movapd %t | count 1

; Shows a dag combine bug that will generate an illegal build vector
; with v2i64 build_vector i32, i32.

define void @test(<2 x double>* %dst, <4 x double> %src) {
entry:
        %tmp7.i = shufflevector <4 x double> %src, <4 x double> undef, <2 x i32> < i32 0, i32 2 >
        store <2 x double> %tmp7.i, <2 x double>* %dst
        ret void
}
