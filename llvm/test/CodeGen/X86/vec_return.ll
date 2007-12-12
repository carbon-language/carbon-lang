; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2
; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 | grep xorps | count 1
; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 | grep movaps | count 1
; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 | not grep shuf

define <2 x double> @test() {
	ret <2 x double> zeroinitializer
}

define <4 x i32> @test2() nounwind  {
	ret <4 x i32> < i32 0, i32 0, i32 1, i32 0 >
}
