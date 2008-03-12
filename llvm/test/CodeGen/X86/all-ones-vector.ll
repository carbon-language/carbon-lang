; RUN: llvm-as < %s | llc -march=x86 -mattr=sse2 | grep pcmpeqd | count 4

define <4 x i32> @ioo() {
        ret <4 x i32> <i32 -1, i32 -1, i32 -1, i32 -1>
}
define <2 x i64> @loo() {
        ret <2 x i64> <i64 -1, i64 -1>
}
define <2 x double> @doo() {
        ret <2 x double> <double 0xffffffffffffffff, double 0xffffffffffffffff>
}
define <4 x float> @foo() {
        ret <4 x float> <float 0xffffffffe0000000, float 0xffffffffe0000000, float 0xffffffffe0000000, float 0xffffffffe0000000>
}
