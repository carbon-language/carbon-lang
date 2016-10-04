// RUN: %clang -v -fxray-instrument -c %s
// XFAIL: armeb, aarch64, aarch64_be, avr, bpfel, bpfeb, hexagon, mips, mipsel, mips64, mips64el, msp430, ppc, ppc64, ppc64le, r600, amdgcn, sparc, sparcv9, sparcel, systemz, tce, thumb, thumbeb, x86-, xcore, nvptx, nvptx64, le32, le64, amdil, amdil64, hsail, hsail64, spir, spir64, kalimba, shave, lanai, wasm32, wasm64, renderscript32, renderscript64
typedef int a;
