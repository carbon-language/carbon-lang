// RUN: llvm-mc -arch=amdgcn -mcpu=tahiti %s | FileCheck -check-prefix=SI %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=tonga %s 2>&1 | FileCheck -check-prefix=NOVI --implicit-check-not=error: %s

s_load_dwordx4 s[100:103], s[2:3], s4
// NOVI: error: register not available on this GPU
// SI: s_load_dwordx4 s[100:103], s[2:3], s4

s_load_dwordx8 s[96:103], s[2:3], s4
// NOVI: error: register not available on this GPU
// SI: 	s_load_dwordx8 s[96:103], s[2:3], s4

s_load_dwordx16 s[88:103], s[2:3], s4
// NOVI: error: register not available on this GPU
// SI: s_load_dwordx16 s[88:103], s[2:3], s4
