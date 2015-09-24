// RUN: llvm-mc -arch=amdgcn -mcpu=tonga -show-encoding %s | FileCheck -check-prefix=GCN -check-prefix=VI %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=tahiti %s 2>&1 | FileCheck -check-prefix=NOSI %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=bonaire %s 2>&1 | FileCheck -check-prefix=NOSI %s

s_dcache_wb
; VI: s_dcache_wb  ; encoding: [0x00,0x00,0x84,0xc0,0x00,0x00,0x00,0x00]
; NOSI: error: instruction not supported on this GPU

s_dcache_wb_vol
; VI: s_dcache_wb_vol ; encoding: [0x00,0x00,0x8c,0xc0,0x00,0x00,0x00,0x00]
; NOSI: error: instruction not supported on this GPU
