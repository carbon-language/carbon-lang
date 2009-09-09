; RUN: llc < %s -march=pic16 | FileCheck %s

@G1 = global i32 4712, section "Address=412"
; CHECK: @G1.412.idata.0.# IDATA 412
; CHECK: @G1 dl 4712

@G2 = global i32 0, section "Address=412"
; CHECK: @G2.412.udata.0.# UDATA 412
; CHECK: @G2 RES 4

@G3 = addrspace(1) constant i32 4712, section "Address=412"
; CHECK: @G3.412.romdata.1.# ROMDATA 412
; CHECK: @G3 rom_dl 4712


