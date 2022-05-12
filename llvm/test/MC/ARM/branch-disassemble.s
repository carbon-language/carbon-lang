@ RUN: llvm-mc -mcpu=cortex-a9 -triple armv7-arm-none-eabi -filetype obj -o - %s \
@ RUN:   | llvm-objdump --mcpu=cortex-a9 --triple=armv7-arm-none-eabi -d - \
@ RUN:   | FileCheck %s -check-prefix CHECK-ARM

@ RUN: llvm-mc -mcpu=cortex-m3 -triple thumbv7m-arm-none-eabi -filetype obj -o - %s \
@ RUN:   | llvm-objdump --mcpu=cortex-m3 --triple=thumbv7m-arm-none-eabi -d - \
@ RUN:   | FileCheck %s -check-prefix CHECK-THUMB

b.w .Lbranch
@ CHECK-ARM: b 0xc <$a.0+0xc> @ imm = #4
@ CHECK-THUMB: b.w 0xc <$t.0+0xc> @ imm = #8
adds r0, r1, #42
adds r1, r2, #42
.Lbranch:
movs r2, r3
