# Instructions that are valid for the current ISA but should be rejected by the assembler (e.g.
# invalid set of operands or operand's restrictions not met).

# RUN: not llvm-mc %s -triple=mips-unknown-linux -mcpu=mips32r3 2>%t1
# RUN: FileCheck %s < %t1

        .text
        .set noreorder
        cache -1, 255($7)    # CHECK: :[[@LINE]]:15: error: expected 5-bit unsigned immediate
        cache 32, 255($7)    # CHECK: :[[@LINE]]:15: error: expected 5-bit unsigned immediate
        jalr.hb $31          # CHECK: :[[@LINE]]:9: error: source and destination must be different
        jalr.hb $31, $31     # CHECK: :[[@LINE]]:9: error: source and destination must be different
        pref -1, 255($7)     # CHECK: :[[@LINE]]:14: error: expected 5-bit unsigned immediate
        pref 32, 255($7)     # CHECK: :[[@LINE]]:14: error: expected 5-bit unsigned immediate
