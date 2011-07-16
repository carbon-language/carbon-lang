//===-- HasAVX.s ---------------------------------------*- x86 Assembly -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#if defined (__i386__) || defined (__x86_64__)

.globl _HasAVX

_HasAVX:
    mov $1, %eax
    cpuid
    and $0x018000000, %ecx
    cmp $0x018000000, %ecx
    jne not_supported
    mov $0, %ecx
.byte 0x0f, 0x01, 0xd0    // xgetbv, for those assemblers that don't know it
    and $0x06, %eax
    cmp $0x06, %eax
    jne not_supported
    mov $1, %eax
    jmp done
not_supported:
    mov $0, %eax
done:
    ret

#endif