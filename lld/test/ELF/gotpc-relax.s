# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -relax-relocations -triple=x86_64-unknown-linux %s -o %t.o
# RUN: ld.lld %t.o -o %t1
# RUN: llvm-readobj -r %t1 | FileCheck --check-prefix=RELOC %s
# RUN: llvm-objdump -d %t1 | FileCheck --check-prefix=DISASM %s

## There is no relocations.
# RELOC:    Relocations [
# RELOC:    ]

# 0x11003 + 7 - 10 = 0x11000
# 0x1100a + 7 - 17 = 0x11000
# 0x11011 + 7 - 23 = 0x11001
# 0x11018 + 7 - 30 = 0x11001
# DISASM:      Disassembly of section .text:
# DISASM-NEXT: foo:
# DISASM-NEXT:   11000: 90 nop
# DISASM:      hid:
# DISASM-NEXT:   11001: 90 nop
# DISASM:      ifunc:
# DISASM-NEXT:   11002: c3 retq
# DISASM:      _start:
# DISASM-NEXT: 11003: 48 8d 05 f6 ff ff ff leaq -10(%rip), %rax
# DISASM-NEXT: 1100a: 48 8d 05 ef ff ff ff leaq -17(%rip), %rax
# DISASM-NEXT: 11011: 48 8d 05 e9 ff ff ff leaq -23(%rip), %rax
# DISASM-NEXT: 11018: 48 8d 05 e2 ff ff ff leaq -30(%rip), %rax
# DISASM-NEXT: 1101f: 48 8b 05 da 0f 00 00 movq 4058(%rip), %rax
# DISASM-NEXT: 11026: 48 8b 05 d3 0f 00 00 movq 4051(%rip), %rax
# DISASM-NEXT: 1102d: 8d 05 cd ff ff ff    leal -51(%rip), %eax
# DISASM-NEXT: 11033: 8d 05 c7 ff ff ff    leal -57(%rip), %eax
# DISASM-NEXT: 11039: 8d 05 c2 ff ff ff    leal -62(%rip), %eax
# DISASM-NEXT: 1103f: 8d 05 bc ff ff ff    leal -68(%rip), %eax
# DISASM-NEXT: 11045: 8b 05 b5 0f 00 00    movl 4021(%rip), %eax
# DISASM-NEXT: 1104b: 8b 05 af 0f 00 00    movl 4015(%rip), %eax

.text
.globl foo
.type foo, @function
foo:
 nop

.globl hid
.hidden hid
.type hid, @function
hid:
 nop

.text
.type ifunc STT_GNU_IFUNC
.globl ifunc
.type ifunc, @function
ifunc:
 ret

.globl _start
.type _start, @function
_start:
 movq foo@GOTPCREL(%rip), %rax
 movq foo@GOTPCREL(%rip), %rax
 movq hid@GOTPCREL(%rip), %rax
 movq hid@GOTPCREL(%rip), %rax
 movq ifunc@GOTPCREL(%rip), %rax
 movq ifunc@GOTPCREL(%rip), %rax
 movl foo@GOTPCREL(%rip), %eax
 movl foo@GOTPCREL(%rip), %eax
 movl hid@GOTPCREL(%rip), %eax
 movl hid@GOTPCREL(%rip), %eax
 movl ifunc@GOTPCREL(%rip), %eax
 movl ifunc@GOTPCREL(%rip), %eax
