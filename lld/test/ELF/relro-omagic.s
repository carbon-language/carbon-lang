# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %p/Inputs/shared.s -o %t2.o
# RUN: ld.lld -shared %t2.o -o %t2.so
# RUN: ld.lld -N %t.o %t2.so -o %t
# RUN: llvm-objdump -section-headers %t | FileCheck --check-prefix=NORELRO %s
# RUN: llvm-readobj --program-headers %t | FileCheck --check-prefix=NOPHDRS %s

# NORELRO:      Sections:
# NORELRO-NEXT: Idx Name          Size      Address          Type
# NORELRO-NEXT:   0               00000000 0000000000000000
# NORELRO-NEXT:   1 .dynsym       00000048 0000000000200120
# NORELRO-NEXT:   2 .hash         00000020 0000000000200168
# NORELRO-NEXT:   3 .dynstr       0000004c 0000000000200188
# NORELRO-NEXT:   4 .rela.dyn     00000018 00000000002001d8
# NORELRO-NEXT:   5 .rela.plt     00000018 00000000002001f0
# NORELRO-NEXT:   6 .text         0000000a 0000000000200208 TEXT DATA
# NORELRO-NEXT:   7 .plt          00000020 0000000000200220 TEXT DATA
# NORELRO-NEXT:   8 .data         00000008 0000000000200240 DATA
# NORELRO-NEXT:   9 .foo          00000004 0000000000200248 DATA
# NORELRO-NEXT:  10 .dynamic      000000f0 0000000000200250
# NORELRO-NEXT:  11 .got          00000008 0000000000200340 DATA
# NORELRO-NEXT:  12 .got.plt      00000020 0000000000200348 DATA
# NORELRO-NEXT:  13 .comment      00000008 0000000000000000
# NORELRO-NEXT:  14 .symtab       00000060 0000000000000000
# NORELRO-NEXT:  15 .shstrtab     0000007b 0000000000000000
# NORELRO-NEXT:  16 .strtab       00000013 0000000000000000

# NOPHDRS:     ProgramHeaders [
# NOPHDRS-NOT: PT_GNU_RELRO

.long bar
jmp *bar2@GOTPCREL(%rip)

.section .data,"aw"
.quad 0

.section .foo,"aw"
.zero 4
