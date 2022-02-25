# REQUIRES: x86
## The i386 psABI uses Elf64_Rela relocation entries. We produce
## Elf32_Rel dynamic relocations by default, but can use Elf32_Rela with -z rela.

# RUN: llvm-mc -filetype=obj -triple=i386 %s -o %t.o
# RUN: ld.lld -shared %t.o -o %t.so --noinhibit-exec
# RUN: llvm-readobj -d -r -x .data -x .got.plt %t.so | FileCheck --check-prefix=REL %s
# RUN: ld.lld -shared -z rel %t.o -o %t1.so
# RUN: llvm-readobj -d -r -x .data -x .got.plt %t1.so | FileCheck --check-prefix=REL %s

# REL:      REL      {{.*}}
# REL-NEXT: RELSZ    32 (bytes)
# REL-NEXT: RELENT   8 (bytes)
# REL-NEXT: RELCOUNT 1
# REL-NEXT: JMPREL   {{.*}}
# REL-NEXT: PLTRELSZ 8 (bytes)
# REL-NEXT: PLTGOT   {{.*}}
# REL-NEXT: PLTREL   REL{{$}}
# REL:      .rel.dyn {
# REL-NEXT:   R_386_RELATIVE -
# REL-NEXT:   R_386_GLOB_DAT func
# REL-NEXT:   R_386_TLS_TPOFF tls
# REL-NEXT:   R_386_32 _start
# REL-NEXT: }
# REL-NEXT: .rel.plt {
# REL-NEXT:   R_386_JUMP_SLOT func
# REL-NEXT: }

# REL-LABEL: Hex dump of section '.data':
# REL-NEXT:  0x000042d0 d0420000 2a000000
##                       ^--- R_386_RELATIVE addend (0x42d0)
##                                ^--- R_386_32 addend (0x2a)
# REL-LABEL: Hex dump of section '.got.plt':
# REL-NEXT:  0x000042d8 48320000 00000000 00000000 36120000
##  R_386_JUMP_SLOT target (0x1236) ---------------^

# RUN: ld.lld -shared -z rel -z rela %t.o -o %t2.so
# RUN: llvm-readobj -d -r -x .data -x .got.plt %t2.so | FileCheck --check-prefix=RELA %s

# RELA:      RELA      {{.*}}
# RELA-NEXT: RELASZ    48 (bytes)
# RELA-NEXT: RELAENT   12 (bytes)
# RELA-NEXT: RELACOUNT 1
# RELA-NEXT: JMPREL    {{.*}}
# RELA-NEXT: PLTRELSZ  12 (bytes)
# RELA-NEXT: PLTGOT    {{.*}}
# RELA-NEXT: PLTREL    RELA
# RELA:      .rela.dyn {
# RELA-NEXT:   R_386_RELATIVE - 0x42F0
# RELA-NEXT:   R_386_GLOB_DAT func 0x0
# RELA-NEXT:   R_386_TLS_TPOFF tls 0x0
# RELA-NEXT:   R_386_32 _start 0x2A
# RELA-NEXT: }
# RELA-NEXT: .rela.plt {
# RELA-NEXT:   R_386_JUMP_SLOT func 0x0
# RELA-NEXT: }

# RELA-LABEL: Hex dump of section '.data':
# RELA-NEXT:  0x000042f0 00000000 2a000000
##                                ^--- R_386_32 addend (0x2a)
## TODO: we should probably clear the R_386_32 addend that was copied from the .o?
## no addend written for R_386_RELATIVE
# RELA-LABEL: Hex dump of section '.got.plt':
# RELA-NEXT:  0x000042f8 68320000 00000000 00000000 56120000
##  R_386_JUMP_SLOT target (0x1256) ----------------^

.globl _start
_start:
  call func@PLT
  movl func@GOT(%eax), %eax

.section .text1,"awx"
  movl tls@GOTNTPOFF(%eax), %eax

.data
  .long .data
  .long _start+42
