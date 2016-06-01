# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -relax-relocations -triple=x86_64-unknown-linux %s -o %t.o
# RUN: ld.lld %t.o -o %t1
# RUN: llvm-readobj -symbols -r %t1 | FileCheck --check-prefix=SYMRELOC %s
# RUN: llvm-objdump -d %t1 | FileCheck --check-prefix=DISASM %s

## There is no relocations.
# SYMRELOC:      Relocations [
# SYMRELOC-NEXT: ]
# SYMRELOC:      Symbols [
# SYMRELOC:       Symbol {
# SYMRELOC:        Name: bar
# SYMRELOC-NEXT:   Value: 0x13000

## 77824 = 0x13000 (bar)
## Notice, that 32bit versions of operations are not relaxed.
# DISASM:      Disassembly of section .text:
# DISASM-NEXT: _start:
# DISASM-NEXT:    11000:  13 05 fa 0f 00 00     adcl  4090(%rip), %eax
# DISASM-NEXT:    11006:  03 1d f4 0f 00 00     addl  4084(%rip), %ebx
# DISASM-NEXT:    1100c:  23 0d ee 0f 00 00     andl  4078(%rip), %ecx
# DISASM-NEXT:    11012:  3b 15 e8 0f 00 00     cmpl  4072(%rip), %edx
# DISASM-NEXT:    11018:  0b 35 e2 0f 00 00     orl   4066(%rip), %esi
# DISASM-NEXT:    1101e:  1b 3d dc 0f 00 00     sbbl  4060(%rip), %edi
# DISASM-NEXT:    11024:  2b 2d d6 0f 00 00     subl  4054(%rip), %ebp
# DISASM-NEXT:    1102a:  44 33 05 cf 0f 00 00  xorl  4047(%rip), %r8d
# DISASM-NEXT:    11031:  44 85 3d c8 0f 00 00  testl 4040(%rip), %r15d
# DISASM-NEXT:    11038:  48 81 d0 00 30 01 00  adcq  $77824, %rax
# DISASM-NEXT:    1103f:  48 81 c3 00 30 01 00  addq  $77824, %rbx
# DISASM-NEXT:    11046:  48 81 e1 00 30 01 00  andq  $77824, %rcx
# DISASM-NEXT:    1104d:  48 81 fa 00 30 01 00  cmpq  $77824, %rdx
# DISASM-NEXT:    11054:  48 81 cf 00 30 01 00  orq   $77824, %rdi
# DISASM-NEXT:    1105b:  48 81 de 00 30 01 00  sbbq  $77824, %rsi
# DISASM-NEXT:    11062:  48 81 ed 00 30 01 00  subq  $77824, %rbp
# DISASM-NEXT:    11069:  49 81 f0 00 30 01 00  xorq  $77824, %r8
# DISASM-NEXT:    11070:  49 f7 c7 00 30 01 00  testq $77824, %r15

# RUN: ld.lld -shared %t.o -o %t2
# RUN: llvm-readobj -s %t2 | FileCheck --check-prefix=SEC-PIC    %s
# RUN: llvm-objdump -d %t2 | FileCheck --check-prefix=DISASM-PIC %s
# SEC-PIC:      Section {
# SEC-PIC:        Index:
# SEC-PIC:        Name: .got
# SEC-PIC-NEXT:   Type: SHT_PROGBITS
# SEC-PIC-NEXT:   Flags [
# SEC-PIC-NEXT:     SHF_ALLOC
# SEC-PIC-NEXT:     SHF_WRITE
# SEC-PIC-NEXT:   ]
# SEC-PIC-NEXT:   Address: 0x2090
# SEC-PIC-NEXT:   Offset: 0x2090
# SEC-PIC-NEXT:   Size: 8
# SEC-PIC-NEXT:   Link:
# SEC-PIC-NEXT:   Info:
# SEC-PIC-NEXT:   AddressAlignment:
# SEC-PIC-NEXT:   EntrySize:
# SEC-PIC-NEXT: }

## Check that there was no relaxation performed. All values refer to got entry.
## Ex: 0x1000 + 4234 + 6 = 0x2090
##     0x102a + 4191 + 7 = 0x2090
# DISASM-PIC:      Disassembly of section .text:
# DISASM-PIC-NEXT: _start:
# DISASM-PIC-NEXT:  1000: 13 05 8a 10 00 00      adcl  4234(%rip), %eax
# DISASM-PIC-NEXT:  1006: 03 1d 84 10 00 00      addl  4228(%rip), %ebx
# DISASM-PIC-NEXT:  100c: 23 0d 7e 10 00 00      andl  4222(%rip), %ecx
# DISASM-PIC-NEXT:  1012: 3b 15 78 10 00 00      cmpl  4216(%rip), %edx
# DISASM-PIC-NEXT:  1018: 0b 35 72 10 00 00      orl   4210(%rip), %esi
# DISASM-PIC-NEXT:  101e: 1b 3d 6c 10 00 00      sbbl  4204(%rip), %edi
# DISASM-PIC-NEXT:  1024: 2b 2d 66 10 00 00      subl  4198(%rip), %ebp
# DISASM-PIC-NEXT:  102a: 44 33 05 5f 10 00 00   xorl  4191(%rip), %r8d
# DISASM-PIC-NEXT:  1031: 44 85 3d 58 10 00 00   testl 4184(%rip), %r15d
# DISASM-PIC-NEXT:  1038: 48 13 05 51 10 00 00   adcq  4177(%rip), %rax
# DISASM-PIC-NEXT:  103f: 48 03 1d 4a 10 00 00   addq  4170(%rip), %rbx
# DISASM-PIC-NEXT:  1046: 48 23 0d 43 10 00 00   andq  4163(%rip), %rcx
# DISASM-PIC-NEXT:  104d: 48 3b 15 3c 10 00 00   cmpq  4156(%rip), %rdx
# DISASM-PIC-NEXT:  1054: 48 0b 3d 35 10 00 00   orq   4149(%rip), %rdi
# DISASM-PIC-NEXT:  105b: 48 1b 35 2e 10 00 00   sbbq  4142(%rip), %rsi
# DISASM-PIC-NEXT:  1062: 48 2b 2d 27 10 00 00   subq  4135(%rip), %rbp
# DISASM-PIC-NEXT:  1069: 4c 33 05 20 10 00 00   xorq  4128(%rip), %r8
# DISASM-PIC-NEXT:  1070: 4c 85 3d 19 10 00 00   testq 4121(%rip), %r15

.data
.type   bar, @object
bar:
 .byte   1
 .size   bar, .-bar

.text
.globl  _start
.type   _start, @function
_start:
  adcl    bar@GOTPCREL(%rip), %eax
  addl    bar@GOTPCREL(%rip), %ebx
  andl    bar@GOTPCREL(%rip), %ecx
  cmpl    bar@GOTPCREL(%rip), %edx
  orl     bar@GOTPCREL(%rip), %esi
  sbbl    bar@GOTPCREL(%rip), %edi
  subl    bar@GOTPCREL(%rip), %ebp
  xorl    bar@GOTPCREL(%rip), %r8d
  testl   %r15d, bar@GOTPCREL(%rip)
  adcq    bar@GOTPCREL(%rip), %rax
  addq    bar@GOTPCREL(%rip), %rbx
  andq    bar@GOTPCREL(%rip), %rcx
  cmpq    bar@GOTPCREL(%rip), %rdx
  orq     bar@GOTPCREL(%rip), %rdi
  sbbq    bar@GOTPCREL(%rip), %rsi
  subq    bar@GOTPCREL(%rip), %rbp
  xorq    bar@GOTPCREL(%rip), %r8
  testq   %r15, bar@GOTPCREL(%rip)
