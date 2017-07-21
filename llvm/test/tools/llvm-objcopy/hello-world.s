# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.o
# RUN: ld.lld %t.o -o %t
# RUN: llvm-objcopy %t %t2
# RUN: llvm-readobj -program-headers %t2 | FileCheck %s

  .global _start
  .text
_start:
  mov $1, %rax
  mov $1, %rdi
  mov $msg, %rsi
  mov $14, %rdx
  syscall

  mov $60, %rax
  mov $0, %rdi
  syscall

  .rodata
msg:
  .ascii  "Hello, World!\n"

# CHECK:       ProgramHeader {
# CHECK:         Type: PT_LOAD
# CHECK-NEXT:    Offset:
# CHECK-NEXT:    VirtualAddress:
# CHECK-NEXT:    PhysicalAddress:
# CHECK-NEXT:    FileSize:
# CHECK-NEXT:    MemSize:
# CHECK-NEXT:    Flags [
# CHECK-NEXT:      PF_R
# CHECK-NEXT:    ]
# CHECK-NEXT:    Alignment: 4096
# CHECK-NEXT:  }

# CHECK:       ProgramHeader {
# CHECK-NEXT:    Type: PT_LOAD
# CHECK-NEXT:    Offset:
# CHECK-NEXT:    VirtualAddress:
# CHECK-NEXT:    PhysicalAddress:
# CHECK-NEXT:    FileSize: 46
# CHECK-NEXT:    MemSize: 46
# CHECK-NEXT:    Flags [
# CHECK-NEXT:      PF_R
# CHECK-NEXT:      PF_X
# CHECK-NEXT:    ]
# CHECK-NEXT:    Alignment:
# CHECK-NEXT:  }
