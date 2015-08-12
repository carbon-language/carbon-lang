# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t
# RUN: lld -flavor gnu2 %t -o %t2
# RUN: llvm-readobj -file-headers -sections -program-headers %t2 | FileCheck %s
# REQUIRES: x86

# exits with return code 42 on linux
.globl _start;
_start:
  mov $60, %rax
  mov $42, %rdi
  syscall

# CHECK:      ElfHeader {
# CHECK-NEXT:   Ident {
# CHECK-NEXT:     Magic: (7F 45 4C 46)
# CHECK-NEXT:     Class: 64-bit (0x2)
# CHECK-NEXT:     DataEncoding: LittleEndian (0x1)
# CHECK-NEXT:     FileVersion: 1
# CHECK-NEXT:     OS/ABI: SystemV (0x0)
# CHECK-NEXT:     ABIVersion: 0
# CHECK-NEXT:     Unused: (00 00 00 00 00 00 00)
# CHECK-NEXT:   }
# CHECK-NEXT:   Type: Executable (0x2)
# CHECK-NEXT:   Machine: EM_X86_64 (0x3E)
# CHECK-NEXT:   Version: 1
# CHECK-NEXT:   Entry: 0x401000
# CHECK-NEXT:   ProgramHeaderOffset: 0x40
# CHECK-NEXT:   SectionHeaderOffset: 0x1030
# CHECK-NEXT:   Flags [ (0x0)
# CHECK-NEXT:   ]
# CHECK-NEXT:   HeaderSize: 64
# CHECK-NEXT:   ProgramHeaderEntrySize: 56
# CHECK-NEXT:   ProgramHeaderCount: 1
# CHECK-NEXT:   SectionHeaderEntrySize: 64
# CHECK-NEXT:   SectionHeaderCount: 5
# CHECK-NEXT:   StringTableSectionIndex: 4
# CHECK-NEXT: }
# CHECK-NEXT: Sections [
# CHECK-NEXT:   Section {
# CHECK-NEXT:     Index: 0
# CHECK-NEXT:     Name:  (0)
# CHECK-NEXT:     Type: SHT_NULL (0x0)
# CHECK-NEXT:     Flags [ (0x0)
# CHECK-NEXT:     ]
# CHECK-NEXT:     Address: 0x0
# CHECK-NEXT:     Offset: 0x0
# CHECK-NEXT:     Size: 0
# CHECK-NEXT:     Link: 0
# CHECK-NEXT:     Info: 0
# CHECK-NEXT:     AddressAlignment: 0
# CHECK-NEXT:     EntrySize: 0
# CHECK-NEXT:   }
# CHECK-NEXT:   Section {
# CHECK-NEXT:     Index: 1
# CHECK-NEXT:     Name: .text
# CHECK-NEXT:     Type: SHT_PROGBITS (0x1)
# CHECK-NEXT:     Flags [ (0x6)
# CHECK-NEXT:       SHF_ALLOC (0x2)
# CHECK-NEXT:       SHF_EXECINSTR (0x4)
# CHECK-NEXT:     ]
# CHECK-NEXT:     Address: 0x1000
# CHECK-NEXT:     Offset: 0x1000
# CHECK-NEXT:     Size: 16
# CHECK-NEXT:     Link: 0
# CHECK-NEXT:     Info: 0
# CHECK-NEXT:     AddressAlignment: 0
# CHECK-NEXT:     EntrySize: 0
# CHECK-NEXT:   }
# CHECK-NEXT:   Section {
# CHECK-NEXT:     Index: 2
# CHECK-NEXT:     Name: .data
# CHECK-NEXT:     Type: SHT_PROGBITS (0x1)
# CHECK-NEXT:     Flags [ (0x3)
# CHECK-NEXT:       SHF_ALLOC (0x2)
# CHECK-NEXT:       SHF_WRITE (0x1)
# CHECK-NEXT:     ]
# CHECK-NEXT:     Address: 0x2000
# CHECK-NEXT:     Offset: 0x0
# CHECK-NEXT:     Size: 0
# CHECK-NEXT:     Link: 0
# CHECK-NEXT:     Info: 0
# CHECK-NEXT:     AddressAlignment: 0
# CHECK-NEXT:     EntrySize: 0
# CHECK-NEXT:   }
# CHECK-NEXT:   Section {
# CHECK-NEXT:     Index: 3
# CHECK-NEXT:     Name: .bss
# CHECK-NEXT:     Type: SHT_NOBITS (0x8)
# CHECK-NEXT:     Flags [ (0x3)
# CHECK-NEXT:       SHF_ALLOC (0x2)
# CHECK-NEXT:       SHF_WRITE (0x1)
# CHECK-NEXT:     ]
# CHECK-NEXT:     Address: 0x2000
# CHECK-NEXT:     Offset: 0x0
# CHECK-NEXT:     Size: 0
# CHECK-NEXT:     Link: 0
# CHECK-NEXT:     Info: 0
# CHECK-NEXT:     AddressAlignment: 0
# CHECK-NEXT:     EntrySize: 0
# CHECK-NEXT:   }
# CHECK-NEXT:   Section {
# CHECK-NEXT:     Index: 4
# CHECK-NEXT:     Name: .strtab (12)
# CHECK-NEXT:     Type: SHT_STRTAB (0x3)
# CHECK-NEXT:     Flags [ (0x0)
# CHECK-NEXT:     ]
# CHECK-NEXT:     Address: 0x0
# CHECK-NEXT:     Offset: 0x1010
# CHECK-NEXT:     Size: 26
# CHECK-NEXT:     Link: 0
# CHECK-NEXT:     Info: 0
# CHECK-NEXT:     AddressAlignment: 1
# CHECK-NEXT:     EntrySize: 0
# CHECK-NEXT:   }
# CHECK-NEXT: ]
# CHECK-NEXT: ProgramHeaders [
# CHECK-NEXT:   ProgramHeader {
# CHECK-NEXT:     Type: PT_LOAD (0x1)
# CHECK-NEXT:     Offset: 0x0
# CHECK-NEXT:     VirtualAddress: 0x400000
# CHECK-NEXT:     PhysicalAddress: 0x400000
# CHECK-NEXT:     FileSize: 4464
# CHECK-NEXT:     MemSize: 4464
# CHECK-NEXT:     Flags [ (0x5)
# CHECK-NEXT:       PF_R (0x4)
# CHECK-NEXT:       PF_X (0x1)
# CHECK-NEXT:     ]
# CHECK-NEXT:     Alignment: 16384
# CHECK-NEXT:   }
# CHECK-NEXT: ]

# RUN: not lld -flavor gnu2 %t 2>&1 | FileCheck --check-prefix=NO_O %s
# NO_O: -o must be specified.

# RUN: not lld -flavor gnu2 %t.foo -o %t2 2>&1 | \
# RUN:  FileCheck --check-prefix=MISSING %s
# MISSING: cannot open {{.*}}.foo: {{[Nn]}}o such file or directory

# RUN: not lld -flavor gnu2 -o %t2 2>&1 | \
# RUN:  FileCheck --check-prefix=NO_INPUT %s
# NO_INPUT: no input files.

# RUN: not lld -flavor gnu2 %t.no.such.file -o %t2 2>&1 | \
# RUN:  FileCheck --check-prefix=CANNOT_OPEN %s
# CANNOT_OPEN: cannot open {{.*}}.no.such.file: {{[Nn]}}o such file or directory

# RUN: not lld -flavor gnu2 %t -o 2>&1 | FileCheck --check-prefix=NO_O_VAL %s
# NO_O_VAL: missing arg value for "-o", expected 1 argument.

# RUN: not lld -flavor gnu2 --foo 2>&1 | FileCheck --check-prefix=UNKNOWN %s
# UNKNOWN: unknown argument: --foo

# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t
# RUN: not lld -flavor gnu2 %t %t -o %t2 2>&1 | FileCheck --check-prefix=DUP %s
# DUP: duplicate symbol: _start
