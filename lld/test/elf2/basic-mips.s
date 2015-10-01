# RUN: llvm-mc -filetype=obj -triple=mipsel-unknown-linux %s -o %t.o
# RUN: lld -flavor gnu2 %t.o -o %t.exe
# RUN: llvm-readobj -file-headers -sections -program-headers -symbols %t.exe \
# RUN:   | FileCheck %s

# REQUIRES: mips

# Exits with return code 1 on Linux.
        .globl  __start
__start:
        li      $a0,1
        li      $v0,4001
        syscall

# CHECK:      ElfHeader {
# CHECK-NEXT:   Ident {
# CHECK-NEXT:     Magic: (7F 45 4C 46)
# CHECK-NEXT:     Class: 32-bit (0x1)
# CHECK-NEXT:     DataEncoding: LittleEndian (0x1)
# CHECK-NEXT:     FileVersion: 1
# CHECK-NEXT:     OS/ABI: SystemV (0x0)
# CHECK-NEXT:     ABIVersion: 0
# CHECK-NEXT:     Unused: (00 00 00 00 00 00 00)
# CHECK-NEXT:   }
# CHECK-NEXT:   Type: Executable (0x2)
# CHECK-NEXT:   Machine: EM_MIPS (0x8)
# CHECK-NEXT:   Version: 1
# CHECK-NEXT:   Entry: 0x12000
# CHECK-NEXT:   ProgramHeaderOffset: 0x34
# CHECK-NEXT:   SectionHeaderOffset: 0x2074
# CHECK-NEXT:   Flags [ (0x0)
# CHECK-NEXT:   ]
# CHECK-NEXT:   HeaderSize: 52
# CHECK-NEXT:   ProgramHeaderEntrySize: 32
# CHECK-NEXT:   ProgramHeaderCount: 2
# CHECK-NEXT:   SectionHeaderEntrySize: 40
# CHECK-NEXT:   SectionHeaderCount: 8
# CHECK-NEXT:   StringTableSectionIndex: 7
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
# CHECK-NEXT:     Name: .reginfo (35)
# CHECK-NEXT:     Type: SHT_MIPS_REGINFO (0x70000006)
# CHECK-NEXT:     Flags [ (0x2)
# CHECK-NEXT:       SHF_ALLOC (0x2)
# CHECK-NEXT:     ]
# CHECK-NEXT:     Address: 0x11000
# CHECK-NEXT:     Offset: 0x1000
# CHECK-NEXT:     Size: 24
# CHECK-NEXT:     Link: 0
# CHECK-NEXT:     Info: 0
# CHECK-NEXT:     AddressAlignment: 4
# CHECK-NEXT:     EntrySize: 0
# CHECK-NEXT:   }
# CHECK-NEXT:   Section {
# CHECK-NEXT:     Index: 2
# CHECK-NEXT:     Name: .MIPS.abiflags (20)
# CHECK-NEXT:     Type: SHT_MIPS_ABIFLAGS (0x7000002A)
# CHECK-NEXT:     Flags [ (0x2)
# CHECK-NEXT:       SHF_ALLOC (0x2)
# CHECK-NEXT:     ]
# CHECK-NEXT:     Address: 0x11018
# CHECK-NEXT:     Offset: 0x1018
# CHECK-NEXT:     Size: 24
# CHECK-NEXT:     Link: 0
# CHECK-NEXT:     Info: 0
# CHECK-NEXT:     AddressAlignment: 8
# CHECK-NEXT:     EntrySize: 0
# CHECK-NEXT:   }
# CHECK-NEXT:   Section {
# CHECK-NEXT:     Index: 3
# CHECK-NEXT:     Name: .text (1)
# CHECK-NEXT:     Type: SHT_PROGBITS (0x1)
# CHECK-NEXT:     Flags [ (0x6)
# CHECK-NEXT:       SHF_ALLOC (0x2)
# CHECK-NEXT:       SHF_EXECINSTR (0x4)
# CHECK-NEXT:     ]
# CHECK-NEXT:     Address: 0x12000
# CHECK-NEXT:     Offset: 0x2000
# CHECK-NEXT:     Size: 12
# CHECK-NEXT:     Link: 0
# CHECK-NEXT:     Info: 0
# CHECK-NEXT:     AddressAlignment: 16
# CHECK-NEXT:     EntrySize: 0
# CHECK-NEXT:   }
# CHECK-NEXT:   Section {
# CHECK-NEXT:     Index: 4
# CHECK-NEXT:     Name: .data (60)
# CHECK-NEXT:     Type: SHT_PROGBITS (0x1)
# CHECK-NEXT:     Flags [ (0x3)
# CHECK-NEXT:       SHF_ALLOC (0x2)
# CHECK-NEXT:       SHF_WRITE (0x1)
# CHECK-NEXT:     ]
# CHECK-NEXT:     Address: 0x12010
# CHECK-NEXT:     Offset: 0x2010
# CHECK-NEXT:     Size: 0
# CHECK-NEXT:     Link: 0
# CHECK-NEXT:     Info: 0
# CHECK-NEXT:     AddressAlignment: 16
# CHECK-NEXT:     EntrySize: 0
# CHECK-NEXT:   }
# CHECK-NEXT:   Section {
# CHECK-NEXT:     Index: 5
# CHECK-NEXT:     Name: .bss (15)
# CHECK-NEXT:     Type: SHT_NOBITS (0x8)
# CHECK-NEXT:     Flags [ (0x3)
# CHECK-NEXT:       SHF_ALLOC (0x2)
# CHECK-NEXT:       SHF_WRITE (0x1)
# CHECK-NEXT:     ]
# CHECK-NEXT:     Address: 0x12010
# CHECK-NEXT:     Offset: 0x2010
# CHECK-NEXT:     Size: 0
# CHECK-NEXT:     Link: 0
# CHECK-NEXT:     Info: 0
# CHECK-NEXT:     AddressAlignment: 16
# CHECK-NEXT:     EntrySize: 0
# CHECK-NEXT:   }
# CHECK-NEXT:   Section {
# CHECK-NEXT:     Index: 6
# CHECK-NEXT:     Name: .symtab (52)
# CHECK-NEXT:     Type: SHT_SYMTAB (0x2)
# CHECK-NEXT:     Flags [ (0x0)
# CHECK-NEXT:     ]
# CHECK-NEXT:     Address: 0x0
# CHECK-NEXT:     Offset: 0x2010
# CHECK-NEXT:     Size: 32
# CHECK-NEXT:     Link: 7
# CHECK-NEXT:     Info: 1
# CHECK-NEXT:     AddressAlignment: 4
# CHECK-NEXT:     EntrySize: 16
# CHECK-NEXT:   }
# CHECK-NEXT:   Section {
# CHECK-NEXT:     Index: 7
# CHECK-NEXT:     Name: .strtab (44)
# CHECK-NEXT:     Type: SHT_STRTAB (0x3)
# CHECK-NEXT:     Flags [ (0x0)
# CHECK-NEXT:     ]
# CHECK-NEXT:     Address: 0x0
# CHECK-NEXT:     Offset: 0x2030
# CHECK-NEXT:     Size: 66
# CHECK-NEXT:     Link: 0
# CHECK-NEXT:     Info: 0
# CHECK-NEXT:     AddressAlignment: 1
# CHECK-NEXT:     EntrySize: 0
# CHECK-NEXT:   }
# CHECK-NEXT: ]
# CHECK-NEXT: Symbols [
# CHECK-NEXT:   Symbol {
# CHECK-NEXT:     Name:  (0)
# CHECK-NEXT:     Value: 0x0
# CHECK-NEXT:     Size: 0
# CHECK-NEXT:     Binding: Local (0x0)
# CHECK-NEXT:     Type: None (0x0)
# CHECK-NEXT:     Other: 0
# CHECK-NEXT:     Section: Undefined (0x0)
# CHECK-NEXT:   }
# CHECK-NEXT:   Symbol {
# CHECK-NEXT:     Name: __start (7)
# CHECK-NEXT:     Value: 0x12000
# CHECK-NEXT:     Size: 0
# CHECK-NEXT:     Binding: Global (0x1)
# CHECK-NEXT:     Type: None (0x0)
# CHECK-NEXT:     Other: 0
# CHECK-NEXT:     Section: .text
# CHECK-NEXT:   }
# CHECK-NEXT: ]
# CHECK-NEXT: ProgramHeaders [
# CHECK-NEXT:   ProgramHeader {
# CHECK-NEXT:     Type: PT_LOAD (0x1)
# CHECK-NEXT:     Offset: 0x0
# CHECK-NEXT:     VirtualAddress: 0x10000
# CHECK-NEXT:     PhysicalAddress: 0x10000
# CHECK-NEXT:     FileSize: 4144
# CHECK-NEXT:     MemSize: 4144
# CHECK-NEXT:     Flags [ (0x4)
# CHECK-NEXT:       PF_R (0x4)
# CHECK-NEXT:     ]
# CHECK-NEXT:     Alignment: 4096
# CHECK-NEXT:   }
# CHECK-NEXT:   ProgramHeader {
# CHECK-NEXT:     Type: PT_LOAD (0x1)
# CHECK-NEXT:     Offset: 0x2000
# CHECK-NEXT:     VirtualAddress: 0x12000
# CHECK-NEXT:     PhysicalAddress: 0x12000
# CHECK-NEXT:     FileSize: 16
# CHECK-NEXT:     MemSize: 16
# CHECK-NEXT:     Flags [ (0x5)
# CHECK-NEXT:       PF_R (0x4)
# CHECK-NEXT:       PF_X (0x1)
# CHECK-NEXT:     ]
# CHECK-NEXT:     Alignment: 4096
# CHECK-NEXT:   }
# CHECK-NEXT: ]
