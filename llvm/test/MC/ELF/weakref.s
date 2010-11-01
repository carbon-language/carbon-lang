// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | elf-dump  | FileCheck %s

// This is a long test that checks that the aliases created by weakref are
// never in the symbol table and that the only case it causes a symbol to
// be output as a weak undefined symbol is if that variable is not defined
// in this file and all the references to it are done via the alias.

        .weakref foo1, bar1

        .weakref foo2, bar2
        .long bar2

        .weakref foo3, bar3
        .long foo3

        .weakref foo4, bar4
        .long foo4
        .long bar4

        .weakref foo5, bar5
        .long bar5
        .long foo5

bar6:
        .weakref foo6, bar6

bar7:
        .weakref foo7, bar7
        .long bar7

bar8:
        .weakref foo8, bar8
        .long foo8

bar9:
        .weakref foo9, bar9
        .long foo9
        .long bar9

bar10:
        .global bar10
        .weakref foo10, bar10
        .long bar10
        .long foo10

bar11:
        .global bar11
        .weakref foo11, bar11

bar12:
        .global bar12
        .weakref foo12, bar12
        .long bar12

bar13:
        .global bar13
        .weakref foo13, bar13
        .long foo13

bar14:
        .global bar14
        .weakref foo14, bar14
        .long foo14
        .long bar14

bar15:
        .global bar15
        .weakref foo15, bar15
        .long bar15
        .long foo15

// CHECK:       # Symbol 0x00000000
// CHECK-NEXT:  (('st_name', 0x00000000) # ''
// CHECK-NEXT:   ('st_bind', 0x00000000)
// CHECK-NEXT:   ('st_type', 0x00000000)
// CHECK-NEXT:   ('st_other', 0x00000000)
// CHECK-NEXT:   ('st_shndx', 0x00000000)
// CHECK-NEXT:   ('st_value', 0x00000000)
// CHECK-NEXT:   ('st_size', 0x00000000)
// CHECK-NEXT:  ),
// CHECK-NEXT:  # Symbol 0x00000001
// CHECK-NEXT:  (('st_name', 0x00000015) # 'bar6'
// CHECK-NEXT:   ('st_bind', 0x00000000)
// CHECK-NEXT:   ('st_type', 0x00000000)
// CHECK-NEXT:   ('st_other', 0x00000000)
// CHECK-NEXT:   ('st_shndx', 0x00000001)
// CHECK-NEXT:   ('st_value', 0x00000018)
// CHECK-NEXT:   ('st_size', 0x00000000)
// CHECK-NEXT:  ),
// CHECK-NEXT:  # Symbol 0x00000002
// CHECK-NEXT:  (('st_name', 0x0000001a) # 'bar7'
// CHECK-NEXT:   ('st_bind', 0x00000000)
// CHECK-NEXT:   ('st_type', 0x00000000)
// CHECK-NEXT:   ('st_other', 0x00000000)
// CHECK-NEXT:   ('st_shndx', 0x00000001)
// CHECK-NEXT:   ('st_value', 0x00000018)
// CHECK-NEXT:   ('st_size', 0x00000000)
// CHECK-NEXT:  ),
// CHECK-NEXT:  # Symbol 0x00000003
// CHECK-NEXT:  (('st_name', 0x0000001f) # 'bar8'
// CHECK-NEXT:   ('st_bind', 0x00000000)
// CHECK-NEXT:   ('st_type', 0x00000000)
// CHECK-NEXT:   ('st_other', 0x00000000)
// CHECK-NEXT:   ('st_shndx', 0x00000001)
// CHECK-NEXT:   ('st_value', 0x0000001c)
// CHECK-NEXT:   ('st_size', 0x00000000)
// CHECK-NEXT:  ),
// CHECK-NEXT:  # Symbol 0x00000004
// CHECK-NEXT:  (('st_name', 0x00000024) # 'bar9'
// CHECK-NEXT:   ('st_bind', 0x00000000)
// CHECK-NEXT:   ('st_type', 0x00000000)
// CHECK-NEXT:   ('st_other', 0x00000000)
// CHECK-NEXT:   ('st_shndx', 0x00000001)
// CHECK-NEXT:   ('st_value', 0x00000020)
// CHECK-NEXT:   ('st_size', 0x00000000)
// CHECK-NEXT:  ),
// CHECK-NEXT:  # Symbol 0x00000005
// CHECK-NEXT:  (('st_name', 0x00000000) # ''
// CHECK-NEXT:   ('st_bind', 0x00000000)
// CHECK-NEXT:   ('st_type', 0x00000003)
// CHECK-NEXT:   ('st_other', 0x00000000)
// CHECK-NEXT:   ('st_shndx', 0x00000001)
// CHECK-NEXT:   ('st_value', 0x00000000)
// CHECK-NEXT:   ('st_size', 0x00000000)
// CHECK-NEXT:  ),
// CHECK-NEXT:  # Symbol 0x00000006
// CHECK-NEXT:  (('st_name', 0x00000000) # ''
// CHECK-NEXT:   ('st_bind', 0x00000000)
// CHECK-NEXT:   ('st_type', 0x00000003)
// CHECK-NEXT:   ('st_other', 0x00000000)
// CHECK-NEXT:   ('st_shndx', 0x00000002)
// CHECK-NEXT:   ('st_value', 0x00000000)
// CHECK-NEXT:   ('st_size', 0x00000000)
// CHECK-NEXT:  ),
// CHECK-NEXT:  # Symbol 0x00000007
// CHECK-NEXT:  (('st_name', 0x00000000) # ''
// CHECK-NEXT:   ('st_bind', 0x00000000)
// CHECK-NEXT:   ('st_type', 0x00000003)
// CHECK-NEXT:   ('st_other', 0x00000000)
// CHECK-NEXT:   ('st_shndx', 0x00000003)
// CHECK-NEXT:   ('st_value', 0x00000000)
// CHECK-NEXT:   ('st_size', 0x00000000)
// CHECK-NEXT:  ),
// CHECK-NEXT:  # Symbol 0x00000008
// CHECK-NEXT:  (('st_name', 0x00000029) # 'bar10'
// CHECK-NEXT:   ('st_bind', 0x00000001)
// CHECK-NEXT:   ('st_type', 0x00000000)
// CHECK-NEXT:   ('st_other', 0x00000000)
// CHECK-NEXT:   ('st_shndx', 0x00000001)
// CHECK-NEXT:   ('st_value', 0x00000028)
// CHECK-NEXT:   ('st_size', 0x00000000)
// CHECK-NEXT:  ),
// CHECK-NEXT:  # Symbol 0x00000009
// CHECK-NEXT:  (('st_name', 0x0000002f) # 'bar11'
// CHECK-NEXT:   ('st_bind', 0x00000001)
// CHECK-NEXT:   ('st_type', 0x00000000)
// CHECK-NEXT:   ('st_other', 0x00000000)
// CHECK-NEXT:   ('st_shndx', 0x00000001)
// CHECK-NEXT:   ('st_value', 0x00000030)
// CHECK-NEXT:   ('st_size', 0x00000000)
// CHECK-NEXT:  ),
// CHECK-NEXT:  # Symbol 0x0000000a
// CHECK-NEXT:  (('st_name', 0x00000035) # 'bar12'
// CHECK-NEXT:   ('st_bind', 0x00000001)
// CHECK-NEXT:   ('st_type', 0x00000000)
// CHECK-NEXT:   ('st_other', 0x00000000)
// CHECK-NEXT:   ('st_shndx', 0x00000001)
// CHECK-NEXT:   ('st_value', 0x00000030)
// CHECK-NEXT:   ('st_size', 0x00000000)
// CHECK-NEXT:  ),
// CHECK-NEXT:  # Symbol 0x0000000b
// CHECK-NEXT:  (('st_name', 0x0000003b) # 'bar13'
// CHECK-NEXT:   ('st_bind', 0x00000001)
// CHECK-NEXT:   ('st_type', 0x00000000)
// CHECK-NEXT:   ('st_other', 0x00000000)
// CHECK-NEXT:   ('st_shndx', 0x00000001)
// CHECK-NEXT:   ('st_value', 0x00000034)
// CHECK-NEXT:   ('st_size', 0x00000000)
// CHECK-NEXT:  ),
// CHECK-NEXT:  # Symbol 0x0000000c
// CHECK-NEXT:  (('st_name', 0x00000041) # 'bar14'
// CHECK-NEXT:   ('st_bind', 0x00000001)
// CHECK-NEXT:   ('st_type', 0x00000000)
// CHECK-NEXT:   ('st_other', 0x00000000)
// CHECK-NEXT:   ('st_shndx', 0x00000001)
// CHECK-NEXT:   ('st_value', 0x00000038)
// CHECK-NEXT:   ('st_size', 0x00000000)
// CHECK-NEXT:  ),
// CHECK-NEXT:  # Symbol 0x0000000d
// CHECK-NEXT:  (('st_name', 0x00000047) # 'bar15'
// CHECK-NEXT:   ('st_bind', 0x00000001)
// CHECK-NEXT:   ('st_type', 0x00000000)
// CHECK-NEXT:   ('st_other', 0x00000000)
// CHECK-NEXT:   ('st_shndx', 0x00000001)
// CHECK-NEXT:   ('st_value', 0x00000040)
// CHECK-NEXT:   ('st_size', 0x00000000)
// CHECK-NEXT:  ),
// CHECK-NEXT:  # Symbol 0x0000000e
// CHECK-NEXT:  (('st_name', 0x00000001) # 'bar2'
// CHECK-NEXT:   ('st_bind', 0x00000001)
// CHECK-NEXT:   ('st_type', 0x00000000)
// CHECK-NEXT:   ('st_other', 0x00000000)
// CHECK-NEXT:   ('st_shndx', 0x00000000)
// CHECK-NEXT:   ('st_value', 0x00000000)
// CHECK-NEXT:   ('st_size', 0x00000000)
// CHECK-NEXT:  ),
// CHECK-NEXT:  # Symbol 0x0000000f
// CHECK-NEXT:  (('st_name', 0x00000006) # 'bar3'
// CHECK-NEXT:   ('st_bind', 0x00000002)
// CHECK-NEXT:   ('st_type', 0x00000000)
// CHECK-NEXT:   ('st_other', 0x00000000)
// CHECK-NEXT:   ('st_shndx', 0x00000000)
// CHECK-NEXT:   ('st_value', 0x00000000)
// CHECK-NEXT:   ('st_size', 0x00000000)
// CHECK-NEXT:  ),
// CHECK-NEXT:  # Symbol 0x00000010
// CHECK-NEXT:  (('st_name', 0x0000000b) # 'bar4'
// CHECK-NEXT:   ('st_bind', 0x00000001)
// CHECK-NEXT:   ('st_type', 0x00000000)
// CHECK-NEXT:   ('st_other', 0x00000000)
// CHECK-NEXT:   ('st_shndx', 0x00000000)
// CHECK-NEXT:   ('st_value', 0x00000000)
// CHECK-NEXT:   ('st_size', 0x00000000)
// CHECK-NEXT:  ),
// CHECK-NEXT:  # Symbol 0x00000011
// CHECK-NEXT:  (('st_name', 0x00000010) # 'bar5'
// CHECK-NEXT:   ('st_bind', 0x00000001)
// CHECK-NEXT:   ('st_type', 0x00000000)
// CHECK-NEXT:   ('st_other', 0x00000000)
// CHECK-NEXT:   ('st_shndx', 0x00000000)
// CHECK-NEXT:   ('st_value', 0x00000000)
// CHECK-NEXT:   ('st_size', 0x00000000)
// CHECK-NEXT:  ),
// CHECK-NEXT: ])
