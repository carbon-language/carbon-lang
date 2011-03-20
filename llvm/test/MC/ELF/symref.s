// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | elf-dump  | FileCheck %s

defined1:
defined2:
defined3:
        .symver defined1, bar1@zed
        .symver undefined1, bar2@zed

        .symver defined2, bar3@@zed

        .symver defined3, bar5@@@zed
        .symver undefined3, bar6@@@zed

        .long defined1
        .long undefined1
        .long defined2
        .long defined3
        .long undefined3

        .global global1
        .symver global1, g1@@zed
global1:


// CHECK:      # Relocation 0x00000000
// CHECK-NEXT: (('r_offset', 0x00000000)
// CHECK-NEXT:  ('r_sym', 0x00000006)
// CHECK-NEXT:  ('r_type', 0x0000000a)
// CHECK-NEXT:  ('r_addend', 0x00000000)
// CHECK-NEXT: ),
// CHECK-NEXT: # Relocation 0x00000001
// CHECK-NEXT: (('r_offset', 0x00000004)
// CHECK-NEXT:  ('r_sym', 0x0000000b)
// CHECK-NEXT:  ('r_type', 0x0000000a)
// CHECK-NEXT:  ('r_addend', 0x00000000)
// CHECK-NEXT: ),
// CHECK-NEXT: # Relocation 0x00000002
// CHECK-NEXT: (('r_offset', 0x00000008)
// CHECK-NEXT:  ('r_sym', 0x00000006)
// CHECK-NEXT:  ('r_type', 0x0000000a)
// CHECK-NEXT:  ('r_addend', 0x00000000)
// CHECK-NEXT: ),
// CHECK-NEXT: # Relocation 0x00000003
// CHECK-NEXT: (('r_offset', 0x0000000c)
// CHECK-NEXT:  ('r_sym', 0x00000006)
// CHECK-NEXT:  ('r_type', 0x0000000a)
// CHECK-NEXT:  ('r_addend', 0x00000000)
// CHECK-NEXT: ),
// CHECK-NEXT: # Relocation 0x00000004
// CHECK-NEXT: (('r_offset', 0x00000010)
// CHECK-NEXT:  ('r_sym', 0x0000000c)
// CHECK-NEXT:  ('r_type', 0x0000000a)
// CHECK-NEXT:  ('r_addend', 0x00000000)
// CHECK-NEXT: ),
// CHECK-NEXT:])

// CHECK:      # Symbol 0x00000001
// CHECK-NEXT: (('st_name', 0x00000013) # 'bar1@zed'
// CHECK-NEXT:  ('st_bind', 0x00000000)
// CHECK-NEXT:  ('st_type', 0x00000000)
// CHECK-NEXT:  ('st_other', 0x00000000)
// CHECK-NEXT:  ('st_shndx', 0x00000001)
// CHECK-NEXT:  ('st_value', 0x0000000000000000)
// CHECK-NEXT:  ('st_size', 0x0000000000000000)
// CHECK-NEXT: ),
// CHECK-NEXT: # Symbol 0x00000002
// CHECK-NEXT: (('st_name', 0x00000025) # 'bar3@@zed'
// CHECK-NEXT:  ('st_bind', 0x00000000)
// CHECK-NEXT:  ('st_type', 0x00000000)
// CHECK-NEXT:  ('st_other', 0x00000000)
// CHECK-NEXT:  ('st_shndx', 0x00000001)
// CHECK-NEXT:  ('st_value', 0x0000000000000000)
// CHECK-NEXT:  ('st_size', 0x0000000000000000)
// CHECK-NEXT: ),
// CHECK-NEXT: # Symbol 0x00000003
// CHECK-NEXT: (('st_name', 0x0000002f) # 'bar5@@zed'
// CHECK-NEXT:  ('st_bind', 0x00000000)
// CHECK-NEXT:  ('st_type', 0x00000000)
// CHECK-NEXT:  ('st_other', 0x00000000)
// CHECK-NEXT:  ('st_shndx', 0x00000001)
// CHECK-NEXT:  ('st_value', 0x0000000000000000)
// CHECK-NEXT:  ('st_size', 0x0000000000000000)
// CHECK-NEXT: ),
// CHECK-NEXT: # Symbol 0x00000004
// CHECK-NEXT: (('st_name', 0x00000001) # 'defined1'
// CHECK-NEXT:  ('st_bind', 0x00000000)
// CHECK-NEXT:  ('st_type', 0x00000000)
// CHECK-NEXT:  ('st_other', 0x00000000)
// CHECK-NEXT:  ('st_shndx', 0x00000001)
// CHECK-NEXT:  ('st_value', 0x0000000000000000)
// CHECK-NEXT:  ('st_size', 0x0000000000000000)
// CHECK-NEXT: ),
// CHECK-NEXT: # Symbol 0x00000005
// CHECK-NEXT: (('st_name', 0x0000000a) # 'defined2'
// CHECK-NEXT:  ('st_bind', 0x00000000)
// CHECK-NEXT:  ('st_type', 0x00000000)
// CHECK-NEXT:  ('st_other', 0x00000000)
// CHECK-NEXT:  ('st_shndx', 0x00000001)
// CHECK-NEXT:  ('st_value', 0x0000000000000000)
// CHECK-NEXT:  ('st_size', 0x0000000000000000)
// CHECK-NEXT: ),
// CHECK-NEXT: # Symbol 0x00000006
// CHECK-NEXT: (('st_name', 0x00000000) # ''
// CHECK-NEXT:  ('st_bind', 0x00000000)
// CHECK-NEXT:  ('st_type', 0x00000003)
// CHECK-NEXT:  ('st_other', 0x00000000)
// CHECK-NEXT:  ('st_shndx', 0x00000001)
// CHECK-NEXT:  ('st_value', 0x0000000000000000)
// CHECK-NEXT:  ('st_size', 0x0000000000000000)
// CHECK-NEXT: ),
// CHECK-NEXT: # Symbol 0x00000007
// CHECK-NEXT: (('st_name', 0x00000000) # ''
// CHECK-NEXT:  ('st_bind', 0x00000000)
// CHECK-NEXT:  ('st_type', 0x00000003)
// CHECK-NEXT:  ('st_other', 0x00000000)
// CHECK-NEXT:  ('st_shndx', 0x00000003)
// CHECK-NEXT:  ('st_value', 0x0000000000000000)
// CHECK-NEXT:  ('st_size', 0x0000000000000000)
// CHECK-NEXT: ),
// CHECK-NEXT: # Symbol 0x00000008
// CHECK-NEXT: (('st_name', 0x00000000) # ''
// CHECK-NEXT:  ('st_bind', 0x00000000)
// CHECK-NEXT:  ('st_type', 0x00000003)
// CHECK-NEXT:  ('st_other', 0x00000000)
// CHECK-NEXT:  ('st_shndx', 0x00000004)
// CHECK-NEXT:  ('st_value', 0x0000000000000000)
// CHECK-NEXT:  ('st_size', 0x0000000000000000)
// CHECK-NEXT: ),
// CHECK-NEXT: # Symbol 0x00000009
// CHECK-NEXT: (('st_name', 0x0000004a) # 'g1@@zed'
// CHECK-NEXT:  ('st_bind', 0x00000001)
// CHECK-NEXT:  ('st_type', 0x00000000)
// CHECK-NEXT:  ('st_other', 0x00000000)
// CHECK-NEXT:  ('st_shndx', 0x00000001)
// CHECK-NEXT:  ('st_value', 0x0000000000000014)
// CHECK-NEXT:  ('st_size', 0x0000000000000000)
// CHECK-NEXT: ),
// CHECK-NEXT: # Symbol 0x0000000a
// CHECK-NEXT: (('st_name', 0x00000042) # 'global1'
// CHECK-NEXT:  ('st_bind', 0x00000001)
// CHECK-NEXT:  ('st_type', 0x00000000)
// CHECK-NEXT:  ('st_other', 0x00000000)
// CHECK-NEXT:  ('st_shndx', 0x00000001)
// CHECK-NEXT:  ('st_value', 0x0000000000000014)
// CHECK-NEXT:  ('st_size', 0x0000000000000000)
// CHECK-NEXT: ),
// CHECK-NEXT: # Symbol 0x0000000b
// CHECK-NEXT: (('st_name', 0x0000001c) # 'bar2@zed'
// CHECK-NEXT:  ('st_bind', 0x00000001)
// CHECK-NEXT:  ('st_type', 0x00000000)
// CHECK-NEXT:  ('st_other', 0x00000000)
// CHECK-NEXT:  ('st_shndx', 0x00000000)
// CHECK-NEXT:  ('st_value', 0x0000000000000000)
// CHECK-NEXT:  ('st_size', 0x0000000000000000)
// CHECK-NEXT: ),
// CHECK-NEXT: # Symbol 0x0000000c
// CHECK-NEXT: (('st_name', 0x00000039) # 'bar6@zed'
// CHECK-NEXT:  ('st_bind', 0x00000001)
// CHECK-NEXT:  ('st_type', 0x00000000)
// CHECK-NEXT:  ('st_other', 0x00000000)
// CHECK-NEXT:  ('st_shndx', 0x00000000)
// CHECK-NEXT:  ('st_value', 0x0000000000000000)
// CHECK-NEXT:  ('st_size', 0x0000000000000000)
// CHECK-NEXT: ),
// CHECK-NEXT:])
