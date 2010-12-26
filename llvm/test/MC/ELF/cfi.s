// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | elf-dump  --dump-section-data | FileCheck %s

f1:
        .cfi_startproc
        nop
        .cfi_endproc

f2:
        .cfi_startproc
        .cfi_personality 0x00, foo
        nop
        .cfi_endproc

f3:
        .cfi_startproc
        nop
        .cfi_endproc

f4:
        .cfi_startproc
        .cfi_personality 0x00, foo
        nop
        .cfi_endproc

// CHECK:      # Section 0x00000004
// CHECK-NEXT: (('sh_name', 0x00000012) # '.eh_frame'
// CHECK-NEXT:  ('sh_type', 0x00000001)
// CHECK-NEXT:  ('sh_flags', 0x00000002)
// CHECK-NEXT:  ('sh_addr', 0x00000000)
// CHECK-NEXT:  ('sh_offset', 0x00000048)
// CHECK-NEXT:  ('sh_size', 0x00000088)
// CHECK-NEXT:  ('sh_link', 0x00000000)
// CHECK-NEXT:  ('sh_info', 0x00000000)
// CHECK-NEXT:  ('sh_addralign', 0x00000008)
// CHECK-NEXT:  ('sh_entsize', 0x00000000)
// CHECK-NEXT:  ('_section_data', '14000000 00000000 017a5200 01781001 1b0c0708 90010000 10000000 1c000000 00000000 01000000 00000000 1c000000 00000000 017a5052 00017810 0a000000 00000000 00001b0c 07089001 10000000 24000000 00000000 01000000 00000000 10000000 64000000 00000000 01000000 00000000 10000000 4c000000 00000000 01000000 00000000')
// CHECK-NEXT: ),

// CHECK:      # Section 0x00000008
// CHECK-NEXT: (('sh_name', 0x00000036) # '.rela.eh_frame'
// CHECK-NEXT:  ('sh_type', 0x00000004)
// CHECK-NEXT:  ('sh_flags', 0x00000000)
// CHECK-NEXT:  ('sh_addr', 0x00000000)
// CHECK-NEXT:  ('sh_offset', 0x00000220)
// CHECK-NEXT:  ('sh_size', 0x00000078)
// CHECK-NEXT:  ('sh_link', 0x00000006)
// CHECK-NEXT:  ('sh_info', 0x00000004)
// CHECK-NEXT:  ('sh_addralign', 0x00000008)
// CHECK-NEXT:  ('sh_entsize', 0x00000018)
// CHECK-NEXT:  ('_relocations', [
// CHECK-NEXT:   # Relocation 0x00000000
// CHECK-NEXT:   (('r_offset', 0x00000020)
// CHECK-NEXT:    ('r_sym', 0x00000005)
// CHECK-NEXT:    ('r_type', 0x00000002)
// CHECK-NEXT:    ('r_addend', 0x00000000)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 0x00000001
// CHECK-NEXT:   (('r_offset', 0x0000003e)
// CHECK-NEXT:    ('r_sym', 0x00000009)
// CHECK-NEXT:    ('r_type', 0x00000001)
// CHECK-NEXT:    ('r_addend', 0x00000000)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 0x00000002
// CHECK-NEXT:   (('r_offset', 0x00000054)
// CHECK-NEXT:    ('r_sym', 0x00000005)
// CHECK-NEXT:    ('r_type', 0x00000002)
// CHECK-NEXT:    ('r_addend', 0x00000001)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 0x00000003
// CHECK-NEXT:   (('r_offset', 0x00000068)
// CHECK-NEXT:    ('r_sym', 0x00000005)
// CHECK-NEXT:    ('r_type', 0x00000002)
// CHECK-NEXT:    ('r_addend', 0x00000002)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 0x00000004
// CHECK-NEXT:   (('r_offset', 0x0000007c)
// CHECK-NEXT:    ('r_sym', 0x00000005)
// CHECK-NEXT:    ('r_type', 0x00000002)
// CHECK-NEXT:    ('r_addend', 0x00000003)
// CHECK-NEXT:   ),
// CHECK-NEXT:  ])
// CHECK-NEXT: ),
