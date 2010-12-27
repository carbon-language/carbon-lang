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

f5:
        .cfi_startproc
        .cfi_personality 0x02, foo
        nop
        .cfi_endproc

f6:
        .cfi_startproc
        .cfi_personality 0x03, foo
        nop
        .cfi_endproc

f7:
        .cfi_startproc
        .cfi_personality 0x04, foo
        nop
        .cfi_endproc

f8:
        .cfi_startproc
        .cfi_personality 0x0a, foo
        nop
        .cfi_endproc

f9:
        .cfi_startproc
        .cfi_personality 0x0b, foo
        nop
        .cfi_endproc

f10:
        .cfi_startproc
        .cfi_personality 0x0c, foo
        nop
        .cfi_endproc

f11:
        .cfi_startproc
        .cfi_personality 0x08, foo
        nop
        .cfi_endproc

f12:
        .cfi_startproc
        .cfi_personality 0x10, foo
        nop
        .cfi_endproc

f13:
        .cfi_startproc
        .cfi_personality 0x12, foo
        nop
        .cfi_endproc

f14:
        .cfi_startproc
        .cfi_personality 0x13, foo
        nop
        .cfi_endproc

f15:
        .cfi_startproc
        .cfi_personality 0x14, foo
        nop
        .cfi_endproc

f16:
        .cfi_startproc
        .cfi_personality 0x1a, foo
        nop
        .cfi_endproc

f17:
        .cfi_startproc
        .cfi_personality 0x1b, foo
        nop
        .cfi_endproc

f18:
        .cfi_startproc
        .cfi_personality 0x1c, foo
        nop
        .cfi_endproc

f19:
        .cfi_startproc
        .cfi_personality 0x18, foo
        nop
        .cfi_endproc

f20:
        .cfi_startproc
        .cfi_personality 0x80, foo
        nop
        .cfi_endproc

f21:
        .cfi_startproc
        .cfi_personality 0x82, foo
        nop
        .cfi_endproc

f22:
        .cfi_startproc
        .cfi_personality 0x83, foo
        nop
        .cfi_endproc

f23:
        .cfi_startproc
        .cfi_personality 0x84, foo
        nop
        .cfi_endproc

f24:
        .cfi_startproc
        .cfi_personality 0x8a, foo
        nop
        .cfi_endproc

f25:
        .cfi_startproc
        .cfi_personality 0x8b, foo
        nop
        .cfi_endproc

f26:
        .cfi_startproc
        .cfi_personality 0x8c, foo
        nop
        .cfi_endproc

f27:
        .cfi_startproc
        .cfi_personality 0x88, foo
        nop
        .cfi_endproc
// CHECK:      # Section 0x00000004
// CHECK-NEXT: (('sh_name', 0x00000012) # '.eh_frame'
// CHECK-NEXT:  ('sh_type', 0x00000001)
// CHECK-NEXT:  ('sh_flags', 0x00000002)
// CHECK-NEXT:  ('sh_addr', 0x00000000)
// CHECK-NEXT:  ('sh_offset', 0x00000060)
// CHECK-NEXT:  ('sh_size', 0x00000508)
// CHECK-NEXT:  ('sh_link', 0x00000000)
// CHECK-NEXT:  ('sh_info', 0x00000000)
// CHECK-NEXT:  ('sh_addralign', 0x00000008)
// CHECK-NEXT:  ('sh_entsize', 0x00000000)
// CHECK-NEXT:  ('_section_data', '14000000 00000000 017a5200 01781001 1b0c0708 90010000 10000000 1c000000 00000000 01000000 00000000 1c000000 00000000 017a5052 00017810 0a000000 00000000 00001b0c 07089001 10000000 24000000 00000000 01000000 00000000 10000000 64000000 00000000 01000000 00000000 10000000 4c000000 00000000 01000000 00000000 18000000 00000000 017a5052 00017810 04020000 1b0c0708 90010000 10000000 20000000 00000000 01000000 00000000 18000000 00000000 017a5052 00017810 06030000 00001b0c 07089001 10000000 20000000 00000000 01000000 00000000 1c000000 00000000 017a5052 00017810 0a040000 00000000 00001b0c 07089001 10000000 24000000 00000000 01000000 00000000 18000000 00000000 017a5052 00017810 040a0000 1b0c0708 90010000 10000000 20000000 00000000 01000000 00000000 18000000 00000000 017a5052 00017810 060b0000 00001b0c 07089001 10000000 20000000 00000000 01000000 00000000 1c000000 00000000 017a5052 00017810 0a0c0000 00000000 00001b0c 07089001 10000000 24000000 00000000 01000000 00000000 1c000000 00000000 017a5052 00017810 0a080000 00000000 00001b0c 07089001 10000000 24000000 00000000 01000000 00000000 1c000000 00000000 017a5052 00017810 0a100000 00000000 00001b0c 07089001 10000000 24000000 00000000 01000000 00000000 18000000 00000000 017a5052 00017810 04120000 1b0c0708 90010000 10000000 20000000 00000000 01000000 00000000 18000000 00000000 017a5052 00017810 06130000 00001b0c 07089001 10000000 20000000 00000000 01000000 00000000 1c000000 00000000 017a5052 00017810 0a140000 00000000 00001b0c 07089001 10000000 24000000 00000000 01000000 00000000 18000000 00000000 017a5052 00017810 041a0000 1b0c0708 90010000 10000000 20000000 00000000 01000000 00000000 18000000 00000000 017a5052 00017810 061b0000 00001b0c 07089001 10000000 20000000 00000000 01000000 00000000 1c000000 00000000 017a5052 00017810 0a1c0000 00000000 00001b0c 07089001 10000000 24000000 00000000 01000000 00000000 1c000000 00000000 017a5052 00017810 0a180000 00000000 00001b0c 07089001 10000000 24000000 00000000 01000000 00000000 1c000000 00000000 017a5052 00017810 0a800000 00000000 00001b0c 07089001 10000000 24000000 00000000 01000000 00000000 18000000 00000000 017a5052 00017810 04820000 1b0c0708 90010000 10000000 20000000 00000000 01000000 00000000 18000000 00000000 017a5052 00017810 06830000 00001b0c 07089001 10000000 20000000 00000000 01000000 00000000 1c000000 00000000 017a5052 00017810 0a840000 00000000 00001b0c 07089001 10000000 24000000 00000000 01000000 00000000 18000000 00000000 017a5052 00017810 048a0000 1b0c0708 90010000 10000000 20000000 00000000 01000000 00000000 18000000 00000000 017a5052 00017810 068b0000 00001b0c 07089001 10000000 20000000 00000000 01000000 00000000 1c000000 00000000 017a5052 00017810 0a8c0000 00000000 00001b0c 07089001 10000000 24000000 00000000 01000000 00000000 1c000000 00000000 017a5052 00017810 0a880000 00000000 00001b0c 07089001 14000000 24000000 00000000 01000000 00000000 00000000')
// CHECK-NEXT: ),

// CHECK:        # Section 0x00000008
// CHECK-NEXT: (('sh_name', 0x00000036) # '.rela.eh_frame'
// CHECK-NEXT:  ('sh_type', 0x00000004)
// CHECK-NEXT:  ('sh_flags', 0x00000000)
// CHECK-NEXT:  ('sh_addr', 0x00000000)
// CHECK-NEXT:  ('sh_offset', 0x00000930)
// CHECK-NEXT:  ('sh_size', 0x000004c8)
// CHECK-NEXT:  ('sh_link', 0x00000006)
// CHECK-NEXT:  ('sh_info', 0x00000004)
// CHECK-NEXT:  ('sh_addralign', 0x00000008)
// CHECK-NEXT:  ('sh_entsize', 0x00000018)
// CHECK-NEXT:  ('_relocations', [
// CHECK-NEXT:   # Relocation 0x00000000
// CHECK-NEXT:   (('r_offset', 0x00000020)
// CHECK-NEXT:    ('r_sym', 0x0000001c)
// CHECK-NEXT:    ('r_type', 0x00000002)
// CHECK-NEXT:    ('r_addend', 0x00000000)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 0x00000001
// CHECK-NEXT:   (('r_offset', 0x0000003e)
// CHECK-NEXT:    ('r_sym', 0x00000020)
// CHECK-NEXT:    ('r_type', 0x00000001)
// CHECK-NEXT:    ('r_addend', 0x00000000)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 0x00000002
// CHECK-NEXT:   (('r_offset', 0x00000054)
// CHECK-NEXT:    ('r_sym', 0x0000001c)
// CHECK-NEXT:    ('r_type', 0x00000002)
// CHECK-NEXT:    ('r_addend', 0x00000001)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 0x00000003
// CHECK-NEXT:   (('r_offset', 0x00000068)
// CHECK-NEXT:    ('r_sym', 0x0000001c)
// CHECK-NEXT:    ('r_type', 0x00000002)
// CHECK-NEXT:    ('r_addend', 0x00000002)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 0x00000004
// CHECK-NEXT:   (('r_offset', 0x0000007c)
// CHECK-NEXT:    ('r_sym', 0x0000001c)
// CHECK-NEXT:    ('r_type', 0x00000002)
// CHECK-NEXT:    ('r_addend', 0x00000003)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 0x00000005
// CHECK-NEXT:   (('r_offset', 0x0000009a)
// CHECK-NEXT:    ('r_sym', 0x00000020)
// CHECK-NEXT:    ('r_type', 0x0000000c)
// CHECK-NEXT:    ('r_addend', 0x00000000)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 0x00000006
// CHECK-NEXT:   (('r_offset', 0x000000ac)
// CHECK-NEXT:    ('r_sym', 0x0000001c)
// CHECK-NEXT:    ('r_type', 0x00000002)
// CHECK-NEXT:    ('r_addend', 0x00000004)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 0x00000007
// CHECK-NEXT:   (('r_offset', 0x000000ca)
// CHECK-NEXT:    ('r_sym', 0x00000020)
// CHECK-NEXT:    ('r_type', 0x0000000a)
// CHECK-NEXT:    ('r_addend', 0x00000000)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 0x00000008
// CHECK-NEXT:   (('r_offset', 0x000000dc)
// CHECK-NEXT:    ('r_sym', 0x0000001c)
// CHECK-NEXT:    ('r_type', 0x00000002)
// CHECK-NEXT:    ('r_addend', 0x00000005)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 0x00000009
// CHECK-NEXT:   (('r_offset', 0x000000fa)
// CHECK-NEXT:    ('r_sym', 0x00000020)
// CHECK-NEXT:    ('r_type', 0x00000001)
// CHECK-NEXT:    ('r_addend', 0x00000000)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 0x0000000a
// CHECK-NEXT:   (('r_offset', 0x00000110)
// CHECK-NEXT:    ('r_sym', 0x0000001c)
// CHECK-NEXT:    ('r_type', 0x00000002)
// CHECK-NEXT:    ('r_addend', 0x00000006)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 0x0000000b
// CHECK-NEXT:   (('r_offset', 0x0000012e)
// CHECK-NEXT:    ('r_sym', 0x00000020)
// CHECK-NEXT:    ('r_type', 0x0000000c)
// CHECK-NEXT:    ('r_addend', 0x00000000)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 0x0000000c
// CHECK-NEXT:   (('r_offset', 0x00000140)
// CHECK-NEXT:    ('r_sym', 0x0000001c)
// CHECK-NEXT:    ('r_type', 0x00000002)
// CHECK-NEXT:    ('r_addend', 0x00000007)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 0x0000000d
// CHECK-NEXT:   (('r_offset', 0x0000015e)
// CHECK-NEXT:    ('r_sym', 0x00000020)
// CHECK-NEXT:    ('r_type', 0x0000000a)
// CHECK-NEXT:    ('r_addend', 0x00000000)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 0x0000000e
// CHECK-NEXT:   (('r_offset', 0x00000170)
// CHECK-NEXT:    ('r_sym', 0x0000001c)
// CHECK-NEXT:    ('r_type', 0x00000002)
// CHECK-NEXT:    ('r_addend', 0x00000008)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 0x0000000f
// CHECK-NEXT:   (('r_offset', 0x0000018e)
// CHECK-NEXT:    ('r_sym', 0x00000020)
// CHECK-NEXT:    ('r_type', 0x00000001)
// CHECK-NEXT:    ('r_addend', 0x00000000)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 0x00000010
// CHECK-NEXT:   (('r_offset', 0x000001a4)
// CHECK-NEXT:    ('r_sym', 0x0000001c)
// CHECK-NEXT:    ('r_type', 0x00000002)
// CHECK-NEXT:    ('r_addend', 0x00000009)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 0x00000011
// CHECK-NEXT:   (('r_offset', 0x000001c2)
// CHECK-NEXT:    ('r_sym', 0x00000020)
// CHECK-NEXT:    ('r_type', 0x00000001)
// CHECK-NEXT:    ('r_addend', 0x00000000)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 0x00000012
// CHECK-NEXT:   (('r_offset', 0x000001d8)
// CHECK-NEXT:    ('r_sym', 0x0000001c)
// CHECK-NEXT:    ('r_type', 0x00000002)
// CHECK-NEXT:    ('r_addend', 0x0000000a)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 0x00000013
// CHECK-NEXT:   (('r_offset', 0x000001f6)
// CHECK-NEXT:    ('r_sym', 0x00000020)
// CHECK-NEXT:    ('r_type', 0x00000018)
// CHECK-NEXT:    ('r_addend', 0x00000000)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 0x00000014
// CHECK-NEXT:   (('r_offset', 0x0000020c)
// CHECK-NEXT:    ('r_sym', 0x0000001c)
// CHECK-NEXT:    ('r_type', 0x00000002)
// CHECK-NEXT:    ('r_addend', 0x0000000b)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 0x00000015
// CHECK-NEXT:   (('r_offset', 0x0000022a)
// CHECK-NEXT:    ('r_sym', 0x00000020)
// CHECK-NEXT:    ('r_type', 0x0000000d)
// CHECK-NEXT:    ('r_addend', 0x00000000)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 0x00000016
// CHECK-NEXT:   (('r_offset', 0x0000023c)
// CHECK-NEXT:    ('r_sym', 0x0000001c)
// CHECK-NEXT:    ('r_type', 0x00000002)
// CHECK-NEXT:    ('r_addend', 0x0000000c)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 0x00000017
// CHECK-NEXT:   (('r_offset', 0x0000025a)
// CHECK-NEXT:    ('r_sym', 0x00000020)
// CHECK-NEXT:    ('r_type', 0x00000002)
// CHECK-NEXT:    ('r_addend', 0x00000000)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 0x00000018
// CHECK-NEXT:   (('r_offset', 0x0000026c)
// CHECK-NEXT:    ('r_sym', 0x0000001c)
// CHECK-NEXT:    ('r_type', 0x00000002)
// CHECK-NEXT:    ('r_addend', 0x0000000d)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 0x00000019
// CHECK-NEXT:   (('r_offset', 0x0000028a)
// CHECK-NEXT:    ('r_sym', 0x00000020)
// CHECK-NEXT:    ('r_type', 0x00000018)
// CHECK-NEXT:    ('r_addend', 0x00000000)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 0x0000001a
// CHECK-NEXT:   (('r_offset', 0x000002a0)
// CHECK-NEXT:    ('r_sym', 0x0000001c)
// CHECK-NEXT:    ('r_type', 0x00000002)
// CHECK-NEXT:    ('r_addend', 0x0000000e)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 0x0000001b
// CHECK-NEXT:   (('r_offset', 0x000002be)
// CHECK-NEXT:    ('r_sym', 0x00000020)
// CHECK-NEXT:    ('r_type', 0x0000000d)
// CHECK-NEXT:    ('r_addend', 0x00000000)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 0x0000001c
// CHECK-NEXT:   (('r_offset', 0x000002d0)
// CHECK-NEXT:    ('r_sym', 0x0000001c)
// CHECK-NEXT:    ('r_type', 0x00000002)
// CHECK-NEXT:    ('r_addend', 0x0000000f)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 0x0000001d
// CHECK-NEXT:   (('r_offset', 0x000002ee)
// CHECK-NEXT:    ('r_sym', 0x00000020)
// CHECK-NEXT:    ('r_type', 0x00000002)
// CHECK-NEXT:    ('r_addend', 0x00000000)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 0x0000001e
// CHECK-NEXT:   (('r_offset', 0x00000300)
// CHECK-NEXT:    ('r_sym', 0x0000001c)
// CHECK-NEXT:    ('r_type', 0x00000002)
// CHECK-NEXT:    ('r_addend', 0x00000010)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 0x0000001f
// CHECK-NEXT:   (('r_offset', 0x0000031e)
// CHECK-NEXT:    ('r_sym', 0x00000020)
// CHECK-NEXT:    ('r_type', 0x00000018)
// CHECK-NEXT:    ('r_addend', 0x00000000)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 0x00000020
// CHECK-NEXT:   (('r_offset', 0x00000334)
// CHECK-NEXT:    ('r_sym', 0x0000001c)
// CHECK-NEXT:    ('r_type', 0x00000002)
// CHECK-NEXT:    ('r_addend', 0x00000011)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 0x00000021
// CHECK-NEXT:   (('r_offset', 0x00000352)
// CHECK-NEXT:    ('r_sym', 0x00000020)
// CHECK-NEXT:    ('r_type', 0x00000018)
// CHECK-NEXT:    ('r_addend', 0x00000000)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 0x00000022
// CHECK-NEXT:   (('r_offset', 0x00000368)
// CHECK-NEXT:    ('r_sym', 0x0000001c)
// CHECK-NEXT:    ('r_type', 0x00000002)
// CHECK-NEXT:    ('r_addend', 0x00000012)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 0x00000023
// CHECK-NEXT:   (('r_offset', 0x00000386)
// CHECK-NEXT:    ('r_sym', 0x00000020)
// CHECK-NEXT:    ('r_type', 0x00000001)
// CHECK-NEXT:    ('r_addend', 0x00000000)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 0x00000024
// CHECK-NEXT:   (('r_offset', 0x0000039c)
// CHECK-NEXT:    ('r_sym', 0x0000001c)
// CHECK-NEXT:    ('r_type', 0x00000002)
// CHECK-NEXT:    ('r_addend', 0x00000013)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 0x00000025
// CHECK-NEXT:   (('r_offset', 0x000003ba)
// CHECK-NEXT:    ('r_sym', 0x00000020)
// CHECK-NEXT:    ('r_type', 0x0000000c)
// CHECK-NEXT:    ('r_addend', 0x00000000)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 0x00000026
// CHECK-NEXT:   (('r_offset', 0x000003cc)
// CHECK-NEXT:    ('r_sym', 0x0000001c)
// CHECK-NEXT:    ('r_type', 0x00000002)
// CHECK-NEXT:    ('r_addend', 0x00000014)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 0x00000027
// CHECK-NEXT:   (('r_offset', 0x000003ea)
// CHECK-NEXT:    ('r_sym', 0x00000020)
// CHECK-NEXT:    ('r_type', 0x0000000a)
// CHECK-NEXT:    ('r_addend', 0x00000000)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 0x00000028
// CHECK-NEXT:   (('r_offset', 0x000003fc)
// CHECK-NEXT:    ('r_sym', 0x0000001c)
// CHECK-NEXT:    ('r_type', 0x00000002)
// CHECK-NEXT:    ('r_addend', 0x00000015)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 0x00000029
// CHECK-NEXT:   (('r_offset', 0x0000041a)
// CHECK-NEXT:    ('r_sym', 0x00000020)
// CHECK-NEXT:    ('r_type', 0x00000001)
// CHECK-NEXT:    ('r_addend', 0x00000000)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 0x0000002a
// CHECK-NEXT:   (('r_offset', 0x00000430)
// CHECK-NEXT:    ('r_sym', 0x0000001c)
// CHECK-NEXT:    ('r_type', 0x00000002)
// CHECK-NEXT:    ('r_addend', 0x00000016)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 0x0000002b
// CHECK-NEXT:   (('r_offset', 0x0000044e)
// CHECK-NEXT:    ('r_sym', 0x00000020)
// CHECK-NEXT:    ('r_type', 0x0000000c)
// CHECK-NEXT:    ('r_addend', 0x00000000)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 0x0000002c
// CHECK-NEXT:   (('r_offset', 0x00000460)
// CHECK-NEXT:    ('r_sym', 0x0000001c)
// CHECK-NEXT:    ('r_type', 0x00000002)
// CHECK-NEXT:    ('r_addend', 0x00000017)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 0x0000002d
// CHECK-NEXT:   (('r_offset', 0x0000047e)
// CHECK-NEXT:    ('r_sym', 0x00000020)
// CHECK-NEXT:    ('r_type', 0x0000000a)
// CHECK-NEXT:    ('r_addend', 0x00000000)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 0x0000002e
// CHECK-NEXT:   (('r_offset', 0x00000490)
// CHECK-NEXT:    ('r_sym', 0x0000001c)
// CHECK-NEXT:    ('r_type', 0x00000002)
// CHECK-NEXT:    ('r_addend', 0x00000018)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 0x0000002f
// CHECK-NEXT:   (('r_offset', 0x000004ae)
// CHECK-NEXT:    ('r_sym', 0x00000020)
// CHECK-NEXT:    ('r_type', 0x00000001)
// CHECK-NEXT:    ('r_addend', 0x00000000)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 0x00000030
// CHECK-NEXT:   (('r_offset', 0x000004c4)
// CHECK-NEXT:    ('r_sym', 0x0000001c)
// CHECK-NEXT:    ('r_type', 0x00000002)
// CHECK-NEXT:    ('r_addend', 0x00000019)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 0x00000031
// CHECK-NEXT:   (('r_offset', 0x000004e2)
// CHECK-NEXT:    ('r_sym', 0x00000020)
// CHECK-NEXT:    ('r_type', 0x00000001)
// CHECK-NEXT:    ('r_addend', 0x00000000)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 0x00000032
// CHECK-NEXT:   (('r_offset', 0x000004f8)
// CHECK-NEXT:    ('r_sym', 0x0000001c)
// CHECK-NEXT:    ('r_type', 0x00000002)
// CHECK-NEXT:    ('r_addend', 0x0000001a)
// CHECK-NEXT:   ),
// CHECK-NEXT:  ])
// CHECK-NEXT: ),
