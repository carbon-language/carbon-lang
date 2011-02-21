// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | elf-dump  --dump-section-data | FileCheck  %s

// Test that we produce the correct relocation.

	loope	0                 # R_X86_64_PC8
	jmp	-256              # R_X86_64_PC32

// CHECK:      # Section 0x00000007
// CHECK-NEXT: (('sh_name', 0x0000002c) # '.rela.text'
// CHECK-NEXT:  ('sh_type', 0x00000004)
// CHECK-NEXT:  ('sh_flags', 0x00000000)
// CHECK-NEXT:  ('sh_addr', 0x00000000)
// CHECK-NEXT:  ('sh_offset', 0x000000e8)
// CHECK-NEXT:  ('sh_size', 0x00000030)
// CHECK-NEXT:  ('sh_link', 0x00000005)
// CHECK-NEXT:  ('sh_info', 0x00000001)
// CHECK-NEXT:  ('sh_addralign', 0x00000008)
// CHECK-NEXT:  ('sh_entsize', 0x00000018)
// CHECK-NEXT:  ('_relocations', [
// CHECK-NEXT:   # Relocation 0x00000000
// CHECK-NEXT:   (('r_offset', 0x00000001)
// CHECK-NEXT:    ('r_sym', 0x00000000)
// CHECK-NEXT:    ('r_type', 0x0000000f)
// CHECK-NEXT:    ('r_addend', 0x00000000)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 0x00000001
// CHECK-NEXT:   (('r_offset', 0x00000003)
// CHECK-NEXT:    ('r_sym', 0x00000000)
// CHECK-NEXT:    ('r_type', 0x00000002)
// CHECK-NEXT:    ('r_addend', 0x00000000)
// CHECK-NEXT:   ),
// CHECK-NEXT:  ])
// CHECK-NEXT: ),
