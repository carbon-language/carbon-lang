// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | elf-dump  --dump-section-data | FileCheck %s

f:
	.cfi_startproc
	subq	$8, %rsp
	.cfi_def_cfa_offset 16
        nop
	addq	$8, %rsp
	.cfi_def_cfa_offset 8
	ret
	.cfi_endproc

// FIXME: This is a correct but really inefficient coding since
// we use a CFA_advance_loc4 for every address change!

// CHECK:      # Section 0x00000004
// CHECK-NEXT: (('sh_name', 0x00000012) # '.eh_frame'
// CHECK-NEXT:  ('sh_type', 0x00000001)
// CHECK-NEXT:  ('sh_flags', 0x00000002)
// CHECK-NEXT:  ('sh_addr', 0x00000000)
// CHECK-NEXT:  ('sh_offset', 0x00000050)
// CHECK-NEXT:  ('sh_size', 0x00000038)
// CHECK-NEXT:  ('sh_link', 0x00000000)
// CHECK-NEXT:  ('sh_info', 0x00000000)
// CHECK-NEXT:  ('sh_addralign', 0x00000008)
// CHECK-NEXT:  ('sh_entsize', 0x00000000)
// CHECK-NEXT:  ('_section_data', '14000000 00000000 017a5200 01781001 1b0c0708 90010000 1c000000 1c000000 00000000 0a000000 00040400 00000e10 04050000 000e0800')
// CHECK-NEXT: ),

// CHECK:       # Section 0x00000008
// CHECK-NEXT: (('sh_name', 0x00000036) # '.rela.eh_frame'
// CHECK-NEXT:  ('sh_type', 0x00000004)
// CHECK-NEXT:  ('sh_flags', 0x00000000)
// CHECK-NEXT:  ('sh_addr', 0x00000000)
// CHECK-NEXT:  ('sh_offset', 0x00000168)
// CHECK-NEXT:  ('sh_size', 0x00000018)
// CHECK-NEXT:  ('sh_link', 0x00000006)
// CHECK-NEXT:  ('sh_info', 0x00000004)
// CHECK-NEXT:  ('sh_addralign', 0x00000008)
// CHECK-NEXT:  ('sh_entsize', 0x00000018)
// CHECK-NEXT:  ('_relocations', [
// CHECK-NEXT:   # Relocation 0x00000000
// CHECK-NEXT:   (('r_offset', 0x00000020)
// CHECK-NEXT:    ('r_sym', 0x00000002)
// CHECK-NEXT:    ('r_type', 0x00000002)
// CHECK-NEXT:    ('r_addend', 0x00000000)
// CHECK-NEXT:   ),
// CHECK-NEXT:  ])
// CHECK-NEXT: ),
