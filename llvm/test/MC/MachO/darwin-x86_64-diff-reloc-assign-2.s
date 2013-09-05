// RUN: llvm-mc -triple x86_64-apple-darwin9 %s -filetype=obj -o - | macho-dump --dump-section-data | FileCheck %s

// Test case for rdar://9356266

// This tests that this expression does not cause a crash and produces these
// four relocation entries:
// Relocation information (__DATA,__data) 4 entries
// address  pcrel length extern type    scattered symbolnum/value
// 00000004 False long   False  SUB     False     2 (__DATA,__data)
// 00000004 False long   False  UNSIGND False     2 (__DATA,__data)
// 00000000 False long   False  SUB     False     2 (__DATA,__data)
// 00000000 False long   False  UNSIGND False     2 (__DATA,__data)

	.data
L_var1:
L_var2:
// This was working fine
	.long L_var2 - L_var1
	
	.set L_var3, .
	.set L_var4, .
// But this was causing a crash
	.long L_var4 - L_var3

// CHECK:  ('_relocations', [
// CHECK:    # Relocation 0
// CHECK:    (('word-0', 0x4),
// CHECK:     ('word-1', 0x54000002)),
// CHECK:    # Relocation 1
// CHECK:    (('word-0', 0x4),
// CHECK:     ('word-1', 0x4000002)),
// CHECK:    # Relocation 2
// CHECK:    (('word-0', 0x0),
// CHECK:     ('word-1', 0x54000002)),
// CHECK:    # Relocation 3
// CHECK:    (('word-0', 0x0),
// CHECK:     ('word-1', 0x4000002)),
// CHECK:  ])
