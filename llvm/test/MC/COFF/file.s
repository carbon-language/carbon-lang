// RUN: llvm-mc -triple i686-windows -filetype obj %s -o - | llvm-objdump -t - \
// RUN:   | FileCheck %s -check-prefix CHECK-PRINT

	.file "null-padded.asm"
// CHECK-PRINT: (nx 1) {{0x[0-9]+}} .file
// CHECK-PRINT-NEXT: AUX null-padded.asm{{$}}

	.file "eighteen-chars.asm"

// CHECK-PRINT: (nx 1) {{0x[0-9]+}} .file
// CHECK-PRINT-NEXT: AUX eighteen-chars.asm{{$}}

	.file "multiple-auxiliary-entries.asm"

// CHECK-PRINT: (nx 2) {{0x[0-9]+}} .file
// CHECK-PRINT-NEXT: AUX multiple-auxiliary-entries.asm{{$}}

