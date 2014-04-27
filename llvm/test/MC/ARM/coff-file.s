// RUN: llvm-mc -triple thumbv7-windows -filetype obj %s -o - | llvm-objdump -t - \
// RUN:   | FileCheck %s

	.file "null-padded.asm"
// CHECK: (nx 1) {{0x[0-9]+}} .file
// CHECK-NEXT: AUX null-padded.asm{{$}}

	.file "eighteen-chars.asm"

// CHECK: (nx 1) {{0x[0-9]+}} .file
// CHECK-NEXT: AUX eighteen-chars.asm{{$}}

	.file "multiple-auxiliary-entries.asm"

// CHECK: (nx 2) {{0x[0-9]+}} .file
// CHECK-NEXT: AUX multiple-auxiliary-entries.asm{{$}}

