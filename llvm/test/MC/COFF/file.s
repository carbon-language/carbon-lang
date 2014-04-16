// RUN: llvm-mc -triple i686-windows -filetype obj %s -o - | llvm-objdump -t - \
// RUN:   | FileCheck %s -check-prefix CHECK-PRINT

	.file "null-padded.asm"
// CHECK-PRINT: .file
// CHECK-PRINT-NEXT: AUX null-padded.asm{{$}}

	.file "eighteen-chars.asm"

// CHECK-PRINT: .file
// CHECK-PRINT-NEXT: AUX eighteen-chars.asm{{$}}

	.file "multiple-auxiliary-entries.asm"

// CHECK-PRINT: .file
// CHECK-PRINT-NEXT: AUX multiple-auxiliary-entries.asm{{$}}

