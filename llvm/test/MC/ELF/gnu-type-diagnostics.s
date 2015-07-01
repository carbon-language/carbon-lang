// RUN: not llvm-mc -triple i686-elf -filetype asm -o /dev/null %s 2>&1 | FileCheck %s
// RUN: not llvm-mc -triple aarch64-elf -filetype asm -o /dev/null %s 2>&1 | FileCheck %s
// RUN: not llvm-mc -triple arm-elf -filetype asm -o /dev/null %s 2>&1 \
// RUN:     | FileCheck -check-prefix=CHECK-INVALID-AT-IN-TYPE-DIRECTIVE %s
// RUN: not llvm-mc -triple armeb-elf -filetype asm -o /dev/null %s 2>&1 \
// RUN:     | FileCheck -check-prefix=CHECK-INVALID-AT-IN-TYPE-DIRECTIVE %s
// RUN: not llvm-mc -triple thumb-elf -filetype asm -o /dev/null %s 2>&1 \
// RUN:     | FileCheck -check-prefix=CHECK-INVALID-AT-IN-TYPE-DIRECTIVE %s
// RUN: not llvm-mc -triple thumbeb-elf -filetype asm -o /dev/null %s 2>&1 \
// RUN:     | FileCheck -check-prefix=CHECK-INVALID-AT-IN-TYPE-DIRECTIVE %s
// RUN: not llvm-mc -triple arm-coff -filetype asm -o /dev/null %s 2>&1 \
// RUN:     | FileCheck -check-prefix=CHECK-INVALID-AT-IN-TYPE-DIRECTIVE %s
// RUN: not llvm-mc -triple armeb-coff -filetype asm -o /dev/null %s 2>&1 \
// RUN:     | FileCheck -check-prefix=CHECK-INVALID-AT-IN-TYPE-DIRECTIVE %s
// RUN: not llvm-mc -triple thumb-coff -filetype asm -o /dev/null %s 2>&1 \
// RUN:     | FileCheck -check-prefix=CHECK-INVALID-AT-IN-TYPE-DIRECTIVE %s
// RUN: not llvm-mc -triple thumbeb-coff -filetype asm -o /dev/null %s 2>&1 \
// RUN:     | FileCheck -check-prefix=CHECK-INVALID-AT-IN-TYPE-DIRECTIVE %s
// RUN: not llvm-mc -triple arm-apple -filetype asm -o /dev/null %s 2>&1 \
// RUN:     | FileCheck -check-prefix=CHECK-INVALID-AT-IN-TYPE-DIRECTIVE %s
// RUN: not llvm-mc -triple armeb-apple -filetype asm -o /dev/null %s 2>&1 \
// RUN:     | FileCheck -check-prefix=CHECK-INVALID-AT-IN-TYPE-DIRECTIVE %s
// RUN: not llvm-mc -triple thumb-apple -filetype asm -o /dev/null %s 2>&1 \
// RUN:     | FileCheck -check-prefix=CHECK-INVALID-AT-IN-TYPE-DIRECTIVE %s
// RUN: not llvm-mc -triple thumbeb-apple -filetype asm -o /dev/null %s 2>&1 \
// RUN:     | FileCheck -check-prefix=CHECK-INVALID-AT-IN-TYPE-DIRECTIVE %s

	.type TYPE FUNC
// CHECK: error: unsupported attribute in '.type' directive
// CHECK: .type TYPE FUNC
// CHECK:            ^

	.type type stt_func
// CHECK: error: unsupported attribute in '.type' directive
// CHECK: .type type stt_func
// CHECK:            ^

	.type symbol 32
// CHECK: error: expected STT_<TYPE_IN_UPPER_CASE>, '#<type>', '@<type>', '%<type>' or "<type>"
// CHECK: .type symbol 32
// CHECK:              ^

// CHECK-INVALID-AT-IN-TYPE-DIRECTIVE: error: expected STT_<TYPE_IN_UPPER_CASE>, '#<type>', '%<type>' or "<type>"
// CHECK-INVALID-AT-IN-TYPE-DIRECTIVE: .type symbol 32
// CHECK-INVALID-AT-IN-TYPE-DIRECTIVE:              ^

