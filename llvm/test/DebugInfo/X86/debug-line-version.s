// RUN: llvm-mc -g -dwarf-version 2 -triple  i686-pc-linux-gnu %s -filetype=obj -o - | llvm-dwarfdump - | FileCheck --check-prefix=DWARF2 %s
// RUN: llvm-mc -g -dwarf-version 3 -triple  i686-pc-linux-gnu %s -filetype=obj -o -  |llvm-dwarfdump - | FileCheck --check-prefix=DWARF3 %s
// RUN: llvm-mc -g -dwarf-version 4 -triple  i686-pc-linux-gnu %s -filetype=obj -o - | llvm-dwarfdump - | FileCheck  --check-prefix=DWARF4 %s


// Check that we generate debug_line version that matches the dwarf version.
// For DWARF-4, additionally check that maximum_operations_per_instruction
// field is emitted.

    .text
    .globl foo
    .type foo, @function
    .align 4
foo:
    ret
    .size foo, .-foo

// DWARF2: Compile Unit: length = {{.*}} version = 0x0002
// DWARF2: Line table prologue:
// DWARF2-NEXT:    total_length: 0x00000077
// DWARF2-NEXT:         version: 2
// DWARF2-NEXT: prologue_length: 0x00000062
// DWARF2-NOT: max_ops_per_inst: 1

// DWARF3: Compile Unit: length = {{.*}} version = 0x0003
// DWARF3: Line table prologue:
// DWARF3-NEXT:    total_length: 0x00000077
// DWARF3-NEXT:         version: 3
// DWARF3-NEXT: prologue_length: 0x00000062
// DWARF3-NOT: max_ops_per_inst: 1


// DWARF4: Compile Unit: length = {{.*}} version = 0x0004
// DWARF4: Line table prologue:
// DWARF4-NEXT:    total_length: 0x00000078
// DWARF4-NEXT:         version: 4
// DWARF4-NEXT: prologue_length: 0x00000063
// DWARF4: max_ops_per_inst: 1

