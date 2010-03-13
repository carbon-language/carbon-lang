// RUN: llvm-mc -n -triple i386-apple-darwin9 %s -filetype=obj -o - | macho-dump --dump-section-data | FileCheck %s

// CHECK: # Relocation 0
// CHECK: (('word-0', 0x1),
// CHECK: ('word-1', 0x5000002)),
// CHECK-NEXT: ])
// CHECK: ('_section_data', '\xe8\xfb\xff\xff\xff')

        .data
        .long 0

        .text
_a:
        call _a
