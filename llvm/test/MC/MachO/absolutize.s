// RUN: llvm-mc -triple i386-apple-darwin9 %s -filetype=obj -o - | macho-dump | FileCheck %s

// CHECK: # Relocation 0
// CHECK: (('word-0', 0xa0000028),
// CHECK:  ('word-1', 0x2b)),
// CHECK: # Relocation 1
// CHECK: (('word-0', 0xa4000020),
// CHECK:  ('word-1', 0x37)),
// CHECK: # Relocation 2
// CHECK: (('word-0', 0xa1000000),
// CHECK:  ('word-1', 0x33)),
// CHECK: # Relocation 3
// CHECK: (('word-0', 0xa4000018),
// CHECK:  ('word-1', 0x33)),
// CHECK: # Relocation 4
// CHECK: (('word-0', 0xa1000000),
// CHECK:  ('word-1', 0x2f)),
// CHECK: # Relocation 5
// CHECK: (('word-0', 0xa4000010),
// CHECK:  ('word-1', 0x2b)),
// CHECK: # Relocation 6
// CHECK: (('word-0', 0xa1000000),
// CHECK:  ('word-1', 0x2f)),
// CHECK-NEXT: ])

_text_a:
        xorl %eax,%eax
_text_b:
        xorl %eax,%eax
Ltext_c:
        xorl %eax,%eax
Ltext_d:        
        xorl %eax,%eax
        
        movl $(_text_a - _text_b), %eax
Ltext_expr_0 = _text_a - _text_b
        movl $(Ltext_expr_0), %eax

        movl $(Ltext_c - _text_b), %eax
Ltext_expr_1 = Ltext_c - _text_b
        movl $(Ltext_expr_1), %eax

        movl $(Ltext_d - Ltext_c), %eax
Ltext_expr_2 = Ltext_d - Ltext_c
        movl $(Ltext_expr_2), %eax

        movl $(_text_a + Ltext_expr_0), %eax

        .data
_data_a:
        .long 0
_data_b:
        .long 0
Ldata_c:
        .long 0
Ldata_d:        
        .long 0
        
        .long _data_a - _data_b
Ldata_expr_0 = _data_a - _data_b
        .long Ldata_expr_0

        .long Ldata_c - _data_b
Ldata_expr_1 = Ldata_c - _data_b
        .long Ldata_expr_1

        .long Ldata_d - Ldata_c
Ldata_expr_2 = Ldata_d - Ldata_c
        .long Ldata_expr_2

        .long _data_a + Ldata_expr_0
