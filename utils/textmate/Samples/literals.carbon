// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// --- string

'C'
"Hello\nWorld\n"
"\"\'\t\r\n"
'''Single line string'''
''' Multi line
    string
'''

// --- true-false

true false

// --- type-literals

array bool char str type ;
i8 i16 i32 i64 i128 i10
u8 u16 u32 u64 u128 u11
f8 f16 f32 f64 f128 f12

// --- numbers

12345 6789 00012345

1234.56789
0.875
0.3e5
1e5
.123
.1e3

0b1011_0010
0b_010
0b2345
0B011

0x2AE5
0x_F109
0xD8_E2.5F_7Bp8
0xabcdef.abcdef
0XA85CE
