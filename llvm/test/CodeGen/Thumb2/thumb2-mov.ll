; RUN: llc < %s -march=thumb -mattr=+thumb2 | FileCheck %s

; Test #<const>

; var 2.1 - 0x00ab00ab
define i32 @t2_const_var2_1_ok_1(i32 %lhs) {
;CHECK: t2_const_var2_1_ok_1:
;CHECK: #11206827
    %ret = add i32 %lhs, 11206827 ; 0x00ab00ab
    ret i32 %ret
}

define i32 @t2_const_var2_1_fail_1(i32 %lhs) {
;CHECK: t2_const_var2_1_fail_1:
;CHECK: movt
    %ret = add i32 %lhs, 11206843 ; 0x00ab00bb
    ret i32 %ret
}

define i32 @t2_const_var2_1_fail_2(i32 %lhs) {
;CHECK: t2_const_var2_1_fail_2:
;CHECK: movt
    %ret = add i32 %lhs, 27984043 ; 0x01ab00ab
    ret i32 %ret
}

define i32 @t2_const_var2_1_fail_3(i32 %lhs) {
;CHECK: t2_const_var2_1_fail_3:
;CHECK: movt
    %ret = add i32 %lhs, 27984299 ; 0x01ab01ab
    ret i32 %ret
}

define i32 @t2_const_var2_1_fail_4(i32 %lhs) {
;CHECK: t2_const_var2_1_fail_4:
;CHECK: movt
    %ret = add i32 %lhs, 28027649 ; 0x01abab01
    ret i32 %ret
}

; var 2.2 - 0xab00ab00
define i32 @t2_const_var2_2_ok_1(i32 %lhs) {
;CHECK: t2_const_var2_2_ok_1:
;CHECK: #-1426019584
    %ret = add i32 %lhs, 2868947712 ; 0xab00ab00
    ret i32 %ret
}

define i32 @t2_const_var2_2_fail_1(i32 %lhs) {
;CHECK: t2_const_var2_2_fail_1:
;CHECK: movt
    %ret = add i32 %lhs, 2868951552 ; 0xab00ba00
    ret i32 %ret
}

define i32 @t2_const_var2_2_fail_2(i32 %lhs) {
;CHECK: t2_const_var2_2_fail_2:
;CHECK: movt
    %ret = add i32 %lhs, 2868947728 ; 0xab00ab10
    ret i32 %ret
}

define i32 @t2_const_var2_2_fail_3(i32 %lhs) {
;CHECK: t2_const_var2_2_fail_3:
;CHECK: movt
    %ret = add i32 %lhs, 2869996304 ; 0xab10ab10
    ret i32 %ret
}

define i32 @t2_const_var2_2_fail_4(i32 %lhs) {
;CHECK: t2_const_var2_2_fail_4:
;CHECK: movt
    %ret = add i32 %lhs, 279685904 ; 0x10abab10
    ret i32 %ret
}

; var 2.3 - 0xabababab
define i32 @t2_const_var2_3_ok_1(i32 %lhs) {
;CHECK: t2_const_var2_3_ok_1:
;CHECK: #-1414812757
    %ret = add i32 %lhs, 2880154539 ; 0xabababab
    ret i32 %ret
}

define i32 @t2_const_var2_3_fail_1(i32 %lhs) {
;CHECK: t2_const_var2_3_fail_1:
;CHECK: movt
    %ret = add i32 %lhs, 2880154554 ; 0xabababba
    ret i32 %ret
}

define i32 @t2_const_var2_3_fail_2(i32 %lhs) {
;CHECK: t2_const_var2_3_fail_2:
;CHECK: movt
    %ret = add i32 %lhs, 2880158379 ; 0xababbaab
    ret i32 %ret
}

define i32 @t2_const_var2_3_fail_3(i32 %lhs) {
;CHECK: t2_const_var2_3_fail_3:
;CHECK: movt
    %ret = add i32 %lhs, 2881137579 ; 0xabbaabab
    ret i32 %ret
}

define i32 @t2_const_var2_3_fail_4(i32 %lhs) {
;CHECK: t2_const_var2_3_fail_4:
;CHECK: movt
    %ret = add i32 %lhs, 3131812779 ; 0xbaababab
    ret i32 %ret
}

; var 3 - 0x0F000000
define i32 @t2_const_var3_1_ok_1(i32 %lhs) {
;CHECK: t2_const_var3_1_ok_1:
;CHECK: #251658240
    %ret = add i32 %lhs, 251658240 ; 0x0F000000
    ret i32 %ret
}

define i32 @t2_const_var3_2_ok_1(i32 %lhs) {
;CHECK: t2_const_var3_2_ok_1:
;CHECK: #3948544
    %ret = add i32 %lhs, 3948544 ; 0b00000000001111000100000000000000
    ret i32 %ret
}

define i32 @t2_const_var3_2_fail_1(i32 %lhs) {
;CHECK: t2_const_var3_2_fail_1:
;CHECK: movt
    %ret = add i32 %lhs, 3940352 ; 0b00000000001111000010000000000000
    ret i32 %ret
}

define i32 @t2_const_var3_3_ok_1(i32 %lhs) {
;CHECK: t2_const_var3_3_ok_1:
;CHECK: #258
    %ret = add i32 %lhs, 258 ; 0b00000000000000000000000100000010
    ret i32 %ret
}

define i32 @t2_const_var3_4_ok_1(i32 %lhs) {
;CHECK: t2_const_var3_4_ok_1:
;CHECK: #-268435456
    %ret = add i32 %lhs, 4026531840 ; 0xF0000000
    ret i32 %ret
}
