; RUN: llc < %s -march=thumb -mattr=+thumb2 | grep add | grep #255
; RUN: llc < %s -march=thumb -mattr=+thumb2 | grep add | grep #256
; RUN: llc < %s -march=thumb -mattr=+thumb2 | grep add | grep #257
; RUN: llc < %s -march=thumb -mattr=+thumb2 | grep add | grep #4094
; RUN: llc < %s -march=thumb -mattr=+thumb2 | grep add | grep #4095
; RUN: llc < %s -march=thumb -mattr=+thumb2 | grep add | grep #4096
; RUN: llc < %s -march=thumb -mattr=+thumb2 | grep add
; RUN: llc < %s -march=thumb -mattr=+thumb2 | grep add | grep lsl | grep #8

define i32 @t2ADDrc_255(i32 %lhs) {
    %Rd = add i32 %lhs, 255;
    ret i32 %Rd
}

define i32 @t2ADDrc_256(i32 %lhs) {
    %Rd = add i32 %lhs, 256;
    ret i32 %Rd
}

define i32 @t2ADDrc_257(i32 %lhs) {
    %Rd = add i32 %lhs, 257;
    ret i32 %Rd
}

define i32 @t2ADDrc_4094(i32 %lhs) {
    %Rd = add i32 %lhs, 4094;
    ret i32 %Rd
}

define i32 @t2ADDrc_4095(i32 %lhs) {
    %Rd = add i32 %lhs, 4095;
    ret i32 %Rd
}

define i32 @t2ADDrc_4096(i32 %lhs) {
    %Rd = add i32 %lhs, 4096;
    ret i32 %Rd
}

define i32 @t2ADDrr(i32 %lhs, i32 %rhs) {
    %Rd = add i32 %lhs, %rhs;
    ret i32 %Rd
}

define i32 @t2ADDrs(i32 %lhs, i32 %rhs) {
    %tmp = shl i32 %rhs, 8
    %Rd = add i32 %lhs, %tmp;
    ret i32 %Rd
}

