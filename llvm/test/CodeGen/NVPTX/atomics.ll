; RUN: llc < %s -march=nvptx -mcpu=sm_20 | FileCheck %s


; CHECK: atom0
define i32 @atom0(i32* %addr, i32 %val) {
; CHECK: atom.add.u32
  %ret = atomicrmw add i32* %addr, i32 %val seq_cst
  ret i32 %ret
}

; CHECK: atom1
define i64 @atom1(i64* %addr, i64 %val) {
; CHECK: atom.add.u64
  %ret = atomicrmw add i64* %addr, i64 %val seq_cst
  ret i64 %ret
}

; CHECK: atom2
define i32 @atom2(i32* %subr, i32 %val) {
; CHECK: neg.s32
; CHECK: atom.add.u32
  %ret = atomicrmw sub i32* %subr, i32 %val seq_cst
  ret i32 %ret
}

; CHECK: atom3
define i64 @atom3(i64* %subr, i64 %val) {
; CHECK: neg.s64
; CHECK: atom.add.u64
  %ret = atomicrmw sub i64* %subr, i64 %val seq_cst
  ret i64 %ret
}

; CHECK: atom4
define i32 @atom4(i32* %subr, i32 %val) {
; CHECK: atom.and.b32
  %ret = atomicrmw and i32* %subr, i32 %val seq_cst
  ret i32 %ret
}

; CHECK: atom5
define i64 @atom5(i64* %subr, i64 %val) {
; CHECK: atom.and.b64
  %ret = atomicrmw and i64* %subr, i64 %val seq_cst
  ret i64 %ret
}

;; NAND not yet supported
;define i32 @atom6(i32* %subr, i32 %val) {
;  %ret = atomicrmw nand i32* %subr, i32 %val seq_cst
;  ret i32 %ret
;}

;define i64 @atom7(i64* %subr, i64 %val) {
;  %ret = atomicrmw nand i64* %subr, i64 %val seq_cst
;  ret i64 %ret
;}

; CHECK: atom8
define i32 @atom8(i32* %subr, i32 %val) {
; CHECK: atom.or.b32
  %ret = atomicrmw or i32* %subr, i32 %val seq_cst
  ret i32 %ret
}

; CHECK: atom9
define i64 @atom9(i64* %subr, i64 %val) {
; CHECK: atom.or.b64
  %ret = atomicrmw or i64* %subr, i64 %val seq_cst
  ret i64 %ret
}

; CHECK: atom10
define i32 @atom10(i32* %subr, i32 %val) {
; CHECK: atom.xor.b32
  %ret = atomicrmw xor i32* %subr, i32 %val seq_cst
  ret i32 %ret
}

; CHECK: atom11
define i64 @atom11(i64* %subr, i64 %val) {
; CHECK: atom.xor.b64
  %ret = atomicrmw xor i64* %subr, i64 %val seq_cst
  ret i64 %ret
}

; CHECK: atom12
define i32 @atom12(i32* %subr, i32 %val) {
; CHECK: atom.max.s32
  %ret = atomicrmw max i32* %subr, i32 %val seq_cst
  ret i32 %ret
}

; CHECK: atom13
define i64 @atom13(i64* %subr, i64 %val) {
; CHECK: atom.max.s64
  %ret = atomicrmw max i64* %subr, i64 %val seq_cst
  ret i64 %ret
}

; CHECK: atom14
define i32 @atom14(i32* %subr, i32 %val) {
; CHECK: atom.min.s32
  %ret = atomicrmw min i32* %subr, i32 %val seq_cst
  ret i32 %ret
}

; CHECK: atom15
define i64 @atom15(i64* %subr, i64 %val) {
; CHECK: atom.min.s64
  %ret = atomicrmw min i64* %subr, i64 %val seq_cst
  ret i64 %ret
}

; CHECK: atom16
define i32 @atom16(i32* %subr, i32 %val) {
; CHECK: atom.max.u32
  %ret = atomicrmw umax i32* %subr, i32 %val seq_cst
  ret i32 %ret
}

; CHECK: atom17
define i64 @atom17(i64* %subr, i64 %val) {
; CHECK: atom.max.u64
  %ret = atomicrmw umax i64* %subr, i64 %val seq_cst
  ret i64 %ret
}

; CHECK: atom18
define i32 @atom18(i32* %subr, i32 %val) {
; CHECK: atom.min.u32
  %ret = atomicrmw umin i32* %subr, i32 %val seq_cst
  ret i32 %ret
}

; CHECK: atom19
define i64 @atom19(i64* %subr, i64 %val) {
; CHECK: atom.min.u64
  %ret = atomicrmw umin i64* %subr, i64 %val seq_cst
  ret i64 %ret
}
