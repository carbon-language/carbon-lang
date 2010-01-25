; RUN: llc < %s -mtriple=arm-linux-gnueabi | FileCheck %s -check-prefix=ELF
; RUN: llc < %s -mtriple=arm-apple-darwin | FileCheck %s -check-prefix=DARWIN

@a = global i1 true
; no alignment

@b = global i8 1
; no alignment

@c = global i16 2
;ELF: .align 2
;ELF: c:
;DARWIN: .align 1
;DARWIN: _c:

@d = global i32 3
;ELF: .align 4
;ELF: d:
;DARWIN: .align 2
;DARWIN: _d:

@e = global i64 4
;ELF: .align 8
;ELF: e
;DARWIN: .align 2
;DARWIN: _e:

@f = global float 5.0
;ELF: .align 4
;ELF: f:
;DARWIN: .align 2
;DARWIN: _f:

@g = global double 6.0
;ELF: .align 8
;ELF: g:
;DARWIN: .align 2
;DARWIN: _g:

@bar = common global [75 x i8] zeroinitializer, align 128
;ELF: .comm bar,75,128
; no alignment on darwin?
;DARWIN: .comm _bar,75
