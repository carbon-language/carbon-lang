; RUN: llc < %s -mtriple=arm64-apple-ios7.0 -mcpu=cyclone | FileCheck %s


; The mask:
; CHECK: lCPI0_0:
; CHECK:  .byte   2                       ; 0x2
; CHECK:  .byte   255                     ; 0xff
; CHECK:  .byte   6                       ; 0x6
; CHECK:  .byte   255                     ; 0xff
; The second vector is legalized to undef and the elements of the first vector
; are used instead.
; CHECK:  .byte   2                       ; 0x2
; CHECK:  .byte   4                       ; 0x4
; CHECK:  .byte   6                       ; 0x6
; CHECK:  .byte   0                       ; 0x0
; CHECK: test1
; CHECK: ldr d[[REG0:[0-9]+]], [{{.*}}, lCPI0_0
; CHECK: movi.8h v[[REG1:[0-9]+]], #1, lsl #8
; CHECK: tbl.8b  v{{[0-9]+}}, { v[[REG1]] }, v[[REG0]]
define <8 x i1> @test1() {
entry:
  %Shuff = shufflevector <8 x i1> <i1 0, i1 1, i1 2, i1 3, i1 4, i1 5, i1 6,
                                   i1 7>,
                         <8 x i1> <i1 0, i1 1, i1 2, i1 3, i1 4, i1 5, i1 6,
                                   i1 7>,
                         <8 x i32> <i32 2, i32 undef, i32 6, i32 undef, i32 10,
                                    i32 12, i32 14, i32 0>
  ret <8 x i1> %Shuff
}

; CHECK: lCPI1_0:
; CHECK:          .byte   2                       ; 0x2
; CHECK:          .byte   255                     ; 0xff
; CHECK:          .byte   6                       ; 0x6
; CHECK:          .byte   255                     ; 0xff
; CHECK:          .byte   10                      ; 0xa
; CHECK:          .byte   12                      ; 0xc
; CHECK:          .byte   14                      ; 0xe
; CHECK:          .byte   0                       ; 0x0
; CHECK: test2
; CHECK: ldr     d[[REG0:[0-9]+]], [{{.*}}, lCPI1_0@PAGEOFF]
; CHECK: adrp    x[[REG2:[0-9]+]], lCPI1_1@PAGE
; CHECK: ldr     q[[REG1:[0-9]+]], [x[[REG2]], lCPI1_1@PAGEOFF]
; CHECK: tbl.8b  v{{[0-9]+}}, { v[[REG1]] }, v[[REG0]]
define <8 x i1>@test2() {
bb:
  %Shuff = shufflevector <8 x i1> zeroinitializer,
     <8 x i1> <i1 0, i1 1, i1 1, i1 0, i1 0, i1 1, i1 0, i1 0>,
     <8 x i32> <i32 2, i32 undef, i32 6, i32 undef, i32 10, i32 12, i32 14,
                i32 0>
  ret <8 x i1> %Shuff
}

; CHECK: lCPI2_0:
; CHECK:         .byte   2                       ; 0x2
; CHECK:         .byte   255                     ; 0xff
; CHECK:         .byte   6                       ; 0x6
; CHECK:         .byte   255                     ; 0xff
; CHECK:         .byte   10                      ; 0xa
; CHECK:         .byte   12                      ; 0xc
; CHECK:         .byte   14                      ; 0xe
; CHECK:         .byte   0                       ; 0x0
; CHECK:         .byte   2                       ; 0x2
; CHECK:         .byte   255                     ; 0xff
; CHECK:         .byte   6                       ; 0x6
; CHECK:         .byte   255                     ; 0xff
; CHECK:         .byte   10                      ; 0xa
; CHECK:         .byte   12                      ; 0xc
; CHECK:         .byte   14                      ; 0xe
; CHECK:         .byte   0                       ; 0x0
; CHECK: test3
; CHECK: adrp    x[[REG3:[0-9]+]], lCPI2_0@PAGE
; CHECK: ldr     q[[REG0:[0-9]+]], [x[[REG3]], lCPI2_0@PAGEOFF]
; CHECK: movi.2d v[[REG1:[0-9]+]], #0000000000000000
; CHECK: tbl.16b v{{[0-9]+}}, { v[[REG1]] }, v[[REG0]]
define <16 x i1> @test3(i1* %ptr, i32 %v) {
bb:
  %Shuff = shufflevector <16 x i1> zeroinitializer, <16 x i1> undef,
     <16 x i32> <i32 2, i32 undef, i32 6, i32 undef, i32 10, i32 12, i32 14,
                 i32 0, i32 2, i32 undef, i32 6, i32 undef, i32 10, i32 12,
                 i32 14, i32 0>
  ret <16 x i1> %Shuff
}
; CHECK: lCPI3_1:
; CHECK:         .byte   2                       ; 0x2
; CHECK:         .byte   1                       ; 0x1
; CHECK:         .byte   6                       ; 0x6
; CHECK:         .byte   18                      ; 0x12
; CHECK:         .byte   10                      ; 0xa
; CHECK:         .byte   12                      ; 0xc
; CHECK:         .byte   14                      ; 0xe
; CHECK:         .byte   0                       ; 0x0
; CHECK:         .byte   2                       ; 0x2
; CHECK:         .byte   31                      ; 0x1f
; CHECK:         .byte   6                       ; 0x6
; CHECK:         .byte   30                      ; 0x1e
; CHECK:         .byte   10                      ; 0xa
; CHECK:         .byte   12                      ; 0xc
; CHECK:         .byte   14                      ; 0xe
; CHECK:         .byte   0                       ; 0x0
; CHECK: _test4:
; CHECK:         ldr     q[[REG1:[0-9]+]]
; CHECK:         movi.2d v[[REG0:[0-9]+]], #0000000000000000
; CHECK:         adrp    x[[REG3:[0-9]+]], lCPI3_1@PAGE
; CHECK:         ldr     q[[REG2:[0-9]+]], [x[[REG3]], lCPI3_1@PAGEOFF]
; CHECK:         tbl.16b v{{[0-9]+}}, { v[[REG0]], v[[REG1]] }, v[[REG2]]
define <16 x i1> @test4(i1* %ptr, i32 %v) {
bb:
  %Shuff = shufflevector <16 x i1> zeroinitializer,
     <16 x i1> <i1 0, i1 1, i1 1, i1 0, i1 0, i1 1, i1 0, i1 0, i1 0, i1 1,
                i1 1, i1 0, i1 0, i1 1, i1 0, i1 0>,
     <16 x i32> <i32 2, i32 1, i32 6, i32 18, i32 10, i32 12, i32 14, i32 0,
                 i32 2, i32 31, i32 6, i32 30, i32 10, i32 12, i32 14, i32 0>
  ret <16 x i1> %Shuff
}
