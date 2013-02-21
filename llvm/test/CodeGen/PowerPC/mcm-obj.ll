; RUN: llc -O0 -mcpu=pwr7 -code-model=medium -filetype=obj %s -o - | \
; RUN: elf-dump --dump-section-data | FileCheck -check-prefix=MEDIUM %s
; RUN: llc -O0 -mcpu=pwr7 -code-model=large -filetype=obj %s -o - | \
; RUN: elf-dump --dump-section-data | FileCheck -check-prefix=LARGE %s

; FIXME: When asm-parse is available, could make this an assembly test.

target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

@ei = external global i32

define signext i32 @test_external() nounwind {
entry:
  %0 = load i32* @ei, align 4
  %inc = add nsw i32 %0, 1
  store i32 %inc, i32* @ei, align 4
  ret i32 %0
}

; Verify generation of R_PPC64_TOC16_HA and R_PPC64_TOC16_LO_DS for
; accessing external variable ei.
;
; MEDIUM:       '.rela.text'
; MEDIUM:       Relocation 0
; MEDIUM-NEXT:  'r_offset'
; MEDIUM-NEXT:  'r_sym', 0x[[SYM1:[0-9]+]]
; MEDIUM-NEXT:  'r_type', 0x00000032
; MEDIUM:       Relocation 1
; MEDIUM-NEXT:  'r_offset'
; MEDIUM-NEXT:  'r_sym', 0x[[SYM1]]
; MEDIUM-NEXT:  'r_type', 0x00000040
;
; LARGE:       '.rela.text'
; LARGE:       Relocation 0
; LARGE-NEXT:  'r_offset'
; LARGE-NEXT:  'r_sym', 0x[[SYM1:[0-9]+]]
; LARGE-NEXT:  'r_type', 0x00000032
; LARGE:       Relocation 1
; LARGE-NEXT:  'r_offset'
; LARGE-NEXT:  'r_sym', 0x[[SYM1]]
; LARGE-NEXT:  'r_type', 0x00000040

@test_fn_static.si = internal global i32 0, align 4

define signext i32 @test_fn_static() nounwind {
entry:
  %0 = load i32* @test_fn_static.si, align 4
  %inc = add nsw i32 %0, 1
  store i32 %inc, i32* @test_fn_static.si, align 4
  ret i32 %0
}

; Verify generation of R_PPC64_TOC16_HA and R_PPC64_TOC16_LO for
; accessing function-scoped variable si.
;
; MEDIUM:       Relocation 2
; MEDIUM-NEXT:  'r_offset'
; MEDIUM-NEXT:  'r_sym', 0x[[SYM2:[0-9]+]]
; MEDIUM-NEXT:  'r_type', 0x00000032
; MEDIUM:       Relocation 3
; MEDIUM-NEXT:  'r_offset'
; MEDIUM-NEXT:  'r_sym', 0x[[SYM2]]
; MEDIUM-NEXT:  'r_type', 0x00000030
;
; Verify generation of R_PPC64_TOC16_HA and R_PPC64_TOC16_LO_DS for
; accessing function-scoped variable si.
;
; LARGE:       Relocation 2
; LARGE-NEXT:  'r_offset'
; LARGE-NEXT:  'r_sym', 0x[[SYM2:[0-9]+]]
; LARGE-NEXT:  'r_type', 0x00000032
; LARGE:       Relocation 3
; LARGE-NEXT:  'r_offset'
; LARGE-NEXT:  'r_sym', 0x[[SYM2]]
; LARGE-NEXT:  'r_type', 0x00000040

@gi = global i32 5, align 4

define signext i32 @test_file_static() nounwind {
entry:
  %0 = load i32* @gi, align 4
  %inc = add nsw i32 %0, 1
  store i32 %inc, i32* @gi, align 4
  ret i32 %0
}

; Verify generation of R_PPC64_TOC16_HA and R_PPC64_TOC16_LO for
; accessing file-scope variable gi.
;
; MEDIUM:       Relocation 4
; MEDIUM-NEXT:  'r_offset'
; MEDIUM-NEXT:  'r_sym', 0x[[SYM3:[0-9]+]]
; MEDIUM-NEXT:  'r_type', 0x00000032
; MEDIUM:       Relocation 5
; MEDIUM-NEXT:  'r_offset'
; MEDIUM-NEXT:  'r_sym', 0x[[SYM3]]
; MEDIUM-NEXT:  'r_type', 0x00000030
;
; Verify generation of R_PPC64_TOC16_HA and R_PPC64_TOC16_LO_DS for
; accessing file-scope variable gi.
;
; LARGE:       Relocation 4
; LARGE-NEXT:  'r_offset'
; LARGE-NEXT:  'r_sym', 0x[[SYM3:[0-9]+]]
; LARGE-NEXT:  'r_type', 0x00000032
; LARGE:       Relocation 5
; LARGE-NEXT:  'r_offset'
; LARGE-NEXT:  'r_sym', 0x[[SYM3]]
; LARGE-NEXT:  'r_type', 0x00000040

define double @test_double_const() nounwind {
entry:
  ret double 0x3F4FD4920B498CF0
}

; Verify generation of R_PPC64_TOC16_HA and R_PPC64_TOC16_LO for
; accessing a constant.
;
; MEDIUM:       Relocation 6
; MEDIUM-NEXT:  'r_offset'
; MEDIUM-NEXT:  'r_sym', 0x[[SYM4:[0-9]+]]
; MEDIUM-NEXT:  'r_type', 0x00000032
; MEDIUM:       Relocation 7
; MEDIUM-NEXT:  'r_offset'
; MEDIUM-NEXT:  'r_sym', 0x[[SYM4]]
; MEDIUM-NEXT:  'r_type', 0x00000030
;
; Verify generation of R_PPC64_TOC16_HA and R_PPC64_TOC16_LO_DS for
; accessing a constant.
;
; LARGE:       Relocation 6
; LARGE-NEXT:  'r_offset'
; LARGE-NEXT:  'r_sym', 0x[[SYM4:[0-9]+]]
; LARGE-NEXT:  'r_type', 0x00000032
; LARGE:       Relocation 7
; LARGE-NEXT:  'r_offset'
; LARGE-NEXT:  'r_sym', 0x[[SYM4]]
; LARGE-NEXT:  'r_type', 0x00000040

define signext i32 @test_jump_table(i32 signext %i) nounwind {
entry:
  %i.addr = alloca i32, align 4
  store i32 %i, i32* %i.addr, align 4
  %0 = load i32* %i.addr, align 4
  switch i32 %0, label %sw.default [
    i32 3, label %sw.bb
    i32 4, label %sw.bb1
    i32 5, label %sw.bb2
    i32 6, label %sw.bb3
  ]

sw.default:                                       ; preds = %entry
  br label %sw.epilog

sw.bb:                                            ; preds = %entry
  %1 = load i32* %i.addr, align 4
  %mul = mul nsw i32 %1, 7
  store i32 %mul, i32* %i.addr, align 4
  br label %sw.bb1

sw.bb1:                                           ; preds = %entry, %sw.bb
  %2 = load i32* %i.addr, align 4
  %dec = add nsw i32 %2, -1
  store i32 %dec, i32* %i.addr, align 4
  br label %sw.bb2

sw.bb2:                                           ; preds = %entry, %sw.bb1
  %3 = load i32* %i.addr, align 4
  %add = add nsw i32 %3, 3
  store i32 %add, i32* %i.addr, align 4
  br label %sw.bb3

sw.bb3:                                           ; preds = %entry, %sw.bb2
  %4 = load i32* %i.addr, align 4
  %shl = shl i32 %4, 1
  store i32 %shl, i32* %i.addr, align 4
  br label %sw.epilog

sw.epilog:                                        ; preds = %sw.bb3, %sw.default
  %5 = load i32* %i.addr, align 4
  ret i32 %5
}

; Verify generation of R_PPC64_TOC16_HA and R_PPC64_TOC16_LO_DS for
; accessing a jump table address.
;
; MEDIUM:       Relocation 8
; MEDIUM-NEXT:  'r_offset'
; MEDIUM-NEXT:  'r_sym', 0x[[SYM5:[0-9]+]]
; MEDIUM-NEXT:  'r_type', 0x00000032
; MEDIUM:       Relocation 9
; MEDIUM-NEXT:  'r_offset'
; MEDIUM-NEXT:  'r_sym', 0x[[SYM5]]
; MEDIUM-NEXT:  'r_type', 0x00000040
;
; LARGE:       Relocation 8
; LARGE-NEXT:  'r_offset'
; LARGE-NEXT:  'r_sym', 0x[[SYM5:[0-9]+]]
; LARGE-NEXT:  'r_type', 0x00000032
; LARGE:       Relocation 9
; LARGE-NEXT:  'r_offset'
; LARGE-NEXT:  'r_sym', 0x[[SYM5]]
; LARGE-NEXT:  'r_type', 0x00000040

@ti = common global i32 0, align 4

define signext i32 @test_tentative() nounwind {
entry:
  %0 = load i32* @ti, align 4
  %inc = add nsw i32 %0, 1
  store i32 %inc, i32* @ti, align 4
  ret i32 %0
}

; Verify generation of R_PPC64_TOC16_HA and R_PPC64_TOC16_LO_DS for
; accessing tentatively declared variable ti.
;
; MEDIUM:       Relocation 10
; MEDIUM-NEXT:  'r_offset'
; MEDIUM-NEXT:  'r_sym', 0x[[SYM6:[0-9]+]]
; MEDIUM-NEXT:  'r_type', 0x00000032
; MEDIUM:       Relocation 11
; MEDIUM-NEXT:  'r_offset'
; MEDIUM-NEXT:  'r_sym', 0x[[SYM6]]
; MEDIUM-NEXT:  'r_type', 0x00000040
;
; LARGE:       Relocation 10
; LARGE-NEXT:  'r_offset'
; LARGE-NEXT:  'r_sym', 0x[[SYM6:[0-9]+]]
; LARGE-NEXT:  'r_type', 0x00000032
; LARGE:       Relocation 11
; LARGE-NEXT:  'r_offset'
; LARGE-NEXT:  'r_sym', 0x[[SYM6]]
; LARGE-NEXT:  'r_type', 0x00000040

define i8* @test_fnaddr() nounwind {
entry:
  %func = alloca i32 (i32)*, align 8
  store i32 (i32)* @foo, i32 (i32)** %func, align 8
  %0 = load i32 (i32)** %func, align 8
  %1 = bitcast i32 (i32)* %0 to i8*
  ret i8* %1
}

declare signext i32 @foo(i32 signext)

; Verify generation of R_PPC64_TOC16_HA and R_PPC64_TOC16_LO_DS for
; accessing function address foo.
;
; MEDIUM:       Relocation 12
; MEDIUM-NEXT:  'r_offset'
; MEDIUM-NEXT:  'r_sym', 0x[[SYM7:[0-9]+]]
; MEDIUM-NEXT:  'r_type', 0x00000032
; MEDIUM:       Relocation 13
; MEDIUM-NEXT:  'r_offset'
; MEDIUM-NEXT:  'r_sym', 0x[[SYM7]]
; MEDIUM-NEXT:  'r_type', 0x00000040
;
; LARGE:       Relocation 12
; LARGE-NEXT:  'r_offset'
; LARGE-NEXT:  'r_sym', 0x[[SYM7:[0-9]+]]
; LARGE-NEXT:  'r_type', 0x00000032
; LARGE:       Relocation 13
; LARGE-NEXT:  'r_offset'
; LARGE-NEXT:  'r_sym', 0x[[SYM7]]
; LARGE-NEXT:  'r_type', 0x00000040

