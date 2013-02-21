; RUN: llc -O1 -mcpu=pwr7 -code-model=medium -filetype=obj %s -o - | \
; RUN: elf-dump --dump-section-data | FileCheck %s

; FIXME: When asm-parse is available, could make this an assembly test.

target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

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
; CHECK:       Relocation 0
; CHECK-NEXT:  'r_offset'
; CHECK-NEXT:  'r_sym', 0x[[SYM2:[0-9]+]]
; CHECK-NEXT:  'r_type', 0x00000032
; CHECK:       Relocation 1
; CHECK-NEXT:  'r_offset'
; CHECK-NEXT:  'r_sym', 0x[[SYM2]]
; CHECK-NEXT:  'r_type', 0x00000030
; CHECK:       Relocation 2
; CHECK-NEXT:  'r_offset'
; CHECK-NEXT:  'r_sym', 0x[[SYM2]]
; CHECK-NEXT:  'r_type', 0x00000030

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
; CHECK:       Relocation 3
; CHECK-NEXT:  'r_offset'
; CHECK-NEXT:  'r_sym', 0x[[SYM3:[0-9]+]]
; CHECK-NEXT:  'r_type', 0x00000032
; CHECK:       Relocation 4
; CHECK-NEXT:  'r_offset'
; CHECK-NEXT:  'r_sym', 0x[[SYM3]]
; CHECK-NEXT:  'r_type', 0x00000030
; CHECK:       Relocation 5
; CHECK-NEXT:  'r_offset'
; CHECK-NEXT:  'r_sym', 0x[[SYM3]]
; CHECK-NEXT:  'r_type', 0x00000030

define double @test_double_const() nounwind {
entry:
  ret double 0x3F4FD4920B498CF0
}

; Verify generation of R_PPC64_TOC16_HA and R_PPC64_TOC16_LO for
; accessing a constant.
;
; CHECK:       Relocation 6
; CHECK-NEXT:  'r_offset'
; CHECK-NEXT:  'r_sym', 0x[[SYM4:[0-9]+]]
; CHECK-NEXT:  'r_type', 0x00000032
; CHECK:       Relocation 7
; CHECK-NEXT:  'r_offset'
; CHECK-NEXT:  'r_sym', 0x[[SYM4]]
; CHECK-NEXT:  'r_type', 0x00000030

