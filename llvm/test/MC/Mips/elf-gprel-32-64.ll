; RUN: llc -filetype=obj -march=mips64el -mcpu=mips64 %s -o - \
; RUN: | llvm-readobj -r \
; RUN: | FileCheck %s

define i32 @test(i32 %c) nounwind {
entry:
  switch i32 %c, label %sw.default [
    i32 0, label %sw.bb
    i32 1, label %sw.bb2
    i32 2, label %sw.bb5
    i32 3, label %sw.bb8
  ]

sw.bb:
  br label %return
sw.bb2:
  br label %return
sw.bb5:
  br label %return
sw.bb8:
  br label %return
sw.default:
  br label %return

return:
  %retval.0 = phi i32 [ -1, %sw.default ], [ 7, %sw.bb8 ], [ 2, %sw.bb5 ], [ 3, %sw.bb2 ], [ 1, %sw.bb ]
  ret i32 %retval.0
}

; Check that the appropriate relocations were created.

; R_MIPS_GPREL32/R_MIPS_64/R_MIPS_NONE
; CHECK:      Relocations [
; CHECK:        Section ({{[a-z0-9]+}}) .rodata {
; CHECK-NEXT:     0x{{[0-9,A-F]+}} R_MIPS_GPREL32/R_MIPS_64/R_MIPS_NONE
; CHECK-NEXT:     0x{{[0-9,A-F]+}} R_MIPS_GPREL32/R_MIPS_64/R_MIPS_NONE
; CHECK-NEXT:     0x{{[0-9,A-F]+}} R_MIPS_GPREL32/R_MIPS_64/R_MIPS_NONE
; CHECK-NEXT:     0x{{[0-9,A-F]+}} R_MIPS_GPREL32/R_MIPS_64/R_MIPS_NONE
; CHECK-NEXT:   }
; CHECK-NEXT: ]
