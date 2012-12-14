; RUN: llc -mcpu=pwr7 -O0 -filetype=obj %s -o - | \
; RUN: elf-dump --dump-section-data | FileCheck %s

; Test correct relocation generation for thread-local storage
; using the initial-exec model and integrated assembly.

target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

@a = external thread_local global i32

define signext i32 @main() nounwind {
entry:
  %retval = alloca i32, align 4
  store i32 0, i32* %retval
  %0 = load i32* @a, align 4
  ret i32 %0
}

; Verify generation of R_PPC64_GOT_TPREL16_DS and R_PPC64_TLS for
; accessing external variable a.
;
; CHECK:       '.rela.text'
; CHECK:       Relocation 0
; CHECK-NEXT:  'r_offset'
; CHECK-NEXT:  'r_sym', 0x[[SYM1:[0-9a-f]+]]
; CHECK-NEXT:  'r_type', 0x0000005a
; CHECK:       Relocation 1
; CHECK-NEXT:  'r_offset'
; CHECK-NEXT:  'r_sym', 0x[[SYM1]]
; CHECK-NEXT:  'r_type', 0x00000058
; CHECK:       Relocation 2
; CHECK-NEXT:  'r_offset'
; CHECK-NEXT:  'r_sym', 0x[[SYM1]]
; CHECK-NEXT:  'r_type', 0x00000043

