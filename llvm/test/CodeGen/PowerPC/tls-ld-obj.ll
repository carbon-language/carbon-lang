; RUN: llc -mcpu=pwr7 -O0 -filetype=obj -relocation-model=pic %s -o - | \
; RUN: elf-dump --dump-section-data | FileCheck %s

; Test correct relocation generation for thread-local storage using
; the local dynamic model.

target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

@a = hidden thread_local global i32 0, align 4

define signext i32 @main() nounwind {
entry:
  %retval = alloca i32, align 4
  store i32 0, i32* %retval
  %0 = load i32* @a, align 4
  ret i32 %0
}

; Verify generation of R_PPC64_GOT_TLSLD16_HA, R_PPC64_GOT_TLSLD16_LO,
; R_PPC64_TLSLD, R_PPC64_DTPREL16_HA, and R_PPC64_DTPREL16_LO for
; accessing external variable a, and R_PPC64_REL24 for the call to
; __tls_get_addr.
;
; CHECK:       '.rela.text'
; CHECK:       Relocation 0
; CHECK-NEXT:  'r_offset'
; CHECK-NEXT:  'r_sym', 0x[[SYM1:[0-9a-f]+]]
; CHECK-NEXT:  'r_type', 0x00000056
; CHECK:       Relocation 1
; CHECK-NEXT:  'r_offset'
; CHECK-NEXT:  'r_sym', 0x[[SYM1]]
; CHECK-NEXT:  'r_type', 0x00000054
; CHECK:       Relocation 2
; CHECK-NEXT:  'r_offset'
; CHECK-NEXT:  'r_sym', 0x[[SYM1]]
; CHECK-NEXT:  'r_type', 0x0000006c
; CHECK:       Relocation 3
; CHECK-NEXT:  'r_offset'
; CHECK-NEXT:  'r_sym', 0x{{[0-9a-f]+}}
; CHECK-NEXT:  'r_type', 0x0000000a
; CHECK:       Relocation 4
; CHECK-NEXT:  'r_offset'
; CHECK-NEXT:  'r_sym', 0x[[SYM1]]
; CHECK-NEXT:  'r_type', 0x0000004d
; CHECK:       Relocation 5
; CHECK-NEXT:  'r_offset'
; CHECK-NEXT:  'r_sym', 0x[[SYM1]]
; CHECK-NEXT:  'r_type', 0x0000004b

