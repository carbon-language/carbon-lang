;; RUN: llc -mtriple=powerpc64-unknown-linux-gnu -filetype=obj %s -o - | \
;; RUN: llvm-readobj -r | FileCheck %s

;; FIXME: this file should be in .s form, change when asm parser is available.

@t = thread_local global i32 0, align 4

define i32* @f() nounwind {
entry:
  ret i32* @t
}

;; Check for a pair of R_PPC64_TPREL16_HA / R_PPC64_TPREL16_LO relocs
;; against the thread-local symbol 't'.
;; CHECK:      Relocations [
;; CHECK:        Section ({{[0-9]+}}) .text {
;; CHECK-NEXT:     0x{{[0-9,A-F]+}} R_PPC64_TPREL16_HA t
;; CHECK-NEXT:     0x{{[0-9,A-F]+}} R_PPC64_TPREL16_LO t
;; CHECK-NEXT:   }
