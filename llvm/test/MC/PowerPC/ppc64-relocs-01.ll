;; RUN: llc -mtriple=powerpc64-unknown-linux-gnu -O3 -code-model=small  \
;; RUN:  -filetype=obj %s -o - | \
;; RUN: llvm-readobj -r | FileCheck %s

;; FIXME: this file need to be in .s form, change when asm parse is done.

@number64 = global i64 10, align 8

define i64 @access_int64(i64 %a) nounwind readonly {
entry:
  %0 = load i64* @number64, align 8
  %cmp = icmp eq i64 %0, %a
  %conv1 = zext i1 %cmp to i64
  ret i64 %conv1
}

declare double @sin(double) nounwind

define double @test_branch24 (double %x) nounwind readonly {
entry:
  %add = call double @sin(double %x) nounwind
  ret double %add
}

;; CHECK:      Relocations [

;; The relocations in .rela.text are the 'number64' load using a
;; R_PPC64_TOC16_DS against the .toc and the 'sin' external function
;; address using a R_PPC64_REL24
;; CHECK:        Section ({{[0-9]+}}) .text {
;; CHECK-NEXT:     0x{{[0-9,A-F]+}} R_PPC64_TOC16_DS .toc
;; CHECK-NEXT:     0x{{[0-9,A-F]+}} R_PPC64_REL24    sin
;; CHECK-NEXT:   }

;; The .opd entry for the 'access_int64' function creates 2 relocations:
;; 1. A R_PPC64_ADDR64 against the .text segment plus addend (the function
;    address itself);
;; 2. And a R_PPC64_TOC against no symbol (the linker will replace for the
;;    module's TOC base).
;; CHECK:        Section ({{[0-9]+}}) .opd {
;; CHECK-NEXT:     0x{{[0-9,A-F]+}} R_PPC64_ADDR64 .text 0x0
;; CHECK-NEXT:     0x{{[0-9,A-F]+}} R_PPC64_TOC - 0x0

;; Finally the TOC creates the relocation for the 'number64'.
;; CHECK:        Section ({{[0-9]+}}) .toc {
;; CHECK-NEXT:     0x{{[0-9,A-F]+}} R_PPC64_ADDR64 number64 0x0
;; CHECK-NEXT:   }

;; CHECK-NEXT: ]
