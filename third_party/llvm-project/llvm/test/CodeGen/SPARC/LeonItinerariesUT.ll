; RUN: llc < %s -O1 -march=sparc | FileCheck %s -check-prefix=NO_ITIN
; RUN: llc < %s -O1 -march=sparc -mcpu=leon2   | FileCheck %s -check-prefix=LEON2_ITIN
; RUN: llc < %s -O1 -march=sparc -mcpu=leon3   | FileCheck %s -check-prefix=LEON3_4_ITIN
; RUN: llc < %s -O1 -march=sparc -mcpu=leon4   | FileCheck %s -check-prefix=LEON3_4_ITIN

; NO_ITIN-LABEL: f32_ops:
; NO_ITIN:       ld
; NO_ITIN-NEXT:  ld
; NO_ITIN-NEXT:  ld
; NO_ITIN-NEXT:  ld
; NO_ITIN-NEXT:  fadds
; NO_ITIN-NEXT:  fsubs
; NO_ITIN-NEXT:  fmuls
; NO_ITIN-NEXT:  retl
; NO_ITIN-NEXT:  fdivs

; LEON2_ITIN-LABEL: f32_ops:
; LEON2_ITIN:       ld
; LEON2_ITIN-NEXT:  ld
; LEON2_ITIN-NEXT:  fadds
; LEON2_ITIN-NEXT:  ld
; LEON2_ITIN-NEXT:  fsubs
; LEON2_ITIN-NEXT:  ld
; LEON2_ITIN-NEXT:  fmuls
; LEON2_ITIN-NEXT:  retl
; LEON2_ITIN-NEXT:  fdivs

; LEON3_4_ITIN-LABEL: f32_ops:
; LEON3_4_ITIN:       ld
; LEON3_4_ITIN-NEXT:  ld
; LEON3_4_ITIN-NEXT:  ld
; LEON3_4_ITIN-NEXT:  fadds
; LEON3_4_ITIN-NEXT:  ld
; LEON3_4_ITIN-NEXT:  fsubs
; LEON3_4_ITIN-NEXT:  fmuls
; LEON3_4_ITIN-NEXT:  retl
; LEON3_4_ITIN-NEXT:  fdivs

define float @f32_ops(float* byval(float) %a, float* byval(float) %b, float* byval(float) %c, float* byval(float) %d) {
entry:
  %0 = load float, float* %a, align 8
  %1 = load float, float* %b, align 8
  %2 = load float, float* %c, align 8
  %3 = load float, float* %d, align 8
  %4 = fadd float %0, %1
  %5 = fsub float %4, %2
  %6 = fmul float %5, %3
  %7 = fdiv float %6, %4
  ret float %7
}
