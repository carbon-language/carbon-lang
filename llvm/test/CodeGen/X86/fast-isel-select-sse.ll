; RUN: llc < %s -mtriple=x86_64-apple-darwin10                                              | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-apple-darwin10 -fast-isel -fast-isel-abort                  | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-apple-darwin10                             -mcpu=corei7-avx | FileCheck %s --check-prefix=AVX
; RUN: llc < %s -mtriple=x86_64-apple-darwin10 -fast-isel -fast-isel-abort -mcpu=corei7-avx | FileCheck %s --check-prefix=AVX

; Test all cmp predicates that can be used with SSE.

define float @select_fcmp_oeq_f32(float %a, float %b, float %c, float %d) {
; CHECK-LABEL: select_fcmp_oeq_f32
; CHECK:       cmpeqss %xmm1, %xmm0
; CHECK-NEXT:  andps   %xmm0, %xmm2
; CHECK-NEXT:  andnps  %xmm3, %xmm0
; CHECK-NEXT:  orps    %xmm2, %xmm0
; AVX-LABEL: select_fcmp_oeq_f32
; AVX:       vcmpeqss %xmm1, %xmm0, %xmm0
; AVX-NEXT:  vandps   %xmm2, %xmm0, %xmm1
; AVX-NEXT:  vandnps  %xmm3, %xmm0, %xmm0
; AVX-NEXT:  vorps    %xmm1, %xmm0, %xmm0
  %1 = fcmp oeq float %a, %b
  %2 = select i1 %1, float %c, float %d
  ret float %2
}

define double @select_fcmp_oeq_f64(double %a, double %b, double %c, double %d) {
; CHECK-LABEL: select_fcmp_oeq_f64
; CHECK:       cmpeqsd %xmm1, %xmm0
; CHECK-NEXT:  andpd   %xmm0, %xmm2
; CHECK-NEXT:  andnpd  %xmm3, %xmm0
; CHECK-NEXT:  orpd    %xmm2, %xmm0
; AVX-LABEL: select_fcmp_oeq_f64
; AVX:       vcmpeqsd %xmm1, %xmm0, %xmm0
; AVX-NEXT:  vandpd   %xmm2, %xmm0, %xmm1
; AVX-NEXT:  vandnpd  %xmm3, %xmm0, %xmm0
; AVX-NEXT:  vorpd    %xmm1, %xmm0, %xmm0
  %1 = fcmp oeq double %a, %b
  %2 = select i1 %1, double %c, double %d
  ret double %2
}

define float @select_fcmp_ogt_f32(float %a, float %b, float %c, float %d) {
; CHECK-LABEL: select_fcmp_ogt_f32
; CHECK:       cmpltss %xmm0, %xmm1
; CHECK-NEXT:  andps   %xmm1, %xmm2
; CHECK-NEXT:  andnps  %xmm3, %xmm1
; CHECK-NEXT:  orps    %xmm2, %xmm1
; AVX-LABEL: select_fcmp_ogt_f32
; AVX:       vcmpltss %xmm0, %xmm1, %xmm0
; AVX-NEXT:  vandps   %xmm2, %xmm0, %xmm1
; AVX-NEXT:  vandnps  %xmm3, %xmm0, %xmm0
; AVX-NEXT:  vorps    %xmm1, %xmm0, %xmm0
  %1 = fcmp ogt float %a, %b
  %2 = select i1 %1, float %c, float %d
  ret float %2
}

define double @select_fcmp_ogt_f64(double %a, double %b, double %c, double %d) {
; CHECK-LABEL: select_fcmp_ogt_f64
; CHECK:       cmpltsd %xmm0, %xmm1
; CHECK-NEXT:  andpd   %xmm1, %xmm2
; CHECK-NEXT:  andnpd  %xmm3, %xmm1
; CHECK-NEXT:  orpd    %xmm2, %xmm1
; AVX-LABEL: select_fcmp_ogt_f64
; AVX:       vcmpltsd %xmm0, %xmm1, %xmm0
; AVX-NEXT:  vandpd   %xmm2, %xmm0, %xmm1
; AVX-NEXT:  vandnpd  %xmm3, %xmm0, %xmm0
; AVX-NEXT:  vorpd    %xmm1, %xmm0, %xmm0
  %1 = fcmp ogt double %a, %b
  %2 = select i1 %1, double %c, double %d
  ret double %2
}

define float @select_fcmp_oge_f32(float %a, float %b, float %c, float %d) {
; CHECK-LABEL: select_fcmp_oge_f32
; CHECK:       cmpless %xmm0, %xmm1
; CHECK-NEXT:  andps   %xmm1, %xmm2
; CHECK-NEXT:  andnps  %xmm3, %xmm1
; CHECK-NEXT:  orps    %xmm2, %xmm1
; AVX-LABEL: select_fcmp_oge_f32
; AVX:       vcmpless %xmm0, %xmm1, %xmm0
; AVX-NEXT:  vandps   %xmm2, %xmm0, %xmm1
; AVX-NEXT:  vandnps  %xmm3, %xmm0, %xmm0
; AVX-NEXT:  vorps    %xmm1, %xmm0, %xmm0
  %1 = fcmp oge float %a, %b
  %2 = select i1 %1, float %c, float %d
  ret float %2
}

define double @select_fcmp_oge_f64(double %a, double %b, double %c, double %d) {
; CHECK-LABEL: select_fcmp_oge_f64
; CHECK:       cmplesd %xmm0, %xmm1
; CHECK-NEXT:  andpd   %xmm1, %xmm2
; CHECK-NEXT:  andnpd  %xmm3, %xmm1
; CHECK-NEXT:  orpd    %xmm2, %xmm1
; AVX-LABEL: select_fcmp_oge_f64
; AVX:       vcmplesd %xmm0, %xmm1, %xmm0
; AVX-NEXT:  vandpd   %xmm2, %xmm0, %xmm1
; AVX-NEXT:  vandnpd  %xmm3, %xmm0, %xmm0
; AVX-NEXT:  vorpd    %xmm1, %xmm0, %xmm0
  %1 = fcmp oge double %a, %b
  %2 = select i1 %1, double %c, double %d
  ret double %2
}

define float @select_fcmp_olt_f32(float %a, float %b, float %c, float %d) {
; CHECK-LABEL: select_fcmp_olt_f32
; CHECK:       cmpltss %xmm1, %xmm0
; CHECK-NEXT:  andps   %xmm0, %xmm2
; CHECK-NEXT:  andnps  %xmm3, %xmm0
; CHECK-NEXT:  orps    %xmm2, %xmm0
; AVX-LABEL: select_fcmp_olt_f32
; AVX:       vcmpltss %xmm1, %xmm0, %xmm0
; AVX-NEXT:  vandps   %xmm2, %xmm0, %xmm1
; AVX-NEXT:  vandnps  %xmm3, %xmm0, %xmm0
; AVX-NEXT:  vorps    %xmm1, %xmm0, %xmm0
  %1 = fcmp olt float %a, %b
  %2 = select i1 %1, float %c, float %d
  ret float %2
}

define double @select_fcmp_olt_f64(double %a, double %b, double %c, double %d) {
; CHECK-LABEL: select_fcmp_olt_f64
; CHECK:       cmpltsd %xmm1, %xmm0
; CHECK-NEXT:  andpd   %xmm0, %xmm2
; CHECK-NEXT:  andnpd  %xmm3, %xmm0
; CHECK-NEXT:  orpd    %xmm2, %xmm0
; AVX-LABEL: select_fcmp_olt_f64
; AVX:       vcmpltsd %xmm1, %xmm0, %xmm0
; AVX-NEXT:  vandpd   %xmm2, %xmm0, %xmm1
; AVX-NEXT:  vandnpd  %xmm3, %xmm0, %xmm0
; AVX-NEXT:  vorpd    %xmm1, %xmm0, %xmm0
  %1 = fcmp olt double %a, %b
  %2 = select i1 %1, double %c, double %d
  ret double %2
}

define float @select_fcmp_ole_f32(float %a, float %b, float %c, float %d) {
; CHECK-LABEL: select_fcmp_ole_f32
; CHECK:       cmpless %xmm1, %xmm0
; CHECK-NEXT:  andps   %xmm0, %xmm2
; CHECK-NEXT:  andnps  %xmm3, %xmm0
; CHECK-NEXT:  orps    %xmm2, %xmm0
; AVX-LABEL: select_fcmp_ole_f32
; AVX:       vcmpless %xmm1, %xmm0, %xmm0
; AVX-NEXT:  vandps   %xmm2, %xmm0, %xmm1
; AVX-NEXT:  vandnps  %xmm3, %xmm0, %xmm0
; AVX-NEXT:  vorps    %xmm1, %xmm0, %xmm0
  %1 = fcmp ole float %a, %b
  %2 = select i1 %1, float %c, float %d
  ret float %2
}

define double @select_fcmp_ole_f64(double %a, double %b, double %c, double %d) {
; CHECK-LABEL: select_fcmp_ole_f64
; CHECK:       cmplesd %xmm1, %xmm0
; CHECK-NEXT:  andpd   %xmm0, %xmm2
; CHECK-NEXT:  andnpd  %xmm3, %xmm0
; CHECK-NEXT:  orpd    %xmm2, %xmm0
; AVX-LABEL: select_fcmp_ole_f64
; AVX:       vcmplesd %xmm1, %xmm0, %xmm0
; AVX-NEXT:  vandpd   %xmm2, %xmm0, %xmm1
; AVX-NEXT:  vandnpd  %xmm3, %xmm0, %xmm0
; AVX-NEXT:  vorpd    %xmm1, %xmm0, %xmm0
  %1 = fcmp ole double %a, %b
  %2 = select i1 %1, double %c, double %d
  ret double %2
}

define float @select_fcmp_ord_f32(float %a, float %b, float %c, float %d) {
; CHECK-LABEL: select_fcmp_ord_f32
; CHECK:       cmpordss %xmm1, %xmm0
; CHECK-NEXT:  andps    %xmm0, %xmm2
; CHECK-NEXT:  andnps   %xmm3, %xmm0
; CHECK-NEXT:  orps     %xmm2, %xmm0
; AVX-LABEL: select_fcmp_ord_f32
; AVX:       vcmpordss %xmm1, %xmm0, %xmm0
; AVX-NEXT:  vandps    %xmm2, %xmm0, %xmm1
; AVX-NEXT:  vandnps   %xmm3, %xmm0, %xmm0
; AVX-NEXT:  vorps     %xmm1, %xmm0, %xmm0
  %1 = fcmp ord float %a, %b
  %2 = select i1 %1, float %c, float %d
  ret float %2
}

define double @select_fcmp_ord_f64(double %a, double %b, double %c, double %d) {
; CHECK-LABEL: select_fcmp_ord_f64
; CHECK:       cmpordsd %xmm1, %xmm0
; CHECK-NEXT:  andpd    %xmm0, %xmm2
; CHECK-NEXT:  andnpd   %xmm3, %xmm0
; CHECK-NEXT:  orpd     %xmm2, %xmm0
; AVX-LABEL: select_fcmp_ord_f64
; AVX:       vcmpordsd %xmm1, %xmm0, %xmm0
; AVX-NEXT:  vandpd    %xmm2, %xmm0, %xmm1
; AVX-NEXT:  vandnpd   %xmm3, %xmm0, %xmm0
; AVX-NEXT:  vorpd     %xmm1, %xmm0, %xmm0
  %1 = fcmp ord double %a, %b
  %2 = select i1 %1, double %c, double %d
  ret double %2
}

define float @select_fcmp_uno_f32(float %a, float %b, float %c, float %d) {
; CHECK-LABEL: select_fcmp_uno_f32
; CHECK:       cmpunordss %xmm1, %xmm0
; CHECK-NEXT:  andps      %xmm0, %xmm2
; CHECK-NEXT:  andnps     %xmm3, %xmm0
; CHECK-NEXT:  orps       %xmm2, %xmm0
; AVX-LABEL: select_fcmp_uno_f32
; AVX:       vcmpunordss %xmm1, %xmm0, %xmm0
; AVX-NEXT:  vandps      %xmm2, %xmm0, %xmm1
; AVX-NEXT:  vandnps     %xmm3, %xmm0, %xmm0
; AVX-NEXT:  vorps       %xmm1, %xmm0, %xmm0
  %1 = fcmp uno float %a, %b
  %2 = select i1 %1, float %c, float %d
  ret float %2
}

define double @select_fcmp_uno_f64(double %a, double %b, double %c, double %d) {
; CHECK-LABEL: select_fcmp_uno_f64
; CHECK:       cmpunordsd %xmm1, %xmm0
; CHECK-NEXT:  andpd      %xmm0, %xmm2
; CHECK-NEXT:  andnpd     %xmm3, %xmm0
; CHECK-NEXT:  orpd       %xmm2, %xmm0
; AVX-LABEL: select_fcmp_uno_f64
; AVX:       vcmpunordsd %xmm1, %xmm0, %xmm0
; AVX-NEXT:  vandpd      %xmm2, %xmm0, %xmm1
; AVX-NEXT:  vandnpd     %xmm3, %xmm0, %xmm0
; AVX-NEXT:  vorpd       %xmm1, %xmm0, %xmm0
  %1 = fcmp uno double %a, %b
  %2 = select i1 %1, double %c, double %d
  ret double %2
}

define float @select_fcmp_ugt_f32(float %a, float %b, float %c, float %d) {
; CHECK-LABEL: select_fcmp_ugt_f32
; CHECK:       cmpnless %xmm1, %xmm0
; CHECK-NEXT:  andps    %xmm0, %xmm2
; CHECK-NEXT:  andnps   %xmm3, %xmm0
; CHECK-NEXT:  orps     %xmm2, %xmm0
; AVX-LABEL: select_fcmp_ugt_f32
; AVX:       vcmpnless %xmm1, %xmm0, %xmm0
; AVX-NEXT:  vandps    %xmm2, %xmm0, %xmm1
; AVX-NEXT:  vandnps   %xmm3, %xmm0, %xmm0
; AVX-NEXT:  vorps     %xmm1, %xmm0, %xmm0
  %1 = fcmp ugt float %a, %b
  %2 = select i1 %1, float %c, float %d
  ret float %2
}

define double @select_fcmp_ugt_f64(double %a, double %b, double %c, double %d) {
; CHECK-LABEL: select_fcmp_ugt_f64
; CHECK:       cmpnlesd %xmm1, %xmm0
; CHECK-NEXT:  andpd    %xmm0, %xmm2
; CHECK-NEXT:  andnpd   %xmm3, %xmm0
; CHECK-NEXT:  orpd     %xmm2, %xmm0
; AVX-LABEL: select_fcmp_ugt_f64
; AVX:       vcmpnlesd %xmm1, %xmm0, %xmm0
; AVX-NEXT:  vandpd    %xmm2, %xmm0, %xmm1
; AVX-NEXT:  vandnpd   %xmm3, %xmm0, %xmm0
; AVX-NEXT:  vorpd     %xmm1, %xmm0, %xmm0
  %1 = fcmp ugt double %a, %b
  %2 = select i1 %1, double %c, double %d
  ret double %2
}

define float @select_fcmp_uge_f32(float %a, float %b, float %c, float %d) {
; CHECK-LABEL: select_fcmp_uge_f32
; CHECK:       cmpnltss %xmm1, %xmm0
; CHECK-NEXT:  andps    %xmm0, %xmm2
; CHECK-NEXT:  andnps   %xmm3, %xmm0
; CHECK-NEXT:  orps     %xmm2, %xmm0
; AVX-LABEL: select_fcmp_uge_f32
; AVX:       vcmpnltss %xmm1, %xmm0, %xmm0
; AVX-NEXT:  vandps    %xmm2, %xmm0, %xmm1
; AVX-NEXT:  vandnps   %xmm3, %xmm0, %xmm0
; AVX-NEXT:  vorps     %xmm1, %xmm0, %xmm0
  %1 = fcmp uge float %a, %b
  %2 = select i1 %1, float %c, float %d
  ret float %2
}

define double @select_fcmp_uge_f64(double %a, double %b, double %c, double %d) {
; CHECK-LABEL: select_fcmp_uge_f64
; CHECK:       cmpnltsd %xmm1, %xmm0
; CHECK-NEXT:  andpd    %xmm0, %xmm2
; CHECK-NEXT:  andnpd   %xmm3, %xmm0
; CHECK-NEXT:  orpd     %xmm2, %xmm0
; AVX-LABEL: select_fcmp_uge_f64
; AVX:       vcmpnltsd %xmm1, %xmm0, %xmm0
; AVX-NEXT:  vandpd    %xmm2, %xmm0, %xmm1
; AVX-NEXT:  vandnpd   %xmm3, %xmm0, %xmm0
; AVX-NEXT:  vorpd     %xmm1, %xmm0, %xmm0
  %1 = fcmp uge double %a, %b
  %2 = select i1 %1, double %c, double %d
  ret double %2
}

define float @select_fcmp_ult_f32(float %a, float %b, float %c, float %d) {
; CHECK-LABEL: select_fcmp_ult_f32
; CHECK:       cmpnless %xmm0, %xmm1
; CHECK-NEXT:  andps    %xmm1, %xmm2
; CHECK-NEXT:  andnps   %xmm3, %xmm1
; CHECK-NEXT:  orps     %xmm2, %xmm1
; AVX-LABEL: select_fcmp_ult_f32
; AVX:       vcmpnless %xmm0, %xmm1, %xmm0
; AVX-NEXT:  vandps    %xmm2, %xmm0, %xmm1
; AVX-NEXT:  vandnps   %xmm3, %xmm0, %xmm0
; AVX-NEXT:  vorps     %xmm1, %xmm0, %xmm0
  %1 = fcmp ult float %a, %b
  %2 = select i1 %1, float %c, float %d
  ret float %2
}

define double @select_fcmp_ult_f64(double %a, double %b, double %c, double %d) {
; CHECK-LABEL: select_fcmp_ult_f64
; CHECK:       cmpnlesd %xmm0, %xmm1
; CHECK-NEXT:  andpd    %xmm1, %xmm2
; CHECK-NEXT:  andnpd   %xmm3, %xmm1
; CHECK-NEXT:  orpd     %xmm2, %xmm1
; AVX-LABEL: select_fcmp_ult_f64
; AVX:       vcmpnlesd %xmm0, %xmm1, %xmm0
; AVX-NEXT:  vandpd    %xmm2, %xmm0, %xmm1
; AVX-NEXT:  vandnpd   %xmm3, %xmm0, %xmm0
; AVX-NEXT:  vorpd     %xmm1, %xmm0, %xmm0
  %1 = fcmp ult double %a, %b
  %2 = select i1 %1, double %c, double %d
  ret double %2
}

define float @select_fcmp_ule_f32(float %a, float %b, float %c, float %d) {
; CHECK-LABEL: select_fcmp_ule_f32
; CHECK:       cmpnltss %xmm0, %xmm1
; CHECK-NEXT:  andps    %xmm1, %xmm2
; CHECK-NEXT:  andnps   %xmm3, %xmm1
; CHECK-NEXT:  orps     %xmm2, %xmm1
; AVX-LABEL: select_fcmp_ule_f32
; AVX:       vcmpnltss %xmm0, %xmm1, %xmm0
; AVX-NEXT:  vandps    %xmm2, %xmm0, %xmm1
; AVX-NEXT:  vandnps   %xmm3, %xmm0, %xmm0
; AVX-NEXT:  vorps     %xmm1, %xmm0, %xmm0
  %1 = fcmp ule float %a, %b
  %2 = select i1 %1, float %c, float %d
  ret float %2
}

define double @select_fcmp_ule_f64(double %a, double %b, double %c, double %d) {
; CHECK-LABEL: select_fcmp_ule_f64
; CHECK:       cmpnltsd %xmm0, %xmm1
; CHECK-NEXT:  andpd    %xmm1, %xmm2
; CHECK-NEXT:  andnpd   %xmm3, %xmm1
; CHECK-NEXT:  orpd     %xmm2, %xmm1
; AVX-LABEL: select_fcmp_ule_f64
; AVX:       vcmpnltsd %xmm0, %xmm1, %xmm0
; AVX-NEXT:  vandpd    %xmm2, %xmm0, %xmm1
; AVX-NEXT:  vandnpd   %xmm3, %xmm0, %xmm0
; AVX-NEXT:  vorpd     %xmm1, %xmm0, %xmm0
  %1 = fcmp ule double %a, %b
  %2 = select i1 %1, double %c, double %d
  ret double %2
}

define float @select_fcmp_une_f32(float %a, float %b, float %c, float %d) {
; CHECK-LABEL: select_fcmp_une_f32
; CHECK:       cmpneqss %xmm1, %xmm0
; CHECK-NEXT:  andps    %xmm0, %xmm2
; CHECK-NEXT:  andnps   %xmm3, %xmm0
; CHECK-NEXT:  orps     %xmm2, %xmm0
; AVX-LABEL: select_fcmp_une_f32
; AVX:       vcmpneqss %xmm1, %xmm0, %xmm0
; AVX-NEXT:  vandps    %xmm2, %xmm0, %xmm1
; AVX-NEXT:  vandnps   %xmm3, %xmm0, %xmm0
; AVX-NEXT:  vorps     %xmm1, %xmm0, %xmm0
  %1 = fcmp une float %a, %b
  %2 = select i1 %1, float %c, float %d
  ret float %2
}

define double @select_fcmp_une_f64(double %a, double %b, double %c, double %d) {
; CHECK-LABEL: select_fcmp_une_f64
; CHECK:       cmpneqsd %xmm1, %xmm0
; CHECK-NEXT:  andpd    %xmm0, %xmm2
; CHECK-NEXT:  andnpd   %xmm3, %xmm0
; CHECK-NEXT:  orpd     %xmm2, %xmm0
; AVX-LABEL: select_fcmp_une_f64
; AVX:       vcmpneqsd %xmm1, %xmm0, %xmm0
; AVX-NEXT:  vandpd    %xmm2, %xmm0, %xmm1
; AVX-NEXT:  vandnpd   %xmm3, %xmm0, %xmm0
; AVX-NEXT:  vorpd     %xmm1, %xmm0, %xmm0
  %1 = fcmp une double %a, %b
  %2 = select i1 %1, double %c, double %d
  ret double %2
}

