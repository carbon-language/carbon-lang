; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s
; RUN: verify-uselistorder %s

; Don't crash on null operands.  (If/when we add a verify check for these, we
; should disable the verifier for this test and remove this comment; the test
; is still important.)
!named = !{!0, !1}
!0 = !MDDerivedType(tag: DW_TAG_pointer_type, baseType: null)
!1 = !MDCompileUnit(language: DW_LANG_C, file: null)

; CHECK: !named = !{!0, !1}
; CHECK: !0 = !MDDerivedType({{.*}}baseType: null{{.*}})
; CHECK: !1 = !MDCompileUnit({{.*}}file: null{{.*}})
