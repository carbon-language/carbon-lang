; RUN: not llvm-ml -filetype=s %s /Fo - 2>&1 | FileCheck %s --implicit-check-not=error:

.code

; CHECK: error: Vararg parameter 'param' should be last in the list of parameters
; CHECK: error: unexpected 'ENDM' in file, no current macro definition
early_vararg_macro MACRO param:vararg, trailing_param
ENDM

; CHECK: error: macro 'colliding_parameters_macro' has multiple parameters named 'paRAM'
; CHECK: error: unexpected 'ENDM' in file, no current macro definition
colliding_parameters_macro MACRO Param, paRAM
ENDM

; CHECK: error: missing parameter qualifier for 'param' in macro 'missing_qualifier_macro'
; CHECK: error: unexpected 'ENDM' in file, no current macro definition
missing_qualifier_macro MACRO param:
ENDM


; CHECK: error: no matching 'endm' in definition
missing_end_macro MACRO

end
