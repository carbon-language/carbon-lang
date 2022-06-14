; RUN: llc -O0 %s -o - | FileCheck %s

target triple = "spirv32-unknown-unknown"

; CHECK-DAG: OpName [[VALUE:%.+]] "value"
; CHECK-DAG: OpName [[IDENTITY:%.+]] "identity"

; CHECK: [[INT:%.+]] = OpTypeInt 32
; CHECK: [[FN:%.+]] = OpTypeFunction [[INT]] [[INT]]

; CHECK: [[IDENTITY]] = OpFunction [[INT]] None [[FN]]
; CHECK-NEXT: [[VALUE]] = OpFunctionParameter [[INT]]
; CHECK-NEXT: {{%.+}} = OpLabel
; CHECK-NEXT: OpReturnValue [[VALUE]]
; CHECK-NEXT: OpFunctionEnd

define i32 @identity(i32 %value) {
  ret i32 %value
}
