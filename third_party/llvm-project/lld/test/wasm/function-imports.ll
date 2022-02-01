; RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown %p/Inputs/ret32.s -o %t.ret32.o
; RUN: llc -filetype=obj %s -o %t.o
; RUN: wasm-ld -o %t.wasm %t.ret32.o %t.o
; RUN: obj2yaml %t.wasm | FileCheck %s

target triple = "wasm32-unknown-unknown"

; Function Attrs: nounwind
define hidden void @_start() local_unnamed_addr #0 {
entry:
  %call = tail call i32 @ret32(float 0.000000e+00) #2
  ret void
}

declare i32 @ret32(float) local_unnamed_addr #1

; CHECK:      Sections:
; CHECK:       - Type:            TYPE
; CHECK-NEXT:    Signatures:
; CHECK-NEXT:      - Index:           0
; CHECK-NEXT:        ParamTypes:
; CHECK-NEXT:          - F32
; CHECK-NEXT:        ReturnTypes:
; CHECK-NEXT:          - I32
; CHECK-NEXT:      - Index:           1
; CHECK-NEXT:        ParamTypes:
; CHECK-NEXT:        ReturnTypes:     []
; CHECK-NEXT:  - Type:            FUNCTION
; CHECK-NEXT:    FunctionTypes:   [ 0, 1 ]
; CHECK:       - Type:            CODE
; CHECK-NEXT:    Functions:
; CHECK:           - Index:       0
; CHECK:           - Index:       1
; CHECK:         Name:            name
; CHECK-NEXT:    FunctionNames:
; CHECK-NEXT:      - Index:           0
; CHECK-NEXT:        Name:            ret32
; CHECK-NEXT:      - Index:           1
; CHECK-NEXT:        Name:            _start
; CHECK-NEXT:    GlobalNames:
; CHECK-NEXT:      - Index:           0
; CHECK-NEXT:        Name:            __stack_pointer
; CHECK-NEXT: ...
