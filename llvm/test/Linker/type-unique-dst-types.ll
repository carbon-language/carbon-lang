; RUN: llvm-link %p/type-unique-dst-types.ll \
; RUN:           %p/Inputs/type-unique-dst-types2.ll \
; RUN:           %p/Inputs/type-unique-dst-types3.ll -S -o - | FileCheck %s

; This tests the importance of keeping track of which types are part of the
; destination module.
; When the second input is merged in, the context gets an unused A.11. When
; the third module is then merged, we should pretend it doesn't exist.

; CHECK: %A = type { %B }
; CHECK-NEXT: %B = type { i8 }
; CHECK-NEXT: %A.11.1 = type opaque

%A = type { %B }
%B = type { i8 }
@g3 = external global %A
