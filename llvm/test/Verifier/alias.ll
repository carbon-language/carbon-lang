; RUN:  not llvm-as %s -o /dev/null 2>&1 | FileCheck %s


declare void @f()
@fa = alias void ()* @f
; CHECK: Alias must point to a definition
; CHECK-NEXT: @fa

@g = external global i32
@ga = alias i32* @g
; CHECK: Alias must point to a definition
; CHECK-NEXT: @ga
