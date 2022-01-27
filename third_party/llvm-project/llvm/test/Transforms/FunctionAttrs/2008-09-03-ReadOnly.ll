; RUN: opt < %s -aa-pipeline=basic-aa -passes=function-attrs -S | FileCheck %s

; CHECK: define i32 @f() #0
define i32 @f() {
entry:
  %tmp = call i32 @e( )
  ret i32 %tmp
}

; CHECK: declare i32 @e() #1
declare i32 @e() readonly

; CHECK: attributes #0 = { nofree readonly }
; CHECK: attributes #1 = { readonly }
