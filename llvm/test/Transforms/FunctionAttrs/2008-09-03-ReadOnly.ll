; RUN: opt < %s -basicaa -functionattrs -S | FileCheck %s

; CHECK: define i32 @f() readonly
define i32 @f() {
entry:
  %tmp = call i32 @e( )
  ret i32 %tmp
}

; CHECK: declare i32 @e() readonly
declare i32 @e() readonly
