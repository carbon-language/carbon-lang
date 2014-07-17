; RUN: opt < %s -instsimplify -S | FileCheck %s

; CHECK-LABEL: @foo
; CHECK-NOT: ashr
define i32 @foo(i32 %x) {
 %o = and i32 %x, 1
 %n = add i32 %o, -1
 %t = ashr i32 %n, 17
 ret i32 %t
}
