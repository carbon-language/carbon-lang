; RUN: llc -mtriple=i686-pc-linux -print-after=postrapseudos < %s 2>&1 | FileCheck %s

; CHECK: MOV8rr ${{[a-d]}}l, implicit killed $e[[R:[a-d]]]x, implicit-def $e[[R]]x
define i32 @foo(i32 %i, i32 %k, i8* %p) {
  %f = icmp ne i32 %i, %k
  %s = zext i1 %f to i8
  %ret = zext i1 %f to i32
  br label %next
next:
  %d = add i8 %s, 5
  store i8 %d, i8* %p
  ret i32 %ret
}
