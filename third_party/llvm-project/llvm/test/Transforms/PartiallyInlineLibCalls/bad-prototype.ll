; RUN: opt -S -partially-inline-libcalls < %s | FileCheck %s
; RUN: opt -S -passes=partially-inline-libcalls < %s | FileCheck %s

target triple = "x86_64-unknown-linux-gnu"

declare i32 @sqrt()
declare float @sqrtf()

; CHECK-LABEL: @foo
define i32 @foo() {
  ; CHECK: call{{.*}}@sqrt
  ; CHECK-NOT: call{{.*}}@sqrt
  %r = call i32 @sqrt()
  ret i32 %r
}

; CHECK-LABEL: @bar
define float @bar() {
  ; CHECK: call{{.*}}@sqrtf
  ; CHECK-NOT: call{{.*}}@sqrtf
  %r = call float @sqrtf()
  ret float %r
}
