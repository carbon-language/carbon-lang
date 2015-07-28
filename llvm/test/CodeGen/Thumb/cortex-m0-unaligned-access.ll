; RUN: llc -mtriple=thumbv6m-apple-unknown-macho -mattr=+strict-align < %s | FileCheck --check-prefix=V6M %s
; RUN: llc -mtriple=thumbv7m-apple-unknown-macho < %s | FileCheck --check-prefix=V7M %s

define i32 @split_load(i32* %p) nounwind {
; V6M-LABEL: split_load
; V6M: ldrh
; V6M: ldrh
; V7M-LABEL: split_load
; V7M-NOT: ldrh
; V7M: bx lr
  %val = load i32, i32* %p, align 2
  ret i32 %val
}
