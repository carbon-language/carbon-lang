; RUN: llc -mtriple=armv7-linux-gnueabihf %s -o - | FileCheck %s

; Check that we don't create an unpredictable STR instruction,
; e.g. str r0, [r0], #4

define i32* @earlyclobber-str-post(i32* %addr) nounwind {
; CHECK: earlyclobber-str-post
; CHECK-NOT: str r[[REG:[0-9]+]], [r[[REG]]], #4
  %val = ptrtoint i32* %addr to i32
  store i32 %val, i32* %addr
  %new = getelementptr i32* %addr, i32 1
  ret i32* %new
}
