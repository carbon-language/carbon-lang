; RUN: llc -mtriple=armv7-linux-gnueabihf %s -o - | FileCheck %s

; Check that we don't create an unpredictable STR instruction,
; e.g. str r0, [r0], #4

define i32* @earlyclobber-str-post(i32* %addr) nounwind {
; CHECK-LABEL: earlyclobber-str-post
; CHECK-NOT: str r[[REG:[0-9]+]], [r[[REG]]], #4
  %val = ptrtoint i32* %addr to i32
  store i32 %val, i32* %addr
  %new = getelementptr i32, i32* %addr, i32 1
  ret i32* %new
}

define i16* @earlyclobber-strh-post(i16* %addr) nounwind {
; CHECK-LABEL: earlyclobber-strh-post
; CHECK-NOT: strh r[[REG:[0-9]+]], [r[[REG]]], #2
  %val = ptrtoint i16* %addr to i32
  %tr = trunc i32 %val to i16
  store i16 %tr, i16* %addr
  %new = getelementptr i16, i16* %addr, i32 1
  ret i16* %new
}

define i8* @earlyclobber-strb-post(i8* %addr) nounwind {
; CHECK-LABEL: earlyclobber-strb-post
; CHECK-NOT: strb r[[REG:[0-9]+]], [r[[REG]]], #1
  %val = ptrtoint i8* %addr to i32
  %tr = trunc i32 %val to i8
  store i8 %tr, i8* %addr
  %new = getelementptr i8, i8* %addr, i32 1
  ret i8* %new
}
