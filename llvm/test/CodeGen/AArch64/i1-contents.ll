; RUN: llc -mtriple=aarch64-linux-gnu -o - %s | FileCheck %s
%big = type i32

@var = dso_local global %big 0

; AAPCS: low 8 bits of %in (== w0) will be either 0 or 1. Need to extend to
; 32-bits.
define dso_local void @consume_i1_arg(i1 %in) {
; CHECK-LABEL: consume_i1_arg:
; CHECK: and [[BOOL32:w[0-9]+]], w0, #{{0x1|0xff}}
; CHECK: str [[BOOL32]], [{{x[0-9]+}}, :lo12:var]
  %val = zext i1 %in to %big
  store %big %val, %big* @var
  ret void
}

; AAPCS: low 8 bits of %val1 (== w0) will be either 0 or 1. Need to extend to
; 32-bits (doesn't really matter if it's from 1 or 8 bits).
define dso_local void @consume_i1_ret() {
; CHECK-LABEL: consume_i1_ret:
; CHECK: bl produce_i1_ret
; CHECK: and [[BOOL32:w[0-9]+]], w0, #{{0x1|0xff}}
; CHECK: str [[BOOL32]], [{{x[0-9]+}}, :lo12:var]
  %val1 = call i1 @produce_i1_ret()
  %val = zext i1 %val1 to %big
  store %big %val, %big* @var
  ret void
}

; AAPCS: low 8 bits of w0 must be either 0 or 1. Need to mask them off.
define dso_local i1 @produce_i1_ret() {
; CHECK-LABEL: produce_i1_ret:
; CHECK: ldr [[VAR32:w[0-9]+]], [{{x[0-9]+}}, :lo12:var]
; CHECK: and w0, [[VAR32]], #{{0x1|0xff}}
  %val = load %big, %big* @var
  %val1 = trunc %big %val to i1
  ret i1 %val1
}

define dso_local void @produce_i1_arg() {
; CHECK-LABEL: produce_i1_arg:
; CHECK: ldr [[VAR32:w[0-9]+]], [{{x[0-9]+}}, :lo12:var]
; CHECK: and w0, [[VAR32]], #{{0x1|0xff}}
; CHECK: bl consume_i1_arg
  %val = load %big, %big* @var
  %val1 = trunc %big %val to i1
  call void @consume_i1_arg(i1 %val1)
  ret void
}


;define zeroext i1 @foo(i8 %in) {
;  %val = trunc i8 %in to i1
;  ret i1 %val
;}
