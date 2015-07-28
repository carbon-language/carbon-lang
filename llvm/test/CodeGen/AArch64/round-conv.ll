; RUN: llc < %s -mtriple=arm64-apple-ios7.0 | FileCheck %s --check-prefix=CHECK-INEXACT
; RUN: llc < %s -mtriple=arm64-apple-ios7.0 -enable-unsafe-fp-math | FileCheck %s --check-prefix=CHECK-FAST
; RUN: llc < %s -mtriple=aarch64-linux-gnu | FileCheck %s --check-prefix=CHECK-FAST

; CHECK-INEXACT-LABEL: testmsws:
; CHECK-INEXACT: fcvtms w0, s0
; CHECK-INEXACT: frintx {{s[0-9]+}}, s0

; CHECK-FAST-LABEL: testmsws:
; CHECK-FAST: fcvtms w0, s0
; CHECK-FAST-NOT: frintx {{s[0-9]+}}, s0
define i32 @testmsws(float %a) {
entry:
  %call = call float @floorf(float %a) nounwind readnone
  %conv = fptosi float %call to i32
  ret i32 %conv
}

; CHECK-INEXACT-LABEL: testmsxs:
; CHECK-INEXACT: fcvtms x0, s0
; CHECK-INEXACT: frintx {{s[0-9]+}}, s0

; CHECK-FAST-LABEL: testmsxs:
; CHECK-FAST: fcvtms x0, s0
; CHECK-FAST-NOT: frintx {{s[0-9]+}}, s0
define i64 @testmsxs(float %a) {
entry:
  %call = call float @floorf(float %a) nounwind readnone
  %conv = fptosi float %call to i64
  ret i64 %conv
}

; CHECK-INEXACT-LABEL: testmswd:
; CHECK-INEXACT: fcvtms w0, d0
; CHECK-INEXACT: frintx {{d[0-9]+}}, d0

; CHECK-FAST-LABEL: testmswd:
; CHECK-FAST: fcvtms w0, d0
; CHECK-FAST-NOT: frintx {{d[0-9]+}}, d0
define i32 @testmswd(double %a) {
entry:
  %call = call double @floor(double %a) nounwind readnone
  %conv = fptosi double %call to i32
  ret i32 %conv
}

; CHECK-INEXACT-LABEL: testmsxd:
; CHECK-INEXACT: fcvtms x0, d0
; CHECK-INEXACT: frintx {{d[0-9]+}}, d0

; CHECK-FAST-LABEL: testmsxd:
; CHECK-FAST: fcvtms x0, d0
; CHECK-FAST-NOT: frintx {{d[0-9]+}}, d0
define i64 @testmsxd(double %a) {
entry:
  %call = call double @floor(double %a) nounwind readnone
  %conv = fptosi double %call to i64
  ret i64 %conv
}

; CHECK-INEXACT-LABEL: testmuws:
; CHECK-INEXACT: fcvtmu w0, s0
; CHECK-INEXACT: frintx {{s[0-9]+}}, s0

; CHECK-FAST-LABEL: testmuws:
; CHECK-FAST: fcvtmu w0, s0
; CHECK-FAST-NOT: frintx {{s[0-9]+}}, s0
define i32 @testmuws(float %a) {
entry:
  %call = call float @floorf(float %a) nounwind readnone
  %conv = fptoui float %call to i32
  ret i32 %conv
}

; CHECK-INEXACT-LABEL: testmuxs:
; CHECK-INEXACT: fcvtmu x0, s0
; CHECK-INEXACT: frintx {{s[0-9]+}}, s0

; CHECK-FAST-LABEL: testmuxs:
; CHECK-FAST: fcvtmu x0, s0
; CHECK-FAST-NOT: frintx {{s[0-9]+}}, s0
define i64 @testmuxs(float %a) {
entry:
  %call = call float @floorf(float %a) nounwind readnone
  %conv = fptoui float %call to i64
  ret i64 %conv
}

; CHECK-INEXACT-LABEL: testmuwd:
; CHECK-INEXACT: fcvtmu w0, d0
; CHECK-INEXACT: frintx {{d[0-9]+}}, d0

; CHECK-FAST-LABEL: testmuwd:
; CHECK-FAST: fcvtmu w0, d0
; CHECK-FAST-NOT: frintx {{d[0-9]+}}, d0
define i32 @testmuwd(double %a) {
entry:
  %call = call double @floor(double %a) nounwind readnone
  %conv = fptoui double %call to i32
  ret i32 %conv
}

; CHECK-INEXACT-LABEL: testmuxd:
; CHECK-INEXACT: fcvtmu x0, d0
; CHECK-INEXACT: frintx {{d[0-9]+}}, d0

; CHECK-FAST-LABEL: testmuxd:
; CHECK-FAST: fcvtmu x0, d0
; CHECK-FAST-NOT: frintx {{d[0-9]+}}, d0
define i64 @testmuxd(double %a) {
entry:
  %call = call double @floor(double %a) nounwind readnone
  %conv = fptoui double %call to i64
  ret i64 %conv
}

; CHECK-INEXACT-LABEL: testpsws:
; CHECK-INEXACT: fcvtps w0, s0
; CHECK-INEXACT: frintx {{s[0-9]+}}, s0

; CHECK-FAST-LABEL: testpsws:
; CHECK-FAST: fcvtps w0, s0
; CHECK-FAST-NOT: frintx {{s[0-9]+}}, s0
define i32 @testpsws(float %a) {
entry:
  %call = call float @ceilf(float %a) nounwind readnone
  %conv = fptosi float %call to i32
  ret i32 %conv
}

; CHECK-INEXACT-LABEL: testpsxs:
; CHECK-INEXACT: fcvtps x0, s0
; CHECK-INEXACT: frintx {{s[0-9]+}}, s0

; CHECK-FAST-LABEL: testpsxs:
; CHECK-FAST: fcvtps x0, s0
; CHECK-FAST-NOT: frintx {{s[0-9]+}}, s0
define i64 @testpsxs(float %a) {
entry:
  %call = call float @ceilf(float %a) nounwind readnone
  %conv = fptosi float %call to i64
  ret i64 %conv
}

; CHECK-INEXACT-LABEL: testpswd:
; CHECK-INEXACT: fcvtps w0, d0
; CHECK-INEXACT: frintx {{d[0-9]+}}, d0

; CHECK-FAST-LABEL: testpswd:
; CHECK-FAST: fcvtps w0, d0
; CHECK-FAST-NOT: frintx {{d[0-9]+}}, d0
define i32 @testpswd(double %a) {
entry:
  %call = call double @ceil(double %a) nounwind readnone
  %conv = fptosi double %call to i32
  ret i32 %conv
}

; CHECK-INEXACT-LABEL: testpsxd:
; CHECK-INEXACT: fcvtps x0, d0
; CHECK-INEXACT: frintx {{d[0-9]+}}, d0

; CHECK-FAST-LABEL: testpsxd:
; CHECK-FAST: fcvtps x0, d0
; CHECK-FAST-NOT: frintx {{d[0-9]+}}, d0
define i64 @testpsxd(double %a) {
entry:
  %call = call double @ceil(double %a) nounwind readnone
  %conv = fptosi double %call to i64
  ret i64 %conv
}

; CHECK-INEXACT-LABEL: testpuws:
; CHECK-INEXACT: fcvtpu w0, s0
; CHECK-INEXACT: frintx {{s[0-9]+}}, s0

; CHECK-FAST-LABEL: testpuws:
; CHECK-FAST: fcvtpu w0, s0
; CHECK-FAST-NOT: frintx {{s[0-9]+}}, s0
define i32 @testpuws(float %a) {
entry:
  %call = call float @ceilf(float %a) nounwind readnone
  %conv = fptoui float %call to i32
  ret i32 %conv
}

; CHECK-INEXACT-LABEL: testpuxs:
; CHECK-INEXACT: fcvtpu x0, s0
; CHECK-INEXACT: frintx {{s[0-9]+}}, s0

; CHECK-FAST-LABEL: testpuxs:
; CHECK-FAST: fcvtpu x0, s0
; CHECK-FAST-NOT: frintx {{s[0-9]+}}, s0
define i64 @testpuxs(float %a) {
entry:
  %call = call float @ceilf(float %a) nounwind readnone
  %conv = fptoui float %call to i64
  ret i64 %conv
}

; CHECK-INEXACT-LABEL: testpuwd:
; CHECK-INEXACT: fcvtpu w0, d0
; CHECK-INEXACT: frintx {{d[0-9]+}}, d0

; CHECK-FAST-LABEL: testpuwd:
; CHECK-FAST: fcvtpu w0, d0
; CHECK-FAST-NOT: frintx {{d[0-9]+}}, d0
define i32 @testpuwd(double %a) {
entry:
  %call = call double @ceil(double %a) nounwind readnone
  %conv = fptoui double %call to i32
  ret i32 %conv
}

; CHECK-INEXACT-LABEL: testpuxd:
; CHECK-INEXACT: fcvtpu x0, d0
; CHECK-INEXACT: frintx {{d[0-9]+}}, d0

; CHECK-FAST-LABEL: testpuxd:
; CHECK-FAST: fcvtpu x0, d0
; CHECK-FAST-NOT: frintx {{d[0-9]+}}, d0
define i64 @testpuxd(double %a) {
entry:
  %call = call double @ceil(double %a) nounwind readnone
  %conv = fptoui double %call to i64
  ret i64 %conv
}

; CHECK-INEXACT-LABEL: testzsws:
; CHECK-INEXACT: fcvtzs w0, s0
; CHECK-INEXACT: frintx {{s[0-9]+}}, s0

; CHECK-FAST-LABEL: testzsws:
; CHECK-FAST: fcvtzs w0, s0
; CHECK-FAST-NOT: frintx {{s[0-9]+}}, s0
define i32 @testzsws(float %a) {
entry:
  %call = call float @truncf(float %a) nounwind readnone
  %conv = fptosi float %call to i32
  ret i32 %conv
}

; CHECK-INEXACT-LABEL: testzsxs:
; CHECK-INEXACT: fcvtzs x0, s0
; CHECK-INEXACT: frintx {{s[0-9]+}}, s0

; CHECK-FAST-LABEL: testzsxs:
; CHECK-FAST: fcvtzs x0, s0
; CHECK-FAST-NOT: frintx {{s[0-9]+}}, s0
define i64 @testzsxs(float %a) {
entry:
  %call = call float @truncf(float %a) nounwind readnone
  %conv = fptosi float %call to i64
  ret i64 %conv
}

; CHECK-INEXACT-LABEL: testzswd:
; CHECK-INEXACT: fcvtzs w0, d0
; CHECK-INEXACT: frintx {{d[0-9]+}}, d0

; CHECK-FAST-LABEL: testzswd:
; CHECK-FAST: fcvtzs w0, d0
; CHECK-FAST-NOT: frintx {{d[0-9]+}}, d0
define i32 @testzswd(double %a) {
entry:
  %call = call double @trunc(double %a) nounwind readnone
  %conv = fptosi double %call to i32
  ret i32 %conv
}

; CHECK-INEXACT-LABEL: testzsxd:
; CHECK-INEXACT: fcvtzs x0, d0
; CHECK-INEXACT: frintx {{d[0-9]+}}, d0

; CHECK-FAST-LABEL: testzsxd:
; CHECK-FAST: fcvtzs x0, d0
; CHECK-FAST-NOT: frintx {{d[0-9]+}}, d0
define i64 @testzsxd(double %a) {
entry:
  %call = call double @trunc(double %a) nounwind readnone
  %conv = fptosi double %call to i64
  ret i64 %conv
}

; CHECK-INEXACT-LABEL: testzuws:
; CHECK-INEXACT: fcvtzu w0, s0
; CHECK-INEXACT: frintx {{s[0-9]+}}, s0

; CHECK-FAST-LABEL: testzuws:
; CHECK-FAST: fcvtzu w0, s0
; CHECK-FAST-NOT: frintx {{s[0-9]+}}, s0
define i32 @testzuws(float %a) {
entry:
  %call = call float @truncf(float %a) nounwind readnone
  %conv = fptoui float %call to i32
  ret i32 %conv
}

; CHECK-INEXACT-LABEL: testzuxs:
; CHECK-INEXACT: fcvtzu x0, s0
; CHECK-INEXACT: frintx {{s[0-9]+}}, s0

; CHECK-FAST-LABEL: testzuxs:
; CHECK-FAST: fcvtzu x0, s0
; CHECK-FAST-NOT: frintx {{s[0-9]+}}, s0
define i64 @testzuxs(float %a) {
entry:
  %call = call float @truncf(float %a) nounwind readnone
  %conv = fptoui float %call to i64
  ret i64 %conv
}

; CHECK-INEXACT-LABEL: testzuwd:
; CHECK-INEXACT: fcvtzu w0, d0
; CHECK-INEXACT: frintx {{d[0-9]+}}, d0

; CHECK-FAST-LABEL: testzuwd:
; CHECK-FAST: fcvtzu w0, d0
; CHECK-FAST-NOT: frintx {{d[0-9]+}}, d0
define i32 @testzuwd(double %a) {
entry:
  %call = call double @trunc(double %a) nounwind readnone
  %conv = fptoui double %call to i32
  ret i32 %conv
}

; CHECK-INEXACT-LABEL: testzuxd:
; CHECK-INEXACT: fcvtzu x0, d0
; CHECK-INEXACT: frintx {{d[0-9]+}}, d0

; CHECK-FAST-LABEL: testzuxd:
; CHECK-FAST: fcvtzu x0, d0
; CHECK-FAST-NOT: frintx {{d[0-9]+}}, d0
define i64 @testzuxd(double %a) {
entry:
  %call = call double @trunc(double %a) nounwind readnone
  %conv = fptoui double %call to i64
  ret i64 %conv
}

; CHECK-INEXACT-LABEL: testasws:
; CHECK-INEXACT: fcvtas w0, s0
; CHECK-INEXACT: frintx {{s[0-9]+}}, s0

; CHECK-FAST-LABEL: testasws:
; CHECK-FAST: fcvtas w0, s0
; CHECK-FAST-NOT: frintx {{s[0-9]+}}, s0
define i32 @testasws(float %a) {
entry:
  %call = call float @roundf(float %a) nounwind readnone
  %conv = fptosi float %call to i32
  ret i32 %conv
}

; CHECK-INEXACT-LABEL: testasxs:
; CHECK-INEXACT: fcvtas x0, s0
; CHECK-INEXACT: frintx {{s[0-9]+}}, s0

; CHECK-FAST-LABEL: testasxs:
; CHECK-FAST: fcvtas x0, s0
; CHECK-FAST-NOT: frintx {{s[0-9]+}}, s0
define i64 @testasxs(float %a) {
entry:
  %call = call float @roundf(float %a) nounwind readnone
  %conv = fptosi float %call to i64
  ret i64 %conv
}

; CHECK-INEXACT-LABEL: testaswd:
; CHECK-INEXACT: fcvtas w0, d0
; CHECK-INEXACT: frintx {{d[0-9]+}}, d0

; CHECK-FAST-LABEL: testaswd:
; CHECK-FAST: fcvtas w0, d0
; CHECK-FAST-NOT: frintx {{d[0-9]+}}, d0
define i32 @testaswd(double %a) {
entry:
  %call = call double @round(double %a) nounwind readnone
  %conv = fptosi double %call to i32
  ret i32 %conv
}

; CHECK-INEXACT-LABEL: testasxd:
; CHECK-INEXACT: fcvtas x0, d0
; CHECK-INEXACT: frintx {{d[0-9]+}}, d0

; CHECK-FAST-LABEL: testasxd:
; CHECK-FAST: fcvtas x0, d0
; CHECK-FAST-NOT: frintx {{d[0-9]+}}, d0
define i64 @testasxd(double %a) {
entry:
  %call = call double @round(double %a) nounwind readnone
  %conv = fptosi double %call to i64
  ret i64 %conv
}

; CHECK-INEXACT-LABEL: testauws:
; CHECK-INEXACT: fcvtau w0, s0
; CHECK-INEXACT: frintx {{s[0-9]+}}, s0

; CHECK-FAST-LABEL: testauws:
; CHECK-FAST: fcvtau w0, s0
; CHECK-FAST-NOT: frintx {{s[0-9]+}}, s0
define i32 @testauws(float %a) {
entry:
  %call = call float @roundf(float %a) nounwind readnone
  %conv = fptoui float %call to i32
  ret i32 %conv
}

; CHECK-INEXACT-LABEL: testauxs:
; CHECK-INEXACT: fcvtau x0, s0
; CHECK-INEXACT: frintx {{s[0-9]+}}, s0

; CHECK-FAST-LABEL: testauxs:
; CHECK-FAST: fcvtau x0, s0
; CHECK-FAST-NOT: frintx {{s[0-9]+}}, s0
define i64 @testauxs(float %a) {
entry:
  %call = call float @roundf(float %a) nounwind readnone
  %conv = fptoui float %call to i64
  ret i64 %conv
}

; CHECK-INEXACT-LABEL: testauwd:
; CHECK-INEXACT: fcvtau w0, d0
; CHECK-INEXACT: frintx {{d[0-9]+}}, d0

; CHECK-FAST-LABEL: testauwd:
; CHECK-FAST: fcvtau w0, d0
; CHECK-FAST-NOT: frintx {{d[0-9]+}}, d0
define i32 @testauwd(double %a) {
entry:
  %call = call double @round(double %a) nounwind readnone
  %conv = fptoui double %call to i32
  ret i32 %conv
}

; CHECK-INEXACT-LABEL: testauxd:
; CHECK-INEXACT: fcvtau x0, d0
; CHECK-INEXACT: frintx {{d[0-9]+}}, d0

; CHECK-FAST-LABEL: testauxd:
; CHECK-FAST: fcvtau x0, d0
; CHECK-FAST-NOT: frintx {{d[0-9]+}}, d0
define i64 @testauxd(double %a) {
entry:
  %call = call double @round(double %a) nounwind readnone
  %conv = fptoui double %call to i64
  ret i64 %conv
}


declare float @floorf(float) nounwind readnone
declare double @floor(double) nounwind readnone
declare float @ceilf(float) nounwind readnone
declare double @ceil(double) nounwind readnone
declare float @truncf(float) nounwind readnone
declare double @trunc(double) nounwind readnone
declare float @roundf(float) nounwind readnone
declare double @round(double) nounwind readnone
