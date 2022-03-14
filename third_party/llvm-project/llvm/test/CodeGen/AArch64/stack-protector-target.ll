; Test target-specific stack cookie location.
; RUN: llc -mtriple=aarch64-linux-android < %s -o - | FileCheck --check-prefix=ANDROID-AARCH64 %s
; RUN: llc -mtriple=aarch64-fuchsia < %s -o - | FileCheck --check-prefixes=FUCHSIA-AARCH64-COMMON,FUCHSIA-AARCH64-USER %s
; RUN: llc -mtriple=aarch64-fuchsia -mattr=+tpidr-el1 < %s -o - | FileCheck --check-prefixes=FUCHSIA-AARCH64-COMMON,FUCHSIA-AARCH64-KERNEL %s
; RUN: llc -mtriple=aarch64-windows < %s -o - | FileCheck --check-prefix=WINDOWS-AARCH64 %s

define void @_Z1fv() sspreq {
entry:
  %x = alloca i32, align 4
  %0 = bitcast i32* %x to i8*
  call void @_Z7CapturePi(i32* nonnull %x)
  ret void
}

declare void @_Z7CapturePi(i32*)

; ANDROID-AARCH64: mrs [[A:.*]], TPIDR_EL0
; ANDROID-AARCH64: ldr [[B:.*]], [[[A]], #40]
; ANDROID-AARCH64: str [[B]], [sp,
; ANDROID-AARCH64: ldr [[C:.*]], [[[A]], #40]
; ANDROID-AARCH64: ldr [[D:.*]], [sp,
; ANDROID-AARCH64: cmp [[C]], [[D]]

; FUCHSIA-AARCH64-USER: mrs [[A:.*]], TPIDR_EL0
; FUCHSIA-AARCH64-KERNEL: mrs [[A:.*]], TPIDR_EL1
; FUCHSIA-AARCH64-COMMON: ldur [[B:.*]], [[[A]], #-16]
; FUCHSIA-AARCH64-COMMON: str [[B]], [sp,
; FUCHSIA-AARCH64-COMMON: ldur [[C:.*]], [[[A]], #-16]
; FUCHSIA-AARCH64-COMMON: ldr [[D:.*]], [sp,
; FUCHSIA-AARCH64-COMMON: cmp [[C]], [[D]]

; WINDOWS-AARCH64: adrp x8, __security_cookie
; WINDOWS-AARCH64: ldr x8, [x8, :lo12:__security_cookie]
; WINDOWS-AARCH64: str x8, [sp, #8]
; WINDOWS-AARCH64: bl  _Z7CapturePi
; WINDOWS-AARCH64: ldr x0, [sp, #8]
; WINDOWS-AARCH64: bl  __security_check_cookie
