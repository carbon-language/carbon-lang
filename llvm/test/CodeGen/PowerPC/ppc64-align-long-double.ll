; RUN: llc -verify-machineinstrs -mcpu=pwr7 -O0 -fast-isel=false -mattr=-vsx < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -mcpu=pwr7 -O0 -fast-isel=false -mattr=+vsx < %s | FileCheck -check-prefix=CHECK-VSX %s
; RUN: llc -verify-machineinstrs -mcpu=pwr9 -O0 -fast-isel=false -mattr=+vsx < %s | FileCheck -check-prefix=CHECK-P9 %s

; Verify internal alignment of long double in a struct.  The double
; argument comes in in GPR3; GPR4 is skipped; GPRs 5 and 6 contain
; the long double.  Check that these are stored to proper locations
; in the parameter save area and loaded from there for return in FPR1/2.

target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

%struct.S = type { double, ppc_fp128 }

define ppc_fp128 @test(%struct.S* byval %x) nounwind {
entry:
  %b = getelementptr inbounds %struct.S, %struct.S* %x, i32 0, i32 1
  %0 = load ppc_fp128, ppc_fp128* %b, align 16
  ret ppc_fp128 %0
}

;; FIXME: Sadly, we now have an extra store to a temp variable here,
;; which comes from (roughly):
;;  store i64 <val> to i64* <frame>
;;  bitcast (load i64* <frame>) to f64
;; The code now can elide the load, making:
;;  store i64 <val> -> <frame>
;;  bitcast i64 <val> to f64
;; Finally, the bitcast itself turns into a store/load pair.
;;
;; This behavior is new, because previously, llvm was accidentally
;; unable to detect that the load came directly from the store, and
;; elide it.

; CHECK: std 6, 72(1)
; CHECK: std 5, 64(1)
; CHECK: std 4, 56(1)
; CHECK: std 3, 48(1)
; CHECK: std 5, -16(1)
; CHECK: std 6, -8(1)
; CHECK: lfd 1, -16(1)
; CHECK: lfd 2, -8(1)

; CHECK-VSX: std 6, 72(1)
; CHECK-VSX: std 5, 64(1)
; CHECK-VSX: std 4, 56(1)
; CHECK-VSX: std 3, 48(1)
; CHECK-VSX: std 5, -16(1)
; CHECK-VSX: std 6, -8(1)
; CHECK-VSX: addi 3, 1, -16
; CHECK-VSX: lxsdx 1, 0, 3
; CHECK-VSX: addi 3, 1, -8
; CHECK-VSX: lxsdx 2, 0, 3

; CHECK-P9: std 6, 72(1)
; CHECK-P9: std 5, 64(1)
; CHECK-P9: std 4, 56(1)
; CHECK-P9: std 3, 48(1)
; CHECK-P9: mtvsrd 1, 5
; CHECK-P9: mtvsrd 2, 6
