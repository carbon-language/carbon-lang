; Test section group of the function with linkonce_odr

; The instantiation of C++ function template will come with linkonce_odr,
; which indicates that the linker can remove the duplicated instantiation.
; However, to make this feature work, we have to group the section properly.
; .text, .ARM.extab, and .ARM.exidx should be grouped together.

; RUN: llc -mtriple arm-unknown-linux-gnueabi \
; RUN:     -arm-enable-ehabi -arm-enable-ehabi-descriptors \
; RUN:     -filetype=obj -o - %s \
; RUN:   | llvm-readobj -s -sd \
; RUN:   | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:64:128-a0:0:64-n32-S64"
target triple = "armv4t--linux-gnueabi"

define void @_Z11instantiatev() {
entry:
  tail call void @_Z4testIidEvT_S0_S0_S0_S0_T0_S1_S1_S1_S1_(i32 1, i32 2, i32 3, i32 4, i32 5, double 1.000000e-01, double 2.000000e-01, double 3.000000e-01, double 4.000000e-01, double 5.000000e-01)
  ret void
}

define linkonce_odr void @_Z4testIidEvT_S0_S0_S0_S0_T0_S1_S1_S1_S1_(i32 %u1, i32 %u2, i32 %u3, i32 %u4, i32 %u5, double %v1, double %v2, double %v3, double %v4, double %v5) {
entry:
  invoke void @_Z5printiiiii(i32 %u1, i32 %u2, i32 %u3, i32 %u4, i32 %u5)
          to label %try.cont unwind label %lpad

lpad:                                             ; preds = %entry
  %0 = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*)
          catch i8* null
  %1 = extractvalue { i8*, i32 } %0, 0
  %2 = tail call i8* @__cxa_begin_catch(i8* %1) nounwind
  invoke void @_Z5printddddd(double %v1, double %v2, double %v3, double %v4, double %v5)
          to label %invoke.cont2 unwind label %lpad1

invoke.cont2:                                     ; preds = %lpad
  tail call void @__cxa_end_catch()
  br label %try.cont

try.cont:                                         ; preds = %entry, %invoke.cont2
  ret void

lpad1:                                            ; preds = %lpad
  %3 = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*)
          cleanup
  invoke void @__cxa_end_catch()
          to label %eh.resume unwind label %terminate.lpad

eh.resume:                                        ; preds = %lpad1
  resume { i8*, i32 } %3

terminate.lpad:                                   ; preds = %lpad1
  %4 = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*)
          catch i8* null
  tail call void @_ZSt9terminatev() noreturn nounwind
  unreachable
}

declare void @_Z5printiiiii(i32, i32, i32, i32, i32)

declare i32 @__gxx_personality_v0(...)

declare i8* @__cxa_begin_catch(i8*)

declare void @_Z5printddddd(double, double, double, double, double)

declare void @__cxa_end_catch()

declare void @_ZSt9terminatev()

; CHECK:        Section {
; CHECK:          Index: 1
; CHECK-NEXT:     Name: .group (47)
; CHECK:          SectionData (
; CHECK-NEXT:       0000: 01000000 0A000000 0C000000 0E000000
; CHECK-NEXT:     )

; CHECK:        Section {
; CHECK:          Index: 10
; CHECK-NEXT:     Name: .text._Z4testIidEvT_S0_S0_S0_S0_T0_S1_S1_S1_S1_ (225)

; CHECK:        Section {
; CHECK:          Index: 12
; CHECK-NEXT:     Name: .ARM.extab.text._Z4testIidEvT_S0_S0_S0_S0_T0_S1_S1_S1_S1_ (215)

; CHECK:        Section {
; CHECK:          Index: 14
; CHECK-NEXT:     Name: .ARM.exidx.text._Z4testIidEvT_S0_S0_S0_S0_T0_S1_S1_S1_S1_ (101)
