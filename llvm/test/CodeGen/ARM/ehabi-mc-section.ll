; RUN: llc -mtriple armv7-unknown-linux-gnueabi \
; RUN:     -arm-enable-ehabi -arm-enable-ehabi-descriptors \
; RUN:     -disable-fp-elim -filetype=obj -o - %s \
; RUN:   | llvm-objdump -s - \
; RUN:   | FileCheck %s --check-prefix=CHECK

; RUN: llc -mtriple armv7-unknown-linux-gnueabi \
; RUN:     -arm-enable-ehabi -arm-enable-ehabi-descriptors \
; RUN:     -filetype=obj -o - %s \
; RUN:   | llvm-objdump -s - \
; RUN:   | FileCheck %s --check-prefix=CHECK-FP-ELIM

define void @_Z4testiiiiiddddd(i32 %u1, i32 %u2, i32 %u3, i32 %u4, i32 %u5, double %v1, double %v2, double %v3, double %v4, double %v5) section ".test_section" {
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

; CHECK: section .test_section
; CHECK: section .ARM.extab.test_section
; CHECK-NEXT: 0000 00000000 c9409b01 b0818484
; CHECK: section .ARM.exidx.test_section
; CHECK-NEXT: 0000 00000000 00000000

; CHECK-FP-ELIM: section .test_section
; CHECK-FP-ELIM: section .ARM.extab.test_section
; CHECK-FP-ELIM-NEXT: 0000 00000000 84c90501 b0b0b0a8
; CHECK-FP-ELIM: section .ARM.exidx.test_section
; CHECK-FP-ELIM-NEXT: 0000 00000000 00000000
