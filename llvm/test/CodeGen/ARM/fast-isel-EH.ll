; RUN: llc < %s -O0 -relocation-model=dynamic-no-pic -mtriple=armv7-apple-darwin | FileCheck %s --check-prefix=ARM
; RUN: llc < %s -O0 -relocation-model=dynamic-no-pic -mtriple=thumbv7-apple-darwin | FileCheck %s --check-prefix=THUMB

@"\01L_OBJC_IMAGE_INFO" = internal constant [2 x i32] [i32 0, i32 16], section "__DATA, __objc_imageinfo, regular, no_dead_strip"
@llvm.used = appending global [1 x i8*] [i8* bitcast ([2 x i32]* @"\01L_OBJC_IMAGE_INFO" to i8*)], section "llvm.metadata"

define i32 @f1(i32 %return_in_finally) {
entry:
  %retval = alloca i32, align 4
  %return_in_finally.addr = alloca i32, align 4
  %finally.for-eh = alloca i1
  %cleanup.dest.slot = alloca i32
  %exn.slot = alloca i8*
  %ehselector.slot = alloca i32
  store i32 %return_in_finally, i32* %return_in_finally.addr, align 4
  store i1 false, i1* %finally.for-eh
  %cleanup.dest.saved = load i32* %cleanup.dest.slot
  %finally.shouldthrow = load i1* %finally.for-eh
  br i1 %finally.shouldthrow, label %finally.rethrow, label %finally.cont

finally.rethrow:                                  ; preds = %entry
  invoke void @objc_exception_rethrow()
          to label %invoke.cont unwind label %lpad

invoke.cont:                                      ; preds = %finally.rethrow
  unreachable

finally.cont:                                     ; preds = %entry
  store i32 %cleanup.dest.saved, i32* %cleanup.dest.slot
  %0 = load i32* %retval
  ret i32 %0

lpad:                                             ; preds = %finally.rethrow
  %1 = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__objc_personality_v0 to i8*)
          cleanup
  %2 = extractvalue { i8*, i32 } %1, 0
  store i8* %2, i8** %exn.slot
  %3 = extractvalue { i8*, i32 } %1, 1
  store i32 %3, i32* %ehselector.slot
  %finally.endcatch = load i1* %finally.for-eh
  br i1 %finally.endcatch, label %finally.endcatch1, label %finally.cleanup.cont

finally.endcatch1:                                ; preds = %lpad
  invoke void @objc_end_catch()
          to label %invoke.cont2 unwind label %terminate.lpad

invoke.cont2:                                     ; preds = %finally.endcatch1
  br label %finally.cleanup.cont

finally.cleanup.cont:                             ; preds = %invoke.cont2, %lpad
  br label %eh.resume

eh.resume:                                        ; preds = %finally.cleanup.cont
; ARM: eh.resume
; ARM: mvn r0, #0
; ARM: ldr r1, [sp, #84]
; ARM: ldr r2, [sp, #80]
; ARM: ldr r3, [sp, #72]
; ARM: str r0, [r3]
; ARM: str r1, [sp, #28]
; ARM: str r2, [sp, #24]

; THUMB: eh.resume
; THUMB: movw r0, #65535
; THUMB: movt r0, #65535
; THUMB: ldr r1, [sp, #80]
; THUMB: ldr r2, [sp, #76]
; THUMB: ldr r3, [sp, #68]
; THUMB: str r0, [r3]
; THUMB: str r1, [sp, #24]
; THUMB: str r2, [sp, #20]

  %exn = load i8** %exn.slot
  %sel = load i32* %ehselector.slot
  %lpad.val = insertvalue { i8*, i32 } undef, i8* %exn, 0
  %lpad.val3 = insertvalue { i8*, i32 } %lpad.val, i32 %sel, 1
  resume { i8*, i32 } %lpad.val3

terminate.lpad:                                   ; preds = %finally.endcatch1
  %4 = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__objc_personality_v0 to i8*)
          catch i8* null
  call void @abort() noreturn nounwind
  unreachable
}

declare i8* @objc_begin_catch(i8*)

declare void @objc_end_catch()

declare void @objc_exception_rethrow()

declare i32 @__objc_personality_v0(...)

declare void @abort()
