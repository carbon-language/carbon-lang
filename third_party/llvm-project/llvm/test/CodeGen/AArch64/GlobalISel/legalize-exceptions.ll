; RUN: llc -O0 -mtriple=aarch64-apple-ios -verify-machineinstrs -global-isel -stop-after=legalizer %s -o - | FileCheck %s

@_ZTIi = external global i8*

declare i32 @foo(i32)
declare i32 @__gxx_personality_v0(...)
declare i32 @llvm.eh.typeid.for(i8*)
declare void @_Unwind_Resume(i8*)

; CHECK: name: bar
; CHECK: body:
; CHECK-NEXT:   bb.1 (%ir-block.0):
; CHECK:     successors: %{{bb.[0-9]+.*}}%[[LP:bb.[0-9]+]]

; CHECK:   [[LP]].{{[a-z]+}} (landing-pad):
; CHECK:     EH_LABEL

; CHECK:     [[PTR:%[0-9]+]]:_(p0) = COPY $x0
; CHECK:     [[SEL_PTR:%[0-9]+]]:_(p0) = COPY $x1
; CHECK:     [[SEL_PTR_INT:%[0-9]+]]:_(s32) = G_PTRTOINT [[SEL_PTR]](p0)
; CHECK:     G_STORE [[PTR]](p0), %0(p0) :: (store (p0) into %ir.exn.slot)
; CHECK:     G_STORE [[SEL_PTR_INT]](s32), %1(p0) :: (store (s32) into %ir.ehselector.slot)

define void @bar() personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
  %exn.slot = alloca i8*
  %ehselector.slot = alloca i32
  %1 = invoke i32 @foo(i32 42) to label %continue unwind label %cleanup

cleanup:
  %2 = landingpad { i8*, i32 } cleanup
  %3 = extractvalue { i8*, i32 } %2, 0
  store i8* %3, i8** %exn.slot, align 8
  %4 = extractvalue { i8*, i32 } %2, 1
  store i32 %4, i32* %ehselector.slot, align 4
  br label %eh.resume

continue:
  ret void

eh.resume:
  %exn = load i8*, i8** %exn.slot, align 8
  call void @_Unwind_Resume(i8* %exn)
  unreachable
}
