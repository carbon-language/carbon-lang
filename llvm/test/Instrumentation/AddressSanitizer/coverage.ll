; RUN: opt < %s -asan -asan-module -asan-coverage=0 -S | FileCheck %s --check-prefix=CHECK0
; RUN: opt < %s -asan -asan-module -asan-coverage=1 -S | FileCheck %s --check-prefix=CHECK1
; RUN: opt < %s -asan -asan-module -asan-coverage=2 -S | FileCheck %s --check-prefix=CHECK2
; RUN: opt < %s -asan -asan-module -asan-coverage=2 -asan-coverage-block-threshold=10 -S | FileCheck %s --check-prefix=CHECK2
; RUN: opt < %s -asan -asan-module -asan-coverage=2 -asan-coverage-block-threshold=1  -S | FileCheck %s --check-prefix=CHECK1
; RUN: opt < %s -asan -asan-module -asan-coverage=3 -asan-coverage-block-threshold=10 -S | FileCheck %s --check-prefix=CHECK3
; RUN: opt < %s -asan -asan-module -asan-coverage=4 -S | FileCheck %s --check-prefix=CHECK4

; RUN: opt < %s -asan -asan-module -asan-coverage=0 -asan-globals=0 -S | \
; RUN:     FileCheck %s --check-prefix=CHECK0
; RUN: opt < %s -asan -asan-module -asan-coverage=1 -asan-globals=0 -S | \
; RUN:     FileCheck %s --check-prefix=CHECK1
; RUN: opt < %s -asan -asan-module -asan-coverage=2 -asan-globals=0 -S | \
; RUN:     FileCheck %s --check-prefix=CHECK2
; RUN: opt < %s -asan -asan-module -asan-coverage=2 -asan-coverage-block-threshold=10 \
; RUN:     -asan-globals=0 -S | FileCheck %s --check-prefix=CHECK2
; RUN: opt < %s -asan -asan-module -asan-coverage=2 -asan-coverage-block-threshold=1 \
; RUN:     -asan-globals=0 -S | FileCheck %s --check-prefix=CHECK1

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"
define void @foo(i32* %a) sanitize_address {
entry:
  %tobool = icmp eq i32* %a, null
  br i1 %tobool, label %if.end, label %if.then

  if.then:                                          ; preds = %entry
  store i32 0, i32* %a, align 4
  br label %if.end

  if.end:                                           ; preds = %entry, %if.then
  ret void
}

; CHECK0-NOT: call void @__sanitizer_cov(
; CHECK0-NOT: call void @__sanitizer_cov_module_init(

; CHECK1-LABEL: define void @foo
; CHECK1: %0 = load atomic i8* @__asan_gen_cov_foo monotonic, align 1
; CHECK1: %1 = icmp eq i8 0, %0
; CHECK1: br i1 %1, label %2, label %3
; CHECK1: call void @__sanitizer_cov
; CHECK1-NOT: call void @__sanitizer_cov
; CHECK1: store atomic i8 1, i8* @__asan_gen_cov_foo monotonic, align 1

; CHECK1-LABEL: define internal void @asan.module_ctor
; CHECK1-NOT: ret
; CHECK1: call void @__sanitizer_cov_module_init(i64 2)
; CHECK1: ret


; CHECK2-LABEL: define void @foo
; CHECK2: call void @__sanitizer_cov
; CHECK2: call void @__sanitizer_cov
; CHECK2: call void @__sanitizer_cov
; CHECK2-NOT: call void @__sanitizer_cov
; CHECK2: ret void

; CHECK2-LABEL: define internal void @asan.module_ctor
; CHECK2-NOT: ret
; CHECK2: call void @__sanitizer_cov_module_init(i64 4)
; CHECK2: ret

; CHECK3-LABEL: define void @foo
; CHECK3: call void @__sanitizer_cov
; CHECK3: call void @__sanitizer_cov
; CHECK3: call void @__sanitizer_cov
; CHECK3: call void @__sanitizer_cov
; CHECK3-NOT: call void @__sanitizer_cov
; CHECK3: ret void


%struct.StructWithVptr = type { i32 (...)** }

define void @CallViaVptr(%struct.StructWithVptr* %foo) uwtable sanitize_address {
entry:
  %0 = bitcast %struct.StructWithVptr* %foo to void (%struct.StructWithVptr*)***
  %vtable = load void (%struct.StructWithVptr*)*** %0, align 8
  %1 = load void (%struct.StructWithVptr*)** %vtable, align 8
  tail call void %1(%struct.StructWithVptr* %foo)
  tail call void %1(%struct.StructWithVptr* %foo)
  tail call void asm sideeffect "", ""()
  ret void
}

; We expect to see two calls to __sanitizer_cov_indir_call16
; with different values of second argument.
; CHECK4-LABEL: define void @CallViaVptr
; CHECK4: call void @__sanitizer_cov_indir_call16({{.*}},[[CACHE:.*]])
; CHECK4-NOT: call void @__sanitizer_cov_indir_call16({{.*}},[[CACHE]])
; CHECK4: ret void
