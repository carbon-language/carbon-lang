; RUN: opt -mergefunc -S < %s | FileCheck %s

;; Make sure that two different sized allocas are not treated as equal.

target datalayout = "e-m:w-p:32:32-i64:64-f80:32-n8:16:32-S32"

%kv1 = type { i32, i32 }
%kv2 = type { i8 }


define void @a(i8 *%f) {
  %v = alloca %kv1, align 8
  %f_2 = bitcast i8* %f to void (%kv1 *)*
  call void %f_2(%kv1 * %v)
  call void %f_2(%kv1 * %v)
  call void %f_2(%kv1 * %v)
  call void %f_2(%kv1 * %v)
  ret void
}

; CHECK-LABEL: define void @b
; CHECK-NOT: call @a
; CHECK: ret

define void @b(i8 *%f) {
  %v = alloca %kv2, align 8
  %f_2 = bitcast i8* %f to void (%kv2 *)*
  call void %f_2(%kv2 * %v)
  call void %f_2(%kv2 * %v)
  call void %f_2(%kv2 * %v)
  call void %f_2(%kv2 * %v)
  ret void
}
