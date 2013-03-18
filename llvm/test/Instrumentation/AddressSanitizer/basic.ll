; Test basic address sanitizer instrumentation.
;
; RUN: opt < %s -asan -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

define i32 @test_load(i32* %a) sanitize_address {
; CHECK: @test_load
; CHECK-NOT: load
; CHECK:   %[[LOAD_ADDR:[^ ]*]] = ptrtoint i32* %a to i64
; CHECK:   lshr i64 %[[LOAD_ADDR]], 3
; CHECK:   {{or|add}}
; CHECK:   %[[LOAD_SHADOW_PTR:[^ ]*]] = inttoptr
; CHECK:   %[[LOAD_SHADOW:[^ ]*]] = load i8* %[[LOAD_SHADOW_PTR]]
; CHECK:   icmp ne i8
; CHECK:   br i1 %{{.*}}, label %{{.*}}, label %{{.*}}
;
; First instrumentation block refines the shadow test.
; CHECK:   and i64 %[[LOAD_ADDR]], 7
; CHECK:   add i64 %{{.*}}, 3
; CHECK:   trunc i64 %{{.*}} to i8
; CHECK:   icmp sge i8 %{{.*}}, %[[LOAD_SHADOW]]
; CHECK:   br i1 %{{.*}}, label %{{.*}}, label %{{.*}}
;
; The crash block reports the error.
; CHECK:   call void @__asan_report_load4(i64 %[[LOAD_ADDR]])
; CHECK:   unreachable
;
; The actual load.
; CHECK:   %tmp1 = load i32* %a
; CHECK:   ret i32 %tmp1



entry:
  %tmp1 = load i32* %a
  ret i32 %tmp1
}

define void @test_store(i32* %a) sanitize_address {
; CHECK: @test_store
; CHECK-NOT: store
; CHECK:   %[[STORE_ADDR:[^ ]*]] = ptrtoint i32* %a to i64
; CHECK:   lshr i64 %[[STORE_ADDR]], 3
; CHECK:   {{or|add}}
; CHECK:   %[[STORE_SHADOW_PTR:[^ ]*]] = inttoptr
; CHECK:   %[[STORE_SHADOW:[^ ]*]] = load i8* %[[STORE_SHADOW_PTR]]
; CHECK:   icmp ne i8
; CHECK:   br i1 %{{.*}}, label %{{.*}}, label %{{.*}}
;
; First instrumentation block refines the shadow test.
; CHECK:   and i64 %[[STORE_ADDR]], 7
; CHECK:   add i64 %{{.*}}, 3
; CHECK:   trunc i64 %{{.*}} to i8
; CHECK:   icmp sge i8 %{{.*}}, %[[STORE_SHADOW]]
; CHECK:   br i1 %{{.*}}, label %{{.*}}, label %{{.*}}
;
; The crash block reports the error.
; CHECK:   call void @__asan_report_store4(i64 %[[STORE_ADDR]])
; CHECK:   unreachable
;
; The actual load.
; CHECK:   store i32 42, i32* %a
; CHECK:   ret void
;

entry:
  store i32 42, i32* %a
  ret void
}

; Check that asan leaves just one alloca.

declare void @alloca_test_use([10 x i8]*)
define void @alloca_test() sanitize_address {
entry:
  %x = alloca [10 x i8], align 1
  %y = alloca [10 x i8], align 1
  %z = alloca [10 x i8], align 1
  call void @alloca_test_use([10 x i8]* %x)
  call void @alloca_test_use([10 x i8]* %y)
  call void @alloca_test_use([10 x i8]* %z)
  ret void
}

; CHECK: define void @alloca_test()
; CHECK: = alloca
; CHECK-NOT: = alloca
; CHECK: ret void

define void @LongDoubleTest(x86_fp80* nocapture %a) nounwind uwtable sanitize_address {
entry:
    store x86_fp80 0xK3FFF8000000000000000, x86_fp80* %a, align 16
    ret void
}

; CHECK: LongDoubleTest
; CHECK: __asan_report_store_n
; CHECK: __asan_report_store_n
; CHECK: ret void


define void @i40test(i40* %a, i40* %b) nounwind uwtable sanitize_address {
  entry:
  %t = load i40* %a
  store i40 %t, i40* %b, align 8
  ret void
}

; CHECK: i40test
; CHECK: __asan_report_load_n{{.*}}, i64 5)
; CHECK: __asan_report_load_n{{.*}}, i64 5)
; CHECK: __asan_report_store_n{{.*}}, i64 5)
; CHECK: __asan_report_store_n{{.*}}, i64 5)
; CHECK: ret void

define void @i80test(i80* %a, i80* %b) nounwind uwtable sanitize_address {
  entry:
  %t = load i80* %a
  store i80 %t, i80* %b, align 8
  ret void
}

; CHECK: i80test
; CHECK: __asan_report_load_n{{.*}}, i64 10)
; CHECK: __asan_report_load_n{{.*}}, i64 10)
; CHECK: __asan_report_store_n{{.*}}, i64 10)
; CHECK: __asan_report_store_n{{.*}}, i64 10)
; CHECK: ret void

; asan should not instrument functions with available_externally linkage.
define available_externally i32 @f_available_externally(i32* %a) sanitize_address  {
entry:
  %tmp1 = load i32* %a
  ret i32 %tmp1
}
; CHECK: @f_available_externally
; CHECK-NOT: __asan_report
; CHECK: ret i32


