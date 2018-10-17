; RUN: %clang_cc1 -S -emit-llvm -o - -fexperimental-new-pass-manager -fsanitize=address %s | FileCheck %s

; CHECK: @llvm.global_ctors = {{.*}}@asan.module_ctor

define i32 @test_load(i32* %a) sanitize_address {
entry:
; CHECK:  %0 = ptrtoint i32* %a to i64
; CHECK:  %1 = lshr i64 %0, 3
; CHECK:  %2 = add i64 %1, 2147450880
; CHECK:  %3 = inttoptr i64 %2 to i8*
; CHECK:  %4 = load i8, i8* %3
; CHECK:  %5 = icmp ne i8 %4, 0
; CHECK:  br i1 %5, label %6, label %12, !prof !0

; CHECK:; <label>:6:                                      ; preds = %entry
; CHECK:  %7 = and i64 %0, 7
; CHECK:  %8 = add i64 %7, 3
; CHECK:  %9 = trunc i64 %8 to i8
; CHECK:  %10 = icmp sge i8 %9, %4
; CHECK:  br i1 %10, label %11, label %12

; CHECK:; <label>:11:                                     ; preds = %6
; CHECK:  call void @__asan_report_load4(i64 %0)
; CHECK:  call void asm sideeffect "", ""()
; CHECK:  unreachable

; CHECK:; <label>:12:                                     ; preds = %6, %entry

  %tmp1 = load i32, i32* %a, align 4
  ret i32 %tmp1
}
