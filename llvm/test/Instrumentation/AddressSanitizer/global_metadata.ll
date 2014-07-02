; RUN: opt < %s -asan -asan-module -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Globals:
@global = global i32 0, align 4
@dyn_init_global = global i32 0, align 4
@blacklisted_global = global i32 0, align 4
@_ZZ4funcvE10static_var = internal global i32 0, align 4
@.str = private unnamed_addr constant [14 x i8] c"Hello, world!\00", align 1
@llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__sub_I_asan_globals.cpp, i8* null }]

; Sanitizer location descriptors:
@.str1 = private unnamed_addr constant [22 x i8] c"/tmp/asan-globals.cpp\00", align 1
@.asan_loc_descr = private unnamed_addr constant { [22 x i8]*, i32, i32 } { [22 x i8]* @.str1, i32 5, i32 5 }
@.asan_loc_descr1 = private unnamed_addr constant { [22 x i8]*, i32, i32 } { [22 x i8]* @.str1, i32 7, i32 5 }
@.asan_loc_descr2 = private unnamed_addr constant { [22 x i8]*, i32, i32 } { [22 x i8]* @.str1, i32 12, i32 14 }
@.asan_loc_descr4 = private unnamed_addr constant { [22 x i8]*, i32, i32 } { [22 x i8]* @.str1, i32 14, i32 25 }

; Check that globals were instrumented, but sanitizer location descriptors weren't:
; CHECK: @global = global { i32, [60 x i8] } zeroinitializer, align 32
; CHECK: @.str = internal unnamed_addr constant { [14 x i8], [50 x i8] } { [14 x i8] c"Hello, world!\00", [50 x i8] zeroinitializer }, align 32
; CHECK: @.asan_loc_descr = private unnamed_addr constant { [22 x i8]*, i32, i32 } { [22 x i8]* @.str1, i32 5, i32 5 }

; Check that location decriptors were passed into __asan_register_globals:
; CHECK: i64 ptrtoint ({ [22 x i8]*, i32, i32 }* @.asan_loc_descr to i64)

; Function Attrs: nounwind sanitize_address
define internal void @__cxx_global_var_init() #0 section ".text.startup" {
entry:
  %0 = load i32* @global, align 4
  store i32 %0, i32* @dyn_init_global, align 4
  ret void
}

; Function Attrs: nounwind sanitize_address
define void @_Z4funcv() #1 {
entry:
  %literal = alloca i8*, align 8
  store i8* getelementptr inbounds ([14 x i8]* @.str, i32 0, i32 0), i8** %literal, align 8
  ret void
}

; Function Attrs: nounwind sanitize_address
define internal void @_GLOBAL__sub_I_asan_globals.cpp() #0 section ".text.startup" {
entry:
  call void @__cxx_global_var_init()
  ret void
}

attributes #0 = { nounwind sanitize_address }
attributes #1 = { nounwind sanitize_address "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.asan.globals = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = metadata !{i32* @global, { [22 x i8]*, i32, i32 }* @.asan_loc_descr, i1 false, i1 false}
!1 = metadata !{i32* @dyn_init_global, { [22 x i8]*, i32, i32 }* @.asan_loc_descr1, i1 true, i1 false}
!2 = metadata !{i32* @blacklisted_global, null, i1 false, i1 true}
!3 = metadata !{i32* @_ZZ4funcvE10static_var, { [22 x i8]*, i32, i32 }* @.asan_loc_descr2, i1 false, i1 false}
!4 = metadata !{[14 x i8]* @.str, { [22 x i8]*, i32, i32 }* @.asan_loc_descr4, i1 false, i1 false}
!5 = metadata !{metadata !"clang version 3.5.0 (211282)"}
