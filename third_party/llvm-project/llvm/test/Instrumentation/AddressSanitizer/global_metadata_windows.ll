; Test that global metadata is placed in a separate section on Windows, and that
; it is in the same comdat group as the instrumented global. This ensures that
; linker dead stripping (/OPT:REF) works as intended.

; FIXME: Later we can use this to instrument linkonce odr string literals.

; RUN: opt < %s -passes='asan-pipeline' -asan-globals-live-support=1 -S | FileCheck %s

target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.0.24215"

$mystr = comdat any

; CHECK: $dead_global = comdat nodeduplicate
; CHECK: $private_str = comdat nodeduplicate

; CHECK: @dead_global = global { i32, [28 x i8] } { i32 42, [28 x i8] zeroinitializer }, comdat, align 32
; CHECK: @private_str = internal constant { [8 x i8], [24 x i8] } { [8 x i8] c"private\00", [24 x i8] zeroinitializer }, comdat, align 32

; CHECK: @__asan_global_dead_global = private global { {{.*}} }, section ".ASAN$GL", comdat($dead_global), align 64, !associated
; CHECK: @__asan_global_private_str = private global { {{.*}} }, section ".ASAN$GL", comdat($private_str), align 64, !associated

; CHECK: @llvm.compiler.used {{.*}} @__asan_global_dead_global {{.*}} @__asan_global_private_str {{.*}} section "llvm.metadata"

@dead_global = local_unnamed_addr global i32 42, align 4
@mystr = linkonce_odr unnamed_addr constant [5 x i8] c"main\00", comdat, align 1

; Private globals will get upgraded to internal linkage.
@private_str = private unnamed_addr constant [8 x i8] c"private\00", align 1

; Function Attrs: nounwind uwtable
define i32 @main() local_unnamed_addr #0 {
entry:
  %call = tail call i32 @puts(i8* getelementptr inbounds ([5 x i8], [5 x i8]* @mystr, i64 0, i64 0))
  %call2 = tail call i32 @puts(i8* getelementptr inbounds ([8 x i8], [8 x i8]* @private_str, i64 0, i64 0))
  ret i32 0
}

; Function Attrs: nounwind
declare i32 @puts(i8* nocapture readonly) local_unnamed_addr #1

attributes #0 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"PIC Level", i32 2}
!1 = !{!"clang version 4.0.0 "}
