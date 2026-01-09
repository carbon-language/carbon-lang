// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/interop/cpp/globals.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/interop/cpp/globals.carbon

// ============================================================================
// Global
// ============================================================================

// --- global.h

class C {};
C global;

// --- import_global.carbon

library "[[@TEST_NAME]]";

import Cpp library "global.h";

fn MyF() {
  let _: Cpp.C = Cpp.global;
}

// ============================================================================
// Inline variable
// ============================================================================

// --- inline.h

struct C {
  C();
  ~C();
  void f();
};
inline C inline_global;

// --- import_inline.carbon

library "[[@TEST_NAME]]";

import Cpp library "inline.h";

fn CallF() {
  Cpp.inline_global.f();
}

// ============================================================================
// Thread-local variable
// ============================================================================

// --- thread_local.h

thread_local int thread_local_global;

int init();
thread_local int thread_local_global_with_init = init();

// --- todo_import_thread_local.carbon

library "[[@TEST_NAME]]";

import Cpp library "thread_local.h";

// TODO: We should call the C++ thread wrapper function here instead of directly
// accessing the global. We fail to trigger the initialization of the global.

fn ThreadLocalStore() {
  Cpp.thread_local_global = 42;
}

fn ThreadLocalTriggerInit() -> i32 {
  return Cpp.thread_local_global_with_init;
}

// CHECK:STDOUT: ; ModuleID = 'import_global.carbon'
// CHECK:STDOUT: source_filename = "import_global.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: %class.C = type { i8 }
// CHECK:STDOUT:
// CHECK:STDOUT: @global = dso_local global %class.C zeroinitializer, align 1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMyF.Main() #0 !dbg !7 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1, !2, !3, !4}
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = !{i32 1, !"wchar_size", i32 4}
// CHECK:STDOUT: !3 = !{i32 8, !"PIC Level", i32 0}
// CHECK:STDOUT: !4 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !6, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !6 = !DIFile(filename: "import_global.carbon", directory: "")
// CHECK:STDOUT: !7 = distinct !DISubprogram(name: "MyF", linkageName: "_CMyF.Main", scope: null, file: !6, line: 6, type: !8, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !8 = !DISubroutineType(types: !9)
// CHECK:STDOUT: !9 = !{null}
// CHECK:STDOUT: !10 = !DILocation(line: 6, column: 1, scope: !7)
// CHECK:STDOUT: ; ModuleID = 'import_inline.carbon'
// CHECK:STDOUT: source_filename = "import_inline.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: %struct.C = type { i8 }
// CHECK:STDOUT:
// CHECK:STDOUT: $inline_global = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @__cxx_global_var_init, ptr @inline_global }]
// CHECK:STDOUT: @llvm.used = appending global [1 x ptr] [ptr @inline_global], section "llvm.metadata"
// CHECK:STDOUT: @inline_global = linkonce_odr dso_local global %struct.C zeroinitializer, comdat, align 1
// CHECK:STDOUT: @_ZGV13inline_global = linkonce_odr dso_local global i64 0, comdat($inline_global), align 8
// CHECK:STDOUT: @__dso_handle = external hidden global i8
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCallF.Main() #0 !dbg !7 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_ZN1C1fEv(ptr @inline_global), !dbg !10
// CHECK:STDOUT:   ret void, !dbg !11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_ZN1C1fEv(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: define internal void @__cxx_global_var_init() #1 section ".text.startup" comdat($inline_global) personality ptr @__gxx_personality_v0 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %exn.slot = alloca ptr, align 8
// CHECK:STDOUT:   %ehselector.slot = alloca i32, align 4
// CHECK:STDOUT:   %0 = load atomic i8, ptr @_ZGV13inline_global acquire, align 8
// CHECK:STDOUT:   %guard.uninitialized = icmp eq i8 %0, 0
// CHECK:STDOUT:   br i1 %guard.uninitialized, label %init.check, label %init.end
// CHECK:STDOUT:
// CHECK:STDOUT: init.check:                                       ; preds = %entry
// CHECK:STDOUT:   %1 = call i32 @__cxa_guard_acquire(ptr @_ZGV13inline_global) #0
// CHECK:STDOUT:   %tobool = icmp ne i32 %1, 0
// CHECK:STDOUT:   br i1 %tobool, label %init, label %init.end
// CHECK:STDOUT:
// CHECK:STDOUT: init:                                             ; preds = %init.check
// CHECK:STDOUT:   invoke void @_ZN1CC1Ev(ptr nonnull align 1 dereferenceable(1) @inline_global)
// CHECK:STDOUT:           to label %invoke.cont unwind label %lpad
// CHECK:STDOUT:
// CHECK:STDOUT: invoke.cont:                                      ; preds = %init
// CHECK:STDOUT:   %2 = call i32 @__cxa_atexit(ptr @_ZN1CD1Ev, ptr @inline_global, ptr @__dso_handle) #0
// CHECK:STDOUT:   call void @__cxa_guard_release(ptr @_ZGV13inline_global) #0
// CHECK:STDOUT:   br label %init.end
// CHECK:STDOUT:
// CHECK:STDOUT: init.end:                                         ; preds = %invoke.cont, %init.check, %entry
// CHECK:STDOUT:   ret void
// CHECK:STDOUT:
// CHECK:STDOUT: lpad:                                             ; preds = %init
// CHECK:STDOUT:   %3 = landingpad { ptr, i32 }
// CHECK:STDOUT:           cleanup
// CHECK:STDOUT:   %4 = extractvalue { ptr, i32 } %3, 0
// CHECK:STDOUT:   store ptr %4, ptr %exn.slot, align 8
// CHECK:STDOUT:   %5 = extractvalue { ptr, i32 } %3, 1
// CHECK:STDOUT:   store i32 %5, ptr %ehselector.slot, align 4
// CHECK:STDOUT:   call void @__cxa_guard_abort(ptr @_ZGV13inline_global) #0
// CHECK:STDOUT:   br label %eh.resume
// CHECK:STDOUT:
// CHECK:STDOUT: eh.resume:                                        ; preds = %lpad
// CHECK:STDOUT:   %exn = load ptr, ptr %exn.slot, align 8
// CHECK:STDOUT:   %sel = load i32, ptr %ehselector.slot, align 4
// CHECK:STDOUT:   %lpad.val = insertvalue { ptr, i32 } poison, ptr %exn, 0
// CHECK:STDOUT:   %lpad.val1 = insertvalue { ptr, i32 } %lpad.val, i32 %sel, 1
// CHECK:STDOUT:   resume { ptr, i32 } %lpad.val1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare i32 @__gxx_personality_v0(...)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: declare i32 @__cxa_guard_acquire(ptr) #0
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_ZN1CC1Ev(ptr nonnull align 1 dereferenceable(1)) unnamed_addr #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: declare void @_ZN1CD1Ev(ptr nonnull align 1 dereferenceable(1)) unnamed_addr #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: declare i32 @__cxa_atexit(ptr, ptr, ptr) #0
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: declare void @__cxa_guard_release(ptr) #0
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: declare void @__cxa_guard_abort(ptr) #0
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @inline_global, { 1, 2, 4, 3, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="0" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { "no-trapping-math"="true" "stack-protector-buffer-size"="0" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #3 = { nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="0" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1, !2, !3, !4}
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = !{i32 1, !"wchar_size", i32 4}
// CHECK:STDOUT: !3 = !{i32 8, !"PIC Level", i32 0}
// CHECK:STDOUT: !4 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !6, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !6 = !DIFile(filename: "import_inline.carbon", directory: "")
// CHECK:STDOUT: !7 = distinct !DISubprogram(name: "CallF", linkageName: "_CCallF.Main", scope: null, file: !6, line: 6, type: !8, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !8 = !DISubroutineType(types: !9)
// CHECK:STDOUT: !9 = !{null}
// CHECK:STDOUT: !10 = !DILocation(line: 7, column: 3, scope: !7)
// CHECK:STDOUT: !11 = !DILocation(line: 6, column: 1, scope: !7)
// CHECK:STDOUT: ; ModuleID = 'todo_import_thread_local.carbon'
// CHECK:STDOUT: source_filename = "todo_import_thread_local.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZTW19thread_local_global = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZTW29thread_local_global_with_init = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: @thread_local_global = dso_local thread_local global i32 0, align 4
// CHECK:STDOUT: @thread_local_global_with_init = dso_local thread_local global i32 0, align 4
// CHECK:STDOUT: @__tls_guard = internal thread_local global i8 0, align 1
// CHECK:STDOUT:
// CHECK:STDOUT: @_ZTH29thread_local_global_with_init = dso_local alias void (), ptr @__tls_init
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CThreadLocalStore.Main() #0 !dbg !7 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   store i32 42, ptr @thread_local_global, align 4, !dbg !10
// CHECK:STDOUT:   ret void, !dbg !11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CThreadLocalTriggerInit.Main() #0 !dbg !12 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc14 = load i32, ptr @thread_local_global_with_init, align 4, !dbg !16
// CHECK:STDOUT:   ret i32 %.loc14, !dbg !17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define weak_odr hidden ptr @_ZTW19thread_local_global() #1 comdat {
// CHECK:STDOUT:   %1 = call align 4 ptr @llvm.threadlocal.address.p0(ptr align 4 @thread_local_global)
// CHECK:STDOUT:   ret ptr %1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
// CHECK:STDOUT: declare nonnull ptr @llvm.threadlocal.address.p0(ptr nonnull) #2
// CHECK:STDOUT:
// CHECK:STDOUT: define weak_odr hidden ptr @_ZTW29thread_local_global_with_init() #1 comdat {
// CHECK:STDOUT:   call void @_ZTH29thread_local_global_with_init()
// CHECK:STDOUT:   %1 = call align 4 ptr @llvm.threadlocal.address.p0(ptr align 4 @thread_local_global_with_init)
// CHECK:STDOUT:   ret ptr %1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define internal void @__tls_init() #3 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %0 = load i8, ptr @__tls_guard, align 1
// CHECK:STDOUT:   %guard.uninitialized = icmp eq i8 %0, 0
// CHECK:STDOUT:   br i1 %guard.uninitialized, label %init, label %exit, !prof !18
// CHECK:STDOUT:
// CHECK:STDOUT: init:                                             ; preds = %entry
// CHECK:STDOUT:   store i8 1, ptr @__tls_guard, align 1
// CHECK:STDOUT:   call void @__cxx_global_var_init()
// CHECK:STDOUT:   br label %exit
// CHECK:STDOUT:
// CHECK:STDOUT: exit:                                             ; preds = %init, %entry
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define internal void @__cxx_global_var_init() #3 section ".text.startup" {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %call = call i32 @_Z4initv()
// CHECK:STDOUT:   %0 = call align 4 ptr @llvm.threadlocal.address.p0(ptr align 4 @thread_local_global_with_init)
// CHECK:STDOUT:   store i32 %call, ptr %0, align 4
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare i32 @_Z4initv() #1
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { "no-trapping-math"="true" "stack-protector-buffer-size"="0" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
// CHECK:STDOUT: attributes #3 = { "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="0" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1, !2, !3, !4}
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = !{i32 1, !"wchar_size", i32 4}
// CHECK:STDOUT: !3 = !{i32 8, !"PIC Level", i32 0}
// CHECK:STDOUT: !4 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !6, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !6 = !DIFile(filename: "todo_import_thread_local.carbon", directory: "")
// CHECK:STDOUT: !7 = distinct !DISubprogram(name: "ThreadLocalStore", linkageName: "_CThreadLocalStore.Main", scope: null, file: !6, line: 9, type: !8, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !8 = !DISubroutineType(types: !9)
// CHECK:STDOUT: !9 = !{null}
// CHECK:STDOUT: !10 = !DILocation(line: 10, column: 3, scope: !7)
// CHECK:STDOUT: !11 = !DILocation(line: 9, column: 1, scope: !7)
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "ThreadLocalTriggerInit", linkageName: "_CThreadLocalTriggerInit.Main", scope: null, file: !6, line: 13, type: !13, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !13 = !DISubroutineType(types: !14)
// CHECK:STDOUT: !14 = !{!15}
// CHECK:STDOUT: !15 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !16 = !DILocation(line: 14, column: 10, scope: !12)
// CHECK:STDOUT: !17 = !DILocation(line: 14, column: 3, scope: !12)
// CHECK:STDOUT: !18 = !{!"branch_weights", i32 1, i32 1023}
