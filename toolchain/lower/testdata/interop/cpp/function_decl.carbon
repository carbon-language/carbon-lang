// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/interop/cpp/function_decl.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/interop/cpp/function_decl.carbon

// ============================================================================
// function_decl
// ============================================================================

// --- function_decl.h

void foo();

// --- import_function_decl.carbon

library "[[@TEST_NAME]]";

import Cpp library "function_decl.h";

fn MyF() {
  Cpp.foo();
}

// ============================================================================
// inline_function_decl
// ============================================================================

// --- inline_function_decl.h

inline void foo() {}

// --- import_inline_function_decl.carbon

library "[[@TEST_NAME]]";

import Cpp library "inline_function_decl.h";

fn MyF() {
  Cpp.foo();
}

// ============================================================================
// inline_used_function_decl
// ============================================================================

// --- inline_used_function_decl.h

inline __attribute__((used)) void foo() {}

// --- import_inline_used_function_decl.carbon

library "[[@TEST_NAME]]";

import Cpp library "inline_used_function_decl.h";

fn MyF() {
  Cpp.foo();
}

// ============================================================================
// multiple_inline_function_decls
// ============================================================================

// --- multiple_inline_function_decls.h

inline void foo1() {}
inline void foo2() {}
inline void foo3() {}

// --- import_multiple_inline_function_decls.carbon

library "[[@TEST_NAME]]";

import Cpp library "multiple_inline_function_decls.h";

fn MyF() {
  Cpp.foo1();
  Cpp.foo2();
  Cpp.foo3();
}

// ============================================================================
// inline_recursive_function_decl
// ============================================================================

// --- inline_recursive_function_decl.h

inline void foo1() {}
inline void foo2() { foo1(); }

// --- import_inline_recursive_function_decl.carbon

library "[[@TEST_NAME]]";

import Cpp library "inline_recursive_function_decl.h";

fn MyF() {
  // This should generate the definition of the inline function `foo1()`.
  Cpp.foo2();
}

// ============================================================================
// Default arguments
// ============================================================================

// --- with_default_args.h

void NoReturnValue(int a = 1, int b = 2);
int SimpleReturnValue(int a = 1, int b = 2);

// --- call_with_default_args.carbon

library "[[@TEST_NAME]]";

import Cpp library "with_default_args.h";

fn MyF() {
  Cpp.NoReturnValue();
  Cpp.NoReturnValue(3);
  Cpp.NoReturnValue(3, 4);

  var value: i32 = Cpp.SimpleReturnValue();
  value = Cpp.SimpleReturnValue(3);
  value = Cpp.SimpleReturnValue(3, 4);
}

// CHECK:STDOUT: ; ModuleID = 'import_function_decl.carbon'
// CHECK:STDOUT: source_filename = "import_function_decl.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CMyF.Main() !dbg !7 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z3foov(), !dbg !10
// CHECK:STDOUT:   ret void, !dbg !11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z3foov()
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1, !2, !3, !4}
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = !{i32 1, !"wchar_size", i32 4}
// CHECK:STDOUT: !3 = !{i32 8, !"PIC Level", i32 0}
// CHECK:STDOUT: !4 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C, file: !6, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !6 = !DIFile(filename: "import_function_decl.carbon", directory: "")
// CHECK:STDOUT: !7 = distinct !DISubprogram(name: "MyF", linkageName: "_CMyF.Main", scope: null, file: !6, line: 6, type: !8, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !8 = !DISubroutineType(types: !9)
// CHECK:STDOUT: !9 = !{}
// CHECK:STDOUT: !10 = !DILocation(line: 7, column: 3, scope: !7)
// CHECK:STDOUT: !11 = !DILocation(line: 6, column: 1, scope: !7)
// CHECK:STDOUT: ; ModuleID = 'import_inline_function_decl.carbon'
// CHECK:STDOUT: source_filename = "import_inline_function_decl.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: $_Z3foov = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CMyF.Main() !dbg !7 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z3foov(), !dbg !10
// CHECK:STDOUT:   ret void, !dbg !11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress noinline nounwind optnone
// CHECK:STDOUT: define linkonce_odr dso_local void @_Z3foov() #0 comdat {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { mustprogress noinline nounwind optnone "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="0" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1, !2, !3, !4}
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = !{i32 1, !"wchar_size", i32 4}
// CHECK:STDOUT: !3 = !{i32 8, !"PIC Level", i32 0}
// CHECK:STDOUT: !4 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C, file: !6, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !6 = !DIFile(filename: "import_inline_function_decl.carbon", directory: "")
// CHECK:STDOUT: !7 = distinct !DISubprogram(name: "MyF", linkageName: "_CMyF.Main", scope: null, file: !6, line: 6, type: !8, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !8 = !DISubroutineType(types: !9)
// CHECK:STDOUT: !9 = !{}
// CHECK:STDOUT: !10 = !DILocation(line: 7, column: 3, scope: !7)
// CHECK:STDOUT: !11 = !DILocation(line: 6, column: 1, scope: !7)
// CHECK:STDOUT: ; ModuleID = 'import_inline_used_function_decl.carbon'
// CHECK:STDOUT: source_filename = "import_inline_used_function_decl.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: $_Z3foov = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: @llvm.compiler.used = appending global [1 x ptr] [ptr @_Z3foov], section "llvm.metadata"
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CMyF.Main() !dbg !7 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z3foov(), !dbg !10
// CHECK:STDOUT:   ret void, !dbg !11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress noinline nounwind optnone
// CHECK:STDOUT: define linkonce_odr dso_local void @_Z3foov() #0 comdat {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_Z3foov, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { mustprogress noinline nounwind optnone "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="0" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1, !2, !3, !4}
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = !{i32 1, !"wchar_size", i32 4}
// CHECK:STDOUT: !3 = !{i32 8, !"PIC Level", i32 0}
// CHECK:STDOUT: !4 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C, file: !6, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !6 = !DIFile(filename: "import_inline_used_function_decl.carbon", directory: "")
// CHECK:STDOUT: !7 = distinct !DISubprogram(name: "MyF", linkageName: "_CMyF.Main", scope: null, file: !6, line: 6, type: !8, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !8 = !DISubroutineType(types: !9)
// CHECK:STDOUT: !9 = !{}
// CHECK:STDOUT: !10 = !DILocation(line: 7, column: 3, scope: !7)
// CHECK:STDOUT: !11 = !DILocation(line: 6, column: 1, scope: !7)
// CHECK:STDOUT: ; ModuleID = 'import_multiple_inline_function_decls.carbon'
// CHECK:STDOUT: source_filename = "import_multiple_inline_function_decls.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: $_Z4foo1v = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_Z4foo2v = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_Z4foo3v = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CMyF.Main() !dbg !7 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z4foo1v(), !dbg !10
// CHECK:STDOUT:   call void @_Z4foo2v(), !dbg !11
// CHECK:STDOUT:   call void @_Z4foo3v(), !dbg !12
// CHECK:STDOUT:   ret void, !dbg !13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress noinline nounwind optnone
// CHECK:STDOUT: define linkonce_odr dso_local void @_Z4foo1v() #0 comdat {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress noinline nounwind optnone
// CHECK:STDOUT: define linkonce_odr dso_local void @_Z4foo2v() #0 comdat {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress noinline nounwind optnone
// CHECK:STDOUT: define linkonce_odr dso_local void @_Z4foo3v() #0 comdat {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { mustprogress noinline nounwind optnone "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="0" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1, !2, !3, !4}
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = !{i32 1, !"wchar_size", i32 4}
// CHECK:STDOUT: !3 = !{i32 8, !"PIC Level", i32 0}
// CHECK:STDOUT: !4 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C, file: !6, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !6 = !DIFile(filename: "import_multiple_inline_function_decls.carbon", directory: "")
// CHECK:STDOUT: !7 = distinct !DISubprogram(name: "MyF", linkageName: "_CMyF.Main", scope: null, file: !6, line: 6, type: !8, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !8 = !DISubroutineType(types: !9)
// CHECK:STDOUT: !9 = !{}
// CHECK:STDOUT: !10 = !DILocation(line: 7, column: 3, scope: !7)
// CHECK:STDOUT: !11 = !DILocation(line: 8, column: 3, scope: !7)
// CHECK:STDOUT: !12 = !DILocation(line: 9, column: 3, scope: !7)
// CHECK:STDOUT: !13 = !DILocation(line: 6, column: 1, scope: !7)
// CHECK:STDOUT: ; ModuleID = 'import_inline_recursive_function_decl.carbon'
// CHECK:STDOUT: source_filename = "import_inline_recursive_function_decl.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: $_Z4foo2v = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_Z4foo1v = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CMyF.Main() !dbg !7 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z4foo2v(), !dbg !10
// CHECK:STDOUT:   ret void, !dbg !11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress noinline optnone
// CHECK:STDOUT: define linkonce_odr dso_local void @_Z4foo2v() #0 comdat {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z4foo1v()
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress noinline nounwind optnone
// CHECK:STDOUT: define linkonce_odr dso_local void @_Z4foo1v() #1 comdat {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { mustprogress noinline optnone "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="0" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #1 = { mustprogress noinline nounwind optnone "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="0" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1, !2, !3, !4}
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = !{i32 1, !"wchar_size", i32 4}
// CHECK:STDOUT: !3 = !{i32 8, !"PIC Level", i32 0}
// CHECK:STDOUT: !4 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C, file: !6, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !6 = !DIFile(filename: "import_inline_recursive_function_decl.carbon", directory: "")
// CHECK:STDOUT: !7 = distinct !DISubprogram(name: "MyF", linkageName: "_CMyF.Main", scope: null, file: !6, line: 6, type: !8, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !8 = !DISubroutineType(types: !9)
// CHECK:STDOUT: !9 = !{}
// CHECK:STDOUT: !10 = !DILocation(line: 8, column: 3, scope: !7)
// CHECK:STDOUT: !11 = !DILocation(line: 6, column: 1, scope: !7)
// CHECK:STDOUT: ; ModuleID = 'call_with_default_args.carbon'
// CHECK:STDOUT: source_filename = "call_with_default_args.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CMyF.Main() !dbg !7 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %value.var = alloca i32, align 4, !dbg !10
// CHECK:STDOUT:   call void @_Z13NoReturnValueii.carbon_thunk0(), !dbg !11
// CHECK:STDOUT:   call void @_Z13NoReturnValueii.carbon_thunk1(i32 3), !dbg !12
// CHECK:STDOUT:   call void @_Z13NoReturnValueii(i32 3, i32 4), !dbg !13
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %value.var), !dbg !10
// CHECK:STDOUT:   %SimpleReturnValue__carbon_thunk.call.loc11 = call i32 @_Z17SimpleReturnValueii.carbon_thunk0(), !dbg !14
// CHECK:STDOUT:   store i32 %SimpleReturnValue__carbon_thunk.call.loc11, ptr %value.var, align 4, !dbg !10
// CHECK:STDOUT:   %SimpleReturnValue__carbon_thunk.call.loc12 = call i32 @_Z17SimpleReturnValueii.carbon_thunk1(i32 3), !dbg !15
// CHECK:STDOUT:   store i32 %SimpleReturnValue__carbon_thunk.call.loc12, ptr %value.var, align 4, !dbg !16
// CHECK:STDOUT:   %SimpleReturnValue.call = call i32 @_Z17SimpleReturnValueii(i32 3, i32 4), !dbg !17
// CHECK:STDOUT:   store i32 %SimpleReturnValue.call, ptr %value.var, align 4, !dbg !18
// CHECK:STDOUT:   ret void, !dbg !19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z13NoReturnValueii(i32, i32)
// CHECK:STDOUT:
// CHECK:STDOUT: declare i32 @_Z17SimpleReturnValueii(i32, i32)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #0
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress
// CHECK:STDOUT: define dso_local void @_Z13NoReturnValueii.carbon_thunk0() #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z13NoReturnValueii(i32 1, i32 2)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress
// CHECK:STDOUT: define dso_local void @_Z13NoReturnValueii.carbon_thunk1(i32 %a) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.addr = alloca i32, align 4
// CHECK:STDOUT:   store i32 %a, ptr %a.addr, align 4
// CHECK:STDOUT:   %0 = load i32, ptr %a.addr, align 4
// CHECK:STDOUT:   call void @_Z13NoReturnValueii(i32 %0, i32 2)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress
// CHECK:STDOUT: define dso_local i32 @_Z17SimpleReturnValueii.carbon_thunk0() #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %call = call i32 @_Z17SimpleReturnValueii(i32 1, i32 2)
// CHECK:STDOUT:   ret i32 %call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress
// CHECK:STDOUT: define dso_local i32 @_Z17SimpleReturnValueii.carbon_thunk1(i32 %a) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.addr = alloca i32, align 4
// CHECK:STDOUT:   store i32 %a, ptr %a.addr, align 4
// CHECK:STDOUT:   %0 = load i32, ptr %a.addr, align 4
// CHECK:STDOUT:   %call = call i32 @_Z17SimpleReturnValueii(i32 %0, i32 2)
// CHECK:STDOUT:   ret i32 %call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="0" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1, !2, !3, !4}
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = !{i32 1, !"wchar_size", i32 4}
// CHECK:STDOUT: !3 = !{i32 8, !"PIC Level", i32 0}
// CHECK:STDOUT: !4 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C, file: !6, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !6 = !DIFile(filename: "call_with_default_args.carbon", directory: "")
// CHECK:STDOUT: !7 = distinct !DISubprogram(name: "MyF", linkageName: "_CMyF.Main", scope: null, file: !6, line: 6, type: !8, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !8 = !DISubroutineType(types: !9)
// CHECK:STDOUT: !9 = !{}
// CHECK:STDOUT: !10 = !DILocation(line: 11, column: 3, scope: !7)
// CHECK:STDOUT: !11 = !DILocation(line: 7, column: 3, scope: !7)
// CHECK:STDOUT: !12 = !DILocation(line: 8, column: 3, scope: !7)
// CHECK:STDOUT: !13 = !DILocation(line: 9, column: 3, scope: !7)
// CHECK:STDOUT: !14 = !DILocation(line: 11, column: 20, scope: !7)
// CHECK:STDOUT: !15 = !DILocation(line: 12, column: 11, scope: !7)
// CHECK:STDOUT: !16 = !DILocation(line: 12, column: 3, scope: !7)
// CHECK:STDOUT: !17 = !DILocation(line: 13, column: 11, scope: !7)
// CHECK:STDOUT: !18 = !DILocation(line: 13, column: 3, scope: !7)
// CHECK:STDOUT: !19 = !DILocation(line: 6, column: 1, scope: !7)
