// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/interop/cpp/function/import/function_decl.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/interop/cpp/function/import/function_decl.carbon

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

inline void foo_inline() {}

// --- import_inline_function_decl.carbon

library "[[@TEST_NAME]]";

import Cpp library "inline_function_decl.h";

fn MyF() {
  Cpp.foo_inline();
}

// ============================================================================
// inline_used_function_decl
// ============================================================================

// --- inline_used_function_decl.h

inline __attribute__((used)) void foo_used() {}

// --- import_inline_used_function_decl.carbon

library "[[@TEST_NAME]]";

import Cpp library "inline_used_function_decl.h";

fn MyF() {
  Cpp.foo_used();
}

// ============================================================================
// multiple_inline_function_decls
// ============================================================================

// --- multiple_inline_function_decls.h

inline void multi_foo1() {}
inline void multi_foo2() {}
inline void multi_foo3() {}

// --- import_multiple_inline_function_decls.carbon

library "[[@TEST_NAME]]";

import Cpp library "multiple_inline_function_decls.h";

fn MyF() {
  Cpp.multi_foo1();
  Cpp.multi_foo2();
  Cpp.multi_foo3();
}

// ============================================================================
// inline_recursive_function_decl
// ============================================================================

// --- inline_recursive_function_decl.h

inline void rec_foo1() {}
inline void rec_foo2() { rec_foo1(); }

// --- import_inline_recursive_function_decl.carbon

library "[[@TEST_NAME]]";

import Cpp library "inline_recursive_function_decl.h";

fn MyF() {
  // This should generate the definition of the inline function `rec_foo1()`.
  Cpp.rec_foo2();
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

fn MyF() -> i32 {
  Cpp.NoReturnValue();
  Cpp.NoReturnValue(3);
  Cpp.NoReturnValue(3, 4);

  var value: i32 = Cpp.SimpleReturnValue();
  value = Cpp.SimpleReturnValue(3);
  value = Cpp.SimpleReturnValue(3, 4);
  return value;
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'import_function_decl.carbon'
// CHECK:STDOUT: source_filename = "import_function_decl.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMyF.Main() #0 !dbg !22 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z3foov(), !dbg !23
// CHECK:STDOUT:   ret void, !dbg !24
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z3foov() #1
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!19}
// CHECK:STDOUT: !llvm.module.flags = !{!25, !26, !27, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!32}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !18 = !DIFile(filename: "import_function_decl.carbon", directory: "")
// CHECK:STDOUT: !19 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !18, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !20 = !{null}
// CHECK:STDOUT: !21 = !DISubroutineType(types: !20)
// CHECK:STDOUT: !22 = distinct !DISubprogram(name: "MyF", linkageName: "_CMyF.Main", scope: null, file: !18, line: 6, type: !21, spFlags: DISPFlagDefinition, unit: !19)
// CHECK:STDOUT: !23 = !DILocation(line: 7, column: 3, scope: !22)
// CHECK:STDOUT: !24 = !DILocation(line: 6, column: 1, scope: !22)
// CHECK:STDOUT: !25 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !26 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !27 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !28 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !29 = !{!"omnipotent char", !28, i64 0}
// CHECK:STDOUT: !30 = !{!"int", !29, i64 0}
// CHECK:STDOUT: !31 = !{!"__libc_errno", !30, i64 0}
// CHECK:STDOUT: !32 = !{!31, !30, i64 0}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'import_inline_function_decl.carbon'
// CHECK:STDOUT: source_filename = "import_inline_function_decl.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: $_Z10foo_inlinev = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMyF.Main() #0 !dbg !35 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z10foo_inlinev(), !dbg !36
// CHECK:STDOUT:   ret void, !dbg !37
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_Z10foo_inlinev() #1 comdat {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { inlinehint mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!34}
// CHECK:STDOUT: !llvm.module.flags = !{!25, !26, !27, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!32}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !{null}
// CHECK:STDOUT: !21 = !DISubroutineType(types: !20)
// CHECK:STDOUT: !25 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !26 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !27 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !28 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !29 = !{!"omnipotent char", !28, i64 0}
// CHECK:STDOUT: !30 = !{!"int", !29, i64 0}
// CHECK:STDOUT: !31 = !{!"__libc_errno", !30, i64 0}
// CHECK:STDOUT: !32 = !{!31, !30, i64 0}
// CHECK:STDOUT: !33 = !DIFile(filename: "import_inline_function_decl.carbon", directory: "")
// CHECK:STDOUT: !34 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !33, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !35 = distinct !DISubprogram(name: "MyF", linkageName: "_CMyF.Main", scope: null, file: !33, line: 6, type: !21, spFlags: DISPFlagDefinition, unit: !34)
// CHECK:STDOUT: !36 = !DILocation(line: 7, column: 3, scope: !35)
// CHECK:STDOUT: !37 = !DILocation(line: 6, column: 1, scope: !35)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'import_inline_used_function_decl.carbon'
// CHECK:STDOUT: source_filename = "import_inline_used_function_decl.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: $_Z8foo_usedv = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: @llvm.compiler.used = appending global [1 x ptr] [ptr @_Z8foo_usedv], section "llvm.metadata"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_Z8foo_usedv() #0 comdat {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMyF.Main() #1 !dbg !40 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z8foo_usedv(), !dbg !41
// CHECK:STDOUT:   ret void, !dbg !42
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_Z8foo_usedv, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { inlinehint mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #1 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!39}
// CHECK:STDOUT: !llvm.module.flags = !{!25, !26, !27, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!32}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !{null}
// CHECK:STDOUT: !21 = !DISubroutineType(types: !20)
// CHECK:STDOUT: !25 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !26 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !27 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !28 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !29 = !{!"omnipotent char", !28, i64 0}
// CHECK:STDOUT: !30 = !{!"int", !29, i64 0}
// CHECK:STDOUT: !31 = !{!"__libc_errno", !30, i64 0}
// CHECK:STDOUT: !32 = !{!31, !30, i64 0}
// CHECK:STDOUT: !38 = !DIFile(filename: "import_inline_used_function_decl.carbon", directory: "")
// CHECK:STDOUT: !39 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !38, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !40 = distinct !DISubprogram(name: "MyF", linkageName: "_CMyF.Main", scope: null, file: !38, line: 6, type: !21, spFlags: DISPFlagDefinition, unit: !39)
// CHECK:STDOUT: !41 = !DILocation(line: 7, column: 3, scope: !40)
// CHECK:STDOUT: !42 = !DILocation(line: 6, column: 1, scope: !40)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'import_multiple_inline_function_decls.carbon'
// CHECK:STDOUT: source_filename = "import_multiple_inline_function_decls.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: $_Z10multi_foo1v = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_Z10multi_foo2v = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_Z10multi_foo3v = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMyF.Main() #0 !dbg !45 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z10multi_foo1v(), !dbg !46
// CHECK:STDOUT:   call void @_Z10multi_foo2v(), !dbg !47
// CHECK:STDOUT:   call void @_Z10multi_foo3v(), !dbg !48
// CHECK:STDOUT:   ret void, !dbg !49
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_Z10multi_foo1v() #1 comdat {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_Z10multi_foo2v() #1 comdat {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_Z10multi_foo3v() #1 comdat {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { inlinehint mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!44}
// CHECK:STDOUT: !llvm.module.flags = !{!25, !26, !27, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!32}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !{null}
// CHECK:STDOUT: !21 = !DISubroutineType(types: !20)
// CHECK:STDOUT: !25 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !26 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !27 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !28 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !29 = !{!"omnipotent char", !28, i64 0}
// CHECK:STDOUT: !30 = !{!"int", !29, i64 0}
// CHECK:STDOUT: !31 = !{!"__libc_errno", !30, i64 0}
// CHECK:STDOUT: !32 = !{!31, !30, i64 0}
// CHECK:STDOUT: !43 = !DIFile(filename: "import_multiple_inline_function_decls.carbon", directory: "")
// CHECK:STDOUT: !44 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !43, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !45 = distinct !DISubprogram(name: "MyF", linkageName: "_CMyF.Main", scope: null, file: !43, line: 6, type: !21, spFlags: DISPFlagDefinition, unit: !44)
// CHECK:STDOUT: !46 = !DILocation(line: 7, column: 3, scope: !45)
// CHECK:STDOUT: !47 = !DILocation(line: 8, column: 3, scope: !45)
// CHECK:STDOUT: !48 = !DILocation(line: 9, column: 3, scope: !45)
// CHECK:STDOUT: !49 = !DILocation(line: 6, column: 1, scope: !45)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'import_inline_recursive_function_decl.carbon'
// CHECK:STDOUT: source_filename = "import_inline_recursive_function_decl.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: $_Z8rec_foo2v = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_Z8rec_foo1v = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMyF.Main() #0 !dbg !52 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z8rec_foo2v(), !dbg !53
// CHECK:STDOUT:   ret void, !dbg !54
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_Z8rec_foo2v() #1 comdat {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z8rec_foo1v()
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_Z8rec_foo1v() #2 comdat {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { inlinehint mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { inlinehint mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!51}
// CHECK:STDOUT: !llvm.module.flags = !{!25, !26, !27, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!32}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !{null}
// CHECK:STDOUT: !21 = !DISubroutineType(types: !20)
// CHECK:STDOUT: !25 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !26 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !27 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !28 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !29 = !{!"omnipotent char", !28, i64 0}
// CHECK:STDOUT: !30 = !{!"int", !29, i64 0}
// CHECK:STDOUT: !31 = !{!"__libc_errno", !30, i64 0}
// CHECK:STDOUT: !32 = !{!31, !30, i64 0}
// CHECK:STDOUT: !50 = !DIFile(filename: "import_inline_recursive_function_decl.carbon", directory: "")
// CHECK:STDOUT: !51 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !50, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !52 = distinct !DISubprogram(name: "MyF", linkageName: "_CMyF.Main", scope: null, file: !50, line: 6, type: !21, spFlags: DISPFlagDefinition, unit: !51)
// CHECK:STDOUT: !53 = !DILocation(line: 8, column: 3, scope: !52)
// CHECK:STDOUT: !54 = !DILocation(line: 6, column: 1, scope: !52)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'call_with_default_args.carbon'
// CHECK:STDOUT: source_filename = "call_with_default_args.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CMyF.Main() #0 !dbg !68 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %value.var = alloca i32, align 4, !dbg !74
// CHECK:STDOUT:   call void @_Z13NoReturnValueii.carbon_thunk.(), !dbg !71
// CHECK:STDOUT:   call void @_Z13NoReturnValueii.carbon_thunk._(i32 3), !dbg !72
// CHECK:STDOUT:   call void @_Z13NoReturnValueii(i32 3, i32 4), !dbg !73
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %value.var), !dbg !74
// CHECK:STDOUT:   %SimpleReturnValue__carbon_thunk.call.loc11 = call i32 @_Z17SimpleReturnValueii.carbon_thunk.(), !dbg !75
// CHECK:STDOUT:   store i32 %SimpleReturnValue__carbon_thunk.call.loc11, ptr %value.var, align 4, !dbg !74
// CHECK:STDOUT:   %SimpleReturnValue__carbon_thunk.call.loc12 = call i32 @_Z17SimpleReturnValueii.carbon_thunk._(i32 3), !dbg !78
// CHECK:STDOUT:   store i32 %SimpleReturnValue__carbon_thunk.call.loc12, ptr %value.var, align 4, !dbg !77
// CHECK:STDOUT:   %SimpleReturnValue.call = call i32 @_Z17SimpleReturnValueii(i32 3, i32 4), !dbg !80
// CHECK:STDOUT:   store i32 %SimpleReturnValue.call, ptr %value.var, align 4, !dbg !79
// CHECK:STDOUT:   %.loc14 = load i32, ptr %value.var, align 4, !dbg !81
// CHECK:STDOUT:   call void @"_COp.f58524831c37807e:core.Destroy.Core"(ptr %value.var), !dbg !74
// CHECK:STDOUT:   ret i32 %.loc14, !dbg !82
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z13NoReturnValueii.carbon_thunk.() #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z13NoReturnValueii(i32 noundef 1, i32 noundef 2)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z13NoReturnValueii.carbon_thunk._(i32 noundef %a) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.addr = alloca i32, align 4
// CHECK:STDOUT:   store i32 %a, ptr %a.addr, align 4, !tbaa !87
// CHECK:STDOUT:   %0 = load i32, ptr %a.addr, align 4, !tbaa !87
// CHECK:STDOUT:   call void @_Z13NoReturnValueii(i32 noundef %0, i32 noundef 2)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z13NoReturnValueii(i32 noundef, i32 noundef) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal noundef i32 @_Z17SimpleReturnValueii.carbon_thunk.() #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %call = call noundef i32 @_Z17SimpleReturnValueii(i32 noundef 1, i32 noundef 2)
// CHECK:STDOUT:   ret i32 %call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal noundef i32 @_Z17SimpleReturnValueii.carbon_thunk._(i32 noundef %a) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.addr = alloca i32, align 4
// CHECK:STDOUT:   store i32 %a, ptr %a.addr, align 4, !tbaa !87
// CHECK:STDOUT:   %0 = load i32, ptr %a.addr, align 4, !tbaa !87
// CHECK:STDOUT:   %call = call noundef i32 @_Z17SimpleReturnValueii(i32 noundef %0, i32 noundef 2)
// CHECK:STDOUT:   ret i32 %call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare noundef i32 @_Z17SimpleReturnValueii(i32 noundef, i32 noundef) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !83 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !85
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_Z13NoReturnValueii, { 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_Z17SimpleReturnValueii, { 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #3 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!56}
// CHECK:STDOUT: !llvm.module.flags = !{!25, !26, !27, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!32}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !25 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !26 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !27 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !28 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !29 = !{!"omnipotent char", !28, i64 0}
// CHECK:STDOUT: !30 = !{!"int", !29, i64 0}
// CHECK:STDOUT: !31 = !{!"__libc_errno", !30, i64 0}
// CHECK:STDOUT: !32 = !{!31, !30, i64 0}
// CHECK:STDOUT: !55 = !DIFile(filename: "call_with_default_args.carbon", directory: "")
// CHECK:STDOUT: !56 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !55, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !57 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !58 = !{!57}
// CHECK:STDOUT: !59 = !DISubroutineType(types: !58)
// CHECK:STDOUT: !60 = !{null, !57}
// CHECK:STDOUT: !61 = !DISubroutineType(types: !60)
// CHECK:STDOUT: !68 = distinct !DISubprogram(name: "MyF", linkageName: "_CMyF.Main", scope: null, file: !55, line: 6, type: !59, spFlags: DISPFlagDefinition, unit: !56)
// CHECK:STDOUT: !71 = !DILocation(line: 7, column: 3, scope: !68)
// CHECK:STDOUT: !72 = !DILocation(line: 8, column: 3, scope: !68)
// CHECK:STDOUT: !73 = !DILocation(line: 9, column: 3, scope: !68)
// CHECK:STDOUT: !74 = !DILocation(line: 11, column: 3, scope: !68)
// CHECK:STDOUT: !75 = !DILocation(line: 11, column: 20, scope: !68)
// CHECK:STDOUT: !77 = !DILocation(line: 12, column: 3, scope: !68)
// CHECK:STDOUT: !78 = !DILocation(line: 12, column: 11, scope: !68)
// CHECK:STDOUT: !79 = !DILocation(line: 13, column: 3, scope: !68)
// CHECK:STDOUT: !80 = !DILocation(line: 13, column: 11, scope: !68)
// CHECK:STDOUT: !81 = !DILocation(line: 14, column: 10, scope: !68)
// CHECK:STDOUT: !82 = !DILocation(line: 14, column: 3, scope: !68)
// CHECK:STDOUT: !83 = distinct !DISubprogram(name: "Op", linkageName: "_COp.f58524831c37807e:core.Destroy.Core", scope: null, file: !55, line: 11, type: !61, spFlags: DISPFlagDefinition, unit: !56, retainedNodes: !86)
// CHECK:STDOUT: !84 = !DILocalVariable(arg: 1, scope: !83, type: !57)
// CHECK:STDOUT: !85 = !DILocation(line: 11, column: 3, scope: !83)
// CHECK:STDOUT: !86 = !{!84}
// CHECK:STDOUT: !87 = !{!30, !30, i64 0}
// CHECK:STDOUT:
