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
// CHECK:STDOUT: !llvm.module.flags = !{!25, !26, !27, !28, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!33}
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
// CHECK:STDOUT: !25 = !{i32 2, !"debugger-tuning", !"gdb"}
// CHECK:STDOUT: !26 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !27 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !28 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !29 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !30 = !{!"omnipotent char", !29, i64 0}
// CHECK:STDOUT: !31 = !{!"int", !30, i64 0}
// CHECK:STDOUT: !32 = !{!"__libc_errno", !31, i64 0}
// CHECK:STDOUT: !33 = !{!32, !31, i64 0}
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
// CHECK:STDOUT: define void @_CMyF.Main() #0 !dbg !36 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z10foo_inlinev(), !dbg !37
// CHECK:STDOUT:   ret void, !dbg !38
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!35}
// CHECK:STDOUT: !llvm.module.flags = !{!25, !26, !27, !28, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!33}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !{null}
// CHECK:STDOUT: !21 = !DISubroutineType(types: !20)
// CHECK:STDOUT: !25 = !{i32 2, !"debugger-tuning", !"gdb"}
// CHECK:STDOUT: !26 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !27 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !28 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !29 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !30 = !{!"omnipotent char", !29, i64 0}
// CHECK:STDOUT: !31 = !{!"int", !30, i64 0}
// CHECK:STDOUT: !32 = !{!"__libc_errno", !31, i64 0}
// CHECK:STDOUT: !33 = !{!32, !31, i64 0}
// CHECK:STDOUT: !34 = !DIFile(filename: "import_inline_function_decl.carbon", directory: "")
// CHECK:STDOUT: !35 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !34, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !36 = distinct !DISubprogram(name: "MyF", linkageName: "_CMyF.Main", scope: null, file: !34, line: 6, type: !21, spFlags: DISPFlagDefinition, unit: !35)
// CHECK:STDOUT: !37 = !DILocation(line: 7, column: 3, scope: !36)
// CHECK:STDOUT: !38 = !DILocation(line: 6, column: 1, scope: !36)
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
// CHECK:STDOUT: define void @_CMyF.Main() #1 !dbg !41 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z8foo_usedv(), !dbg !42
// CHECK:STDOUT:   ret void, !dbg !43
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_Z8foo_usedv, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { inlinehint mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #1 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!40}
// CHECK:STDOUT: !llvm.module.flags = !{!25, !26, !27, !28, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!33}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !{null}
// CHECK:STDOUT: !21 = !DISubroutineType(types: !20)
// CHECK:STDOUT: !25 = !{i32 2, !"debugger-tuning", !"gdb"}
// CHECK:STDOUT: !26 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !27 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !28 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !29 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !30 = !{!"omnipotent char", !29, i64 0}
// CHECK:STDOUT: !31 = !{!"int", !30, i64 0}
// CHECK:STDOUT: !32 = !{!"__libc_errno", !31, i64 0}
// CHECK:STDOUT: !33 = !{!32, !31, i64 0}
// CHECK:STDOUT: !39 = !DIFile(filename: "import_inline_used_function_decl.carbon", directory: "")
// CHECK:STDOUT: !40 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !39, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !41 = distinct !DISubprogram(name: "MyF", linkageName: "_CMyF.Main", scope: null, file: !39, line: 6, type: !21, spFlags: DISPFlagDefinition, unit: !40)
// CHECK:STDOUT: !42 = !DILocation(line: 7, column: 3, scope: !41)
// CHECK:STDOUT: !43 = !DILocation(line: 6, column: 1, scope: !41)
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
// CHECK:STDOUT: define void @_CMyF.Main() #0 !dbg !46 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z10multi_foo1v(), !dbg !47
// CHECK:STDOUT:   call void @_Z10multi_foo2v(), !dbg !48
// CHECK:STDOUT:   call void @_Z10multi_foo3v(), !dbg !49
// CHECK:STDOUT:   ret void, !dbg !50
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!45}
// CHECK:STDOUT: !llvm.module.flags = !{!25, !26, !27, !28, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!33}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !{null}
// CHECK:STDOUT: !21 = !DISubroutineType(types: !20)
// CHECK:STDOUT: !25 = !{i32 2, !"debugger-tuning", !"gdb"}
// CHECK:STDOUT: !26 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !27 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !28 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !29 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !30 = !{!"omnipotent char", !29, i64 0}
// CHECK:STDOUT: !31 = !{!"int", !30, i64 0}
// CHECK:STDOUT: !32 = !{!"__libc_errno", !31, i64 0}
// CHECK:STDOUT: !33 = !{!32, !31, i64 0}
// CHECK:STDOUT: !44 = !DIFile(filename: "import_multiple_inline_function_decls.carbon", directory: "")
// CHECK:STDOUT: !45 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !44, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !46 = distinct !DISubprogram(name: "MyF", linkageName: "_CMyF.Main", scope: null, file: !44, line: 6, type: !21, spFlags: DISPFlagDefinition, unit: !45)
// CHECK:STDOUT: !47 = !DILocation(line: 7, column: 3, scope: !46)
// CHECK:STDOUT: !48 = !DILocation(line: 8, column: 3, scope: !46)
// CHECK:STDOUT: !49 = !DILocation(line: 9, column: 3, scope: !46)
// CHECK:STDOUT: !50 = !DILocation(line: 6, column: 1, scope: !46)
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
// CHECK:STDOUT: define void @_CMyF.Main() #0 !dbg !53 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z8rec_foo2v(), !dbg !54
// CHECK:STDOUT:   ret void, !dbg !55
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!52}
// CHECK:STDOUT: !llvm.module.flags = !{!25, !26, !27, !28, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!33}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !{null}
// CHECK:STDOUT: !21 = !DISubroutineType(types: !20)
// CHECK:STDOUT: !25 = !{i32 2, !"debugger-tuning", !"gdb"}
// CHECK:STDOUT: !26 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !27 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !28 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !29 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !30 = !{!"omnipotent char", !29, i64 0}
// CHECK:STDOUT: !31 = !{!"int", !30, i64 0}
// CHECK:STDOUT: !32 = !{!"__libc_errno", !31, i64 0}
// CHECK:STDOUT: !33 = !{!32, !31, i64 0}
// CHECK:STDOUT: !51 = !DIFile(filename: "import_inline_recursive_function_decl.carbon", directory: "")
// CHECK:STDOUT: !52 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !51, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !53 = distinct !DISubprogram(name: "MyF", linkageName: "_CMyF.Main", scope: null, file: !51, line: 6, type: !21, spFlags: DISPFlagDefinition, unit: !52)
// CHECK:STDOUT: !54 = !DILocation(line: 8, column: 3, scope: !53)
// CHECK:STDOUT: !55 = !DILocation(line: 6, column: 1, scope: !53)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'call_with_default_args.carbon'
// CHECK:STDOUT: source_filename = "call_with_default_args.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CMyF.Main() #0 !dbg !69 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %value.var = alloca i32, align 4, !dbg !75
// CHECK:STDOUT:   call void @_Z13NoReturnValueii.carbon_thunk.(), !dbg !72
// CHECK:STDOUT:   call void @_Z13NoReturnValueii.carbon_thunk._(i32 3), !dbg !73
// CHECK:STDOUT:   call void @_Z13NoReturnValueii(i32 3, i32 4), !dbg !74
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %value.var), !dbg !75
// CHECK:STDOUT:   %SimpleReturnValue__carbon_thunk.call.loc11 = call i32 @_Z17SimpleReturnValueii.carbon_thunk.(), !dbg !76
// CHECK:STDOUT:   store i32 %SimpleReturnValue__carbon_thunk.call.loc11, ptr %value.var, align 4, !dbg !75
// CHECK:STDOUT:   %SimpleReturnValue__carbon_thunk.call.loc12 = call i32 @_Z17SimpleReturnValueii.carbon_thunk._(i32 3), !dbg !79
// CHECK:STDOUT:   store i32 %SimpleReturnValue__carbon_thunk.call.loc12, ptr %value.var, align 4, !dbg !78
// CHECK:STDOUT:   %SimpleReturnValue.call = call i32 @_Z17SimpleReturnValueii(i32 3, i32 4), !dbg !81
// CHECK:STDOUT:   store i32 %SimpleReturnValue.call, ptr %value.var, align 4, !dbg !80
// CHECK:STDOUT:   %.loc14 = load i32, ptr %value.var, align 4, !dbg !82
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f127f30143696d51(ptr %value.var), !dbg !75
// CHECK:STDOUT:   ret i32 %.loc14, !dbg !83
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
// CHECK:STDOUT:   store i32 %a, ptr %a.addr, align 4, !tbaa !91
// CHECK:STDOUT:   %0 = load i32, ptr %a.addr, align 4, !tbaa !91
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
// CHECK:STDOUT:   store i32 %a, ptr %a.addr, align 4, !tbaa !91
// CHECK:STDOUT:   %0 = load i32, ptr %a.addr, align 4, !tbaa !91
// CHECK:STDOUT:   %call = call noundef i32 @_Z17SimpleReturnValueii(i32 noundef %0, i32 noundef 2)
// CHECK:STDOUT:   ret i32 %call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare noundef i32 @_Z17SimpleReturnValueii(i32 noundef, i32 noundef) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.f127f30143696d51(ptr %t) #0 !dbg !84 {
// CHECK:STDOUT:   ret void, !dbg !89
// CHECK:STDOUT: }
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!57}
// CHECK:STDOUT: !llvm.module.flags = !{!25, !26, !27, !28, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!33}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !10 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !25 = !{i32 2, !"debugger-tuning", !"gdb"}
// CHECK:STDOUT: !26 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !27 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !28 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !29 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !30 = !{!"omnipotent char", !29, i64 0}
// CHECK:STDOUT: !31 = !{!"int", !30, i64 0}
// CHECK:STDOUT: !32 = !{!"__libc_errno", !31, i64 0}
// CHECK:STDOUT: !33 = !{!32, !31, i64 0}
// CHECK:STDOUT: !56 = !DIFile(filename: "call_with_default_args.carbon", directory: "")
// CHECK:STDOUT: !57 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !56, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !58 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !59 = !{!58}
// CHECK:STDOUT: !60 = !DISubroutineType(types: !59)
// CHECK:STDOUT: !61 = !{null, !58}
// CHECK:STDOUT: !62 = !DISubroutineType(types: !61)
// CHECK:STDOUT: !69 = distinct !DISubprogram(name: "MyF", linkageName: "_CMyF.Main", scope: null, file: !56, line: 6, type: !60, spFlags: DISPFlagDefinition, unit: !57)
// CHECK:STDOUT: !72 = !DILocation(line: 7, column: 3, scope: !69)
// CHECK:STDOUT: !73 = !DILocation(line: 8, column: 3, scope: !69)
// CHECK:STDOUT: !74 = !DILocation(line: 9, column: 3, scope: !69)
// CHECK:STDOUT: !75 = !DILocation(line: 11, column: 3, scope: !69)
// CHECK:STDOUT: !76 = !DILocation(line: 11, column: 20, scope: !69)
// CHECK:STDOUT: !78 = !DILocation(line: 12, column: 3, scope: !69)
// CHECK:STDOUT: !79 = !DILocation(line: 12, column: 11, scope: !69)
// CHECK:STDOUT: !80 = !DILocation(line: 13, column: 3, scope: !69)
// CHECK:STDOUT: !81 = !DILocation(line: 13, column: 11, scope: !69)
// CHECK:STDOUT: !82 = !DILocation(line: 14, column: 10, scope: !69)
// CHECK:STDOUT: !83 = !DILocation(line: 14, column: 3, scope: !69)
// CHECK:STDOUT: !84 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.f127f30143696d51", scope: null, file: !10, line: 32, type: !62, spFlags: DISPFlagDefinition, unit: !57, retainedNodes: !90)
// CHECK:STDOUT: !85 = !DILocalVariable(arg: 1, scope: !84, type: !58)
// CHECK:STDOUT: !89 = !DILocation(line: 32, column: 1, scope: !84)
// CHECK:STDOUT: !90 = !{!85}
// CHECK:STDOUT: !91 = !{!31, !31, i64 0}
// CHECK:STDOUT:
