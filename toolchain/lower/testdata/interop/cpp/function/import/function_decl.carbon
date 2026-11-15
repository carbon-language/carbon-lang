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
// CHECK:STDOUT: define void @_CMyF.Main() #0 !dbg !24 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z3foov(), !dbg !25
// CHECK:STDOUT:   ret void, !dbg !26
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z3foov() #1
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!21}
// CHECK:STDOUT: !llvm.module.flags = !{!27, !28, !29, !30, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!35}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIFile(filename: "import_function_decl.carbon", directory: "")
// CHECK:STDOUT: !21 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !20, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !22 = !{null}
// CHECK:STDOUT: !23 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !24 = distinct !DISubprogram(name: "MyF", linkageName: "_CMyF.Main", scope: null, file: !20, line: 6, type: !23, spFlags: DISPFlagDefinition, unit: !21)
// CHECK:STDOUT: !25 = !DILocation(line: 7, column: 3, scope: !24)
// CHECK:STDOUT: !26 = !DILocation(line: 6, column: 1, scope: !24)
// CHECK:STDOUT: !27 = !{i32 2, !"debugger-tuning", !"gdb"}
// CHECK:STDOUT: !28 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !29 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !30 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !31 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !32 = !{!"omnipotent char", !31, i64 0}
// CHECK:STDOUT: !33 = !{!"int", !32, i64 0}
// CHECK:STDOUT: !34 = !{!"__libc_errno", !33, i64 0}
// CHECK:STDOUT: !35 = !{!34, !33, i64 0}
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
// CHECK:STDOUT: define void @_CMyF.Main() #0 !dbg !38 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z10foo_inlinev(), !dbg !39
// CHECK:STDOUT:   ret void, !dbg !40
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!37}
// CHECK:STDOUT: !llvm.module.flags = !{!27, !28, !29, !30, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!35}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !22 = !{null}
// CHECK:STDOUT: !23 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !27 = !{i32 2, !"debugger-tuning", !"gdb"}
// CHECK:STDOUT: !28 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !29 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !30 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !31 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !32 = !{!"omnipotent char", !31, i64 0}
// CHECK:STDOUT: !33 = !{!"int", !32, i64 0}
// CHECK:STDOUT: !34 = !{!"__libc_errno", !33, i64 0}
// CHECK:STDOUT: !35 = !{!34, !33, i64 0}
// CHECK:STDOUT: !36 = !DIFile(filename: "import_inline_function_decl.carbon", directory: "")
// CHECK:STDOUT: !37 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !36, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !38 = distinct !DISubprogram(name: "MyF", linkageName: "_CMyF.Main", scope: null, file: !36, line: 6, type: !23, spFlags: DISPFlagDefinition, unit: !37)
// CHECK:STDOUT: !39 = !DILocation(line: 7, column: 3, scope: !38)
// CHECK:STDOUT: !40 = !DILocation(line: 6, column: 1, scope: !38)
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
// CHECK:STDOUT: define void @_CMyF.Main() #1 !dbg !43 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z8foo_usedv(), !dbg !44
// CHECK:STDOUT:   ret void, !dbg !45
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_Z8foo_usedv, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { inlinehint mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #1 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!42}
// CHECK:STDOUT: !llvm.module.flags = !{!27, !28, !29, !30, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!35}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !22 = !{null}
// CHECK:STDOUT: !23 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !27 = !{i32 2, !"debugger-tuning", !"gdb"}
// CHECK:STDOUT: !28 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !29 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !30 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !31 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !32 = !{!"omnipotent char", !31, i64 0}
// CHECK:STDOUT: !33 = !{!"int", !32, i64 0}
// CHECK:STDOUT: !34 = !{!"__libc_errno", !33, i64 0}
// CHECK:STDOUT: !35 = !{!34, !33, i64 0}
// CHECK:STDOUT: !41 = !DIFile(filename: "import_inline_used_function_decl.carbon", directory: "")
// CHECK:STDOUT: !42 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !41, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !43 = distinct !DISubprogram(name: "MyF", linkageName: "_CMyF.Main", scope: null, file: !41, line: 6, type: !23, spFlags: DISPFlagDefinition, unit: !42)
// CHECK:STDOUT: !44 = !DILocation(line: 7, column: 3, scope: !43)
// CHECK:STDOUT: !45 = !DILocation(line: 6, column: 1, scope: !43)
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
// CHECK:STDOUT: define void @_CMyF.Main() #0 !dbg !48 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z10multi_foo1v(), !dbg !49
// CHECK:STDOUT:   call void @_Z10multi_foo2v(), !dbg !50
// CHECK:STDOUT:   call void @_Z10multi_foo3v(), !dbg !51
// CHECK:STDOUT:   ret void, !dbg !52
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!47}
// CHECK:STDOUT: !llvm.module.flags = !{!27, !28, !29, !30, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!35}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !22 = !{null}
// CHECK:STDOUT: !23 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !27 = !{i32 2, !"debugger-tuning", !"gdb"}
// CHECK:STDOUT: !28 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !29 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !30 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !31 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !32 = !{!"omnipotent char", !31, i64 0}
// CHECK:STDOUT: !33 = !{!"int", !32, i64 0}
// CHECK:STDOUT: !34 = !{!"__libc_errno", !33, i64 0}
// CHECK:STDOUT: !35 = !{!34, !33, i64 0}
// CHECK:STDOUT: !46 = !DIFile(filename: "import_multiple_inline_function_decls.carbon", directory: "")
// CHECK:STDOUT: !47 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !46, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !48 = distinct !DISubprogram(name: "MyF", linkageName: "_CMyF.Main", scope: null, file: !46, line: 6, type: !23, spFlags: DISPFlagDefinition, unit: !47)
// CHECK:STDOUT: !49 = !DILocation(line: 7, column: 3, scope: !48)
// CHECK:STDOUT: !50 = !DILocation(line: 8, column: 3, scope: !48)
// CHECK:STDOUT: !51 = !DILocation(line: 9, column: 3, scope: !48)
// CHECK:STDOUT: !52 = !DILocation(line: 6, column: 1, scope: !48)
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
// CHECK:STDOUT: define void @_CMyF.Main() #0 !dbg !55 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z8rec_foo2v(), !dbg !56
// CHECK:STDOUT:   ret void, !dbg !57
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!54}
// CHECK:STDOUT: !llvm.module.flags = !{!27, !28, !29, !30, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!35}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !22 = !{null}
// CHECK:STDOUT: !23 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !27 = !{i32 2, !"debugger-tuning", !"gdb"}
// CHECK:STDOUT: !28 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !29 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !30 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !31 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !32 = !{!"omnipotent char", !31, i64 0}
// CHECK:STDOUT: !33 = !{!"int", !32, i64 0}
// CHECK:STDOUT: !34 = !{!"__libc_errno", !33, i64 0}
// CHECK:STDOUT: !35 = !{!34, !33, i64 0}
// CHECK:STDOUT: !53 = !DIFile(filename: "import_inline_recursive_function_decl.carbon", directory: "")
// CHECK:STDOUT: !54 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !53, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !55 = distinct !DISubprogram(name: "MyF", linkageName: "_CMyF.Main", scope: null, file: !53, line: 6, type: !23, spFlags: DISPFlagDefinition, unit: !54)
// CHECK:STDOUT: !56 = !DILocation(line: 8, column: 3, scope: !55)
// CHECK:STDOUT: !57 = !DILocation(line: 6, column: 1, scope: !55)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'call_with_default_args.carbon'
// CHECK:STDOUT: source_filename = "call_with_default_args.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CMyF.Main() #0 !dbg !71 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %value.var = alloca i32, align 4, !dbg !77
// CHECK:STDOUT:   call void @_Z13NoReturnValueii.carbon_thunk.(), !dbg !74
// CHECK:STDOUT:   call void @_Z13NoReturnValueii.carbon_thunk._(i32 3), !dbg !75
// CHECK:STDOUT:   call void @_Z13NoReturnValueii(i32 3, i32 4), !dbg !76
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %value.var), !dbg !77
// CHECK:STDOUT:   %SimpleReturnValue__carbon_thunk.call.loc11 = call i32 @_Z17SimpleReturnValueii.carbon_thunk.(), !dbg !78
// CHECK:STDOUT:   store i32 %SimpleReturnValue__carbon_thunk.call.loc11, ptr %value.var, align 4, !dbg !77
// CHECK:STDOUT:   %SimpleReturnValue__carbon_thunk.call.loc12 = call i32 @_Z17SimpleReturnValueii.carbon_thunk._(i32 3), !dbg !81
// CHECK:STDOUT:   store i32 %SimpleReturnValue__carbon_thunk.call.loc12, ptr %value.var, align 4, !dbg !80
// CHECK:STDOUT:   %SimpleReturnValue.call = call i32 @_Z17SimpleReturnValueii(i32 3, i32 4), !dbg !83
// CHECK:STDOUT:   store i32 %SimpleReturnValue.call, ptr %value.var, align 4, !dbg !82
// CHECK:STDOUT:   %.loc14 = load i32, ptr %value.var, align 4, !dbg !84
// CHECK:STDOUT:   ret i32 %.loc14, !dbg !85
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
// CHECK:STDOUT:   store i32 %a, ptr %a.addr, align 4, !tbaa !86
// CHECK:STDOUT:   %0 = load i32, ptr %a.addr, align 4, !tbaa !86
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
// CHECK:STDOUT:   store i32 %a, ptr %a.addr, align 4, !tbaa !86
// CHECK:STDOUT:   %0 = load i32, ptr %a.addr, align 4, !tbaa !86
// CHECK:STDOUT:   %call = call noundef i32 @_Z17SimpleReturnValueii(i32 noundef %0, i32 noundef 2)
// CHECK:STDOUT:   ret i32 %call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare noundef i32 @_Z17SimpleReturnValueii(i32 noundef, i32 noundef) #2
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!59}
// CHECK:STDOUT: !llvm.module.flags = !{!27, !28, !29, !30, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!35}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !27 = !{i32 2, !"debugger-tuning", !"gdb"}
// CHECK:STDOUT: !28 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !29 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !30 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !31 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !32 = !{!"omnipotent char", !31, i64 0}
// CHECK:STDOUT: !33 = !{!"int", !32, i64 0}
// CHECK:STDOUT: !34 = !{!"__libc_errno", !33, i64 0}
// CHECK:STDOUT: !35 = !{!34, !33, i64 0}
// CHECK:STDOUT: !58 = !DIFile(filename: "call_with_default_args.carbon", directory: "")
// CHECK:STDOUT: !59 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !58, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !60 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !61 = !{!60}
// CHECK:STDOUT: !62 = !DISubroutineType(types: !61)
// CHECK:STDOUT: !71 = distinct !DISubprogram(name: "MyF", linkageName: "_CMyF.Main", scope: null, file: !58, line: 6, type: !62, spFlags: DISPFlagDefinition, unit: !59)
// CHECK:STDOUT: !74 = !DILocation(line: 7, column: 3, scope: !71)
// CHECK:STDOUT: !75 = !DILocation(line: 8, column: 3, scope: !71)
// CHECK:STDOUT: !76 = !DILocation(line: 9, column: 3, scope: !71)
// CHECK:STDOUT: !77 = !DILocation(line: 11, column: 3, scope: !71)
// CHECK:STDOUT: !78 = !DILocation(line: 11, column: 20, scope: !71)
// CHECK:STDOUT: !80 = !DILocation(line: 12, column: 3, scope: !71)
// CHECK:STDOUT: !81 = !DILocation(line: 12, column: 11, scope: !71)
// CHECK:STDOUT: !82 = !DILocation(line: 13, column: 3, scope: !71)
// CHECK:STDOUT: !83 = !DILocation(line: 13, column: 11, scope: !71)
// CHECK:STDOUT: !84 = !DILocation(line: 14, column: 10, scope: !71)
// CHECK:STDOUT: !85 = !DILocation(line: 14, column: 3, scope: !71)
// CHECK:STDOUT: !86 = !{!33, !33, i64 0}
// CHECK:STDOUT:
