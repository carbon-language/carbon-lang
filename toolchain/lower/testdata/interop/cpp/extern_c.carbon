// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/interop/cpp/extern_c.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/interop/cpp/extern_c.carbon

// ============================================================================
// extern "C" function
// ============================================================================

// --- extern_c_function.h

extern "C" void foo();

// --- import_extern_c_function.carbon

library "[[@TEST_NAME]]";

import Cpp library "extern_c_function.h";

fn MyF() {
  Cpp.foo();
}

// ============================================================================
// Overload set of extern "C" function and a C++ function
// ============================================================================

// --- extern_c_function_with_cpp_overload_set.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
extern "C" void foo();
void foo(int x);
''';

fn MyF() {
  Cpp.foo();
  Cpp.foo(5);
}

// ============================================================================
// extern "C" variable
// ============================================================================

// --- extern_c_variable.h

extern "C" int foo_var;

// --- import_extern_c_variable.carbon

library "[[@TEST_NAME]]";

import Cpp library "extern_c_variable.h";

fn MyF() -> i32 {
  return Cpp.foo_var;
}

// ============================================================================
// extern "C" function with C++ special name
// ============================================================================

// --- extern_c_with_special_name.h

struct X {};

extern "C" X operator+(X, X);

// --- import_extern_c_with_special_name.carbon

library "[[@TEST_NAME]]";

import Cpp library "extern_c_with_special_name.h";

fn MyF(a: Cpp.X, b: Cpp.X) -> Cpp.X {
  return a + b;
}

// ============================================================================
// extern "C" function with asm label
// ============================================================================

// --- extern_c_with_asm_label.h

extern "C" void foo_with_asm_label() __asm__("bar");

// --- import_extern_c_with_asm_label.carbon

library "[[@TEST_NAME]]";

import Cpp library "extern_c_with_asm_label.h";

fn MyF() {
  Cpp.foo_with_asm_label();
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'import_extern_c_function.carbon'
// CHECK:STDOUT: source_filename = "import_extern_c_function.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMyF.Main() #0 !dbg !22 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @foo(), !dbg !23
// CHECK:STDOUT:   ret void, !dbg !24
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @foo() #1
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
// CHECK:STDOUT: !18 = !DIFile(filename: "import_extern_c_function.carbon", directory: "")
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
// CHECK:STDOUT: ; ModuleID = 'extern_c_function_with_cpp_overload_set.carbon'
// CHECK:STDOUT: source_filename = "extern_c_function_with_cpp_overload_set.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMyF.Main() #0 !dbg !38 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @foo(), !dbg !39
// CHECK:STDOUT:   call void @_Z3fooi(i32 5), !dbg !40
// CHECK:STDOUT:   ret void, !dbg !41
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @foo() #1
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z3fooi(i32 noundef) #1
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
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
// CHECK:STDOUT: !33 = !DIFile(filename: "extern_c_function_with_cpp_overload_set.carbon", directory: "")
// CHECK:STDOUT: !34 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !33, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !38 = distinct !DISubprogram(name: "MyF", linkageName: "_CMyF.Main", scope: null, file: !33, line: 9, type: !21, spFlags: DISPFlagDefinition, unit: !34)
// CHECK:STDOUT: !39 = !DILocation(line: 10, column: 3, scope: !38)
// CHECK:STDOUT: !40 = !DILocation(line: 11, column: 3, scope: !38)
// CHECK:STDOUT: !41 = !DILocation(line: 9, column: 1, scope: !38)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'import_extern_c_variable.carbon'
// CHECK:STDOUT: source_filename = "import_extern_c_variable.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: @foo_var = external global i32, align 4
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CMyF.Main() #0 !dbg !46 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7 = load i32, ptr @foo_var, align 4, !dbg !49
// CHECK:STDOUT:   ret i32 %.loc7, !dbg !50
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!43}
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
// CHECK:STDOUT: !35 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !42 = !DIFile(filename: "import_extern_c_variable.carbon", directory: "")
// CHECK:STDOUT: !43 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !42, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !44 = !{!35}
// CHECK:STDOUT: !45 = !DISubroutineType(types: !44)
// CHECK:STDOUT: !46 = distinct !DISubprogram(name: "MyF", linkageName: "_CMyF.Main", scope: null, file: !42, line: 6, type: !45, spFlags: DISPFlagDefinition, unit: !43)
// CHECK:STDOUT: !49 = !DILocation(line: 7, column: 10, scope: !46)
// CHECK:STDOUT: !50 = !DILocation(line: 7, column: 3, scope: !46)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'import_extern_c_with_special_name.carbon'
// CHECK:STDOUT: source_filename = "import_extern_c_with_special_name.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: %struct.X = type { i8 }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMyF.Main(ptr sret({}) %return, ptr %a, ptr %b) #0 !dbg !58 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Zpl1XS_.carbon_thunk.__(ptr %a, ptr %b, ptr %return), !dbg !65
// CHECK:STDOUT:   ret void, !dbg !67
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Zpl1XS_.carbon_thunk.__(ptr noundef %0, ptr noundef %1, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   %.addr1 = alloca ptr, align 8
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   %agg.tmp = alloca %struct.X, align 1
// CHECK:STDOUT:   %agg.tmp2 = alloca %struct.X, align 1
// CHECK:STDOUT:   %undef.agg.tmp = alloca %struct.X, align 1
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !71
// CHECK:STDOUT:   store ptr %1, ptr %.addr1, align 8, !tbaa !71
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !71
// CHECK:STDOUT:   %2 = load ptr, ptr %return.addr, align 8, !tbaa !71
// CHECK:STDOUT:   %3 = load ptr, ptr %.addr, align 8, !tbaa !71
// CHECK:STDOUT:   %4 = load ptr, ptr %.addr1, align 8, !tbaa !71
// CHECK:STDOUT:   call void @_Zpl1XS_()
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Zpl1XS_() #2
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!52}
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
// CHECK:STDOUT: !51 = !DIFile(filename: "import_extern_c_with_special_name.carbon", directory: "")
// CHECK:STDOUT: !52 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !51, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !53 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !54 = !{!53, !53, !53}
// CHECK:STDOUT: !55 = !DISubroutineType(types: !54)
// CHECK:STDOUT: !58 = distinct !DISubprogram(name: "MyF", linkageName: "_CMyF.Main", scope: null, file: !51, line: 6, type: !55, spFlags: DISPFlagDefinition, unit: !52, retainedNodes: !68)
// CHECK:STDOUT: !59 = !DILocalVariable(arg: 1, scope: !58, type: !53)
// CHECK:STDOUT: !60 = !DILocalVariable(arg: 2, scope: !58, type: !53)
// CHECK:STDOUT: !65 = !DILocation(line: 7, column: 10, scope: !58)
// CHECK:STDOUT: !67 = !DILocation(line: 7, column: 3, scope: !58)
// CHECK:STDOUT: !68 = !{!59, !60}
// CHECK:STDOUT: !69 = !{!"any pointer", !29, i64 0}
// CHECK:STDOUT: !70 = !{!"p1 _ZTS1X", !69, i64 0}
// CHECK:STDOUT: !71 = !{!70, !70, i64 0}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'import_extern_c_with_asm_label.carbon'
// CHECK:STDOUT: source_filename = "import_extern_c_with_asm_label.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMyF.Main() #0 !dbg !75 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @bar(), !dbg !76
// CHECK:STDOUT:   ret void, !dbg !77
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @bar() #1
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!74}
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
// CHECK:STDOUT: !73 = !DIFile(filename: "import_extern_c_with_asm_label.carbon", directory: "")
// CHECK:STDOUT: !74 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !73, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !75 = distinct !DISubprogram(name: "MyF", linkageName: "_CMyF.Main", scope: null, file: !73, line: 6, type: !21, spFlags: DISPFlagDefinition, unit: !74)
// CHECK:STDOUT: !76 = !DILocation(line: 7, column: 3, scope: !75)
// CHECK:STDOUT: !77 = !DILocation(line: 6, column: 1, scope: !75)
// CHECK:STDOUT:
