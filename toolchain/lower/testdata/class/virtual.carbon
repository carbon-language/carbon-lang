// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/class/virtual.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/class/virtual.carbon

// --- classes.carbon

package Classes;

base class Base {
}

base class Intermediate {
  extend base: Base;
  virtual fn Fn();
}

class Derived {
  extend base: Intermediate;
  impl fn Fn();
}

// --- create.carbon

package Create;

import Classes;

fn Create() {
  var b: Classes.Base;
  var i: Classes.Intermediate;
  var d: Classes.Derived;
}

fn Use(v: Classes.Intermediate*) {
  v->Fn();
}

// --- member_init.carbon

package MemberInit;

base class Base {
  var m: i32;
  virtual fn Fn();
}

fn Fn() {
  var i: i32 = 3;
  // TODO: make `i` initialize `m` here (it currently initializes the vptr instead)
  var v: Base = {.m = i};
  // TODO: fix crash here
  // var v: Base = {.m = 3};
  v.m = 5;
}

// CHECK:STDOUT: ; ModuleID = 'classes.carbon'
// CHECK:STDOUT: source_filename = "classes.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CFn.Intermediate.Classes()
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CFn.Derived.Classes()
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "classes.carbon", directory: "")
// CHECK:STDOUT: ; ModuleID = 'create.carbon'
// CHECK:STDOUT: source_filename = "create.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CCreate.Create() !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %b.var = alloca {}, align 8, !dbg !7
// CHECK:STDOUT:   %i.var = alloca { ptr, {} }, align 8, !dbg !8
// CHECK:STDOUT:   %d.var = alloca { { ptr, {} } }, align 8, !dbg !9
// CHECK:STDOUT:   ret void, !dbg !10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CUse.Create(ptr %v) !dbg !11 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CFn.Intermediate.Classes(), !dbg !12
// CHECK:STDOUT:   ret void, !dbg !13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CFn.Intermediate.Classes()
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "create.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Create", linkageName: "_CCreate.Create", scope: null, file: !3, line: 6, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 7, column: 7, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 8, column: 7, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 9, column: 7, scope: !4)
// CHECK:STDOUT: !10 = !DILocation(line: 6, column: 1, scope: !4)
// CHECK:STDOUT: !11 = distinct !DISubprogram(name: "Use", linkageName: "_CUse.Create", scope: null, file: !3, line: 12, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !12 = !DILocation(line: 13, column: 3, scope: !11)
// CHECK:STDOUT: !13 = !DILocation(line: 12, column: 1, scope: !11)
// CHECK:STDOUT: ; ModuleID = 'member_init.carbon'
// CHECK:STDOUT: source_filename = "member_init.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CFn.Base.MemberInit()
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CFn.MemberInit() !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %i.var = alloca i32, align 4, !dbg !7
// CHECK:STDOUT:   store i32 3, ptr %i.var, align 4, !dbg !8
// CHECK:STDOUT:   %v.var = alloca { ptr, i32 }, align 8, !dbg !9
// CHECK:STDOUT:   %.loc12_23 = load i32, ptr %i.var, align 4, !dbg !10
// CHECK:STDOUT:   %.loc12_24.2.m = getelementptr inbounds nuw { ptr, i32 }, ptr %v.var, i32 0, i32 1, !dbg !11
// CHECK:STDOUT:   store i32 %.loc12_23, ptr %.loc12_24.2.m, align 4, !dbg !11
// CHECK:STDOUT:   %.loc15_4.m = getelementptr inbounds nuw { ptr, i32 }, ptr %v.var, i32 0, i32 1, !dbg !12
// CHECK:STDOUT:   store i32 5, ptr %.loc15_4.m, align 4, !dbg !12
// CHECK:STDOUT:   ret void, !dbg !13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "member_init.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Fn", linkageName: "_CFn.MemberInit", scope: null, file: !3, line: 9, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 10, column: 7, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 10, column: 3, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 12, column: 7, scope: !4)
// CHECK:STDOUT: !10 = !DILocation(line: 12, column: 23, scope: !4)
// CHECK:STDOUT: !11 = !DILocation(line: 12, column: 17, scope: !4)
// CHECK:STDOUT: !12 = !DILocation(line: 15, column: 3, scope: !4)
// CHECK:STDOUT: !13 = !DILocation(line: 9, column: 1, scope: !4)
