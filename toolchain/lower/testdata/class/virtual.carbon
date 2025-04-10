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
  virtual fn Fn[self: Self]() { }
}

class Derived {
  extend base: Intermediate;
  impl fn Fn[self: Self]() { }
}

// --- create.carbon

package Create;

import Classes;

fn Create() {
  var b: Classes.Base = {};
  var i: Classes.Intermediate = {.base = {}};
  var d: Classes.Derived = {.base = {.base = {}}};
  // TODO: Support vptr initialization without explicit source initializers.
  var d2: Classes.Derived;
}

fn Use(v: Classes.Intermediate*) {
  v->Fn();
}

// --- member_init.carbon

package MemberInit;

base class Base {
  var m: i32;
  virtual fn Fn[self: Self]() { }
}

fn Fn() {
  var i: i32 = 3;
  var v: Base = {.m = i};
  v.m = 5;
  var u: Base = {.m = 3};
}

// --- member_brace_init.carbon

library "[[@TEST_NAME]]";

base class Base {
  virtual fn F[self: Self]();
}

class Derived {
  extend base: Base;
}

fn Use() {
  var v : Derived = {.base = {}};
}

// CHECK:STDOUT: ; ModuleID = 'classes.carbon'
// CHECK:STDOUT: source_filename = "classes.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @"_CIntermediate.Classes.$vtable" = unnamed_addr constant [1 x i32] [i32 trunc (i64 sub (i64 ptrtoint (ptr @_CFn.Intermediate.Classes to i64), i64 ptrtoint (ptr @"_CIntermediate.Classes.$vtable" to i64)) to i32)]
// CHECK:STDOUT: @"_CDerived.Classes.$vtable" = unnamed_addr constant [1 x i32] [i32 trunc (i64 sub (i64 ptrtoint (ptr @_CFn.Derived.Classes to i64), i64 ptrtoint (ptr @"_CDerived.Classes.$vtable" to i64)) to i32)]
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CFn.Intermediate.Classes(ptr %self) !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CFn.Derived.Classes(ptr %self) !dbg !8 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "classes.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Fn", linkageName: "_CFn.Intermediate.Classes", scope: null, file: !3, line: 9, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 9, column: 3, scope: !4)
// CHECK:STDOUT: !8 = distinct !DISubprogram(name: "Fn", linkageName: "_CFn.Derived.Classes", scope: null, file: !3, line: 14, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !9 = !DILocation(line: 14, column: 3, scope: !8)
// CHECK:STDOUT: ; ModuleID = 'create.carbon'
// CHECK:STDOUT: source_filename = "create.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @"_CIntermediate.Classes.$vtable" = external unnamed_addr constant ptr
// CHECK:STDOUT: @"_CDerived.Classes.$vtable" = external unnamed_addr constant ptr
// CHECK:STDOUT: @Base.val.loc7_3.2 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CCreate.Create() !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %b.var = alloca {}, align 8, !dbg !7
// CHECK:STDOUT:   %i.var = alloca { ptr, {} }, align 8, !dbg !7
// CHECK:STDOUT:   %d.var = alloca { { ptr, {} } }, align 8, !dbg !7
// CHECK:STDOUT:   %d2.var = alloca { { ptr, {} } }, align 8, !dbg !7
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(i64 0, ptr %b.var), !dbg !7
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %b.var, ptr align 1 @Base.val.loc7_3.2, i64 0, i1 false), !dbg !7
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(i64 8, ptr %i.var), !dbg !7
// CHECK:STDOUT:   %.loc8_44.2.vptr = getelementptr inbounds nuw { ptr, {} }, ptr %i.var, i32 0, i32 0, !dbg !8
// CHECK:STDOUT:   store ptr @"_CIntermediate.Classes.$vtable", ptr %.loc8_44.2.vptr, align 8, !dbg !8
// CHECK:STDOUT:   %.loc8_44.5.base = getelementptr inbounds nuw { ptr, {} }, ptr %i.var, i32 0, i32 1, !dbg !8
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %.loc8_44.5.base, ptr align 1 @Base.val.loc7_3.2, i64 0, i1 false), !dbg !8
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(i64 8, ptr %d.var), !dbg !7
// CHECK:STDOUT:   %.loc9_49.2.base = getelementptr inbounds nuw { { ptr, {} } }, ptr %d.var, i32 0, i32 0, !dbg !9
// CHECK:STDOUT:   %.loc9_48.2.vptr = getelementptr inbounds nuw { ptr, {} }, ptr %.loc9_49.2.base, i32 0, i32 0, !dbg !10
// CHECK:STDOUT:   store ptr @"_CDerived.Classes.$vtable", ptr %.loc9_48.2.vptr, align 8, !dbg !10
// CHECK:STDOUT:   %.loc9_48.5.base = getelementptr inbounds nuw { ptr, {} }, ptr %.loc9_49.2.base, i32 0, i32 1, !dbg !10
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %.loc9_48.5.base, ptr align 1 @Base.val.loc7_3.2, i64 0, i1 false), !dbg !10
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(i64 8, ptr %d2.var), !dbg !7
// CHECK:STDOUT:   ret void, !dbg !11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CUse.Create(ptr %v) !dbg !12 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CFn.Intermediate.Classes(ptr %v), !dbg !13
// CHECK:STDOUT:   ret void, !dbg !14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CFn.Intermediate.Classes(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(i64 immarg, ptr captures(none)) #0
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder i32 1, { 1, 3, 0, 2, 4, 5 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
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
// CHECK:STDOUT: !7 = !DILocation(line: 7, column: 3, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 8, column: 33, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 9, column: 28, scope: !4)
// CHECK:STDOUT: !10 = !DILocation(line: 9, column: 37, scope: !4)
// CHECK:STDOUT: !11 = !DILocation(line: 6, column: 1, scope: !4)
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "Use", linkageName: "_CUse.Create", scope: null, file: !3, line: 14, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !13 = !DILocation(line: 15, column: 3, scope: !12)
// CHECK:STDOUT: !14 = !DILocation(line: 14, column: 1, scope: !12)
// CHECK:STDOUT: ; ModuleID = 'member_init.carbon'
// CHECK:STDOUT: source_filename = "member_init.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @"_CBase.MemberInit.$vtable" = unnamed_addr constant [1 x i32] [i32 trunc (i64 sub (i64 ptrtoint (ptr @_CFn.Base.MemberInit to i64), i64 ptrtoint (ptr @"_CBase.MemberInit.$vtable" to i64)) to i32)]
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CFn.Base.MemberInit(ptr %self) !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CFn.MemberInit() !dbg !8 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %i.var = alloca i32, align 4, !dbg !9
// CHECK:STDOUT:   %v.var = alloca { ptr, i32 }, align 8, !dbg !9
// CHECK:STDOUT:   %u.var = alloca { ptr, i32 }, align 8, !dbg !9
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(i64 4, ptr %i.var), !dbg !9
// CHECK:STDOUT:   store i32 3, ptr %i.var, align 4, !dbg !9
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(i64 16, ptr %v.var), !dbg !9
// CHECK:STDOUT:   %.loc11_24.2.vptr = getelementptr inbounds nuw { ptr, i32 }, ptr %v.var, i32 0, i32 0, !dbg !10
// CHECK:STDOUT:   store ptr @"_CBase.MemberInit.$vtable", ptr %.loc11_24.2.vptr, align 8, !dbg !10
// CHECK:STDOUT:   %.loc11_23 = load i32, ptr %i.var, align 4, !dbg !11
// CHECK:STDOUT:   %.loc11_24.5.m = getelementptr inbounds nuw { ptr, i32 }, ptr %v.var, i32 0, i32 1, !dbg !10
// CHECK:STDOUT:   store i32 %.loc11_23, ptr %.loc11_24.5.m, align 4, !dbg !10
// CHECK:STDOUT:   %.loc12_4.m = getelementptr inbounds nuw { ptr, i32 }, ptr %v.var, i32 0, i32 1, !dbg !12
// CHECK:STDOUT:   store i32 5, ptr %.loc12_4.m, align 4, !dbg !12
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(i64 16, ptr %u.var), !dbg !9
// CHECK:STDOUT:   %.loc13_24.2.vptr = getelementptr inbounds nuw { ptr, i32 }, ptr %u.var, i32 0, i32 0, !dbg !13
// CHECK:STDOUT:   store ptr @"_CBase.MemberInit.$vtable", ptr %.loc13_24.2.vptr, align 8, !dbg !13
// CHECK:STDOUT:   %.loc13_24.6.m = getelementptr inbounds nuw { ptr, i32 }, ptr %u.var, i32 0, i32 1, !dbg !13
// CHECK:STDOUT:   store i32 3, ptr %.loc13_24.6.m, align 4, !dbg !13
// CHECK:STDOUT:   ret void, !dbg !14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(i64 immarg, ptr captures(none)) #0
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder i32 1, { 0, 2, 3, 1, 4, 5 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "member_init.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Fn", linkageName: "_CFn.Base.MemberInit", scope: null, file: !3, line: 6, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 6, column: 3, scope: !4)
// CHECK:STDOUT: !8 = distinct !DISubprogram(name: "Fn", linkageName: "_CFn.MemberInit", scope: null, file: !3, line: 9, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !9 = !DILocation(line: 10, column: 3, scope: !8)
// CHECK:STDOUT: !10 = !DILocation(line: 11, column: 17, scope: !8)
// CHECK:STDOUT: !11 = !DILocation(line: 11, column: 23, scope: !8)
// CHECK:STDOUT: !12 = !DILocation(line: 12, column: 3, scope: !8)
// CHECK:STDOUT: !13 = !DILocation(line: 13, column: 17, scope: !8)
// CHECK:STDOUT: !14 = !DILocation(line: 9, column: 1, scope: !8)
// CHECK:STDOUT: ; ModuleID = 'member_brace_init.carbon'
// CHECK:STDOUT: source_filename = "member_brace_init.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @"_CBase.Main.$vtable" = unnamed_addr constant [1 x i32] [i32 trunc (i64 sub (i64 ptrtoint (ptr @_CF.Base.Main to i64), i64 ptrtoint (ptr @"_CBase.Main.$vtable" to i64)) to i32)]
// CHECK:STDOUT: @"_CDerived.Main.$vtable" = unnamed_addr constant [1 x i32] [i32 trunc (i64 sub (i64 ptrtoint (ptr @_CF.Base.Main to i64), i64 ptrtoint (ptr @"_CDerived.Main.$vtable" to i64)) to i32)]
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CF.Base.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CUse.Main() !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %v.var = alloca { { ptr } }, align 8, !dbg !7
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(i64 8, ptr %v.var), !dbg !7
// CHECK:STDOUT:   %.loc13_32.2.base = getelementptr inbounds nuw { { ptr } }, ptr %v.var, i32 0, i32 0, !dbg !8
// CHECK:STDOUT:   %.loc13_31.2.vptr = getelementptr inbounds nuw { ptr }, ptr %.loc13_32.2.base, i32 0, i32 0, !dbg !9
// CHECK:STDOUT:   store ptr @"_CDerived.Main.$vtable", ptr %.loc13_31.2.vptr, align 8, !dbg !9
// CHECK:STDOUT:   ret void, !dbg !10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(i64 immarg, ptr captures(none)) #0
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "member_brace_init.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Use", linkageName: "_CUse.Main", scope: null, file: !3, line: 12, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 13, column: 3, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 13, column: 21, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 13, column: 30, scope: !4)
// CHECK:STDOUT: !10 = !DILocation(line: 12, column: 1, scope: !4)
