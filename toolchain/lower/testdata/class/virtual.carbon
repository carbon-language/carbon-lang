// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
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
  override fn Fn[self: Self]() { }
}

// --- create.carbon

package Create;

import Classes;

fn Create() {
  var b: Classes.Base = {};
  var i: Classes.Intermediate = {.base = {}};
  var d: Classes.Derived = {.base = {.base = {}}};
  // Implicit initialization creates an object with the unformed state, which
  // doesn't include vptr initialization.
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

// --- call.carbon

library "[[@TEST_NAME]]";

base class Base {
  virtual fn F[self: Self]();
}

fn Use(b: Base) {
  b.F();
}

// --- call_impl.carbon

library "[[@TEST_NAME]]";

base class Base {
  virtual fn F[self: Self]();
}

class Derived {
  extend base: Base;
  override fn F[self: Self]();
}

fn Use() {
  var v : Derived = {.base = {}};
  v.F();
}

// --- generic_noop.carbon

library "[[@TEST_NAME]]";

base class Base(T:! type) {
  virtual fn F[self: Self]() { }
}

// --- generic_use.carbon

library "[[@TEST_NAME]]";

base class Base(T:! type) {
  virtual fn F[self: Self]() {
    var v: T;
  }
}

class T1 { }
class T2 {
  var v: T1;
}

fn F() {
  var t1: Base(T1) = {};
  var t2: Base(T2) = {};
}

// --- generic_base.carbon

library "[[@TEST_NAME]]";

base class Base(T:! type) {
  virtual fn F[self: Self]() { }
}

class T1;

class Derived {
  extend base: Base(T1);
}

fn Make() {
  var v: Derived;
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
// CHECK:STDOUT: @Base.val.loc7_3 = internal constant {} zeroinitializer
// CHECK:STDOUT: @Intermediate.val.ec2.loc8_3 = internal constant { ptr, {} } { ptr @"_CIntermediate.Classes.$vtable", {} zeroinitializer }
// CHECK:STDOUT: @Derived.val.loc9_3 = internal constant { { ptr, {} } } { { ptr, {} } { ptr @"_CDerived.Classes.$vtable", {} zeroinitializer } }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CCreate.Create() !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %b.var = alloca {}, align 8, !dbg !7
// CHECK:STDOUT:   %i.var = alloca { ptr, {} }, align 8, !dbg !8
// CHECK:STDOUT:   %d.var = alloca { { ptr, {} } }, align 8, !dbg !9
// CHECK:STDOUT:   %d2.var = alloca { { ptr, {} } }, align 8, !dbg !10
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %b.var), !dbg !7
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %b.var, ptr align 1 @Base.val.loc7_3, i64 0, i1 false), !dbg !7
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %i.var), !dbg !8
// CHECK:STDOUT:   %.loc8_44.2.vptr = getelementptr inbounds nuw { ptr, {} }, ptr %i.var, i32 0, i32 0, !dbg !11
// CHECK:STDOUT:   %.loc8_44.4.base = getelementptr inbounds nuw { ptr, {} }, ptr %i.var, i32 0, i32 1, !dbg !11
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 %i.var, ptr align 8 @Intermediate.val.ec2.loc8_3, i64 8, i1 false), !dbg !8
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %d.var), !dbg !9
// CHECK:STDOUT:   %.loc9_49.2.base = getelementptr inbounds nuw { { ptr, {} } }, ptr %d.var, i32 0, i32 0, !dbg !12
// CHECK:STDOUT:   %.loc9_48.2.vptr = getelementptr inbounds nuw { ptr, {} }, ptr %.loc9_49.2.base, i32 0, i32 0, !dbg !13
// CHECK:STDOUT:   %.loc9_48.4.base = getelementptr inbounds nuw { ptr, {} }, ptr %.loc9_49.2.base, i32 0, i32 1, !dbg !13
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 %d.var, ptr align 8 @Derived.val.loc9_3, i64 8, i1 false), !dbg !9
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %d2.var), !dbg !10
// CHECK:STDOUT:   ret void, !dbg !14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CFn.Intermediate.Classes(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CFn.Derived.Classes(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CUse.Create(ptr %v) !dbg !15 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Intermediate.Fn.call.vtable = load ptr, ptr %v, align 8, !dbg !16
// CHECK:STDOUT:   %Intermediate.Fn.call = call ptr @llvm.load.relative.i32(ptr %Intermediate.Fn.call.vtable, i32 0), !dbg !16
// CHECK:STDOUT:   call void %Intermediate.Fn.call(ptr %v), !dbg !16
// CHECK:STDOUT:   ret void, !dbg !17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #0
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: read)
// CHECK:STDOUT: declare ptr @llvm.load.relative.i32(ptr, i32) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: read) }
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
// CHECK:STDOUT: !8 = !DILocation(line: 8, column: 3, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 9, column: 3, scope: !4)
// CHECK:STDOUT: !10 = !DILocation(line: 12, column: 3, scope: !4)
// CHECK:STDOUT: !11 = !DILocation(line: 8, column: 33, scope: !4)
// CHECK:STDOUT: !12 = !DILocation(line: 9, column: 28, scope: !4)
// CHECK:STDOUT: !13 = !DILocation(line: 9, column: 37, scope: !4)
// CHECK:STDOUT: !14 = !DILocation(line: 6, column: 1, scope: !4)
// CHECK:STDOUT: !15 = distinct !DISubprogram(name: "Use", linkageName: "_CUse.Create", scope: null, file: !3, line: 15, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !16 = !DILocation(line: 16, column: 3, scope: !15)
// CHECK:STDOUT: !17 = !DILocation(line: 15, column: 1, scope: !15)
// CHECK:STDOUT: ; ModuleID = 'member_init.carbon'
// CHECK:STDOUT: source_filename = "member_init.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @"_CBase.MemberInit.$vtable" = unnamed_addr constant [1 x i32] [i32 trunc (i64 sub (i64 ptrtoint (ptr @_CFn.Base.MemberInit to i64), i64 ptrtoint (ptr @"_CBase.MemberInit.$vtable" to i64)) to i32)]
// CHECK:STDOUT: @Base.val.loc13_3 = internal constant { ptr, i32 } { ptr @"_CBase.MemberInit.$vtable", i32 3 }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CFn.Base.MemberInit(ptr %self) !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CFn.MemberInit() !dbg !8 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %i.var = alloca i32, align 4, !dbg !9
// CHECK:STDOUT:   %v.var = alloca { ptr, i32 }, align 8, !dbg !10
// CHECK:STDOUT:   %u.var = alloca { ptr, i32 }, align 8, !dbg !11
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %i.var), !dbg !9
// CHECK:STDOUT:   store i32 3, ptr %i.var, align 4, !dbg !9
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %v.var), !dbg !10
// CHECK:STDOUT:   %.loc11_24.2.vptr = getelementptr inbounds nuw { ptr, i32 }, ptr %v.var, i32 0, i32 0, !dbg !12
// CHECK:STDOUT:   %.loc11_23 = load i32, ptr %i.var, align 4, !dbg !13
// CHECK:STDOUT:   %.loc11_24.4.m = getelementptr inbounds nuw { ptr, i32 }, ptr %v.var, i32 0, i32 1, !dbg !12
// CHECK:STDOUT:   store i32 %.loc11_23, ptr %.loc11_24.4.m, align 4, !dbg !12
// CHECK:STDOUT:   store ptr @"_CBase.MemberInit.$vtable", ptr %.loc11_24.2.vptr, align 8, !dbg !12
// CHECK:STDOUT:   %.loc12_4.m = getelementptr inbounds nuw { ptr, i32 }, ptr %v.var, i32 0, i32 1, !dbg !14
// CHECK:STDOUT:   store i32 5, ptr %.loc12_4.m, align 4, !dbg !14
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %u.var), !dbg !11
// CHECK:STDOUT:   %.loc13_24.2.vptr = getelementptr inbounds nuw { ptr, i32 }, ptr %u.var, i32 0, i32 0, !dbg !15
// CHECK:STDOUT:   %.loc13_24.5.m = getelementptr inbounds nuw { ptr, i32 }, ptr %u.var, i32 0, i32 1, !dbg !15
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 %u.var, ptr align 8 @Base.val.loc13_3, i64 16, i1 false), !dbg !11
// CHECK:STDOUT:   ret void, !dbg !16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #0
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 2, 1, 0 }
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
// CHECK:STDOUT: !3 = !DIFile(filename: "member_init.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Fn", linkageName: "_CFn.Base.MemberInit", scope: null, file: !3, line: 6, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 6, column: 3, scope: !4)
// CHECK:STDOUT: !8 = distinct !DISubprogram(name: "Fn", linkageName: "_CFn.MemberInit", scope: null, file: !3, line: 9, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !9 = !DILocation(line: 10, column: 3, scope: !8)
// CHECK:STDOUT: !10 = !DILocation(line: 11, column: 3, scope: !8)
// CHECK:STDOUT: !11 = !DILocation(line: 13, column: 3, scope: !8)
// CHECK:STDOUT: !12 = !DILocation(line: 11, column: 17, scope: !8)
// CHECK:STDOUT: !13 = !DILocation(line: 11, column: 23, scope: !8)
// CHECK:STDOUT: !14 = !DILocation(line: 12, column: 3, scope: !8)
// CHECK:STDOUT: !15 = !DILocation(line: 13, column: 17, scope: !8)
// CHECK:STDOUT: !16 = !DILocation(line: 9, column: 1, scope: !8)
// CHECK:STDOUT: ; ModuleID = 'member_brace_init.carbon'
// CHECK:STDOUT: source_filename = "member_brace_init.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @"_CBase.Main.$vtable" = unnamed_addr constant [1 x i32] [i32 trunc (i64 sub (i64 ptrtoint (ptr @_CF.Base.Main to i64), i64 ptrtoint (ptr @"_CBase.Main.$vtable" to i64)) to i32)]
// CHECK:STDOUT: @"_CDerived.Main.$vtable" = unnamed_addr constant [1 x i32] [i32 trunc (i64 sub (i64 ptrtoint (ptr @_CF.Base.Main to i64), i64 ptrtoint (ptr @"_CDerived.Main.$vtable" to i64)) to i32)]
// CHECK:STDOUT: @Derived.val.loc13_3 = internal constant { { ptr } } { { ptr } { ptr @"_CDerived.Main.$vtable" } }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CF.Base.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CUse.Main() !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %v.var = alloca { { ptr } }, align 8, !dbg !7
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %v.var), !dbg !7
// CHECK:STDOUT:   %.loc13_32.2.base = getelementptr inbounds nuw { { ptr } }, ptr %v.var, i32 0, i32 0, !dbg !8
// CHECK:STDOUT:   %.loc13_31.2.vptr = getelementptr inbounds nuw { ptr }, ptr %.loc13_32.2.base, i32 0, i32 0, !dbg !9
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 %v.var, ptr align 8 @Derived.val.loc13_3, i64 8, i1 false), !dbg !7
// CHECK:STDOUT:   ret void, !dbg !10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #0
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
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
// CHECK:STDOUT: !3 = !DIFile(filename: "member_brace_init.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Use", linkageName: "_CUse.Main", scope: null, file: !3, line: 12, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 13, column: 3, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 13, column: 21, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 13, column: 30, scope: !4)
// CHECK:STDOUT: !10 = !DILocation(line: 12, column: 1, scope: !4)
// CHECK:STDOUT: ; ModuleID = 'call.carbon'
// CHECK:STDOUT: source_filename = "call.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @"_CBase.Main.$vtable" = unnamed_addr constant [1 x i32] [i32 trunc (i64 sub (i64 ptrtoint (ptr @_CF.Base.Main to i64), i64 ptrtoint (ptr @"_CBase.Main.$vtable" to i64)) to i32)]
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CF.Base.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CUse.Main(ptr %b) !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Base.F.call.vtable = load ptr, ptr %b, align 8, !dbg !7
// CHECK:STDOUT:   %Base.F.call = call ptr @llvm.load.relative.i32(ptr %Base.F.call.vtable, i32 0), !dbg !7
// CHECK:STDOUT:   call void %Base.F.call(ptr %b), !dbg !7
// CHECK:STDOUT:   ret void, !dbg !8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: read)
// CHECK:STDOUT: declare ptr @llvm.load.relative.i32(ptr, i32) #0
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nocallback nofree nosync nounwind willreturn memory(argmem: read) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "call.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Use", linkageName: "_CUse.Main", scope: null, file: !3, line: 8, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 9, column: 3, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 8, column: 1, scope: !4)
// CHECK:STDOUT: ; ModuleID = 'call_impl.carbon'
// CHECK:STDOUT: source_filename = "call_impl.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @"_CBase.Main.$vtable" = unnamed_addr constant [1 x i32] [i32 trunc (i64 sub (i64 ptrtoint (ptr @_CF.Base.Main to i64), i64 ptrtoint (ptr @"_CBase.Main.$vtable" to i64)) to i32)]
// CHECK:STDOUT: @"_CDerived.Main.$vtable" = unnamed_addr constant [1 x i32] [i32 trunc (i64 sub (i64 ptrtoint (ptr @_CF.Derived.Main to i64), i64 ptrtoint (ptr @"_CDerived.Main.$vtable" to i64)) to i32)]
// CHECK:STDOUT: @Derived.val.loc14_3 = internal constant { { ptr } } { { ptr } { ptr @"_CDerived.Main.$vtable" } }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CF.Base.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CF.Derived.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CUse.Main() !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %v.var = alloca { { ptr } }, align 8, !dbg !7
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %v.var), !dbg !7
// CHECK:STDOUT:   %.loc14_32.2.base = getelementptr inbounds nuw { { ptr } }, ptr %v.var, i32 0, i32 0, !dbg !8
// CHECK:STDOUT:   %.loc14_31.2.vptr = getelementptr inbounds nuw { ptr }, ptr %.loc14_32.2.base, i32 0, i32 0, !dbg !9
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 %v.var, ptr align 8 @Derived.val.loc14_3, i64 8, i1 false), !dbg !7
// CHECK:STDOUT:   %Derived.F.call.vtable = load ptr, ptr %v.var, align 8, !dbg !10
// CHECK:STDOUT:   %Derived.F.call = call ptr @llvm.load.relative.i32(ptr %Derived.F.call.vtable, i32 0), !dbg !10
// CHECK:STDOUT:   call void %Derived.F.call(ptr %v.var), !dbg !10
// CHECK:STDOUT:   ret void, !dbg !11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #0
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: read)
// CHECK:STDOUT: declare ptr @llvm.load.relative.i32(ptr, i32) #2
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: read) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "call_impl.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Use", linkageName: "_CUse.Main", scope: null, file: !3, line: 13, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 14, column: 3, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 14, column: 21, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 14, column: 30, scope: !4)
// CHECK:STDOUT: !10 = !DILocation(line: 15, column: 3, scope: !4)
// CHECK:STDOUT: !11 = !DILocation(line: 13, column: 1, scope: !4)
// CHECK:STDOUT: ; ModuleID = 'generic_noop.carbon'
// CHECK:STDOUT: source_filename = "generic_noop.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "generic_noop.carbon", directory: "")
// CHECK:STDOUT: ; ModuleID = 'generic_use.carbon'
// CHECK:STDOUT: source_filename = "generic_use.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @"_CBase.Main.$vtable.4d2ffa01ebfb7a1d" = unnamed_addr constant [1 x i32] [i32 trunc (i64 sub (i64 ptrtoint (ptr @_CF.Base.Main.4d2ffa01ebfb7a1d to i64), i64 ptrtoint (ptr @"_CBase.Main.$vtable.4d2ffa01ebfb7a1d" to i64)) to i32)]
// CHECK:STDOUT: @"_CBase.Main.$vtable.4f4310253a4de9ab" = unnamed_addr constant [1 x i32] [i32 trunc (i64 sub (i64 ptrtoint (ptr @_CF.Base.Main.4f4310253a4de9ab to i64), i64 ptrtoint (ptr @"_CBase.Main.$vtable.4f4310253a4de9ab" to i64)) to i32)]
// CHECK:STDOUT: @Base.val.20a.loc16_3 = internal constant { ptr } { ptr @"_CBase.Main.$vtable.4d2ffa01ebfb7a1d" }
// CHECK:STDOUT: @Base.val.f56.loc17_3 = internal constant { ptr } { ptr @"_CBase.Main.$vtable.4f4310253a4de9ab" }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CF.Main() !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %t1.var = alloca { ptr }, align 8, !dbg !7
// CHECK:STDOUT:   %t2.var = alloca { ptr }, align 8, !dbg !8
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %t1.var), !dbg !7
// CHECK:STDOUT:   %.loc16_23.2.vptr = getelementptr inbounds nuw { ptr }, ptr %t1.var, i32 0, i32 0, !dbg !9
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 %t1.var, ptr align 8 @Base.val.20a.loc16_3, i64 8, i1 false), !dbg !7
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %t2.var), !dbg !8
// CHECK:STDOUT:   %.loc17_23.2.vptr = getelementptr inbounds nuw { ptr }, ptr %t2.var, i32 0, i32 0, !dbg !10
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 %t2.var, ptr align 8 @Base.val.f56.loc17_3, i64 8, i1 false), !dbg !8
// CHECK:STDOUT:   ret void, !dbg !11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr void @_CF.Base.Main.4d2ffa01ebfb7a1d(ptr %self) !dbg !12 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %v.var = alloca {}, align 8, !dbg !13
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %v.var), !dbg !13
// CHECK:STDOUT:   ret void, !dbg !14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr void @_CF.Base.Main.4f4310253a4de9ab(ptr %self) !dbg !15 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %v.var = alloca { {} }, align 8, !dbg !16
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %v.var), !dbg !16
// CHECK:STDOUT:   ret void, !dbg !17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #0
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 1, 0 }
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
// CHECK:STDOUT: !3 = !DIFile(filename: "generic_use.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !3, line: 15, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 16, column: 3, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 17, column: 3, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 16, column: 22, scope: !4)
// CHECK:STDOUT: !10 = !DILocation(line: 17, column: 22, scope: !4)
// CHECK:STDOUT: !11 = !DILocation(line: 15, column: 1, scope: !4)
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "F", linkageName: "_CF.Base.Main.4d2ffa01ebfb7a1d", scope: null, file: !3, line: 5, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !13 = !DILocation(line: 6, column: 5, scope: !12)
// CHECK:STDOUT: !14 = !DILocation(line: 5, column: 3, scope: !12)
// CHECK:STDOUT: !15 = distinct !DISubprogram(name: "F", linkageName: "_CF.Base.Main.4f4310253a4de9ab", scope: null, file: !3, line: 5, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !16 = !DILocation(line: 6, column: 5, scope: !15)
// CHECK:STDOUT: !17 = !DILocation(line: 5, column: 3, scope: !15)
// CHECK:STDOUT: ; ModuleID = 'generic_base.carbon'
// CHECK:STDOUT: source_filename = "generic_base.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @"_CDerived.Main.$vtable" = unnamed_addr constant [1 x i32] [i32 trunc (i64 sub (i64 ptrtoint (ptr @_CF.Base.Main.4d2ffa01ebfb7a1d to i64), i64 ptrtoint (ptr @"_CDerived.Main.$vtable" to i64)) to i32)]
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CMake.Main() !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %v.var = alloca { { ptr } }, align 8, !dbg !7
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %v.var), !dbg !7
// CHECK:STDOUT:   ret void, !dbg !8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr void @_CF.Base.Main.4d2ffa01ebfb7a1d(ptr %self) !dbg !9 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #0
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "generic_base.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Make", linkageName: "_CMake.Main", scope: null, file: !3, line: 14, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 15, column: 3, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 14, column: 1, scope: !4)
// CHECK:STDOUT: !9 = distinct !DISubprogram(name: "F", linkageName: "_CF.Base.Main.4d2ffa01ebfb7a1d", scope: null, file: !3, line: 5, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !10 = !DILocation(line: 5, column: 3, scope: !9)
