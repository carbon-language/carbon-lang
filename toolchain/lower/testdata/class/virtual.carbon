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
  virtual fn Fn[unused self: Self]() { }
}

class Derived {
  extend base: Intermediate;
  override fn Fn[unused self: Self]() { }
}

// --- create.carbon

package Create;

import Classes;

fn Create() {
  var _: Classes.Base = {};
  var _: Classes.Intermediate = {.base = {}};
  var _: Classes.Derived = {.base = {.base = {}}};
  // Implicit initialization creates an object with the unformed state, which
  // doesn't include vptr initialization.
  var _: Classes.Derived;
}

fn Use(v: Classes.Intermediate*) {
  v->Fn();
}

// --- create_named_ctor.carbon

library "[[@TEST_NAME]]";

import Classes;

fn MakePartialBase() -> partial Classes.Base {
  return {};
}

fn MakePartialIntermediate() -> partial Classes.Intermediate {
  return {.base = MakePartialBase()};
}

fn MakeBase() -> Classes.Base {
  return MakePartialBase();
}

fn MakeIntermediate() -> Classes.Intermediate {
  return MakePartialIntermediate();
}

fn MakeDerivedFromPartial() -> Classes.Derived {
  return {.base = MakePartialIntermediate()};
}

fn MakeDerivedFromNonPartial() -> Classes.Derived {
  return {.base = MakeIntermediate()};
}

// --- member_init.carbon

package MemberInit;

base class Base {
  var m: i32;
  virtual fn Fn[unused self: Self]() { }
}

fn Fn() {
  var i: i32 = 3;
  var v: Base = {.m = i};
  v.m = 5;
  var _: Base = {.m = 3};
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
  var _ : Derived = {.base = {}};
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
  virtual fn F[unused self: Self]() { }
}

// --- generic_use.carbon

library "[[@TEST_NAME]]";

base class Base(T:! Core.Destroy) {
  virtual fn F[unused self: Self]() {
    var _: T;
  }
}

class T1 { }
class T2 {
  var v: T1;
}

fn F() {
  var _: Base(T1) = {};
  var _: Base(T2) = {};
}

// --- generic_base.carbon

library "[[@TEST_NAME]]";

base class Base(T:! type) {
  virtual fn F[unused self: Self]() { }
}

class T1;

class Derived {
  extend base: Base(T1);
}

fn Make() {
  var _: Derived;
}

// --- thunk.carbon

library "[[@TEST_NAME]]";

class From {}
class To {}

impl From as Core.ImplicitAs(To) {
  fn Convert[self: From]() -> To;
}

base class Base {
  virtual fn F[ref self: Self](n: From) -> To;
}

class Derived {
  extend base: Base;
  override fn F[unused self: Self](unused n: To) -> From {
    return {};
  }
}

fn Use() {
  var d: Derived = {.base = {}};
  let p: Base* = &d;
  p->F({});
}

// CHECK:STDOUT: ; ModuleID = 'classes.carbon'
// CHECK:STDOUT: source_filename = "classes.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @"_CIntermediate.Classes.$vtable" = unnamed_addr constant [1 x i32] [i32 trunc (i64 sub (i64 ptrtoint (ptr @_CFn.Intermediate.Classes to i64), i64 ptrtoint (ptr @"_CIntermediate.Classes.$vtable" to i64)) to i32)]
// CHECK:STDOUT: @"_CDerived.Classes.$vtable" = unnamed_addr constant [1 x i32] [i32 trunc (i64 sub (i64 ptrtoint (ptr @_CFn.Derived.Classes to i64), i64 ptrtoint (ptr @"_CDerived.Classes.$vtable" to i64)) to i32)]
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CFn.Intermediate.Classes(ptr %self) #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CFn.Derived.Classes(ptr %self) #0 !dbg !11 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "classes.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Fn", linkageName: "_CFn.Intermediate.Classes", scope: null, file: !3, line: 9, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !8)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{null, !7}
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !8 = !{!9}
// CHECK:STDOUT: !9 = !DILocalVariable(arg: 1, scope: !4, type: !7)
// CHECK:STDOUT: !10 = !DILocation(line: 9, column: 3, scope: !4)
// CHECK:STDOUT: !11 = distinct !DISubprogram(name: "Fn", linkageName: "_CFn.Derived.Classes", scope: null, file: !3, line: 14, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !12)
// CHECK:STDOUT: !12 = !{!13}
// CHECK:STDOUT: !13 = !DILocalVariable(arg: 1, scope: !11, type: !7)
// CHECK:STDOUT: !14 = !DILocation(line: 14, column: 3, scope: !11)
// CHECK:STDOUT: ; ModuleID = 'create.carbon'
// CHECK:STDOUT: source_filename = "create.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @"_CIntermediate.Classes.$vtable" = external unnamed_addr constant ptr
// CHECK:STDOUT: @"_CDerived.Classes.$vtable" = external unnamed_addr constant ptr
// CHECK:STDOUT: @Base.val.loc7_3 = internal constant {} zeroinitializer
// CHECK:STDOUT: @Intermediate.val.ec2.loc8_3 = internal constant { ptr, {} } { ptr @"_CIntermediate.Classes.$vtable", {} zeroinitializer }
// CHECK:STDOUT: @Derived.val.loc9_49.5 = internal constant { { ptr, {} } } { { ptr, {} } { ptr poison, {} zeroinitializer } }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCreate.Create() #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var.loc7 = alloca {}, align 8, !dbg !7
// CHECK:STDOUT:   %_.var.loc8 = alloca { ptr, {} }, align 8, !dbg !8
// CHECK:STDOUT:   %_.var.loc9 = alloca { { ptr, {} } }, align 8, !dbg !9
// CHECK:STDOUT:   %_.var.loc12 = alloca { { ptr, {} } }, align 8, !dbg !10
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc7), !dbg !7
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %_.var.loc7, ptr align 1 @Base.val.loc7_3, i64 0, i1 false), !dbg !7
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc8), !dbg !8
// CHECK:STDOUT:   %.loc8_44.2.vptr = getelementptr inbounds nuw { ptr, {} }, ptr %_.var.loc8, i32 0, i32 0, !dbg !11
// CHECK:STDOUT:   %.loc8_44.4.base = getelementptr inbounds nuw { ptr, {} }, ptr %_.var.loc8, i32 0, i32 1, !dbg !11
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 %_.var.loc8, ptr align 8 @Intermediate.val.ec2.loc8_3, i64 8, i1 false), !dbg !8
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc9), !dbg !9
// CHECK:STDOUT:   %.loc9_49.2.base = getelementptr inbounds nuw { { ptr, {} } }, ptr %_.var.loc9, i32 0, i32 0, !dbg !12
// CHECK:STDOUT:   %.loc9_48.2.vptr = getelementptr inbounds nuw { ptr, {} }, ptr %.loc9_49.2.base, i32 0, i32 0, !dbg !13
// CHECK:STDOUT:   %.loc9_48.4.base = getelementptr inbounds nuw { ptr, {} }, ptr %.loc9_49.2.base, i32 0, i32 1, !dbg !13
// CHECK:STDOUT:   %.loc9_49.6.base = getelementptr inbounds nuw { { ptr, {} } }, ptr %_.var.loc9, i32 0, i32 0, !dbg !12
// CHECK:STDOUT:   %.loc9_49.7.vptr = getelementptr inbounds nuw { ptr, {} }, ptr %.loc9_49.6.base, i32 0, i32 0, !dbg !12
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 %_.var.loc9, ptr align 8 @Derived.val.loc9_49.5, i64 8, i1 false), !dbg !12
// CHECK:STDOUT:   store ptr @"_CDerived.Classes.$vtable", ptr %.loc9_49.7.vptr, align 8, !dbg !12
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc12), !dbg !10
// CHECK:STDOUT:   call void @"_COp.428e66b4e6184a3f:core.Destroy.Core"(ptr %_.var.loc12), !dbg !10
// CHECK:STDOUT:   call void @"_COp.428e66b4e6184a3f:core.Destroy.Core"(ptr %_.var.loc9), !dbg !9
// CHECK:STDOUT:   call void @"_COp.74903cc702c34bd7:core.Destroy.Core"(ptr %_.var.loc8), !dbg !8
// CHECK:STDOUT:   call void @"_COp.46f79b7aa2c021ce:core.Destroy.Core"(ptr %_.var.loc7), !dbg !7
// CHECK:STDOUT:   ret void, !dbg !14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CFn.Intermediate.Classes(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CFn.Derived.Classes(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.46f79b7aa2c021ce:core.Destroy.Core"(ptr %self) #0 !dbg !15 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.27bf1b16dfaccdce:core.Destroy.Core"(ptr %self) #0 !dbg !22 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !25
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.74903cc702c34bd7:core.Destroy.Core"(ptr %self) #0 !dbg !26 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !29
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.d9cbe8847a86e0d2:core.Destroy.Core"(ptr %self) #0 !dbg !30 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !33
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.428e66b4e6184a3f:core.Destroy.Core"(ptr %self) #0 !dbg !34 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !37
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CUse.Create(ptr %v) #0 !dbg !38 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Intermediate.Fn.call.vtable = load ptr, ptr %v, align 8, !dbg !41
// CHECK:STDOUT:   %Intermediate.Fn.call = call ptr @llvm.load.relative.i32(ptr %Intermediate.Fn.call.vtable, i32 0), !dbg !41
// CHECK:STDOUT:   call void %Intermediate.Fn.call(ptr %v), !dbg !41
// CHECK:STDOUT:   ret void, !dbg !42
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: read)
// CHECK:STDOUT: declare ptr @llvm.load.relative.i32(ptr, i32) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @"_COp.428e66b4e6184a3f:core.Destroy.Core", { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #3 = { nocallback nofree nosync nounwind willreturn memory(argmem: read) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "create.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Create", linkageName: "_CCreate.Create", scope: null, file: !3, line: 6, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{null}
// CHECK:STDOUT: !7 = !DILocation(line: 7, column: 3, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 8, column: 3, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 9, column: 3, scope: !4)
// CHECK:STDOUT: !10 = !DILocation(line: 12, column: 3, scope: !4)
// CHECK:STDOUT: !11 = !DILocation(line: 8, column: 33, scope: !4)
// CHECK:STDOUT: !12 = !DILocation(line: 9, column: 28, scope: !4)
// CHECK:STDOUT: !13 = !DILocation(line: 9, column: 37, scope: !4)
// CHECK:STDOUT: !14 = !DILocation(line: 6, column: 1, scope: !4)
// CHECK:STDOUT: !15 = distinct !DISubprogram(name: "Op", linkageName: "_COp.46f79b7aa2c021ce:core.Destroy.Core", scope: null, file: !3, line: 12, type: !16, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !19)
// CHECK:STDOUT: !16 = !DISubroutineType(types: !17)
// CHECK:STDOUT: !17 = !{null, !18}
// CHECK:STDOUT: !18 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !19 = !{!20}
// CHECK:STDOUT: !20 = !DILocalVariable(arg: 1, scope: !15, type: !18)
// CHECK:STDOUT: !21 = !DILocation(line: 12, column: 3, scope: !15)
// CHECK:STDOUT: !22 = distinct !DISubprogram(name: "Op", linkageName: "_COp.27bf1b16dfaccdce:core.Destroy.Core", scope: null, file: !3, line: 12, type: !16, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !23)
// CHECK:STDOUT: !23 = !{!24}
// CHECK:STDOUT: !24 = !DILocalVariable(arg: 1, scope: !22, type: !18)
// CHECK:STDOUT: !25 = !DILocation(line: 12, column: 3, scope: !22)
// CHECK:STDOUT: !26 = distinct !DISubprogram(name: "Op", linkageName: "_COp.74903cc702c34bd7:core.Destroy.Core", scope: null, file: !3, line: 12, type: !16, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !27)
// CHECK:STDOUT: !27 = !{!28}
// CHECK:STDOUT: !28 = !DILocalVariable(arg: 1, scope: !26, type: !18)
// CHECK:STDOUT: !29 = !DILocation(line: 12, column: 3, scope: !26)
// CHECK:STDOUT: !30 = distinct !DISubprogram(name: "Op", linkageName: "_COp.d9cbe8847a86e0d2:core.Destroy.Core", scope: null, file: !3, line: 12, type: !16, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !31)
// CHECK:STDOUT: !31 = !{!32}
// CHECK:STDOUT: !32 = !DILocalVariable(arg: 1, scope: !30, type: !18)
// CHECK:STDOUT: !33 = !DILocation(line: 12, column: 3, scope: !30)
// CHECK:STDOUT: !34 = distinct !DISubprogram(name: "Op", linkageName: "_COp.428e66b4e6184a3f:core.Destroy.Core", scope: null, file: !3, line: 12, type: !16, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !35)
// CHECK:STDOUT: !35 = !{!36}
// CHECK:STDOUT: !36 = !DILocalVariable(arg: 1, scope: !34, type: !18)
// CHECK:STDOUT: !37 = !DILocation(line: 12, column: 3, scope: !34)
// CHECK:STDOUT: !38 = distinct !DISubprogram(name: "Use", linkageName: "_CUse.Create", scope: null, file: !3, line: 15, type: !16, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !39)
// CHECK:STDOUT: !39 = !{!40}
// CHECK:STDOUT: !40 = !DILocalVariable(arg: 1, scope: !38, type: !18)
// CHECK:STDOUT: !41 = !DILocation(line: 16, column: 3, scope: !38)
// CHECK:STDOUT: !42 = !DILocation(line: 15, column: 1, scope: !38)
// CHECK:STDOUT: ; ModuleID = 'create_named_ctor.carbon'
// CHECK:STDOUT: source_filename = "create_named_ctor.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @"_CIntermediate.Classes.$vtable" = external unnamed_addr constant ptr
// CHECK:STDOUT: @"_CDerived.Classes.$vtable" = external unnamed_addr constant ptr
// CHECK:STDOUT: @empty_struct.d84.loc7_12 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMakePartialBase.Main(ptr sret({}) %return) #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %return, ptr align 1 @empty_struct.d84.loc7_12, i64 0, i1 false), !dbg !8
// CHECK:STDOUT:   ret void, !dbg !8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMakePartialIntermediate.Main(ptr sret({ ptr, {} }) %return) #0 !dbg !9 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc11_36.1.base = getelementptr inbounds nuw { ptr, {} }, ptr %return, i32 0, i32 1, !dbg !10
// CHECK:STDOUT:   call void @_CMakePartialBase.Main(ptr %.loc11_36.1.base), !dbg !11
// CHECK:STDOUT:   %.loc11_36.3.vptr = getelementptr inbounds nuw { ptr, {} }, ptr %return, i32 0, i32 0, !dbg !10
// CHECK:STDOUT:   store ptr poison, ptr %.loc11_36.3.vptr, align 8, !dbg !10
// CHECK:STDOUT:   ret void, !dbg !12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMakeBase.Main(ptr sret({}) %return) #0 !dbg !13 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CMakePartialBase.Main(ptr %return), !dbg !14
// CHECK:STDOUT:   ret void, !dbg !15
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMakeIntermediate.Main(ptr sret({ ptr, {} }) %return) #0 !dbg !16 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CMakePartialIntermediate.Main(ptr %return), !dbg !17
// CHECK:STDOUT:   %.loc19_34.1.vptr = getelementptr inbounds nuw { ptr, {} }, ptr %return, i32 0, i32 0, !dbg !17
// CHECK:STDOUT:   store ptr @"_CIntermediate.Classes.$vtable", ptr %.loc19_34.1.vptr, align 8, !dbg !17
// CHECK:STDOUT:   ret void, !dbg !18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CFn.Intermediate.Classes(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMakeDerivedFromPartial.Main(ptr sret({ { ptr, {} } }) %return) #0 !dbg !19 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc23_44.1.base = getelementptr inbounds nuw { { ptr, {} } }, ptr %return, i32 0, i32 0, !dbg !20
// CHECK:STDOUT:   call void @_CMakePartialIntermediate.Main(ptr %.loc23_44.1.base), !dbg !21
// CHECK:STDOUT:   %.loc23_44.5.base = getelementptr inbounds nuw { { ptr, {} } }, ptr %return, i32 0, i32 0, !dbg !20
// CHECK:STDOUT:   %.loc23_44.6.vptr = getelementptr inbounds nuw { ptr, {} }, ptr %.loc23_44.5.base, i32 0, i32 0, !dbg !20
// CHECK:STDOUT:   store ptr @"_CDerived.Classes.$vtable", ptr %.loc23_44.6.vptr, align 8, !dbg !20
// CHECK:STDOUT:   ret void, !dbg !22
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CFn.Derived.Classes(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMakeDerivedFromNonPartial.Main(ptr sret({ { ptr, {} } }) %return) #0 !dbg !23 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc27_37.1.base = getelementptr inbounds nuw { { ptr, {} } }, ptr %return, i32 0, i32 0, !dbg !24
// CHECK:STDOUT:   call void @_CMakeIntermediate.Main(ptr %.loc27_37.1.base), !dbg !25
// CHECK:STDOUT:   %.loc27_37.7.base = getelementptr inbounds nuw { { ptr, {} } }, ptr %return, i32 0, i32 0, !dbg !24
// CHECK:STDOUT:   %.loc27_37.8.vptr = getelementptr inbounds nuw { ptr, {} }, ptr %.loc27_37.7.base, i32 0, i32 0, !dbg !24
// CHECK:STDOUT:   store ptr @"_CDerived.Classes.$vtable", ptr %.loc27_37.8.vptr, align 8, !dbg !24
// CHECK:STDOUT:   ret void, !dbg !26
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "create_named_ctor.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "MakePartialBase", linkageName: "_CMakePartialBase.Main", scope: null, file: !3, line: 6, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{!7}
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !8 = !DILocation(line: 7, column: 3, scope: !4)
// CHECK:STDOUT: !9 = distinct !DISubprogram(name: "MakePartialIntermediate", linkageName: "_CMakePartialIntermediate.Main", scope: null, file: !3, line: 10, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !10 = !DILocation(line: 11, column: 10, scope: !9)
// CHECK:STDOUT: !11 = !DILocation(line: 11, column: 19, scope: !9)
// CHECK:STDOUT: !12 = !DILocation(line: 11, column: 3, scope: !9)
// CHECK:STDOUT: !13 = distinct !DISubprogram(name: "MakeBase", linkageName: "_CMakeBase.Main", scope: null, file: !3, line: 14, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !14 = !DILocation(line: 15, column: 10, scope: !13)
// CHECK:STDOUT: !15 = !DILocation(line: 15, column: 3, scope: !13)
// CHECK:STDOUT: !16 = distinct !DISubprogram(name: "MakeIntermediate", linkageName: "_CMakeIntermediate.Main", scope: null, file: !3, line: 18, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !17 = !DILocation(line: 19, column: 10, scope: !16)
// CHECK:STDOUT: !18 = !DILocation(line: 19, column: 3, scope: !16)
// CHECK:STDOUT: !19 = distinct !DISubprogram(name: "MakeDerivedFromPartial", linkageName: "_CMakeDerivedFromPartial.Main", scope: null, file: !3, line: 22, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !20 = !DILocation(line: 23, column: 10, scope: !19)
// CHECK:STDOUT: !21 = !DILocation(line: 23, column: 19, scope: !19)
// CHECK:STDOUT: !22 = !DILocation(line: 23, column: 3, scope: !19)
// CHECK:STDOUT: !23 = distinct !DISubprogram(name: "MakeDerivedFromNonPartial", linkageName: "_CMakeDerivedFromNonPartial.Main", scope: null, file: !3, line: 26, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !24 = !DILocation(line: 27, column: 10, scope: !23)
// CHECK:STDOUT: !25 = !DILocation(line: 27, column: 19, scope: !23)
// CHECK:STDOUT: !26 = !DILocation(line: 27, column: 3, scope: !23)
// CHECK:STDOUT: ; ModuleID = 'member_init.carbon'
// CHECK:STDOUT: source_filename = "member_init.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @"_CBase.MemberInit.$vtable" = unnamed_addr constant [1 x i32] [i32 trunc (i64 sub (i64 ptrtoint (ptr @_CFn.Base.MemberInit to i64), i64 ptrtoint (ptr @"_CBase.MemberInit.$vtable" to i64)) to i32)]
// CHECK:STDOUT: @Base.val.loc13_3 = internal constant <{ ptr, i32 }> <{ ptr @"_CBase.MemberInit.$vtable", i32 3 }>
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CFn.Base.MemberInit(ptr %self) #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CFn.MemberInit() #0 !dbg !11 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %i.var = alloca i32, align 4, !dbg !14
// CHECK:STDOUT:   %v.var = alloca <{ ptr, i32 }>, align 8, !dbg !15
// CHECK:STDOUT:   %_.var = alloca <{ ptr, i32 }>, align 8, !dbg !16
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %i.var), !dbg !14
// CHECK:STDOUT:   store i32 3, ptr %i.var, align 4, !dbg !14
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %v.var), !dbg !15
// CHECK:STDOUT:   %.loc11_24.2.vptr = getelementptr inbounds nuw <{ ptr, i32 }>, ptr %v.var, i32 0, i32 0, !dbg !17
// CHECK:STDOUT:   %.loc11_23 = load i32, ptr %i.var, align 4, !dbg !18
// CHECK:STDOUT:   %.loc11_24.4.m = getelementptr inbounds nuw <{ ptr, i32 }>, ptr %v.var, i32 0, i32 1, !dbg !17
// CHECK:STDOUT:   store i32 %.loc11_23, ptr %.loc11_24.4.m, align 4, !dbg !17
// CHECK:STDOUT:   store ptr @"_CBase.MemberInit.$vtable", ptr %.loc11_24.2.vptr, align 8, !dbg !17
// CHECK:STDOUT:   %.loc12_4.m = getelementptr inbounds nuw <{ ptr, i32 }>, ptr %v.var, i32 0, i32 1, !dbg !19
// CHECK:STDOUT:   store i32 5, ptr %.loc12_4.m, align 4, !dbg !19
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !16
// CHECK:STDOUT:   %.loc13_24.2.vptr = getelementptr inbounds nuw <{ ptr, i32 }>, ptr %_.var, i32 0, i32 0, !dbg !20
// CHECK:STDOUT:   %.loc13_24.5.m = getelementptr inbounds nuw <{ ptr, i32 }>, ptr %_.var, i32 0, i32 1, !dbg !20
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %_.var, ptr align 1 @Base.val.loc13_3, i64 12, i1 false), !dbg !16
// CHECK:STDOUT:   call void @"_COp.dbf7676be30a7aa8:core.Destroy.Core"(ptr %_.var), !dbg !16
// CHECK:STDOUT:   call void @"_COp.dbf7676be30a7aa8:core.Destroy.Core"(ptr %v.var), !dbg !15
// CHECK:STDOUT:   call void @"_COp.7e389eab4a7e5487:core.Destroy.Core"(ptr %i.var), !dbg !14
// CHECK:STDOUT:   ret void, !dbg !21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.7e389eab4a7e5487:core.Destroy.Core"(ptr %self) #0 !dbg !22 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !28
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.a832eab4299edb30:core.Destroy.Core"(ptr %self) #0 !dbg !29 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.dbf7676be30a7aa8:core.Destroy.Core"(ptr %self) #0 !dbg !33 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !36
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @"_COp.dbf7676be30a7aa8:core.Destroy.Core", { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "member_init.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Fn", linkageName: "_CFn.Base.MemberInit", scope: null, file: !3, line: 6, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !8)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{null, !7}
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !8 = !{!9}
// CHECK:STDOUT: !9 = !DILocalVariable(arg: 1, scope: !4, type: !7)
// CHECK:STDOUT: !10 = !DILocation(line: 6, column: 3, scope: !4)
// CHECK:STDOUT: !11 = distinct !DISubprogram(name: "Fn", linkageName: "_CFn.MemberInit", scope: null, file: !3, line: 9, type: !12, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !12 = !DISubroutineType(types: !13)
// CHECK:STDOUT: !13 = !{null}
// CHECK:STDOUT: !14 = !DILocation(line: 10, column: 3, scope: !11)
// CHECK:STDOUT: !15 = !DILocation(line: 11, column: 3, scope: !11)
// CHECK:STDOUT: !16 = !DILocation(line: 13, column: 3, scope: !11)
// CHECK:STDOUT: !17 = !DILocation(line: 11, column: 17, scope: !11)
// CHECK:STDOUT: !18 = !DILocation(line: 11, column: 23, scope: !11)
// CHECK:STDOUT: !19 = !DILocation(line: 12, column: 3, scope: !11)
// CHECK:STDOUT: !20 = !DILocation(line: 13, column: 17, scope: !11)
// CHECK:STDOUT: !21 = !DILocation(line: 9, column: 1, scope: !11)
// CHECK:STDOUT: !22 = distinct !DISubprogram(name: "Op", linkageName: "_COp.7e389eab4a7e5487:core.Destroy.Core", scope: null, file: !3, line: 13, type: !23, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !26)
// CHECK:STDOUT: !23 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !24 = !{null, !25}
// CHECK:STDOUT: !25 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !26 = !{!27}
// CHECK:STDOUT: !27 = !DILocalVariable(arg: 1, scope: !22, type: !25)
// CHECK:STDOUT: !28 = !DILocation(line: 13, column: 3, scope: !22)
// CHECK:STDOUT: !29 = distinct !DISubprogram(name: "Op", linkageName: "_COp.a832eab4299edb30:core.Destroy.Core", scope: null, file: !3, line: 13, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !30)
// CHECK:STDOUT: !30 = !{!31}
// CHECK:STDOUT: !31 = !DILocalVariable(arg: 1, scope: !29, type: !7)
// CHECK:STDOUT: !32 = !DILocation(line: 13, column: 3, scope: !29)
// CHECK:STDOUT: !33 = distinct !DISubprogram(name: "Op", linkageName: "_COp.dbf7676be30a7aa8:core.Destroy.Core", scope: null, file: !3, line: 13, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !34)
// CHECK:STDOUT: !34 = !{!35}
// CHECK:STDOUT: !35 = !DILocalVariable(arg: 1, scope: !33, type: !7)
// CHECK:STDOUT: !36 = !DILocation(line: 13, column: 3, scope: !33)
// CHECK:STDOUT: ; ModuleID = 'member_brace_init.carbon'
// CHECK:STDOUT: source_filename = "member_brace_init.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @"_CBase.Main.$vtable" = unnamed_addr constant [1 x i32] [i32 trunc (i64 sub (i64 ptrtoint (ptr @_CF.Base.Main to i64), i64 ptrtoint (ptr @"_CBase.Main.$vtable" to i64)) to i32)]
// CHECK:STDOUT: @"_CDerived.Main.$vtable" = unnamed_addr constant [1 x i32] [i32 trunc (i64 sub (i64 ptrtoint (ptr @_CF.Base.Main to i64), i64 ptrtoint (ptr @"_CDerived.Main.$vtable" to i64)) to i32)]
// CHECK:STDOUT: @Derived.val.loc13_32.5 = internal constant { { ptr } } poison
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CF.Base.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CUse.Main() #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca { { ptr } }, align 8, !dbg !7
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !7
// CHECK:STDOUT:   %.loc13_32.2.base = getelementptr inbounds nuw { { ptr } }, ptr %_.var, i32 0, i32 0, !dbg !8
// CHECK:STDOUT:   %.loc13_31.2.vptr = getelementptr inbounds nuw { ptr }, ptr %.loc13_32.2.base, i32 0, i32 0, !dbg !9
// CHECK:STDOUT:   %.loc13_32.6.base = getelementptr inbounds nuw { { ptr } }, ptr %_.var, i32 0, i32 0, !dbg !8
// CHECK:STDOUT:   %.loc13_32.7.vptr = getelementptr inbounds nuw { ptr }, ptr %.loc13_32.6.base, i32 0, i32 0, !dbg !8
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 %_.var, ptr align 8 @Derived.val.loc13_32.5, i64 8, i1 false), !dbg !8
// CHECK:STDOUT:   store ptr @"_CDerived.Main.$vtable", ptr %.loc13_32.7.vptr, align 8, !dbg !8
// CHECK:STDOUT:   call void @"_COp.12e0d0434542305a:core.Destroy.Core"(ptr %_.var), !dbg !7
// CHECK:STDOUT:   ret void, !dbg !10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.a8f528cd70a29ff8:core.Destroy.Core"(ptr %self) #0 !dbg !11 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.96a711151205ee34:core.Destroy.Core"(ptr %self) #0 !dbg !18 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.4bfa84f6e0b7c617:core.Destroy.Core"(ptr %self) #0 !dbg !22 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !25
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.12e0d0434542305a:core.Destroy.Core"(ptr %self) #0 !dbg !26 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !29
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #2
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "member_brace_init.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Use", linkageName: "_CUse.Main", scope: null, file: !3, line: 12, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{null}
// CHECK:STDOUT: !7 = !DILocation(line: 13, column: 3, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 13, column: 21, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 13, column: 30, scope: !4)
// CHECK:STDOUT: !10 = !DILocation(line: 12, column: 1, scope: !4)
// CHECK:STDOUT: !11 = distinct !DISubprogram(name: "Op", linkageName: "_COp.a8f528cd70a29ff8:core.Destroy.Core", scope: null, file: !3, line: 13, type: !12, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !15)
// CHECK:STDOUT: !12 = !DISubroutineType(types: !13)
// CHECK:STDOUT: !13 = !{null, !14}
// CHECK:STDOUT: !14 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !15 = !{!16}
// CHECK:STDOUT: !16 = !DILocalVariable(arg: 1, scope: !11, type: !14)
// CHECK:STDOUT: !17 = !DILocation(line: 13, column: 3, scope: !11)
// CHECK:STDOUT: !18 = distinct !DISubprogram(name: "Op", linkageName: "_COp.96a711151205ee34:core.Destroy.Core", scope: null, file: !3, line: 13, type: !12, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !19)
// CHECK:STDOUT: !19 = !{!20}
// CHECK:STDOUT: !20 = !DILocalVariable(arg: 1, scope: !18, type: !14)
// CHECK:STDOUT: !21 = !DILocation(line: 13, column: 3, scope: !18)
// CHECK:STDOUT: !22 = distinct !DISubprogram(name: "Op", linkageName: "_COp.4bfa84f6e0b7c617:core.Destroy.Core", scope: null, file: !3, line: 13, type: !12, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !23)
// CHECK:STDOUT: !23 = !{!24}
// CHECK:STDOUT: !24 = !DILocalVariable(arg: 1, scope: !22, type: !14)
// CHECK:STDOUT: !25 = !DILocation(line: 13, column: 3, scope: !22)
// CHECK:STDOUT: !26 = distinct !DISubprogram(name: "Op", linkageName: "_COp.12e0d0434542305a:core.Destroy.Core", scope: null, file: !3, line: 13, type: !12, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !27)
// CHECK:STDOUT: !27 = !{!28}
// CHECK:STDOUT: !28 = !DILocalVariable(arg: 1, scope: !26, type: !14)
// CHECK:STDOUT: !29 = !DILocation(line: 13, column: 3, scope: !26)
// CHECK:STDOUT: ; ModuleID = 'call.carbon'
// CHECK:STDOUT: source_filename = "call.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @"_CBase.Main.$vtable" = unnamed_addr constant [1 x i32] [i32 trunc (i64 sub (i64 ptrtoint (ptr @_CF.Base.Main to i64), i64 ptrtoint (ptr @"_CBase.Main.$vtable" to i64)) to i32)]
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CF.Base.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CUse.Main(ptr %b) #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Base.F.call.vtable = load ptr, ptr %b, align 8, !dbg !10
// CHECK:STDOUT:   %Base.F.call = call ptr @llvm.load.relative.i32(ptr %Base.F.call.vtable, i32 0), !dbg !10
// CHECK:STDOUT:   call void %Base.F.call(ptr %b), !dbg !10
// CHECK:STDOUT:   ret void, !dbg !11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: read)
// CHECK:STDOUT: declare ptr @llvm.load.relative.i32(ptr, i32) #1
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: read) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "call.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Use", linkageName: "_CUse.Main", scope: null, file: !3, line: 8, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !8)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{null, !7}
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !8 = !{!9}
// CHECK:STDOUT: !9 = !DILocalVariable(arg: 1, scope: !4, type: !7)
// CHECK:STDOUT: !10 = !DILocation(line: 9, column: 3, scope: !4)
// CHECK:STDOUT: !11 = !DILocation(line: 8, column: 1, scope: !4)
// CHECK:STDOUT: ; ModuleID = 'call_impl.carbon'
// CHECK:STDOUT: source_filename = "call_impl.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @"_CBase.Main.$vtable" = unnamed_addr constant [1 x i32] [i32 trunc (i64 sub (i64 ptrtoint (ptr @_CF.Base.Main to i64), i64 ptrtoint (ptr @"_CBase.Main.$vtable" to i64)) to i32)]
// CHECK:STDOUT: @"_CDerived.Main.$vtable" = unnamed_addr constant [1 x i32] [i32 trunc (i64 sub (i64 ptrtoint (ptr @_CF.Derived.Main to i64), i64 ptrtoint (ptr @"_CDerived.Main.$vtable" to i64)) to i32)]
// CHECK:STDOUT: @Derived.val.loc14_32.5 = internal constant { { ptr } } poison
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CF.Base.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CF.Derived.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CUse.Main() #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %v.var = alloca { { ptr } }, align 8, !dbg !7
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %v.var), !dbg !7
// CHECK:STDOUT:   %.loc14_32.2.base = getelementptr inbounds nuw { { ptr } }, ptr %v.var, i32 0, i32 0, !dbg !8
// CHECK:STDOUT:   %.loc14_31.2.vptr = getelementptr inbounds nuw { ptr }, ptr %.loc14_32.2.base, i32 0, i32 0, !dbg !9
// CHECK:STDOUT:   %.loc14_32.6.base = getelementptr inbounds nuw { { ptr } }, ptr %v.var, i32 0, i32 0, !dbg !8
// CHECK:STDOUT:   %.loc14_32.7.vptr = getelementptr inbounds nuw { ptr }, ptr %.loc14_32.6.base, i32 0, i32 0, !dbg !8
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 %v.var, ptr align 8 @Derived.val.loc14_32.5, i64 8, i1 false), !dbg !8
// CHECK:STDOUT:   store ptr @"_CDerived.Main.$vtable", ptr %.loc14_32.7.vptr, align 8, !dbg !8
// CHECK:STDOUT:   %.loc15_3.1.base = getelementptr inbounds nuw { { ptr } }, ptr %v.var, i32 0, i32 0, !dbg !10
// CHECK:STDOUT:   %Base.F.call.vtable = load ptr, ptr %.loc15_3.1.base, align 8, !dbg !10
// CHECK:STDOUT:   %Base.F.call = call ptr @llvm.load.relative.i32(ptr %Base.F.call.vtable, i32 0), !dbg !10
// CHECK:STDOUT:   call void %Base.F.call(ptr %.loc15_3.1.base), !dbg !10
// CHECK:STDOUT:   call void @"_COp.12e0d0434542305a:core.Destroy.Core"(ptr %v.var), !dbg !7
// CHECK:STDOUT:   ret void, !dbg !11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.a8f528cd70a29ff8:core.Destroy.Core"(ptr %self) #0 !dbg !12 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.96a711151205ee34:core.Destroy.Core"(ptr %self) #0 !dbg !19 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !22
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.4bfa84f6e0b7c617:core.Destroy.Core"(ptr %self) #0 !dbg !23 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !26
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.12e0d0434542305a:core.Destroy.Core"(ptr %self) #0 !dbg !27 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !30
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: read)
// CHECK:STDOUT: declare ptr @llvm.load.relative.i32(ptr, i32) #3
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #3 = { nocallback nofree nosync nounwind willreturn memory(argmem: read) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "call_impl.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Use", linkageName: "_CUse.Main", scope: null, file: !3, line: 13, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{null}
// CHECK:STDOUT: !7 = !DILocation(line: 14, column: 3, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 14, column: 21, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 14, column: 30, scope: !4)
// CHECK:STDOUT: !10 = !DILocation(line: 15, column: 3, scope: !4)
// CHECK:STDOUT: !11 = !DILocation(line: 13, column: 1, scope: !4)
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "Op", linkageName: "_COp.a8f528cd70a29ff8:core.Destroy.Core", scope: null, file: !3, line: 14, type: !13, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !16)
// CHECK:STDOUT: !13 = !DISubroutineType(types: !14)
// CHECK:STDOUT: !14 = !{null, !15}
// CHECK:STDOUT: !15 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !16 = !{!17}
// CHECK:STDOUT: !17 = !DILocalVariable(arg: 1, scope: !12, type: !15)
// CHECK:STDOUT: !18 = !DILocation(line: 14, column: 3, scope: !12)
// CHECK:STDOUT: !19 = distinct !DISubprogram(name: "Op", linkageName: "_COp.96a711151205ee34:core.Destroy.Core", scope: null, file: !3, line: 14, type: !13, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !20)
// CHECK:STDOUT: !20 = !{!21}
// CHECK:STDOUT: !21 = !DILocalVariable(arg: 1, scope: !19, type: !15)
// CHECK:STDOUT: !22 = !DILocation(line: 14, column: 3, scope: !19)
// CHECK:STDOUT: !23 = distinct !DISubprogram(name: "Op", linkageName: "_COp.4bfa84f6e0b7c617:core.Destroy.Core", scope: null, file: !3, line: 14, type: !13, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !24)
// CHECK:STDOUT: !24 = !{!25}
// CHECK:STDOUT: !25 = !DILocalVariable(arg: 1, scope: !23, type: !15)
// CHECK:STDOUT: !26 = !DILocation(line: 14, column: 3, scope: !23)
// CHECK:STDOUT: !27 = distinct !DISubprogram(name: "Op", linkageName: "_COp.12e0d0434542305a:core.Destroy.Core", scope: null, file: !3, line: 14, type: !13, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !28)
// CHECK:STDOUT: !28 = !{!29}
// CHECK:STDOUT: !29 = !DILocalVariable(arg: 1, scope: !27, type: !15)
// CHECK:STDOUT: !30 = !DILocation(line: 14, column: 3, scope: !27)
// CHECK:STDOUT: ; ModuleID = 'generic_noop.carbon'
// CHECK:STDOUT: source_filename = "generic_noop.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "generic_noop.carbon", directory: "")
// CHECK:STDOUT: ; ModuleID = 'generic_use.carbon'
// CHECK:STDOUT: source_filename = "generic_use.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @"_CBase.Main.$vtable.0a7c28d4c8b7893d" = unnamed_addr constant [1 x i32] [i32 trunc (i64 sub (i64 ptrtoint (ptr @_CF.Base.Main.0a7c28d4c8b7893d to i64), i64 ptrtoint (ptr @"_CBase.Main.$vtable.0a7c28d4c8b7893d" to i64)) to i32)]
// CHECK:STDOUT: @"_CBase.Main.$vtable.bdccbdc18ce76efa" = unnamed_addr constant [1 x i32] [i32 trunc (i64 sub (i64 ptrtoint (ptr @_CF.Base.Main.bdccbdc18ce76efa to i64), i64 ptrtoint (ptr @"_CBase.Main.$vtable.bdccbdc18ce76efa" to i64)) to i32)]
// CHECK:STDOUT: @Base.val.0c4.loc16_3 = internal constant { ptr } { ptr @"_CBase.Main.$vtable.0a7c28d4c8b7893d" }
// CHECK:STDOUT: @Base.val.c5e.loc17_3 = internal constant { ptr } { ptr @"_CBase.Main.$vtable.bdccbdc18ce76efa" }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var.loc16 = alloca { ptr }, align 8, !dbg !7
// CHECK:STDOUT:   %_.var.loc17 = alloca { ptr }, align 8, !dbg !8
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc16), !dbg !7
// CHECK:STDOUT:   %.loc16_22.2.vptr = getelementptr inbounds nuw { ptr }, ptr %_.var.loc16, i32 0, i32 0, !dbg !9
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 %_.var.loc16, ptr align 8 @Base.val.0c4.loc16_3, i64 8, i1 false), !dbg !7
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc17), !dbg !8
// CHECK:STDOUT:   %.loc17_22.2.vptr = getelementptr inbounds nuw { ptr }, ptr %_.var.loc17, i32 0, i32 0, !dbg !10
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 %_.var.loc17, ptr align 8 @Base.val.c5e.loc17_3, i64 8, i1 false), !dbg !8
// CHECK:STDOUT:   call void @"_COp.98cef783bfe00900:core.Destroy.Core"(ptr %_.var.loc17), !dbg !8
// CHECK:STDOUT:   call void @"_COp.4cabc7f3a3e0371f:core.Destroy.Core"(ptr %_.var.loc16), !dbg !7
// CHECK:STDOUT:   ret void, !dbg !11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.b8216ad694fdad1d:core.Destroy.Core"(ptr %self) #0 !dbg !12 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.a65450da9a7ba013:core.Destroy.Core"(ptr %self) #0 !dbg !19 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !22
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.47037bb76eb05287:core.Destroy.Core"(ptr %self) #0 !dbg !23 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !26
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.a8f528cd70a29ff8:core.Destroy.Core"(ptr %self) #0 !dbg !27 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !30
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.98cef783bfe00900:core.Destroy.Core"(ptr %self) #0 !dbg !31 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !34
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.4cabc7f3a3e0371f:core.Destroy.Core"(ptr %self) #0 !dbg !35 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !38
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Base.Main.0a7c28d4c8b7893d(ptr %self) #0 !dbg !39 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca {}, align 8, !dbg !42
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !42
// CHECK:STDOUT:   call void @"_COp.b8216ad694fdad1d:core.Destroy.Core"(ptr %_.var), !dbg !42
// CHECK:STDOUT:   ret void, !dbg !43
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Base.Main.bdccbdc18ce76efa(ptr %self) #0 !dbg !44 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca { {} }, align 8, !dbg !47
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !47
// CHECK:STDOUT:   call void @"_COp.47037bb76eb05287:core.Destroy.Core"(ptr %_.var), !dbg !47
// CHECK:STDOUT:   ret void, !dbg !48
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "generic_use.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !3, line: 15, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{null}
// CHECK:STDOUT: !7 = !DILocation(line: 16, column: 3, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 17, column: 3, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 16, column: 21, scope: !4)
// CHECK:STDOUT: !10 = !DILocation(line: 17, column: 21, scope: !4)
// CHECK:STDOUT: !11 = !DILocation(line: 15, column: 1, scope: !4)
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "Op", linkageName: "_COp.b8216ad694fdad1d:core.Destroy.Core", scope: null, file: !3, line: 16, type: !13, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !16)
// CHECK:STDOUT: !13 = !DISubroutineType(types: !14)
// CHECK:STDOUT: !14 = !{null, !15}
// CHECK:STDOUT: !15 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !16 = !{!17}
// CHECK:STDOUT: !17 = !DILocalVariable(arg: 1, scope: !12, type: !15)
// CHECK:STDOUT: !18 = !DILocation(line: 16, column: 10, scope: !12)
// CHECK:STDOUT: !19 = distinct !DISubprogram(name: "Op", linkageName: "_COp.a65450da9a7ba013:core.Destroy.Core", scope: null, file: !3, line: 17, type: !13, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !20)
// CHECK:STDOUT: !20 = !{!21}
// CHECK:STDOUT: !21 = !DILocalVariable(arg: 1, scope: !19, type: !15)
// CHECK:STDOUT: !22 = !DILocation(line: 17, column: 10, scope: !19)
// CHECK:STDOUT: !23 = distinct !DISubprogram(name: "Op", linkageName: "_COp.47037bb76eb05287:core.Destroy.Core", scope: null, file: !3, line: 17, type: !13, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !24)
// CHECK:STDOUT: !24 = !{!25}
// CHECK:STDOUT: !25 = !DILocalVariable(arg: 1, scope: !23, type: !15)
// CHECK:STDOUT: !26 = !DILocation(line: 17, column: 10, scope: !23)
// CHECK:STDOUT: !27 = distinct !DISubprogram(name: "Op", linkageName: "_COp.a8f528cd70a29ff8:core.Destroy.Core", scope: null, file: !3, line: 17, type: !13, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !28)
// CHECK:STDOUT: !28 = !{!29}
// CHECK:STDOUT: !29 = !DILocalVariable(arg: 1, scope: !27, type: !15)
// CHECK:STDOUT: !30 = !DILocation(line: 17, column: 3, scope: !27)
// CHECK:STDOUT: !31 = distinct !DISubprogram(name: "Op", linkageName: "_COp.98cef783bfe00900:core.Destroy.Core", scope: null, file: !3, line: 17, type: !13, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !32)
// CHECK:STDOUT: !32 = !{!33}
// CHECK:STDOUT: !33 = !DILocalVariable(arg: 1, scope: !31, type: !15)
// CHECK:STDOUT: !34 = !DILocation(line: 17, column: 3, scope: !31)
// CHECK:STDOUT: !35 = distinct !DISubprogram(name: "Op", linkageName: "_COp.4cabc7f3a3e0371f:core.Destroy.Core", scope: null, file: !3, line: 16, type: !13, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !36)
// CHECK:STDOUT: !36 = !{!37}
// CHECK:STDOUT: !37 = !DILocalVariable(arg: 1, scope: !35, type: !15)
// CHECK:STDOUT: !38 = !DILocation(line: 16, column: 3, scope: !35)
// CHECK:STDOUT: !39 = distinct !DISubprogram(name: "F", linkageName: "_CF.Base.Main.0a7c28d4c8b7893d", scope: null, file: !3, line: 5, type: !13, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !40)
// CHECK:STDOUT: !40 = !{!41}
// CHECK:STDOUT: !41 = !DILocalVariable(arg: 1, scope: !39, type: !15)
// CHECK:STDOUT: !42 = !DILocation(line: 6, column: 5, scope: !39)
// CHECK:STDOUT: !43 = !DILocation(line: 5, column: 3, scope: !39)
// CHECK:STDOUT: !44 = distinct !DISubprogram(name: "F", linkageName: "_CF.Base.Main.bdccbdc18ce76efa", scope: null, file: !3, line: 5, type: !13, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !45)
// CHECK:STDOUT: !45 = !{!46}
// CHECK:STDOUT: !46 = !DILocalVariable(arg: 1, scope: !44, type: !15)
// CHECK:STDOUT: !47 = !DILocation(line: 6, column: 5, scope: !44)
// CHECK:STDOUT: !48 = !DILocation(line: 5, column: 3, scope: !44)
// CHECK:STDOUT: ; ModuleID = 'generic_base.carbon'
// CHECK:STDOUT: source_filename = "generic_base.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @"_CDerived.Main.$vtable" = unnamed_addr constant [1 x i32] [i32 trunc (i64 sub (i64 ptrtoint (ptr @_CF.Base.Main.c914672a22d4cb10 to i64), i64 ptrtoint (ptr @"_CDerived.Main.$vtable" to i64)) to i32)]
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMake.Main() #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca { { ptr } }, align 8, !dbg !7
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !7
// CHECK:STDOUT:   call void @"_COp.12e0d0434542305a:core.Destroy.Core"(ptr %_.var), !dbg !7
// CHECK:STDOUT:   ret void, !dbg !8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.a8f528cd70a29ff8:core.Destroy.Core"(ptr %self) #0 !dbg !9 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !15
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.8ac3c4c0cf7cf991:core.Destroy.Core"(ptr %self) #0 !dbg !16 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.16489017edc6364e:core.Destroy.Core"(ptr %self) #0 !dbg !20 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !23
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.12e0d0434542305a:core.Destroy.Core"(ptr %self) #0 !dbg !24 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !27
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Base.Main.c914672a22d4cb10(ptr %self) #0 !dbg !28 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !31
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "generic_base.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Make", linkageName: "_CMake.Main", scope: null, file: !3, line: 14, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{null}
// CHECK:STDOUT: !7 = !DILocation(line: 15, column: 3, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 14, column: 1, scope: !4)
// CHECK:STDOUT: !9 = distinct !DISubprogram(name: "Op", linkageName: "_COp.a8f528cd70a29ff8:core.Destroy.Core", scope: null, file: !3, line: 15, type: !10, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !13)
// CHECK:STDOUT: !10 = !DISubroutineType(types: !11)
// CHECK:STDOUT: !11 = !{null, !12}
// CHECK:STDOUT: !12 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !13 = !{!14}
// CHECK:STDOUT: !14 = !DILocalVariable(arg: 1, scope: !9, type: !12)
// CHECK:STDOUT: !15 = !DILocation(line: 15, column: 3, scope: !9)
// CHECK:STDOUT: !16 = distinct !DISubprogram(name: "Op", linkageName: "_COp.8ac3c4c0cf7cf991:core.Destroy.Core", scope: null, file: !3, line: 15, type: !10, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !17)
// CHECK:STDOUT: !17 = !{!18}
// CHECK:STDOUT: !18 = !DILocalVariable(arg: 1, scope: !16, type: !12)
// CHECK:STDOUT: !19 = !DILocation(line: 15, column: 3, scope: !16)
// CHECK:STDOUT: !20 = distinct !DISubprogram(name: "Op", linkageName: "_COp.16489017edc6364e:core.Destroy.Core", scope: null, file: !3, line: 15, type: !10, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !21)
// CHECK:STDOUT: !21 = !{!22}
// CHECK:STDOUT: !22 = !DILocalVariable(arg: 1, scope: !20, type: !12)
// CHECK:STDOUT: !23 = !DILocation(line: 15, column: 3, scope: !20)
// CHECK:STDOUT: !24 = distinct !DISubprogram(name: "Op", linkageName: "_COp.12e0d0434542305a:core.Destroy.Core", scope: null, file: !3, line: 15, type: !10, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !25)
// CHECK:STDOUT: !25 = !{!26}
// CHECK:STDOUT: !26 = !DILocalVariable(arg: 1, scope: !24, type: !12)
// CHECK:STDOUT: !27 = !DILocation(line: 15, column: 3, scope: !24)
// CHECK:STDOUT: !28 = distinct !DISubprogram(name: "F", linkageName: "_CF.Base.Main.c914672a22d4cb10", scope: null, file: !3, line: 5, type: !10, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !29)
// CHECK:STDOUT: !29 = !{!30}
// CHECK:STDOUT: !30 = !DILocalVariable(arg: 1, scope: !28, type: !12)
// CHECK:STDOUT: !31 = !DILocation(line: 5, column: 3, scope: !28)
// CHECK:STDOUT: ; ModuleID = 'thunk.carbon'
// CHECK:STDOUT: source_filename = "thunk.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @"_CBase.Main.$vtable" = unnamed_addr constant [1 x i32] [i32 trunc (i64 sub (i64 ptrtoint (ptr @_CF.Base.Main to i64), i64 ptrtoint (ptr @"_CBase.Main.$vtable" to i64)) to i32)]
// CHECK:STDOUT: @"_CDerived.Main.$vtable" = unnamed_addr constant [1 x i32] [i32 trunc (i64 sub (i64 ptrtoint (ptr @"_CF:thunk:Base.Main:Derived.Main" to i64), i64 ptrtoint (ptr @"_CDerived.Main.$vtable" to i64)) to i32)]
// CHECK:STDOUT: @From.val = internal constant {} zeroinitializer
// CHECK:STDOUT: @From.val.loc18_14 = internal constant {} zeroinitializer
// CHECK:STDOUT: @Derived.val.loc23_31.5 = internal constant { { ptr } } poison
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_CConvert.From.Main:ImplicitAs.a0663b0b66554b38.Core"(ptr sret({}), ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CF.Base.Main(ptr sret({}), ptr, ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Derived.Main(ptr sret({}) %return, ptr %self, ptr %n) #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %return, ptr align 1 @From.val.loc18_14, i64 0, i1 false), !dbg !11
// CHECK:STDOUT:   ret void, !dbg !11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define void @"_CF:thunk:Base.Main:Derived.Main"(ptr sret({}) %return, ptr %self, ptr %n) #1 !dbg !12 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc17_58.1.temp = alloca {}, align 8, !dbg !16
// CHECK:STDOUT:   %.loc12_33.1.temp = alloca {}, align 8, !dbg !17
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc17_58.1.temp), !dbg !16
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc12_33.1.temp), !dbg !17
// CHECK:STDOUT:   call void @"_CConvert.From.Main:ImplicitAs.a0663b0b66554b38.Core"(ptr %.loc12_33.1.temp, ptr %n), !dbg !17
// CHECK:STDOUT:   call void @_CF.Derived.Main(ptr %.loc17_58.1.temp, ptr %self, ptr %.loc12_33.1.temp), !dbg !16
// CHECK:STDOUT:   call void @"_CConvert.From.Main:ImplicitAs.a0663b0b66554b38.Core"(ptr %return, ptr %.loc17_58.1.temp), !dbg !16
// CHECK:STDOUT:   call void @"_COp.b59cd0c4bb4fc669:core.Destroy.Core"(ptr %.loc17_58.1.temp), !dbg !16
// CHECK:STDOUT:   call void @"_COp.17affc8c9ed5036b:core.Destroy.Core"(ptr %.loc12_33.1.temp), !dbg !17
// CHECK:STDOUT:   ret void, !dbg !16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.b59cd0c4bb4fc669:core.Destroy.Core"(ptr %self) #0 !dbg !18 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !23
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.17affc8c9ed5036b:core.Destroy.Core"(ptr %self) #0 !dbg !24 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !27
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CUse.Main() #0 !dbg !28 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %d.var = alloca { { ptr } }, align 8, !dbg !31
// CHECK:STDOUT:   %.loc25_10.1.temp = alloca {}, align 8, !dbg !32
// CHECK:STDOUT:   %.loc25_9.2.temp = alloca {}, align 8, !dbg !33
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %d.var), !dbg !31
// CHECK:STDOUT:   %.loc23_31.2.base = getelementptr inbounds nuw { { ptr } }, ptr %d.var, i32 0, i32 0, !dbg !34
// CHECK:STDOUT:   %.loc23_30.2.vptr = getelementptr inbounds nuw { ptr }, ptr %.loc23_31.2.base, i32 0, i32 0, !dbg !35
// CHECK:STDOUT:   %.loc23_31.6.base = getelementptr inbounds nuw { { ptr } }, ptr %d.var, i32 0, i32 0, !dbg !34
// CHECK:STDOUT:   %.loc23_31.7.vptr = getelementptr inbounds nuw { ptr }, ptr %.loc23_31.6.base, i32 0, i32 0, !dbg !34
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 %d.var, ptr align 8 @Derived.val.loc23_31.5, i64 8, i1 false), !dbg !34
// CHECK:STDOUT:   store ptr @"_CDerived.Main.$vtable", ptr %.loc23_31.7.vptr, align 8, !dbg !34
// CHECK:STDOUT:   %.loc24_18.2.base = getelementptr inbounds nuw { { ptr } }, ptr %d.var, i32 0, i32 0, !dbg !36
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc25_10.1.temp), !dbg !32
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc25_9.2.temp), !dbg !33
// CHECK:STDOUT:   %Base.F.call.vtable = load ptr, ptr %.loc25_10.1.temp, align 8, !dbg !32
// CHECK:STDOUT:   %Base.F.call = call ptr @llvm.load.relative.i32(ptr %Base.F.call.vtable, i32 0), !dbg !32
// CHECK:STDOUT:   call void %Base.F.call(ptr %.loc25_10.1.temp, ptr %.loc24_18.2.base, ptr @From.val.loc18_14), !dbg !32
// CHECK:STDOUT:   call void @"_COp.17affc8c9ed5036b:core.Destroy.Core"(ptr %.loc25_10.1.temp), !dbg !32
// CHECK:STDOUT:   call void @"_COp.b59cd0c4bb4fc669:core.Destroy.Core"(ptr @From.val), !dbg !33
// CHECK:STDOUT:   call void @"_COp.12e0d0434542305a:core.Destroy.Core"(ptr %d.var), !dbg !31
// CHECK:STDOUT:   ret void, !dbg !37
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.a8f528cd70a29ff8:core.Destroy.Core"(ptr %self) #0 !dbg !38 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !41
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.96a711151205ee34:core.Destroy.Core"(ptr %self) #0 !dbg !42 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !45
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.4bfa84f6e0b7c617:core.Destroy.Core"(ptr %self) #0 !dbg !46 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !49
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.12e0d0434542305a:core.Destroy.Core"(ptr %self) #0 !dbg !50 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !53
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: read)
// CHECK:STDOUT: declare ptr @llvm.load.relative.i32(ptr, i32) #4
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 4, 3, 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline nounwind }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #3 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #4 = { nocallback nofree nosync nounwind willreturn memory(argmem: read) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "thunk.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "F", linkageName: "_CF.Derived.Main", scope: null, file: !3, line: 17, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !8)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{!7, !7, !7}
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !8 = !{!9, !10}
// CHECK:STDOUT: !9 = !DILocalVariable(arg: 1, scope: !4, type: !7)
// CHECK:STDOUT: !10 = !DILocalVariable(arg: 2, scope: !4, type: !7)
// CHECK:STDOUT: !11 = !DILocation(line: 18, column: 5, scope: !4)
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "F", linkageName: "_CF:thunk:Base.Main:Derived.Main", scope: null, file: !3, line: 17, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !13)
// CHECK:STDOUT: !13 = !{!14, !15}
// CHECK:STDOUT: !14 = !DILocalVariable(arg: 1, scope: !12, type: !7)
// CHECK:STDOUT: !15 = !DILocalVariable(arg: 2, scope: !12, type: !7)
// CHECK:STDOUT: !16 = !DILocation(line: 17, column: 3, scope: !12)
// CHECK:STDOUT: !17 = !DILocation(line: 12, column: 32, scope: !12)
// CHECK:STDOUT: !18 = distinct !DISubprogram(name: "Op", linkageName: "_COp.b59cd0c4bb4fc669:core.Destroy.Core", scope: null, file: !3, line: 17, type: !19, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !21)
// CHECK:STDOUT: !19 = !DISubroutineType(types: !20)
// CHECK:STDOUT: !20 = !{null, !7}
// CHECK:STDOUT: !21 = !{!22}
// CHECK:STDOUT: !22 = !DILocalVariable(arg: 1, scope: !18, type: !7)
// CHECK:STDOUT: !23 = !DILocation(line: 17, column: 3, scope: !18)
// CHECK:STDOUT: !24 = distinct !DISubprogram(name: "Op", linkageName: "_COp.17affc8c9ed5036b:core.Destroy.Core", scope: null, file: !3, line: 12, type: !19, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !25)
// CHECK:STDOUT: !25 = !{!26}
// CHECK:STDOUT: !26 = !DILocalVariable(arg: 1, scope: !24, type: !7)
// CHECK:STDOUT: !27 = !DILocation(line: 12, column: 32, scope: !24)
// CHECK:STDOUT: !28 = distinct !DISubprogram(name: "Use", linkageName: "_CUse.Main", scope: null, file: !3, line: 22, type: !29, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !29 = !DISubroutineType(types: !30)
// CHECK:STDOUT: !30 = !{null}
// CHECK:STDOUT: !31 = !DILocation(line: 23, column: 3, scope: !28)
// CHECK:STDOUT: !32 = !DILocation(line: 25, column: 3, scope: !28)
// CHECK:STDOUT: !33 = !DILocation(line: 25, column: 8, scope: !28)
// CHECK:STDOUT: !34 = !DILocation(line: 23, column: 20, scope: !28)
// CHECK:STDOUT: !35 = !DILocation(line: 23, column: 29, scope: !28)
// CHECK:STDOUT: !36 = !DILocation(line: 24, column: 18, scope: !28)
// CHECK:STDOUT: !37 = !DILocation(line: 22, column: 1, scope: !28)
// CHECK:STDOUT: !38 = distinct !DISubprogram(name: "Op", linkageName: "_COp.a8f528cd70a29ff8:core.Destroy.Core", scope: null, file: !3, line: 23, type: !19, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !39)
// CHECK:STDOUT: !39 = !{!40}
// CHECK:STDOUT: !40 = !DILocalVariable(arg: 1, scope: !38, type: !7)
// CHECK:STDOUT: !41 = !DILocation(line: 23, column: 3, scope: !38)
// CHECK:STDOUT: !42 = distinct !DISubprogram(name: "Op", linkageName: "_COp.96a711151205ee34:core.Destroy.Core", scope: null, file: !3, line: 23, type: !19, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !43)
// CHECK:STDOUT: !43 = !{!44}
// CHECK:STDOUT: !44 = !DILocalVariable(arg: 1, scope: !42, type: !7)
// CHECK:STDOUT: !45 = !DILocation(line: 23, column: 3, scope: !42)
// CHECK:STDOUT: !46 = distinct !DISubprogram(name: "Op", linkageName: "_COp.4bfa84f6e0b7c617:core.Destroy.Core", scope: null, file: !3, line: 23, type: !19, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !47)
// CHECK:STDOUT: !47 = !{!48}
// CHECK:STDOUT: !48 = !DILocalVariable(arg: 1, scope: !46, type: !7)
// CHECK:STDOUT: !49 = !DILocation(line: 23, column: 3, scope: !46)
// CHECK:STDOUT: !50 = distinct !DISubprogram(name: "Op", linkageName: "_COp.12e0d0434542305a:core.Destroy.Core", scope: null, file: !3, line: 23, type: !19, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !51)
// CHECK:STDOUT: !51 = !{!52}
// CHECK:STDOUT: !52 = !DILocalVariable(arg: 1, scope: !50, type: !7)
// CHECK:STDOUT: !53 = !DILocation(line: 23, column: 3, scope: !50)
