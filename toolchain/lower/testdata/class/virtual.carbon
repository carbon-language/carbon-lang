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
  virtual fn Fn(unused self) { }
}

class Derived {
  extend base: Intermediate;
  override fn Fn(unused self) { }
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
  virtual fn Fn(unused self) { }
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
  virtual fn F(self);
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
  virtual fn F(self);
}

fn Use(b: Base) {
  b.F();
}

// --- call_impl.carbon

library "[[@TEST_NAME]]";

base class Base {
  virtual fn F(self);
}

class Derived {
  extend base: Base;
  override fn F(self);
}

fn Use() {
  var v : Derived = {.base = {}};
  v.F();
}

// --- generic_noop.carbon

library "[[@TEST_NAME]]";

base class Base(T: type) {
  virtual fn F(unused self) { }
}

// --- generic_use.carbon

library "[[@TEST_NAME]]";

base class Base(T: Core.Destroy) {
  virtual fn F(unused self) {
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

base class Base(T: type) {
  virtual fn F(unused self) { }
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
  fn Convert(self: From) -> To;
}

base class Base {
  virtual fn F(ref self, n: From) -> To;
}

class Derived {
  extend base: Base;
  override fn F(unused self, unused n: To) -> From {
    return {};
  }
}

fn Use() {
  var d: Derived = {.base = {}};
  let p: Base* = &d;
  p->F({});
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'classes.carbon'
// CHECK:STDOUT: source_filename = "classes.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @"_CIntermediate.Classes.$vtable" = unnamed_addr constant [1 x i32] [i32 trunc (i64 sub (i64 ptrtoint (ptr @_CFn.Intermediate.Classes to i64), i64 ptrtoint (ptr @"_CIntermediate.Classes.$vtable" to i64)) to i32)]
// CHECK:STDOUT: @"_CDerived.Classes.$vtable" = unnamed_addr constant [1 x i32] [i32 trunc (i64 sub (i64 ptrtoint (ptr @"_CFn:thunk:Intermediate.Classes:Derived.Classes" to i64), i64 ptrtoint (ptr @"_CDerived.Classes.$vtable" to i64)) to i32)]
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CFn.Intermediate.Classes(ptr %self) #0 !dbg !25 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !28
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CFn.Derived.Classes(ptr %self) #0 !dbg !30 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !33
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_CFn:thunk:Intermediate.Classes:Derived.Classes"(ptr %self) #1 !dbg !35 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CFn.Derived.Classes(ptr %self), !dbg !38
// CHECK:STDOUT:   ret void, !dbg !38
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!21}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIFile(filename: "classes.carbon", directory: "")
// CHECK:STDOUT: !21 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !20, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !22 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !23 = !{null, !22}
// CHECK:STDOUT: !24 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !25 = distinct !DISubprogram(name: "Fn", linkageName: "_CFn.Intermediate.Classes", scope: null, file: !20, line: 9, type: !24, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !29)
// CHECK:STDOUT: !26 = !DILocalVariable(arg: 1, scope: !25, type: !22)
// CHECK:STDOUT: !28 = !DILocation(line: 9, column: 3, scope: !25)
// CHECK:STDOUT: !29 = !{!26}
// CHECK:STDOUT: !30 = distinct !DISubprogram(name: "Fn", linkageName: "_CFn.Derived.Classes", scope: null, file: !20, line: 14, type: !24, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !34)
// CHECK:STDOUT: !31 = !DILocalVariable(arg: 1, scope: !30, type: !22)
// CHECK:STDOUT: !33 = !DILocation(line: 14, column: 3, scope: !30)
// CHECK:STDOUT: !34 = !{!31}
// CHECK:STDOUT: !35 = distinct !DISubprogram(name: "Fn", linkageName: "_CFn:thunk:Intermediate.Classes:Derived.Classes", scope: null, file: !20, line: 14, type: !24, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !39)
// CHECK:STDOUT: !36 = !DILocalVariable(arg: 1, scope: !35, type: !22)
// CHECK:STDOUT: !38 = !DILocation(line: 14, column: 3, scope: !35)
// CHECK:STDOUT: !39 = !{!36}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'create.carbon'
// CHECK:STDOUT: source_filename = "create.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @"_CIntermediate.Classes.$vtable" = external unnamed_addr constant ptr
// CHECK:STDOUT: @"_CDerived.Classes.$vtable" = external unnamed_addr constant ptr
// CHECK:STDOUT: @Base.val.loc7_3 = internal constant {} zeroinitializer
// CHECK:STDOUT: @Intermediate.val.54b.loc8_3 = internal constant { ptr, {} } { ptr @"_CIntermediate.Classes.$vtable", {} zeroinitializer }
// CHECK:STDOUT: @Derived.val.loc9_49.5 = internal constant { { ptr, {} } } { { ptr, {} } { ptr poison, {} zeroinitializer } }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCreate.Create() #0 !dbg !44 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var.loc7 = alloca {}, align 1, !dbg !45
// CHECK:STDOUT:   %_.var.loc8 = alloca { ptr, {} }, align 8, !dbg !47
// CHECK:STDOUT:   %_.var.loc9 = alloca { { ptr, {} } }, align 8, !dbg !50
// CHECK:STDOUT:   %_.var.loc12 = alloca { { ptr, {} } }, align 8, !dbg !54
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc7), !dbg !45
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %_.var.loc7, ptr align 1 @Base.val.loc7_3, i64 0, i1 false), !dbg !45
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc8), !dbg !47
// CHECK:STDOUT:   %.loc8_44.2.vptr = getelementptr inbounds nuw { ptr, {} }, ptr %_.var.loc8, i32 0, i32 0, !dbg !48
// CHECK:STDOUT:   %.loc8_44.4.base = getelementptr inbounds nuw { ptr, {} }, ptr %_.var.loc8, i32 0, i32 1, !dbg !48
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 %_.var.loc8, ptr align 8 @Intermediate.val.54b.loc8_3, i64 8, i1 false), !dbg !47
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc9), !dbg !50
// CHECK:STDOUT:   %.loc9_49.2.base = getelementptr inbounds nuw { { ptr, {} } }, ptr %_.var.loc9, i32 0, i32 0, !dbg !51
// CHECK:STDOUT:   %.loc9_48.2.vptr = getelementptr inbounds nuw { ptr, {} }, ptr %.loc9_49.2.base, i32 0, i32 0, !dbg !52
// CHECK:STDOUT:   %.loc9_48.4.base = getelementptr inbounds nuw { ptr, {} }, ptr %.loc9_49.2.base, i32 0, i32 1, !dbg !52
// CHECK:STDOUT:   %.loc9_49.6.base = getelementptr inbounds nuw { { ptr, {} } }, ptr %_.var.loc9, i32 0, i32 0, !dbg !51
// CHECK:STDOUT:   %.loc9_49.7.vptr = getelementptr inbounds nuw { ptr, {} }, ptr %.loc9_49.6.base, i32 0, i32 0, !dbg !51
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 %_.var.loc9, ptr align 8 @Derived.val.loc9_49.5, i64 8, i1 false), !dbg !51
// CHECK:STDOUT:   store ptr @"_CDerived.Classes.$vtable", ptr %.loc9_49.7.vptr, align 8, !dbg !51
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc12), !dbg !54
// CHECK:STDOUT:   ret void, !dbg !56
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CFn.Intermediate.Classes(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_CFn:thunk:Intermediate.Classes:Derived.Classes"(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CUse.Create(ptr %v) #0 !dbg !57 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Intermediate.Fn.call.vtable = load ptr, ptr %v, align 8, !dbg !60
// CHECK:STDOUT:   %Intermediate.Fn.call = call ptr @llvm.load.relative.i32(ptr %Intermediate.Fn.call.vtable, i32 0), !dbg !60
// CHECK:STDOUT:   call void %Intermediate.Fn.call(ptr %v), !dbg !60
// CHECK:STDOUT:   ret void, !dbg !61
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: read)
// CHECK:STDOUT: declare ptr @llvm.load.relative.i32(ptr, i32) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: read) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!41}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !22 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !23 = !{null, !22}
// CHECK:STDOUT: !24 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !40 = !DIFile(filename: "create.carbon", directory: "")
// CHECK:STDOUT: !41 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !40, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !42 = !{null}
// CHECK:STDOUT: !43 = !DISubroutineType(types: !42)
// CHECK:STDOUT: !44 = distinct !DISubprogram(name: "Create", linkageName: "_CCreate.Create", scope: null, file: !40, line: 6, type: !43, spFlags: DISPFlagDefinition, unit: !41)
// CHECK:STDOUT: !45 = !DILocation(line: 7, column: 3, scope: !44)
// CHECK:STDOUT: !47 = !DILocation(line: 8, column: 3, scope: !44)
// CHECK:STDOUT: !48 = !DILocation(line: 8, column: 33, scope: !44)
// CHECK:STDOUT: !50 = !DILocation(line: 9, column: 3, scope: !44)
// CHECK:STDOUT: !51 = !DILocation(line: 9, column: 28, scope: !44)
// CHECK:STDOUT: !52 = !DILocation(line: 9, column: 37, scope: !44)
// CHECK:STDOUT: !54 = !DILocation(line: 12, column: 3, scope: !44)
// CHECK:STDOUT: !56 = !DILocation(line: 6, column: 1, scope: !44)
// CHECK:STDOUT: !57 = distinct !DISubprogram(name: "Use", linkageName: "_CUse.Create", scope: null, file: !40, line: 15, type: !24, spFlags: DISPFlagDefinition, unit: !41, retainedNodes: !62)
// CHECK:STDOUT: !58 = !DILocalVariable(arg: 1, scope: !57, type: !22)
// CHECK:STDOUT: !60 = !DILocation(line: 16, column: 3, scope: !57)
// CHECK:STDOUT: !61 = !DILocation(line: 15, column: 1, scope: !57)
// CHECK:STDOUT: !62 = !{!58}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'create_named_ctor.carbon'
// CHECK:STDOUT: source_filename = "create_named_ctor.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @"_CIntermediate.Classes.$vtable" = external unnamed_addr constant ptr
// CHECK:STDOUT: @"_CDerived.Classes.$vtable" = external unnamed_addr constant ptr
// CHECK:STDOUT: @empty_struct.ac8.loc7_12 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMakePartialBase.Main(ptr sret({}) %return) #0 !dbg !67 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %return, ptr align 1 @empty_struct.ac8.loc7_12, i64 0, i1 false), !dbg !70
// CHECK:STDOUT:   ret void, !dbg !70
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMakePartialIntermediate.Main(ptr sret({ ptr, {} }) %return) #0 !dbg !71 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc11_36.1.base = getelementptr inbounds nuw { ptr, {} }, ptr %return, i32 0, i32 1, !dbg !74
// CHECK:STDOUT:   call void @_CMakePartialBase.Main(ptr %.loc11_36.1.base), !dbg !75
// CHECK:STDOUT:   %.loc11_36.3.vptr = getelementptr inbounds nuw { ptr, {} }, ptr %return, i32 0, i32 0, !dbg !74
// CHECK:STDOUT:   store ptr poison, ptr %.loc11_36.3.vptr, align 8, !dbg !74
// CHECK:STDOUT:   ret void, !dbg !76
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMakeBase.Main(ptr sret({}) %return) #0 !dbg !77 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CMakePartialBase.Main(ptr %return), !dbg !80
// CHECK:STDOUT:   ret void, !dbg !81
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMakeIntermediate.Main(ptr sret({ ptr, {} }) %return) #0 !dbg !82 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CMakePartialIntermediate.Main(ptr %return), !dbg !85
// CHECK:STDOUT:   %.loc19_34.1.vptr = getelementptr inbounds nuw { ptr, {} }, ptr %return, i32 0, i32 0, !dbg !85
// CHECK:STDOUT:   store ptr @"_CIntermediate.Classes.$vtable", ptr %.loc19_34.1.vptr, align 8, !dbg !85
// CHECK:STDOUT:   ret void, !dbg !86
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CFn.Intermediate.Classes(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMakeDerivedFromPartial.Main(ptr sret({ { ptr, {} } }) %return) #0 !dbg !87 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc23_44.1.base = getelementptr inbounds nuw { { ptr, {} } }, ptr %return, i32 0, i32 0, !dbg !90
// CHECK:STDOUT:   call void @_CMakePartialIntermediate.Main(ptr %.loc23_44.1.base), !dbg !91
// CHECK:STDOUT:   %.loc23_44.5.base = getelementptr inbounds nuw { { ptr, {} } }, ptr %return, i32 0, i32 0, !dbg !90
// CHECK:STDOUT:   %.loc23_44.6.vptr = getelementptr inbounds nuw { ptr, {} }, ptr %.loc23_44.5.base, i32 0, i32 0, !dbg !90
// CHECK:STDOUT:   store ptr @"_CDerived.Classes.$vtable", ptr %.loc23_44.6.vptr, align 8, !dbg !90
// CHECK:STDOUT:   ret void, !dbg !92
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_CFn:thunk:Intermediate.Classes:Derived.Classes"(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMakeDerivedFromNonPartial.Main(ptr sret({ { ptr, {} } }) %return) #0 !dbg !93 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc27_37.1.base = getelementptr inbounds nuw { { ptr, {} } }, ptr %return, i32 0, i32 0, !dbg !96
// CHECK:STDOUT:   call void @_CMakeIntermediate.Main(ptr %.loc27_37.1.base), !dbg !97
// CHECK:STDOUT:   %.loc27_37.7.base = getelementptr inbounds nuw { { ptr, {} } }, ptr %return, i32 0, i32 0, !dbg !96
// CHECK:STDOUT:   %.loc27_37.8.vptr = getelementptr inbounds nuw { ptr, {} }, ptr %.loc27_37.7.base, i32 0, i32 0, !dbg !96
// CHECK:STDOUT:   store ptr @"_CDerived.Classes.$vtable", ptr %.loc27_37.8.vptr, align 8, !dbg !96
// CHECK:STDOUT:   ret void, !dbg !98
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!64}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !22 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !63 = !DIFile(filename: "create_named_ctor.carbon", directory: "")
// CHECK:STDOUT: !64 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !63, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !65 = !{!22}
// CHECK:STDOUT: !66 = !DISubroutineType(types: !65)
// CHECK:STDOUT: !67 = distinct !DISubprogram(name: "MakePartialBase", linkageName: "_CMakePartialBase.Main", scope: null, file: !63, line: 6, type: !66, spFlags: DISPFlagDefinition, unit: !64)
// CHECK:STDOUT: !70 = !DILocation(line: 7, column: 3, scope: !67)
// CHECK:STDOUT: !71 = distinct !DISubprogram(name: "MakePartialIntermediate", linkageName: "_CMakePartialIntermediate.Main", scope: null, file: !63, line: 10, type: !66, spFlags: DISPFlagDefinition, unit: !64)
// CHECK:STDOUT: !74 = !DILocation(line: 11, column: 10, scope: !71)
// CHECK:STDOUT: !75 = !DILocation(line: 11, column: 19, scope: !71)
// CHECK:STDOUT: !76 = !DILocation(line: 11, column: 3, scope: !71)
// CHECK:STDOUT: !77 = distinct !DISubprogram(name: "MakeBase", linkageName: "_CMakeBase.Main", scope: null, file: !63, line: 14, type: !66, spFlags: DISPFlagDefinition, unit: !64)
// CHECK:STDOUT: !80 = !DILocation(line: 15, column: 10, scope: !77)
// CHECK:STDOUT: !81 = !DILocation(line: 15, column: 3, scope: !77)
// CHECK:STDOUT: !82 = distinct !DISubprogram(name: "MakeIntermediate", linkageName: "_CMakeIntermediate.Main", scope: null, file: !63, line: 18, type: !66, spFlags: DISPFlagDefinition, unit: !64)
// CHECK:STDOUT: !85 = !DILocation(line: 19, column: 10, scope: !82)
// CHECK:STDOUT: !86 = !DILocation(line: 19, column: 3, scope: !82)
// CHECK:STDOUT: !87 = distinct !DISubprogram(name: "MakeDerivedFromPartial", linkageName: "_CMakeDerivedFromPartial.Main", scope: null, file: !63, line: 22, type: !66, spFlags: DISPFlagDefinition, unit: !64)
// CHECK:STDOUT: !90 = !DILocation(line: 23, column: 10, scope: !87)
// CHECK:STDOUT: !91 = !DILocation(line: 23, column: 19, scope: !87)
// CHECK:STDOUT: !92 = !DILocation(line: 23, column: 3, scope: !87)
// CHECK:STDOUT: !93 = distinct !DISubprogram(name: "MakeDerivedFromNonPartial", linkageName: "_CMakeDerivedFromNonPartial.Main", scope: null, file: !63, line: 26, type: !66, spFlags: DISPFlagDefinition, unit: !64)
// CHECK:STDOUT: !96 = !DILocation(line: 27, column: 10, scope: !93)
// CHECK:STDOUT: !97 = !DILocation(line: 27, column: 19, scope: !93)
// CHECK:STDOUT: !98 = !DILocation(line: 27, column: 3, scope: !93)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'member_init.carbon'
// CHECK:STDOUT: source_filename = "member_init.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @"_CBase.MemberInit.$vtable" = unnamed_addr constant [1 x i32] [i32 trunc (i64 sub (i64 ptrtoint (ptr @_CFn.Base.MemberInit to i64), i64 ptrtoint (ptr @"_CBase.MemberInit.$vtable" to i64)) to i32)]
// CHECK:STDOUT: @Base.val.loc13_3 = internal constant <{ ptr, i32 }> <{ ptr @"_CBase.MemberInit.$vtable", i32 3 }>
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CFn.Base.MemberInit(ptr %self) #0 !dbg !102 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !105
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CFn.MemberInit() #0 !dbg !107 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %i.var = alloca i32, align 4, !dbg !108
// CHECK:STDOUT:   %v.var = alloca <{ ptr, i32 }>, align 8, !dbg !110
// CHECK:STDOUT:   %_.var = alloca <{ ptr, i32 }>, align 8, !dbg !115
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %i.var), !dbg !108
// CHECK:STDOUT:   store i32 3, ptr %i.var, align 4, !dbg !108
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %v.var), !dbg !110
// CHECK:STDOUT:   %.loc11_24.2.vptr = getelementptr inbounds nuw <{ ptr, i32 }>, ptr %v.var, i32 0, i32 0, !dbg !112
// CHECK:STDOUT:   %.loc11_23 = load i32, ptr %i.var, align 4, !dbg !111
// CHECK:STDOUT:   %.loc11_24.4.m = getelementptr inbounds nuw <{ ptr, i32 }>, ptr %v.var, i32 0, i32 1, !dbg !112
// CHECK:STDOUT:   store i32 %.loc11_23, ptr %.loc11_24.4.m, align 4, !dbg !112
// CHECK:STDOUT:   store ptr @"_CBase.MemberInit.$vtable", ptr %.loc11_24.2.vptr, align 8, !dbg !112
// CHECK:STDOUT:   %.loc12_4.m = getelementptr inbounds nuw <{ ptr, i32 }>, ptr %v.var, i32 0, i32 1, !dbg !114
// CHECK:STDOUT:   store i32 5, ptr %.loc12_4.m, align 4, !dbg !114
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !115
// CHECK:STDOUT:   %.loc13_24.2.vptr = getelementptr inbounds nuw <{ ptr, i32 }>, ptr %_.var, i32 0, i32 0, !dbg !116
// CHECK:STDOUT:   %.loc13_24.5.m = getelementptr inbounds nuw <{ ptr, i32 }>, ptr %_.var, i32 0, i32 1, !dbg !116
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 %_.var, ptr align 8 @Base.val.loc13_3, i64 12, i1 false), !dbg !115
// CHECK:STDOUT:   ret void, !dbg !118
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!100}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !22 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !23 = !{null, !22}
// CHECK:STDOUT: !24 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !42 = !{null}
// CHECK:STDOUT: !43 = !DISubroutineType(types: !42)
// CHECK:STDOUT: !99 = !DIFile(filename: "member_init.carbon", directory: "")
// CHECK:STDOUT: !100 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !99, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !102 = distinct !DISubprogram(name: "Fn", linkageName: "_CFn.Base.MemberInit", scope: null, file: !99, line: 6, type: !24, spFlags: DISPFlagDefinition, unit: !100, retainedNodes: !106)
// CHECK:STDOUT: !103 = !DILocalVariable(arg: 1, scope: !102, type: !22)
// CHECK:STDOUT: !105 = !DILocation(line: 6, column: 3, scope: !102)
// CHECK:STDOUT: !106 = !{!103}
// CHECK:STDOUT: !107 = distinct !DISubprogram(name: "Fn", linkageName: "_CFn.MemberInit", scope: null, file: !99, line: 9, type: !43, spFlags: DISPFlagDefinition, unit: !100)
// CHECK:STDOUT: !108 = !DILocation(line: 10, column: 3, scope: !107)
// CHECK:STDOUT: !110 = !DILocation(line: 11, column: 3, scope: !107)
// CHECK:STDOUT: !111 = !DILocation(line: 11, column: 23, scope: !107)
// CHECK:STDOUT: !112 = !DILocation(line: 11, column: 17, scope: !107)
// CHECK:STDOUT: !114 = !DILocation(line: 12, column: 3, scope: !107)
// CHECK:STDOUT: !115 = !DILocation(line: 13, column: 3, scope: !107)
// CHECK:STDOUT: !116 = !DILocation(line: 13, column: 17, scope: !107)
// CHECK:STDOUT: !118 = !DILocation(line: 9, column: 1, scope: !107)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
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
// CHECK:STDOUT: define void @_CUse.Main() #0 !dbg !121 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca { { ptr } }, align 8, !dbg !122
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !122
// CHECK:STDOUT:   %.loc13_32.2.base = getelementptr inbounds nuw { { ptr } }, ptr %_.var, i32 0, i32 0, !dbg !123
// CHECK:STDOUT:   %.loc13_31.2.vptr = getelementptr inbounds nuw { ptr }, ptr %.loc13_32.2.base, i32 0, i32 0, !dbg !124
// CHECK:STDOUT:   %.loc13_32.6.base = getelementptr inbounds nuw { { ptr } }, ptr %_.var, i32 0, i32 0, !dbg !123
// CHECK:STDOUT:   %.loc13_32.7.vptr = getelementptr inbounds nuw { ptr }, ptr %.loc13_32.6.base, i32 0, i32 0, !dbg !123
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 %_.var, ptr align 8 @Derived.val.loc13_32.5, i64 8, i1 false), !dbg !123
// CHECK:STDOUT:   store ptr @"_CDerived.Main.$vtable", ptr %.loc13_32.7.vptr, align 8, !dbg !123
// CHECK:STDOUT:   ret void, !dbg !126
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!120}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !42 = !{null}
// CHECK:STDOUT: !43 = !DISubroutineType(types: !42)
// CHECK:STDOUT: !119 = !DIFile(filename: "member_brace_init.carbon", directory: "")
// CHECK:STDOUT: !120 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !119, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !121 = distinct !DISubprogram(name: "Use", linkageName: "_CUse.Main", scope: null, file: !119, line: 12, type: !43, spFlags: DISPFlagDefinition, unit: !120)
// CHECK:STDOUT: !122 = !DILocation(line: 13, column: 3, scope: !121)
// CHECK:STDOUT: !123 = !DILocation(line: 13, column: 21, scope: !121)
// CHECK:STDOUT: !124 = !DILocation(line: 13, column: 30, scope: !121)
// CHECK:STDOUT: !126 = !DILocation(line: 12, column: 1, scope: !121)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'call.carbon'
// CHECK:STDOUT: source_filename = "call.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @"_CBase.Main.$vtable" = unnamed_addr constant [1 x i32] [i32 trunc (i64 sub (i64 ptrtoint (ptr @_CF.Base.Main to i64), i64 ptrtoint (ptr @"_CBase.Main.$vtable" to i64)) to i32)]
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CF.Base.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CUse.Main(ptr %b) #0 !dbg !129 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Base.F.call.vtable = load ptr, ptr %b, align 8, !dbg !132
// CHECK:STDOUT:   %Base.F.call = call ptr @llvm.load.relative.i32(ptr %Base.F.call.vtable, i32 0), !dbg !132
// CHECK:STDOUT:   call void %Base.F.call(ptr %b), !dbg !132
// CHECK:STDOUT:   ret void, !dbg !133
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: read)
// CHECK:STDOUT: declare ptr @llvm.load.relative.i32(ptr, i32) #1
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: read) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!128}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !22 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !23 = !{null, !22}
// CHECK:STDOUT: !24 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !127 = !DIFile(filename: "call.carbon", directory: "")
// CHECK:STDOUT: !128 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !127, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !129 = distinct !DISubprogram(name: "Use", linkageName: "_CUse.Main", scope: null, file: !127, line: 8, type: !24, spFlags: DISPFlagDefinition, unit: !128, retainedNodes: !134)
// CHECK:STDOUT: !130 = !DILocalVariable(arg: 1, scope: !129, type: !22)
// CHECK:STDOUT: !132 = !DILocation(line: 9, column: 3, scope: !129)
// CHECK:STDOUT: !133 = !DILocation(line: 8, column: 1, scope: !129)
// CHECK:STDOUT: !134 = !{!130}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'call_impl.carbon'
// CHECK:STDOUT: source_filename = "call_impl.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @"_CBase.Main.$vtable" = unnamed_addr constant [1 x i32] [i32 trunc (i64 sub (i64 ptrtoint (ptr @_CF.Base.Main to i64), i64 ptrtoint (ptr @"_CBase.Main.$vtable" to i64)) to i32)]
// CHECK:STDOUT: @"_CDerived.Main.$vtable" = unnamed_addr constant [1 x i32] [i32 trunc (i64 sub (i64 ptrtoint (ptr @"_CF:thunk:Base.Main:Derived.Main" to i64), i64 ptrtoint (ptr @"_CDerived.Main.$vtable" to i64)) to i32)]
// CHECK:STDOUT: @Derived.val.loc14_32.5 = internal constant { { ptr } } poison
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CF.Base.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CF.Derived.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_CF:thunk:Base.Main:Derived.Main"(ptr %self) #0 !dbg !137 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Derived.Main(ptr %self), !dbg !140
// CHECK:STDOUT:   ret void, !dbg !140
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CUse.Main() #1 !dbg !142 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %v.var = alloca { { ptr } }, align 8, !dbg !143
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %v.var), !dbg !143
// CHECK:STDOUT:   %.loc14_32.2.base = getelementptr inbounds nuw { { ptr } }, ptr %v.var, i32 0, i32 0, !dbg !144
// CHECK:STDOUT:   %.loc14_31.2.vptr = getelementptr inbounds nuw { ptr }, ptr %.loc14_32.2.base, i32 0, i32 0, !dbg !145
// CHECK:STDOUT:   %.loc14_32.6.base = getelementptr inbounds nuw { { ptr } }, ptr %v.var, i32 0, i32 0, !dbg !144
// CHECK:STDOUT:   %.loc14_32.7.vptr = getelementptr inbounds nuw { ptr }, ptr %.loc14_32.6.base, i32 0, i32 0, !dbg !144
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 %v.var, ptr align 8 @Derived.val.loc14_32.5, i64 8, i1 false), !dbg !144
// CHECK:STDOUT:   store ptr @"_CDerived.Main.$vtable", ptr %.loc14_32.7.vptr, align 8, !dbg !144
// CHECK:STDOUT:   %.loc15_3.1.base = getelementptr inbounds nuw { { ptr } }, ptr %v.var, i32 0, i32 0, !dbg !147
// CHECK:STDOUT:   %Base.F.call.vtable = load ptr, ptr %.loc15_3.1.base, align 8, !dbg !147
// CHECK:STDOUT:   %Base.F.call = call ptr @llvm.load.relative.i32(ptr %Base.F.call.vtable, i32 0), !dbg !147
// CHECK:STDOUT:   call void %Base.F.call(ptr %.loc15_3.1.base), !dbg !147
// CHECK:STDOUT:   ret void, !dbg !148
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: read)
// CHECK:STDOUT: declare ptr @llvm.load.relative.i32(ptr, i32) #3
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { alwaysinline nounwind }
// CHECK:STDOUT: attributes #1 = { nounwind }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #3 = { nocallback nofree nosync nounwind willreturn memory(argmem: read) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!136}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !22 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !23 = !{null, !22}
// CHECK:STDOUT: !24 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !42 = !{null}
// CHECK:STDOUT: !43 = !DISubroutineType(types: !42)
// CHECK:STDOUT: !135 = !DIFile(filename: "call_impl.carbon", directory: "")
// CHECK:STDOUT: !136 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !135, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !137 = distinct !DISubprogram(name: "F", linkageName: "_CF:thunk:Base.Main:Derived.Main", scope: null, file: !135, line: 10, type: !24, spFlags: DISPFlagDefinition, unit: !136, retainedNodes: !141)
// CHECK:STDOUT: !138 = !DILocalVariable(arg: 1, scope: !137, type: !22)
// CHECK:STDOUT: !140 = !DILocation(line: 10, column: 3, scope: !137)
// CHECK:STDOUT: !141 = !{!138}
// CHECK:STDOUT: !142 = distinct !DISubprogram(name: "Use", linkageName: "_CUse.Main", scope: null, file: !135, line: 13, type: !43, spFlags: DISPFlagDefinition, unit: !136)
// CHECK:STDOUT: !143 = !DILocation(line: 14, column: 3, scope: !142)
// CHECK:STDOUT: !144 = !DILocation(line: 14, column: 21, scope: !142)
// CHECK:STDOUT: !145 = !DILocation(line: 14, column: 30, scope: !142)
// CHECK:STDOUT: !147 = !DILocation(line: 15, column: 3, scope: !142)
// CHECK:STDOUT: !148 = !DILocation(line: 13, column: 1, scope: !142)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'generic_noop.carbon'
// CHECK:STDOUT: source_filename = "generic_noop.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!150}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !149 = !DIFile(filename: "generic_noop.carbon", directory: "")
// CHECK:STDOUT: !150 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !149, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'generic_use.carbon'
// CHECK:STDOUT: source_filename = "generic_use.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @"_CBase.Main.$vtable.947fa7adaf50ddfd" = unnamed_addr constant [1 x i32] [i32 trunc (i64 sub (i64 ptrtoint (ptr @_CF.Base.Main.947fa7adaf50ddfd to i64), i64 ptrtoint (ptr @"_CBase.Main.$vtable.947fa7adaf50ddfd" to i64)) to i32)]
// CHECK:STDOUT: @"_CBase.Main.$vtable.1977607cd7948753" = unnamed_addr constant [1 x i32] [i32 trunc (i64 sub (i64 ptrtoint (ptr @_CF.Base.Main.1977607cd7948753 to i64), i64 ptrtoint (ptr @"_CBase.Main.$vtable.1977607cd7948753" to i64)) to i32)]
// CHECK:STDOUT: @Base.val.65a.loc16_3 = internal constant { ptr } { ptr @"_CBase.Main.$vtable.947fa7adaf50ddfd" }
// CHECK:STDOUT: @Base.val.b6e.loc17_3 = internal constant { ptr } { ptr @"_CBase.Main.$vtable.1977607cd7948753" }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !153 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var.loc16 = alloca { ptr }, align 8, !dbg !154
// CHECK:STDOUT:   %_.var.loc17 = alloca { ptr }, align 8, !dbg !157
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc16), !dbg !154
// CHECK:STDOUT:   %.loc16_22.2.vptr = getelementptr inbounds nuw { ptr }, ptr %_.var.loc16, i32 0, i32 0, !dbg !155
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 %_.var.loc16, ptr align 8 @Base.val.65a.loc16_3, i64 8, i1 false), !dbg !154
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc17), !dbg !157
// CHECK:STDOUT:   %.loc17_22.2.vptr = getelementptr inbounds nuw { ptr }, ptr %_.var.loc17, i32 0, i32 0, !dbg !158
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 %_.var.loc17, ptr align 8 @Base.val.b6e.loc17_3, i64 8, i1 false), !dbg !157
// CHECK:STDOUT:   ret void, !dbg !160
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Base.Main.947fa7adaf50ddfd(ptr %self) #0 !dbg !161 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca {}, align 1, !dbg !164
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !164
// CHECK:STDOUT:   ret void, !dbg !166
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Base.Main.1977607cd7948753(ptr %self) #0 !dbg !168 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca { {} }, align 1, !dbg !171
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !171
// CHECK:STDOUT:   ret void, !dbg !173
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!152}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !22 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !23 = !{null, !22}
// CHECK:STDOUT: !24 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !42 = !{null}
// CHECK:STDOUT: !43 = !DISubroutineType(types: !42)
// CHECK:STDOUT: !151 = !DIFile(filename: "generic_use.carbon", directory: "")
// CHECK:STDOUT: !152 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !151, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !153 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !151, line: 15, type: !43, spFlags: DISPFlagDefinition, unit: !152)
// CHECK:STDOUT: !154 = !DILocation(line: 16, column: 3, scope: !153)
// CHECK:STDOUT: !155 = !DILocation(line: 16, column: 21, scope: !153)
// CHECK:STDOUT: !157 = !DILocation(line: 17, column: 3, scope: !153)
// CHECK:STDOUT: !158 = !DILocation(line: 17, column: 21, scope: !153)
// CHECK:STDOUT: !160 = !DILocation(line: 15, column: 1, scope: !153)
// CHECK:STDOUT: !161 = distinct !DISubprogram(name: "F", linkageName: "_CF.Base.Main.947fa7adaf50ddfd", scope: null, file: !151, line: 5, type: !24, spFlags: DISPFlagDefinition, unit: !152, retainedNodes: !167)
// CHECK:STDOUT: !162 = !DILocalVariable(arg: 1, scope: !161, type: !22)
// CHECK:STDOUT: !164 = !DILocation(line: 6, column: 5, scope: !161)
// CHECK:STDOUT: !166 = !DILocation(line: 5, column: 3, scope: !161)
// CHECK:STDOUT: !167 = !{!162}
// CHECK:STDOUT: !168 = distinct !DISubprogram(name: "F", linkageName: "_CF.Base.Main.1977607cd7948753", scope: null, file: !151, line: 5, type: !24, spFlags: DISPFlagDefinition, unit: !152, retainedNodes: !174)
// CHECK:STDOUT: !169 = !DILocalVariable(arg: 1, scope: !168, type: !22)
// CHECK:STDOUT: !171 = !DILocation(line: 6, column: 5, scope: !168)
// CHECK:STDOUT: !173 = !DILocation(line: 5, column: 3, scope: !168)
// CHECK:STDOUT: !174 = !{!169}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'generic_base.carbon'
// CHECK:STDOUT: source_filename = "generic_base.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @"_CDerived.Main.$vtable" = unnamed_addr constant [1 x i32] [i32 trunc (i64 sub (i64 ptrtoint (ptr @_CF.Base.Main.6d6a4a2512463e59 to i64), i64 ptrtoint (ptr @"_CDerived.Main.$vtable" to i64)) to i32)]
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMake.Main() #0 !dbg !177 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca { { ptr } }, align 8, !dbg !178
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !178
// CHECK:STDOUT:   ret void, !dbg !180
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Base.Main.6d6a4a2512463e59(ptr %self) #0 !dbg !181 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !184
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!176}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !22 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !23 = !{null, !22}
// CHECK:STDOUT: !24 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !42 = !{null}
// CHECK:STDOUT: !43 = !DISubroutineType(types: !42)
// CHECK:STDOUT: !175 = !DIFile(filename: "generic_base.carbon", directory: "")
// CHECK:STDOUT: !176 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !175, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !177 = distinct !DISubprogram(name: "Make", linkageName: "_CMake.Main", scope: null, file: !175, line: 14, type: !43, spFlags: DISPFlagDefinition, unit: !176)
// CHECK:STDOUT: !178 = !DILocation(line: 15, column: 3, scope: !177)
// CHECK:STDOUT: !180 = !DILocation(line: 14, column: 1, scope: !177)
// CHECK:STDOUT: !181 = distinct !DISubprogram(name: "F", linkageName: "_CF.Base.Main.6d6a4a2512463e59", scope: null, file: !175, line: 5, type: !24, spFlags: DISPFlagDefinition, unit: !176, retainedNodes: !185)
// CHECK:STDOUT: !182 = !DILocalVariable(arg: 1, scope: !181, type: !22)
// CHECK:STDOUT: !184 = !DILocation(line: 5, column: 3, scope: !181)
// CHECK:STDOUT: !185 = !{!182}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'thunk.carbon'
// CHECK:STDOUT: source_filename = "thunk.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @"_CBase.Main.$vtable" = unnamed_addr constant [1 x i32] [i32 trunc (i64 sub (i64 ptrtoint (ptr @_CF.Base.Main to i64), i64 ptrtoint (ptr @"_CBase.Main.$vtable" to i64)) to i32)]
// CHECK:STDOUT: @"_CDerived.Main.$vtable" = unnamed_addr constant [1 x i32] [i32 trunc (i64 sub (i64 ptrtoint (ptr @"_CF:thunk:Base.Main:Derived.Main" to i64), i64 ptrtoint (ptr @"_CDerived.Main.$vtable" to i64)) to i32)]
// CHECK:STDOUT: @From.val = internal constant {} zeroinitializer
// CHECK:STDOUT: @From.val.loc18_14 = internal constant {} zeroinitializer
// CHECK:STDOUT: @Derived.val.loc23_31.5 = internal constant { { ptr } } poison
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_CConvert.From.Main:ImplicitAs.a3bb440f6426a7d6.Core"(ptr sret({}), ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CF.Base.Main(ptr sret({}), ptr, ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Derived.Main(ptr sret({}) %return, ptr %self, ptr %n) #0 !dbg !192 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %return, ptr align 1 @From.val.loc18_14, i64 0, i1 false), !dbg !199
// CHECK:STDOUT:   ret void, !dbg !199
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_CF:thunk:Base.Main:Derived.Main"(ptr sret({}) %return, ptr %self, ptr %n) #1 !dbg !201 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc17_52.2.temp = alloca {}, align 1, !dbg !205
// CHECK:STDOUT:   %.loc17_52.4.temp = alloca {}, align 1, !dbg !205
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc17_52.2.temp), !dbg !205
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc17_52.4.temp), !dbg !205
// CHECK:STDOUT:   call void @"_CConvert.From.Main:ImplicitAs.a3bb440f6426a7d6.Core"(ptr %.loc17_52.4.temp, ptr %n), !dbg !205
// CHECK:STDOUT:   call void @_CF.Derived.Main(ptr %.loc17_52.2.temp, ptr %self, ptr %.loc17_52.4.temp), !dbg !205
// CHECK:STDOUT:   call void @"_CConvert.From.Main:ImplicitAs.a3bb440f6426a7d6.Core"(ptr %return, ptr %.loc17_52.2.temp), !dbg !205
// CHECK:STDOUT:   ret void, !dbg !205
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CUse.Main() #0 !dbg !207 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %d.var = alloca { { ptr } }, align 8, !dbg !208
// CHECK:STDOUT:   %.loc25_10.1.temp = alloca {}, align 1, !dbg !216
// CHECK:STDOUT:   %.loc25_9.2.temp = alloca {}, align 1, !dbg !217
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %d.var), !dbg !208
// CHECK:STDOUT:   %.loc23_31.2.base = getelementptr inbounds nuw { { ptr } }, ptr %d.var, i32 0, i32 0, !dbg !209
// CHECK:STDOUT:   %.loc23_30.2.vptr = getelementptr inbounds nuw { ptr }, ptr %.loc23_31.2.base, i32 0, i32 0, !dbg !210
// CHECK:STDOUT:   %.loc23_31.6.base = getelementptr inbounds nuw { { ptr } }, ptr %d.var, i32 0, i32 0, !dbg !209
// CHECK:STDOUT:   %.loc23_31.7.vptr = getelementptr inbounds nuw { ptr }, ptr %.loc23_31.6.base, i32 0, i32 0, !dbg !209
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 %d.var, ptr align 8 @Derived.val.loc23_31.5, i64 8, i1 false), !dbg !209
// CHECK:STDOUT:   store ptr @"_CDerived.Main.$vtable", ptr %.loc23_31.7.vptr, align 8, !dbg !209
// CHECK:STDOUT:   %.loc24_18.2.base = getelementptr inbounds nuw { { ptr } }, ptr %d.var, i32 0, i32 0, !dbg !213
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc25_10.1.temp), !dbg !216
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc25_9.2.temp), !dbg !217
// CHECK:STDOUT:   %Base.F.call.vtable = load ptr, ptr %.loc25_10.1.temp, align 8, !dbg !216
// CHECK:STDOUT:   %Base.F.call = call ptr @llvm.load.relative.i32(ptr %Base.F.call.vtable, i32 0), !dbg !216
// CHECK:STDOUT:   call void %Base.F.call(ptr %.loc25_10.1.temp, ptr %.loc24_18.2.base, ptr @From.val.loc18_14), !dbg !216
// CHECK:STDOUT:   ret void, !dbg !218
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: read)
// CHECK:STDOUT: declare ptr @llvm.load.relative.i32(ptr, i32) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 4, 3, 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline nounwind }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #3 = { nocallback nofree nosync nounwind willreturn memory(argmem: read) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!187}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !22 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !42 = !{null}
// CHECK:STDOUT: !43 = !DISubroutineType(types: !42)
// CHECK:STDOUT: !186 = !DIFile(filename: "thunk.carbon", directory: "")
// CHECK:STDOUT: !187 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !186, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !190 = !{!22, !22, !22}
// CHECK:STDOUT: !191 = !DISubroutineType(types: !190)
// CHECK:STDOUT: !192 = distinct !DISubprogram(name: "F", linkageName: "_CF.Derived.Main", scope: null, file: !186, line: 17, type: !191, spFlags: DISPFlagDefinition, unit: !187, retainedNodes: !200)
// CHECK:STDOUT: !193 = !DILocalVariable(arg: 1, scope: !192, type: !22)
// CHECK:STDOUT: !194 = !DILocalVariable(arg: 2, scope: !192, type: !22)
// CHECK:STDOUT: !199 = !DILocation(line: 18, column: 5, scope: !192)
// CHECK:STDOUT: !200 = !{!193, !194}
// CHECK:STDOUT: !201 = distinct !DISubprogram(name: "F", linkageName: "_CF:thunk:Base.Main:Derived.Main", scope: null, file: !186, line: 17, type: !191, spFlags: DISPFlagDefinition, unit: !187, retainedNodes: !206)
// CHECK:STDOUT: !202 = !DILocalVariable(arg: 1, scope: !201, type: !22)
// CHECK:STDOUT: !203 = !DILocalVariable(arg: 2, scope: !201, type: !22)
// CHECK:STDOUT: !205 = !DILocation(line: 17, column: 3, scope: !201)
// CHECK:STDOUT: !206 = !{!202, !203}
// CHECK:STDOUT: !207 = distinct !DISubprogram(name: "Use", linkageName: "_CUse.Main", scope: null, file: !186, line: 22, type: !43, spFlags: DISPFlagDefinition, unit: !187)
// CHECK:STDOUT: !208 = !DILocation(line: 23, column: 3, scope: !207)
// CHECK:STDOUT: !209 = !DILocation(line: 23, column: 20, scope: !207)
// CHECK:STDOUT: !210 = !DILocation(line: 23, column: 29, scope: !207)
// CHECK:STDOUT: !213 = !DILocation(line: 24, column: 18, scope: !207)
// CHECK:STDOUT: !216 = !DILocation(line: 25, column: 3, scope: !207)
// CHECK:STDOUT: !217 = !DILocation(line: 25, column: 8, scope: !207)
// CHECK:STDOUT: !218 = !DILocation(line: 22, column: 1, scope: !207)
// CHECK:STDOUT:
