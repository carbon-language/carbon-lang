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
// CHECK:STDOUT: define void @_CFn.Intermediate.Classes(ptr %self) #0 !dbg !23 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !26
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CFn.Derived.Classes(ptr %self) #0 !dbg !28 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !31
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_CFn:thunk:Intermediate.Classes:Derived.Classes"(ptr %self) #1 !dbg !33 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CFn.Derived.Classes(ptr %self), !dbg !36
// CHECK:STDOUT:   ret void, !dbg !36
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!19}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !18 = !DIFile(filename: "classes.carbon", directory: "")
// CHECK:STDOUT: !19 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !18, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !20 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !21 = !{null, !20}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !23 = distinct !DISubprogram(name: "Fn", linkageName: "_CFn.Intermediate.Classes", scope: null, file: !18, line: 9, type: !22, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !27)
// CHECK:STDOUT: !24 = !DILocalVariable(arg: 1, scope: !23, type: !20)
// CHECK:STDOUT: !26 = !DILocation(line: 9, column: 3, scope: !23)
// CHECK:STDOUT: !27 = !{!24}
// CHECK:STDOUT: !28 = distinct !DISubprogram(name: "Fn", linkageName: "_CFn.Derived.Classes", scope: null, file: !18, line: 14, type: !22, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !32)
// CHECK:STDOUT: !29 = !DILocalVariable(arg: 1, scope: !28, type: !20)
// CHECK:STDOUT: !31 = !DILocation(line: 14, column: 3, scope: !28)
// CHECK:STDOUT: !32 = !{!29}
// CHECK:STDOUT: !33 = distinct !DISubprogram(name: "Fn", linkageName: "_CFn:thunk:Intermediate.Classes:Derived.Classes", scope: null, file: !18, line: 14, type: !22, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !37)
// CHECK:STDOUT: !34 = !DILocalVariable(arg: 1, scope: !33, type: !20)
// CHECK:STDOUT: !36 = !DILocation(line: 14, column: 3, scope: !33)
// CHECK:STDOUT: !37 = !{!34}
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
// CHECK:STDOUT: define void @_CCreate.Create() #0 !dbg !42 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var.loc7 = alloca {}, align 1, !dbg !43
// CHECK:STDOUT:   %_.var.loc8 = alloca { ptr, {} }, align 8, !dbg !45
// CHECK:STDOUT:   %_.var.loc9 = alloca { { ptr, {} } }, align 8, !dbg !48
// CHECK:STDOUT:   %_.var.loc12 = alloca { { ptr, {} } }, align 8, !dbg !52
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc7), !dbg !43
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %_.var.loc7, ptr align 1 @Base.val.loc7_3, i64 0, i1 false), !dbg !43
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc8), !dbg !45
// CHECK:STDOUT:   %.loc8_44.2.vptr = getelementptr inbounds nuw { ptr, {} }, ptr %_.var.loc8, i32 0, i32 0, !dbg !46
// CHECK:STDOUT:   %.loc8_44.4.base = getelementptr inbounds nuw { ptr, {} }, ptr %_.var.loc8, i32 0, i32 1, !dbg !46
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 %_.var.loc8, ptr align 8 @Intermediate.val.54b.loc8_3, i64 8, i1 false), !dbg !45
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc9), !dbg !48
// CHECK:STDOUT:   %.loc9_49.2.base = getelementptr inbounds nuw { { ptr, {} } }, ptr %_.var.loc9, i32 0, i32 0, !dbg !49
// CHECK:STDOUT:   %.loc9_48.2.vptr = getelementptr inbounds nuw { ptr, {} }, ptr %.loc9_49.2.base, i32 0, i32 0, !dbg !50
// CHECK:STDOUT:   %.loc9_48.4.base = getelementptr inbounds nuw { ptr, {} }, ptr %.loc9_49.2.base, i32 0, i32 1, !dbg !50
// CHECK:STDOUT:   %.loc9_49.6.base = getelementptr inbounds nuw { { ptr, {} } }, ptr %_.var.loc9, i32 0, i32 0, !dbg !49
// CHECK:STDOUT:   %.loc9_49.7.vptr = getelementptr inbounds nuw { ptr, {} }, ptr %.loc9_49.6.base, i32 0, i32 0, !dbg !49
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 %_.var.loc9, ptr align 8 @Derived.val.loc9_49.5, i64 8, i1 false), !dbg !49
// CHECK:STDOUT:   store ptr @"_CDerived.Classes.$vtable", ptr %.loc9_49.7.vptr, align 8, !dbg !49
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc12), !dbg !52
// CHECK:STDOUT:   call void @"_COp.90ba497113bea86f:core.Destroy.Core"(ptr %_.var.loc12), !dbg !52
// CHECK:STDOUT:   call void @"_COp.90ba497113bea86f:core.Destroy.Core"(ptr %_.var.loc9), !dbg !48
// CHECK:STDOUT:   call void @"_COp.21751b0492a59c62:core.Destroy.Core"(ptr %_.var.loc8), !dbg !45
// CHECK:STDOUT:   call void @"_COp.bbc7087405c1f238:core.Destroy.Core"(ptr %_.var.loc7), !dbg !43
// CHECK:STDOUT:   ret void, !dbg !54
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CFn.Intermediate.Classes(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_CFn:thunk:Intermediate.Classes:Derived.Classes"(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.bbc7087405c1f238:core.Destroy.Core"(ptr %self) #0 !dbg !55 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !57
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.f2ac7e7561e4076c:core.Destroy.Core"(ptr %self) #0 !dbg !59 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !61
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.21751b0492a59c62:core.Destroy.Core"(ptr %self) #0 !dbg !63 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !65
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.558f48c06e972c2d:core.Destroy.Core"(ptr %self) #0 !dbg !67 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !69
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.90ba497113bea86f:core.Destroy.Core"(ptr %self) #0 !dbg !71 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !73
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CUse.Create(ptr %v) #0 !dbg !75 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Intermediate.Fn.call.vtable = load ptr, ptr %v, align 8, !dbg !78
// CHECK:STDOUT:   %Intermediate.Fn.call = call ptr @llvm.load.relative.i32(ptr %Intermediate.Fn.call.vtable, i32 0), !dbg !78
// CHECK:STDOUT:   call void %Intermediate.Fn.call(ptr %v), !dbg !78
// CHECK:STDOUT:   ret void, !dbg !79
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
// CHECK:STDOUT: uselistorder ptr @"_COp.90ba497113bea86f:core.Destroy.Core", { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: read) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!39}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !21 = !{null, !20}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !38 = !DIFile(filename: "create.carbon", directory: "")
// CHECK:STDOUT: !39 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !38, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !40 = !{null}
// CHECK:STDOUT: !41 = !DISubroutineType(types: !40)
// CHECK:STDOUT: !42 = distinct !DISubprogram(name: "Create", linkageName: "_CCreate.Create", scope: null, file: !38, line: 6, type: !41, spFlags: DISPFlagDefinition, unit: !39)
// CHECK:STDOUT: !43 = !DILocation(line: 7, column: 3, scope: !42)
// CHECK:STDOUT: !45 = !DILocation(line: 8, column: 3, scope: !42)
// CHECK:STDOUT: !46 = !DILocation(line: 8, column: 33, scope: !42)
// CHECK:STDOUT: !48 = !DILocation(line: 9, column: 3, scope: !42)
// CHECK:STDOUT: !49 = !DILocation(line: 9, column: 28, scope: !42)
// CHECK:STDOUT: !50 = !DILocation(line: 9, column: 37, scope: !42)
// CHECK:STDOUT: !52 = !DILocation(line: 12, column: 3, scope: !42)
// CHECK:STDOUT: !54 = !DILocation(line: 6, column: 1, scope: !42)
// CHECK:STDOUT: !55 = distinct !DISubprogram(name: "Op", linkageName: "_COp.bbc7087405c1f238:core.Destroy.Core", scope: null, file: !38, line: 12, type: !22, spFlags: DISPFlagDefinition, unit: !39, retainedNodes: !58)
// CHECK:STDOUT: !56 = !DILocalVariable(arg: 1, scope: !55, type: !20)
// CHECK:STDOUT: !57 = !DILocation(line: 12, column: 3, scope: !55)
// CHECK:STDOUT: !58 = !{!56}
// CHECK:STDOUT: !59 = distinct !DISubprogram(name: "Op", linkageName: "_COp.f2ac7e7561e4076c:core.Destroy.Core", scope: null, file: !38, line: 12, type: !22, spFlags: DISPFlagDefinition, unit: !39, retainedNodes: !62)
// CHECK:STDOUT: !60 = !DILocalVariable(arg: 1, scope: !59, type: !20)
// CHECK:STDOUT: !61 = !DILocation(line: 12, column: 3, scope: !59)
// CHECK:STDOUT: !62 = !{!60}
// CHECK:STDOUT: !63 = distinct !DISubprogram(name: "Op", linkageName: "_COp.21751b0492a59c62:core.Destroy.Core", scope: null, file: !38, line: 12, type: !22, spFlags: DISPFlagDefinition, unit: !39, retainedNodes: !66)
// CHECK:STDOUT: !64 = !DILocalVariable(arg: 1, scope: !63, type: !20)
// CHECK:STDOUT: !65 = !DILocation(line: 12, column: 3, scope: !63)
// CHECK:STDOUT: !66 = !{!64}
// CHECK:STDOUT: !67 = distinct !DISubprogram(name: "Op", linkageName: "_COp.558f48c06e972c2d:core.Destroy.Core", scope: null, file: !38, line: 12, type: !22, spFlags: DISPFlagDefinition, unit: !39, retainedNodes: !70)
// CHECK:STDOUT: !68 = !DILocalVariable(arg: 1, scope: !67, type: !20)
// CHECK:STDOUT: !69 = !DILocation(line: 12, column: 3, scope: !67)
// CHECK:STDOUT: !70 = !{!68}
// CHECK:STDOUT: !71 = distinct !DISubprogram(name: "Op", linkageName: "_COp.90ba497113bea86f:core.Destroy.Core", scope: null, file: !38, line: 12, type: !22, spFlags: DISPFlagDefinition, unit: !39, retainedNodes: !74)
// CHECK:STDOUT: !72 = !DILocalVariable(arg: 1, scope: !71, type: !20)
// CHECK:STDOUT: !73 = !DILocation(line: 12, column: 3, scope: !71)
// CHECK:STDOUT: !74 = !{!72}
// CHECK:STDOUT: !75 = distinct !DISubprogram(name: "Use", linkageName: "_CUse.Create", scope: null, file: !38, line: 15, type: !22, spFlags: DISPFlagDefinition, unit: !39, retainedNodes: !80)
// CHECK:STDOUT: !76 = !DILocalVariable(arg: 1, scope: !75, type: !20)
// CHECK:STDOUT: !78 = !DILocation(line: 16, column: 3, scope: !75)
// CHECK:STDOUT: !79 = !DILocation(line: 15, column: 1, scope: !75)
// CHECK:STDOUT: !80 = !{!76}
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
// CHECK:STDOUT: define void @_CMakePartialBase.Main(ptr sret({}) %return) #0 !dbg !85 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %return, ptr align 1 @empty_struct.ac8.loc7_12, i64 0, i1 false), !dbg !88
// CHECK:STDOUT:   ret void, !dbg !88
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMakePartialIntermediate.Main(ptr sret({ ptr, {} }) %return) #0 !dbg !89 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc11_36.1.base = getelementptr inbounds nuw { ptr, {} }, ptr %return, i32 0, i32 1, !dbg !92
// CHECK:STDOUT:   call void @_CMakePartialBase.Main(ptr %.loc11_36.1.base), !dbg !93
// CHECK:STDOUT:   %.loc11_36.3.vptr = getelementptr inbounds nuw { ptr, {} }, ptr %return, i32 0, i32 0, !dbg !92
// CHECK:STDOUT:   store ptr poison, ptr %.loc11_36.3.vptr, align 8, !dbg !92
// CHECK:STDOUT:   ret void, !dbg !94
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMakeBase.Main(ptr sret({}) %return) #0 !dbg !95 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CMakePartialBase.Main(ptr %return), !dbg !98
// CHECK:STDOUT:   ret void, !dbg !99
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMakeIntermediate.Main(ptr sret({ ptr, {} }) %return) #0 !dbg !100 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CMakePartialIntermediate.Main(ptr %return), !dbg !103
// CHECK:STDOUT:   %.loc19_34.1.vptr = getelementptr inbounds nuw { ptr, {} }, ptr %return, i32 0, i32 0, !dbg !103
// CHECK:STDOUT:   store ptr @"_CIntermediate.Classes.$vtable", ptr %.loc19_34.1.vptr, align 8, !dbg !103
// CHECK:STDOUT:   ret void, !dbg !104
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CFn.Intermediate.Classes(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMakeDerivedFromPartial.Main(ptr sret({ { ptr, {} } }) %return) #0 !dbg !105 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc23_44.1.base = getelementptr inbounds nuw { { ptr, {} } }, ptr %return, i32 0, i32 0, !dbg !108
// CHECK:STDOUT:   call void @_CMakePartialIntermediate.Main(ptr %.loc23_44.1.base), !dbg !109
// CHECK:STDOUT:   %.loc23_44.5.base = getelementptr inbounds nuw { { ptr, {} } }, ptr %return, i32 0, i32 0, !dbg !108
// CHECK:STDOUT:   %.loc23_44.6.vptr = getelementptr inbounds nuw { ptr, {} }, ptr %.loc23_44.5.base, i32 0, i32 0, !dbg !108
// CHECK:STDOUT:   store ptr @"_CDerived.Classes.$vtable", ptr %.loc23_44.6.vptr, align 8, !dbg !108
// CHECK:STDOUT:   ret void, !dbg !110
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_CFn:thunk:Intermediate.Classes:Derived.Classes"(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMakeDerivedFromNonPartial.Main(ptr sret({ { ptr, {} } }) %return) #0 !dbg !111 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc27_37.1.base = getelementptr inbounds nuw { { ptr, {} } }, ptr %return, i32 0, i32 0, !dbg !114
// CHECK:STDOUT:   call void @_CMakeIntermediate.Main(ptr %.loc27_37.1.base), !dbg !115
// CHECK:STDOUT:   %.loc27_37.7.base = getelementptr inbounds nuw { { ptr, {} } }, ptr %return, i32 0, i32 0, !dbg !114
// CHECK:STDOUT:   %.loc27_37.8.vptr = getelementptr inbounds nuw { ptr, {} }, ptr %.loc27_37.7.base, i32 0, i32 0, !dbg !114
// CHECK:STDOUT:   store ptr @"_CDerived.Classes.$vtable", ptr %.loc27_37.8.vptr, align 8, !dbg !114
// CHECK:STDOUT:   ret void, !dbg !116
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!82}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !81 = !DIFile(filename: "create_named_ctor.carbon", directory: "")
// CHECK:STDOUT: !82 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !81, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !83 = !{!20}
// CHECK:STDOUT: !84 = !DISubroutineType(types: !83)
// CHECK:STDOUT: !85 = distinct !DISubprogram(name: "MakePartialBase", linkageName: "_CMakePartialBase.Main", scope: null, file: !81, line: 6, type: !84, spFlags: DISPFlagDefinition, unit: !82)
// CHECK:STDOUT: !88 = !DILocation(line: 7, column: 3, scope: !85)
// CHECK:STDOUT: !89 = distinct !DISubprogram(name: "MakePartialIntermediate", linkageName: "_CMakePartialIntermediate.Main", scope: null, file: !81, line: 10, type: !84, spFlags: DISPFlagDefinition, unit: !82)
// CHECK:STDOUT: !92 = !DILocation(line: 11, column: 10, scope: !89)
// CHECK:STDOUT: !93 = !DILocation(line: 11, column: 19, scope: !89)
// CHECK:STDOUT: !94 = !DILocation(line: 11, column: 3, scope: !89)
// CHECK:STDOUT: !95 = distinct !DISubprogram(name: "MakeBase", linkageName: "_CMakeBase.Main", scope: null, file: !81, line: 14, type: !84, spFlags: DISPFlagDefinition, unit: !82)
// CHECK:STDOUT: !98 = !DILocation(line: 15, column: 10, scope: !95)
// CHECK:STDOUT: !99 = !DILocation(line: 15, column: 3, scope: !95)
// CHECK:STDOUT: !100 = distinct !DISubprogram(name: "MakeIntermediate", linkageName: "_CMakeIntermediate.Main", scope: null, file: !81, line: 18, type: !84, spFlags: DISPFlagDefinition, unit: !82)
// CHECK:STDOUT: !103 = !DILocation(line: 19, column: 10, scope: !100)
// CHECK:STDOUT: !104 = !DILocation(line: 19, column: 3, scope: !100)
// CHECK:STDOUT: !105 = distinct !DISubprogram(name: "MakeDerivedFromPartial", linkageName: "_CMakeDerivedFromPartial.Main", scope: null, file: !81, line: 22, type: !84, spFlags: DISPFlagDefinition, unit: !82)
// CHECK:STDOUT: !108 = !DILocation(line: 23, column: 10, scope: !105)
// CHECK:STDOUT: !109 = !DILocation(line: 23, column: 19, scope: !105)
// CHECK:STDOUT: !110 = !DILocation(line: 23, column: 3, scope: !105)
// CHECK:STDOUT: !111 = distinct !DISubprogram(name: "MakeDerivedFromNonPartial", linkageName: "_CMakeDerivedFromNonPartial.Main", scope: null, file: !81, line: 26, type: !84, spFlags: DISPFlagDefinition, unit: !82)
// CHECK:STDOUT: !114 = !DILocation(line: 27, column: 10, scope: !111)
// CHECK:STDOUT: !115 = !DILocation(line: 27, column: 19, scope: !111)
// CHECK:STDOUT: !116 = !DILocation(line: 27, column: 3, scope: !111)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'member_init.carbon'
// CHECK:STDOUT: source_filename = "member_init.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @"_CBase.MemberInit.$vtable" = unnamed_addr constant [1 x i32] [i32 trunc (i64 sub (i64 ptrtoint (ptr @_CFn.Base.MemberInit to i64), i64 ptrtoint (ptr @"_CBase.MemberInit.$vtable" to i64)) to i32)]
// CHECK:STDOUT: @Base.val.loc13_3 = internal constant <{ ptr, i32 }> <{ ptr @"_CBase.MemberInit.$vtable", i32 3 }>
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CFn.Base.MemberInit(ptr %self) #0 !dbg !122 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !125
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CFn.MemberInit() #0 !dbg !127 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %i.var = alloca i32, align 4, !dbg !128
// CHECK:STDOUT:   %v.var = alloca <{ ptr, i32 }>, align 8, !dbg !130
// CHECK:STDOUT:   %_.var = alloca <{ ptr, i32 }>, align 8, !dbg !135
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %i.var), !dbg !128
// CHECK:STDOUT:   store i32 3, ptr %i.var, align 4, !dbg !128
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %v.var), !dbg !130
// CHECK:STDOUT:   %.loc11_24.2.vptr = getelementptr inbounds nuw <{ ptr, i32 }>, ptr %v.var, i32 0, i32 0, !dbg !132
// CHECK:STDOUT:   %.loc11_23 = load i32, ptr %i.var, align 4, !dbg !131
// CHECK:STDOUT:   %.loc11_24.4.m = getelementptr inbounds nuw <{ ptr, i32 }>, ptr %v.var, i32 0, i32 1, !dbg !132
// CHECK:STDOUT:   store i32 %.loc11_23, ptr %.loc11_24.4.m, align 4, !dbg !132
// CHECK:STDOUT:   store ptr @"_CBase.MemberInit.$vtable", ptr %.loc11_24.2.vptr, align 8, !dbg !132
// CHECK:STDOUT:   %.loc12_4.m = getelementptr inbounds nuw <{ ptr, i32 }>, ptr %v.var, i32 0, i32 1, !dbg !134
// CHECK:STDOUT:   store i32 5, ptr %.loc12_4.m, align 4, !dbg !134
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !135
// CHECK:STDOUT:   %.loc13_24.2.vptr = getelementptr inbounds nuw <{ ptr, i32 }>, ptr %_.var, i32 0, i32 0, !dbg !136
// CHECK:STDOUT:   %.loc13_24.5.m = getelementptr inbounds nuw <{ ptr, i32 }>, ptr %_.var, i32 0, i32 1, !dbg !136
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 %_.var, ptr align 8 @Base.val.loc13_3, i64 12, i1 false), !dbg !135
// CHECK:STDOUT:   call void @"_COp.a6a094e0f6bc9348:core.Destroy.Core"(ptr %_.var), !dbg !135
// CHECK:STDOUT:   call void @"_COp.a6a094e0f6bc9348:core.Destroy.Core"(ptr %v.var), !dbg !130
// CHECK:STDOUT:   call void @"_COp.70db2d92ffddbef7:core.Destroy.Core"(ptr %i.var), !dbg !128
// CHECK:STDOUT:   ret void, !dbg !138
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.70db2d92ffddbef7:core.Destroy.Core"(ptr %self) #0 !dbg !139 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !141
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.bef9da9070867db2:core.Destroy.Core"(ptr %self) #0 !dbg !143 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !145
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.a6a094e0f6bc9348:core.Destroy.Core"(ptr %self) #0 !dbg !147 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !149
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @"_COp.a6a094e0f6bc9348:core.Destroy.Core", { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!118}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !21 = !{null, !20}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !40 = !{null}
// CHECK:STDOUT: !41 = !DISubroutineType(types: !40)
// CHECK:STDOUT: !117 = !DIFile(filename: "member_init.carbon", directory: "")
// CHECK:STDOUT: !118 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !117, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !119 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !120 = !{null, !119}
// CHECK:STDOUT: !121 = !DISubroutineType(types: !120)
// CHECK:STDOUT: !122 = distinct !DISubprogram(name: "Fn", linkageName: "_CFn.Base.MemberInit", scope: null, file: !117, line: 6, type: !22, spFlags: DISPFlagDefinition, unit: !118, retainedNodes: !126)
// CHECK:STDOUT: !123 = !DILocalVariable(arg: 1, scope: !122, type: !20)
// CHECK:STDOUT: !125 = !DILocation(line: 6, column: 3, scope: !122)
// CHECK:STDOUT: !126 = !{!123}
// CHECK:STDOUT: !127 = distinct !DISubprogram(name: "Fn", linkageName: "_CFn.MemberInit", scope: null, file: !117, line: 9, type: !41, spFlags: DISPFlagDefinition, unit: !118)
// CHECK:STDOUT: !128 = !DILocation(line: 10, column: 3, scope: !127)
// CHECK:STDOUT: !130 = !DILocation(line: 11, column: 3, scope: !127)
// CHECK:STDOUT: !131 = !DILocation(line: 11, column: 23, scope: !127)
// CHECK:STDOUT: !132 = !DILocation(line: 11, column: 17, scope: !127)
// CHECK:STDOUT: !134 = !DILocation(line: 12, column: 3, scope: !127)
// CHECK:STDOUT: !135 = !DILocation(line: 13, column: 3, scope: !127)
// CHECK:STDOUT: !136 = !DILocation(line: 13, column: 17, scope: !127)
// CHECK:STDOUT: !138 = !DILocation(line: 9, column: 1, scope: !127)
// CHECK:STDOUT: !139 = distinct !DISubprogram(name: "Op", linkageName: "_COp.70db2d92ffddbef7:core.Destroy.Core", scope: null, file: !117, line: 13, type: !121, spFlags: DISPFlagDefinition, unit: !118, retainedNodes: !142)
// CHECK:STDOUT: !140 = !DILocalVariable(arg: 1, scope: !139, type: !119)
// CHECK:STDOUT: !141 = !DILocation(line: 13, column: 3, scope: !139)
// CHECK:STDOUT: !142 = !{!140}
// CHECK:STDOUT: !143 = distinct !DISubprogram(name: "Op", linkageName: "_COp.bef9da9070867db2:core.Destroy.Core", scope: null, file: !117, line: 13, type: !22, spFlags: DISPFlagDefinition, unit: !118, retainedNodes: !146)
// CHECK:STDOUT: !144 = !DILocalVariable(arg: 1, scope: !143, type: !20)
// CHECK:STDOUT: !145 = !DILocation(line: 13, column: 3, scope: !143)
// CHECK:STDOUT: !146 = !{!144}
// CHECK:STDOUT: !147 = distinct !DISubprogram(name: "Op", linkageName: "_COp.a6a094e0f6bc9348:core.Destroy.Core", scope: null, file: !117, line: 13, type: !22, spFlags: DISPFlagDefinition, unit: !118, retainedNodes: !150)
// CHECK:STDOUT: !148 = !DILocalVariable(arg: 1, scope: !147, type: !20)
// CHECK:STDOUT: !149 = !DILocation(line: 13, column: 3, scope: !147)
// CHECK:STDOUT: !150 = !{!148}
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
// CHECK:STDOUT: define void @_CUse.Main() #0 !dbg !153 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca { { ptr } }, align 8, !dbg !154
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !154
// CHECK:STDOUT:   %.loc13_32.2.base = getelementptr inbounds nuw { { ptr } }, ptr %_.var, i32 0, i32 0, !dbg !155
// CHECK:STDOUT:   %.loc13_31.2.vptr = getelementptr inbounds nuw { ptr }, ptr %.loc13_32.2.base, i32 0, i32 0, !dbg !156
// CHECK:STDOUT:   %.loc13_32.6.base = getelementptr inbounds nuw { { ptr } }, ptr %_.var, i32 0, i32 0, !dbg !155
// CHECK:STDOUT:   %.loc13_32.7.vptr = getelementptr inbounds nuw { ptr }, ptr %.loc13_32.6.base, i32 0, i32 0, !dbg !155
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 %_.var, ptr align 8 @Derived.val.loc13_32.5, i64 8, i1 false), !dbg !155
// CHECK:STDOUT:   store ptr @"_CDerived.Main.$vtable", ptr %.loc13_32.7.vptr, align 8, !dbg !155
// CHECK:STDOUT:   call void @"_COp.d8aa8f8dee0ff18e:core.Destroy.Core"(ptr %_.var), !dbg !154
// CHECK:STDOUT:   ret void, !dbg !158
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.8ea19ac2d92d0842:core.Destroy.Core"(ptr %self) #0 !dbg !159 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !161
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.26b62f84bf985626:core.Destroy.Core"(ptr %self) #0 !dbg !163 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !165
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.66f0cc6b13851099:core.Destroy.Core"(ptr %self) #0 !dbg !167 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !169
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.d8aa8f8dee0ff18e:core.Destroy.Core"(ptr %self) #0 !dbg !171 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !173
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!152}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !21 = !{null, !20}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !40 = !{null}
// CHECK:STDOUT: !41 = !DISubroutineType(types: !40)
// CHECK:STDOUT: !151 = !DIFile(filename: "member_brace_init.carbon", directory: "")
// CHECK:STDOUT: !152 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !151, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !153 = distinct !DISubprogram(name: "Use", linkageName: "_CUse.Main", scope: null, file: !151, line: 12, type: !41, spFlags: DISPFlagDefinition, unit: !152)
// CHECK:STDOUT: !154 = !DILocation(line: 13, column: 3, scope: !153)
// CHECK:STDOUT: !155 = !DILocation(line: 13, column: 21, scope: !153)
// CHECK:STDOUT: !156 = !DILocation(line: 13, column: 30, scope: !153)
// CHECK:STDOUT: !158 = !DILocation(line: 12, column: 1, scope: !153)
// CHECK:STDOUT: !159 = distinct !DISubprogram(name: "Op", linkageName: "_COp.8ea19ac2d92d0842:core.Destroy.Core", scope: null, file: !151, line: 13, type: !22, spFlags: DISPFlagDefinition, unit: !152, retainedNodes: !162)
// CHECK:STDOUT: !160 = !DILocalVariable(arg: 1, scope: !159, type: !20)
// CHECK:STDOUT: !161 = !DILocation(line: 13, column: 3, scope: !159)
// CHECK:STDOUT: !162 = !{!160}
// CHECK:STDOUT: !163 = distinct !DISubprogram(name: "Op", linkageName: "_COp.26b62f84bf985626:core.Destroy.Core", scope: null, file: !151, line: 13, type: !22, spFlags: DISPFlagDefinition, unit: !152, retainedNodes: !166)
// CHECK:STDOUT: !164 = !DILocalVariable(arg: 1, scope: !163, type: !20)
// CHECK:STDOUT: !165 = !DILocation(line: 13, column: 3, scope: !163)
// CHECK:STDOUT: !166 = !{!164}
// CHECK:STDOUT: !167 = distinct !DISubprogram(name: "Op", linkageName: "_COp.66f0cc6b13851099:core.Destroy.Core", scope: null, file: !151, line: 13, type: !22, spFlags: DISPFlagDefinition, unit: !152, retainedNodes: !170)
// CHECK:STDOUT: !168 = !DILocalVariable(arg: 1, scope: !167, type: !20)
// CHECK:STDOUT: !169 = !DILocation(line: 13, column: 3, scope: !167)
// CHECK:STDOUT: !170 = !{!168}
// CHECK:STDOUT: !171 = distinct !DISubprogram(name: "Op", linkageName: "_COp.d8aa8f8dee0ff18e:core.Destroy.Core", scope: null, file: !151, line: 13, type: !22, spFlags: DISPFlagDefinition, unit: !152, retainedNodes: !174)
// CHECK:STDOUT: !172 = !DILocalVariable(arg: 1, scope: !171, type: !20)
// CHECK:STDOUT: !173 = !DILocation(line: 13, column: 3, scope: !171)
// CHECK:STDOUT: !174 = !{!172}
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
// CHECK:STDOUT: define void @_CUse.Main(ptr %b) #0 !dbg !177 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Base.F.call.vtable = load ptr, ptr %b, align 8, !dbg !180
// CHECK:STDOUT:   %Base.F.call = call ptr @llvm.load.relative.i32(ptr %Base.F.call.vtable, i32 0), !dbg !180
// CHECK:STDOUT:   call void %Base.F.call(ptr %b), !dbg !180
// CHECK:STDOUT:   ret void, !dbg !181
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: read)
// CHECK:STDOUT: declare ptr @llvm.load.relative.i32(ptr, i32) #1
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: read) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!176}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !21 = !{null, !20}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !175 = !DIFile(filename: "call.carbon", directory: "")
// CHECK:STDOUT: !176 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !175, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !177 = distinct !DISubprogram(name: "Use", linkageName: "_CUse.Main", scope: null, file: !175, line: 8, type: !22, spFlags: DISPFlagDefinition, unit: !176, retainedNodes: !182)
// CHECK:STDOUT: !178 = !DILocalVariable(arg: 1, scope: !177, type: !20)
// CHECK:STDOUT: !180 = !DILocation(line: 9, column: 3, scope: !177)
// CHECK:STDOUT: !181 = !DILocation(line: 8, column: 1, scope: !177)
// CHECK:STDOUT: !182 = !{!178}
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
// CHECK:STDOUT: define weak_odr void @"_CF:thunk:Base.Main:Derived.Main"(ptr %self) #0 !dbg !185 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Derived.Main(ptr %self), !dbg !188
// CHECK:STDOUT:   ret void, !dbg !188
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CUse.Main() #1 !dbg !190 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %v.var = alloca { { ptr } }, align 8, !dbg !191
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %v.var), !dbg !191
// CHECK:STDOUT:   %.loc14_32.2.base = getelementptr inbounds nuw { { ptr } }, ptr %v.var, i32 0, i32 0, !dbg !192
// CHECK:STDOUT:   %.loc14_31.2.vptr = getelementptr inbounds nuw { ptr }, ptr %.loc14_32.2.base, i32 0, i32 0, !dbg !193
// CHECK:STDOUT:   %.loc14_32.6.base = getelementptr inbounds nuw { { ptr } }, ptr %v.var, i32 0, i32 0, !dbg !192
// CHECK:STDOUT:   %.loc14_32.7.vptr = getelementptr inbounds nuw { ptr }, ptr %.loc14_32.6.base, i32 0, i32 0, !dbg !192
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 %v.var, ptr align 8 @Derived.val.loc14_32.5, i64 8, i1 false), !dbg !192
// CHECK:STDOUT:   store ptr @"_CDerived.Main.$vtable", ptr %.loc14_32.7.vptr, align 8, !dbg !192
// CHECK:STDOUT:   %.loc15_3.2.vptr = getelementptr inbounds nuw { ptr }, ptr %v.var, i32 0, i32 0, !dbg !195
// CHECK:STDOUT:   %Base.F.call.vtable = load ptr, ptr %.loc15_3.2.vptr, align 8, !dbg !195
// CHECK:STDOUT:   %Base.F.call = call ptr @llvm.load.relative.i32(ptr %Base.F.call.vtable, i32 0), !dbg !195
// CHECK:STDOUT:   call void %Base.F.call(ptr %.loc15_3.2.vptr), !dbg !195
// CHECK:STDOUT:   call void @"_COp.d8aa8f8dee0ff18e:core.Destroy.Core"(ptr %v.var), !dbg !191
// CHECK:STDOUT:   ret void, !dbg !196
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.8ea19ac2d92d0842:core.Destroy.Core"(ptr %self) #1 !dbg !197 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !199
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.26b62f84bf985626:core.Destroy.Core"(ptr %self) #1 !dbg !201 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !203
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.66f0cc6b13851099:core.Destroy.Core"(ptr %self) #1 !dbg !205 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !207
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.d8aa8f8dee0ff18e:core.Destroy.Core"(ptr %self) #1 !dbg !209 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !211
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!184}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !21 = !{null, !20}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !40 = !{null}
// CHECK:STDOUT: !41 = !DISubroutineType(types: !40)
// CHECK:STDOUT: !183 = !DIFile(filename: "call_impl.carbon", directory: "")
// CHECK:STDOUT: !184 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !183, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !185 = distinct !DISubprogram(name: "F", linkageName: "_CF:thunk:Base.Main:Derived.Main", scope: null, file: !183, line: 10, type: !22, spFlags: DISPFlagDefinition, unit: !184, retainedNodes: !189)
// CHECK:STDOUT: !186 = !DILocalVariable(arg: 1, scope: !185, type: !20)
// CHECK:STDOUT: !188 = !DILocation(line: 10, column: 3, scope: !185)
// CHECK:STDOUT: !189 = !{!186}
// CHECK:STDOUT: !190 = distinct !DISubprogram(name: "Use", linkageName: "_CUse.Main", scope: null, file: !183, line: 13, type: !41, spFlags: DISPFlagDefinition, unit: !184)
// CHECK:STDOUT: !191 = !DILocation(line: 14, column: 3, scope: !190)
// CHECK:STDOUT: !192 = !DILocation(line: 14, column: 21, scope: !190)
// CHECK:STDOUT: !193 = !DILocation(line: 14, column: 30, scope: !190)
// CHECK:STDOUT: !195 = !DILocation(line: 15, column: 3, scope: !190)
// CHECK:STDOUT: !196 = !DILocation(line: 13, column: 1, scope: !190)
// CHECK:STDOUT: !197 = distinct !DISubprogram(name: "Op", linkageName: "_COp.8ea19ac2d92d0842:core.Destroy.Core", scope: null, file: !183, line: 14, type: !22, spFlags: DISPFlagDefinition, unit: !184, retainedNodes: !200)
// CHECK:STDOUT: !198 = !DILocalVariable(arg: 1, scope: !197, type: !20)
// CHECK:STDOUT: !199 = !DILocation(line: 14, column: 3, scope: !197)
// CHECK:STDOUT: !200 = !{!198}
// CHECK:STDOUT: !201 = distinct !DISubprogram(name: "Op", linkageName: "_COp.26b62f84bf985626:core.Destroy.Core", scope: null, file: !183, line: 14, type: !22, spFlags: DISPFlagDefinition, unit: !184, retainedNodes: !204)
// CHECK:STDOUT: !202 = !DILocalVariable(arg: 1, scope: !201, type: !20)
// CHECK:STDOUT: !203 = !DILocation(line: 14, column: 3, scope: !201)
// CHECK:STDOUT: !204 = !{!202}
// CHECK:STDOUT: !205 = distinct !DISubprogram(name: "Op", linkageName: "_COp.66f0cc6b13851099:core.Destroy.Core", scope: null, file: !183, line: 14, type: !22, spFlags: DISPFlagDefinition, unit: !184, retainedNodes: !208)
// CHECK:STDOUT: !206 = !DILocalVariable(arg: 1, scope: !205, type: !20)
// CHECK:STDOUT: !207 = !DILocation(line: 14, column: 3, scope: !205)
// CHECK:STDOUT: !208 = !{!206}
// CHECK:STDOUT: !209 = distinct !DISubprogram(name: "Op", linkageName: "_COp.d8aa8f8dee0ff18e:core.Destroy.Core", scope: null, file: !183, line: 14, type: !22, spFlags: DISPFlagDefinition, unit: !184, retainedNodes: !212)
// CHECK:STDOUT: !210 = !DILocalVariable(arg: 1, scope: !209, type: !20)
// CHECK:STDOUT: !211 = !DILocation(line: 14, column: 3, scope: !209)
// CHECK:STDOUT: !212 = !{!210}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'generic_noop.carbon'
// CHECK:STDOUT: source_filename = "generic_noop.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!214}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !213 = !DIFile(filename: "generic_noop.carbon", directory: "")
// CHECK:STDOUT: !214 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !213, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'generic_use.carbon'
// CHECK:STDOUT: source_filename = "generic_use.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @"_CBase.Main.$vtable.9203bf580ba2de56" = unnamed_addr constant [1 x i32] [i32 trunc (i64 sub (i64 ptrtoint (ptr @_CF.Base.Main.9203bf580ba2de56 to i64), i64 ptrtoint (ptr @"_CBase.Main.$vtable.9203bf580ba2de56" to i64)) to i32)]
// CHECK:STDOUT: @"_CBase.Main.$vtable.d35bf058397cdfd9" = unnamed_addr constant [1 x i32] [i32 trunc (i64 sub (i64 ptrtoint (ptr @_CF.Base.Main.d35bf058397cdfd9 to i64), i64 ptrtoint (ptr @"_CBase.Main.$vtable.d35bf058397cdfd9" to i64)) to i32)]
// CHECK:STDOUT: @Base.val.389.loc16_3 = internal constant { ptr } { ptr @"_CBase.Main.$vtable.9203bf580ba2de56" }
// CHECK:STDOUT: @Base.val.7e9.loc17_3 = internal constant { ptr } { ptr @"_CBase.Main.$vtable.d35bf058397cdfd9" }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !217 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var.loc16 = alloca { ptr }, align 8, !dbg !218
// CHECK:STDOUT:   %_.var.loc17 = alloca { ptr }, align 8, !dbg !221
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc16), !dbg !218
// CHECK:STDOUT:   %.loc16_22.2.vptr = getelementptr inbounds nuw { ptr }, ptr %_.var.loc16, i32 0, i32 0, !dbg !219
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 %_.var.loc16, ptr align 8 @Base.val.389.loc16_3, i64 8, i1 false), !dbg !218
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc17), !dbg !221
// CHECK:STDOUT:   %.loc17_22.2.vptr = getelementptr inbounds nuw { ptr }, ptr %_.var.loc17, i32 0, i32 0, !dbg !222
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 %_.var.loc17, ptr align 8 @Base.val.7e9.loc17_3, i64 8, i1 false), !dbg !221
// CHECK:STDOUT:   call void @"_COp.6a1d86cdd5547971:core.Destroy.Core"(ptr %_.var.loc17), !dbg !221
// CHECK:STDOUT:   call void @"_COp.40b7c6c0efee38e9:core.Destroy.Core"(ptr %_.var.loc16), !dbg !218
// CHECK:STDOUT:   ret void, !dbg !224
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.2a8a57e92efa0931:core.Destroy.Core"(ptr %self) #0 !dbg !225 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !227
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.1afb52e20d18d857:core.Destroy.Core"(ptr %self) #0 !dbg !229 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !231
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.bbfe224db279cb80:core.Destroy.Core"(ptr %self) #0 !dbg !233 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !235
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.8ea19ac2d92d0842:core.Destroy.Core"(ptr %self) #0 !dbg !237 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !239
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.6a1d86cdd5547971:core.Destroy.Core"(ptr %self) #0 !dbg !241 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !243
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.40b7c6c0efee38e9:core.Destroy.Core"(ptr %self) #0 !dbg !245 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !247
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Base.Main.9203bf580ba2de56(ptr %self) #0 !dbg !249 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca {}, align 1, !dbg !252
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !252
// CHECK:STDOUT:   call void @"_COp.2a8a57e92efa0931:core.Destroy.Core"(ptr %_.var), !dbg !252
// CHECK:STDOUT:   ret void, !dbg !254
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Base.Main.d35bf058397cdfd9(ptr %self) #0 !dbg !256 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca { {} }, align 1, !dbg !259
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !259
// CHECK:STDOUT:   call void @"_COp.bbfe224db279cb80:core.Destroy.Core"(ptr %_.var), !dbg !259
// CHECK:STDOUT:   ret void, !dbg !261
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!216}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !21 = !{null, !20}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !40 = !{null}
// CHECK:STDOUT: !41 = !DISubroutineType(types: !40)
// CHECK:STDOUT: !215 = !DIFile(filename: "generic_use.carbon", directory: "")
// CHECK:STDOUT: !216 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !215, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !217 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !215, line: 15, type: !41, spFlags: DISPFlagDefinition, unit: !216)
// CHECK:STDOUT: !218 = !DILocation(line: 16, column: 3, scope: !217)
// CHECK:STDOUT: !219 = !DILocation(line: 16, column: 21, scope: !217)
// CHECK:STDOUT: !221 = !DILocation(line: 17, column: 3, scope: !217)
// CHECK:STDOUT: !222 = !DILocation(line: 17, column: 21, scope: !217)
// CHECK:STDOUT: !224 = !DILocation(line: 15, column: 1, scope: !217)
// CHECK:STDOUT: !225 = distinct !DISubprogram(name: "Op", linkageName: "_COp.2a8a57e92efa0931:core.Destroy.Core", scope: null, file: !215, line: 16, type: !22, spFlags: DISPFlagDefinition, unit: !216, retainedNodes: !228)
// CHECK:STDOUT: !226 = !DILocalVariable(arg: 1, scope: !225, type: !20)
// CHECK:STDOUT: !227 = !DILocation(line: 16, column: 10, scope: !225)
// CHECK:STDOUT: !228 = !{!226}
// CHECK:STDOUT: !229 = distinct !DISubprogram(name: "Op", linkageName: "_COp.1afb52e20d18d857:core.Destroy.Core", scope: null, file: !215, line: 17, type: !22, spFlags: DISPFlagDefinition, unit: !216, retainedNodes: !232)
// CHECK:STDOUT: !230 = !DILocalVariable(arg: 1, scope: !229, type: !20)
// CHECK:STDOUT: !231 = !DILocation(line: 17, column: 10, scope: !229)
// CHECK:STDOUT: !232 = !{!230}
// CHECK:STDOUT: !233 = distinct !DISubprogram(name: "Op", linkageName: "_COp.bbfe224db279cb80:core.Destroy.Core", scope: null, file: !215, line: 17, type: !22, spFlags: DISPFlagDefinition, unit: !216, retainedNodes: !236)
// CHECK:STDOUT: !234 = !DILocalVariable(arg: 1, scope: !233, type: !20)
// CHECK:STDOUT: !235 = !DILocation(line: 17, column: 10, scope: !233)
// CHECK:STDOUT: !236 = !{!234}
// CHECK:STDOUT: !237 = distinct !DISubprogram(name: "Op", linkageName: "_COp.8ea19ac2d92d0842:core.Destroy.Core", scope: null, file: !215, line: 17, type: !22, spFlags: DISPFlagDefinition, unit: !216, retainedNodes: !240)
// CHECK:STDOUT: !238 = !DILocalVariable(arg: 1, scope: !237, type: !20)
// CHECK:STDOUT: !239 = !DILocation(line: 17, column: 3, scope: !237)
// CHECK:STDOUT: !240 = !{!238}
// CHECK:STDOUT: !241 = distinct !DISubprogram(name: "Op", linkageName: "_COp.6a1d86cdd5547971:core.Destroy.Core", scope: null, file: !215, line: 17, type: !22, spFlags: DISPFlagDefinition, unit: !216, retainedNodes: !244)
// CHECK:STDOUT: !242 = !DILocalVariable(arg: 1, scope: !241, type: !20)
// CHECK:STDOUT: !243 = !DILocation(line: 17, column: 3, scope: !241)
// CHECK:STDOUT: !244 = !{!242}
// CHECK:STDOUT: !245 = distinct !DISubprogram(name: "Op", linkageName: "_COp.40b7c6c0efee38e9:core.Destroy.Core", scope: null, file: !215, line: 16, type: !22, spFlags: DISPFlagDefinition, unit: !216, retainedNodes: !248)
// CHECK:STDOUT: !246 = !DILocalVariable(arg: 1, scope: !245, type: !20)
// CHECK:STDOUT: !247 = !DILocation(line: 16, column: 3, scope: !245)
// CHECK:STDOUT: !248 = !{!246}
// CHECK:STDOUT: !249 = distinct !DISubprogram(name: "F", linkageName: "_CF.Base.Main.9203bf580ba2de56", scope: null, file: !215, line: 5, type: !22, spFlags: DISPFlagDefinition, unit: !216, retainedNodes: !255)
// CHECK:STDOUT: !250 = !DILocalVariable(arg: 1, scope: !249, type: !20)
// CHECK:STDOUT: !252 = !DILocation(line: 6, column: 5, scope: !249)
// CHECK:STDOUT: !254 = !DILocation(line: 5, column: 3, scope: !249)
// CHECK:STDOUT: !255 = !{!250}
// CHECK:STDOUT: !256 = distinct !DISubprogram(name: "F", linkageName: "_CF.Base.Main.d35bf058397cdfd9", scope: null, file: !215, line: 5, type: !22, spFlags: DISPFlagDefinition, unit: !216, retainedNodes: !262)
// CHECK:STDOUT: !257 = !DILocalVariable(arg: 1, scope: !256, type: !20)
// CHECK:STDOUT: !259 = !DILocation(line: 6, column: 5, scope: !256)
// CHECK:STDOUT: !261 = !DILocation(line: 5, column: 3, scope: !256)
// CHECK:STDOUT: !262 = !{!257}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'generic_base.carbon'
// CHECK:STDOUT: source_filename = "generic_base.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @"_CDerived.Main.$vtable" = unnamed_addr constant [1 x i32] [i32 trunc (i64 sub (i64 ptrtoint (ptr @_CF.Base.Main.6d6a4a2512463e59 to i64), i64 ptrtoint (ptr @"_CDerived.Main.$vtable" to i64)) to i32)]
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMake.Main() #0 !dbg !265 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca { { ptr } }, align 8, !dbg !266
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !266
// CHECK:STDOUT:   call void @"_COp.d8aa8f8dee0ff18e:core.Destroy.Core"(ptr %_.var), !dbg !266
// CHECK:STDOUT:   ret void, !dbg !268
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.8ea19ac2d92d0842:core.Destroy.Core"(ptr %self) #0 !dbg !269 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !271
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.898d07056c7ee329:core.Destroy.Core"(ptr %self) #0 !dbg !273 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !275
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.869a2dea742ac75c:core.Destroy.Core"(ptr %self) #0 !dbg !277 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !279
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.d8aa8f8dee0ff18e:core.Destroy.Core"(ptr %self) #0 !dbg !281 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !283
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Base.Main.6d6a4a2512463e59(ptr %self) #0 !dbg !285 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !288
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!264}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !21 = !{null, !20}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !40 = !{null}
// CHECK:STDOUT: !41 = !DISubroutineType(types: !40)
// CHECK:STDOUT: !263 = !DIFile(filename: "generic_base.carbon", directory: "")
// CHECK:STDOUT: !264 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !263, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !265 = distinct !DISubprogram(name: "Make", linkageName: "_CMake.Main", scope: null, file: !263, line: 14, type: !41, spFlags: DISPFlagDefinition, unit: !264)
// CHECK:STDOUT: !266 = !DILocation(line: 15, column: 3, scope: !265)
// CHECK:STDOUT: !268 = !DILocation(line: 14, column: 1, scope: !265)
// CHECK:STDOUT: !269 = distinct !DISubprogram(name: "Op", linkageName: "_COp.8ea19ac2d92d0842:core.Destroy.Core", scope: null, file: !263, line: 15, type: !22, spFlags: DISPFlagDefinition, unit: !264, retainedNodes: !272)
// CHECK:STDOUT: !270 = !DILocalVariable(arg: 1, scope: !269, type: !20)
// CHECK:STDOUT: !271 = !DILocation(line: 15, column: 3, scope: !269)
// CHECK:STDOUT: !272 = !{!270}
// CHECK:STDOUT: !273 = distinct !DISubprogram(name: "Op", linkageName: "_COp.898d07056c7ee329:core.Destroy.Core", scope: null, file: !263, line: 15, type: !22, spFlags: DISPFlagDefinition, unit: !264, retainedNodes: !276)
// CHECK:STDOUT: !274 = !DILocalVariable(arg: 1, scope: !273, type: !20)
// CHECK:STDOUT: !275 = !DILocation(line: 15, column: 3, scope: !273)
// CHECK:STDOUT: !276 = !{!274}
// CHECK:STDOUT: !277 = distinct !DISubprogram(name: "Op", linkageName: "_COp.869a2dea742ac75c:core.Destroy.Core", scope: null, file: !263, line: 15, type: !22, spFlags: DISPFlagDefinition, unit: !264, retainedNodes: !280)
// CHECK:STDOUT: !278 = !DILocalVariable(arg: 1, scope: !277, type: !20)
// CHECK:STDOUT: !279 = !DILocation(line: 15, column: 3, scope: !277)
// CHECK:STDOUT: !280 = !{!278}
// CHECK:STDOUT: !281 = distinct !DISubprogram(name: "Op", linkageName: "_COp.d8aa8f8dee0ff18e:core.Destroy.Core", scope: null, file: !263, line: 15, type: !22, spFlags: DISPFlagDefinition, unit: !264, retainedNodes: !284)
// CHECK:STDOUT: !282 = !DILocalVariable(arg: 1, scope: !281, type: !20)
// CHECK:STDOUT: !283 = !DILocation(line: 15, column: 3, scope: !281)
// CHECK:STDOUT: !284 = !{!282}
// CHECK:STDOUT: !285 = distinct !DISubprogram(name: "F", linkageName: "_CF.Base.Main.6d6a4a2512463e59", scope: null, file: !263, line: 5, type: !22, spFlags: DISPFlagDefinition, unit: !264, retainedNodes: !289)
// CHECK:STDOUT: !286 = !DILocalVariable(arg: 1, scope: !285, type: !20)
// CHECK:STDOUT: !288 = !DILocation(line: 5, column: 3, scope: !285)
// CHECK:STDOUT: !289 = !{!286}
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
// CHECK:STDOUT: define void @_CF.Derived.Main(ptr sret({}) %return, ptr %self, ptr %n) #0 !dbg !296 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %return, ptr align 1 @From.val.loc18_14, i64 0, i1 false), !dbg !303
// CHECK:STDOUT:   ret void, !dbg !303
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_CF:thunk:Base.Main:Derived.Main"(ptr sret({}) %return, ptr %self, ptr %n) #1 !dbg !305 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc17_52.2.temp = alloca {}, align 1, !dbg !309
// CHECK:STDOUT:   %.5.temp = alloca {}, align 1, !dbg !308
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc17_52.2.temp), !dbg !309
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.5.temp), !dbg !308
// CHECK:STDOUT:   call void @"_CConvert.From.Main:ImplicitAs.a3bb440f6426a7d6.Core"(ptr %.5.temp, ptr %n), !dbg !308
// CHECK:STDOUT:   call void @_CF.Derived.Main(ptr %.loc17_52.2.temp, ptr %self, ptr %.5.temp), !dbg !309
// CHECK:STDOUT:   call void @"_CConvert.From.Main:ImplicitAs.a3bb440f6426a7d6.Core"(ptr %return, ptr %.loc17_52.2.temp), !dbg !309
// CHECK:STDOUT:   call void @"_COp.5f7e8652ea6df95d:core.Destroy.Core"(ptr %.loc17_52.2.temp), !dbg !309
// CHECK:STDOUT:   call void @"_COp.ce04c064f6006102:core.Destroy.Core"(ptr %.5.temp), !dbg !308
// CHECK:STDOUT:   ret void, !dbg !309
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.5f7e8652ea6df95d:core.Destroy.Core"(ptr %self) #0 !dbg !311 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !313
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.ce04c064f6006102:core.Destroy.Core"(ptr %self) #0 !dbg !315 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !317
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CUse.Main() #0 !dbg !319 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %d.var = alloca { { ptr } }, align 8, !dbg !320
// CHECK:STDOUT:   %.loc25_10.1.temp = alloca {}, align 1, !dbg !328
// CHECK:STDOUT:   %.loc25_9.2.temp = alloca {}, align 1, !dbg !329
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %d.var), !dbg !320
// CHECK:STDOUT:   %.loc23_31.2.base = getelementptr inbounds nuw { { ptr } }, ptr %d.var, i32 0, i32 0, !dbg !321
// CHECK:STDOUT:   %.loc23_30.2.vptr = getelementptr inbounds nuw { ptr }, ptr %.loc23_31.2.base, i32 0, i32 0, !dbg !322
// CHECK:STDOUT:   %.loc23_31.6.base = getelementptr inbounds nuw { { ptr } }, ptr %d.var, i32 0, i32 0, !dbg !321
// CHECK:STDOUT:   %.loc23_31.7.vptr = getelementptr inbounds nuw { ptr }, ptr %.loc23_31.6.base, i32 0, i32 0, !dbg !321
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 %d.var, ptr align 8 @Derived.val.loc23_31.5, i64 8, i1 false), !dbg !321
// CHECK:STDOUT:   store ptr @"_CDerived.Main.$vtable", ptr %.loc23_31.7.vptr, align 8, !dbg !321
// CHECK:STDOUT:   %.loc24_18.2.base = getelementptr inbounds nuw { { ptr } }, ptr %d.var, i32 0, i32 0, !dbg !325
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc25_10.1.temp), !dbg !328
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc25_9.2.temp), !dbg !329
// CHECK:STDOUT:   %Base.F.call.vtable = load ptr, ptr %.loc25_10.1.temp, align 8, !dbg !328
// CHECK:STDOUT:   %Base.F.call = call ptr @llvm.load.relative.i32(ptr %Base.F.call.vtable, i32 0), !dbg !328
// CHECK:STDOUT:   call void %Base.F.call(ptr %.loc25_10.1.temp, ptr %.loc24_18.2.base, ptr @From.val.loc18_14), !dbg !328
// CHECK:STDOUT:   call void @"_COp.ce04c064f6006102:core.Destroy.Core"(ptr %.loc25_10.1.temp), !dbg !328
// CHECK:STDOUT:   call void @"_COp.5f7e8652ea6df95d:core.Destroy.Core"(ptr @From.val), !dbg !329
// CHECK:STDOUT:   call void @"_COp.d8aa8f8dee0ff18e:core.Destroy.Core"(ptr %d.var), !dbg !320
// CHECK:STDOUT:   ret void, !dbg !330
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.8ea19ac2d92d0842:core.Destroy.Core"(ptr %self) #0 !dbg !331 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !333
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.26b62f84bf985626:core.Destroy.Core"(ptr %self) #0 !dbg !335 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !337
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.66f0cc6b13851099:core.Destroy.Core"(ptr %self) #0 !dbg !339 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !341
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.d8aa8f8dee0ff18e:core.Destroy.Core"(ptr %self) #0 !dbg !343 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !345
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!291}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !21 = !{null, !20}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !40 = !{null}
// CHECK:STDOUT: !41 = !DISubroutineType(types: !40)
// CHECK:STDOUT: !290 = !DIFile(filename: "thunk.carbon", directory: "")
// CHECK:STDOUT: !291 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !290, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !294 = !{!20, !20, !20}
// CHECK:STDOUT: !295 = !DISubroutineType(types: !294)
// CHECK:STDOUT: !296 = distinct !DISubprogram(name: "F", linkageName: "_CF.Derived.Main", scope: null, file: !290, line: 17, type: !295, spFlags: DISPFlagDefinition, unit: !291, retainedNodes: !304)
// CHECK:STDOUT: !297 = !DILocalVariable(arg: 1, scope: !296, type: !20)
// CHECK:STDOUT: !298 = !DILocalVariable(arg: 2, scope: !296, type: !20)
// CHECK:STDOUT: !303 = !DILocation(line: 18, column: 5, scope: !296)
// CHECK:STDOUT: !304 = !{!297, !298}
// CHECK:STDOUT: !305 = distinct !DISubprogram(name: "F", linkageName: "_CF:thunk:Base.Main:Derived.Main", scope: null, file: !290, line: 17, type: !295, spFlags: DISPFlagDefinition, unit: !291, retainedNodes: !310)
// CHECK:STDOUT: !306 = !DILocalVariable(arg: 1, scope: !305, type: !20)
// CHECK:STDOUT: !307 = !DILocalVariable(arg: 2, scope: !305, type: !20)
// CHECK:STDOUT: !308 = !DILocation(line: 0, scope: !305)
// CHECK:STDOUT: !309 = !DILocation(line: 17, column: 3, scope: !305)
// CHECK:STDOUT: !310 = !{!306, !307}
// CHECK:STDOUT: !311 = distinct !DISubprogram(name: "Op", linkageName: "_COp.5f7e8652ea6df95d:core.Destroy.Core", scope: null, file: !290, line: 17, type: !22, spFlags: DISPFlagDefinition, unit: !291, retainedNodes: !314)
// CHECK:STDOUT: !312 = !DILocalVariable(arg: 1, scope: !311, type: !20)
// CHECK:STDOUT: !313 = !DILocation(line: 17, column: 3, scope: !311)
// CHECK:STDOUT: !314 = !{!312}
// CHECK:STDOUT: !315 = distinct !DISubprogram(name: "Op", linkageName: "_COp.ce04c064f6006102:core.Destroy.Core", scope: null, file: !290, type: !22, spFlags: DISPFlagDefinition, unit: !291, retainedNodes: !318)
// CHECK:STDOUT: !316 = !DILocalVariable(arg: 1, scope: !315, type: !20)
// CHECK:STDOUT: !317 = !DILocation(line: 0, scope: !315)
// CHECK:STDOUT: !318 = !{!316}
// CHECK:STDOUT: !319 = distinct !DISubprogram(name: "Use", linkageName: "_CUse.Main", scope: null, file: !290, line: 22, type: !41, spFlags: DISPFlagDefinition, unit: !291)
// CHECK:STDOUT: !320 = !DILocation(line: 23, column: 3, scope: !319)
// CHECK:STDOUT: !321 = !DILocation(line: 23, column: 20, scope: !319)
// CHECK:STDOUT: !322 = !DILocation(line: 23, column: 29, scope: !319)
// CHECK:STDOUT: !325 = !DILocation(line: 24, column: 18, scope: !319)
// CHECK:STDOUT: !328 = !DILocation(line: 25, column: 3, scope: !319)
// CHECK:STDOUT: !329 = !DILocation(line: 25, column: 8, scope: !319)
// CHECK:STDOUT: !330 = !DILocation(line: 22, column: 1, scope: !319)
// CHECK:STDOUT: !331 = distinct !DISubprogram(name: "Op", linkageName: "_COp.8ea19ac2d92d0842:core.Destroy.Core", scope: null, file: !290, line: 23, type: !22, spFlags: DISPFlagDefinition, unit: !291, retainedNodes: !334)
// CHECK:STDOUT: !332 = !DILocalVariable(arg: 1, scope: !331, type: !20)
// CHECK:STDOUT: !333 = !DILocation(line: 23, column: 3, scope: !331)
// CHECK:STDOUT: !334 = !{!332}
// CHECK:STDOUT: !335 = distinct !DISubprogram(name: "Op", linkageName: "_COp.26b62f84bf985626:core.Destroy.Core", scope: null, file: !290, line: 23, type: !22, spFlags: DISPFlagDefinition, unit: !291, retainedNodes: !338)
// CHECK:STDOUT: !336 = !DILocalVariable(arg: 1, scope: !335, type: !20)
// CHECK:STDOUT: !337 = !DILocation(line: 23, column: 3, scope: !335)
// CHECK:STDOUT: !338 = !{!336}
// CHECK:STDOUT: !339 = distinct !DISubprogram(name: "Op", linkageName: "_COp.66f0cc6b13851099:core.Destroy.Core", scope: null, file: !290, line: 23, type: !22, spFlags: DISPFlagDefinition, unit: !291, retainedNodes: !342)
// CHECK:STDOUT: !340 = !DILocalVariable(arg: 1, scope: !339, type: !20)
// CHECK:STDOUT: !341 = !DILocation(line: 23, column: 3, scope: !339)
// CHECK:STDOUT: !342 = !{!340}
// CHECK:STDOUT: !343 = distinct !DISubprogram(name: "Op", linkageName: "_COp.d8aa8f8dee0ff18e:core.Destroy.Core", scope: null, file: !290, line: 23, type: !22, spFlags: DISPFlagDefinition, unit: !291, retainedNodes: !346)
// CHECK:STDOUT: !344 = !DILocalVariable(arg: 1, scope: !343, type: !20)
// CHECK:STDOUT: !345 = !DILocation(line: 23, column: 3, scope: !343)
// CHECK:STDOUT: !346 = !{!344}
// CHECK:STDOUT:
