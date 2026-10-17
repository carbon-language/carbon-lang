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
// CHECK:STDOUT: @Base.val.loc7_3.1 = internal constant {} zeroinitializer
// CHECK:STDOUT: @Intermediate.val.54b.loc8_3.1 = internal constant { ptr, {} } { ptr @"_CIntermediate.Classes.$vtable", {} zeroinitializer }
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
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %_.var.loc7, ptr align 1 @Base.val.loc7_3.1, i64 0, i1 false), !dbg !43
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc8), !dbg !45
// CHECK:STDOUT:   %.loc8_44.2.vptr = getelementptr inbounds nuw { ptr, {} }, ptr %_.var.loc8, i32 0, i32 0, !dbg !46
// CHECK:STDOUT:   %.loc8_44.4.base = getelementptr inbounds nuw { ptr, {} }, ptr %_.var.loc8, i32 0, i32 1, !dbg !46
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 %_.var.loc8, ptr align 8 @Intermediate.val.54b.loc8_3.1, i64 8, i1 false), !dbg !45
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc9), !dbg !48
// CHECK:STDOUT:   %.loc9_49.2.base = getelementptr inbounds nuw { { ptr, {} } }, ptr %_.var.loc9, i32 0, i32 0, !dbg !49
// CHECK:STDOUT:   %.loc9_48.2.vptr = getelementptr inbounds nuw { ptr, {} }, ptr %.loc9_49.2.base, i32 0, i32 0, !dbg !50
// CHECK:STDOUT:   %.loc9_48.4.base = getelementptr inbounds nuw { ptr, {} }, ptr %.loc9_49.2.base, i32 0, i32 1, !dbg !50
// CHECK:STDOUT:   %.loc9_49.6.base = getelementptr inbounds nuw { { ptr, {} } }, ptr %_.var.loc9, i32 0, i32 0, !dbg !49
// CHECK:STDOUT:   %.loc9_49.7.vptr = getelementptr inbounds nuw { ptr, {} }, ptr %.loc9_49.6.base, i32 0, i32 0, !dbg !49
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 %_.var.loc9, ptr align 8 @Derived.val.loc9_49.5, i64 8, i1 false), !dbg !49
// CHECK:STDOUT:   store ptr @"_CDerived.Classes.$vtable", ptr %.loc9_49.7.vptr, align 8, !dbg !49
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc12), !dbg !52
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.4ebe230b4b7eb612(ptr %_.var.loc12), !dbg !52
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.4ebe230b4b7eb612(ptr %_.var.loc9), !dbg !48
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.36e476cc7e22676a(ptr %_.var.loc8), !dbg !45
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.9dd568d598bdbcbd(ptr %_.var.loc7), !dbg !43
// CHECK:STDOUT:   ret void, !dbg !54
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CFn.Intermediate.Classes(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_CFn:thunk:Intermediate.Classes:Derived.Classes"(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CUse.Create(ptr %v) #0 !dbg !55 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Intermediate.Fn.call.vtable = load ptr, ptr %v, align 8, !dbg !58
// CHECK:STDOUT:   %Intermediate.Fn.call = call ptr @llvm.load.relative.i32(ptr %Intermediate.Fn.call.vtable, i32 0), !dbg !58
// CHECK:STDOUT:   call void %Intermediate.Fn.call(ptr %v), !dbg !58
// CHECK:STDOUT:   ret void, !dbg !59
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.4ebe230b4b7eb612(ptr %t) #0 !dbg !61 {
// CHECK:STDOUT:   ret void, !dbg !66
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.36e476cc7e22676a(ptr %t) #0 !dbg !68 {
// CHECK:STDOUT:   ret void, !dbg !73
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.9dd568d598bdbcbd(ptr %t) #0 !dbg !75 {
// CHECK:STDOUT:   ret void, !dbg !80
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: read)
// CHECK:STDOUT: declare ptr @llvm.load.relative.i32(ptr, i32) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.4ebe230b4b7eb612, { 1, 0 }
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
// CHECK:STDOUT: !10 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
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
// CHECK:STDOUT: !55 = distinct !DISubprogram(name: "Use", linkageName: "_CUse.Create", scope: null, file: !38, line: 15, type: !22, spFlags: DISPFlagDefinition, unit: !39, retainedNodes: !60)
// CHECK:STDOUT: !56 = !DILocalVariable(arg: 1, scope: !55, type: !20)
// CHECK:STDOUT: !58 = !DILocation(line: 16, column: 3, scope: !55)
// CHECK:STDOUT: !59 = !DILocation(line: 15, column: 1, scope: !55)
// CHECK:STDOUT: !60 = !{!56}
// CHECK:STDOUT: !61 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.4ebe230b4b7eb612", scope: null, file: !10, line: 30, type: !22, spFlags: DISPFlagDefinition, unit: !39, retainedNodes: !67)
// CHECK:STDOUT: !62 = !DILocalVariable(arg: 1, scope: !61, type: !20)
// CHECK:STDOUT: !66 = !DILocation(line: 30, column: 1, scope: !61)
// CHECK:STDOUT: !67 = !{!62}
// CHECK:STDOUT: !68 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.36e476cc7e22676a", scope: null, file: !10, line: 30, type: !22, spFlags: DISPFlagDefinition, unit: !39, retainedNodes: !74)
// CHECK:STDOUT: !69 = !DILocalVariable(arg: 1, scope: !68, type: !20)
// CHECK:STDOUT: !73 = !DILocation(line: 30, column: 1, scope: !68)
// CHECK:STDOUT: !74 = !{!69}
// CHECK:STDOUT: !75 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.9dd568d598bdbcbd", scope: null, file: !10, line: 30, type: !22, spFlags: DISPFlagDefinition, unit: !39, retainedNodes: !81)
// CHECK:STDOUT: !76 = !DILocalVariable(arg: 1, scope: !75, type: !20)
// CHECK:STDOUT: !80 = !DILocation(line: 30, column: 1, scope: !75)
// CHECK:STDOUT: !81 = !{!76}
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
// CHECK:STDOUT: define void @_CMakePartialBase.Main(ptr sret({}) %return) #0 !dbg !86 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %return, ptr align 1 @empty_struct.ac8.loc7_12, i64 0, i1 false), !dbg !89
// CHECK:STDOUT:   ret void, !dbg !89
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMakePartialIntermediate.Main(ptr sret({ ptr, {} }) %return) #0 !dbg !90 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc11_36.1.base = getelementptr inbounds nuw { ptr, {} }, ptr %return, i32 0, i32 1, !dbg !93
// CHECK:STDOUT:   call void @_CMakePartialBase.Main(ptr %.loc11_36.1.base), !dbg !94
// CHECK:STDOUT:   %.loc11_36.3.vptr = getelementptr inbounds nuw { ptr, {} }, ptr %return, i32 0, i32 0, !dbg !93
// CHECK:STDOUT:   store ptr poison, ptr %.loc11_36.3.vptr, align 8, !dbg !93
// CHECK:STDOUT:   ret void, !dbg !95
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMakeBase.Main(ptr sret({}) %return) #0 !dbg !96 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CMakePartialBase.Main(ptr %return), !dbg !99
// CHECK:STDOUT:   ret void, !dbg !100
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMakeIntermediate.Main(ptr sret({ ptr, {} }) %return) #0 !dbg !101 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CMakePartialIntermediate.Main(ptr %return), !dbg !104
// CHECK:STDOUT:   %.loc19_34.1.vptr = getelementptr inbounds nuw { ptr, {} }, ptr %return, i32 0, i32 0, !dbg !104
// CHECK:STDOUT:   store ptr @"_CIntermediate.Classes.$vtable", ptr %.loc19_34.1.vptr, align 8, !dbg !104
// CHECK:STDOUT:   ret void, !dbg !105
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CFn.Intermediate.Classes(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMakeDerivedFromPartial.Main(ptr sret({ { ptr, {} } }) %return) #0 !dbg !106 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc23_44.1.base = getelementptr inbounds nuw { { ptr, {} } }, ptr %return, i32 0, i32 0, !dbg !109
// CHECK:STDOUT:   call void @_CMakePartialIntermediate.Main(ptr %.loc23_44.1.base), !dbg !110
// CHECK:STDOUT:   %.loc23_44.5.base = getelementptr inbounds nuw { { ptr, {} } }, ptr %return, i32 0, i32 0, !dbg !109
// CHECK:STDOUT:   %.loc23_44.6.vptr = getelementptr inbounds nuw { ptr, {} }, ptr %.loc23_44.5.base, i32 0, i32 0, !dbg !109
// CHECK:STDOUT:   store ptr @"_CDerived.Classes.$vtable", ptr %.loc23_44.6.vptr, align 8, !dbg !109
// CHECK:STDOUT:   ret void, !dbg !111
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_CFn:thunk:Intermediate.Classes:Derived.Classes"(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMakeDerivedFromNonPartial.Main(ptr sret({ { ptr, {} } }) %return) #0 !dbg !112 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc27_37.1.base = getelementptr inbounds nuw { { ptr, {} } }, ptr %return, i32 0, i32 0, !dbg !115
// CHECK:STDOUT:   call void @_CMakeIntermediate.Main(ptr %.loc27_37.1.base), !dbg !116
// CHECK:STDOUT:   %.loc27_37.7.base = getelementptr inbounds nuw { { ptr, {} } }, ptr %return, i32 0, i32 0, !dbg !115
// CHECK:STDOUT:   %.loc27_37.8.vptr = getelementptr inbounds nuw { ptr, {} }, ptr %.loc27_37.7.base, i32 0, i32 0, !dbg !115
// CHECK:STDOUT:   store ptr @"_CDerived.Classes.$vtable", ptr %.loc27_37.8.vptr, align 8, !dbg !115
// CHECK:STDOUT:   ret void, !dbg !117
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!83}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !82 = !DIFile(filename: "create_named_ctor.carbon", directory: "")
// CHECK:STDOUT: !83 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !82, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !84 = !{!20}
// CHECK:STDOUT: !85 = !DISubroutineType(types: !84)
// CHECK:STDOUT: !86 = distinct !DISubprogram(name: "MakePartialBase", linkageName: "_CMakePartialBase.Main", scope: null, file: !82, line: 6, type: !85, spFlags: DISPFlagDefinition, unit: !83)
// CHECK:STDOUT: !89 = !DILocation(line: 7, column: 3, scope: !86)
// CHECK:STDOUT: !90 = distinct !DISubprogram(name: "MakePartialIntermediate", linkageName: "_CMakePartialIntermediate.Main", scope: null, file: !82, line: 10, type: !85, spFlags: DISPFlagDefinition, unit: !83)
// CHECK:STDOUT: !93 = !DILocation(line: 11, column: 10, scope: !90)
// CHECK:STDOUT: !94 = !DILocation(line: 11, column: 19, scope: !90)
// CHECK:STDOUT: !95 = !DILocation(line: 11, column: 3, scope: !90)
// CHECK:STDOUT: !96 = distinct !DISubprogram(name: "MakeBase", linkageName: "_CMakeBase.Main", scope: null, file: !82, line: 14, type: !85, spFlags: DISPFlagDefinition, unit: !83)
// CHECK:STDOUT: !99 = !DILocation(line: 15, column: 10, scope: !96)
// CHECK:STDOUT: !100 = !DILocation(line: 15, column: 3, scope: !96)
// CHECK:STDOUT: !101 = distinct !DISubprogram(name: "MakeIntermediate", linkageName: "_CMakeIntermediate.Main", scope: null, file: !82, line: 18, type: !85, spFlags: DISPFlagDefinition, unit: !83)
// CHECK:STDOUT: !104 = !DILocation(line: 19, column: 10, scope: !101)
// CHECK:STDOUT: !105 = !DILocation(line: 19, column: 3, scope: !101)
// CHECK:STDOUT: !106 = distinct !DISubprogram(name: "MakeDerivedFromPartial", linkageName: "_CMakeDerivedFromPartial.Main", scope: null, file: !82, line: 22, type: !85, spFlags: DISPFlagDefinition, unit: !83)
// CHECK:STDOUT: !109 = !DILocation(line: 23, column: 10, scope: !106)
// CHECK:STDOUT: !110 = !DILocation(line: 23, column: 19, scope: !106)
// CHECK:STDOUT: !111 = !DILocation(line: 23, column: 3, scope: !106)
// CHECK:STDOUT: !112 = distinct !DISubprogram(name: "MakeDerivedFromNonPartial", linkageName: "_CMakeDerivedFromNonPartial.Main", scope: null, file: !82, line: 26, type: !85, spFlags: DISPFlagDefinition, unit: !83)
// CHECK:STDOUT: !115 = !DILocation(line: 27, column: 10, scope: !112)
// CHECK:STDOUT: !116 = !DILocation(line: 27, column: 19, scope: !112)
// CHECK:STDOUT: !117 = !DILocation(line: 27, column: 3, scope: !112)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'member_init.carbon'
// CHECK:STDOUT: source_filename = "member_init.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @"_CBase.MemberInit.$vtable" = unnamed_addr constant [1 x i32] [i32 trunc (i64 sub (i64 ptrtoint (ptr @_CFn.Base.MemberInit to i64), i64 ptrtoint (ptr @"_CBase.MemberInit.$vtable" to i64)) to i32)]
// CHECK:STDOUT: @Base.val.loc13_3.1 = internal constant <{ ptr, i32 }> <{ ptr @"_CBase.MemberInit.$vtable", i32 3 }>
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CFn.Base.MemberInit(ptr %self) #0 !dbg !121 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !124
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CFn.MemberInit() #0 !dbg !126 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %i.var = alloca i32, align 4, !dbg !127
// CHECK:STDOUT:   %v.var = alloca <{ ptr, i32 }>, align 8, !dbg !129
// CHECK:STDOUT:   %_.var = alloca <{ ptr, i32 }>, align 8, !dbg !134
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %i.var), !dbg !127
// CHECK:STDOUT:   store i32 3, ptr %i.var, align 4, !dbg !127
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %v.var), !dbg !129
// CHECK:STDOUT:   %.loc11_24.2.vptr = getelementptr inbounds nuw <{ ptr, i32 }>, ptr %v.var, i32 0, i32 0, !dbg !131
// CHECK:STDOUT:   %.loc11_23 = load i32, ptr %i.var, align 4, !dbg !130
// CHECK:STDOUT:   %.loc11_24.4.m = getelementptr inbounds nuw <{ ptr, i32 }>, ptr %v.var, i32 0, i32 1, !dbg !131
// CHECK:STDOUT:   store i32 %.loc11_23, ptr %.loc11_24.4.m, align 4, !dbg !131
// CHECK:STDOUT:   store ptr @"_CBase.MemberInit.$vtable", ptr %.loc11_24.2.vptr, align 8, !dbg !131
// CHECK:STDOUT:   %.loc12_4.m = getelementptr inbounds nuw <{ ptr, i32 }>, ptr %v.var, i32 0, i32 1, !dbg !133
// CHECK:STDOUT:   store i32 5, ptr %.loc12_4.m, align 4, !dbg !133
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !134
// CHECK:STDOUT:   %.loc13_24.2.vptr = getelementptr inbounds nuw <{ ptr, i32 }>, ptr %_.var, i32 0, i32 0, !dbg !135
// CHECK:STDOUT:   %.loc13_24.5.m = getelementptr inbounds nuw <{ ptr, i32 }>, ptr %_.var, i32 0, i32 1, !dbg !135
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 %_.var, ptr align 8 @Base.val.loc13_3.1, i64 12, i1 false), !dbg !134
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.a9d732896496568f(ptr %_.var), !dbg !134
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.a9d732896496568f(ptr %v.var), !dbg !129
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.dc5658e0ac74b2aa(ptr %i.var), !dbg !127
// CHECK:STDOUT:   ret void, !dbg !139
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.a9d732896496568f(ptr %t) #0 !dbg !140 {
// CHECK:STDOUT:   ret void, !dbg !145
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.dc5658e0ac74b2aa(ptr %t) #0 !dbg !147 {
// CHECK:STDOUT:   ret void, !dbg !152
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.a9d732896496568f, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!119}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !10 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !20 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !21 = !{null, !20}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !40 = !{null}
// CHECK:STDOUT: !41 = !DISubroutineType(types: !40)
// CHECK:STDOUT: !118 = !DIFile(filename: "member_init.carbon", directory: "")
// CHECK:STDOUT: !119 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !118, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !120 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !121 = distinct !DISubprogram(name: "Fn", linkageName: "_CFn.Base.MemberInit", scope: null, file: !118, line: 6, type: !22, spFlags: DISPFlagDefinition, unit: !119, retainedNodes: !125)
// CHECK:STDOUT: !122 = !DILocalVariable(arg: 1, scope: !121, type: !20)
// CHECK:STDOUT: !124 = !DILocation(line: 6, column: 3, scope: !121)
// CHECK:STDOUT: !125 = !{!122}
// CHECK:STDOUT: !126 = distinct !DISubprogram(name: "Fn", linkageName: "_CFn.MemberInit", scope: null, file: !118, line: 9, type: !41, spFlags: DISPFlagDefinition, unit: !119)
// CHECK:STDOUT: !127 = !DILocation(line: 10, column: 3, scope: !126)
// CHECK:STDOUT: !129 = !DILocation(line: 11, column: 3, scope: !126)
// CHECK:STDOUT: !130 = !DILocation(line: 11, column: 23, scope: !126)
// CHECK:STDOUT: !131 = !DILocation(line: 11, column: 17, scope: !126)
// CHECK:STDOUT: !133 = !DILocation(line: 12, column: 3, scope: !126)
// CHECK:STDOUT: !134 = !DILocation(line: 13, column: 3, scope: !126)
// CHECK:STDOUT: !135 = !DILocation(line: 13, column: 17, scope: !126)
// CHECK:STDOUT: !137 = !{null, !120}
// CHECK:STDOUT: !138 = !DISubroutineType(types: !137)
// CHECK:STDOUT: !139 = !DILocation(line: 9, column: 1, scope: !126)
// CHECK:STDOUT: !140 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.a9d732896496568f", scope: null, file: !10, line: 30, type: !22, spFlags: DISPFlagDefinition, unit: !119, retainedNodes: !146)
// CHECK:STDOUT: !141 = !DILocalVariable(arg: 1, scope: !140, type: !20)
// CHECK:STDOUT: !145 = !DILocation(line: 30, column: 1, scope: !140)
// CHECK:STDOUT: !146 = !{!141}
// CHECK:STDOUT: !147 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.dc5658e0ac74b2aa", scope: null, file: !10, line: 30, type: !138, spFlags: DISPFlagDefinition, unit: !119, retainedNodes: !153)
// CHECK:STDOUT: !148 = !DILocalVariable(arg: 1, scope: !147, type: !120)
// CHECK:STDOUT: !152 = !DILocation(line: 30, column: 1, scope: !147)
// CHECK:STDOUT: !153 = !{!148}
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
// CHECK:STDOUT: define void @_CUse.Main() #0 !dbg !156 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca { { ptr } }, align 8, !dbg !157
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !157
// CHECK:STDOUT:   %.loc13_32.2.base = getelementptr inbounds nuw { { ptr } }, ptr %_.var, i32 0, i32 0, !dbg !158
// CHECK:STDOUT:   %.loc13_31.2.vptr = getelementptr inbounds nuw { ptr }, ptr %.loc13_32.2.base, i32 0, i32 0, !dbg !159
// CHECK:STDOUT:   %.loc13_32.6.base = getelementptr inbounds nuw { { ptr } }, ptr %_.var, i32 0, i32 0, !dbg !158
// CHECK:STDOUT:   %.loc13_32.7.vptr = getelementptr inbounds nuw { ptr }, ptr %.loc13_32.6.base, i32 0, i32 0, !dbg !158
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 %_.var, ptr align 8 @Derived.val.loc13_32.5, i64 8, i1 false), !dbg !158
// CHECK:STDOUT:   store ptr @"_CDerived.Main.$vtable", ptr %.loc13_32.7.vptr, align 8, !dbg !158
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.d81ad81840dd2e3c(ptr %_.var), !dbg !157
// CHECK:STDOUT:   ret void, !dbg !161
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.d81ad81840dd2e3c(ptr %t) #0 !dbg !162 {
// CHECK:STDOUT:   ret void, !dbg !167
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!155}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !10 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !20 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !21 = !{null, !20}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !40 = !{null}
// CHECK:STDOUT: !41 = !DISubroutineType(types: !40)
// CHECK:STDOUT: !154 = !DIFile(filename: "member_brace_init.carbon", directory: "")
// CHECK:STDOUT: !155 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !154, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !156 = distinct !DISubprogram(name: "Use", linkageName: "_CUse.Main", scope: null, file: !154, line: 12, type: !41, spFlags: DISPFlagDefinition, unit: !155)
// CHECK:STDOUT: !157 = !DILocation(line: 13, column: 3, scope: !156)
// CHECK:STDOUT: !158 = !DILocation(line: 13, column: 21, scope: !156)
// CHECK:STDOUT: !159 = !DILocation(line: 13, column: 30, scope: !156)
// CHECK:STDOUT: !161 = !DILocation(line: 12, column: 1, scope: !156)
// CHECK:STDOUT: !162 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.d81ad81840dd2e3c", scope: null, file: !10, line: 30, type: !22, spFlags: DISPFlagDefinition, unit: !155, retainedNodes: !168)
// CHECK:STDOUT: !163 = !DILocalVariable(arg: 1, scope: !162, type: !20)
// CHECK:STDOUT: !167 = !DILocation(line: 30, column: 1, scope: !162)
// CHECK:STDOUT: !168 = !{!163}
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
// CHECK:STDOUT: define void @_CUse.Main(ptr %b) #0 !dbg !171 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Base.F.call.vtable = load ptr, ptr %b, align 8, !dbg !174
// CHECK:STDOUT:   %Base.F.call = call ptr @llvm.load.relative.i32(ptr %Base.F.call.vtable, i32 0), !dbg !174
// CHECK:STDOUT:   call void %Base.F.call(ptr %b), !dbg !174
// CHECK:STDOUT:   ret void, !dbg !175
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: read)
// CHECK:STDOUT: declare ptr @llvm.load.relative.i32(ptr, i32) #1
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: read) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!170}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !21 = !{null, !20}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !169 = !DIFile(filename: "call.carbon", directory: "")
// CHECK:STDOUT: !170 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !169, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !171 = distinct !DISubprogram(name: "Use", linkageName: "_CUse.Main", scope: null, file: !169, line: 8, type: !22, spFlags: DISPFlagDefinition, unit: !170, retainedNodes: !176)
// CHECK:STDOUT: !172 = !DILocalVariable(arg: 1, scope: !171, type: !20)
// CHECK:STDOUT: !174 = !DILocation(line: 9, column: 3, scope: !171)
// CHECK:STDOUT: !175 = !DILocation(line: 8, column: 1, scope: !171)
// CHECK:STDOUT: !176 = !{!172}
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
// CHECK:STDOUT: define weak_odr void @"_CF:thunk:Base.Main:Derived.Main"(ptr %self) #0 !dbg !179 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Derived.Main(ptr %self), !dbg !182
// CHECK:STDOUT:   ret void, !dbg !182
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CUse.Main() #1 !dbg !184 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %v.var = alloca { { ptr } }, align 8, !dbg !185
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %v.var), !dbg !185
// CHECK:STDOUT:   %.loc14_32.2.base = getelementptr inbounds nuw { { ptr } }, ptr %v.var, i32 0, i32 0, !dbg !186
// CHECK:STDOUT:   %.loc14_31.2.vptr = getelementptr inbounds nuw { ptr }, ptr %.loc14_32.2.base, i32 0, i32 0, !dbg !187
// CHECK:STDOUT:   %.loc14_32.6.base = getelementptr inbounds nuw { { ptr } }, ptr %v.var, i32 0, i32 0, !dbg !186
// CHECK:STDOUT:   %.loc14_32.7.vptr = getelementptr inbounds nuw { ptr }, ptr %.loc14_32.6.base, i32 0, i32 0, !dbg !186
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 %v.var, ptr align 8 @Derived.val.loc14_32.5, i64 8, i1 false), !dbg !186
// CHECK:STDOUT:   store ptr @"_CDerived.Main.$vtable", ptr %.loc14_32.7.vptr, align 8, !dbg !186
// CHECK:STDOUT:   %.loc15_3.1.base = getelementptr inbounds nuw { { ptr } }, ptr %v.var, i32 0, i32 0, !dbg !189
// CHECK:STDOUT:   %Base.F.call.vtable = load ptr, ptr %.loc15_3.1.base, align 8, !dbg !189
// CHECK:STDOUT:   %Base.F.call = call ptr @llvm.load.relative.i32(ptr %Base.F.call.vtable, i32 0), !dbg !189
// CHECK:STDOUT:   call void %Base.F.call(ptr %.loc15_3.1.base), !dbg !189
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.d81ad81840dd2e3c(ptr %v.var), !dbg !185
// CHECK:STDOUT:   ret void, !dbg !190
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
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.d81ad81840dd2e3c(ptr %t) #1 !dbg !191 {
// CHECK:STDOUT:   ret void, !dbg !196
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { alwaysinline nounwind }
// CHECK:STDOUT: attributes #1 = { nounwind }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #3 = { nocallback nofree nosync nounwind willreturn memory(argmem: read) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!178}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !10 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !20 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !21 = !{null, !20}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !40 = !{null}
// CHECK:STDOUT: !41 = !DISubroutineType(types: !40)
// CHECK:STDOUT: !177 = !DIFile(filename: "call_impl.carbon", directory: "")
// CHECK:STDOUT: !178 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !177, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !179 = distinct !DISubprogram(name: "F", linkageName: "_CF:thunk:Base.Main:Derived.Main", scope: null, file: !177, line: 10, type: !22, spFlags: DISPFlagDefinition, unit: !178, retainedNodes: !183)
// CHECK:STDOUT: !180 = !DILocalVariable(arg: 1, scope: !179, type: !20)
// CHECK:STDOUT: !182 = !DILocation(line: 10, column: 3, scope: !179)
// CHECK:STDOUT: !183 = !{!180}
// CHECK:STDOUT: !184 = distinct !DISubprogram(name: "Use", linkageName: "_CUse.Main", scope: null, file: !177, line: 13, type: !41, spFlags: DISPFlagDefinition, unit: !178)
// CHECK:STDOUT: !185 = !DILocation(line: 14, column: 3, scope: !184)
// CHECK:STDOUT: !186 = !DILocation(line: 14, column: 21, scope: !184)
// CHECK:STDOUT: !187 = !DILocation(line: 14, column: 30, scope: !184)
// CHECK:STDOUT: !189 = !DILocation(line: 15, column: 3, scope: !184)
// CHECK:STDOUT: !190 = !DILocation(line: 13, column: 1, scope: !184)
// CHECK:STDOUT: !191 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.d81ad81840dd2e3c", scope: null, file: !10, line: 30, type: !22, spFlags: DISPFlagDefinition, unit: !178, retainedNodes: !197)
// CHECK:STDOUT: !192 = !DILocalVariable(arg: 1, scope: !191, type: !20)
// CHECK:STDOUT: !196 = !DILocation(line: 30, column: 1, scope: !191)
// CHECK:STDOUT: !197 = !{!192}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'generic_noop.carbon'
// CHECK:STDOUT: source_filename = "generic_noop.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!199}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !198 = !DIFile(filename: "generic_noop.carbon", directory: "")
// CHECK:STDOUT: !199 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !198, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'generic_use.carbon'
// CHECK:STDOUT: source_filename = "generic_use.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @"_CBase.Main.$vtable.2c02d4d6b54f9d0d" = unnamed_addr constant [1 x i32] [i32 trunc (i64 sub (i64 ptrtoint (ptr @_CF.Base.Main.2c02d4d6b54f9d0d to i64), i64 ptrtoint (ptr @"_CBase.Main.$vtable.2c02d4d6b54f9d0d" to i64)) to i32)]
// CHECK:STDOUT: @"_CBase.Main.$vtable.9b8326f304a33dd3" = unnamed_addr constant [1 x i32] [i32 trunc (i64 sub (i64 ptrtoint (ptr @_CF.Base.Main.9b8326f304a33dd3 to i64), i64 ptrtoint (ptr @"_CBase.Main.$vtable.9b8326f304a33dd3" to i64)) to i32)]
// CHECK:STDOUT: @Base.val.d42.loc16_3.1 = internal constant { ptr } { ptr @"_CBase.Main.$vtable.2c02d4d6b54f9d0d" }
// CHECK:STDOUT: @Base.val.52a.loc17_3.1 = internal constant { ptr } { ptr @"_CBase.Main.$vtable.9b8326f304a33dd3" }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !202 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var.loc16 = alloca { ptr }, align 8, !dbg !203
// CHECK:STDOUT:   %_.var.loc17 = alloca { ptr }, align 8, !dbg !206
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc16), !dbg !203
// CHECK:STDOUT:   %.loc16_22.2.vptr = getelementptr inbounds nuw { ptr }, ptr %_.var.loc16, i32 0, i32 0, !dbg !204
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 %_.var.loc16, ptr align 8 @Base.val.d42.loc16_3.1, i64 8, i1 false), !dbg !203
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc17), !dbg !206
// CHECK:STDOUT:   %.loc17_22.2.vptr = getelementptr inbounds nuw { ptr }, ptr %_.var.loc17, i32 0, i32 0, !dbg !207
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 %_.var.loc17, ptr align 8 @Base.val.52a.loc17_3.1, i64 8, i1 false), !dbg !206
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.613a848e97ed9e84(ptr %_.var.loc17), !dbg !206
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.300593fdd4019f45(ptr %_.var.loc16), !dbg !203
// CHECK:STDOUT:   ret void, !dbg !209
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Base.Main.2c02d4d6b54f9d0d(ptr %self) #0 !dbg !210 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca {}, align 1, !dbg !213
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !213
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.2c02d4d6b54f9d0d(ptr %_.var), !dbg !213
// CHECK:STDOUT:   ret void, !dbg !215
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Base.Main.9b8326f304a33dd3(ptr %self) #0 !dbg !217 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca { {} }, align 1, !dbg !220
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !220
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.9b8326f304a33dd3(ptr %_.var), !dbg !220
// CHECK:STDOUT:   ret void, !dbg !222
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.613a848e97ed9e84(ptr %t) #0 !dbg !224 {
// CHECK:STDOUT:   ret void, !dbg !229
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.300593fdd4019f45(ptr %t) #0 !dbg !231 {
// CHECK:STDOUT:   ret void, !dbg !236
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.2c02d4d6b54f9d0d(ptr %t) #0 !dbg !238 {
// CHECK:STDOUT:   ret void, !dbg !243
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.9b8326f304a33dd3(ptr %t) #0 !dbg !245 {
// CHECK:STDOUT:   ret void, !dbg !250
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!201}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !10 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !20 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !21 = !{null, !20}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !40 = !{null}
// CHECK:STDOUT: !41 = !DISubroutineType(types: !40)
// CHECK:STDOUT: !200 = !DIFile(filename: "generic_use.carbon", directory: "")
// CHECK:STDOUT: !201 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !200, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !202 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !200, line: 15, type: !41, spFlags: DISPFlagDefinition, unit: !201)
// CHECK:STDOUT: !203 = !DILocation(line: 16, column: 3, scope: !202)
// CHECK:STDOUT: !204 = !DILocation(line: 16, column: 21, scope: !202)
// CHECK:STDOUT: !206 = !DILocation(line: 17, column: 3, scope: !202)
// CHECK:STDOUT: !207 = !DILocation(line: 17, column: 21, scope: !202)
// CHECK:STDOUT: !209 = !DILocation(line: 15, column: 1, scope: !202)
// CHECK:STDOUT: !210 = distinct !DISubprogram(name: "F", linkageName: "_CF.Base.Main.2c02d4d6b54f9d0d", scope: null, file: !200, line: 5, type: !22, spFlags: DISPFlagDefinition, unit: !201, retainedNodes: !216)
// CHECK:STDOUT: !211 = !DILocalVariable(arg: 1, scope: !210, type: !20)
// CHECK:STDOUT: !213 = !DILocation(line: 6, column: 5, scope: !210)
// CHECK:STDOUT: !215 = !DILocation(line: 5, column: 3, scope: !210)
// CHECK:STDOUT: !216 = !{!211}
// CHECK:STDOUT: !217 = distinct !DISubprogram(name: "F", linkageName: "_CF.Base.Main.9b8326f304a33dd3", scope: null, file: !200, line: 5, type: !22, spFlags: DISPFlagDefinition, unit: !201, retainedNodes: !223)
// CHECK:STDOUT: !218 = !DILocalVariable(arg: 1, scope: !217, type: !20)
// CHECK:STDOUT: !220 = !DILocation(line: 6, column: 5, scope: !217)
// CHECK:STDOUT: !222 = !DILocation(line: 5, column: 3, scope: !217)
// CHECK:STDOUT: !223 = !{!218}
// CHECK:STDOUT: !224 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.613a848e97ed9e84", scope: null, file: !10, line: 30, type: !22, spFlags: DISPFlagDefinition, unit: !201, retainedNodes: !230)
// CHECK:STDOUT: !225 = !DILocalVariable(arg: 1, scope: !224, type: !20)
// CHECK:STDOUT: !229 = !DILocation(line: 30, column: 1, scope: !224)
// CHECK:STDOUT: !230 = !{!225}
// CHECK:STDOUT: !231 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.300593fdd4019f45", scope: null, file: !10, line: 30, type: !22, spFlags: DISPFlagDefinition, unit: !201, retainedNodes: !237)
// CHECK:STDOUT: !232 = !DILocalVariable(arg: 1, scope: !231, type: !20)
// CHECK:STDOUT: !236 = !DILocation(line: 30, column: 1, scope: !231)
// CHECK:STDOUT: !237 = !{!232}
// CHECK:STDOUT: !238 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.2c02d4d6b54f9d0d", scope: null, file: !10, line: 30, type: !22, spFlags: DISPFlagDefinition, unit: !201, retainedNodes: !244)
// CHECK:STDOUT: !239 = !DILocalVariable(arg: 1, scope: !238, type: !20)
// CHECK:STDOUT: !243 = !DILocation(line: 30, column: 1, scope: !238)
// CHECK:STDOUT: !244 = !{!239}
// CHECK:STDOUT: !245 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.9b8326f304a33dd3", scope: null, file: !10, line: 30, type: !22, spFlags: DISPFlagDefinition, unit: !201, retainedNodes: !251)
// CHECK:STDOUT: !246 = !DILocalVariable(arg: 1, scope: !245, type: !20)
// CHECK:STDOUT: !250 = !DILocation(line: 30, column: 1, scope: !245)
// CHECK:STDOUT: !251 = !{!246}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'generic_base.carbon'
// CHECK:STDOUT: source_filename = "generic_base.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @"_CDerived.Main.$vtable" = unnamed_addr constant [1 x i32] [i32 trunc (i64 sub (i64 ptrtoint (ptr @_CF.Base.Main.6d6a4a2512463e59 to i64), i64 ptrtoint (ptr @"_CDerived.Main.$vtable" to i64)) to i32)]
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMake.Main() #0 !dbg !254 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca { { ptr } }, align 8, !dbg !255
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !255
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.d81ad81840dd2e3c(ptr %_.var), !dbg !255
// CHECK:STDOUT:   ret void, !dbg !257
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Base.Main.6d6a4a2512463e59(ptr %self) #0 !dbg !258 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !261
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.d81ad81840dd2e3c(ptr %t) #0 !dbg !263 {
// CHECK:STDOUT:   ret void, !dbg !268
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!253}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !10 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !20 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !21 = !{null, !20}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !40 = !{null}
// CHECK:STDOUT: !41 = !DISubroutineType(types: !40)
// CHECK:STDOUT: !252 = !DIFile(filename: "generic_base.carbon", directory: "")
// CHECK:STDOUT: !253 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !252, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !254 = distinct !DISubprogram(name: "Make", linkageName: "_CMake.Main", scope: null, file: !252, line: 14, type: !41, spFlags: DISPFlagDefinition, unit: !253)
// CHECK:STDOUT: !255 = !DILocation(line: 15, column: 3, scope: !254)
// CHECK:STDOUT: !257 = !DILocation(line: 14, column: 1, scope: !254)
// CHECK:STDOUT: !258 = distinct !DISubprogram(name: "F", linkageName: "_CF.Base.Main.6d6a4a2512463e59", scope: null, file: !252, line: 5, type: !22, spFlags: DISPFlagDefinition, unit: !253, retainedNodes: !262)
// CHECK:STDOUT: !259 = !DILocalVariable(arg: 1, scope: !258, type: !20)
// CHECK:STDOUT: !261 = !DILocation(line: 5, column: 3, scope: !258)
// CHECK:STDOUT: !262 = !{!259}
// CHECK:STDOUT: !263 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.d81ad81840dd2e3c", scope: null, file: !10, line: 30, type: !22, spFlags: DISPFlagDefinition, unit: !253, retainedNodes: !269)
// CHECK:STDOUT: !264 = !DILocalVariable(arg: 1, scope: !263, type: !20)
// CHECK:STDOUT: !268 = !DILocation(line: 30, column: 1, scope: !263)
// CHECK:STDOUT: !269 = !{!264}
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
// CHECK:STDOUT: define void @_CF.Derived.Main(ptr sret({}) %return, ptr %self, ptr %n) #0 !dbg !276 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %return, ptr align 1 @From.val.loc18_14, i64 0, i1 false), !dbg !283
// CHECK:STDOUT:   ret void, !dbg !283
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_CF:thunk:Base.Main:Derived.Main"(ptr sret({}) %return, ptr %self, ptr %n) #1 !dbg !285 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc17_52.2.temp = alloca {}, align 1, !dbg !289
// CHECK:STDOUT:   %.5.temp = alloca {}, align 1, !dbg !288
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc17_52.2.temp), !dbg !289
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.5.temp), !dbg !288
// CHECK:STDOUT:   call void @"_CConvert.From.Main:ImplicitAs.a3bb440f6426a7d6.Core"(ptr %.5.temp, ptr %n), !dbg !288
// CHECK:STDOUT:   call void @_CF.Derived.Main(ptr %.loc17_52.2.temp, ptr %self, ptr %.5.temp), !dbg !289
// CHECK:STDOUT:   call void @"_CConvert.From.Main:ImplicitAs.a3bb440f6426a7d6.Core"(ptr %return, ptr %.loc17_52.2.temp), !dbg !289
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.3f61691f292a1f88(ptr %.loc17_52.2.temp), !dbg !289
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.36c1478c1cb9545e(ptr %.5.temp), !dbg !288
// CHECK:STDOUT:   ret void, !dbg !289
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CUse.Main() #0 !dbg !291 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %d.var = alloca { { ptr } }, align 8, !dbg !292
// CHECK:STDOUT:   %.loc25_10.1.temp = alloca {}, align 1, !dbg !300
// CHECK:STDOUT:   %.loc25_9.2.temp = alloca {}, align 1, !dbg !301
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %d.var), !dbg !292
// CHECK:STDOUT:   %.loc23_31.2.base = getelementptr inbounds nuw { { ptr } }, ptr %d.var, i32 0, i32 0, !dbg !293
// CHECK:STDOUT:   %.loc23_30.2.vptr = getelementptr inbounds nuw { ptr }, ptr %.loc23_31.2.base, i32 0, i32 0, !dbg !294
// CHECK:STDOUT:   %.loc23_31.6.base = getelementptr inbounds nuw { { ptr } }, ptr %d.var, i32 0, i32 0, !dbg !293
// CHECK:STDOUT:   %.loc23_31.7.vptr = getelementptr inbounds nuw { ptr }, ptr %.loc23_31.6.base, i32 0, i32 0, !dbg !293
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 %d.var, ptr align 8 @Derived.val.loc23_31.5, i64 8, i1 false), !dbg !293
// CHECK:STDOUT:   store ptr @"_CDerived.Main.$vtable", ptr %.loc23_31.7.vptr, align 8, !dbg !293
// CHECK:STDOUT:   %.loc24_18.2.base = getelementptr inbounds nuw { { ptr } }, ptr %d.var, i32 0, i32 0, !dbg !297
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc25_10.1.temp), !dbg !300
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc25_9.2.temp), !dbg !301
// CHECK:STDOUT:   %Base.F.call.vtable = load ptr, ptr %.loc25_10.1.temp, align 8, !dbg !300
// CHECK:STDOUT:   %Base.F.call = call ptr @llvm.load.relative.i32(ptr %Base.F.call.vtable, i32 0), !dbg !300
// CHECK:STDOUT:   call void %Base.F.call(ptr %.loc25_10.1.temp, ptr %.loc24_18.2.base, ptr @From.val.loc18_14), !dbg !300
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.36c1478c1cb9545e(ptr %.loc25_10.1.temp), !dbg !300
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.3f61691f292a1f88(ptr @From.val), !dbg !301
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.d81ad81840dd2e3c(ptr %d.var), !dbg !292
// CHECK:STDOUT:   ret void, !dbg !302
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.3f61691f292a1f88(ptr %t) #0 !dbg !303 {
// CHECK:STDOUT:   ret void, !dbg !308
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.36c1478c1cb9545e(ptr %t) #0 !dbg !310 {
// CHECK:STDOUT:   ret void, !dbg !315
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: read)
// CHECK:STDOUT: declare ptr @llvm.load.relative.i32(ptr, i32) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.d81ad81840dd2e3c(ptr %t) #0 !dbg !317 {
// CHECK:STDOUT:   ret void, !dbg !322
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 4, 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.3f61691f292a1f88, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.36c1478c1cb9545e, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline nounwind }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #3 = { nocallback nofree nosync nounwind willreturn memory(argmem: read) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!271}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !10 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !20 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !21 = !{null, !20}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !40 = !{null}
// CHECK:STDOUT: !41 = !DISubroutineType(types: !40)
// CHECK:STDOUT: !270 = !DIFile(filename: "thunk.carbon", directory: "")
// CHECK:STDOUT: !271 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !270, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !274 = !{!20, !20, !20}
// CHECK:STDOUT: !275 = !DISubroutineType(types: !274)
// CHECK:STDOUT: !276 = distinct !DISubprogram(name: "F", linkageName: "_CF.Derived.Main", scope: null, file: !270, line: 17, type: !275, spFlags: DISPFlagDefinition, unit: !271, retainedNodes: !284)
// CHECK:STDOUT: !277 = !DILocalVariable(arg: 1, scope: !276, type: !20)
// CHECK:STDOUT: !278 = !DILocalVariable(arg: 2, scope: !276, type: !20)
// CHECK:STDOUT: !283 = !DILocation(line: 18, column: 5, scope: !276)
// CHECK:STDOUT: !284 = !{!277, !278}
// CHECK:STDOUT: !285 = distinct !DISubprogram(name: "F", linkageName: "_CF:thunk:Base.Main:Derived.Main", scope: null, file: !270, line: 17, type: !275, spFlags: DISPFlagDefinition, unit: !271, retainedNodes: !290)
// CHECK:STDOUT: !286 = !DILocalVariable(arg: 1, scope: !285, type: !20)
// CHECK:STDOUT: !287 = !DILocalVariable(arg: 2, scope: !285, type: !20)
// CHECK:STDOUT: !288 = !DILocation(line: 0, scope: !285)
// CHECK:STDOUT: !289 = !DILocation(line: 17, column: 3, scope: !285)
// CHECK:STDOUT: !290 = !{!286, !287}
// CHECK:STDOUT: !291 = distinct !DISubprogram(name: "Use", linkageName: "_CUse.Main", scope: null, file: !270, line: 22, type: !41, spFlags: DISPFlagDefinition, unit: !271)
// CHECK:STDOUT: !292 = !DILocation(line: 23, column: 3, scope: !291)
// CHECK:STDOUT: !293 = !DILocation(line: 23, column: 20, scope: !291)
// CHECK:STDOUT: !294 = !DILocation(line: 23, column: 29, scope: !291)
// CHECK:STDOUT: !297 = !DILocation(line: 24, column: 18, scope: !291)
// CHECK:STDOUT: !300 = !DILocation(line: 25, column: 3, scope: !291)
// CHECK:STDOUT: !301 = !DILocation(line: 25, column: 8, scope: !291)
// CHECK:STDOUT: !302 = !DILocation(line: 22, column: 1, scope: !291)
// CHECK:STDOUT: !303 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.3f61691f292a1f88", scope: null, file: !10, line: 30, type: !22, spFlags: DISPFlagDefinition, unit: !271, retainedNodes: !309)
// CHECK:STDOUT: !304 = !DILocalVariable(arg: 1, scope: !303, type: !20)
// CHECK:STDOUT: !308 = !DILocation(line: 30, column: 1, scope: !303)
// CHECK:STDOUT: !309 = !{!304}
// CHECK:STDOUT: !310 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.36c1478c1cb9545e", scope: null, file: !10, line: 30, type: !22, spFlags: DISPFlagDefinition, unit: !271, retainedNodes: !316)
// CHECK:STDOUT: !311 = !DILocalVariable(arg: 1, scope: !310, type: !20)
// CHECK:STDOUT: !315 = !DILocation(line: 30, column: 1, scope: !310)
// CHECK:STDOUT: !316 = !{!311}
// CHECK:STDOUT: !317 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.d81ad81840dd2e3c", scope: null, file: !10, line: 30, type: !22, spFlags: DISPFlagDefinition, unit: !271, retainedNodes: !323)
// CHECK:STDOUT: !318 = !DILocalVariable(arg: 1, scope: !317, type: !20)
// CHECK:STDOUT: !322 = !DILocation(line: 30, column: 1, scope: !317)
// CHECK:STDOUT: !323 = !{!318}
// CHECK:STDOUT:
