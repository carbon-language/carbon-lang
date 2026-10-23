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
// CHECK:STDOUT: define weak_odr void @"_COp.f7b08e0f33b30342:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !55 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc12_3.1.base = getelementptr inbounds nuw { ptr, {} }, ptr %self, i32 0, i32 1, !dbg !57
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.9dd568d598bdbcbd(ptr %.loc12_3.1.base), !dbg !57
// CHECK:STDOUT:   %.loc12_3.4.vptr = getelementptr inbounds nuw { ptr, {} }, ptr %self, i32 0, i32 0, !dbg !57
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.648208d69a8e7988(ptr %.loc12_3.4.vptr), !dbg !57
// CHECK:STDOUT:   ret void, !dbg !57
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.4487e305711510e9:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !59 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc12_3.1.base = getelementptr inbounds nuw { ptr, {} }, ptr %self, i32 0, i32 1, !dbg !61
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.9dd568d598bdbcbd(ptr %.loc12_3.1.base), !dbg !61
// CHECK:STDOUT:   ret void, !dbg !61
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.3c3534e993db7939:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !63 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc12_3.1.base = getelementptr inbounds nuw { { ptr, {} } }, ptr %self, i32 0, i32 0, !dbg !65
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.36e476cc7e22676a(ptr %.loc12_3.1.base), !dbg !65
// CHECK:STDOUT:   ret void, !dbg !65
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.43ed5d0e08fc7556:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !67 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc12_3.1.base = getelementptr inbounds nuw { { ptr, {} } }, ptr %self, i32 0, i32 0, !dbg !69
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.36e476cc7e22676a(ptr %.loc12_3.1.base), !dbg !69
// CHECK:STDOUT:   ret void, !dbg !69
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CUse.Create(ptr %v) #0 !dbg !71 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Intermediate.Fn.call.vtable = load ptr, ptr %v, align 8, !dbg !74
// CHECK:STDOUT:   %Intermediate.Fn.call = call ptr @llvm.load.relative.i32(ptr %Intermediate.Fn.call.vtable, i32 0), !dbg !74
// CHECK:STDOUT:   call void %Intermediate.Fn.call(ptr %v), !dbg !74
// CHECK:STDOUT:   ret void, !dbg !75
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.4ebe230b4b7eb612(ptr %t) #0 !dbg !77 {
// CHECK:STDOUT:   call void @"_COp.43ed5d0e08fc7556:core.SubobjectDestroy.Core"(ptr %t), !dbg !81
// CHECK:STDOUT:   ret void, !dbg !82
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.36e476cc7e22676a(ptr %t) #0 !dbg !84 {
// CHECK:STDOUT:   call void @"_COp.4487e305711510e9:core.SubobjectDestroy.Core"(ptr %t), !dbg !88
// CHECK:STDOUT:   ret void, !dbg !89
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.9dd568d598bdbcbd(ptr %t) #0 !dbg !91 {
// CHECK:STDOUT:   ret void, !dbg !96
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.648208d69a8e7988(ptr %t) #0 !dbg !98 {
// CHECK:STDOUT:   ret void, !dbg !103
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: read)
// CHECK:STDOUT: declare ptr @llvm.load.relative.i32(ptr, i32) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.4ebe230b4b7eb612, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.36e476cc7e22676a, { 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.9dd568d598bdbcbd, { 2, 1, 0 }
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
// CHECK:STDOUT: !55 = distinct !DISubprogram(name: "Op", linkageName: "_COp.f7b08e0f33b30342:core.SubobjectDestroy.Core", scope: null, file: !38, line: 12, type: !22, spFlags: DISPFlagDefinition, unit: !39, retainedNodes: !58)
// CHECK:STDOUT: !56 = !DILocalVariable(arg: 1, scope: !55, type: !20)
// CHECK:STDOUT: !57 = !DILocation(line: 12, column: 3, scope: !55)
// CHECK:STDOUT: !58 = !{!56}
// CHECK:STDOUT: !59 = distinct !DISubprogram(name: "Op", linkageName: "_COp.4487e305711510e9:core.SubobjectDestroy.Core", scope: null, file: !38, line: 12, type: !22, spFlags: DISPFlagDefinition, unit: !39, retainedNodes: !62)
// CHECK:STDOUT: !60 = !DILocalVariable(arg: 1, scope: !59, type: !20)
// CHECK:STDOUT: !61 = !DILocation(line: 12, column: 3, scope: !59)
// CHECK:STDOUT: !62 = !{!60}
// CHECK:STDOUT: !63 = distinct !DISubprogram(name: "Op", linkageName: "_COp.3c3534e993db7939:core.SubobjectDestroy.Core", scope: null, file: !38, line: 12, type: !22, spFlags: DISPFlagDefinition, unit: !39, retainedNodes: !66)
// CHECK:STDOUT: !64 = !DILocalVariable(arg: 1, scope: !63, type: !20)
// CHECK:STDOUT: !65 = !DILocation(line: 12, column: 3, scope: !63)
// CHECK:STDOUT: !66 = !{!64}
// CHECK:STDOUT: !67 = distinct !DISubprogram(name: "Op", linkageName: "_COp.43ed5d0e08fc7556:core.SubobjectDestroy.Core", scope: null, file: !38, line: 12, type: !22, spFlags: DISPFlagDefinition, unit: !39, retainedNodes: !70)
// CHECK:STDOUT: !68 = !DILocalVariable(arg: 1, scope: !67, type: !20)
// CHECK:STDOUT: !69 = !DILocation(line: 12, column: 3, scope: !67)
// CHECK:STDOUT: !70 = !{!68}
// CHECK:STDOUT: !71 = distinct !DISubprogram(name: "Use", linkageName: "_CUse.Create", scope: null, file: !38, line: 15, type: !22, spFlags: DISPFlagDefinition, unit: !39, retainedNodes: !76)
// CHECK:STDOUT: !72 = !DILocalVariable(arg: 1, scope: !71, type: !20)
// CHECK:STDOUT: !74 = !DILocation(line: 16, column: 3, scope: !71)
// CHECK:STDOUT: !75 = !DILocation(line: 15, column: 1, scope: !71)
// CHECK:STDOUT: !76 = !{!72}
// CHECK:STDOUT: !77 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.4ebe230b4b7eb612", scope: null, file: !10, line: 30, type: !22, spFlags: DISPFlagDefinition, unit: !39, retainedNodes: !83)
// CHECK:STDOUT: !78 = !DILocalVariable(arg: 1, scope: !77, type: !20)
// CHECK:STDOUT: !81 = !DILocation(line: 32, column: 5, scope: !77)
// CHECK:STDOUT: !82 = !DILocation(line: 30, column: 1, scope: !77)
// CHECK:STDOUT: !83 = !{!78}
// CHECK:STDOUT: !84 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.36e476cc7e22676a", scope: null, file: !10, line: 30, type: !22, spFlags: DISPFlagDefinition, unit: !39, retainedNodes: !90)
// CHECK:STDOUT: !85 = !DILocalVariable(arg: 1, scope: !84, type: !20)
// CHECK:STDOUT: !88 = !DILocation(line: 32, column: 5, scope: !84)
// CHECK:STDOUT: !89 = !DILocation(line: 30, column: 1, scope: !84)
// CHECK:STDOUT: !90 = !{!85}
// CHECK:STDOUT: !91 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.9dd568d598bdbcbd", scope: null, file: !10, line: 30, type: !22, spFlags: DISPFlagDefinition, unit: !39, retainedNodes: !97)
// CHECK:STDOUT: !92 = !DILocalVariable(arg: 1, scope: !91, type: !20)
// CHECK:STDOUT: !96 = !DILocation(line: 30, column: 1, scope: !91)
// CHECK:STDOUT: !97 = !{!92}
// CHECK:STDOUT: !98 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.648208d69a8e7988", scope: null, file: !10, line: 30, type: !22, spFlags: DISPFlagDefinition, unit: !39, retainedNodes: !104)
// CHECK:STDOUT: !99 = !DILocalVariable(arg: 1, scope: !98, type: !20)
// CHECK:STDOUT: !103 = !DILocation(line: 30, column: 1, scope: !98)
// CHECK:STDOUT: !104 = !{!99}
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
// CHECK:STDOUT: define void @_CMakePartialBase.Main(ptr sret({}) %return) #0 !dbg !109 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %return, ptr align 1 @empty_struct.ac8.loc7_12, i64 0, i1 false), !dbg !112
// CHECK:STDOUT:   ret void, !dbg !112
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMakePartialIntermediate.Main(ptr sret({ ptr, {} }) %return) #0 !dbg !113 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc11_36.1.base = getelementptr inbounds nuw { ptr, {} }, ptr %return, i32 0, i32 1, !dbg !116
// CHECK:STDOUT:   call void @_CMakePartialBase.Main(ptr %.loc11_36.1.base), !dbg !117
// CHECK:STDOUT:   %.loc11_36.3.vptr = getelementptr inbounds nuw { ptr, {} }, ptr %return, i32 0, i32 0, !dbg !116
// CHECK:STDOUT:   store ptr poison, ptr %.loc11_36.3.vptr, align 8, !dbg !116
// CHECK:STDOUT:   ret void, !dbg !118
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMakeBase.Main(ptr sret({}) %return) #0 !dbg !119 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CMakePartialBase.Main(ptr %return), !dbg !122
// CHECK:STDOUT:   ret void, !dbg !123
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMakeIntermediate.Main(ptr sret({ ptr, {} }) %return) #0 !dbg !124 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CMakePartialIntermediate.Main(ptr %return), !dbg !127
// CHECK:STDOUT:   %.loc19_34.1.vptr = getelementptr inbounds nuw { ptr, {} }, ptr %return, i32 0, i32 0, !dbg !127
// CHECK:STDOUT:   store ptr @"_CIntermediate.Classes.$vtable", ptr %.loc19_34.1.vptr, align 8, !dbg !127
// CHECK:STDOUT:   ret void, !dbg !128
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CFn.Intermediate.Classes(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMakeDerivedFromPartial.Main(ptr sret({ { ptr, {} } }) %return) #0 !dbg !129 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc23_44.1.base = getelementptr inbounds nuw { { ptr, {} } }, ptr %return, i32 0, i32 0, !dbg !132
// CHECK:STDOUT:   call void @_CMakePartialIntermediate.Main(ptr %.loc23_44.1.base), !dbg !133
// CHECK:STDOUT:   %.loc23_44.5.base = getelementptr inbounds nuw { { ptr, {} } }, ptr %return, i32 0, i32 0, !dbg !132
// CHECK:STDOUT:   %.loc23_44.6.vptr = getelementptr inbounds nuw { ptr, {} }, ptr %.loc23_44.5.base, i32 0, i32 0, !dbg !132
// CHECK:STDOUT:   store ptr @"_CDerived.Classes.$vtable", ptr %.loc23_44.6.vptr, align 8, !dbg !132
// CHECK:STDOUT:   ret void, !dbg !134
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_CFn:thunk:Intermediate.Classes:Derived.Classes"(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMakeDerivedFromNonPartial.Main(ptr sret({ { ptr, {} } }) %return) #0 !dbg !135 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc27_37.1.base = getelementptr inbounds nuw { { ptr, {} } }, ptr %return, i32 0, i32 0, !dbg !138
// CHECK:STDOUT:   call void @_CMakeIntermediate.Main(ptr %.loc27_37.1.base), !dbg !139
// CHECK:STDOUT:   %.loc27_37.7.base = getelementptr inbounds nuw { { ptr, {} } }, ptr %return, i32 0, i32 0, !dbg !138
// CHECK:STDOUT:   %.loc27_37.8.vptr = getelementptr inbounds nuw { ptr, {} }, ptr %.loc27_37.7.base, i32 0, i32 0, !dbg !138
// CHECK:STDOUT:   store ptr @"_CDerived.Classes.$vtable", ptr %.loc27_37.8.vptr, align 8, !dbg !138
// CHECK:STDOUT:   ret void, !dbg !140
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!106}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !105 = !DIFile(filename: "create_named_ctor.carbon", directory: "")
// CHECK:STDOUT: !106 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !105, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !107 = !{!20}
// CHECK:STDOUT: !108 = !DISubroutineType(types: !107)
// CHECK:STDOUT: !109 = distinct !DISubprogram(name: "MakePartialBase", linkageName: "_CMakePartialBase.Main", scope: null, file: !105, line: 6, type: !108, spFlags: DISPFlagDefinition, unit: !106)
// CHECK:STDOUT: !112 = !DILocation(line: 7, column: 3, scope: !109)
// CHECK:STDOUT: !113 = distinct !DISubprogram(name: "MakePartialIntermediate", linkageName: "_CMakePartialIntermediate.Main", scope: null, file: !105, line: 10, type: !108, spFlags: DISPFlagDefinition, unit: !106)
// CHECK:STDOUT: !116 = !DILocation(line: 11, column: 10, scope: !113)
// CHECK:STDOUT: !117 = !DILocation(line: 11, column: 19, scope: !113)
// CHECK:STDOUT: !118 = !DILocation(line: 11, column: 3, scope: !113)
// CHECK:STDOUT: !119 = distinct !DISubprogram(name: "MakeBase", linkageName: "_CMakeBase.Main", scope: null, file: !105, line: 14, type: !108, spFlags: DISPFlagDefinition, unit: !106)
// CHECK:STDOUT: !122 = !DILocation(line: 15, column: 10, scope: !119)
// CHECK:STDOUT: !123 = !DILocation(line: 15, column: 3, scope: !119)
// CHECK:STDOUT: !124 = distinct !DISubprogram(name: "MakeIntermediate", linkageName: "_CMakeIntermediate.Main", scope: null, file: !105, line: 18, type: !108, spFlags: DISPFlagDefinition, unit: !106)
// CHECK:STDOUT: !127 = !DILocation(line: 19, column: 10, scope: !124)
// CHECK:STDOUT: !128 = !DILocation(line: 19, column: 3, scope: !124)
// CHECK:STDOUT: !129 = distinct !DISubprogram(name: "MakeDerivedFromPartial", linkageName: "_CMakeDerivedFromPartial.Main", scope: null, file: !105, line: 22, type: !108, spFlags: DISPFlagDefinition, unit: !106)
// CHECK:STDOUT: !132 = !DILocation(line: 23, column: 10, scope: !129)
// CHECK:STDOUT: !133 = !DILocation(line: 23, column: 19, scope: !129)
// CHECK:STDOUT: !134 = !DILocation(line: 23, column: 3, scope: !129)
// CHECK:STDOUT: !135 = distinct !DISubprogram(name: "MakeDerivedFromNonPartial", linkageName: "_CMakeDerivedFromNonPartial.Main", scope: null, file: !105, line: 26, type: !108, spFlags: DISPFlagDefinition, unit: !106)
// CHECK:STDOUT: !138 = !DILocation(line: 27, column: 10, scope: !135)
// CHECK:STDOUT: !139 = !DILocation(line: 27, column: 19, scope: !135)
// CHECK:STDOUT: !140 = !DILocation(line: 27, column: 3, scope: !135)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'member_init.carbon'
// CHECK:STDOUT: source_filename = "member_init.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @"_CBase.MemberInit.$vtable" = unnamed_addr constant [1 x i32] [i32 trunc (i64 sub (i64 ptrtoint (ptr @_CFn.Base.MemberInit to i64), i64 ptrtoint (ptr @"_CBase.MemberInit.$vtable" to i64)) to i32)]
// CHECK:STDOUT: @Base.val.loc13_3.1 = internal constant <{ ptr, i32 }> <{ ptr @"_CBase.MemberInit.$vtable", i32 3 }>
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CFn.Base.MemberInit(ptr %self) #0 !dbg !144 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !147
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CFn.MemberInit() #0 !dbg !149 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %i.var = alloca i32, align 4, !dbg !150
// CHECK:STDOUT:   %v.var = alloca <{ ptr, i32 }>, align 8, !dbg !152
// CHECK:STDOUT:   %_.var = alloca <{ ptr, i32 }>, align 8, !dbg !157
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %i.var), !dbg !150
// CHECK:STDOUT:   store i32 3, ptr %i.var, align 4, !dbg !150
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %v.var), !dbg !152
// CHECK:STDOUT:   %.loc11_24.2.vptr = getelementptr inbounds nuw <{ ptr, i32 }>, ptr %v.var, i32 0, i32 0, !dbg !154
// CHECK:STDOUT:   %.loc11_23 = load i32, ptr %i.var, align 4, !dbg !153
// CHECK:STDOUT:   %.loc11_24.4.m = getelementptr inbounds nuw <{ ptr, i32 }>, ptr %v.var, i32 0, i32 1, !dbg !154
// CHECK:STDOUT:   store i32 %.loc11_23, ptr %.loc11_24.4.m, align 4, !dbg !154
// CHECK:STDOUT:   store ptr @"_CBase.MemberInit.$vtable", ptr %.loc11_24.2.vptr, align 8, !dbg !154
// CHECK:STDOUT:   %.loc12_4.m = getelementptr inbounds nuw <{ ptr, i32 }>, ptr %v.var, i32 0, i32 1, !dbg !156
// CHECK:STDOUT:   store i32 5, ptr %.loc12_4.m, align 4, !dbg !156
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !157
// CHECK:STDOUT:   %.loc13_24.2.vptr = getelementptr inbounds nuw <{ ptr, i32 }>, ptr %_.var, i32 0, i32 0, !dbg !158
// CHECK:STDOUT:   %.loc13_24.5.m = getelementptr inbounds nuw <{ ptr, i32 }>, ptr %_.var, i32 0, i32 1, !dbg !158
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 %_.var, ptr align 8 @Base.val.loc13_3.1, i64 12, i1 false), !dbg !157
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.a9d732896496568f(ptr %_.var), !dbg !157
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.a9d732896496568f(ptr %v.var), !dbg !152
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.dc5658e0ac74b2aa(ptr %i.var), !dbg !150
// CHECK:STDOUT:   ret void, !dbg !162
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.9bdba0482390b229:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !163 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc13_3.1.m = getelementptr inbounds nuw <{ ptr, i32 }>, ptr %self, i32 0, i32 1, !dbg !165
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.dc5658e0ac74b2aa(ptr %.loc13_3.1.m), !dbg !165
// CHECK:STDOUT:   %.loc13_3.4.vptr = getelementptr inbounds nuw <{ ptr, i32 }>, ptr %self, i32 0, i32 0, !dbg !165
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.648208d69a8e7988(ptr %.loc13_3.4.vptr), !dbg !165
// CHECK:STDOUT:   ret void, !dbg !165
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.c6dbe0017b4c8d16:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !167 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc13_3.1.m = getelementptr inbounds nuw <{ ptr, i32 }>, ptr %self, i32 0, i32 1, !dbg !169
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.dc5658e0ac74b2aa(ptr %.loc13_3.1.m), !dbg !169
// CHECK:STDOUT:   ret void, !dbg !169
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.a9d732896496568f(ptr %t) #0 !dbg !171 {
// CHECK:STDOUT:   call void @"_COp.c6dbe0017b4c8d16:core.SubobjectDestroy.Core"(ptr %t), !dbg !175
// CHECK:STDOUT:   ret void, !dbg !176
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.dc5658e0ac74b2aa(ptr %t) #0 !dbg !178 {
// CHECK:STDOUT:   ret void, !dbg !183
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.648208d69a8e7988(ptr %t) #0 !dbg !185 {
// CHECK:STDOUT:   ret void, !dbg !190
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.a9d732896496568f, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.dc5658e0ac74b2aa, { 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!142}
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
// CHECK:STDOUT: !141 = !DIFile(filename: "member_init.carbon", directory: "")
// CHECK:STDOUT: !142 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !141, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !143 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !144 = distinct !DISubprogram(name: "Fn", linkageName: "_CFn.Base.MemberInit", scope: null, file: !141, line: 6, type: !22, spFlags: DISPFlagDefinition, unit: !142, retainedNodes: !148)
// CHECK:STDOUT: !145 = !DILocalVariable(arg: 1, scope: !144, type: !20)
// CHECK:STDOUT: !147 = !DILocation(line: 6, column: 3, scope: !144)
// CHECK:STDOUT: !148 = !{!145}
// CHECK:STDOUT: !149 = distinct !DISubprogram(name: "Fn", linkageName: "_CFn.MemberInit", scope: null, file: !141, line: 9, type: !41, spFlags: DISPFlagDefinition, unit: !142)
// CHECK:STDOUT: !150 = !DILocation(line: 10, column: 3, scope: !149)
// CHECK:STDOUT: !152 = !DILocation(line: 11, column: 3, scope: !149)
// CHECK:STDOUT: !153 = !DILocation(line: 11, column: 23, scope: !149)
// CHECK:STDOUT: !154 = !DILocation(line: 11, column: 17, scope: !149)
// CHECK:STDOUT: !156 = !DILocation(line: 12, column: 3, scope: !149)
// CHECK:STDOUT: !157 = !DILocation(line: 13, column: 3, scope: !149)
// CHECK:STDOUT: !158 = !DILocation(line: 13, column: 17, scope: !149)
// CHECK:STDOUT: !160 = !{null, !143}
// CHECK:STDOUT: !161 = !DISubroutineType(types: !160)
// CHECK:STDOUT: !162 = !DILocation(line: 9, column: 1, scope: !149)
// CHECK:STDOUT: !163 = distinct !DISubprogram(name: "Op", linkageName: "_COp.9bdba0482390b229:core.SubobjectDestroy.Core", scope: null, file: !141, line: 13, type: !22, spFlags: DISPFlagDefinition, unit: !142, retainedNodes: !166)
// CHECK:STDOUT: !164 = !DILocalVariable(arg: 1, scope: !163, type: !20)
// CHECK:STDOUT: !165 = !DILocation(line: 13, column: 3, scope: !163)
// CHECK:STDOUT: !166 = !{!164}
// CHECK:STDOUT: !167 = distinct !DISubprogram(name: "Op", linkageName: "_COp.c6dbe0017b4c8d16:core.SubobjectDestroy.Core", scope: null, file: !141, line: 13, type: !22, spFlags: DISPFlagDefinition, unit: !142, retainedNodes: !170)
// CHECK:STDOUT: !168 = !DILocalVariable(arg: 1, scope: !167, type: !20)
// CHECK:STDOUT: !169 = !DILocation(line: 13, column: 3, scope: !167)
// CHECK:STDOUT: !170 = !{!168}
// CHECK:STDOUT: !171 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.a9d732896496568f", scope: null, file: !10, line: 30, type: !22, spFlags: DISPFlagDefinition, unit: !142, retainedNodes: !177)
// CHECK:STDOUT: !172 = !DILocalVariable(arg: 1, scope: !171, type: !20)
// CHECK:STDOUT: !175 = !DILocation(line: 32, column: 5, scope: !171)
// CHECK:STDOUT: !176 = !DILocation(line: 30, column: 1, scope: !171)
// CHECK:STDOUT: !177 = !{!172}
// CHECK:STDOUT: !178 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.dc5658e0ac74b2aa", scope: null, file: !10, line: 30, type: !161, spFlags: DISPFlagDefinition, unit: !142, retainedNodes: !184)
// CHECK:STDOUT: !179 = !DILocalVariable(arg: 1, scope: !178, type: !143)
// CHECK:STDOUT: !183 = !DILocation(line: 30, column: 1, scope: !178)
// CHECK:STDOUT: !184 = !{!179}
// CHECK:STDOUT: !185 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.648208d69a8e7988", scope: null, file: !10, line: 30, type: !22, spFlags: DISPFlagDefinition, unit: !142, retainedNodes: !191)
// CHECK:STDOUT: !186 = !DILocalVariable(arg: 1, scope: !185, type: !20)
// CHECK:STDOUT: !190 = !DILocation(line: 30, column: 1, scope: !185)
// CHECK:STDOUT: !191 = !{!186}
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
// CHECK:STDOUT: define void @_CUse.Main() #0 !dbg !194 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca { { ptr } }, align 8, !dbg !195
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !195
// CHECK:STDOUT:   %.loc13_32.2.base = getelementptr inbounds nuw { { ptr } }, ptr %_.var, i32 0, i32 0, !dbg !196
// CHECK:STDOUT:   %.loc13_31.2.vptr = getelementptr inbounds nuw { ptr }, ptr %.loc13_32.2.base, i32 0, i32 0, !dbg !197
// CHECK:STDOUT:   %.loc13_32.6.base = getelementptr inbounds nuw { { ptr } }, ptr %_.var, i32 0, i32 0, !dbg !196
// CHECK:STDOUT:   %.loc13_32.7.vptr = getelementptr inbounds nuw { ptr }, ptr %.loc13_32.6.base, i32 0, i32 0, !dbg !196
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 %_.var, ptr align 8 @Derived.val.loc13_32.5, i64 8, i1 false), !dbg !196
// CHECK:STDOUT:   store ptr @"_CDerived.Main.$vtable", ptr %.loc13_32.7.vptr, align 8, !dbg !196
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.d81ad81840dd2e3c(ptr %_.var), !dbg !195
// CHECK:STDOUT:   ret void, !dbg !199
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.95528a86e1127e2d:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !200 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc13_3.1.vptr = getelementptr inbounds nuw { ptr }, ptr %self, i32 0, i32 0, !dbg !202
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.648208d69a8e7988(ptr %.loc13_3.1.vptr), !dbg !202
// CHECK:STDOUT:   ret void, !dbg !202
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.294df737cd074d60:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !204 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !206
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.caed8a96bf328d09:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !208 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc13_3.1.base = getelementptr inbounds nuw { { ptr } }, ptr %self, i32 0, i32 0, !dbg !210
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.8f745713e9c32bf6(ptr %.loc13_3.1.base), !dbg !210
// CHECK:STDOUT:   ret void, !dbg !210
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.b25525d21a9bd472:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !212 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc13_3.1.base = getelementptr inbounds nuw { { ptr } }, ptr %self, i32 0, i32 0, !dbg !214
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.8f745713e9c32bf6(ptr %.loc13_3.1.base), !dbg !214
// CHECK:STDOUT:   ret void, !dbg !214
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.d81ad81840dd2e3c(ptr %t) #0 !dbg !216 {
// CHECK:STDOUT:   call void @"_COp.b25525d21a9bd472:core.SubobjectDestroy.Core"(ptr %t), !dbg !220
// CHECK:STDOUT:   ret void, !dbg !221
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.648208d69a8e7988(ptr %t) #0 !dbg !223 {
// CHECK:STDOUT:   ret void, !dbg !228
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.8f745713e9c32bf6(ptr %t) #0 !dbg !230 {
// CHECK:STDOUT:   call void @"_COp.294df737cd074d60:core.SubobjectDestroy.Core"(ptr %t), !dbg !234
// CHECK:STDOUT:   ret void, !dbg !235
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.8f745713e9c32bf6, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!193}
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
// CHECK:STDOUT: !192 = !DIFile(filename: "member_brace_init.carbon", directory: "")
// CHECK:STDOUT: !193 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !192, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !194 = distinct !DISubprogram(name: "Use", linkageName: "_CUse.Main", scope: null, file: !192, line: 12, type: !41, spFlags: DISPFlagDefinition, unit: !193)
// CHECK:STDOUT: !195 = !DILocation(line: 13, column: 3, scope: !194)
// CHECK:STDOUT: !196 = !DILocation(line: 13, column: 21, scope: !194)
// CHECK:STDOUT: !197 = !DILocation(line: 13, column: 30, scope: !194)
// CHECK:STDOUT: !199 = !DILocation(line: 12, column: 1, scope: !194)
// CHECK:STDOUT: !200 = distinct !DISubprogram(name: "Op", linkageName: "_COp.95528a86e1127e2d:core.SubobjectDestroy.Core", scope: null, file: !192, line: 13, type: !22, spFlags: DISPFlagDefinition, unit: !193, retainedNodes: !203)
// CHECK:STDOUT: !201 = !DILocalVariable(arg: 1, scope: !200, type: !20)
// CHECK:STDOUT: !202 = !DILocation(line: 13, column: 3, scope: !200)
// CHECK:STDOUT: !203 = !{!201}
// CHECK:STDOUT: !204 = distinct !DISubprogram(name: "Op", linkageName: "_COp.294df737cd074d60:core.SubobjectDestroy.Core", scope: null, file: !192, line: 13, type: !22, spFlags: DISPFlagDefinition, unit: !193, retainedNodes: !207)
// CHECK:STDOUT: !205 = !DILocalVariable(arg: 1, scope: !204, type: !20)
// CHECK:STDOUT: !206 = !DILocation(line: 13, column: 3, scope: !204)
// CHECK:STDOUT: !207 = !{!205}
// CHECK:STDOUT: !208 = distinct !DISubprogram(name: "Op", linkageName: "_COp.caed8a96bf328d09:core.SubobjectDestroy.Core", scope: null, file: !192, line: 13, type: !22, spFlags: DISPFlagDefinition, unit: !193, retainedNodes: !211)
// CHECK:STDOUT: !209 = !DILocalVariable(arg: 1, scope: !208, type: !20)
// CHECK:STDOUT: !210 = !DILocation(line: 13, column: 3, scope: !208)
// CHECK:STDOUT: !211 = !{!209}
// CHECK:STDOUT: !212 = distinct !DISubprogram(name: "Op", linkageName: "_COp.b25525d21a9bd472:core.SubobjectDestroy.Core", scope: null, file: !192, line: 13, type: !22, spFlags: DISPFlagDefinition, unit: !193, retainedNodes: !215)
// CHECK:STDOUT: !213 = !DILocalVariable(arg: 1, scope: !212, type: !20)
// CHECK:STDOUT: !214 = !DILocation(line: 13, column: 3, scope: !212)
// CHECK:STDOUT: !215 = !{!213}
// CHECK:STDOUT: !216 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.d81ad81840dd2e3c", scope: null, file: !10, line: 30, type: !22, spFlags: DISPFlagDefinition, unit: !193, retainedNodes: !222)
// CHECK:STDOUT: !217 = !DILocalVariable(arg: 1, scope: !216, type: !20)
// CHECK:STDOUT: !220 = !DILocation(line: 32, column: 5, scope: !216)
// CHECK:STDOUT: !221 = !DILocation(line: 30, column: 1, scope: !216)
// CHECK:STDOUT: !222 = !{!217}
// CHECK:STDOUT: !223 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.648208d69a8e7988", scope: null, file: !10, line: 30, type: !22, spFlags: DISPFlagDefinition, unit: !193, retainedNodes: !229)
// CHECK:STDOUT: !224 = !DILocalVariable(arg: 1, scope: !223, type: !20)
// CHECK:STDOUT: !228 = !DILocation(line: 30, column: 1, scope: !223)
// CHECK:STDOUT: !229 = !{!224}
// CHECK:STDOUT: !230 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.8f745713e9c32bf6", scope: null, file: !10, line: 30, type: !22, spFlags: DISPFlagDefinition, unit: !193, retainedNodes: !236)
// CHECK:STDOUT: !231 = !DILocalVariable(arg: 1, scope: !230, type: !20)
// CHECK:STDOUT: !234 = !DILocation(line: 32, column: 5, scope: !230)
// CHECK:STDOUT: !235 = !DILocation(line: 30, column: 1, scope: !230)
// CHECK:STDOUT: !236 = !{!231}
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
// CHECK:STDOUT: define void @_CUse.Main(ptr %b) #0 !dbg !239 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Base.F.call.vtable = load ptr, ptr %b, align 8, !dbg !242
// CHECK:STDOUT:   %Base.F.call = call ptr @llvm.load.relative.i32(ptr %Base.F.call.vtable, i32 0), !dbg !242
// CHECK:STDOUT:   call void %Base.F.call(ptr %b), !dbg !242
// CHECK:STDOUT:   ret void, !dbg !243
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: read)
// CHECK:STDOUT: declare ptr @llvm.load.relative.i32(ptr, i32) #1
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: read) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!238}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !21 = !{null, !20}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !237 = !DIFile(filename: "call.carbon", directory: "")
// CHECK:STDOUT: !238 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !237, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !239 = distinct !DISubprogram(name: "Use", linkageName: "_CUse.Main", scope: null, file: !237, line: 8, type: !22, spFlags: DISPFlagDefinition, unit: !238, retainedNodes: !244)
// CHECK:STDOUT: !240 = !DILocalVariable(arg: 1, scope: !239, type: !20)
// CHECK:STDOUT: !242 = !DILocation(line: 9, column: 3, scope: !239)
// CHECK:STDOUT: !243 = !DILocation(line: 8, column: 1, scope: !239)
// CHECK:STDOUT: !244 = !{!240}
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
// CHECK:STDOUT: define weak_odr void @"_CF:thunk:Base.Main:Derived.Main"(ptr %self) #0 !dbg !247 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Derived.Main(ptr %self), !dbg !250
// CHECK:STDOUT:   ret void, !dbg !250
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CUse.Main() #1 !dbg !252 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %v.var = alloca { { ptr } }, align 8, !dbg !253
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %v.var), !dbg !253
// CHECK:STDOUT:   %.loc14_32.2.base = getelementptr inbounds nuw { { ptr } }, ptr %v.var, i32 0, i32 0, !dbg !254
// CHECK:STDOUT:   %.loc14_31.2.vptr = getelementptr inbounds nuw { ptr }, ptr %.loc14_32.2.base, i32 0, i32 0, !dbg !255
// CHECK:STDOUT:   %.loc14_32.6.base = getelementptr inbounds nuw { { ptr } }, ptr %v.var, i32 0, i32 0, !dbg !254
// CHECK:STDOUT:   %.loc14_32.7.vptr = getelementptr inbounds nuw { ptr }, ptr %.loc14_32.6.base, i32 0, i32 0, !dbg !254
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 %v.var, ptr align 8 @Derived.val.loc14_32.5, i64 8, i1 false), !dbg !254
// CHECK:STDOUT:   store ptr @"_CDerived.Main.$vtable", ptr %.loc14_32.7.vptr, align 8, !dbg !254
// CHECK:STDOUT:   %.loc15_3.1.base = getelementptr inbounds nuw { { ptr } }, ptr %v.var, i32 0, i32 0, !dbg !257
// CHECK:STDOUT:   %Base.F.call.vtable = load ptr, ptr %.loc15_3.1.base, align 8, !dbg !257
// CHECK:STDOUT:   %Base.F.call = call ptr @llvm.load.relative.i32(ptr %Base.F.call.vtable, i32 0), !dbg !257
// CHECK:STDOUT:   call void %Base.F.call(ptr %.loc15_3.1.base), !dbg !257
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.d81ad81840dd2e3c(ptr %v.var), !dbg !253
// CHECK:STDOUT:   ret void, !dbg !258
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.95528a86e1127e2d:core.SubobjectDestroy.Core"(ptr %self) #1 !dbg !259 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc14_3.1.vptr = getelementptr inbounds nuw { ptr }, ptr %self, i32 0, i32 0, !dbg !261
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.648208d69a8e7988(ptr %.loc14_3.1.vptr), !dbg !261
// CHECK:STDOUT:   ret void, !dbg !261
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.294df737cd074d60:core.SubobjectDestroy.Core"(ptr %self) #1 !dbg !263 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !265
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.caed8a96bf328d09:core.SubobjectDestroy.Core"(ptr %self) #1 !dbg !267 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc14_3.1.base = getelementptr inbounds nuw { { ptr } }, ptr %self, i32 0, i32 0, !dbg !269
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.8f745713e9c32bf6(ptr %.loc14_3.1.base), !dbg !269
// CHECK:STDOUT:   ret void, !dbg !269
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.b25525d21a9bd472:core.SubobjectDestroy.Core"(ptr %self) #1 !dbg !271 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc14_3.1.base = getelementptr inbounds nuw { { ptr } }, ptr %self, i32 0, i32 0, !dbg !273
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.8f745713e9c32bf6(ptr %.loc14_3.1.base), !dbg !273
// CHECK:STDOUT:   ret void, !dbg !273
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
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.d81ad81840dd2e3c(ptr %t) #1 !dbg !275 {
// CHECK:STDOUT:   call void @"_COp.b25525d21a9bd472:core.SubobjectDestroy.Core"(ptr %t), !dbg !279
// CHECK:STDOUT:   ret void, !dbg !280
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.648208d69a8e7988(ptr %t) #1 !dbg !282 {
// CHECK:STDOUT:   ret void, !dbg !287
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.8f745713e9c32bf6(ptr %t) #1 !dbg !289 {
// CHECK:STDOUT:   call void @"_COp.294df737cd074d60:core.SubobjectDestroy.Core"(ptr %t), !dbg !293
// CHECK:STDOUT:   ret void, !dbg !294
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.8f745713e9c32bf6, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { alwaysinline nounwind }
// CHECK:STDOUT: attributes #1 = { nounwind }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #3 = { nocallback nofree nosync nounwind willreturn memory(argmem: read) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!246}
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
// CHECK:STDOUT: !245 = !DIFile(filename: "call_impl.carbon", directory: "")
// CHECK:STDOUT: !246 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !245, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !247 = distinct !DISubprogram(name: "F", linkageName: "_CF:thunk:Base.Main:Derived.Main", scope: null, file: !245, line: 10, type: !22, spFlags: DISPFlagDefinition, unit: !246, retainedNodes: !251)
// CHECK:STDOUT: !248 = !DILocalVariable(arg: 1, scope: !247, type: !20)
// CHECK:STDOUT: !250 = !DILocation(line: 10, column: 3, scope: !247)
// CHECK:STDOUT: !251 = !{!248}
// CHECK:STDOUT: !252 = distinct !DISubprogram(name: "Use", linkageName: "_CUse.Main", scope: null, file: !245, line: 13, type: !41, spFlags: DISPFlagDefinition, unit: !246)
// CHECK:STDOUT: !253 = !DILocation(line: 14, column: 3, scope: !252)
// CHECK:STDOUT: !254 = !DILocation(line: 14, column: 21, scope: !252)
// CHECK:STDOUT: !255 = !DILocation(line: 14, column: 30, scope: !252)
// CHECK:STDOUT: !257 = !DILocation(line: 15, column: 3, scope: !252)
// CHECK:STDOUT: !258 = !DILocation(line: 13, column: 1, scope: !252)
// CHECK:STDOUT: !259 = distinct !DISubprogram(name: "Op", linkageName: "_COp.95528a86e1127e2d:core.SubobjectDestroy.Core", scope: null, file: !245, line: 14, type: !22, spFlags: DISPFlagDefinition, unit: !246, retainedNodes: !262)
// CHECK:STDOUT: !260 = !DILocalVariable(arg: 1, scope: !259, type: !20)
// CHECK:STDOUT: !261 = !DILocation(line: 14, column: 3, scope: !259)
// CHECK:STDOUT: !262 = !{!260}
// CHECK:STDOUT: !263 = distinct !DISubprogram(name: "Op", linkageName: "_COp.294df737cd074d60:core.SubobjectDestroy.Core", scope: null, file: !245, line: 14, type: !22, spFlags: DISPFlagDefinition, unit: !246, retainedNodes: !266)
// CHECK:STDOUT: !264 = !DILocalVariable(arg: 1, scope: !263, type: !20)
// CHECK:STDOUT: !265 = !DILocation(line: 14, column: 3, scope: !263)
// CHECK:STDOUT: !266 = !{!264}
// CHECK:STDOUT: !267 = distinct !DISubprogram(name: "Op", linkageName: "_COp.caed8a96bf328d09:core.SubobjectDestroy.Core", scope: null, file: !245, line: 14, type: !22, spFlags: DISPFlagDefinition, unit: !246, retainedNodes: !270)
// CHECK:STDOUT: !268 = !DILocalVariable(arg: 1, scope: !267, type: !20)
// CHECK:STDOUT: !269 = !DILocation(line: 14, column: 3, scope: !267)
// CHECK:STDOUT: !270 = !{!268}
// CHECK:STDOUT: !271 = distinct !DISubprogram(name: "Op", linkageName: "_COp.b25525d21a9bd472:core.SubobjectDestroy.Core", scope: null, file: !245, line: 14, type: !22, spFlags: DISPFlagDefinition, unit: !246, retainedNodes: !274)
// CHECK:STDOUT: !272 = !DILocalVariable(arg: 1, scope: !271, type: !20)
// CHECK:STDOUT: !273 = !DILocation(line: 14, column: 3, scope: !271)
// CHECK:STDOUT: !274 = !{!272}
// CHECK:STDOUT: !275 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.d81ad81840dd2e3c", scope: null, file: !10, line: 30, type: !22, spFlags: DISPFlagDefinition, unit: !246, retainedNodes: !281)
// CHECK:STDOUT: !276 = !DILocalVariable(arg: 1, scope: !275, type: !20)
// CHECK:STDOUT: !279 = !DILocation(line: 32, column: 5, scope: !275)
// CHECK:STDOUT: !280 = !DILocation(line: 30, column: 1, scope: !275)
// CHECK:STDOUT: !281 = !{!276}
// CHECK:STDOUT: !282 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.648208d69a8e7988", scope: null, file: !10, line: 30, type: !22, spFlags: DISPFlagDefinition, unit: !246, retainedNodes: !288)
// CHECK:STDOUT: !283 = !DILocalVariable(arg: 1, scope: !282, type: !20)
// CHECK:STDOUT: !287 = !DILocation(line: 30, column: 1, scope: !282)
// CHECK:STDOUT: !288 = !{!283}
// CHECK:STDOUT: !289 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.8f745713e9c32bf6", scope: null, file: !10, line: 30, type: !22, spFlags: DISPFlagDefinition, unit: !246, retainedNodes: !295)
// CHECK:STDOUT: !290 = !DILocalVariable(arg: 1, scope: !289, type: !20)
// CHECK:STDOUT: !293 = !DILocation(line: 32, column: 5, scope: !289)
// CHECK:STDOUT: !294 = !DILocation(line: 30, column: 1, scope: !289)
// CHECK:STDOUT: !295 = !{!290}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'generic_noop.carbon'
// CHECK:STDOUT: source_filename = "generic_noop.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!297}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !296 = !DIFile(filename: "generic_noop.carbon", directory: "")
// CHECK:STDOUT: !297 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !296, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
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
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !300 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var.loc16 = alloca { ptr }, align 8, !dbg !301
// CHECK:STDOUT:   %_.var.loc17 = alloca { ptr }, align 8, !dbg !304
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc16), !dbg !301
// CHECK:STDOUT:   %.loc16_22.2.vptr = getelementptr inbounds nuw { ptr }, ptr %_.var.loc16, i32 0, i32 0, !dbg !302
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 %_.var.loc16, ptr align 8 @Base.val.d42.loc16_3.1, i64 8, i1 false), !dbg !301
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc17), !dbg !304
// CHECK:STDOUT:   %.loc17_22.2.vptr = getelementptr inbounds nuw { ptr }, ptr %_.var.loc17, i32 0, i32 0, !dbg !305
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 %_.var.loc17, ptr align 8 @Base.val.52a.loc17_3.1, i64 8, i1 false), !dbg !304
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.613a848e97ed9e84(ptr %_.var.loc17), !dbg !304
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.300593fdd4019f45(ptr %_.var.loc16), !dbg !301
// CHECK:STDOUT:   ret void, !dbg !307
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.f2f12ab22a0a6b87:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !308 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc17_17.1.v = getelementptr inbounds nuw { {} }, ptr %self, i32 0, i32 0, !dbg !310
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.2c02d4d6b54f9d0d(ptr %.loc17_17.1.v), !dbg !310
// CHECK:STDOUT:   ret void, !dbg !310
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.8578661a61c8bbbf:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !312 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc17_17.1.v = getelementptr inbounds nuw { {} }, ptr %self, i32 0, i32 0, !dbg !314
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.2c02d4d6b54f9d0d(ptr %.loc17_17.1.v), !dbg !314
// CHECK:STDOUT:   ret void, !dbg !314
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.95528a86e1127e2d:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !316 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc17_3.1.vptr = getelementptr inbounds nuw { ptr }, ptr %self, i32 0, i32 0, !dbg !318
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.648208d69a8e7988(ptr %.loc17_3.1.vptr), !dbg !318
// CHECK:STDOUT:   ret void, !dbg !318
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.ce451d5fcc93b1f6:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !320 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !322
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.239e9a7b889667cd:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !324 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !326
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Base.Main.2c02d4d6b54f9d0d(ptr %self) #0 !dbg !328 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca {}, align 1, !dbg !331
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !331
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.2c02d4d6b54f9d0d(ptr %_.var), !dbg !331
// CHECK:STDOUT:   ret void, !dbg !333
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Base.Main.9b8326f304a33dd3(ptr %self) #0 !dbg !335 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca { {} }, align 1, !dbg !338
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !338
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.9b8326f304a33dd3(ptr %_.var), !dbg !338
// CHECK:STDOUT:   ret void, !dbg !340
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.613a848e97ed9e84(ptr %t) #0 !dbg !342 {
// CHECK:STDOUT:   call void @"_COp.ce451d5fcc93b1f6:core.SubobjectDestroy.Core"(ptr %t), !dbg !346
// CHECK:STDOUT:   ret void, !dbg !347
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.300593fdd4019f45(ptr %t) #0 !dbg !349 {
// CHECK:STDOUT:   call void @"_COp.239e9a7b889667cd:core.SubobjectDestroy.Core"(ptr %t), !dbg !353
// CHECK:STDOUT:   ret void, !dbg !354
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.2c02d4d6b54f9d0d(ptr %t) #0 !dbg !356 {
// CHECK:STDOUT:   ret void, !dbg !361
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.648208d69a8e7988(ptr %t) #0 !dbg !363 {
// CHECK:STDOUT:   ret void, !dbg !368
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.9b8326f304a33dd3(ptr %t) #0 !dbg !370 {
// CHECK:STDOUT:   call void @"_COp.8578661a61c8bbbf:core.SubobjectDestroy.Core"(ptr %t), !dbg !374
// CHECK:STDOUT:   ret void, !dbg !375
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.2c02d4d6b54f9d0d, { 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!299}
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
// CHECK:STDOUT: !298 = !DIFile(filename: "generic_use.carbon", directory: "")
// CHECK:STDOUT: !299 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !298, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !300 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !298, line: 15, type: !41, spFlags: DISPFlagDefinition, unit: !299)
// CHECK:STDOUT: !301 = !DILocation(line: 16, column: 3, scope: !300)
// CHECK:STDOUT: !302 = !DILocation(line: 16, column: 21, scope: !300)
// CHECK:STDOUT: !304 = !DILocation(line: 17, column: 3, scope: !300)
// CHECK:STDOUT: !305 = !DILocation(line: 17, column: 21, scope: !300)
// CHECK:STDOUT: !307 = !DILocation(line: 15, column: 1, scope: !300)
// CHECK:STDOUT: !308 = distinct !DISubprogram(name: "Op", linkageName: "_COp.f2f12ab22a0a6b87:core.SubobjectDestroy.Core", scope: null, file: !298, line: 17, type: !22, spFlags: DISPFlagDefinition, unit: !299, retainedNodes: !311)
// CHECK:STDOUT: !309 = !DILocalVariable(arg: 1, scope: !308, type: !20)
// CHECK:STDOUT: !310 = !DILocation(line: 17, column: 10, scope: !308)
// CHECK:STDOUT: !311 = !{!309}
// CHECK:STDOUT: !312 = distinct !DISubprogram(name: "Op", linkageName: "_COp.8578661a61c8bbbf:core.SubobjectDestroy.Core", scope: null, file: !298, line: 17, type: !22, spFlags: DISPFlagDefinition, unit: !299, retainedNodes: !315)
// CHECK:STDOUT: !313 = !DILocalVariable(arg: 1, scope: !312, type: !20)
// CHECK:STDOUT: !314 = !DILocation(line: 17, column: 10, scope: !312)
// CHECK:STDOUT: !315 = !{!313}
// CHECK:STDOUT: !316 = distinct !DISubprogram(name: "Op", linkageName: "_COp.95528a86e1127e2d:core.SubobjectDestroy.Core", scope: null, file: !298, line: 17, type: !22, spFlags: DISPFlagDefinition, unit: !299, retainedNodes: !319)
// CHECK:STDOUT: !317 = !DILocalVariable(arg: 1, scope: !316, type: !20)
// CHECK:STDOUT: !318 = !DILocation(line: 17, column: 3, scope: !316)
// CHECK:STDOUT: !319 = !{!317}
// CHECK:STDOUT: !320 = distinct !DISubprogram(name: "Op", linkageName: "_COp.ce451d5fcc93b1f6:core.SubobjectDestroy.Core", scope: null, file: !298, line: 17, type: !22, spFlags: DISPFlagDefinition, unit: !299, retainedNodes: !323)
// CHECK:STDOUT: !321 = !DILocalVariable(arg: 1, scope: !320, type: !20)
// CHECK:STDOUT: !322 = !DILocation(line: 17, column: 3, scope: !320)
// CHECK:STDOUT: !323 = !{!321}
// CHECK:STDOUT: !324 = distinct !DISubprogram(name: "Op", linkageName: "_COp.239e9a7b889667cd:core.SubobjectDestroy.Core", scope: null, file: !298, line: 16, type: !22, spFlags: DISPFlagDefinition, unit: !299, retainedNodes: !327)
// CHECK:STDOUT: !325 = !DILocalVariable(arg: 1, scope: !324, type: !20)
// CHECK:STDOUT: !326 = !DILocation(line: 16, column: 3, scope: !324)
// CHECK:STDOUT: !327 = !{!325}
// CHECK:STDOUT: !328 = distinct !DISubprogram(name: "F", linkageName: "_CF.Base.Main.2c02d4d6b54f9d0d", scope: null, file: !298, line: 5, type: !22, spFlags: DISPFlagDefinition, unit: !299, retainedNodes: !334)
// CHECK:STDOUT: !329 = !DILocalVariable(arg: 1, scope: !328, type: !20)
// CHECK:STDOUT: !331 = !DILocation(line: 6, column: 5, scope: !328)
// CHECK:STDOUT: !333 = !DILocation(line: 5, column: 3, scope: !328)
// CHECK:STDOUT: !334 = !{!329}
// CHECK:STDOUT: !335 = distinct !DISubprogram(name: "F", linkageName: "_CF.Base.Main.9b8326f304a33dd3", scope: null, file: !298, line: 5, type: !22, spFlags: DISPFlagDefinition, unit: !299, retainedNodes: !341)
// CHECK:STDOUT: !336 = !DILocalVariable(arg: 1, scope: !335, type: !20)
// CHECK:STDOUT: !338 = !DILocation(line: 6, column: 5, scope: !335)
// CHECK:STDOUT: !340 = !DILocation(line: 5, column: 3, scope: !335)
// CHECK:STDOUT: !341 = !{!336}
// CHECK:STDOUT: !342 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.613a848e97ed9e84", scope: null, file: !10, line: 30, type: !22, spFlags: DISPFlagDefinition, unit: !299, retainedNodes: !348)
// CHECK:STDOUT: !343 = !DILocalVariable(arg: 1, scope: !342, type: !20)
// CHECK:STDOUT: !346 = !DILocation(line: 32, column: 5, scope: !342)
// CHECK:STDOUT: !347 = !DILocation(line: 30, column: 1, scope: !342)
// CHECK:STDOUT: !348 = !{!343}
// CHECK:STDOUT: !349 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.300593fdd4019f45", scope: null, file: !10, line: 30, type: !22, spFlags: DISPFlagDefinition, unit: !299, retainedNodes: !355)
// CHECK:STDOUT: !350 = !DILocalVariable(arg: 1, scope: !349, type: !20)
// CHECK:STDOUT: !353 = !DILocation(line: 32, column: 5, scope: !349)
// CHECK:STDOUT: !354 = !DILocation(line: 30, column: 1, scope: !349)
// CHECK:STDOUT: !355 = !{!350}
// CHECK:STDOUT: !356 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.2c02d4d6b54f9d0d", scope: null, file: !10, line: 30, type: !22, spFlags: DISPFlagDefinition, unit: !299, retainedNodes: !362)
// CHECK:STDOUT: !357 = !DILocalVariable(arg: 1, scope: !356, type: !20)
// CHECK:STDOUT: !361 = !DILocation(line: 30, column: 1, scope: !356)
// CHECK:STDOUT: !362 = !{!357}
// CHECK:STDOUT: !363 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.648208d69a8e7988", scope: null, file: !10, line: 30, type: !22, spFlags: DISPFlagDefinition, unit: !299, retainedNodes: !369)
// CHECK:STDOUT: !364 = !DILocalVariable(arg: 1, scope: !363, type: !20)
// CHECK:STDOUT: !368 = !DILocation(line: 30, column: 1, scope: !363)
// CHECK:STDOUT: !369 = !{!364}
// CHECK:STDOUT: !370 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.9b8326f304a33dd3", scope: null, file: !10, line: 30, type: !22, spFlags: DISPFlagDefinition, unit: !299, retainedNodes: !376)
// CHECK:STDOUT: !371 = !DILocalVariable(arg: 1, scope: !370, type: !20)
// CHECK:STDOUT: !374 = !DILocation(line: 32, column: 5, scope: !370)
// CHECK:STDOUT: !375 = !DILocation(line: 30, column: 1, scope: !370)
// CHECK:STDOUT: !376 = !{!371}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'generic_base.carbon'
// CHECK:STDOUT: source_filename = "generic_base.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @"_CDerived.Main.$vtable" = unnamed_addr constant [1 x i32] [i32 trunc (i64 sub (i64 ptrtoint (ptr @_CF.Base.Main.6d6a4a2512463e59 to i64), i64 ptrtoint (ptr @"_CDerived.Main.$vtable" to i64)) to i32)]
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMake.Main() #0 !dbg !379 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca { { ptr } }, align 8, !dbg !380
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !380
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.d81ad81840dd2e3c(ptr %_.var), !dbg !380
// CHECK:STDOUT:   ret void, !dbg !382
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.95528a86e1127e2d:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !383 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc15_3.1.vptr = getelementptr inbounds nuw { ptr }, ptr %self, i32 0, i32 0, !dbg !385
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.648208d69a8e7988(ptr %.loc15_3.1.vptr), !dbg !385
// CHECK:STDOUT:   ret void, !dbg !385
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.fe75085f268e05b2:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !387 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !389
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.66111a67d7c24ae0:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !391 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc15_3.1.base = getelementptr inbounds nuw { { ptr } }, ptr %self, i32 0, i32 0, !dbg !393
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.a782d97849f79bb1(ptr %.loc15_3.1.base), !dbg !393
// CHECK:STDOUT:   ret void, !dbg !393
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.b25525d21a9bd472:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !395 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc15_3.1.base = getelementptr inbounds nuw { { ptr } }, ptr %self, i32 0, i32 0, !dbg !397
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.a782d97849f79bb1(ptr %.loc15_3.1.base), !dbg !397
// CHECK:STDOUT:   ret void, !dbg !397
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Base.Main.6d6a4a2512463e59(ptr %self) #0 !dbg !399 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !402
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.d81ad81840dd2e3c(ptr %t) #0 !dbg !404 {
// CHECK:STDOUT:   call void @"_COp.b25525d21a9bd472:core.SubobjectDestroy.Core"(ptr %t), !dbg !408
// CHECK:STDOUT:   ret void, !dbg !409
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.648208d69a8e7988(ptr %t) #0 !dbg !411 {
// CHECK:STDOUT:   ret void, !dbg !416
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.a782d97849f79bb1(ptr %t) #0 !dbg !418 {
// CHECK:STDOUT:   call void @"_COp.fe75085f268e05b2:core.SubobjectDestroy.Core"(ptr %t), !dbg !422
// CHECK:STDOUT:   ret void, !dbg !423
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.a782d97849f79bb1, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!378}
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
// CHECK:STDOUT: !377 = !DIFile(filename: "generic_base.carbon", directory: "")
// CHECK:STDOUT: !378 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !377, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !379 = distinct !DISubprogram(name: "Make", linkageName: "_CMake.Main", scope: null, file: !377, line: 14, type: !41, spFlags: DISPFlagDefinition, unit: !378)
// CHECK:STDOUT: !380 = !DILocation(line: 15, column: 3, scope: !379)
// CHECK:STDOUT: !382 = !DILocation(line: 14, column: 1, scope: !379)
// CHECK:STDOUT: !383 = distinct !DISubprogram(name: "Op", linkageName: "_COp.95528a86e1127e2d:core.SubobjectDestroy.Core", scope: null, file: !377, line: 15, type: !22, spFlags: DISPFlagDefinition, unit: !378, retainedNodes: !386)
// CHECK:STDOUT: !384 = !DILocalVariable(arg: 1, scope: !383, type: !20)
// CHECK:STDOUT: !385 = !DILocation(line: 15, column: 3, scope: !383)
// CHECK:STDOUT: !386 = !{!384}
// CHECK:STDOUT: !387 = distinct !DISubprogram(name: "Op", linkageName: "_COp.fe75085f268e05b2:core.SubobjectDestroy.Core", scope: null, file: !377, line: 15, type: !22, spFlags: DISPFlagDefinition, unit: !378, retainedNodes: !390)
// CHECK:STDOUT: !388 = !DILocalVariable(arg: 1, scope: !387, type: !20)
// CHECK:STDOUT: !389 = !DILocation(line: 15, column: 3, scope: !387)
// CHECK:STDOUT: !390 = !{!388}
// CHECK:STDOUT: !391 = distinct !DISubprogram(name: "Op", linkageName: "_COp.66111a67d7c24ae0:core.SubobjectDestroy.Core", scope: null, file: !377, line: 15, type: !22, spFlags: DISPFlagDefinition, unit: !378, retainedNodes: !394)
// CHECK:STDOUT: !392 = !DILocalVariable(arg: 1, scope: !391, type: !20)
// CHECK:STDOUT: !393 = !DILocation(line: 15, column: 3, scope: !391)
// CHECK:STDOUT: !394 = !{!392}
// CHECK:STDOUT: !395 = distinct !DISubprogram(name: "Op", linkageName: "_COp.b25525d21a9bd472:core.SubobjectDestroy.Core", scope: null, file: !377, line: 15, type: !22, spFlags: DISPFlagDefinition, unit: !378, retainedNodes: !398)
// CHECK:STDOUT: !396 = !DILocalVariable(arg: 1, scope: !395, type: !20)
// CHECK:STDOUT: !397 = !DILocation(line: 15, column: 3, scope: !395)
// CHECK:STDOUT: !398 = !{!396}
// CHECK:STDOUT: !399 = distinct !DISubprogram(name: "F", linkageName: "_CF.Base.Main.6d6a4a2512463e59", scope: null, file: !377, line: 5, type: !22, spFlags: DISPFlagDefinition, unit: !378, retainedNodes: !403)
// CHECK:STDOUT: !400 = !DILocalVariable(arg: 1, scope: !399, type: !20)
// CHECK:STDOUT: !402 = !DILocation(line: 5, column: 3, scope: !399)
// CHECK:STDOUT: !403 = !{!400}
// CHECK:STDOUT: !404 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.d81ad81840dd2e3c", scope: null, file: !10, line: 30, type: !22, spFlags: DISPFlagDefinition, unit: !378, retainedNodes: !410)
// CHECK:STDOUT: !405 = !DILocalVariable(arg: 1, scope: !404, type: !20)
// CHECK:STDOUT: !408 = !DILocation(line: 32, column: 5, scope: !404)
// CHECK:STDOUT: !409 = !DILocation(line: 30, column: 1, scope: !404)
// CHECK:STDOUT: !410 = !{!405}
// CHECK:STDOUT: !411 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.648208d69a8e7988", scope: null, file: !10, line: 30, type: !22, spFlags: DISPFlagDefinition, unit: !378, retainedNodes: !417)
// CHECK:STDOUT: !412 = !DILocalVariable(arg: 1, scope: !411, type: !20)
// CHECK:STDOUT: !416 = !DILocation(line: 30, column: 1, scope: !411)
// CHECK:STDOUT: !417 = !{!412}
// CHECK:STDOUT: !418 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.a782d97849f79bb1", scope: null, file: !10, line: 30, type: !22, spFlags: DISPFlagDefinition, unit: !378, retainedNodes: !424)
// CHECK:STDOUT: !419 = !DILocalVariable(arg: 1, scope: !418, type: !20)
// CHECK:STDOUT: !422 = !DILocation(line: 32, column: 5, scope: !418)
// CHECK:STDOUT: !423 = !DILocation(line: 30, column: 1, scope: !418)
// CHECK:STDOUT: !424 = !{!419}
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
// CHECK:STDOUT: define void @_CF.Derived.Main(ptr sret({}) %return, ptr %self, ptr %n) #0 !dbg !431 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %return, ptr align 1 @From.val.loc18_14, i64 0, i1 false), !dbg !438
// CHECK:STDOUT:   ret void, !dbg !438
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_CF:thunk:Base.Main:Derived.Main"(ptr sret({}) %return, ptr %self, ptr %n) #1 !dbg !440 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc17_52.2.temp = alloca {}, align 1, !dbg !444
// CHECK:STDOUT:   %.loc17_52.4.temp = alloca {}, align 1, !dbg !444
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc17_52.2.temp), !dbg !444
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc17_52.4.temp), !dbg !444
// CHECK:STDOUT:   call void @"_CConvert.From.Main:ImplicitAs.a3bb440f6426a7d6.Core"(ptr %.loc17_52.4.temp, ptr %n), !dbg !444
// CHECK:STDOUT:   call void @_CF.Derived.Main(ptr %.loc17_52.2.temp, ptr %self, ptr %.loc17_52.4.temp), !dbg !444
// CHECK:STDOUT:   call void @"_CConvert.From.Main:ImplicitAs.a3bb440f6426a7d6.Core"(ptr %return, ptr %.loc17_52.2.temp), !dbg !444
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.3f61691f292a1f88(ptr %.loc17_52.2.temp), !dbg !444
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.36c1478c1cb9545e(ptr %.loc17_52.4.temp), !dbg !444
// CHECK:STDOUT:   ret void, !dbg !444
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CUse.Main() #0 !dbg !446 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %d.var = alloca { { ptr } }, align 8, !dbg !447
// CHECK:STDOUT:   %.loc25_10.1.temp = alloca {}, align 1, !dbg !455
// CHECK:STDOUT:   %.loc25_9.2.temp = alloca {}, align 1, !dbg !456
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %d.var), !dbg !447
// CHECK:STDOUT:   %.loc23_31.2.base = getelementptr inbounds nuw { { ptr } }, ptr %d.var, i32 0, i32 0, !dbg !448
// CHECK:STDOUT:   %.loc23_30.2.vptr = getelementptr inbounds nuw { ptr }, ptr %.loc23_31.2.base, i32 0, i32 0, !dbg !449
// CHECK:STDOUT:   %.loc23_31.6.base = getelementptr inbounds nuw { { ptr } }, ptr %d.var, i32 0, i32 0, !dbg !448
// CHECK:STDOUT:   %.loc23_31.7.vptr = getelementptr inbounds nuw { ptr }, ptr %.loc23_31.6.base, i32 0, i32 0, !dbg !448
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 %d.var, ptr align 8 @Derived.val.loc23_31.5, i64 8, i1 false), !dbg !448
// CHECK:STDOUT:   store ptr @"_CDerived.Main.$vtable", ptr %.loc23_31.7.vptr, align 8, !dbg !448
// CHECK:STDOUT:   %.loc24_18.2.base = getelementptr inbounds nuw { { ptr } }, ptr %d.var, i32 0, i32 0, !dbg !452
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc25_10.1.temp), !dbg !455
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc25_9.2.temp), !dbg !456
// CHECK:STDOUT:   %Base.F.call.vtable = load ptr, ptr %.loc25_10.1.temp, align 8, !dbg !455
// CHECK:STDOUT:   %Base.F.call = call ptr @llvm.load.relative.i32(ptr %Base.F.call.vtable, i32 0), !dbg !455
// CHECK:STDOUT:   call void %Base.F.call(ptr %.loc25_10.1.temp, ptr %.loc24_18.2.base, ptr @From.val.loc18_14), !dbg !455
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.36c1478c1cb9545e(ptr %.loc25_10.1.temp), !dbg !455
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.3f61691f292a1f88(ptr @From.val), !dbg !456
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.d81ad81840dd2e3c(ptr %d.var), !dbg !447
// CHECK:STDOUT:   ret void, !dbg !457
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.95528a86e1127e2d:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !458 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc23_3.1.vptr = getelementptr inbounds nuw { ptr }, ptr %self, i32 0, i32 0, !dbg !460
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.648208d69a8e7988(ptr %.loc23_3.1.vptr), !dbg !460
// CHECK:STDOUT:   ret void, !dbg !460
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.294df737cd074d60:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !462 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !464
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.caed8a96bf328d09:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !466 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc23_3.1.base = getelementptr inbounds nuw { { ptr } }, ptr %self, i32 0, i32 0, !dbg !468
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.8f745713e9c32bf6(ptr %.loc23_3.1.base), !dbg !468
// CHECK:STDOUT:   ret void, !dbg !468
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.b25525d21a9bd472:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !470 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc23_3.1.base = getelementptr inbounds nuw { { ptr } }, ptr %self, i32 0, i32 0, !dbg !472
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.8f745713e9c32bf6(ptr %.loc23_3.1.base), !dbg !472
// CHECK:STDOUT:   ret void, !dbg !472
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.3f61691f292a1f88(ptr %t) #0 !dbg !474 {
// CHECK:STDOUT:   ret void, !dbg !479
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.36c1478c1cb9545e(ptr %t) #0 !dbg !481 {
// CHECK:STDOUT:   ret void, !dbg !486
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: read)
// CHECK:STDOUT: declare ptr @llvm.load.relative.i32(ptr, i32) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.d81ad81840dd2e3c(ptr %t) #0 !dbg !488 {
// CHECK:STDOUT:   call void @"_COp.b25525d21a9bd472:core.SubobjectDestroy.Core"(ptr %t), !dbg !492
// CHECK:STDOUT:   ret void, !dbg !493
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.648208d69a8e7988(ptr %t) #0 !dbg !495 {
// CHECK:STDOUT:   ret void, !dbg !500
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.8f745713e9c32bf6(ptr %t) #0 !dbg !502 {
// CHECK:STDOUT:   call void @"_COp.294df737cd074d60:core.SubobjectDestroy.Core"(ptr %t), !dbg !506
// CHECK:STDOUT:   ret void, !dbg !507
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 4, 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.3f61691f292a1f88, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.36c1478c1cb9545e, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.8f745713e9c32bf6, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline nounwind }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #3 = { nocallback nofree nosync nounwind willreturn memory(argmem: read) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!426}
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
// CHECK:STDOUT: !425 = !DIFile(filename: "thunk.carbon", directory: "")
// CHECK:STDOUT: !426 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !425, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !429 = !{!20, !20, !20}
// CHECK:STDOUT: !430 = !DISubroutineType(types: !429)
// CHECK:STDOUT: !431 = distinct !DISubprogram(name: "F", linkageName: "_CF.Derived.Main", scope: null, file: !425, line: 17, type: !430, spFlags: DISPFlagDefinition, unit: !426, retainedNodes: !439)
// CHECK:STDOUT: !432 = !DILocalVariable(arg: 1, scope: !431, type: !20)
// CHECK:STDOUT: !433 = !DILocalVariable(arg: 2, scope: !431, type: !20)
// CHECK:STDOUT: !438 = !DILocation(line: 18, column: 5, scope: !431)
// CHECK:STDOUT: !439 = !{!432, !433}
// CHECK:STDOUT: !440 = distinct !DISubprogram(name: "F", linkageName: "_CF:thunk:Base.Main:Derived.Main", scope: null, file: !425, line: 17, type: !430, spFlags: DISPFlagDefinition, unit: !426, retainedNodes: !445)
// CHECK:STDOUT: !441 = !DILocalVariable(arg: 1, scope: !440, type: !20)
// CHECK:STDOUT: !442 = !DILocalVariable(arg: 2, scope: !440, type: !20)
// CHECK:STDOUT: !444 = !DILocation(line: 17, column: 3, scope: !440)
// CHECK:STDOUT: !445 = !{!441, !442}
// CHECK:STDOUT: !446 = distinct !DISubprogram(name: "Use", linkageName: "_CUse.Main", scope: null, file: !425, line: 22, type: !41, spFlags: DISPFlagDefinition, unit: !426)
// CHECK:STDOUT: !447 = !DILocation(line: 23, column: 3, scope: !446)
// CHECK:STDOUT: !448 = !DILocation(line: 23, column: 20, scope: !446)
// CHECK:STDOUT: !449 = !DILocation(line: 23, column: 29, scope: !446)
// CHECK:STDOUT: !452 = !DILocation(line: 24, column: 18, scope: !446)
// CHECK:STDOUT: !455 = !DILocation(line: 25, column: 3, scope: !446)
// CHECK:STDOUT: !456 = !DILocation(line: 25, column: 8, scope: !446)
// CHECK:STDOUT: !457 = !DILocation(line: 22, column: 1, scope: !446)
// CHECK:STDOUT: !458 = distinct !DISubprogram(name: "Op", linkageName: "_COp.95528a86e1127e2d:core.SubobjectDestroy.Core", scope: null, file: !425, line: 23, type: !22, spFlags: DISPFlagDefinition, unit: !426, retainedNodes: !461)
// CHECK:STDOUT: !459 = !DILocalVariable(arg: 1, scope: !458, type: !20)
// CHECK:STDOUT: !460 = !DILocation(line: 23, column: 3, scope: !458)
// CHECK:STDOUT: !461 = !{!459}
// CHECK:STDOUT: !462 = distinct !DISubprogram(name: "Op", linkageName: "_COp.294df737cd074d60:core.SubobjectDestroy.Core", scope: null, file: !425, line: 23, type: !22, spFlags: DISPFlagDefinition, unit: !426, retainedNodes: !465)
// CHECK:STDOUT: !463 = !DILocalVariable(arg: 1, scope: !462, type: !20)
// CHECK:STDOUT: !464 = !DILocation(line: 23, column: 3, scope: !462)
// CHECK:STDOUT: !465 = !{!463}
// CHECK:STDOUT: !466 = distinct !DISubprogram(name: "Op", linkageName: "_COp.caed8a96bf328d09:core.SubobjectDestroy.Core", scope: null, file: !425, line: 23, type: !22, spFlags: DISPFlagDefinition, unit: !426, retainedNodes: !469)
// CHECK:STDOUT: !467 = !DILocalVariable(arg: 1, scope: !466, type: !20)
// CHECK:STDOUT: !468 = !DILocation(line: 23, column: 3, scope: !466)
// CHECK:STDOUT: !469 = !{!467}
// CHECK:STDOUT: !470 = distinct !DISubprogram(name: "Op", linkageName: "_COp.b25525d21a9bd472:core.SubobjectDestroy.Core", scope: null, file: !425, line: 23, type: !22, spFlags: DISPFlagDefinition, unit: !426, retainedNodes: !473)
// CHECK:STDOUT: !471 = !DILocalVariable(arg: 1, scope: !470, type: !20)
// CHECK:STDOUT: !472 = !DILocation(line: 23, column: 3, scope: !470)
// CHECK:STDOUT: !473 = !{!471}
// CHECK:STDOUT: !474 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.3f61691f292a1f88", scope: null, file: !10, line: 30, type: !22, spFlags: DISPFlagDefinition, unit: !426, retainedNodes: !480)
// CHECK:STDOUT: !475 = !DILocalVariable(arg: 1, scope: !474, type: !20)
// CHECK:STDOUT: !479 = !DILocation(line: 30, column: 1, scope: !474)
// CHECK:STDOUT: !480 = !{!475}
// CHECK:STDOUT: !481 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.36c1478c1cb9545e", scope: null, file: !10, line: 30, type: !22, spFlags: DISPFlagDefinition, unit: !426, retainedNodes: !487)
// CHECK:STDOUT: !482 = !DILocalVariable(arg: 1, scope: !481, type: !20)
// CHECK:STDOUT: !486 = !DILocation(line: 30, column: 1, scope: !481)
// CHECK:STDOUT: !487 = !{!482}
// CHECK:STDOUT: !488 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.d81ad81840dd2e3c", scope: null, file: !10, line: 30, type: !22, spFlags: DISPFlagDefinition, unit: !426, retainedNodes: !494)
// CHECK:STDOUT: !489 = !DILocalVariable(arg: 1, scope: !488, type: !20)
// CHECK:STDOUT: !492 = !DILocation(line: 32, column: 5, scope: !488)
// CHECK:STDOUT: !493 = !DILocation(line: 30, column: 1, scope: !488)
// CHECK:STDOUT: !494 = !{!489}
// CHECK:STDOUT: !495 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.648208d69a8e7988", scope: null, file: !10, line: 30, type: !22, spFlags: DISPFlagDefinition, unit: !426, retainedNodes: !501)
// CHECK:STDOUT: !496 = !DILocalVariable(arg: 1, scope: !495, type: !20)
// CHECK:STDOUT: !500 = !DILocation(line: 30, column: 1, scope: !495)
// CHECK:STDOUT: !501 = !{!496}
// CHECK:STDOUT: !502 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.8f745713e9c32bf6", scope: null, file: !10, line: 30, type: !22, spFlags: DISPFlagDefinition, unit: !426, retainedNodes: !508)
// CHECK:STDOUT: !503 = !DILocalVariable(arg: 1, scope: !502, type: !20)
// CHECK:STDOUT: !506 = !DILocation(line: 32, column: 5, scope: !502)
// CHECK:STDOUT: !507 = !DILocation(line: 30, column: 1, scope: !502)
// CHECK:STDOUT: !508 = !{!503}
// CHECK:STDOUT:
