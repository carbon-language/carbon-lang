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
// CHECK:STDOUT:   call void @"_CSelfDestruct.43ed5d0e08fc7556:core.Destroy.Core"(ptr %_.var.loc12), !dbg !52
// CHECK:STDOUT:   call void @"_CSelfDestruct.43ed5d0e08fc7556:core.Destroy.Core"(ptr %_.var.loc9), !dbg !48
// CHECK:STDOUT:   call void @"_CSelfDestruct.4487e305711510e9:core.Destroy.Core"(ptr %_.var.loc8), !dbg !45
// CHECK:STDOUT:   call void @"_CSelfDestruct.bf085e1f873e33df:core.Destroy.Core"(ptr %_.var.loc7), !dbg !43
// CHECK:STDOUT:   ret void, !dbg !54
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CFn.Intermediate.Classes(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_CFn:thunk:Intermediate.Classes:Derived.Classes"(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.454fecbcedcd86e1:core.Destroy.Core"(ptr %self) #0 !dbg !55 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !57
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.cb8adced19392b99:core.Destroy.Core"(ptr %self) #0 !dbg !59 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !61
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.bf085e1f873e33df:core.Destroy.Core"(ptr %self) #0 !dbg !63 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !65
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.f7b08e0f33b30342:core.Destroy.Core"(ptr %self) #0 !dbg !67 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !69
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.f7b08e0f33b30342:core.Destroy.Core"(ptr %self) #0 !dbg !71 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.f7b08e0f33b30342:core.Destroy.Core"(ptr %self), !dbg !73
// CHECK:STDOUT:   ret void, !dbg !73
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.4487e305711510e9:core.Destroy.Core"(ptr %self) #0 !dbg !75 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !77
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.4487e305711510e9:core.Destroy.Core"(ptr %self) #0 !dbg !79 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.4487e305711510e9:core.Destroy.Core"(ptr %self), !dbg !81
// CHECK:STDOUT:   ret void, !dbg !81
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.3c3534e993db7939:core.Destroy.Core"(ptr %self) #0 !dbg !83 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !85
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.3c3534e993db7939:core.Destroy.Core"(ptr %self) #0 !dbg !87 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.3c3534e993db7939:core.Destroy.Core"(ptr %self), !dbg !89
// CHECK:STDOUT:   ret void, !dbg !89
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.43ed5d0e08fc7556:core.Destroy.Core"(ptr %self) #0 !dbg !91 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !93
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.43ed5d0e08fc7556:core.Destroy.Core"(ptr %self) #0 !dbg !95 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.43ed5d0e08fc7556:core.Destroy.Core"(ptr %self), !dbg !97
// CHECK:STDOUT:   ret void, !dbg !97
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CUse.Create(ptr %v) #0 !dbg !99 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Intermediate.Fn.call.vtable = load ptr, ptr %v, align 8, !dbg !102
// CHECK:STDOUT:   %Intermediate.Fn.call = call ptr @llvm.load.relative.i32(ptr %Intermediate.Fn.call.vtable, i32 0), !dbg !102
// CHECK:STDOUT:   call void %Intermediate.Fn.call(ptr %v), !dbg !102
// CHECK:STDOUT:   ret void, !dbg !103
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
// CHECK:STDOUT: uselistorder ptr @"_CSelfDestruct.43ed5d0e08fc7556:core.Destroy.Core", { 1, 0 }
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
// CHECK:STDOUT: !55 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.454fecbcedcd86e1:core.Destroy.Core", scope: null, file: !38, line: 12, type: !22, spFlags: DISPFlagDefinition, unit: !39, retainedNodes: !58)
// CHECK:STDOUT: !56 = !DILocalVariable(arg: 1, scope: !55, type: !20)
// CHECK:STDOUT: !57 = !DILocation(line: 12, column: 3, scope: !55)
// CHECK:STDOUT: !58 = !{!56}
// CHECK:STDOUT: !59 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.cb8adced19392b99:core.Destroy.Core", scope: null, file: !38, line: 12, type: !22, spFlags: DISPFlagDefinition, unit: !39, retainedNodes: !62)
// CHECK:STDOUT: !60 = !DILocalVariable(arg: 1, scope: !59, type: !20)
// CHECK:STDOUT: !61 = !DILocation(line: 12, column: 3, scope: !59)
// CHECK:STDOUT: !62 = !{!60}
// CHECK:STDOUT: !63 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.bf085e1f873e33df:core.Destroy.Core", scope: null, file: !38, line: 12, type: !22, spFlags: DISPFlagDefinition, unit: !39, retainedNodes: !66)
// CHECK:STDOUT: !64 = !DILocalVariable(arg: 1, scope: !63, type: !20)
// CHECK:STDOUT: !65 = !DILocation(line: 12, column: 3, scope: !63)
// CHECK:STDOUT: !66 = !{!64}
// CHECK:STDOUT: !67 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.f7b08e0f33b30342:core.Destroy.Core", scope: null, file: !38, line: 12, type: !22, spFlags: DISPFlagDefinition, unit: !39, retainedNodes: !70)
// CHECK:STDOUT: !68 = !DILocalVariable(arg: 1, scope: !67, type: !20)
// CHECK:STDOUT: !69 = !DILocation(line: 12, column: 3, scope: !67)
// CHECK:STDOUT: !70 = !{!68}
// CHECK:STDOUT: !71 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.f7b08e0f33b30342:core.Destroy.Core", scope: null, file: !38, line: 12, type: !22, spFlags: DISPFlagDefinition, unit: !39, retainedNodes: !74)
// CHECK:STDOUT: !72 = !DILocalVariable(arg: 1, scope: !71, type: !20)
// CHECK:STDOUT: !73 = !DILocation(line: 12, column: 3, scope: !71)
// CHECK:STDOUT: !74 = !{!72}
// CHECK:STDOUT: !75 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.4487e305711510e9:core.Destroy.Core", scope: null, file: !38, line: 12, type: !22, spFlags: DISPFlagDefinition, unit: !39, retainedNodes: !78)
// CHECK:STDOUT: !76 = !DILocalVariable(arg: 1, scope: !75, type: !20)
// CHECK:STDOUT: !77 = !DILocation(line: 12, column: 3, scope: !75)
// CHECK:STDOUT: !78 = !{!76}
// CHECK:STDOUT: !79 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.4487e305711510e9:core.Destroy.Core", scope: null, file: !38, line: 12, type: !22, spFlags: DISPFlagDefinition, unit: !39, retainedNodes: !82)
// CHECK:STDOUT: !80 = !DILocalVariable(arg: 1, scope: !79, type: !20)
// CHECK:STDOUT: !81 = !DILocation(line: 12, column: 3, scope: !79)
// CHECK:STDOUT: !82 = !{!80}
// CHECK:STDOUT: !83 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.3c3534e993db7939:core.Destroy.Core", scope: null, file: !38, line: 12, type: !22, spFlags: DISPFlagDefinition, unit: !39, retainedNodes: !86)
// CHECK:STDOUT: !84 = !DILocalVariable(arg: 1, scope: !83, type: !20)
// CHECK:STDOUT: !85 = !DILocation(line: 12, column: 3, scope: !83)
// CHECK:STDOUT: !86 = !{!84}
// CHECK:STDOUT: !87 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.3c3534e993db7939:core.Destroy.Core", scope: null, file: !38, line: 12, type: !22, spFlags: DISPFlagDefinition, unit: !39, retainedNodes: !90)
// CHECK:STDOUT: !88 = !DILocalVariable(arg: 1, scope: !87, type: !20)
// CHECK:STDOUT: !89 = !DILocation(line: 12, column: 3, scope: !87)
// CHECK:STDOUT: !90 = !{!88}
// CHECK:STDOUT: !91 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.43ed5d0e08fc7556:core.Destroy.Core", scope: null, file: !38, line: 12, type: !22, spFlags: DISPFlagDefinition, unit: !39, retainedNodes: !94)
// CHECK:STDOUT: !92 = !DILocalVariable(arg: 1, scope: !91, type: !20)
// CHECK:STDOUT: !93 = !DILocation(line: 12, column: 3, scope: !91)
// CHECK:STDOUT: !94 = !{!92}
// CHECK:STDOUT: !95 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.43ed5d0e08fc7556:core.Destroy.Core", scope: null, file: !38, line: 12, type: !22, spFlags: DISPFlagDefinition, unit: !39, retainedNodes: !98)
// CHECK:STDOUT: !96 = !DILocalVariable(arg: 1, scope: !95, type: !20)
// CHECK:STDOUT: !97 = !DILocation(line: 12, column: 3, scope: !95)
// CHECK:STDOUT: !98 = !{!96}
// CHECK:STDOUT: !99 = distinct !DISubprogram(name: "Use", linkageName: "_CUse.Create", scope: null, file: !38, line: 15, type: !22, spFlags: DISPFlagDefinition, unit: !39, retainedNodes: !104)
// CHECK:STDOUT: !100 = !DILocalVariable(arg: 1, scope: !99, type: !20)
// CHECK:STDOUT: !102 = !DILocation(line: 16, column: 3, scope: !99)
// CHECK:STDOUT: !103 = !DILocation(line: 15, column: 1, scope: !99)
// CHECK:STDOUT: !104 = !{!100}
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
// CHECK:STDOUT: @Base.val.loc13_3 = internal constant <{ ptr, i32 }> <{ ptr @"_CBase.MemberInit.$vtable", i32 3 }>
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CFn.Base.MemberInit(ptr %self) #0 !dbg !146 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !149
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CFn.MemberInit() #0 !dbg !151 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %i.var = alloca i32, align 4, !dbg !152
// CHECK:STDOUT:   %v.var = alloca <{ ptr, i32 }>, align 8, !dbg !154
// CHECK:STDOUT:   %_.var = alloca <{ ptr, i32 }>, align 8, !dbg !159
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %i.var), !dbg !152
// CHECK:STDOUT:   store i32 3, ptr %i.var, align 4, !dbg !152
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %v.var), !dbg !154
// CHECK:STDOUT:   %.loc11_24.2.vptr = getelementptr inbounds nuw <{ ptr, i32 }>, ptr %v.var, i32 0, i32 0, !dbg !156
// CHECK:STDOUT:   %.loc11_23 = load i32, ptr %i.var, align 4, !dbg !155
// CHECK:STDOUT:   %.loc11_24.4.m = getelementptr inbounds nuw <{ ptr, i32 }>, ptr %v.var, i32 0, i32 1, !dbg !156
// CHECK:STDOUT:   store i32 %.loc11_23, ptr %.loc11_24.4.m, align 4, !dbg !156
// CHECK:STDOUT:   store ptr @"_CBase.MemberInit.$vtable", ptr %.loc11_24.2.vptr, align 8, !dbg !156
// CHECK:STDOUT:   %.loc12_4.m = getelementptr inbounds nuw <{ ptr, i32 }>, ptr %v.var, i32 0, i32 1, !dbg !158
// CHECK:STDOUT:   store i32 5, ptr %.loc12_4.m, align 4, !dbg !158
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !159
// CHECK:STDOUT:   %.loc13_24.2.vptr = getelementptr inbounds nuw <{ ptr, i32 }>, ptr %_.var, i32 0, i32 0, !dbg !160
// CHECK:STDOUT:   %.loc13_24.5.m = getelementptr inbounds nuw <{ ptr, i32 }>, ptr %_.var, i32 0, i32 1, !dbg !160
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 %_.var, ptr align 8 @Base.val.loc13_3, i64 12, i1 false), !dbg !159
// CHECK:STDOUT:   call void @"_CSelfDestruct.c6dbe0017b4c8d16:core.Destroy.Core"(ptr %_.var), !dbg !159
// CHECK:STDOUT:   call void @"_CSelfDestruct.c6dbe0017b4c8d16:core.Destroy.Core"(ptr %v.var), !dbg !154
// CHECK:STDOUT:   call void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %i.var), !dbg !152
// CHECK:STDOUT:   ret void, !dbg !162
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.454fecbcedcd86e1:core.Destroy.Core"(ptr %self) #0 !dbg !163 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !165
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !167 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !169
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.fd9d4a98e93fc879:core.Destroy.Core"(ptr %self) #0 !dbg !171 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !173
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.fd9d4a98e93fc879:core.Destroy.Core"(ptr %self) #0 !dbg !175 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.fd9d4a98e93fc879:core.Destroy.Core"(ptr %self), !dbg !177
// CHECK:STDOUT:   ret void, !dbg !177
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.c6dbe0017b4c8d16:core.Destroy.Core"(ptr %self) #0 !dbg !179 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !181
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.c6dbe0017b4c8d16:core.Destroy.Core"(ptr %self) #0 !dbg !183 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.c6dbe0017b4c8d16:core.Destroy.Core"(ptr %self), !dbg !185
// CHECK:STDOUT:   ret void, !dbg !185
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @"_CSelfDestruct.c6dbe0017b4c8d16:core.Destroy.Core", { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!142}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !21 = !{null, !20}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !40 = !{null}
// CHECK:STDOUT: !41 = !DISubroutineType(types: !40)
// CHECK:STDOUT: !141 = !DIFile(filename: "member_init.carbon", directory: "")
// CHECK:STDOUT: !142 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !141, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !143 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !144 = !{null, !143}
// CHECK:STDOUT: !145 = !DISubroutineType(types: !144)
// CHECK:STDOUT: !146 = distinct !DISubprogram(name: "Fn", linkageName: "_CFn.Base.MemberInit", scope: null, file: !141, line: 6, type: !22, spFlags: DISPFlagDefinition, unit: !142, retainedNodes: !150)
// CHECK:STDOUT: !147 = !DILocalVariable(arg: 1, scope: !146, type: !20)
// CHECK:STDOUT: !149 = !DILocation(line: 6, column: 3, scope: !146)
// CHECK:STDOUT: !150 = !{!147}
// CHECK:STDOUT: !151 = distinct !DISubprogram(name: "Fn", linkageName: "_CFn.MemberInit", scope: null, file: !141, line: 9, type: !41, spFlags: DISPFlagDefinition, unit: !142)
// CHECK:STDOUT: !152 = !DILocation(line: 10, column: 3, scope: !151)
// CHECK:STDOUT: !154 = !DILocation(line: 11, column: 3, scope: !151)
// CHECK:STDOUT: !155 = !DILocation(line: 11, column: 23, scope: !151)
// CHECK:STDOUT: !156 = !DILocation(line: 11, column: 17, scope: !151)
// CHECK:STDOUT: !158 = !DILocation(line: 12, column: 3, scope: !151)
// CHECK:STDOUT: !159 = !DILocation(line: 13, column: 3, scope: !151)
// CHECK:STDOUT: !160 = !DILocation(line: 13, column: 17, scope: !151)
// CHECK:STDOUT: !162 = !DILocation(line: 9, column: 1, scope: !151)
// CHECK:STDOUT: !163 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.454fecbcedcd86e1:core.Destroy.Core", scope: null, file: !141, line: 13, type: !22, spFlags: DISPFlagDefinition, unit: !142, retainedNodes: !166)
// CHECK:STDOUT: !164 = !DILocalVariable(arg: 1, scope: !163, type: !20)
// CHECK:STDOUT: !165 = !DILocation(line: 13, column: 3, scope: !163)
// CHECK:STDOUT: !166 = !{!164}
// CHECK:STDOUT: !167 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.f58524831c37807e:core.Destroy.Core", scope: null, file: !141, line: 13, type: !145, spFlags: DISPFlagDefinition, unit: !142, retainedNodes: !170)
// CHECK:STDOUT: !168 = !DILocalVariable(arg: 1, scope: !167, type: !143)
// CHECK:STDOUT: !169 = !DILocation(line: 13, column: 3, scope: !167)
// CHECK:STDOUT: !170 = !{!168}
// CHECK:STDOUT: !171 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.fd9d4a98e93fc879:core.Destroy.Core", scope: null, file: !141, line: 13, type: !22, spFlags: DISPFlagDefinition, unit: !142, retainedNodes: !174)
// CHECK:STDOUT: !172 = !DILocalVariable(arg: 1, scope: !171, type: !20)
// CHECK:STDOUT: !173 = !DILocation(line: 13, column: 3, scope: !171)
// CHECK:STDOUT: !174 = !{!172}
// CHECK:STDOUT: !175 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.fd9d4a98e93fc879:core.Destroy.Core", scope: null, file: !141, line: 13, type: !22, spFlags: DISPFlagDefinition, unit: !142, retainedNodes: !178)
// CHECK:STDOUT: !176 = !DILocalVariable(arg: 1, scope: !175, type: !20)
// CHECK:STDOUT: !177 = !DILocation(line: 13, column: 3, scope: !175)
// CHECK:STDOUT: !178 = !{!176}
// CHECK:STDOUT: !179 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.c6dbe0017b4c8d16:core.Destroy.Core", scope: null, file: !141, line: 13, type: !22, spFlags: DISPFlagDefinition, unit: !142, retainedNodes: !182)
// CHECK:STDOUT: !180 = !DILocalVariable(arg: 1, scope: !179, type: !20)
// CHECK:STDOUT: !181 = !DILocation(line: 13, column: 3, scope: !179)
// CHECK:STDOUT: !182 = !{!180}
// CHECK:STDOUT: !183 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.c6dbe0017b4c8d16:core.Destroy.Core", scope: null, file: !141, line: 13, type: !22, spFlags: DISPFlagDefinition, unit: !142, retainedNodes: !186)
// CHECK:STDOUT: !184 = !DILocalVariable(arg: 1, scope: !183, type: !20)
// CHECK:STDOUT: !185 = !DILocation(line: 13, column: 3, scope: !183)
// CHECK:STDOUT: !186 = !{!184}
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
// CHECK:STDOUT: define void @_CUse.Main() #0 !dbg !189 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca { { ptr } }, align 8, !dbg !190
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !190
// CHECK:STDOUT:   %.loc13_32.2.base = getelementptr inbounds nuw { { ptr } }, ptr %_.var, i32 0, i32 0, !dbg !191
// CHECK:STDOUT:   %.loc13_31.2.vptr = getelementptr inbounds nuw { ptr }, ptr %.loc13_32.2.base, i32 0, i32 0, !dbg !192
// CHECK:STDOUT:   %.loc13_32.6.base = getelementptr inbounds nuw { { ptr } }, ptr %_.var, i32 0, i32 0, !dbg !191
// CHECK:STDOUT:   %.loc13_32.7.vptr = getelementptr inbounds nuw { ptr }, ptr %.loc13_32.6.base, i32 0, i32 0, !dbg !191
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 %_.var, ptr align 8 @Derived.val.loc13_32.5, i64 8, i1 false), !dbg !191
// CHECK:STDOUT:   store ptr @"_CDerived.Main.$vtable", ptr %.loc13_32.7.vptr, align 8, !dbg !191
// CHECK:STDOUT:   call void @"_CSelfDestruct.b25525d21a9bd472:core.Destroy.Core"(ptr %_.var), !dbg !190
// CHECK:STDOUT:   ret void, !dbg !194
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.454fecbcedcd86e1:core.Destroy.Core"(ptr %self) #0 !dbg !195 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !197
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.95528a86e1127e2d:core.Destroy.Core"(ptr %self) #0 !dbg !199 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !201
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.95528a86e1127e2d:core.Destroy.Core"(ptr %self) #0 !dbg !203 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.95528a86e1127e2d:core.Destroy.Core"(ptr %self), !dbg !205
// CHECK:STDOUT:   ret void, !dbg !205
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.294df737cd074d60:core.Destroy.Core"(ptr %self) #0 !dbg !207 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !209
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.294df737cd074d60:core.Destroy.Core"(ptr %self) #0 !dbg !211 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.294df737cd074d60:core.Destroy.Core"(ptr %self), !dbg !213
// CHECK:STDOUT:   ret void, !dbg !213
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.caed8a96bf328d09:core.Destroy.Core"(ptr %self) #0 !dbg !215 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !217
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.caed8a96bf328d09:core.Destroy.Core"(ptr %self) #0 !dbg !219 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.caed8a96bf328d09:core.Destroy.Core"(ptr %self), !dbg !221
// CHECK:STDOUT:   ret void, !dbg !221
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.b25525d21a9bd472:core.Destroy.Core"(ptr %self) #0 !dbg !223 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !225
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.b25525d21a9bd472:core.Destroy.Core"(ptr %self) #0 !dbg !227 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.b25525d21a9bd472:core.Destroy.Core"(ptr %self), !dbg !229
// CHECK:STDOUT:   ret void, !dbg !229
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!188}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !21 = !{null, !20}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !40 = !{null}
// CHECK:STDOUT: !41 = !DISubroutineType(types: !40)
// CHECK:STDOUT: !187 = !DIFile(filename: "member_brace_init.carbon", directory: "")
// CHECK:STDOUT: !188 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !187, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !189 = distinct !DISubprogram(name: "Use", linkageName: "_CUse.Main", scope: null, file: !187, line: 12, type: !41, spFlags: DISPFlagDefinition, unit: !188)
// CHECK:STDOUT: !190 = !DILocation(line: 13, column: 3, scope: !189)
// CHECK:STDOUT: !191 = !DILocation(line: 13, column: 21, scope: !189)
// CHECK:STDOUT: !192 = !DILocation(line: 13, column: 30, scope: !189)
// CHECK:STDOUT: !194 = !DILocation(line: 12, column: 1, scope: !189)
// CHECK:STDOUT: !195 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.454fecbcedcd86e1:core.Destroy.Core", scope: null, file: !187, line: 13, type: !22, spFlags: DISPFlagDefinition, unit: !188, retainedNodes: !198)
// CHECK:STDOUT: !196 = !DILocalVariable(arg: 1, scope: !195, type: !20)
// CHECK:STDOUT: !197 = !DILocation(line: 13, column: 3, scope: !195)
// CHECK:STDOUT: !198 = !{!196}
// CHECK:STDOUT: !199 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.95528a86e1127e2d:core.Destroy.Core", scope: null, file: !187, line: 13, type: !22, spFlags: DISPFlagDefinition, unit: !188, retainedNodes: !202)
// CHECK:STDOUT: !200 = !DILocalVariable(arg: 1, scope: !199, type: !20)
// CHECK:STDOUT: !201 = !DILocation(line: 13, column: 3, scope: !199)
// CHECK:STDOUT: !202 = !{!200}
// CHECK:STDOUT: !203 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.95528a86e1127e2d:core.Destroy.Core", scope: null, file: !187, line: 13, type: !22, spFlags: DISPFlagDefinition, unit: !188, retainedNodes: !206)
// CHECK:STDOUT: !204 = !DILocalVariable(arg: 1, scope: !203, type: !20)
// CHECK:STDOUT: !205 = !DILocation(line: 13, column: 3, scope: !203)
// CHECK:STDOUT: !206 = !{!204}
// CHECK:STDOUT: !207 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.294df737cd074d60:core.Destroy.Core", scope: null, file: !187, line: 13, type: !22, spFlags: DISPFlagDefinition, unit: !188, retainedNodes: !210)
// CHECK:STDOUT: !208 = !DILocalVariable(arg: 1, scope: !207, type: !20)
// CHECK:STDOUT: !209 = !DILocation(line: 13, column: 3, scope: !207)
// CHECK:STDOUT: !210 = !{!208}
// CHECK:STDOUT: !211 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.294df737cd074d60:core.Destroy.Core", scope: null, file: !187, line: 13, type: !22, spFlags: DISPFlagDefinition, unit: !188, retainedNodes: !214)
// CHECK:STDOUT: !212 = !DILocalVariable(arg: 1, scope: !211, type: !20)
// CHECK:STDOUT: !213 = !DILocation(line: 13, column: 3, scope: !211)
// CHECK:STDOUT: !214 = !{!212}
// CHECK:STDOUT: !215 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.caed8a96bf328d09:core.Destroy.Core", scope: null, file: !187, line: 13, type: !22, spFlags: DISPFlagDefinition, unit: !188, retainedNodes: !218)
// CHECK:STDOUT: !216 = !DILocalVariable(arg: 1, scope: !215, type: !20)
// CHECK:STDOUT: !217 = !DILocation(line: 13, column: 3, scope: !215)
// CHECK:STDOUT: !218 = !{!216}
// CHECK:STDOUT: !219 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.caed8a96bf328d09:core.Destroy.Core", scope: null, file: !187, line: 13, type: !22, spFlags: DISPFlagDefinition, unit: !188, retainedNodes: !222)
// CHECK:STDOUT: !220 = !DILocalVariable(arg: 1, scope: !219, type: !20)
// CHECK:STDOUT: !221 = !DILocation(line: 13, column: 3, scope: !219)
// CHECK:STDOUT: !222 = !{!220}
// CHECK:STDOUT: !223 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.b25525d21a9bd472:core.Destroy.Core", scope: null, file: !187, line: 13, type: !22, spFlags: DISPFlagDefinition, unit: !188, retainedNodes: !226)
// CHECK:STDOUT: !224 = !DILocalVariable(arg: 1, scope: !223, type: !20)
// CHECK:STDOUT: !225 = !DILocation(line: 13, column: 3, scope: !223)
// CHECK:STDOUT: !226 = !{!224}
// CHECK:STDOUT: !227 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.b25525d21a9bd472:core.Destroy.Core", scope: null, file: !187, line: 13, type: !22, spFlags: DISPFlagDefinition, unit: !188, retainedNodes: !230)
// CHECK:STDOUT: !228 = !DILocalVariable(arg: 1, scope: !227, type: !20)
// CHECK:STDOUT: !229 = !DILocation(line: 13, column: 3, scope: !227)
// CHECK:STDOUT: !230 = !{!228}
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
// CHECK:STDOUT: define void @_CUse.Main(ptr %b) #0 !dbg !233 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Base.F.call.vtable = load ptr, ptr %b, align 8, !dbg !236
// CHECK:STDOUT:   %Base.F.call = call ptr @llvm.load.relative.i32(ptr %Base.F.call.vtable, i32 0), !dbg !236
// CHECK:STDOUT:   call void %Base.F.call(ptr %b), !dbg !236
// CHECK:STDOUT:   ret void, !dbg !237
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: read)
// CHECK:STDOUT: declare ptr @llvm.load.relative.i32(ptr, i32) #1
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: read) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!232}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !21 = !{null, !20}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !231 = !DIFile(filename: "call.carbon", directory: "")
// CHECK:STDOUT: !232 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !231, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !233 = distinct !DISubprogram(name: "Use", linkageName: "_CUse.Main", scope: null, file: !231, line: 8, type: !22, spFlags: DISPFlagDefinition, unit: !232, retainedNodes: !238)
// CHECK:STDOUT: !234 = !DILocalVariable(arg: 1, scope: !233, type: !20)
// CHECK:STDOUT: !236 = !DILocation(line: 9, column: 3, scope: !233)
// CHECK:STDOUT: !237 = !DILocation(line: 8, column: 1, scope: !233)
// CHECK:STDOUT: !238 = !{!234}
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
// CHECK:STDOUT: define weak_odr void @"_CF:thunk:Base.Main:Derived.Main"(ptr %self) #0 !dbg !241 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Derived.Main(ptr %self), !dbg !244
// CHECK:STDOUT:   ret void, !dbg !244
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CUse.Main() #1 !dbg !246 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %v.var = alloca { { ptr } }, align 8, !dbg !247
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %v.var), !dbg !247
// CHECK:STDOUT:   %.loc14_32.2.base = getelementptr inbounds nuw { { ptr } }, ptr %v.var, i32 0, i32 0, !dbg !248
// CHECK:STDOUT:   %.loc14_31.2.vptr = getelementptr inbounds nuw { ptr }, ptr %.loc14_32.2.base, i32 0, i32 0, !dbg !249
// CHECK:STDOUT:   %.loc14_32.6.base = getelementptr inbounds nuw { { ptr } }, ptr %v.var, i32 0, i32 0, !dbg !248
// CHECK:STDOUT:   %.loc14_32.7.vptr = getelementptr inbounds nuw { ptr }, ptr %.loc14_32.6.base, i32 0, i32 0, !dbg !248
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 %v.var, ptr align 8 @Derived.val.loc14_32.5, i64 8, i1 false), !dbg !248
// CHECK:STDOUT:   store ptr @"_CDerived.Main.$vtable", ptr %.loc14_32.7.vptr, align 8, !dbg !248
// CHECK:STDOUT:   %.loc15_3.1.base = getelementptr inbounds nuw { { ptr } }, ptr %v.var, i32 0, i32 0, !dbg !251
// CHECK:STDOUT:   %Base.F.call.vtable = load ptr, ptr %.loc15_3.1.base, align 8, !dbg !251
// CHECK:STDOUT:   %Base.F.call = call ptr @llvm.load.relative.i32(ptr %Base.F.call.vtable, i32 0), !dbg !251
// CHECK:STDOUT:   call void %Base.F.call(ptr %.loc15_3.1.base), !dbg !251
// CHECK:STDOUT:   call void @"_CSelfDestruct.b25525d21a9bd472:core.Destroy.Core"(ptr %v.var), !dbg !247
// CHECK:STDOUT:   ret void, !dbg !252
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.454fecbcedcd86e1:core.Destroy.Core"(ptr %self) #1 !dbg !253 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !255
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.95528a86e1127e2d:core.Destroy.Core"(ptr %self) #1 !dbg !257 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !259
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.95528a86e1127e2d:core.Destroy.Core"(ptr %self) #1 !dbg !261 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.95528a86e1127e2d:core.Destroy.Core"(ptr %self), !dbg !263
// CHECK:STDOUT:   ret void, !dbg !263
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.294df737cd074d60:core.Destroy.Core"(ptr %self) #1 !dbg !265 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !267
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.294df737cd074d60:core.Destroy.Core"(ptr %self) #1 !dbg !269 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.294df737cd074d60:core.Destroy.Core"(ptr %self), !dbg !271
// CHECK:STDOUT:   ret void, !dbg !271
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.caed8a96bf328d09:core.Destroy.Core"(ptr %self) #1 !dbg !273 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !275
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.caed8a96bf328d09:core.Destroy.Core"(ptr %self) #1 !dbg !277 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.caed8a96bf328d09:core.Destroy.Core"(ptr %self), !dbg !279
// CHECK:STDOUT:   ret void, !dbg !279
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.b25525d21a9bd472:core.Destroy.Core"(ptr %self) #1 !dbg !281 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !283
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.b25525d21a9bd472:core.Destroy.Core"(ptr %self) #1 !dbg !285 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.b25525d21a9bd472:core.Destroy.Core"(ptr %self), !dbg !287
// CHECK:STDOUT:   ret void, !dbg !287
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!240}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !21 = !{null, !20}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !40 = !{null}
// CHECK:STDOUT: !41 = !DISubroutineType(types: !40)
// CHECK:STDOUT: !239 = !DIFile(filename: "call_impl.carbon", directory: "")
// CHECK:STDOUT: !240 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !239, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !241 = distinct !DISubprogram(name: "F", linkageName: "_CF:thunk:Base.Main:Derived.Main", scope: null, file: !239, line: 10, type: !22, spFlags: DISPFlagDefinition, unit: !240, retainedNodes: !245)
// CHECK:STDOUT: !242 = !DILocalVariable(arg: 1, scope: !241, type: !20)
// CHECK:STDOUT: !244 = !DILocation(line: 10, column: 3, scope: !241)
// CHECK:STDOUT: !245 = !{!242}
// CHECK:STDOUT: !246 = distinct !DISubprogram(name: "Use", linkageName: "_CUse.Main", scope: null, file: !239, line: 13, type: !41, spFlags: DISPFlagDefinition, unit: !240)
// CHECK:STDOUT: !247 = !DILocation(line: 14, column: 3, scope: !246)
// CHECK:STDOUT: !248 = !DILocation(line: 14, column: 21, scope: !246)
// CHECK:STDOUT: !249 = !DILocation(line: 14, column: 30, scope: !246)
// CHECK:STDOUT: !251 = !DILocation(line: 15, column: 3, scope: !246)
// CHECK:STDOUT: !252 = !DILocation(line: 13, column: 1, scope: !246)
// CHECK:STDOUT: !253 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.454fecbcedcd86e1:core.Destroy.Core", scope: null, file: !239, line: 14, type: !22, spFlags: DISPFlagDefinition, unit: !240, retainedNodes: !256)
// CHECK:STDOUT: !254 = !DILocalVariable(arg: 1, scope: !253, type: !20)
// CHECK:STDOUT: !255 = !DILocation(line: 14, column: 3, scope: !253)
// CHECK:STDOUT: !256 = !{!254}
// CHECK:STDOUT: !257 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.95528a86e1127e2d:core.Destroy.Core", scope: null, file: !239, line: 14, type: !22, spFlags: DISPFlagDefinition, unit: !240, retainedNodes: !260)
// CHECK:STDOUT: !258 = !DILocalVariable(arg: 1, scope: !257, type: !20)
// CHECK:STDOUT: !259 = !DILocation(line: 14, column: 3, scope: !257)
// CHECK:STDOUT: !260 = !{!258}
// CHECK:STDOUT: !261 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.95528a86e1127e2d:core.Destroy.Core", scope: null, file: !239, line: 14, type: !22, spFlags: DISPFlagDefinition, unit: !240, retainedNodes: !264)
// CHECK:STDOUT: !262 = !DILocalVariable(arg: 1, scope: !261, type: !20)
// CHECK:STDOUT: !263 = !DILocation(line: 14, column: 3, scope: !261)
// CHECK:STDOUT: !264 = !{!262}
// CHECK:STDOUT: !265 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.294df737cd074d60:core.Destroy.Core", scope: null, file: !239, line: 14, type: !22, spFlags: DISPFlagDefinition, unit: !240, retainedNodes: !268)
// CHECK:STDOUT: !266 = !DILocalVariable(arg: 1, scope: !265, type: !20)
// CHECK:STDOUT: !267 = !DILocation(line: 14, column: 3, scope: !265)
// CHECK:STDOUT: !268 = !{!266}
// CHECK:STDOUT: !269 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.294df737cd074d60:core.Destroy.Core", scope: null, file: !239, line: 14, type: !22, spFlags: DISPFlagDefinition, unit: !240, retainedNodes: !272)
// CHECK:STDOUT: !270 = !DILocalVariable(arg: 1, scope: !269, type: !20)
// CHECK:STDOUT: !271 = !DILocation(line: 14, column: 3, scope: !269)
// CHECK:STDOUT: !272 = !{!270}
// CHECK:STDOUT: !273 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.caed8a96bf328d09:core.Destroy.Core", scope: null, file: !239, line: 14, type: !22, spFlags: DISPFlagDefinition, unit: !240, retainedNodes: !276)
// CHECK:STDOUT: !274 = !DILocalVariable(arg: 1, scope: !273, type: !20)
// CHECK:STDOUT: !275 = !DILocation(line: 14, column: 3, scope: !273)
// CHECK:STDOUT: !276 = !{!274}
// CHECK:STDOUT: !277 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.caed8a96bf328d09:core.Destroy.Core", scope: null, file: !239, line: 14, type: !22, spFlags: DISPFlagDefinition, unit: !240, retainedNodes: !280)
// CHECK:STDOUT: !278 = !DILocalVariable(arg: 1, scope: !277, type: !20)
// CHECK:STDOUT: !279 = !DILocation(line: 14, column: 3, scope: !277)
// CHECK:STDOUT: !280 = !{!278}
// CHECK:STDOUT: !281 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.b25525d21a9bd472:core.Destroy.Core", scope: null, file: !239, line: 14, type: !22, spFlags: DISPFlagDefinition, unit: !240, retainedNodes: !284)
// CHECK:STDOUT: !282 = !DILocalVariable(arg: 1, scope: !281, type: !20)
// CHECK:STDOUT: !283 = !DILocation(line: 14, column: 3, scope: !281)
// CHECK:STDOUT: !284 = !{!282}
// CHECK:STDOUT: !285 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.b25525d21a9bd472:core.Destroy.Core", scope: null, file: !239, line: 14, type: !22, spFlags: DISPFlagDefinition, unit: !240, retainedNodes: !288)
// CHECK:STDOUT: !286 = !DILocalVariable(arg: 1, scope: !285, type: !20)
// CHECK:STDOUT: !287 = !DILocation(line: 14, column: 3, scope: !285)
// CHECK:STDOUT: !288 = !{!286}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'generic_noop.carbon'
// CHECK:STDOUT: source_filename = "generic_noop.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!290}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !289 = !DIFile(filename: "generic_noop.carbon", directory: "")
// CHECK:STDOUT: !290 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !289, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'generic_use.carbon'
// CHECK:STDOUT: source_filename = "generic_use.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @"_CBase.Main.$vtable.2f8a0042309f394e" = unnamed_addr constant [1 x i32] [i32 trunc (i64 sub (i64 ptrtoint (ptr @_CF.Base.Main.2f8a0042309f394e to i64), i64 ptrtoint (ptr @"_CBase.Main.$vtable.2f8a0042309f394e" to i64)) to i32)]
// CHECK:STDOUT: @"_CBase.Main.$vtable.d7d44a05ee79f1b8" = unnamed_addr constant [1 x i32] [i32 trunc (i64 sub (i64 ptrtoint (ptr @_CF.Base.Main.d7d44a05ee79f1b8 to i64), i64 ptrtoint (ptr @"_CBase.Main.$vtable.d7d44a05ee79f1b8" to i64)) to i32)]
// CHECK:STDOUT: @Base.val.a7b.loc16_3 = internal constant { ptr } { ptr @"_CBase.Main.$vtable.2f8a0042309f394e" }
// CHECK:STDOUT: @Base.val.3f1.loc17_3 = internal constant { ptr } { ptr @"_CBase.Main.$vtable.d7d44a05ee79f1b8" }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !293 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var.loc16 = alloca { ptr }, align 8, !dbg !294
// CHECK:STDOUT:   %_.var.loc17 = alloca { ptr }, align 8, !dbg !297
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc16), !dbg !294
// CHECK:STDOUT:   %.loc16_22.2.vptr = getelementptr inbounds nuw { ptr }, ptr %_.var.loc16, i32 0, i32 0, !dbg !295
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 %_.var.loc16, ptr align 8 @Base.val.a7b.loc16_3, i64 8, i1 false), !dbg !294
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc17), !dbg !297
// CHECK:STDOUT:   %.loc17_22.2.vptr = getelementptr inbounds nuw { ptr }, ptr %_.var.loc17, i32 0, i32 0, !dbg !298
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 %_.var.loc17, ptr align 8 @Base.val.3f1.loc17_3, i64 8, i1 false), !dbg !297
// CHECK:STDOUT:   call void @"_CSelfDestruct.c3ce2497e05f4476:core.Destroy.Core"(ptr %_.var.loc17), !dbg !297
// CHECK:STDOUT:   call void @"_CSelfDestruct.b64fd0f098bf5c8e:core.Destroy.Core"(ptr %_.var.loc16), !dbg !294
// CHECK:STDOUT:   ret void, !dbg !300
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.cb8adced19392b99:core.Destroy.Core"(ptr %self) #0 !dbg !301 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !303
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.13f86630f401913f:core.Destroy.Core"(ptr %self) #0 !dbg !305 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !307
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.f2f12ab22a0a6b87:core.Destroy.Core"(ptr %self) #0 !dbg !309 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !311
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.f2f12ab22a0a6b87:core.Destroy.Core"(ptr %self) #0 !dbg !313 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.f2f12ab22a0a6b87:core.Destroy.Core"(ptr %self), !dbg !315
// CHECK:STDOUT:   ret void, !dbg !315
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.8578661a61c8bbbf:core.Destroy.Core"(ptr %self) #0 !dbg !317 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !319
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.8578661a61c8bbbf:core.Destroy.Core"(ptr %self) #0 !dbg !321 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.8578661a61c8bbbf:core.Destroy.Core"(ptr %self), !dbg !323
// CHECK:STDOUT:   ret void, !dbg !323
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.454fecbcedcd86e1:core.Destroy.Core"(ptr %self) #0 !dbg !325 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !327
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.95528a86e1127e2d:core.Destroy.Core"(ptr %self) #0 !dbg !329 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !331
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.95528a86e1127e2d:core.Destroy.Core"(ptr %self) #0 !dbg !333 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.95528a86e1127e2d:core.Destroy.Core"(ptr %self), !dbg !335
// CHECK:STDOUT:   ret void, !dbg !335
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.c3ce2497e05f4476:core.Destroy.Core"(ptr %self) #0 !dbg !337 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !339
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.c3ce2497e05f4476:core.Destroy.Core"(ptr %self) #0 !dbg !341 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.c3ce2497e05f4476:core.Destroy.Core"(ptr %self), !dbg !343
// CHECK:STDOUT:   ret void, !dbg !343
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.b64fd0f098bf5c8e:core.Destroy.Core"(ptr %self) #0 !dbg !345 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !347
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.b64fd0f098bf5c8e:core.Destroy.Core"(ptr %self) #0 !dbg !349 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.b64fd0f098bf5c8e:core.Destroy.Core"(ptr %self), !dbg !351
// CHECK:STDOUT:   ret void, !dbg !351
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Base.Main.2f8a0042309f394e(ptr %self) #0 !dbg !353 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca {}, align 1, !dbg !356
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !356
// CHECK:STDOUT:   call void @"_CSelfDestruct.13f86630f401913f:core.Destroy.Core"(ptr %_.var), !dbg !356
// CHECK:STDOUT:   ret void, !dbg !358
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Base.Main.d7d44a05ee79f1b8(ptr %self) #0 !dbg !360 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca { {} }, align 1, !dbg !363
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !363
// CHECK:STDOUT:   call void @"_CSelfDestruct.8578661a61c8bbbf:core.Destroy.Core"(ptr %_.var), !dbg !363
// CHECK:STDOUT:   ret void, !dbg !365
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!292}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !21 = !{null, !20}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !40 = !{null}
// CHECK:STDOUT: !41 = !DISubroutineType(types: !40)
// CHECK:STDOUT: !291 = !DIFile(filename: "generic_use.carbon", directory: "")
// CHECK:STDOUT: !292 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !291, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !293 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !291, line: 15, type: !41, spFlags: DISPFlagDefinition, unit: !292)
// CHECK:STDOUT: !294 = !DILocation(line: 16, column: 3, scope: !293)
// CHECK:STDOUT: !295 = !DILocation(line: 16, column: 21, scope: !293)
// CHECK:STDOUT: !297 = !DILocation(line: 17, column: 3, scope: !293)
// CHECK:STDOUT: !298 = !DILocation(line: 17, column: 21, scope: !293)
// CHECK:STDOUT: !300 = !DILocation(line: 15, column: 1, scope: !293)
// CHECK:STDOUT: !301 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.cb8adced19392b99:core.Destroy.Core", scope: null, file: !291, line: 16, type: !22, spFlags: DISPFlagDefinition, unit: !292, retainedNodes: !304)
// CHECK:STDOUT: !302 = !DILocalVariable(arg: 1, scope: !301, type: !20)
// CHECK:STDOUT: !303 = !DILocation(line: 16, column: 10, scope: !301)
// CHECK:STDOUT: !304 = !{!302}
// CHECK:STDOUT: !305 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.13f86630f401913f:core.Destroy.Core", scope: null, file: !291, line: 16, type: !22, spFlags: DISPFlagDefinition, unit: !292, retainedNodes: !308)
// CHECK:STDOUT: !306 = !DILocalVariable(arg: 1, scope: !305, type: !20)
// CHECK:STDOUT: !307 = !DILocation(line: 16, column: 10, scope: !305)
// CHECK:STDOUT: !308 = !{!306}
// CHECK:STDOUT: !309 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.f2f12ab22a0a6b87:core.Destroy.Core", scope: null, file: !291, line: 17, type: !22, spFlags: DISPFlagDefinition, unit: !292, retainedNodes: !312)
// CHECK:STDOUT: !310 = !DILocalVariable(arg: 1, scope: !309, type: !20)
// CHECK:STDOUT: !311 = !DILocation(line: 17, column: 10, scope: !309)
// CHECK:STDOUT: !312 = !{!310}
// CHECK:STDOUT: !313 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.f2f12ab22a0a6b87:core.Destroy.Core", scope: null, file: !291, line: 17, type: !22, spFlags: DISPFlagDefinition, unit: !292, retainedNodes: !316)
// CHECK:STDOUT: !314 = !DILocalVariable(arg: 1, scope: !313, type: !20)
// CHECK:STDOUT: !315 = !DILocation(line: 17, column: 10, scope: !313)
// CHECK:STDOUT: !316 = !{!314}
// CHECK:STDOUT: !317 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.8578661a61c8bbbf:core.Destroy.Core", scope: null, file: !291, line: 17, type: !22, spFlags: DISPFlagDefinition, unit: !292, retainedNodes: !320)
// CHECK:STDOUT: !318 = !DILocalVariable(arg: 1, scope: !317, type: !20)
// CHECK:STDOUT: !319 = !DILocation(line: 17, column: 10, scope: !317)
// CHECK:STDOUT: !320 = !{!318}
// CHECK:STDOUT: !321 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.8578661a61c8bbbf:core.Destroy.Core", scope: null, file: !291, line: 17, type: !22, spFlags: DISPFlagDefinition, unit: !292, retainedNodes: !324)
// CHECK:STDOUT: !322 = !DILocalVariable(arg: 1, scope: !321, type: !20)
// CHECK:STDOUT: !323 = !DILocation(line: 17, column: 10, scope: !321)
// CHECK:STDOUT: !324 = !{!322}
// CHECK:STDOUT: !325 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.454fecbcedcd86e1:core.Destroy.Core", scope: null, file: !291, line: 17, type: !22, spFlags: DISPFlagDefinition, unit: !292, retainedNodes: !328)
// CHECK:STDOUT: !326 = !DILocalVariable(arg: 1, scope: !325, type: !20)
// CHECK:STDOUT: !327 = !DILocation(line: 17, column: 3, scope: !325)
// CHECK:STDOUT: !328 = !{!326}
// CHECK:STDOUT: !329 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.95528a86e1127e2d:core.Destroy.Core", scope: null, file: !291, line: 17, type: !22, spFlags: DISPFlagDefinition, unit: !292, retainedNodes: !332)
// CHECK:STDOUT: !330 = !DILocalVariable(arg: 1, scope: !329, type: !20)
// CHECK:STDOUT: !331 = !DILocation(line: 17, column: 3, scope: !329)
// CHECK:STDOUT: !332 = !{!330}
// CHECK:STDOUT: !333 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.95528a86e1127e2d:core.Destroy.Core", scope: null, file: !291, line: 17, type: !22, spFlags: DISPFlagDefinition, unit: !292, retainedNodes: !336)
// CHECK:STDOUT: !334 = !DILocalVariable(arg: 1, scope: !333, type: !20)
// CHECK:STDOUT: !335 = !DILocation(line: 17, column: 3, scope: !333)
// CHECK:STDOUT: !336 = !{!334}
// CHECK:STDOUT: !337 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.c3ce2497e05f4476:core.Destroy.Core", scope: null, file: !291, line: 17, type: !22, spFlags: DISPFlagDefinition, unit: !292, retainedNodes: !340)
// CHECK:STDOUT: !338 = !DILocalVariable(arg: 1, scope: !337, type: !20)
// CHECK:STDOUT: !339 = !DILocation(line: 17, column: 3, scope: !337)
// CHECK:STDOUT: !340 = !{!338}
// CHECK:STDOUT: !341 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.c3ce2497e05f4476:core.Destroy.Core", scope: null, file: !291, line: 17, type: !22, spFlags: DISPFlagDefinition, unit: !292, retainedNodes: !344)
// CHECK:STDOUT: !342 = !DILocalVariable(arg: 1, scope: !341, type: !20)
// CHECK:STDOUT: !343 = !DILocation(line: 17, column: 3, scope: !341)
// CHECK:STDOUT: !344 = !{!342}
// CHECK:STDOUT: !345 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.b64fd0f098bf5c8e:core.Destroy.Core", scope: null, file: !291, line: 16, type: !22, spFlags: DISPFlagDefinition, unit: !292, retainedNodes: !348)
// CHECK:STDOUT: !346 = !DILocalVariable(arg: 1, scope: !345, type: !20)
// CHECK:STDOUT: !347 = !DILocation(line: 16, column: 3, scope: !345)
// CHECK:STDOUT: !348 = !{!346}
// CHECK:STDOUT: !349 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.b64fd0f098bf5c8e:core.Destroy.Core", scope: null, file: !291, line: 16, type: !22, spFlags: DISPFlagDefinition, unit: !292, retainedNodes: !352)
// CHECK:STDOUT: !350 = !DILocalVariable(arg: 1, scope: !349, type: !20)
// CHECK:STDOUT: !351 = !DILocation(line: 16, column: 3, scope: !349)
// CHECK:STDOUT: !352 = !{!350}
// CHECK:STDOUT: !353 = distinct !DISubprogram(name: "F", linkageName: "_CF.Base.Main.2f8a0042309f394e", scope: null, file: !291, line: 5, type: !22, spFlags: DISPFlagDefinition, unit: !292, retainedNodes: !359)
// CHECK:STDOUT: !354 = !DILocalVariable(arg: 1, scope: !353, type: !20)
// CHECK:STDOUT: !356 = !DILocation(line: 6, column: 5, scope: !353)
// CHECK:STDOUT: !358 = !DILocation(line: 5, column: 3, scope: !353)
// CHECK:STDOUT: !359 = !{!354}
// CHECK:STDOUT: !360 = distinct !DISubprogram(name: "F", linkageName: "_CF.Base.Main.d7d44a05ee79f1b8", scope: null, file: !291, line: 5, type: !22, spFlags: DISPFlagDefinition, unit: !292, retainedNodes: !366)
// CHECK:STDOUT: !361 = !DILocalVariable(arg: 1, scope: !360, type: !20)
// CHECK:STDOUT: !363 = !DILocation(line: 6, column: 5, scope: !360)
// CHECK:STDOUT: !365 = !DILocation(line: 5, column: 3, scope: !360)
// CHECK:STDOUT: !366 = !{!361}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'generic_base.carbon'
// CHECK:STDOUT: source_filename = "generic_base.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @"_CDerived.Main.$vtable" = unnamed_addr constant [1 x i32] [i32 trunc (i64 sub (i64 ptrtoint (ptr @_CF.Base.Main.6d6a4a2512463e59 to i64), i64 ptrtoint (ptr @"_CDerived.Main.$vtable" to i64)) to i32)]
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMake.Main() #0 !dbg !369 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca { { ptr } }, align 8, !dbg !370
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !370
// CHECK:STDOUT:   call void @"_CSelfDestruct.b25525d21a9bd472:core.Destroy.Core"(ptr %_.var), !dbg !370
// CHECK:STDOUT:   ret void, !dbg !372
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.454fecbcedcd86e1:core.Destroy.Core"(ptr %self) #0 !dbg !373 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !375
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.95528a86e1127e2d:core.Destroy.Core"(ptr %self) #0 !dbg !377 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !379
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.95528a86e1127e2d:core.Destroy.Core"(ptr %self) #0 !dbg !381 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.95528a86e1127e2d:core.Destroy.Core"(ptr %self), !dbg !383
// CHECK:STDOUT:   ret void, !dbg !383
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.5b23b4f14feb87e1:core.Destroy.Core"(ptr %self) #0 !dbg !385 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !387
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.5b23b4f14feb87e1:core.Destroy.Core"(ptr %self) #0 !dbg !389 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.5b23b4f14feb87e1:core.Destroy.Core"(ptr %self), !dbg !391
// CHECK:STDOUT:   ret void, !dbg !391
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.2f190654c863951f:core.Destroy.Core"(ptr %self) #0 !dbg !393 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !395
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.2f190654c863951f:core.Destroy.Core"(ptr %self) #0 !dbg !397 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.2f190654c863951f:core.Destroy.Core"(ptr %self), !dbg !399
// CHECK:STDOUT:   ret void, !dbg !399
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.b25525d21a9bd472:core.Destroy.Core"(ptr %self) #0 !dbg !401 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !403
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.b25525d21a9bd472:core.Destroy.Core"(ptr %self) #0 !dbg !405 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.b25525d21a9bd472:core.Destroy.Core"(ptr %self), !dbg !407
// CHECK:STDOUT:   ret void, !dbg !407
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Base.Main.6d6a4a2512463e59(ptr %self) #0 !dbg !409 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !412
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!368}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !21 = !{null, !20}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !40 = !{null}
// CHECK:STDOUT: !41 = !DISubroutineType(types: !40)
// CHECK:STDOUT: !367 = !DIFile(filename: "generic_base.carbon", directory: "")
// CHECK:STDOUT: !368 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !367, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !369 = distinct !DISubprogram(name: "Make", linkageName: "_CMake.Main", scope: null, file: !367, line: 14, type: !41, spFlags: DISPFlagDefinition, unit: !368)
// CHECK:STDOUT: !370 = !DILocation(line: 15, column: 3, scope: !369)
// CHECK:STDOUT: !372 = !DILocation(line: 14, column: 1, scope: !369)
// CHECK:STDOUT: !373 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.454fecbcedcd86e1:core.Destroy.Core", scope: null, file: !367, line: 15, type: !22, spFlags: DISPFlagDefinition, unit: !368, retainedNodes: !376)
// CHECK:STDOUT: !374 = !DILocalVariable(arg: 1, scope: !373, type: !20)
// CHECK:STDOUT: !375 = !DILocation(line: 15, column: 3, scope: !373)
// CHECK:STDOUT: !376 = !{!374}
// CHECK:STDOUT: !377 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.95528a86e1127e2d:core.Destroy.Core", scope: null, file: !367, line: 15, type: !22, spFlags: DISPFlagDefinition, unit: !368, retainedNodes: !380)
// CHECK:STDOUT: !378 = !DILocalVariable(arg: 1, scope: !377, type: !20)
// CHECK:STDOUT: !379 = !DILocation(line: 15, column: 3, scope: !377)
// CHECK:STDOUT: !380 = !{!378}
// CHECK:STDOUT: !381 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.95528a86e1127e2d:core.Destroy.Core", scope: null, file: !367, line: 15, type: !22, spFlags: DISPFlagDefinition, unit: !368, retainedNodes: !384)
// CHECK:STDOUT: !382 = !DILocalVariable(arg: 1, scope: !381, type: !20)
// CHECK:STDOUT: !383 = !DILocation(line: 15, column: 3, scope: !381)
// CHECK:STDOUT: !384 = !{!382}
// CHECK:STDOUT: !385 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.5b23b4f14feb87e1:core.Destroy.Core", scope: null, file: !367, line: 15, type: !22, spFlags: DISPFlagDefinition, unit: !368, retainedNodes: !388)
// CHECK:STDOUT: !386 = !DILocalVariable(arg: 1, scope: !385, type: !20)
// CHECK:STDOUT: !387 = !DILocation(line: 15, column: 3, scope: !385)
// CHECK:STDOUT: !388 = !{!386}
// CHECK:STDOUT: !389 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.5b23b4f14feb87e1:core.Destroy.Core", scope: null, file: !367, line: 15, type: !22, spFlags: DISPFlagDefinition, unit: !368, retainedNodes: !392)
// CHECK:STDOUT: !390 = !DILocalVariable(arg: 1, scope: !389, type: !20)
// CHECK:STDOUT: !391 = !DILocation(line: 15, column: 3, scope: !389)
// CHECK:STDOUT: !392 = !{!390}
// CHECK:STDOUT: !393 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.2f190654c863951f:core.Destroy.Core", scope: null, file: !367, line: 15, type: !22, spFlags: DISPFlagDefinition, unit: !368, retainedNodes: !396)
// CHECK:STDOUT: !394 = !DILocalVariable(arg: 1, scope: !393, type: !20)
// CHECK:STDOUT: !395 = !DILocation(line: 15, column: 3, scope: !393)
// CHECK:STDOUT: !396 = !{!394}
// CHECK:STDOUT: !397 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.2f190654c863951f:core.Destroy.Core", scope: null, file: !367, line: 15, type: !22, spFlags: DISPFlagDefinition, unit: !368, retainedNodes: !400)
// CHECK:STDOUT: !398 = !DILocalVariable(arg: 1, scope: !397, type: !20)
// CHECK:STDOUT: !399 = !DILocation(line: 15, column: 3, scope: !397)
// CHECK:STDOUT: !400 = !{!398}
// CHECK:STDOUT: !401 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.b25525d21a9bd472:core.Destroy.Core", scope: null, file: !367, line: 15, type: !22, spFlags: DISPFlagDefinition, unit: !368, retainedNodes: !404)
// CHECK:STDOUT: !402 = !DILocalVariable(arg: 1, scope: !401, type: !20)
// CHECK:STDOUT: !403 = !DILocation(line: 15, column: 3, scope: !401)
// CHECK:STDOUT: !404 = !{!402}
// CHECK:STDOUT: !405 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.b25525d21a9bd472:core.Destroy.Core", scope: null, file: !367, line: 15, type: !22, spFlags: DISPFlagDefinition, unit: !368, retainedNodes: !408)
// CHECK:STDOUT: !406 = !DILocalVariable(arg: 1, scope: !405, type: !20)
// CHECK:STDOUT: !407 = !DILocation(line: 15, column: 3, scope: !405)
// CHECK:STDOUT: !408 = !{!406}
// CHECK:STDOUT: !409 = distinct !DISubprogram(name: "F", linkageName: "_CF.Base.Main.6d6a4a2512463e59", scope: null, file: !367, line: 5, type: !22, spFlags: DISPFlagDefinition, unit: !368, retainedNodes: !413)
// CHECK:STDOUT: !410 = !DILocalVariable(arg: 1, scope: !409, type: !20)
// CHECK:STDOUT: !412 = !DILocation(line: 5, column: 3, scope: !409)
// CHECK:STDOUT: !413 = !{!410}
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
// CHECK:STDOUT: define void @_CF.Derived.Main(ptr sret({}) %return, ptr %self, ptr %n) #0 !dbg !420 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %return, ptr align 1 @From.val.loc18_14, i64 0, i1 false), !dbg !427
// CHECK:STDOUT:   ret void, !dbg !427
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_CF:thunk:Base.Main:Derived.Main"(ptr sret({}) %return, ptr %self, ptr %n) #1 !dbg !429 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc17_52.2.temp = alloca {}, align 1, !dbg !433
// CHECK:STDOUT:   %.5.temp = alloca {}, align 1, !dbg !432
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc17_52.2.temp), !dbg !433
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.5.temp), !dbg !432
// CHECK:STDOUT:   call void @"_CConvert.From.Main:ImplicitAs.a3bb440f6426a7d6.Core"(ptr %.5.temp, ptr %n), !dbg !432
// CHECK:STDOUT:   call void @_CF.Derived.Main(ptr %.loc17_52.2.temp, ptr %self, ptr %.5.temp), !dbg !433
// CHECK:STDOUT:   call void @"_CConvert.From.Main:ImplicitAs.a3bb440f6426a7d6.Core"(ptr %return, ptr %.loc17_52.2.temp), !dbg !433
// CHECK:STDOUT:   call void @"_CSelfDestruct.c2a743abd057cb33:core.Destroy.Core"(ptr %.loc17_52.2.temp), !dbg !433
// CHECK:STDOUT:   call void @"_CSelfDestruct.7ef2e8c2f12d163a:core.Destroy.Core"(ptr %.5.temp), !dbg !432
// CHECK:STDOUT:   ret void, !dbg !433
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.cb8adced19392b99:core.Destroy.Core"(ptr %self) #0 !dbg !435 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !437
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.c2a743abd057cb33:core.Destroy.Core"(ptr %self) #0 !dbg !439 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !441
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.7ef2e8c2f12d163a:core.Destroy.Core"(ptr %self) #0 !dbg !443 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !445
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CUse.Main() #0 !dbg !447 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %d.var = alloca { { ptr } }, align 8, !dbg !448
// CHECK:STDOUT:   %.loc25_10.1.temp = alloca {}, align 1, !dbg !456
// CHECK:STDOUT:   %.loc25_9.2.temp = alloca {}, align 1, !dbg !457
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %d.var), !dbg !448
// CHECK:STDOUT:   %.loc23_31.2.base = getelementptr inbounds nuw { { ptr } }, ptr %d.var, i32 0, i32 0, !dbg !449
// CHECK:STDOUT:   %.loc23_30.2.vptr = getelementptr inbounds nuw { ptr }, ptr %.loc23_31.2.base, i32 0, i32 0, !dbg !450
// CHECK:STDOUT:   %.loc23_31.6.base = getelementptr inbounds nuw { { ptr } }, ptr %d.var, i32 0, i32 0, !dbg !449
// CHECK:STDOUT:   %.loc23_31.7.vptr = getelementptr inbounds nuw { ptr }, ptr %.loc23_31.6.base, i32 0, i32 0, !dbg !449
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 %d.var, ptr align 8 @Derived.val.loc23_31.5, i64 8, i1 false), !dbg !449
// CHECK:STDOUT:   store ptr @"_CDerived.Main.$vtable", ptr %.loc23_31.7.vptr, align 8, !dbg !449
// CHECK:STDOUT:   %.loc24_18.2.base = getelementptr inbounds nuw { { ptr } }, ptr %d.var, i32 0, i32 0, !dbg !453
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc25_10.1.temp), !dbg !456
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc25_9.2.temp), !dbg !457
// CHECK:STDOUT:   %Base.F.call.vtable = load ptr, ptr %.loc25_10.1.temp, align 8, !dbg !456
// CHECK:STDOUT:   %Base.F.call = call ptr @llvm.load.relative.i32(ptr %Base.F.call.vtable, i32 0), !dbg !456
// CHECK:STDOUT:   call void %Base.F.call(ptr %.loc25_10.1.temp, ptr %.loc24_18.2.base, ptr @From.val.loc18_14), !dbg !456
// CHECK:STDOUT:   call void @"_CSelfDestruct.7ef2e8c2f12d163a:core.Destroy.Core"(ptr %.loc25_10.1.temp), !dbg !456
// CHECK:STDOUT:   call void @"_CSelfDestruct.c2a743abd057cb33:core.Destroy.Core"(ptr @From.val), !dbg !457
// CHECK:STDOUT:   call void @"_CSelfDestruct.b25525d21a9bd472:core.Destroy.Core"(ptr %d.var), !dbg !448
// CHECK:STDOUT:   ret void, !dbg !458
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.454fecbcedcd86e1:core.Destroy.Core"(ptr %self) #0 !dbg !459 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !461
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.95528a86e1127e2d:core.Destroy.Core"(ptr %self) #0 !dbg !463 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !465
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.95528a86e1127e2d:core.Destroy.Core"(ptr %self) #0 !dbg !467 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.95528a86e1127e2d:core.Destroy.Core"(ptr %self), !dbg !469
// CHECK:STDOUT:   ret void, !dbg !469
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.294df737cd074d60:core.Destroy.Core"(ptr %self) #0 !dbg !471 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !473
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.294df737cd074d60:core.Destroy.Core"(ptr %self) #0 !dbg !475 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.294df737cd074d60:core.Destroy.Core"(ptr %self), !dbg !477
// CHECK:STDOUT:   ret void, !dbg !477
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.caed8a96bf328d09:core.Destroy.Core"(ptr %self) #0 !dbg !479 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !481
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.caed8a96bf328d09:core.Destroy.Core"(ptr %self) #0 !dbg !483 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.caed8a96bf328d09:core.Destroy.Core"(ptr %self), !dbg !485
// CHECK:STDOUT:   ret void, !dbg !485
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.b25525d21a9bd472:core.Destroy.Core"(ptr %self) #0 !dbg !487 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !489
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.b25525d21a9bd472:core.Destroy.Core"(ptr %self) #0 !dbg !491 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.b25525d21a9bd472:core.Destroy.Core"(ptr %self), !dbg !493
// CHECK:STDOUT:   ret void, !dbg !493
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!415}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !21 = !{null, !20}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !40 = !{null}
// CHECK:STDOUT: !41 = !DISubroutineType(types: !40)
// CHECK:STDOUT: !414 = !DIFile(filename: "thunk.carbon", directory: "")
// CHECK:STDOUT: !415 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !414, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !418 = !{!20, !20, !20}
// CHECK:STDOUT: !419 = !DISubroutineType(types: !418)
// CHECK:STDOUT: !420 = distinct !DISubprogram(name: "F", linkageName: "_CF.Derived.Main", scope: null, file: !414, line: 17, type: !419, spFlags: DISPFlagDefinition, unit: !415, retainedNodes: !428)
// CHECK:STDOUT: !421 = !DILocalVariable(arg: 1, scope: !420, type: !20)
// CHECK:STDOUT: !422 = !DILocalVariable(arg: 2, scope: !420, type: !20)
// CHECK:STDOUT: !427 = !DILocation(line: 18, column: 5, scope: !420)
// CHECK:STDOUT: !428 = !{!421, !422}
// CHECK:STDOUT: !429 = distinct !DISubprogram(name: "F", linkageName: "_CF:thunk:Base.Main:Derived.Main", scope: null, file: !414, line: 17, type: !419, spFlags: DISPFlagDefinition, unit: !415, retainedNodes: !434)
// CHECK:STDOUT: !430 = !DILocalVariable(arg: 1, scope: !429, type: !20)
// CHECK:STDOUT: !431 = !DILocalVariable(arg: 2, scope: !429, type: !20)
// CHECK:STDOUT: !432 = !DILocation(line: 0, scope: !429)
// CHECK:STDOUT: !433 = !DILocation(line: 17, column: 3, scope: !429)
// CHECK:STDOUT: !434 = !{!430, !431}
// CHECK:STDOUT: !435 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.cb8adced19392b99:core.Destroy.Core", scope: null, file: !414, line: 17, type: !22, spFlags: DISPFlagDefinition, unit: !415, retainedNodes: !438)
// CHECK:STDOUT: !436 = !DILocalVariable(arg: 1, scope: !435, type: !20)
// CHECK:STDOUT: !437 = !DILocation(line: 17, column: 3, scope: !435)
// CHECK:STDOUT: !438 = !{!436}
// CHECK:STDOUT: !439 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.c2a743abd057cb33:core.Destroy.Core", scope: null, file: !414, line: 17, type: !22, spFlags: DISPFlagDefinition, unit: !415, retainedNodes: !442)
// CHECK:STDOUT: !440 = !DILocalVariable(arg: 1, scope: !439, type: !20)
// CHECK:STDOUT: !441 = !DILocation(line: 17, column: 3, scope: !439)
// CHECK:STDOUT: !442 = !{!440}
// CHECK:STDOUT: !443 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.7ef2e8c2f12d163a:core.Destroy.Core", scope: null, file: !414, type: !22, spFlags: DISPFlagDefinition, unit: !415, retainedNodes: !446)
// CHECK:STDOUT: !444 = !DILocalVariable(arg: 1, scope: !443, type: !20)
// CHECK:STDOUT: !445 = !DILocation(line: 0, scope: !443)
// CHECK:STDOUT: !446 = !{!444}
// CHECK:STDOUT: !447 = distinct !DISubprogram(name: "Use", linkageName: "_CUse.Main", scope: null, file: !414, line: 22, type: !41, spFlags: DISPFlagDefinition, unit: !415)
// CHECK:STDOUT: !448 = !DILocation(line: 23, column: 3, scope: !447)
// CHECK:STDOUT: !449 = !DILocation(line: 23, column: 20, scope: !447)
// CHECK:STDOUT: !450 = !DILocation(line: 23, column: 29, scope: !447)
// CHECK:STDOUT: !453 = !DILocation(line: 24, column: 18, scope: !447)
// CHECK:STDOUT: !456 = !DILocation(line: 25, column: 3, scope: !447)
// CHECK:STDOUT: !457 = !DILocation(line: 25, column: 8, scope: !447)
// CHECK:STDOUT: !458 = !DILocation(line: 22, column: 1, scope: !447)
// CHECK:STDOUT: !459 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.454fecbcedcd86e1:core.Destroy.Core", scope: null, file: !414, line: 23, type: !22, spFlags: DISPFlagDefinition, unit: !415, retainedNodes: !462)
// CHECK:STDOUT: !460 = !DILocalVariable(arg: 1, scope: !459, type: !20)
// CHECK:STDOUT: !461 = !DILocation(line: 23, column: 3, scope: !459)
// CHECK:STDOUT: !462 = !{!460}
// CHECK:STDOUT: !463 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.95528a86e1127e2d:core.Destroy.Core", scope: null, file: !414, line: 23, type: !22, spFlags: DISPFlagDefinition, unit: !415, retainedNodes: !466)
// CHECK:STDOUT: !464 = !DILocalVariable(arg: 1, scope: !463, type: !20)
// CHECK:STDOUT: !465 = !DILocation(line: 23, column: 3, scope: !463)
// CHECK:STDOUT: !466 = !{!464}
// CHECK:STDOUT: !467 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.95528a86e1127e2d:core.Destroy.Core", scope: null, file: !414, line: 23, type: !22, spFlags: DISPFlagDefinition, unit: !415, retainedNodes: !470)
// CHECK:STDOUT: !468 = !DILocalVariable(arg: 1, scope: !467, type: !20)
// CHECK:STDOUT: !469 = !DILocation(line: 23, column: 3, scope: !467)
// CHECK:STDOUT: !470 = !{!468}
// CHECK:STDOUT: !471 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.294df737cd074d60:core.Destroy.Core", scope: null, file: !414, line: 23, type: !22, spFlags: DISPFlagDefinition, unit: !415, retainedNodes: !474)
// CHECK:STDOUT: !472 = !DILocalVariable(arg: 1, scope: !471, type: !20)
// CHECK:STDOUT: !473 = !DILocation(line: 23, column: 3, scope: !471)
// CHECK:STDOUT: !474 = !{!472}
// CHECK:STDOUT: !475 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.294df737cd074d60:core.Destroy.Core", scope: null, file: !414, line: 23, type: !22, spFlags: DISPFlagDefinition, unit: !415, retainedNodes: !478)
// CHECK:STDOUT: !476 = !DILocalVariable(arg: 1, scope: !475, type: !20)
// CHECK:STDOUT: !477 = !DILocation(line: 23, column: 3, scope: !475)
// CHECK:STDOUT: !478 = !{!476}
// CHECK:STDOUT: !479 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.caed8a96bf328d09:core.Destroy.Core", scope: null, file: !414, line: 23, type: !22, spFlags: DISPFlagDefinition, unit: !415, retainedNodes: !482)
// CHECK:STDOUT: !480 = !DILocalVariable(arg: 1, scope: !479, type: !20)
// CHECK:STDOUT: !481 = !DILocation(line: 23, column: 3, scope: !479)
// CHECK:STDOUT: !482 = !{!480}
// CHECK:STDOUT: !483 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.caed8a96bf328d09:core.Destroy.Core", scope: null, file: !414, line: 23, type: !22, spFlags: DISPFlagDefinition, unit: !415, retainedNodes: !486)
// CHECK:STDOUT: !484 = !DILocalVariable(arg: 1, scope: !483, type: !20)
// CHECK:STDOUT: !485 = !DILocation(line: 23, column: 3, scope: !483)
// CHECK:STDOUT: !486 = !{!484}
// CHECK:STDOUT: !487 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.b25525d21a9bd472:core.Destroy.Core", scope: null, file: !414, line: 23, type: !22, spFlags: DISPFlagDefinition, unit: !415, retainedNodes: !490)
// CHECK:STDOUT: !488 = !DILocalVariable(arg: 1, scope: !487, type: !20)
// CHECK:STDOUT: !489 = !DILocation(line: 23, column: 3, scope: !487)
// CHECK:STDOUT: !490 = !{!488}
// CHECK:STDOUT: !491 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.b25525d21a9bd472:core.Destroy.Core", scope: null, file: !414, line: 23, type: !22, spFlags: DISPFlagDefinition, unit: !415, retainedNodes: !494)
// CHECK:STDOUT: !492 = !DILocalVariable(arg: 1, scope: !491, type: !20)
// CHECK:STDOUT: !493 = !DILocation(line: 23, column: 3, scope: !491)
// CHECK:STDOUT: !494 = !{!492}
// CHECK:STDOUT:
