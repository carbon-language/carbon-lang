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
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.bf085e1f873e33df:core.Destroy.Core"(ptr %self) #0 !dbg !63 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !65
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.bf085e1f873e33df:core.Destroy.Core"(ptr %self) #0 !dbg !67 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.bf085e1f873e33df:core.Destroy.Core"(ptr %self), !dbg !69
// CHECK:STDOUT:   ret void, !dbg !69
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.f7b08e0f33b30342:core.Destroy.Core"(ptr %self) #0 !dbg !71 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc12_3.1.vptr = getelementptr inbounds nuw { ptr, {} }, ptr %self, i32 0, i32 0, !dbg !73
// CHECK:STDOUT:   call void @"_CSelfDestruct.454fecbcedcd86e1:core.Destroy.Core"(ptr %.loc12_3.1.vptr), !dbg !73
// CHECK:STDOUT:   %.loc12_3.2.base = getelementptr inbounds nuw { ptr, {} }, ptr %self, i32 0, i32 1, !dbg !73
// CHECK:STDOUT:   call void @"_CSelfDestruct.bf085e1f873e33df:core.Destroy.Core"(ptr %.loc12_3.2.base), !dbg !73
// CHECK:STDOUT:   ret void, !dbg !73
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.f7b08e0f33b30342:core.Destroy.Core"(ptr %self) #0 !dbg !75 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.f7b08e0f33b30342:core.Destroy.Core"(ptr %self), !dbg !77
// CHECK:STDOUT:   ret void, !dbg !77
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.4487e305711510e9:core.Destroy.Core"(ptr %self) #0 !dbg !79 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !81
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.4487e305711510e9:core.Destroy.Core"(ptr %self) #0 !dbg !83 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.4487e305711510e9:core.Destroy.Core"(ptr %self), !dbg !85
// CHECK:STDOUT:   ret void, !dbg !85
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.3c3534e993db7939:core.Destroy.Core"(ptr %self) #0 !dbg !87 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc12.base = getelementptr inbounds nuw { { ptr, {} } }, ptr %self, i32 0, i32 0, !dbg !89
// CHECK:STDOUT:   call void @"_CSelfDestruct.4487e305711510e9:core.Destroy.Core"(ptr %.loc12.base), !dbg !89
// CHECK:STDOUT:   ret void, !dbg !89
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.3c3534e993db7939:core.Destroy.Core"(ptr %self) #0 !dbg !91 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.3c3534e993db7939:core.Destroy.Core"(ptr %self), !dbg !93
// CHECK:STDOUT:   ret void, !dbg !93
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.43ed5d0e08fc7556:core.Destroy.Core"(ptr %self) #0 !dbg !95 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !97
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.43ed5d0e08fc7556:core.Destroy.Core"(ptr %self) #0 !dbg !99 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.43ed5d0e08fc7556:core.Destroy.Core"(ptr %self), !dbg !101
// CHECK:STDOUT:   ret void, !dbg !101
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CUse.Create(ptr %v) #0 !dbg !103 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Intermediate.Fn.call.vtable = load ptr, ptr %v, align 8, !dbg !106
// CHECK:STDOUT:   %Intermediate.Fn.call = call ptr @llvm.load.relative.i32(ptr %Intermediate.Fn.call.vtable, i32 0), !dbg !106
// CHECK:STDOUT:   call void %Intermediate.Fn.call(ptr %v), !dbg !106
// CHECK:STDOUT:   ret void, !dbg !107
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
// CHECK:STDOUT: !63 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.bf085e1f873e33df:core.Destroy.Core", scope: null, file: !38, line: 12, type: !22, spFlags: DISPFlagDefinition, unit: !39, retainedNodes: !66)
// CHECK:STDOUT: !64 = !DILocalVariable(arg: 1, scope: !63, type: !20)
// CHECK:STDOUT: !65 = !DILocation(line: 12, column: 3, scope: !63)
// CHECK:STDOUT: !66 = !{!64}
// CHECK:STDOUT: !67 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.bf085e1f873e33df:core.Destroy.Core", scope: null, file: !38, line: 12, type: !22, spFlags: DISPFlagDefinition, unit: !39, retainedNodes: !70)
// CHECK:STDOUT: !68 = !DILocalVariable(arg: 1, scope: !67, type: !20)
// CHECK:STDOUT: !69 = !DILocation(line: 12, column: 3, scope: !67)
// CHECK:STDOUT: !70 = !{!68}
// CHECK:STDOUT: !71 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.f7b08e0f33b30342:core.Destroy.Core", scope: null, file: !38, line: 12, type: !22, spFlags: DISPFlagDefinition, unit: !39, retainedNodes: !74)
// CHECK:STDOUT: !72 = !DILocalVariable(arg: 1, scope: !71, type: !20)
// CHECK:STDOUT: !73 = !DILocation(line: 12, column: 3, scope: !71)
// CHECK:STDOUT: !74 = !{!72}
// CHECK:STDOUT: !75 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.f7b08e0f33b30342:core.Destroy.Core", scope: null, file: !38, line: 12, type: !22, spFlags: DISPFlagDefinition, unit: !39, retainedNodes: !78)
// CHECK:STDOUT: !76 = !DILocalVariable(arg: 1, scope: !75, type: !20)
// CHECK:STDOUT: !77 = !DILocation(line: 12, column: 3, scope: !75)
// CHECK:STDOUT: !78 = !{!76}
// CHECK:STDOUT: !79 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.4487e305711510e9:core.Destroy.Core", scope: null, file: !38, line: 12, type: !22, spFlags: DISPFlagDefinition, unit: !39, retainedNodes: !82)
// CHECK:STDOUT: !80 = !DILocalVariable(arg: 1, scope: !79, type: !20)
// CHECK:STDOUT: !81 = !DILocation(line: 12, column: 3, scope: !79)
// CHECK:STDOUT: !82 = !{!80}
// CHECK:STDOUT: !83 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.4487e305711510e9:core.Destroy.Core", scope: null, file: !38, line: 12, type: !22, spFlags: DISPFlagDefinition, unit: !39, retainedNodes: !86)
// CHECK:STDOUT: !84 = !DILocalVariable(arg: 1, scope: !83, type: !20)
// CHECK:STDOUT: !85 = !DILocation(line: 12, column: 3, scope: !83)
// CHECK:STDOUT: !86 = !{!84}
// CHECK:STDOUT: !87 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.3c3534e993db7939:core.Destroy.Core", scope: null, file: !38, line: 12, type: !22, spFlags: DISPFlagDefinition, unit: !39, retainedNodes: !90)
// CHECK:STDOUT: !88 = !DILocalVariable(arg: 1, scope: !87, type: !20)
// CHECK:STDOUT: !89 = !DILocation(line: 12, column: 3, scope: !87)
// CHECK:STDOUT: !90 = !{!88}
// CHECK:STDOUT: !91 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.3c3534e993db7939:core.Destroy.Core", scope: null, file: !38, line: 12, type: !22, spFlags: DISPFlagDefinition, unit: !39, retainedNodes: !94)
// CHECK:STDOUT: !92 = !DILocalVariable(arg: 1, scope: !91, type: !20)
// CHECK:STDOUT: !93 = !DILocation(line: 12, column: 3, scope: !91)
// CHECK:STDOUT: !94 = !{!92}
// CHECK:STDOUT: !95 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.43ed5d0e08fc7556:core.Destroy.Core", scope: null, file: !38, line: 12, type: !22, spFlags: DISPFlagDefinition, unit: !39, retainedNodes: !98)
// CHECK:STDOUT: !96 = !DILocalVariable(arg: 1, scope: !95, type: !20)
// CHECK:STDOUT: !97 = !DILocation(line: 12, column: 3, scope: !95)
// CHECK:STDOUT: !98 = !{!96}
// CHECK:STDOUT: !99 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.43ed5d0e08fc7556:core.Destroy.Core", scope: null, file: !38, line: 12, type: !22, spFlags: DISPFlagDefinition, unit: !39, retainedNodes: !102)
// CHECK:STDOUT: !100 = !DILocalVariable(arg: 1, scope: !99, type: !20)
// CHECK:STDOUT: !101 = !DILocation(line: 12, column: 3, scope: !99)
// CHECK:STDOUT: !102 = !{!100}
// CHECK:STDOUT: !103 = distinct !DISubprogram(name: "Use", linkageName: "_CUse.Create", scope: null, file: !38, line: 15, type: !22, spFlags: DISPFlagDefinition, unit: !39, retainedNodes: !108)
// CHECK:STDOUT: !104 = !DILocalVariable(arg: 1, scope: !103, type: !20)
// CHECK:STDOUT: !106 = !DILocation(line: 16, column: 3, scope: !103)
// CHECK:STDOUT: !107 = !DILocation(line: 15, column: 1, scope: !103)
// CHECK:STDOUT: !108 = !{!104}
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
// CHECK:STDOUT: define void @_CMakePartialBase.Main(ptr sret({}) %return) #0 !dbg !113 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %return, ptr align 1 @empty_struct.ac8.loc7_12, i64 0, i1 false), !dbg !116
// CHECK:STDOUT:   ret void, !dbg !116
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMakePartialIntermediate.Main(ptr sret({ ptr, {} }) %return) #0 !dbg !117 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc11_36.1.base = getelementptr inbounds nuw { ptr, {} }, ptr %return, i32 0, i32 1, !dbg !120
// CHECK:STDOUT:   call void @_CMakePartialBase.Main(ptr %.loc11_36.1.base), !dbg !121
// CHECK:STDOUT:   %.loc11_36.3.vptr = getelementptr inbounds nuw { ptr, {} }, ptr %return, i32 0, i32 0, !dbg !120
// CHECK:STDOUT:   store ptr poison, ptr %.loc11_36.3.vptr, align 8, !dbg !120
// CHECK:STDOUT:   ret void, !dbg !122
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMakeBase.Main(ptr sret({}) %return) #0 !dbg !123 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CMakePartialBase.Main(ptr %return), !dbg !126
// CHECK:STDOUT:   ret void, !dbg !127
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMakeIntermediate.Main(ptr sret({ ptr, {} }) %return) #0 !dbg !128 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CMakePartialIntermediate.Main(ptr %return), !dbg !131
// CHECK:STDOUT:   %.loc19_34.1.vptr = getelementptr inbounds nuw { ptr, {} }, ptr %return, i32 0, i32 0, !dbg !131
// CHECK:STDOUT:   store ptr @"_CIntermediate.Classes.$vtable", ptr %.loc19_34.1.vptr, align 8, !dbg !131
// CHECK:STDOUT:   ret void, !dbg !132
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CFn.Intermediate.Classes(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMakeDerivedFromPartial.Main(ptr sret({ { ptr, {} } }) %return) #0 !dbg !133 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc23_44.1.base = getelementptr inbounds nuw { { ptr, {} } }, ptr %return, i32 0, i32 0, !dbg !136
// CHECK:STDOUT:   call void @_CMakePartialIntermediate.Main(ptr %.loc23_44.1.base), !dbg !137
// CHECK:STDOUT:   %.loc23_44.5.base = getelementptr inbounds nuw { { ptr, {} } }, ptr %return, i32 0, i32 0, !dbg !136
// CHECK:STDOUT:   %.loc23_44.6.vptr = getelementptr inbounds nuw { ptr, {} }, ptr %.loc23_44.5.base, i32 0, i32 0, !dbg !136
// CHECK:STDOUT:   store ptr @"_CDerived.Classes.$vtable", ptr %.loc23_44.6.vptr, align 8, !dbg !136
// CHECK:STDOUT:   ret void, !dbg !138
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_CFn:thunk:Intermediate.Classes:Derived.Classes"(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMakeDerivedFromNonPartial.Main(ptr sret({ { ptr, {} } }) %return) #0 !dbg !139 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc27_37.1.base = getelementptr inbounds nuw { { ptr, {} } }, ptr %return, i32 0, i32 0, !dbg !142
// CHECK:STDOUT:   call void @_CMakeIntermediate.Main(ptr %.loc27_37.1.base), !dbg !143
// CHECK:STDOUT:   %.loc27_37.7.base = getelementptr inbounds nuw { { ptr, {} } }, ptr %return, i32 0, i32 0, !dbg !142
// CHECK:STDOUT:   %.loc27_37.8.vptr = getelementptr inbounds nuw { ptr, {} }, ptr %.loc27_37.7.base, i32 0, i32 0, !dbg !142
// CHECK:STDOUT:   store ptr @"_CDerived.Classes.$vtable", ptr %.loc27_37.8.vptr, align 8, !dbg !142
// CHECK:STDOUT:   ret void, !dbg !144
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!110}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !109 = !DIFile(filename: "create_named_ctor.carbon", directory: "")
// CHECK:STDOUT: !110 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !109, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !111 = !{!20}
// CHECK:STDOUT: !112 = !DISubroutineType(types: !111)
// CHECK:STDOUT: !113 = distinct !DISubprogram(name: "MakePartialBase", linkageName: "_CMakePartialBase.Main", scope: null, file: !109, line: 6, type: !112, spFlags: DISPFlagDefinition, unit: !110)
// CHECK:STDOUT: !116 = !DILocation(line: 7, column: 3, scope: !113)
// CHECK:STDOUT: !117 = distinct !DISubprogram(name: "MakePartialIntermediate", linkageName: "_CMakePartialIntermediate.Main", scope: null, file: !109, line: 10, type: !112, spFlags: DISPFlagDefinition, unit: !110)
// CHECK:STDOUT: !120 = !DILocation(line: 11, column: 10, scope: !117)
// CHECK:STDOUT: !121 = !DILocation(line: 11, column: 19, scope: !117)
// CHECK:STDOUT: !122 = !DILocation(line: 11, column: 3, scope: !117)
// CHECK:STDOUT: !123 = distinct !DISubprogram(name: "MakeBase", linkageName: "_CMakeBase.Main", scope: null, file: !109, line: 14, type: !112, spFlags: DISPFlagDefinition, unit: !110)
// CHECK:STDOUT: !126 = !DILocation(line: 15, column: 10, scope: !123)
// CHECK:STDOUT: !127 = !DILocation(line: 15, column: 3, scope: !123)
// CHECK:STDOUT: !128 = distinct !DISubprogram(name: "MakeIntermediate", linkageName: "_CMakeIntermediate.Main", scope: null, file: !109, line: 18, type: !112, spFlags: DISPFlagDefinition, unit: !110)
// CHECK:STDOUT: !131 = !DILocation(line: 19, column: 10, scope: !128)
// CHECK:STDOUT: !132 = !DILocation(line: 19, column: 3, scope: !128)
// CHECK:STDOUT: !133 = distinct !DISubprogram(name: "MakeDerivedFromPartial", linkageName: "_CMakeDerivedFromPartial.Main", scope: null, file: !109, line: 22, type: !112, spFlags: DISPFlagDefinition, unit: !110)
// CHECK:STDOUT: !136 = !DILocation(line: 23, column: 10, scope: !133)
// CHECK:STDOUT: !137 = !DILocation(line: 23, column: 19, scope: !133)
// CHECK:STDOUT: !138 = !DILocation(line: 23, column: 3, scope: !133)
// CHECK:STDOUT: !139 = distinct !DISubprogram(name: "MakeDerivedFromNonPartial", linkageName: "_CMakeDerivedFromNonPartial.Main", scope: null, file: !109, line: 26, type: !112, spFlags: DISPFlagDefinition, unit: !110)
// CHECK:STDOUT: !142 = !DILocation(line: 27, column: 10, scope: !139)
// CHECK:STDOUT: !143 = !DILocation(line: 27, column: 19, scope: !139)
// CHECK:STDOUT: !144 = !DILocation(line: 27, column: 3, scope: !139)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'member_init.carbon'
// CHECK:STDOUT: source_filename = "member_init.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @"_CBase.MemberInit.$vtable" = unnamed_addr constant [1 x i32] [i32 trunc (i64 sub (i64 ptrtoint (ptr @_CFn.Base.MemberInit to i64), i64 ptrtoint (ptr @"_CBase.MemberInit.$vtable" to i64)) to i32)]
// CHECK:STDOUT: @Base.val.loc13_3 = internal constant <{ ptr, i32 }> <{ ptr @"_CBase.MemberInit.$vtable", i32 3 }>
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CFn.Base.MemberInit(ptr %self) #0 !dbg !150 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !153
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CFn.MemberInit() #0 !dbg !155 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %i.var = alloca i32, align 4, !dbg !156
// CHECK:STDOUT:   %v.var = alloca <{ ptr, i32 }>, align 8, !dbg !158
// CHECK:STDOUT:   %_.var = alloca <{ ptr, i32 }>, align 8, !dbg !163
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %i.var), !dbg !156
// CHECK:STDOUT:   store i32 3, ptr %i.var, align 4, !dbg !156
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %v.var), !dbg !158
// CHECK:STDOUT:   %.loc11_24.2.vptr = getelementptr inbounds nuw <{ ptr, i32 }>, ptr %v.var, i32 0, i32 0, !dbg !160
// CHECK:STDOUT:   %.loc11_23 = load i32, ptr %i.var, align 4, !dbg !159
// CHECK:STDOUT:   %.loc11_24.4.m = getelementptr inbounds nuw <{ ptr, i32 }>, ptr %v.var, i32 0, i32 1, !dbg !160
// CHECK:STDOUT:   store i32 %.loc11_23, ptr %.loc11_24.4.m, align 4, !dbg !160
// CHECK:STDOUT:   store ptr @"_CBase.MemberInit.$vtable", ptr %.loc11_24.2.vptr, align 8, !dbg !160
// CHECK:STDOUT:   %.loc12_4.m = getelementptr inbounds nuw <{ ptr, i32 }>, ptr %v.var, i32 0, i32 1, !dbg !162
// CHECK:STDOUT:   store i32 5, ptr %.loc12_4.m, align 4, !dbg !162
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !163
// CHECK:STDOUT:   %.loc13_24.2.vptr = getelementptr inbounds nuw <{ ptr, i32 }>, ptr %_.var, i32 0, i32 0, !dbg !164
// CHECK:STDOUT:   %.loc13_24.5.m = getelementptr inbounds nuw <{ ptr, i32 }>, ptr %_.var, i32 0, i32 1, !dbg !164
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 %_.var, ptr align 8 @Base.val.loc13_3, i64 12, i1 false), !dbg !163
// CHECK:STDOUT:   call void @"_CSelfDestruct.c6dbe0017b4c8d16:core.Destroy.Core"(ptr %_.var), !dbg !163
// CHECK:STDOUT:   call void @"_CSelfDestruct.c6dbe0017b4c8d16:core.Destroy.Core"(ptr %v.var), !dbg !158
// CHECK:STDOUT:   call void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %i.var), !dbg !156
// CHECK:STDOUT:   ret void, !dbg !166
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.454fecbcedcd86e1:core.Destroy.Core"(ptr %self) #0 !dbg !167 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !169
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.6dae29ece9a708c4:core.Destroy.Core"(ptr %self) #0 !dbg !171 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !173
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !175 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !177
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !179 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self), !dbg !181
// CHECK:STDOUT:   ret void, !dbg !181
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.fd9d4a98e93fc879:core.Destroy.Core"(ptr %self) #0 !dbg !183 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc13_3.1.vptr = getelementptr inbounds nuw <{ ptr, i32 }>, ptr %self, i32 0, i32 0, !dbg !185
// CHECK:STDOUT:   call void @"_CSelfDestruct.454fecbcedcd86e1:core.Destroy.Core"(ptr %.loc13_3.1.vptr), !dbg !185
// CHECK:STDOUT:   %.loc13_3.2.m = getelementptr inbounds nuw <{ ptr, i32 }>, ptr %self, i32 0, i32 1, !dbg !185
// CHECK:STDOUT:   call void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %.loc13_3.2.m), !dbg !185
// CHECK:STDOUT:   ret void, !dbg !185
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.fd9d4a98e93fc879:core.Destroy.Core"(ptr %self) #0 !dbg !187 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.fd9d4a98e93fc879:core.Destroy.Core"(ptr %self), !dbg !189
// CHECK:STDOUT:   ret void, !dbg !189
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.c6dbe0017b4c8d16:core.Destroy.Core"(ptr %self) #0 !dbg !191 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc13.m = getelementptr inbounds nuw <{ ptr, i32 }>, ptr %self, i32 0, i32 1, !dbg !193
// CHECK:STDOUT:   call void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %.loc13.m), !dbg !193
// CHECK:STDOUT:   ret void, !dbg !193
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.c6dbe0017b4c8d16:core.Destroy.Core"(ptr %self) #0 !dbg !195 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.c6dbe0017b4c8d16:core.Destroy.Core"(ptr %self), !dbg !197
// CHECK:STDOUT:   ret void, !dbg !197
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!146}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !21 = !{null, !20}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !40 = !{null}
// CHECK:STDOUT: !41 = !DISubroutineType(types: !40)
// CHECK:STDOUT: !145 = !DIFile(filename: "member_init.carbon", directory: "")
// CHECK:STDOUT: !146 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !145, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !147 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !148 = !{null, !147}
// CHECK:STDOUT: !149 = !DISubroutineType(types: !148)
// CHECK:STDOUT: !150 = distinct !DISubprogram(name: "Fn", linkageName: "_CFn.Base.MemberInit", scope: null, file: !145, line: 6, type: !22, spFlags: DISPFlagDefinition, unit: !146, retainedNodes: !154)
// CHECK:STDOUT: !151 = !DILocalVariable(arg: 1, scope: !150, type: !20)
// CHECK:STDOUT: !153 = !DILocation(line: 6, column: 3, scope: !150)
// CHECK:STDOUT: !154 = !{!151}
// CHECK:STDOUT: !155 = distinct !DISubprogram(name: "Fn", linkageName: "_CFn.MemberInit", scope: null, file: !145, line: 9, type: !41, spFlags: DISPFlagDefinition, unit: !146)
// CHECK:STDOUT: !156 = !DILocation(line: 10, column: 3, scope: !155)
// CHECK:STDOUT: !158 = !DILocation(line: 11, column: 3, scope: !155)
// CHECK:STDOUT: !159 = !DILocation(line: 11, column: 23, scope: !155)
// CHECK:STDOUT: !160 = !DILocation(line: 11, column: 17, scope: !155)
// CHECK:STDOUT: !162 = !DILocation(line: 12, column: 3, scope: !155)
// CHECK:STDOUT: !163 = !DILocation(line: 13, column: 3, scope: !155)
// CHECK:STDOUT: !164 = !DILocation(line: 13, column: 17, scope: !155)
// CHECK:STDOUT: !166 = !DILocation(line: 9, column: 1, scope: !155)
// CHECK:STDOUT: !167 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.454fecbcedcd86e1:core.Destroy.Core", scope: null, file: !145, line: 13, type: !22, spFlags: DISPFlagDefinition, unit: !146, retainedNodes: !170)
// CHECK:STDOUT: !168 = !DILocalVariable(arg: 1, scope: !167, type: !20)
// CHECK:STDOUT: !169 = !DILocation(line: 13, column: 3, scope: !167)
// CHECK:STDOUT: !170 = !{!168}
// CHECK:STDOUT: !171 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.6dae29ece9a708c4:core.Destroy.Core", scope: null, file: !145, line: 13, type: !149, spFlags: DISPFlagDefinition, unit: !146, retainedNodes: !174)
// CHECK:STDOUT: !172 = !DILocalVariable(arg: 1, scope: !171, type: !147)
// CHECK:STDOUT: !173 = !DILocation(line: 13, column: 3, scope: !171)
// CHECK:STDOUT: !174 = !{!172}
// CHECK:STDOUT: !175 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core", scope: null, file: !145, line: 13, type: !149, spFlags: DISPFlagDefinition, unit: !146, retainedNodes: !178)
// CHECK:STDOUT: !176 = !DILocalVariable(arg: 1, scope: !175, type: !147)
// CHECK:STDOUT: !177 = !DILocation(line: 13, column: 3, scope: !175)
// CHECK:STDOUT: !178 = !{!176}
// CHECK:STDOUT: !179 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.f58524831c37807e:core.Destroy.Core", scope: null, file: !145, line: 13, type: !149, spFlags: DISPFlagDefinition, unit: !146, retainedNodes: !182)
// CHECK:STDOUT: !180 = !DILocalVariable(arg: 1, scope: !179, type: !147)
// CHECK:STDOUT: !181 = !DILocation(line: 13, column: 3, scope: !179)
// CHECK:STDOUT: !182 = !{!180}
// CHECK:STDOUT: !183 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.fd9d4a98e93fc879:core.Destroy.Core", scope: null, file: !145, line: 13, type: !22, spFlags: DISPFlagDefinition, unit: !146, retainedNodes: !186)
// CHECK:STDOUT: !184 = !DILocalVariable(arg: 1, scope: !183, type: !20)
// CHECK:STDOUT: !185 = !DILocation(line: 13, column: 3, scope: !183)
// CHECK:STDOUT: !186 = !{!184}
// CHECK:STDOUT: !187 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.fd9d4a98e93fc879:core.Destroy.Core", scope: null, file: !145, line: 13, type: !22, spFlags: DISPFlagDefinition, unit: !146, retainedNodes: !190)
// CHECK:STDOUT: !188 = !DILocalVariable(arg: 1, scope: !187, type: !20)
// CHECK:STDOUT: !189 = !DILocation(line: 13, column: 3, scope: !187)
// CHECK:STDOUT: !190 = !{!188}
// CHECK:STDOUT: !191 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.c6dbe0017b4c8d16:core.Destroy.Core", scope: null, file: !145, line: 13, type: !22, spFlags: DISPFlagDefinition, unit: !146, retainedNodes: !194)
// CHECK:STDOUT: !192 = !DILocalVariable(arg: 1, scope: !191, type: !20)
// CHECK:STDOUT: !193 = !DILocation(line: 13, column: 3, scope: !191)
// CHECK:STDOUT: !194 = !{!192}
// CHECK:STDOUT: !195 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.c6dbe0017b4c8d16:core.Destroy.Core", scope: null, file: !145, line: 13, type: !22, spFlags: DISPFlagDefinition, unit: !146, retainedNodes: !198)
// CHECK:STDOUT: !196 = !DILocalVariable(arg: 1, scope: !195, type: !20)
// CHECK:STDOUT: !197 = !DILocation(line: 13, column: 3, scope: !195)
// CHECK:STDOUT: !198 = !{!196}
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
// CHECK:STDOUT: define void @_CUse.Main() #0 !dbg !201 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca { { ptr } }, align 8, !dbg !202
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !202
// CHECK:STDOUT:   %.loc13_32.2.base = getelementptr inbounds nuw { { ptr } }, ptr %_.var, i32 0, i32 0, !dbg !203
// CHECK:STDOUT:   %.loc13_31.2.vptr = getelementptr inbounds nuw { ptr }, ptr %.loc13_32.2.base, i32 0, i32 0, !dbg !204
// CHECK:STDOUT:   %.loc13_32.6.base = getelementptr inbounds nuw { { ptr } }, ptr %_.var, i32 0, i32 0, !dbg !203
// CHECK:STDOUT:   %.loc13_32.7.vptr = getelementptr inbounds nuw { ptr }, ptr %.loc13_32.6.base, i32 0, i32 0, !dbg !203
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 %_.var, ptr align 8 @Derived.val.loc13_32.5, i64 8, i1 false), !dbg !203
// CHECK:STDOUT:   store ptr @"_CDerived.Main.$vtable", ptr %.loc13_32.7.vptr, align 8, !dbg !203
// CHECK:STDOUT:   call void @"_CSelfDestruct.b25525d21a9bd472:core.Destroy.Core"(ptr %_.var), !dbg !202
// CHECK:STDOUT:   ret void, !dbg !206
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.454fecbcedcd86e1:core.Destroy.Core"(ptr %self) #0 !dbg !207 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !209
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.95528a86e1127e2d:core.Destroy.Core"(ptr %self) #0 !dbg !211 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc13.vptr = getelementptr inbounds nuw { ptr }, ptr %self, i32 0, i32 0, !dbg !213
// CHECK:STDOUT:   call void @"_CSelfDestruct.454fecbcedcd86e1:core.Destroy.Core"(ptr %.loc13.vptr), !dbg !213
// CHECK:STDOUT:   ret void, !dbg !213
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.95528a86e1127e2d:core.Destroy.Core"(ptr %self) #0 !dbg !215 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.95528a86e1127e2d:core.Destroy.Core"(ptr %self), !dbg !217
// CHECK:STDOUT:   ret void, !dbg !217
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.294df737cd074d60:core.Destroy.Core"(ptr %self) #0 !dbg !219 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !221
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.294df737cd074d60:core.Destroy.Core"(ptr %self) #0 !dbg !223 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.294df737cd074d60:core.Destroy.Core"(ptr %self), !dbg !225
// CHECK:STDOUT:   ret void, !dbg !225
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.caed8a96bf328d09:core.Destroy.Core"(ptr %self) #0 !dbg !227 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc13.base = getelementptr inbounds nuw { { ptr } }, ptr %self, i32 0, i32 0, !dbg !229
// CHECK:STDOUT:   call void @"_CSelfDestruct.294df737cd074d60:core.Destroy.Core"(ptr %.loc13.base), !dbg !229
// CHECK:STDOUT:   ret void, !dbg !229
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.caed8a96bf328d09:core.Destroy.Core"(ptr %self) #0 !dbg !231 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.caed8a96bf328d09:core.Destroy.Core"(ptr %self), !dbg !233
// CHECK:STDOUT:   ret void, !dbg !233
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.b25525d21a9bd472:core.Destroy.Core"(ptr %self) #0 !dbg !235 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !237
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.b25525d21a9bd472:core.Destroy.Core"(ptr %self) #0 !dbg !239 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.b25525d21a9bd472:core.Destroy.Core"(ptr %self), !dbg !241
// CHECK:STDOUT:   ret void, !dbg !241
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!200}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !21 = !{null, !20}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !40 = !{null}
// CHECK:STDOUT: !41 = !DISubroutineType(types: !40)
// CHECK:STDOUT: !199 = !DIFile(filename: "member_brace_init.carbon", directory: "")
// CHECK:STDOUT: !200 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !199, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !201 = distinct !DISubprogram(name: "Use", linkageName: "_CUse.Main", scope: null, file: !199, line: 12, type: !41, spFlags: DISPFlagDefinition, unit: !200)
// CHECK:STDOUT: !202 = !DILocation(line: 13, column: 3, scope: !201)
// CHECK:STDOUT: !203 = !DILocation(line: 13, column: 21, scope: !201)
// CHECK:STDOUT: !204 = !DILocation(line: 13, column: 30, scope: !201)
// CHECK:STDOUT: !206 = !DILocation(line: 12, column: 1, scope: !201)
// CHECK:STDOUT: !207 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.454fecbcedcd86e1:core.Destroy.Core", scope: null, file: !199, line: 13, type: !22, spFlags: DISPFlagDefinition, unit: !200, retainedNodes: !210)
// CHECK:STDOUT: !208 = !DILocalVariable(arg: 1, scope: !207, type: !20)
// CHECK:STDOUT: !209 = !DILocation(line: 13, column: 3, scope: !207)
// CHECK:STDOUT: !210 = !{!208}
// CHECK:STDOUT: !211 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.95528a86e1127e2d:core.Destroy.Core", scope: null, file: !199, line: 13, type: !22, spFlags: DISPFlagDefinition, unit: !200, retainedNodes: !214)
// CHECK:STDOUT: !212 = !DILocalVariable(arg: 1, scope: !211, type: !20)
// CHECK:STDOUT: !213 = !DILocation(line: 13, column: 3, scope: !211)
// CHECK:STDOUT: !214 = !{!212}
// CHECK:STDOUT: !215 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.95528a86e1127e2d:core.Destroy.Core", scope: null, file: !199, line: 13, type: !22, spFlags: DISPFlagDefinition, unit: !200, retainedNodes: !218)
// CHECK:STDOUT: !216 = !DILocalVariable(arg: 1, scope: !215, type: !20)
// CHECK:STDOUT: !217 = !DILocation(line: 13, column: 3, scope: !215)
// CHECK:STDOUT: !218 = !{!216}
// CHECK:STDOUT: !219 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.294df737cd074d60:core.Destroy.Core", scope: null, file: !199, line: 13, type: !22, spFlags: DISPFlagDefinition, unit: !200, retainedNodes: !222)
// CHECK:STDOUT: !220 = !DILocalVariable(arg: 1, scope: !219, type: !20)
// CHECK:STDOUT: !221 = !DILocation(line: 13, column: 3, scope: !219)
// CHECK:STDOUT: !222 = !{!220}
// CHECK:STDOUT: !223 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.294df737cd074d60:core.Destroy.Core", scope: null, file: !199, line: 13, type: !22, spFlags: DISPFlagDefinition, unit: !200, retainedNodes: !226)
// CHECK:STDOUT: !224 = !DILocalVariable(arg: 1, scope: !223, type: !20)
// CHECK:STDOUT: !225 = !DILocation(line: 13, column: 3, scope: !223)
// CHECK:STDOUT: !226 = !{!224}
// CHECK:STDOUT: !227 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.caed8a96bf328d09:core.Destroy.Core", scope: null, file: !199, line: 13, type: !22, spFlags: DISPFlagDefinition, unit: !200, retainedNodes: !230)
// CHECK:STDOUT: !228 = !DILocalVariable(arg: 1, scope: !227, type: !20)
// CHECK:STDOUT: !229 = !DILocation(line: 13, column: 3, scope: !227)
// CHECK:STDOUT: !230 = !{!228}
// CHECK:STDOUT: !231 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.caed8a96bf328d09:core.Destroy.Core", scope: null, file: !199, line: 13, type: !22, spFlags: DISPFlagDefinition, unit: !200, retainedNodes: !234)
// CHECK:STDOUT: !232 = !DILocalVariable(arg: 1, scope: !231, type: !20)
// CHECK:STDOUT: !233 = !DILocation(line: 13, column: 3, scope: !231)
// CHECK:STDOUT: !234 = !{!232}
// CHECK:STDOUT: !235 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.b25525d21a9bd472:core.Destroy.Core", scope: null, file: !199, line: 13, type: !22, spFlags: DISPFlagDefinition, unit: !200, retainedNodes: !238)
// CHECK:STDOUT: !236 = !DILocalVariable(arg: 1, scope: !235, type: !20)
// CHECK:STDOUT: !237 = !DILocation(line: 13, column: 3, scope: !235)
// CHECK:STDOUT: !238 = !{!236}
// CHECK:STDOUT: !239 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.b25525d21a9bd472:core.Destroy.Core", scope: null, file: !199, line: 13, type: !22, spFlags: DISPFlagDefinition, unit: !200, retainedNodes: !242)
// CHECK:STDOUT: !240 = !DILocalVariable(arg: 1, scope: !239, type: !20)
// CHECK:STDOUT: !241 = !DILocation(line: 13, column: 3, scope: !239)
// CHECK:STDOUT: !242 = !{!240}
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
// CHECK:STDOUT: define void @_CUse.Main(ptr %b) #0 !dbg !245 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Base.F.call.vtable = load ptr, ptr %b, align 8, !dbg !248
// CHECK:STDOUT:   %Base.F.call = call ptr @llvm.load.relative.i32(ptr %Base.F.call.vtable, i32 0), !dbg !248
// CHECK:STDOUT:   call void %Base.F.call(ptr %b), !dbg !248
// CHECK:STDOUT:   ret void, !dbg !249
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: read)
// CHECK:STDOUT: declare ptr @llvm.load.relative.i32(ptr, i32) #1
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: read) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!244}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !21 = !{null, !20}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !243 = !DIFile(filename: "call.carbon", directory: "")
// CHECK:STDOUT: !244 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !243, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !245 = distinct !DISubprogram(name: "Use", linkageName: "_CUse.Main", scope: null, file: !243, line: 8, type: !22, spFlags: DISPFlagDefinition, unit: !244, retainedNodes: !250)
// CHECK:STDOUT: !246 = !DILocalVariable(arg: 1, scope: !245, type: !20)
// CHECK:STDOUT: !248 = !DILocation(line: 9, column: 3, scope: !245)
// CHECK:STDOUT: !249 = !DILocation(line: 8, column: 1, scope: !245)
// CHECK:STDOUT: !250 = !{!246}
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
// CHECK:STDOUT: define weak_odr void @"_CF:thunk:Base.Main:Derived.Main"(ptr %self) #0 !dbg !253 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Derived.Main(ptr %self), !dbg !256
// CHECK:STDOUT:   ret void, !dbg !256
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CUse.Main() #1 !dbg !258 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %v.var = alloca { { ptr } }, align 8, !dbg !259
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %v.var), !dbg !259
// CHECK:STDOUT:   %.loc14_32.2.base = getelementptr inbounds nuw { { ptr } }, ptr %v.var, i32 0, i32 0, !dbg !260
// CHECK:STDOUT:   %.loc14_31.2.vptr = getelementptr inbounds nuw { ptr }, ptr %.loc14_32.2.base, i32 0, i32 0, !dbg !261
// CHECK:STDOUT:   %.loc14_32.6.base = getelementptr inbounds nuw { { ptr } }, ptr %v.var, i32 0, i32 0, !dbg !260
// CHECK:STDOUT:   %.loc14_32.7.vptr = getelementptr inbounds nuw { ptr }, ptr %.loc14_32.6.base, i32 0, i32 0, !dbg !260
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 %v.var, ptr align 8 @Derived.val.loc14_32.5, i64 8, i1 false), !dbg !260
// CHECK:STDOUT:   store ptr @"_CDerived.Main.$vtable", ptr %.loc14_32.7.vptr, align 8, !dbg !260
// CHECK:STDOUT:   %.loc15_3.1.base = getelementptr inbounds nuw { { ptr } }, ptr %v.var, i32 0, i32 0, !dbg !263
// CHECK:STDOUT:   %Base.F.call.vtable = load ptr, ptr %.loc15_3.1.base, align 8, !dbg !263
// CHECK:STDOUT:   %Base.F.call = call ptr @llvm.load.relative.i32(ptr %Base.F.call.vtable, i32 0), !dbg !263
// CHECK:STDOUT:   call void %Base.F.call(ptr %.loc15_3.1.base), !dbg !263
// CHECK:STDOUT:   call void @"_CSelfDestruct.b25525d21a9bd472:core.Destroy.Core"(ptr %v.var), !dbg !259
// CHECK:STDOUT:   ret void, !dbg !264
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.454fecbcedcd86e1:core.Destroy.Core"(ptr %self) #1 !dbg !265 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !267
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.95528a86e1127e2d:core.Destroy.Core"(ptr %self) #1 !dbg !269 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc14.vptr = getelementptr inbounds nuw { ptr }, ptr %self, i32 0, i32 0, !dbg !271
// CHECK:STDOUT:   call void @"_CSelfDestruct.454fecbcedcd86e1:core.Destroy.Core"(ptr %.loc14.vptr), !dbg !271
// CHECK:STDOUT:   ret void, !dbg !271
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.95528a86e1127e2d:core.Destroy.Core"(ptr %self) #1 !dbg !273 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.95528a86e1127e2d:core.Destroy.Core"(ptr %self), !dbg !275
// CHECK:STDOUT:   ret void, !dbg !275
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.294df737cd074d60:core.Destroy.Core"(ptr %self) #1 !dbg !277 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !279
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.294df737cd074d60:core.Destroy.Core"(ptr %self) #1 !dbg !281 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.294df737cd074d60:core.Destroy.Core"(ptr %self), !dbg !283
// CHECK:STDOUT:   ret void, !dbg !283
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.caed8a96bf328d09:core.Destroy.Core"(ptr %self) #1 !dbg !285 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc14.base = getelementptr inbounds nuw { { ptr } }, ptr %self, i32 0, i32 0, !dbg !287
// CHECK:STDOUT:   call void @"_CSelfDestruct.294df737cd074d60:core.Destroy.Core"(ptr %.loc14.base), !dbg !287
// CHECK:STDOUT:   ret void, !dbg !287
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.caed8a96bf328d09:core.Destroy.Core"(ptr %self) #1 !dbg !289 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.caed8a96bf328d09:core.Destroy.Core"(ptr %self), !dbg !291
// CHECK:STDOUT:   ret void, !dbg !291
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.b25525d21a9bd472:core.Destroy.Core"(ptr %self) #1 !dbg !293 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !295
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.b25525d21a9bd472:core.Destroy.Core"(ptr %self) #1 !dbg !297 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.b25525d21a9bd472:core.Destroy.Core"(ptr %self), !dbg !299
// CHECK:STDOUT:   ret void, !dbg !299
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!252}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !21 = !{null, !20}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !40 = !{null}
// CHECK:STDOUT: !41 = !DISubroutineType(types: !40)
// CHECK:STDOUT: !251 = !DIFile(filename: "call_impl.carbon", directory: "")
// CHECK:STDOUT: !252 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !251, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !253 = distinct !DISubprogram(name: "F", linkageName: "_CF:thunk:Base.Main:Derived.Main", scope: null, file: !251, line: 10, type: !22, spFlags: DISPFlagDefinition, unit: !252, retainedNodes: !257)
// CHECK:STDOUT: !254 = !DILocalVariable(arg: 1, scope: !253, type: !20)
// CHECK:STDOUT: !256 = !DILocation(line: 10, column: 3, scope: !253)
// CHECK:STDOUT: !257 = !{!254}
// CHECK:STDOUT: !258 = distinct !DISubprogram(name: "Use", linkageName: "_CUse.Main", scope: null, file: !251, line: 13, type: !41, spFlags: DISPFlagDefinition, unit: !252)
// CHECK:STDOUT: !259 = !DILocation(line: 14, column: 3, scope: !258)
// CHECK:STDOUT: !260 = !DILocation(line: 14, column: 21, scope: !258)
// CHECK:STDOUT: !261 = !DILocation(line: 14, column: 30, scope: !258)
// CHECK:STDOUT: !263 = !DILocation(line: 15, column: 3, scope: !258)
// CHECK:STDOUT: !264 = !DILocation(line: 13, column: 1, scope: !258)
// CHECK:STDOUT: !265 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.454fecbcedcd86e1:core.Destroy.Core", scope: null, file: !251, line: 14, type: !22, spFlags: DISPFlagDefinition, unit: !252, retainedNodes: !268)
// CHECK:STDOUT: !266 = !DILocalVariable(arg: 1, scope: !265, type: !20)
// CHECK:STDOUT: !267 = !DILocation(line: 14, column: 3, scope: !265)
// CHECK:STDOUT: !268 = !{!266}
// CHECK:STDOUT: !269 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.95528a86e1127e2d:core.Destroy.Core", scope: null, file: !251, line: 14, type: !22, spFlags: DISPFlagDefinition, unit: !252, retainedNodes: !272)
// CHECK:STDOUT: !270 = !DILocalVariable(arg: 1, scope: !269, type: !20)
// CHECK:STDOUT: !271 = !DILocation(line: 14, column: 3, scope: !269)
// CHECK:STDOUT: !272 = !{!270}
// CHECK:STDOUT: !273 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.95528a86e1127e2d:core.Destroy.Core", scope: null, file: !251, line: 14, type: !22, spFlags: DISPFlagDefinition, unit: !252, retainedNodes: !276)
// CHECK:STDOUT: !274 = !DILocalVariable(arg: 1, scope: !273, type: !20)
// CHECK:STDOUT: !275 = !DILocation(line: 14, column: 3, scope: !273)
// CHECK:STDOUT: !276 = !{!274}
// CHECK:STDOUT: !277 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.294df737cd074d60:core.Destroy.Core", scope: null, file: !251, line: 14, type: !22, spFlags: DISPFlagDefinition, unit: !252, retainedNodes: !280)
// CHECK:STDOUT: !278 = !DILocalVariable(arg: 1, scope: !277, type: !20)
// CHECK:STDOUT: !279 = !DILocation(line: 14, column: 3, scope: !277)
// CHECK:STDOUT: !280 = !{!278}
// CHECK:STDOUT: !281 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.294df737cd074d60:core.Destroy.Core", scope: null, file: !251, line: 14, type: !22, spFlags: DISPFlagDefinition, unit: !252, retainedNodes: !284)
// CHECK:STDOUT: !282 = !DILocalVariable(arg: 1, scope: !281, type: !20)
// CHECK:STDOUT: !283 = !DILocation(line: 14, column: 3, scope: !281)
// CHECK:STDOUT: !284 = !{!282}
// CHECK:STDOUT: !285 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.caed8a96bf328d09:core.Destroy.Core", scope: null, file: !251, line: 14, type: !22, spFlags: DISPFlagDefinition, unit: !252, retainedNodes: !288)
// CHECK:STDOUT: !286 = !DILocalVariable(arg: 1, scope: !285, type: !20)
// CHECK:STDOUT: !287 = !DILocation(line: 14, column: 3, scope: !285)
// CHECK:STDOUT: !288 = !{!286}
// CHECK:STDOUT: !289 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.caed8a96bf328d09:core.Destroy.Core", scope: null, file: !251, line: 14, type: !22, spFlags: DISPFlagDefinition, unit: !252, retainedNodes: !292)
// CHECK:STDOUT: !290 = !DILocalVariable(arg: 1, scope: !289, type: !20)
// CHECK:STDOUT: !291 = !DILocation(line: 14, column: 3, scope: !289)
// CHECK:STDOUT: !292 = !{!290}
// CHECK:STDOUT: !293 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.b25525d21a9bd472:core.Destroy.Core", scope: null, file: !251, line: 14, type: !22, spFlags: DISPFlagDefinition, unit: !252, retainedNodes: !296)
// CHECK:STDOUT: !294 = !DILocalVariable(arg: 1, scope: !293, type: !20)
// CHECK:STDOUT: !295 = !DILocation(line: 14, column: 3, scope: !293)
// CHECK:STDOUT: !296 = !{!294}
// CHECK:STDOUT: !297 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.b25525d21a9bd472:core.Destroy.Core", scope: null, file: !251, line: 14, type: !22, spFlags: DISPFlagDefinition, unit: !252, retainedNodes: !300)
// CHECK:STDOUT: !298 = !DILocalVariable(arg: 1, scope: !297, type: !20)
// CHECK:STDOUT: !299 = !DILocation(line: 14, column: 3, scope: !297)
// CHECK:STDOUT: !300 = !{!298}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'generic_noop.carbon'
// CHECK:STDOUT: source_filename = "generic_noop.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!302}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !301 = !DIFile(filename: "generic_noop.carbon", directory: "")
// CHECK:STDOUT: !302 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !301, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
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
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !305 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var.loc16 = alloca { ptr }, align 8, !dbg !306
// CHECK:STDOUT:   %_.var.loc17 = alloca { ptr }, align 8, !dbg !309
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc16), !dbg !306
// CHECK:STDOUT:   %.loc16_22.2.vptr = getelementptr inbounds nuw { ptr }, ptr %_.var.loc16, i32 0, i32 0, !dbg !307
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 %_.var.loc16, ptr align 8 @Base.val.a7b.loc16_3, i64 8, i1 false), !dbg !306
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc17), !dbg !309
// CHECK:STDOUT:   %.loc17_22.2.vptr = getelementptr inbounds nuw { ptr }, ptr %_.var.loc17, i32 0, i32 0, !dbg !310
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 %_.var.loc17, ptr align 8 @Base.val.3f1.loc17_3, i64 8, i1 false), !dbg !309
// CHECK:STDOUT:   call void @"_CSelfDestruct.c3ce2497e05f4476:core.Destroy.Core"(ptr %_.var.loc17), !dbg !309
// CHECK:STDOUT:   call void @"_CSelfDestruct.b64fd0f098bf5c8e:core.Destroy.Core"(ptr %_.var.loc16), !dbg !306
// CHECK:STDOUT:   ret void, !dbg !312
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.cb8adced19392b99:core.Destroy.Core"(ptr %self) #0 !dbg !313 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !315
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.13f86630f401913f:core.Destroy.Core"(ptr %self) #0 !dbg !317 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !319
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.13f86630f401913f:core.Destroy.Core"(ptr %self) #0 !dbg !321 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.13f86630f401913f:core.Destroy.Core"(ptr %self), !dbg !323
// CHECK:STDOUT:   ret void, !dbg !323
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.f2f12ab22a0a6b87:core.Destroy.Core"(ptr %self) #0 !dbg !325 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc17.v = getelementptr inbounds nuw { {} }, ptr %self, i32 0, i32 0, !dbg !327
// CHECK:STDOUT:   call void @"_CSelfDestruct.13f86630f401913f:core.Destroy.Core"(ptr %.loc17.v), !dbg !327
// CHECK:STDOUT:   ret void, !dbg !327
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.f2f12ab22a0a6b87:core.Destroy.Core"(ptr %self) #0 !dbg !329 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.f2f12ab22a0a6b87:core.Destroy.Core"(ptr %self), !dbg !331
// CHECK:STDOUT:   ret void, !dbg !331
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.8578661a61c8bbbf:core.Destroy.Core"(ptr %self) #0 !dbg !333 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc17.v = getelementptr inbounds nuw { {} }, ptr %self, i32 0, i32 0, !dbg !335
// CHECK:STDOUT:   call void @"_CSelfDestruct.13f86630f401913f:core.Destroy.Core"(ptr %.loc17.v), !dbg !335
// CHECK:STDOUT:   ret void, !dbg !335
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.8578661a61c8bbbf:core.Destroy.Core"(ptr %self) #0 !dbg !337 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.8578661a61c8bbbf:core.Destroy.Core"(ptr %self), !dbg !339
// CHECK:STDOUT:   ret void, !dbg !339
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.454fecbcedcd86e1:core.Destroy.Core"(ptr %self) #0 !dbg !341 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !343
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.95528a86e1127e2d:core.Destroy.Core"(ptr %self) #0 !dbg !345 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc17.vptr = getelementptr inbounds nuw { ptr }, ptr %self, i32 0, i32 0, !dbg !347
// CHECK:STDOUT:   call void @"_CSelfDestruct.454fecbcedcd86e1:core.Destroy.Core"(ptr %.loc17.vptr), !dbg !347
// CHECK:STDOUT:   ret void, !dbg !347
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.95528a86e1127e2d:core.Destroy.Core"(ptr %self) #0 !dbg !349 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.95528a86e1127e2d:core.Destroy.Core"(ptr %self), !dbg !351
// CHECK:STDOUT:   ret void, !dbg !351
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.c3ce2497e05f4476:core.Destroy.Core"(ptr %self) #0 !dbg !353 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !355
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.c3ce2497e05f4476:core.Destroy.Core"(ptr %self) #0 !dbg !357 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.c3ce2497e05f4476:core.Destroy.Core"(ptr %self), !dbg !359
// CHECK:STDOUT:   ret void, !dbg !359
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.b64fd0f098bf5c8e:core.Destroy.Core"(ptr %self) #0 !dbg !361 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !363
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.b64fd0f098bf5c8e:core.Destroy.Core"(ptr %self) #0 !dbg !365 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.b64fd0f098bf5c8e:core.Destroy.Core"(ptr %self), !dbg !367
// CHECK:STDOUT:   ret void, !dbg !367
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Base.Main.2f8a0042309f394e(ptr %self) #0 !dbg !369 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca {}, align 1, !dbg !372
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !372
// CHECK:STDOUT:   call void @"_CSelfDestruct.13f86630f401913f:core.Destroy.Core"(ptr %_.var), !dbg !372
// CHECK:STDOUT:   ret void, !dbg !374
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Base.Main.d7d44a05ee79f1b8(ptr %self) #0 !dbg !376 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca { {} }, align 1, !dbg !379
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !379
// CHECK:STDOUT:   call void @"_CSelfDestruct.8578661a61c8bbbf:core.Destroy.Core"(ptr %_.var), !dbg !379
// CHECK:STDOUT:   ret void, !dbg !381
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!304}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !21 = !{null, !20}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !40 = !{null}
// CHECK:STDOUT: !41 = !DISubroutineType(types: !40)
// CHECK:STDOUT: !303 = !DIFile(filename: "generic_use.carbon", directory: "")
// CHECK:STDOUT: !304 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !303, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !305 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !303, line: 15, type: !41, spFlags: DISPFlagDefinition, unit: !304)
// CHECK:STDOUT: !306 = !DILocation(line: 16, column: 3, scope: !305)
// CHECK:STDOUT: !307 = !DILocation(line: 16, column: 21, scope: !305)
// CHECK:STDOUT: !309 = !DILocation(line: 17, column: 3, scope: !305)
// CHECK:STDOUT: !310 = !DILocation(line: 17, column: 21, scope: !305)
// CHECK:STDOUT: !312 = !DILocation(line: 15, column: 1, scope: !305)
// CHECK:STDOUT: !313 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.cb8adced19392b99:core.Destroy.Core", scope: null, file: !303, line: 16, type: !22, spFlags: DISPFlagDefinition, unit: !304, retainedNodes: !316)
// CHECK:STDOUT: !314 = !DILocalVariable(arg: 1, scope: !313, type: !20)
// CHECK:STDOUT: !315 = !DILocation(line: 16, column: 10, scope: !313)
// CHECK:STDOUT: !316 = !{!314}
// CHECK:STDOUT: !317 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.13f86630f401913f:core.Destroy.Core", scope: null, file: !303, line: 16, type: !22, spFlags: DISPFlagDefinition, unit: !304, retainedNodes: !320)
// CHECK:STDOUT: !318 = !DILocalVariable(arg: 1, scope: !317, type: !20)
// CHECK:STDOUT: !319 = !DILocation(line: 16, column: 10, scope: !317)
// CHECK:STDOUT: !320 = !{!318}
// CHECK:STDOUT: !321 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.13f86630f401913f:core.Destroy.Core", scope: null, file: !303, line: 16, type: !22, spFlags: DISPFlagDefinition, unit: !304, retainedNodes: !324)
// CHECK:STDOUT: !322 = !DILocalVariable(arg: 1, scope: !321, type: !20)
// CHECK:STDOUT: !323 = !DILocation(line: 16, column: 10, scope: !321)
// CHECK:STDOUT: !324 = !{!322}
// CHECK:STDOUT: !325 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.f2f12ab22a0a6b87:core.Destroy.Core", scope: null, file: !303, line: 17, type: !22, spFlags: DISPFlagDefinition, unit: !304, retainedNodes: !328)
// CHECK:STDOUT: !326 = !DILocalVariable(arg: 1, scope: !325, type: !20)
// CHECK:STDOUT: !327 = !DILocation(line: 17, column: 10, scope: !325)
// CHECK:STDOUT: !328 = !{!326}
// CHECK:STDOUT: !329 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.f2f12ab22a0a6b87:core.Destroy.Core", scope: null, file: !303, line: 17, type: !22, spFlags: DISPFlagDefinition, unit: !304, retainedNodes: !332)
// CHECK:STDOUT: !330 = !DILocalVariable(arg: 1, scope: !329, type: !20)
// CHECK:STDOUT: !331 = !DILocation(line: 17, column: 10, scope: !329)
// CHECK:STDOUT: !332 = !{!330}
// CHECK:STDOUT: !333 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.8578661a61c8bbbf:core.Destroy.Core", scope: null, file: !303, line: 17, type: !22, spFlags: DISPFlagDefinition, unit: !304, retainedNodes: !336)
// CHECK:STDOUT: !334 = !DILocalVariable(arg: 1, scope: !333, type: !20)
// CHECK:STDOUT: !335 = !DILocation(line: 17, column: 10, scope: !333)
// CHECK:STDOUT: !336 = !{!334}
// CHECK:STDOUT: !337 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.8578661a61c8bbbf:core.Destroy.Core", scope: null, file: !303, line: 17, type: !22, spFlags: DISPFlagDefinition, unit: !304, retainedNodes: !340)
// CHECK:STDOUT: !338 = !DILocalVariable(arg: 1, scope: !337, type: !20)
// CHECK:STDOUT: !339 = !DILocation(line: 17, column: 10, scope: !337)
// CHECK:STDOUT: !340 = !{!338}
// CHECK:STDOUT: !341 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.454fecbcedcd86e1:core.Destroy.Core", scope: null, file: !303, line: 17, type: !22, spFlags: DISPFlagDefinition, unit: !304, retainedNodes: !344)
// CHECK:STDOUT: !342 = !DILocalVariable(arg: 1, scope: !341, type: !20)
// CHECK:STDOUT: !343 = !DILocation(line: 17, column: 3, scope: !341)
// CHECK:STDOUT: !344 = !{!342}
// CHECK:STDOUT: !345 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.95528a86e1127e2d:core.Destroy.Core", scope: null, file: !303, line: 17, type: !22, spFlags: DISPFlagDefinition, unit: !304, retainedNodes: !348)
// CHECK:STDOUT: !346 = !DILocalVariable(arg: 1, scope: !345, type: !20)
// CHECK:STDOUT: !347 = !DILocation(line: 17, column: 3, scope: !345)
// CHECK:STDOUT: !348 = !{!346}
// CHECK:STDOUT: !349 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.95528a86e1127e2d:core.Destroy.Core", scope: null, file: !303, line: 17, type: !22, spFlags: DISPFlagDefinition, unit: !304, retainedNodes: !352)
// CHECK:STDOUT: !350 = !DILocalVariable(arg: 1, scope: !349, type: !20)
// CHECK:STDOUT: !351 = !DILocation(line: 17, column: 3, scope: !349)
// CHECK:STDOUT: !352 = !{!350}
// CHECK:STDOUT: !353 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.c3ce2497e05f4476:core.Destroy.Core", scope: null, file: !303, line: 17, type: !22, spFlags: DISPFlagDefinition, unit: !304, retainedNodes: !356)
// CHECK:STDOUT: !354 = !DILocalVariable(arg: 1, scope: !353, type: !20)
// CHECK:STDOUT: !355 = !DILocation(line: 17, column: 3, scope: !353)
// CHECK:STDOUT: !356 = !{!354}
// CHECK:STDOUT: !357 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.c3ce2497e05f4476:core.Destroy.Core", scope: null, file: !303, line: 17, type: !22, spFlags: DISPFlagDefinition, unit: !304, retainedNodes: !360)
// CHECK:STDOUT: !358 = !DILocalVariable(arg: 1, scope: !357, type: !20)
// CHECK:STDOUT: !359 = !DILocation(line: 17, column: 3, scope: !357)
// CHECK:STDOUT: !360 = !{!358}
// CHECK:STDOUT: !361 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.b64fd0f098bf5c8e:core.Destroy.Core", scope: null, file: !303, line: 16, type: !22, spFlags: DISPFlagDefinition, unit: !304, retainedNodes: !364)
// CHECK:STDOUT: !362 = !DILocalVariable(arg: 1, scope: !361, type: !20)
// CHECK:STDOUT: !363 = !DILocation(line: 16, column: 3, scope: !361)
// CHECK:STDOUT: !364 = !{!362}
// CHECK:STDOUT: !365 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.b64fd0f098bf5c8e:core.Destroy.Core", scope: null, file: !303, line: 16, type: !22, spFlags: DISPFlagDefinition, unit: !304, retainedNodes: !368)
// CHECK:STDOUT: !366 = !DILocalVariable(arg: 1, scope: !365, type: !20)
// CHECK:STDOUT: !367 = !DILocation(line: 16, column: 3, scope: !365)
// CHECK:STDOUT: !368 = !{!366}
// CHECK:STDOUT: !369 = distinct !DISubprogram(name: "F", linkageName: "_CF.Base.Main.2f8a0042309f394e", scope: null, file: !303, line: 5, type: !22, spFlags: DISPFlagDefinition, unit: !304, retainedNodes: !375)
// CHECK:STDOUT: !370 = !DILocalVariable(arg: 1, scope: !369, type: !20)
// CHECK:STDOUT: !372 = !DILocation(line: 6, column: 5, scope: !369)
// CHECK:STDOUT: !374 = !DILocation(line: 5, column: 3, scope: !369)
// CHECK:STDOUT: !375 = !{!370}
// CHECK:STDOUT: !376 = distinct !DISubprogram(name: "F", linkageName: "_CF.Base.Main.d7d44a05ee79f1b8", scope: null, file: !303, line: 5, type: !22, spFlags: DISPFlagDefinition, unit: !304, retainedNodes: !382)
// CHECK:STDOUT: !377 = !DILocalVariable(arg: 1, scope: !376, type: !20)
// CHECK:STDOUT: !379 = !DILocation(line: 6, column: 5, scope: !376)
// CHECK:STDOUT: !381 = !DILocation(line: 5, column: 3, scope: !376)
// CHECK:STDOUT: !382 = !{!377}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'generic_base.carbon'
// CHECK:STDOUT: source_filename = "generic_base.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @"_CDerived.Main.$vtable" = unnamed_addr constant [1 x i32] [i32 trunc (i64 sub (i64 ptrtoint (ptr @_CF.Base.Main.6d6a4a2512463e59 to i64), i64 ptrtoint (ptr @"_CDerived.Main.$vtable" to i64)) to i32)]
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMake.Main() #0 !dbg !385 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca { { ptr } }, align 8, !dbg !386
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !386
// CHECK:STDOUT:   call void @"_CSelfDestruct.b25525d21a9bd472:core.Destroy.Core"(ptr %_.var), !dbg !386
// CHECK:STDOUT:   ret void, !dbg !388
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.454fecbcedcd86e1:core.Destroy.Core"(ptr %self) #0 !dbg !389 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !391
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.95528a86e1127e2d:core.Destroy.Core"(ptr %self) #0 !dbg !393 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc15.vptr = getelementptr inbounds nuw { ptr }, ptr %self, i32 0, i32 0, !dbg !395
// CHECK:STDOUT:   call void @"_CSelfDestruct.454fecbcedcd86e1:core.Destroy.Core"(ptr %.loc15.vptr), !dbg !395
// CHECK:STDOUT:   ret void, !dbg !395
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.95528a86e1127e2d:core.Destroy.Core"(ptr %self) #0 !dbg !397 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.95528a86e1127e2d:core.Destroy.Core"(ptr %self), !dbg !399
// CHECK:STDOUT:   ret void, !dbg !399
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.5b23b4f14feb87e1:core.Destroy.Core"(ptr %self) #0 !dbg !401 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !403
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.5b23b4f14feb87e1:core.Destroy.Core"(ptr %self) #0 !dbg !405 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.5b23b4f14feb87e1:core.Destroy.Core"(ptr %self), !dbg !407
// CHECK:STDOUT:   ret void, !dbg !407
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.2f190654c863951f:core.Destroy.Core"(ptr %self) #0 !dbg !409 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc15.base = getelementptr inbounds nuw { { ptr } }, ptr %self, i32 0, i32 0, !dbg !411
// CHECK:STDOUT:   call void @"_CSelfDestruct.5b23b4f14feb87e1:core.Destroy.Core"(ptr %.loc15.base), !dbg !411
// CHECK:STDOUT:   ret void, !dbg !411
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.2f190654c863951f:core.Destroy.Core"(ptr %self) #0 !dbg !413 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.2f190654c863951f:core.Destroy.Core"(ptr %self), !dbg !415
// CHECK:STDOUT:   ret void, !dbg !415
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.b25525d21a9bd472:core.Destroy.Core"(ptr %self) #0 !dbg !417 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !419
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.b25525d21a9bd472:core.Destroy.Core"(ptr %self) #0 !dbg !421 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.b25525d21a9bd472:core.Destroy.Core"(ptr %self), !dbg !423
// CHECK:STDOUT:   ret void, !dbg !423
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Base.Main.6d6a4a2512463e59(ptr %self) #0 !dbg !425 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !428
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!384}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !21 = !{null, !20}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !40 = !{null}
// CHECK:STDOUT: !41 = !DISubroutineType(types: !40)
// CHECK:STDOUT: !383 = !DIFile(filename: "generic_base.carbon", directory: "")
// CHECK:STDOUT: !384 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !383, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !385 = distinct !DISubprogram(name: "Make", linkageName: "_CMake.Main", scope: null, file: !383, line: 14, type: !41, spFlags: DISPFlagDefinition, unit: !384)
// CHECK:STDOUT: !386 = !DILocation(line: 15, column: 3, scope: !385)
// CHECK:STDOUT: !388 = !DILocation(line: 14, column: 1, scope: !385)
// CHECK:STDOUT: !389 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.454fecbcedcd86e1:core.Destroy.Core", scope: null, file: !383, line: 15, type: !22, spFlags: DISPFlagDefinition, unit: !384, retainedNodes: !392)
// CHECK:STDOUT: !390 = !DILocalVariable(arg: 1, scope: !389, type: !20)
// CHECK:STDOUT: !391 = !DILocation(line: 15, column: 3, scope: !389)
// CHECK:STDOUT: !392 = !{!390}
// CHECK:STDOUT: !393 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.95528a86e1127e2d:core.Destroy.Core", scope: null, file: !383, line: 15, type: !22, spFlags: DISPFlagDefinition, unit: !384, retainedNodes: !396)
// CHECK:STDOUT: !394 = !DILocalVariable(arg: 1, scope: !393, type: !20)
// CHECK:STDOUT: !395 = !DILocation(line: 15, column: 3, scope: !393)
// CHECK:STDOUT: !396 = !{!394}
// CHECK:STDOUT: !397 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.95528a86e1127e2d:core.Destroy.Core", scope: null, file: !383, line: 15, type: !22, spFlags: DISPFlagDefinition, unit: !384, retainedNodes: !400)
// CHECK:STDOUT: !398 = !DILocalVariable(arg: 1, scope: !397, type: !20)
// CHECK:STDOUT: !399 = !DILocation(line: 15, column: 3, scope: !397)
// CHECK:STDOUT: !400 = !{!398}
// CHECK:STDOUT: !401 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.5b23b4f14feb87e1:core.Destroy.Core", scope: null, file: !383, line: 15, type: !22, spFlags: DISPFlagDefinition, unit: !384, retainedNodes: !404)
// CHECK:STDOUT: !402 = !DILocalVariable(arg: 1, scope: !401, type: !20)
// CHECK:STDOUT: !403 = !DILocation(line: 15, column: 3, scope: !401)
// CHECK:STDOUT: !404 = !{!402}
// CHECK:STDOUT: !405 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.5b23b4f14feb87e1:core.Destroy.Core", scope: null, file: !383, line: 15, type: !22, spFlags: DISPFlagDefinition, unit: !384, retainedNodes: !408)
// CHECK:STDOUT: !406 = !DILocalVariable(arg: 1, scope: !405, type: !20)
// CHECK:STDOUT: !407 = !DILocation(line: 15, column: 3, scope: !405)
// CHECK:STDOUT: !408 = !{!406}
// CHECK:STDOUT: !409 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.2f190654c863951f:core.Destroy.Core", scope: null, file: !383, line: 15, type: !22, spFlags: DISPFlagDefinition, unit: !384, retainedNodes: !412)
// CHECK:STDOUT: !410 = !DILocalVariable(arg: 1, scope: !409, type: !20)
// CHECK:STDOUT: !411 = !DILocation(line: 15, column: 3, scope: !409)
// CHECK:STDOUT: !412 = !{!410}
// CHECK:STDOUT: !413 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.2f190654c863951f:core.Destroy.Core", scope: null, file: !383, line: 15, type: !22, spFlags: DISPFlagDefinition, unit: !384, retainedNodes: !416)
// CHECK:STDOUT: !414 = !DILocalVariable(arg: 1, scope: !413, type: !20)
// CHECK:STDOUT: !415 = !DILocation(line: 15, column: 3, scope: !413)
// CHECK:STDOUT: !416 = !{!414}
// CHECK:STDOUT: !417 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.b25525d21a9bd472:core.Destroy.Core", scope: null, file: !383, line: 15, type: !22, spFlags: DISPFlagDefinition, unit: !384, retainedNodes: !420)
// CHECK:STDOUT: !418 = !DILocalVariable(arg: 1, scope: !417, type: !20)
// CHECK:STDOUT: !419 = !DILocation(line: 15, column: 3, scope: !417)
// CHECK:STDOUT: !420 = !{!418}
// CHECK:STDOUT: !421 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.b25525d21a9bd472:core.Destroy.Core", scope: null, file: !383, line: 15, type: !22, spFlags: DISPFlagDefinition, unit: !384, retainedNodes: !424)
// CHECK:STDOUT: !422 = !DILocalVariable(arg: 1, scope: !421, type: !20)
// CHECK:STDOUT: !423 = !DILocation(line: 15, column: 3, scope: !421)
// CHECK:STDOUT: !424 = !{!422}
// CHECK:STDOUT: !425 = distinct !DISubprogram(name: "F", linkageName: "_CF.Base.Main.6d6a4a2512463e59", scope: null, file: !383, line: 5, type: !22, spFlags: DISPFlagDefinition, unit: !384, retainedNodes: !429)
// CHECK:STDOUT: !426 = !DILocalVariable(arg: 1, scope: !425, type: !20)
// CHECK:STDOUT: !428 = !DILocation(line: 5, column: 3, scope: !425)
// CHECK:STDOUT: !429 = !{!426}
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
// CHECK:STDOUT: define void @_CF.Derived.Main(ptr sret({}) %return, ptr %self, ptr %n) #0 !dbg !436 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %return, ptr align 1 @From.val.loc18_14, i64 0, i1 false), !dbg !443
// CHECK:STDOUT:   ret void, !dbg !443
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_CF:thunk:Base.Main:Derived.Main"(ptr sret({}) %return, ptr %self, ptr %n) #1 !dbg !445 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc17_52.2.temp = alloca {}, align 1, !dbg !449
// CHECK:STDOUT:   %.5.temp = alloca {}, align 1, !dbg !448
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc17_52.2.temp), !dbg !449
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.5.temp), !dbg !448
// CHECK:STDOUT:   call void @"_CConvert.From.Main:ImplicitAs.a3bb440f6426a7d6.Core"(ptr %.5.temp, ptr %n), !dbg !448
// CHECK:STDOUT:   call void @_CF.Derived.Main(ptr %.loc17_52.2.temp, ptr %self, ptr %.5.temp), !dbg !449
// CHECK:STDOUT:   call void @"_CConvert.From.Main:ImplicitAs.a3bb440f6426a7d6.Core"(ptr %return, ptr %.loc17_52.2.temp), !dbg !449
// CHECK:STDOUT:   call void @"_CSelfDestruct.c2a743abd057cb33:core.Destroy.Core"(ptr %.loc17_52.2.temp), !dbg !449
// CHECK:STDOUT:   call void @"_CSelfDestruct.7ef2e8c2f12d163a:core.Destroy.Core"(ptr %.5.temp), !dbg !448
// CHECK:STDOUT:   ret void, !dbg !449
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.cb8adced19392b99:core.Destroy.Core"(ptr %self) #0 !dbg !451 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !453
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.c2a743abd057cb33:core.Destroy.Core"(ptr %self) #0 !dbg !455 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !457
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.c2a743abd057cb33:core.Destroy.Core"(ptr %self) #0 !dbg !459 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.c2a743abd057cb33:core.Destroy.Core"(ptr %self), !dbg !461
// CHECK:STDOUT:   ret void, !dbg !461
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.7ef2e8c2f12d163a:core.Destroy.Core"(ptr %self) #0 !dbg !463 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !465
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.7ef2e8c2f12d163a:core.Destroy.Core"(ptr %self) #0 !dbg !467 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.7ef2e8c2f12d163a:core.Destroy.Core"(ptr %self), !dbg !469
// CHECK:STDOUT:   ret void, !dbg !469
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CUse.Main() #0 !dbg !471 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %d.var = alloca { { ptr } }, align 8, !dbg !472
// CHECK:STDOUT:   %.loc25_10.1.temp = alloca {}, align 1, !dbg !480
// CHECK:STDOUT:   %.loc25_9.2.temp = alloca {}, align 1, !dbg !481
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %d.var), !dbg !472
// CHECK:STDOUT:   %.loc23_31.2.base = getelementptr inbounds nuw { { ptr } }, ptr %d.var, i32 0, i32 0, !dbg !473
// CHECK:STDOUT:   %.loc23_30.2.vptr = getelementptr inbounds nuw { ptr }, ptr %.loc23_31.2.base, i32 0, i32 0, !dbg !474
// CHECK:STDOUT:   %.loc23_31.6.base = getelementptr inbounds nuw { { ptr } }, ptr %d.var, i32 0, i32 0, !dbg !473
// CHECK:STDOUT:   %.loc23_31.7.vptr = getelementptr inbounds nuw { ptr }, ptr %.loc23_31.6.base, i32 0, i32 0, !dbg !473
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 %d.var, ptr align 8 @Derived.val.loc23_31.5, i64 8, i1 false), !dbg !473
// CHECK:STDOUT:   store ptr @"_CDerived.Main.$vtable", ptr %.loc23_31.7.vptr, align 8, !dbg !473
// CHECK:STDOUT:   %.loc24_18.2.base = getelementptr inbounds nuw { { ptr } }, ptr %d.var, i32 0, i32 0, !dbg !477
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc25_10.1.temp), !dbg !480
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc25_9.2.temp), !dbg !481
// CHECK:STDOUT:   %Base.F.call.vtable = load ptr, ptr %.loc25_10.1.temp, align 8, !dbg !480
// CHECK:STDOUT:   %Base.F.call = call ptr @llvm.load.relative.i32(ptr %Base.F.call.vtable, i32 0), !dbg !480
// CHECK:STDOUT:   call void %Base.F.call(ptr %.loc25_10.1.temp, ptr %.loc24_18.2.base, ptr @From.val.loc18_14), !dbg !480
// CHECK:STDOUT:   call void @"_CSelfDestruct.7ef2e8c2f12d163a:core.Destroy.Core"(ptr %.loc25_10.1.temp), !dbg !480
// CHECK:STDOUT:   call void @"_CSelfDestruct.c2a743abd057cb33:core.Destroy.Core"(ptr @From.val), !dbg !481
// CHECK:STDOUT:   call void @"_CSelfDestruct.b25525d21a9bd472:core.Destroy.Core"(ptr %d.var), !dbg !472
// CHECK:STDOUT:   ret void, !dbg !482
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.454fecbcedcd86e1:core.Destroy.Core"(ptr %self) #0 !dbg !483 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !485
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.95528a86e1127e2d:core.Destroy.Core"(ptr %self) #0 !dbg !487 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc23.vptr = getelementptr inbounds nuw { ptr }, ptr %self, i32 0, i32 0, !dbg !489
// CHECK:STDOUT:   call void @"_CSelfDestruct.454fecbcedcd86e1:core.Destroy.Core"(ptr %.loc23.vptr), !dbg !489
// CHECK:STDOUT:   ret void, !dbg !489
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.95528a86e1127e2d:core.Destroy.Core"(ptr %self) #0 !dbg !491 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.95528a86e1127e2d:core.Destroy.Core"(ptr %self), !dbg !493
// CHECK:STDOUT:   ret void, !dbg !493
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.294df737cd074d60:core.Destroy.Core"(ptr %self) #0 !dbg !495 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !497
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.294df737cd074d60:core.Destroy.Core"(ptr %self) #0 !dbg !499 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.294df737cd074d60:core.Destroy.Core"(ptr %self), !dbg !501
// CHECK:STDOUT:   ret void, !dbg !501
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.caed8a96bf328d09:core.Destroy.Core"(ptr %self) #0 !dbg !503 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc23.base = getelementptr inbounds nuw { { ptr } }, ptr %self, i32 0, i32 0, !dbg !505
// CHECK:STDOUT:   call void @"_CSelfDestruct.294df737cd074d60:core.Destroy.Core"(ptr %.loc23.base), !dbg !505
// CHECK:STDOUT:   ret void, !dbg !505
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.caed8a96bf328d09:core.Destroy.Core"(ptr %self) #0 !dbg !507 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.caed8a96bf328d09:core.Destroy.Core"(ptr %self), !dbg !509
// CHECK:STDOUT:   ret void, !dbg !509
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.b25525d21a9bd472:core.Destroy.Core"(ptr %self) #0 !dbg !511 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !513
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.b25525d21a9bd472:core.Destroy.Core"(ptr %self) #0 !dbg !515 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.b25525d21a9bd472:core.Destroy.Core"(ptr %self), !dbg !517
// CHECK:STDOUT:   ret void, !dbg !517
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!431}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !21 = !{null, !20}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !40 = !{null}
// CHECK:STDOUT: !41 = !DISubroutineType(types: !40)
// CHECK:STDOUT: !430 = !DIFile(filename: "thunk.carbon", directory: "")
// CHECK:STDOUT: !431 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !430, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !434 = !{!20, !20, !20}
// CHECK:STDOUT: !435 = !DISubroutineType(types: !434)
// CHECK:STDOUT: !436 = distinct !DISubprogram(name: "F", linkageName: "_CF.Derived.Main", scope: null, file: !430, line: 17, type: !435, spFlags: DISPFlagDefinition, unit: !431, retainedNodes: !444)
// CHECK:STDOUT: !437 = !DILocalVariable(arg: 1, scope: !436, type: !20)
// CHECK:STDOUT: !438 = !DILocalVariable(arg: 2, scope: !436, type: !20)
// CHECK:STDOUT: !443 = !DILocation(line: 18, column: 5, scope: !436)
// CHECK:STDOUT: !444 = !{!437, !438}
// CHECK:STDOUT: !445 = distinct !DISubprogram(name: "F", linkageName: "_CF:thunk:Base.Main:Derived.Main", scope: null, file: !430, line: 17, type: !435, spFlags: DISPFlagDefinition, unit: !431, retainedNodes: !450)
// CHECK:STDOUT: !446 = !DILocalVariable(arg: 1, scope: !445, type: !20)
// CHECK:STDOUT: !447 = !DILocalVariable(arg: 2, scope: !445, type: !20)
// CHECK:STDOUT: !448 = !DILocation(line: 0, scope: !445)
// CHECK:STDOUT: !449 = !DILocation(line: 17, column: 3, scope: !445)
// CHECK:STDOUT: !450 = !{!446, !447}
// CHECK:STDOUT: !451 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.cb8adced19392b99:core.Destroy.Core", scope: null, file: !430, line: 17, type: !22, spFlags: DISPFlagDefinition, unit: !431, retainedNodes: !454)
// CHECK:STDOUT: !452 = !DILocalVariable(arg: 1, scope: !451, type: !20)
// CHECK:STDOUT: !453 = !DILocation(line: 17, column: 3, scope: !451)
// CHECK:STDOUT: !454 = !{!452}
// CHECK:STDOUT: !455 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.c2a743abd057cb33:core.Destroy.Core", scope: null, file: !430, line: 17, type: !22, spFlags: DISPFlagDefinition, unit: !431, retainedNodes: !458)
// CHECK:STDOUT: !456 = !DILocalVariable(arg: 1, scope: !455, type: !20)
// CHECK:STDOUT: !457 = !DILocation(line: 17, column: 3, scope: !455)
// CHECK:STDOUT: !458 = !{!456}
// CHECK:STDOUT: !459 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.c2a743abd057cb33:core.Destroy.Core", scope: null, file: !430, line: 17, type: !22, spFlags: DISPFlagDefinition, unit: !431, retainedNodes: !462)
// CHECK:STDOUT: !460 = !DILocalVariable(arg: 1, scope: !459, type: !20)
// CHECK:STDOUT: !461 = !DILocation(line: 17, column: 3, scope: !459)
// CHECK:STDOUT: !462 = !{!460}
// CHECK:STDOUT: !463 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.7ef2e8c2f12d163a:core.Destroy.Core", scope: null, file: !430, type: !22, spFlags: DISPFlagDefinition, unit: !431, retainedNodes: !466)
// CHECK:STDOUT: !464 = !DILocalVariable(arg: 1, scope: !463, type: !20)
// CHECK:STDOUT: !465 = !DILocation(line: 0, scope: !463)
// CHECK:STDOUT: !466 = !{!464}
// CHECK:STDOUT: !467 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.7ef2e8c2f12d163a:core.Destroy.Core", scope: null, file: !430, type: !22, spFlags: DISPFlagDefinition, unit: !431, retainedNodes: !470)
// CHECK:STDOUT: !468 = !DILocalVariable(arg: 1, scope: !467, type: !20)
// CHECK:STDOUT: !469 = !DILocation(line: 0, scope: !467)
// CHECK:STDOUT: !470 = !{!468}
// CHECK:STDOUT: !471 = distinct !DISubprogram(name: "Use", linkageName: "_CUse.Main", scope: null, file: !430, line: 22, type: !41, spFlags: DISPFlagDefinition, unit: !431)
// CHECK:STDOUT: !472 = !DILocation(line: 23, column: 3, scope: !471)
// CHECK:STDOUT: !473 = !DILocation(line: 23, column: 20, scope: !471)
// CHECK:STDOUT: !474 = !DILocation(line: 23, column: 29, scope: !471)
// CHECK:STDOUT: !477 = !DILocation(line: 24, column: 18, scope: !471)
// CHECK:STDOUT: !480 = !DILocation(line: 25, column: 3, scope: !471)
// CHECK:STDOUT: !481 = !DILocation(line: 25, column: 8, scope: !471)
// CHECK:STDOUT: !482 = !DILocation(line: 22, column: 1, scope: !471)
// CHECK:STDOUT: !483 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.454fecbcedcd86e1:core.Destroy.Core", scope: null, file: !430, line: 23, type: !22, spFlags: DISPFlagDefinition, unit: !431, retainedNodes: !486)
// CHECK:STDOUT: !484 = !DILocalVariable(arg: 1, scope: !483, type: !20)
// CHECK:STDOUT: !485 = !DILocation(line: 23, column: 3, scope: !483)
// CHECK:STDOUT: !486 = !{!484}
// CHECK:STDOUT: !487 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.95528a86e1127e2d:core.Destroy.Core", scope: null, file: !430, line: 23, type: !22, spFlags: DISPFlagDefinition, unit: !431, retainedNodes: !490)
// CHECK:STDOUT: !488 = !DILocalVariable(arg: 1, scope: !487, type: !20)
// CHECK:STDOUT: !489 = !DILocation(line: 23, column: 3, scope: !487)
// CHECK:STDOUT: !490 = !{!488}
// CHECK:STDOUT: !491 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.95528a86e1127e2d:core.Destroy.Core", scope: null, file: !430, line: 23, type: !22, spFlags: DISPFlagDefinition, unit: !431, retainedNodes: !494)
// CHECK:STDOUT: !492 = !DILocalVariable(arg: 1, scope: !491, type: !20)
// CHECK:STDOUT: !493 = !DILocation(line: 23, column: 3, scope: !491)
// CHECK:STDOUT: !494 = !{!492}
// CHECK:STDOUT: !495 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.294df737cd074d60:core.Destroy.Core", scope: null, file: !430, line: 23, type: !22, spFlags: DISPFlagDefinition, unit: !431, retainedNodes: !498)
// CHECK:STDOUT: !496 = !DILocalVariable(arg: 1, scope: !495, type: !20)
// CHECK:STDOUT: !497 = !DILocation(line: 23, column: 3, scope: !495)
// CHECK:STDOUT: !498 = !{!496}
// CHECK:STDOUT: !499 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.294df737cd074d60:core.Destroy.Core", scope: null, file: !430, line: 23, type: !22, spFlags: DISPFlagDefinition, unit: !431, retainedNodes: !502)
// CHECK:STDOUT: !500 = !DILocalVariable(arg: 1, scope: !499, type: !20)
// CHECK:STDOUT: !501 = !DILocation(line: 23, column: 3, scope: !499)
// CHECK:STDOUT: !502 = !{!500}
// CHECK:STDOUT: !503 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.caed8a96bf328d09:core.Destroy.Core", scope: null, file: !430, line: 23, type: !22, spFlags: DISPFlagDefinition, unit: !431, retainedNodes: !506)
// CHECK:STDOUT: !504 = !DILocalVariable(arg: 1, scope: !503, type: !20)
// CHECK:STDOUT: !505 = !DILocation(line: 23, column: 3, scope: !503)
// CHECK:STDOUT: !506 = !{!504}
// CHECK:STDOUT: !507 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.caed8a96bf328d09:core.Destroy.Core", scope: null, file: !430, line: 23, type: !22, spFlags: DISPFlagDefinition, unit: !431, retainedNodes: !510)
// CHECK:STDOUT: !508 = !DILocalVariable(arg: 1, scope: !507, type: !20)
// CHECK:STDOUT: !509 = !DILocation(line: 23, column: 3, scope: !507)
// CHECK:STDOUT: !510 = !{!508}
// CHECK:STDOUT: !511 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.b25525d21a9bd472:core.Destroy.Core", scope: null, file: !430, line: 23, type: !22, spFlags: DISPFlagDefinition, unit: !431, retainedNodes: !514)
// CHECK:STDOUT: !512 = !DILocalVariable(arg: 1, scope: !511, type: !20)
// CHECK:STDOUT: !513 = !DILocation(line: 23, column: 3, scope: !511)
// CHECK:STDOUT: !514 = !{!512}
// CHECK:STDOUT: !515 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.b25525d21a9bd472:core.Destroy.Core", scope: null, file: !430, line: 23, type: !22, spFlags: DISPFlagDefinition, unit: !431, retainedNodes: !518)
// CHECK:STDOUT: !516 = !DILocalVariable(arg: 1, scope: !515, type: !20)
// CHECK:STDOUT: !517 = !DILocation(line: 23, column: 3, scope: !515)
// CHECK:STDOUT: !518 = !{!516}
// CHECK:STDOUT:
