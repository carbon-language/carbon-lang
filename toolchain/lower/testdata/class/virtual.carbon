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
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.b25b39addefdf798(ptr %_.var.loc12), !dbg !52
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.b25b39addefdf798(ptr %_.var.loc9), !dbg !48
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.a22a3bceb8a14791(ptr %_.var.loc8), !dbg !45
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.094fc998e8aeb2cd(ptr %_.var.loc7), !dbg !43
// CHECK:STDOUT:   ret void, !dbg !54
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CFn.Intermediate.Classes(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_CFn:thunk:Intermediate.Classes:Derived.Classes"(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.bf085e1f873e33df:core.Destroy.Core"(ptr %self) #0 !dbg !55 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !57
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.f7b08e0f33b30342:core.Destroy.Core"(ptr %self) #0 !dbg !59 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !61
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.4487e305711510e9:core.Destroy.Core"(ptr %self) #0 !dbg !63 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !65
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.3c3534e993db7939:core.Destroy.Core"(ptr %self) #0 !dbg !67 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !69
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.43ed5d0e08fc7556:core.Destroy.Core"(ptr %self) #0 !dbg !71 {
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
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.b25b39addefdf798(ptr %self) #0 !dbg !81 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.43ed5d0e08fc7556:core.Destroy.Core"(ptr %self), !dbg !85
// CHECK:STDOUT:   ret void, !dbg !86
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.a22a3bceb8a14791(ptr %self) #0 !dbg !88 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.4487e305711510e9:core.Destroy.Core"(ptr %self), !dbg !92
// CHECK:STDOUT:   ret void, !dbg !93
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.094fc998e8aeb2cd(ptr %self) #0 !dbg !95 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.bf085e1f873e33df:core.Destroy.Core"(ptr %self), !dbg !99
// CHECK:STDOUT:   ret void, !dbg !100
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: read)
// CHECK:STDOUT: declare ptr @llvm.load.relative.i32(ptr, i32) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Destroy.Core.b25b39addefdf798, { 1, 0 }
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
// CHECK:STDOUT: !14 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
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
// CHECK:STDOUT: !55 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.bf085e1f873e33df:core.Destroy.Core", scope: null, file: !38, line: 12, type: !22, spFlags: DISPFlagDefinition, unit: !39, retainedNodes: !58)
// CHECK:STDOUT: !56 = !DILocalVariable(arg: 1, scope: !55, type: !20)
// CHECK:STDOUT: !57 = !DILocation(line: 12, column: 3, scope: !55)
// CHECK:STDOUT: !58 = !{!56}
// CHECK:STDOUT: !59 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.f7b08e0f33b30342:core.Destroy.Core", scope: null, file: !38, line: 12, type: !22, spFlags: DISPFlagDefinition, unit: !39, retainedNodes: !62)
// CHECK:STDOUT: !60 = !DILocalVariable(arg: 1, scope: !59, type: !20)
// CHECK:STDOUT: !61 = !DILocation(line: 12, column: 3, scope: !59)
// CHECK:STDOUT: !62 = !{!60}
// CHECK:STDOUT: !63 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.4487e305711510e9:core.Destroy.Core", scope: null, file: !38, line: 12, type: !22, spFlags: DISPFlagDefinition, unit: !39, retainedNodes: !66)
// CHECK:STDOUT: !64 = !DILocalVariable(arg: 1, scope: !63, type: !20)
// CHECK:STDOUT: !65 = !DILocation(line: 12, column: 3, scope: !63)
// CHECK:STDOUT: !66 = !{!64}
// CHECK:STDOUT: !67 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.3c3534e993db7939:core.Destroy.Core", scope: null, file: !38, line: 12, type: !22, spFlags: DISPFlagDefinition, unit: !39, retainedNodes: !70)
// CHECK:STDOUT: !68 = !DILocalVariable(arg: 1, scope: !67, type: !20)
// CHECK:STDOUT: !69 = !DILocation(line: 12, column: 3, scope: !67)
// CHECK:STDOUT: !70 = !{!68}
// CHECK:STDOUT: !71 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.43ed5d0e08fc7556:core.Destroy.Core", scope: null, file: !38, line: 12, type: !22, spFlags: DISPFlagDefinition, unit: !39, retainedNodes: !74)
// CHECK:STDOUT: !72 = !DILocalVariable(arg: 1, scope: !71, type: !20)
// CHECK:STDOUT: !73 = !DILocation(line: 12, column: 3, scope: !71)
// CHECK:STDOUT: !74 = !{!72}
// CHECK:STDOUT: !75 = distinct !DISubprogram(name: "Use", linkageName: "_CUse.Create", scope: null, file: !38, line: 15, type: !22, spFlags: DISPFlagDefinition, unit: !39, retainedNodes: !80)
// CHECK:STDOUT: !76 = !DILocalVariable(arg: 1, scope: !75, type: !20)
// CHECK:STDOUT: !78 = !DILocation(line: 16, column: 3, scope: !75)
// CHECK:STDOUT: !79 = !DILocation(line: 15, column: 1, scope: !75)
// CHECK:STDOUT: !80 = !{!76}
// CHECK:STDOUT: !81 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.b25b39addefdf798", scope: null, file: !14, line: 24, type: !22, spFlags: DISPFlagDefinition, unit: !39, retainedNodes: !87)
// CHECK:STDOUT: !82 = !DILocalVariable(arg: 1, scope: !81, type: !20)
// CHECK:STDOUT: !85 = !DILocation(line: 26, column: 5, scope: !81)
// CHECK:STDOUT: !86 = !DILocation(line: 24, column: 3, scope: !81)
// CHECK:STDOUT: !87 = !{!82}
// CHECK:STDOUT: !88 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.a22a3bceb8a14791", scope: null, file: !14, line: 24, type: !22, spFlags: DISPFlagDefinition, unit: !39, retainedNodes: !94)
// CHECK:STDOUT: !89 = !DILocalVariable(arg: 1, scope: !88, type: !20)
// CHECK:STDOUT: !92 = !DILocation(line: 26, column: 5, scope: !88)
// CHECK:STDOUT: !93 = !DILocation(line: 24, column: 3, scope: !88)
// CHECK:STDOUT: !94 = !{!89}
// CHECK:STDOUT: !95 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.094fc998e8aeb2cd", scope: null, file: !14, line: 24, type: !22, spFlags: DISPFlagDefinition, unit: !39, retainedNodes: !101)
// CHECK:STDOUT: !96 = !DILocalVariable(arg: 1, scope: !95, type: !20)
// CHECK:STDOUT: !99 = !DILocation(line: 26, column: 5, scope: !95)
// CHECK:STDOUT: !100 = !DILocation(line: 24, column: 3, scope: !95)
// CHECK:STDOUT: !101 = !{!96}
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
// CHECK:STDOUT: define void @_CMakePartialBase.Main(ptr sret({}) %return) #0 !dbg !106 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %return, ptr align 1 @empty_struct.ac8.loc7_12, i64 0, i1 false), !dbg !109
// CHECK:STDOUT:   ret void, !dbg !109
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMakePartialIntermediate.Main(ptr sret({ ptr, {} }) %return) #0 !dbg !110 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc11_36.1.base = getelementptr inbounds nuw { ptr, {} }, ptr %return, i32 0, i32 1, !dbg !113
// CHECK:STDOUT:   call void @_CMakePartialBase.Main(ptr %.loc11_36.1.base), !dbg !114
// CHECK:STDOUT:   %.loc11_36.3.vptr = getelementptr inbounds nuw { ptr, {} }, ptr %return, i32 0, i32 0, !dbg !113
// CHECK:STDOUT:   store ptr poison, ptr %.loc11_36.3.vptr, align 8, !dbg !113
// CHECK:STDOUT:   ret void, !dbg !115
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMakeBase.Main(ptr sret({}) %return) #0 !dbg !116 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CMakePartialBase.Main(ptr %return), !dbg !119
// CHECK:STDOUT:   ret void, !dbg !120
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMakeIntermediate.Main(ptr sret({ ptr, {} }) %return) #0 !dbg !121 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CMakePartialIntermediate.Main(ptr %return), !dbg !124
// CHECK:STDOUT:   %.loc19_34.1.vptr = getelementptr inbounds nuw { ptr, {} }, ptr %return, i32 0, i32 0, !dbg !124
// CHECK:STDOUT:   store ptr @"_CIntermediate.Classes.$vtable", ptr %.loc19_34.1.vptr, align 8, !dbg !124
// CHECK:STDOUT:   ret void, !dbg !125
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CFn.Intermediate.Classes(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMakeDerivedFromPartial.Main(ptr sret({ { ptr, {} } }) %return) #0 !dbg !126 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc23_44.1.base = getelementptr inbounds nuw { { ptr, {} } }, ptr %return, i32 0, i32 0, !dbg !129
// CHECK:STDOUT:   call void @_CMakePartialIntermediate.Main(ptr %.loc23_44.1.base), !dbg !130
// CHECK:STDOUT:   %.loc23_44.5.base = getelementptr inbounds nuw { { ptr, {} } }, ptr %return, i32 0, i32 0, !dbg !129
// CHECK:STDOUT:   %.loc23_44.6.vptr = getelementptr inbounds nuw { ptr, {} }, ptr %.loc23_44.5.base, i32 0, i32 0, !dbg !129
// CHECK:STDOUT:   store ptr @"_CDerived.Classes.$vtable", ptr %.loc23_44.6.vptr, align 8, !dbg !129
// CHECK:STDOUT:   ret void, !dbg !131
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_CFn:thunk:Intermediate.Classes:Derived.Classes"(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMakeDerivedFromNonPartial.Main(ptr sret({ { ptr, {} } }) %return) #0 !dbg !132 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc27_37.1.base = getelementptr inbounds nuw { { ptr, {} } }, ptr %return, i32 0, i32 0, !dbg !135
// CHECK:STDOUT:   call void @_CMakeIntermediate.Main(ptr %.loc27_37.1.base), !dbg !136
// CHECK:STDOUT:   %.loc27_37.7.base = getelementptr inbounds nuw { { ptr, {} } }, ptr %return, i32 0, i32 0, !dbg !135
// CHECK:STDOUT:   %.loc27_37.8.vptr = getelementptr inbounds nuw { ptr, {} }, ptr %.loc27_37.7.base, i32 0, i32 0, !dbg !135
// CHECK:STDOUT:   store ptr @"_CDerived.Classes.$vtable", ptr %.loc27_37.8.vptr, align 8, !dbg !135
// CHECK:STDOUT:   ret void, !dbg !137
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!103}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !102 = !DIFile(filename: "create_named_ctor.carbon", directory: "")
// CHECK:STDOUT: !103 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !102, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !104 = !{!20}
// CHECK:STDOUT: !105 = !DISubroutineType(types: !104)
// CHECK:STDOUT: !106 = distinct !DISubprogram(name: "MakePartialBase", linkageName: "_CMakePartialBase.Main", scope: null, file: !102, line: 6, type: !105, spFlags: DISPFlagDefinition, unit: !103)
// CHECK:STDOUT: !109 = !DILocation(line: 7, column: 3, scope: !106)
// CHECK:STDOUT: !110 = distinct !DISubprogram(name: "MakePartialIntermediate", linkageName: "_CMakePartialIntermediate.Main", scope: null, file: !102, line: 10, type: !105, spFlags: DISPFlagDefinition, unit: !103)
// CHECK:STDOUT: !113 = !DILocation(line: 11, column: 10, scope: !110)
// CHECK:STDOUT: !114 = !DILocation(line: 11, column: 19, scope: !110)
// CHECK:STDOUT: !115 = !DILocation(line: 11, column: 3, scope: !110)
// CHECK:STDOUT: !116 = distinct !DISubprogram(name: "MakeBase", linkageName: "_CMakeBase.Main", scope: null, file: !102, line: 14, type: !105, spFlags: DISPFlagDefinition, unit: !103)
// CHECK:STDOUT: !119 = !DILocation(line: 15, column: 10, scope: !116)
// CHECK:STDOUT: !120 = !DILocation(line: 15, column: 3, scope: !116)
// CHECK:STDOUT: !121 = distinct !DISubprogram(name: "MakeIntermediate", linkageName: "_CMakeIntermediate.Main", scope: null, file: !102, line: 18, type: !105, spFlags: DISPFlagDefinition, unit: !103)
// CHECK:STDOUT: !124 = !DILocation(line: 19, column: 10, scope: !121)
// CHECK:STDOUT: !125 = !DILocation(line: 19, column: 3, scope: !121)
// CHECK:STDOUT: !126 = distinct !DISubprogram(name: "MakeDerivedFromPartial", linkageName: "_CMakeDerivedFromPartial.Main", scope: null, file: !102, line: 22, type: !105, spFlags: DISPFlagDefinition, unit: !103)
// CHECK:STDOUT: !129 = !DILocation(line: 23, column: 10, scope: !126)
// CHECK:STDOUT: !130 = !DILocation(line: 23, column: 19, scope: !126)
// CHECK:STDOUT: !131 = !DILocation(line: 23, column: 3, scope: !126)
// CHECK:STDOUT: !132 = distinct !DISubprogram(name: "MakeDerivedFromNonPartial", linkageName: "_CMakeDerivedFromNonPartial.Main", scope: null, file: !102, line: 26, type: !105, spFlags: DISPFlagDefinition, unit: !103)
// CHECK:STDOUT: !135 = !DILocation(line: 27, column: 10, scope: !132)
// CHECK:STDOUT: !136 = !DILocation(line: 27, column: 19, scope: !132)
// CHECK:STDOUT: !137 = !DILocation(line: 27, column: 3, scope: !132)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'member_init.carbon'
// CHECK:STDOUT: source_filename = "member_init.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @"_CBase.MemberInit.$vtable" = unnamed_addr constant [1 x i32] [i32 trunc (i64 sub (i64 ptrtoint (ptr @_CFn.Base.MemberInit to i64), i64 ptrtoint (ptr @"_CBase.MemberInit.$vtable" to i64)) to i32)]
// CHECK:STDOUT: @Base.val.loc13_3.1 = internal constant <{ ptr, i32 }> <{ ptr @"_CBase.MemberInit.$vtable", i32 3 }>
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CFn.Base.MemberInit(ptr %self) #0 !dbg !143 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !146
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CFn.MemberInit() #0 !dbg !148 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %i.var = alloca i32, align 4, !dbg !149
// CHECK:STDOUT:   %v.var = alloca <{ ptr, i32 }>, align 8, !dbg !151
// CHECK:STDOUT:   %_.var = alloca <{ ptr, i32 }>, align 8, !dbg !156
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %i.var), !dbg !149
// CHECK:STDOUT:   store i32 3, ptr %i.var, align 4, !dbg !149
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %v.var), !dbg !151
// CHECK:STDOUT:   %.loc11_24.2.vptr = getelementptr inbounds nuw <{ ptr, i32 }>, ptr %v.var, i32 0, i32 0, !dbg !153
// CHECK:STDOUT:   %.loc11_23 = load i32, ptr %i.var, align 4, !dbg !152
// CHECK:STDOUT:   %.loc11_24.4.m = getelementptr inbounds nuw <{ ptr, i32 }>, ptr %v.var, i32 0, i32 1, !dbg !153
// CHECK:STDOUT:   store i32 %.loc11_23, ptr %.loc11_24.4.m, align 4, !dbg !153
// CHECK:STDOUT:   store ptr @"_CBase.MemberInit.$vtable", ptr %.loc11_24.2.vptr, align 8, !dbg !153
// CHECK:STDOUT:   %.loc12_4.m = getelementptr inbounds nuw <{ ptr, i32 }>, ptr %v.var, i32 0, i32 1, !dbg !155
// CHECK:STDOUT:   store i32 5, ptr %.loc12_4.m, align 4, !dbg !155
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !156
// CHECK:STDOUT:   %.loc13_24.2.vptr = getelementptr inbounds nuw <{ ptr, i32 }>, ptr %_.var, i32 0, i32 0, !dbg !157
// CHECK:STDOUT:   %.loc13_24.5.m = getelementptr inbounds nuw <{ ptr, i32 }>, ptr %_.var, i32 0, i32 1, !dbg !157
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 %_.var, ptr align 8 @Base.val.loc13_3.1, i64 12, i1 false), !dbg !156
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.318f60746ae7b99b(ptr %_.var), !dbg !156
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.318f60746ae7b99b(ptr %v.var), !dbg !151
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.6580e8d2c4d79a48(ptr %i.var), !dbg !149
// CHECK:STDOUT:   ret void, !dbg !159
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !160 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !162
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.fd9d4a98e93fc879:core.Destroy.Core"(ptr %self) #0 !dbg !164 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !166
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.c6dbe0017b4c8d16:core.Destroy.Core"(ptr %self) #0 !dbg !168 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !170
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.318f60746ae7b99b(ptr %self) #0 !dbg !172 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.c6dbe0017b4c8d16:core.Destroy.Core"(ptr %self), !dbg !176
// CHECK:STDOUT:   ret void, !dbg !177
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.6580e8d2c4d79a48(ptr %self) #0 !dbg !179 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self), !dbg !183
// CHECK:STDOUT:   ret void, !dbg !184
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Destroy.Core.318f60746ae7b99b, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!139}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !14 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !20 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !21 = !{null, !20}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !40 = !{null}
// CHECK:STDOUT: !41 = !DISubroutineType(types: !40)
// CHECK:STDOUT: !138 = !DIFile(filename: "member_init.carbon", directory: "")
// CHECK:STDOUT: !139 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !138, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !140 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !141 = !{null, !140}
// CHECK:STDOUT: !142 = !DISubroutineType(types: !141)
// CHECK:STDOUT: !143 = distinct !DISubprogram(name: "Fn", linkageName: "_CFn.Base.MemberInit", scope: null, file: !138, line: 6, type: !22, spFlags: DISPFlagDefinition, unit: !139, retainedNodes: !147)
// CHECK:STDOUT: !144 = !DILocalVariable(arg: 1, scope: !143, type: !20)
// CHECK:STDOUT: !146 = !DILocation(line: 6, column: 3, scope: !143)
// CHECK:STDOUT: !147 = !{!144}
// CHECK:STDOUT: !148 = distinct !DISubprogram(name: "Fn", linkageName: "_CFn.MemberInit", scope: null, file: !138, line: 9, type: !41, spFlags: DISPFlagDefinition, unit: !139)
// CHECK:STDOUT: !149 = !DILocation(line: 10, column: 3, scope: !148)
// CHECK:STDOUT: !151 = !DILocation(line: 11, column: 3, scope: !148)
// CHECK:STDOUT: !152 = !DILocation(line: 11, column: 23, scope: !148)
// CHECK:STDOUT: !153 = !DILocation(line: 11, column: 17, scope: !148)
// CHECK:STDOUT: !155 = !DILocation(line: 12, column: 3, scope: !148)
// CHECK:STDOUT: !156 = !DILocation(line: 13, column: 3, scope: !148)
// CHECK:STDOUT: !157 = !DILocation(line: 13, column: 17, scope: !148)
// CHECK:STDOUT: !159 = !DILocation(line: 9, column: 1, scope: !148)
// CHECK:STDOUT: !160 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core", scope: null, file: !138, line: 13, type: !142, spFlags: DISPFlagDefinition, unit: !139, retainedNodes: !163)
// CHECK:STDOUT: !161 = !DILocalVariable(arg: 1, scope: !160, type: !140)
// CHECK:STDOUT: !162 = !DILocation(line: 13, column: 3, scope: !160)
// CHECK:STDOUT: !163 = !{!161}
// CHECK:STDOUT: !164 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.fd9d4a98e93fc879:core.Destroy.Core", scope: null, file: !138, line: 13, type: !22, spFlags: DISPFlagDefinition, unit: !139, retainedNodes: !167)
// CHECK:STDOUT: !165 = !DILocalVariable(arg: 1, scope: !164, type: !20)
// CHECK:STDOUT: !166 = !DILocation(line: 13, column: 3, scope: !164)
// CHECK:STDOUT: !167 = !{!165}
// CHECK:STDOUT: !168 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.c6dbe0017b4c8d16:core.Destroy.Core", scope: null, file: !138, line: 13, type: !22, spFlags: DISPFlagDefinition, unit: !139, retainedNodes: !171)
// CHECK:STDOUT: !169 = !DILocalVariable(arg: 1, scope: !168, type: !20)
// CHECK:STDOUT: !170 = !DILocation(line: 13, column: 3, scope: !168)
// CHECK:STDOUT: !171 = !{!169}
// CHECK:STDOUT: !172 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.318f60746ae7b99b", scope: null, file: !14, line: 24, type: !22, spFlags: DISPFlagDefinition, unit: !139, retainedNodes: !178)
// CHECK:STDOUT: !173 = !DILocalVariable(arg: 1, scope: !172, type: !20)
// CHECK:STDOUT: !176 = !DILocation(line: 26, column: 5, scope: !172)
// CHECK:STDOUT: !177 = !DILocation(line: 24, column: 3, scope: !172)
// CHECK:STDOUT: !178 = !{!173}
// CHECK:STDOUT: !179 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.6580e8d2c4d79a48", scope: null, file: !14, line: 24, type: !142, spFlags: DISPFlagDefinition, unit: !139, retainedNodes: !185)
// CHECK:STDOUT: !180 = !DILocalVariable(arg: 1, scope: !179, type: !140)
// CHECK:STDOUT: !183 = !DILocation(line: 26, column: 5, scope: !179)
// CHECK:STDOUT: !184 = !DILocation(line: 24, column: 3, scope: !179)
// CHECK:STDOUT: !185 = !{!180}
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
// CHECK:STDOUT: define void @_CUse.Main() #0 !dbg !188 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca { { ptr } }, align 8, !dbg !189
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !189
// CHECK:STDOUT:   %.loc13_32.2.base = getelementptr inbounds nuw { { ptr } }, ptr %_.var, i32 0, i32 0, !dbg !190
// CHECK:STDOUT:   %.loc13_31.2.vptr = getelementptr inbounds nuw { ptr }, ptr %.loc13_32.2.base, i32 0, i32 0, !dbg !191
// CHECK:STDOUT:   %.loc13_32.6.base = getelementptr inbounds nuw { { ptr } }, ptr %_.var, i32 0, i32 0, !dbg !190
// CHECK:STDOUT:   %.loc13_32.7.vptr = getelementptr inbounds nuw { ptr }, ptr %.loc13_32.6.base, i32 0, i32 0, !dbg !190
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 %_.var, ptr align 8 @Derived.val.loc13_32.5, i64 8, i1 false), !dbg !190
// CHECK:STDOUT:   store ptr @"_CDerived.Main.$vtable", ptr %.loc13_32.7.vptr, align 8, !dbg !190
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.acf75410c7781b1f(ptr %_.var), !dbg !189
// CHECK:STDOUT:   ret void, !dbg !193
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.95528a86e1127e2d:core.Destroy.Core"(ptr %self) #0 !dbg !194 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !196
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.294df737cd074d60:core.Destroy.Core"(ptr %self) #0 !dbg !198 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !200
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.caed8a96bf328d09:core.Destroy.Core"(ptr %self) #0 !dbg !202 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !204
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.b25525d21a9bd472:core.Destroy.Core"(ptr %self) #0 !dbg !206 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !208
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.acf75410c7781b1f(ptr %self) #0 !dbg !210 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.b25525d21a9bd472:core.Destroy.Core"(ptr %self), !dbg !214
// CHECK:STDOUT:   ret void, !dbg !215
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!187}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !14 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !20 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !21 = !{null, !20}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !40 = !{null}
// CHECK:STDOUT: !41 = !DISubroutineType(types: !40)
// CHECK:STDOUT: !186 = !DIFile(filename: "member_brace_init.carbon", directory: "")
// CHECK:STDOUT: !187 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !186, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !188 = distinct !DISubprogram(name: "Use", linkageName: "_CUse.Main", scope: null, file: !186, line: 12, type: !41, spFlags: DISPFlagDefinition, unit: !187)
// CHECK:STDOUT: !189 = !DILocation(line: 13, column: 3, scope: !188)
// CHECK:STDOUT: !190 = !DILocation(line: 13, column: 21, scope: !188)
// CHECK:STDOUT: !191 = !DILocation(line: 13, column: 30, scope: !188)
// CHECK:STDOUT: !193 = !DILocation(line: 12, column: 1, scope: !188)
// CHECK:STDOUT: !194 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.95528a86e1127e2d:core.Destroy.Core", scope: null, file: !186, line: 13, type: !22, spFlags: DISPFlagDefinition, unit: !187, retainedNodes: !197)
// CHECK:STDOUT: !195 = !DILocalVariable(arg: 1, scope: !194, type: !20)
// CHECK:STDOUT: !196 = !DILocation(line: 13, column: 3, scope: !194)
// CHECK:STDOUT: !197 = !{!195}
// CHECK:STDOUT: !198 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.294df737cd074d60:core.Destroy.Core", scope: null, file: !186, line: 13, type: !22, spFlags: DISPFlagDefinition, unit: !187, retainedNodes: !201)
// CHECK:STDOUT: !199 = !DILocalVariable(arg: 1, scope: !198, type: !20)
// CHECK:STDOUT: !200 = !DILocation(line: 13, column: 3, scope: !198)
// CHECK:STDOUT: !201 = !{!199}
// CHECK:STDOUT: !202 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.caed8a96bf328d09:core.Destroy.Core", scope: null, file: !186, line: 13, type: !22, spFlags: DISPFlagDefinition, unit: !187, retainedNodes: !205)
// CHECK:STDOUT: !203 = !DILocalVariable(arg: 1, scope: !202, type: !20)
// CHECK:STDOUT: !204 = !DILocation(line: 13, column: 3, scope: !202)
// CHECK:STDOUT: !205 = !{!203}
// CHECK:STDOUT: !206 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.b25525d21a9bd472:core.Destroy.Core", scope: null, file: !186, line: 13, type: !22, spFlags: DISPFlagDefinition, unit: !187, retainedNodes: !209)
// CHECK:STDOUT: !207 = !DILocalVariable(arg: 1, scope: !206, type: !20)
// CHECK:STDOUT: !208 = !DILocation(line: 13, column: 3, scope: !206)
// CHECK:STDOUT: !209 = !{!207}
// CHECK:STDOUT: !210 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.acf75410c7781b1f", scope: null, file: !14, line: 24, type: !22, spFlags: DISPFlagDefinition, unit: !187, retainedNodes: !216)
// CHECK:STDOUT: !211 = !DILocalVariable(arg: 1, scope: !210, type: !20)
// CHECK:STDOUT: !214 = !DILocation(line: 26, column: 5, scope: !210)
// CHECK:STDOUT: !215 = !DILocation(line: 24, column: 3, scope: !210)
// CHECK:STDOUT: !216 = !{!211}
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
// CHECK:STDOUT: define void @_CUse.Main(ptr %b) #0 !dbg !219 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Base.F.call.vtable = load ptr, ptr %b, align 8, !dbg !222
// CHECK:STDOUT:   %Base.F.call = call ptr @llvm.load.relative.i32(ptr %Base.F.call.vtable, i32 0), !dbg !222
// CHECK:STDOUT:   call void %Base.F.call(ptr %b), !dbg !222
// CHECK:STDOUT:   ret void, !dbg !223
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: read)
// CHECK:STDOUT: declare ptr @llvm.load.relative.i32(ptr, i32) #1
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: read) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!218}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !21 = !{null, !20}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !217 = !DIFile(filename: "call.carbon", directory: "")
// CHECK:STDOUT: !218 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !217, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !219 = distinct !DISubprogram(name: "Use", linkageName: "_CUse.Main", scope: null, file: !217, line: 8, type: !22, spFlags: DISPFlagDefinition, unit: !218, retainedNodes: !224)
// CHECK:STDOUT: !220 = !DILocalVariable(arg: 1, scope: !219, type: !20)
// CHECK:STDOUT: !222 = !DILocation(line: 9, column: 3, scope: !219)
// CHECK:STDOUT: !223 = !DILocation(line: 8, column: 1, scope: !219)
// CHECK:STDOUT: !224 = !{!220}
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
// CHECK:STDOUT: define weak_odr void @"_CF:thunk:Base.Main:Derived.Main"(ptr %self) #0 !dbg !227 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Derived.Main(ptr %self), !dbg !230
// CHECK:STDOUT:   ret void, !dbg !230
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CUse.Main() #1 !dbg !232 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %v.var = alloca { { ptr } }, align 8, !dbg !233
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %v.var), !dbg !233
// CHECK:STDOUT:   %.loc14_32.2.base = getelementptr inbounds nuw { { ptr } }, ptr %v.var, i32 0, i32 0, !dbg !234
// CHECK:STDOUT:   %.loc14_31.2.vptr = getelementptr inbounds nuw { ptr }, ptr %.loc14_32.2.base, i32 0, i32 0, !dbg !235
// CHECK:STDOUT:   %.loc14_32.6.base = getelementptr inbounds nuw { { ptr } }, ptr %v.var, i32 0, i32 0, !dbg !234
// CHECK:STDOUT:   %.loc14_32.7.vptr = getelementptr inbounds nuw { ptr }, ptr %.loc14_32.6.base, i32 0, i32 0, !dbg !234
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 %v.var, ptr align 8 @Derived.val.loc14_32.5, i64 8, i1 false), !dbg !234
// CHECK:STDOUT:   store ptr @"_CDerived.Main.$vtable", ptr %.loc14_32.7.vptr, align 8, !dbg !234
// CHECK:STDOUT:   %.loc15_3.1.base = getelementptr inbounds nuw { { ptr } }, ptr %v.var, i32 0, i32 0, !dbg !237
// CHECK:STDOUT:   %Base.F.call.vtable = load ptr, ptr %.loc15_3.1.base, align 8, !dbg !237
// CHECK:STDOUT:   %Base.F.call = call ptr @llvm.load.relative.i32(ptr %Base.F.call.vtable, i32 0), !dbg !237
// CHECK:STDOUT:   call void %Base.F.call(ptr %.loc15_3.1.base), !dbg !237
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.acf75410c7781b1f(ptr %v.var), !dbg !233
// CHECK:STDOUT:   ret void, !dbg !238
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.95528a86e1127e2d:core.Destroy.Core"(ptr %self) #1 !dbg !239 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !241
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.294df737cd074d60:core.Destroy.Core"(ptr %self) #1 !dbg !243 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !245
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.caed8a96bf328d09:core.Destroy.Core"(ptr %self) #1 !dbg !247 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !249
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.b25525d21a9bd472:core.Destroy.Core"(ptr %self) #1 !dbg !251 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !253
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
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.acf75410c7781b1f(ptr %self) #1 !dbg !255 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.b25525d21a9bd472:core.Destroy.Core"(ptr %self), !dbg !259
// CHECK:STDOUT:   ret void, !dbg !260
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { alwaysinline nounwind }
// CHECK:STDOUT: attributes #1 = { nounwind }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #3 = { nocallback nofree nosync nounwind willreturn memory(argmem: read) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!226}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !14 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !20 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !21 = !{null, !20}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !40 = !{null}
// CHECK:STDOUT: !41 = !DISubroutineType(types: !40)
// CHECK:STDOUT: !225 = !DIFile(filename: "call_impl.carbon", directory: "")
// CHECK:STDOUT: !226 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !225, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !227 = distinct !DISubprogram(name: "F", linkageName: "_CF:thunk:Base.Main:Derived.Main", scope: null, file: !225, line: 10, type: !22, spFlags: DISPFlagDefinition, unit: !226, retainedNodes: !231)
// CHECK:STDOUT: !228 = !DILocalVariable(arg: 1, scope: !227, type: !20)
// CHECK:STDOUT: !230 = !DILocation(line: 10, column: 3, scope: !227)
// CHECK:STDOUT: !231 = !{!228}
// CHECK:STDOUT: !232 = distinct !DISubprogram(name: "Use", linkageName: "_CUse.Main", scope: null, file: !225, line: 13, type: !41, spFlags: DISPFlagDefinition, unit: !226)
// CHECK:STDOUT: !233 = !DILocation(line: 14, column: 3, scope: !232)
// CHECK:STDOUT: !234 = !DILocation(line: 14, column: 21, scope: !232)
// CHECK:STDOUT: !235 = !DILocation(line: 14, column: 30, scope: !232)
// CHECK:STDOUT: !237 = !DILocation(line: 15, column: 3, scope: !232)
// CHECK:STDOUT: !238 = !DILocation(line: 13, column: 1, scope: !232)
// CHECK:STDOUT: !239 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.95528a86e1127e2d:core.Destroy.Core", scope: null, file: !225, line: 14, type: !22, spFlags: DISPFlagDefinition, unit: !226, retainedNodes: !242)
// CHECK:STDOUT: !240 = !DILocalVariable(arg: 1, scope: !239, type: !20)
// CHECK:STDOUT: !241 = !DILocation(line: 14, column: 3, scope: !239)
// CHECK:STDOUT: !242 = !{!240}
// CHECK:STDOUT: !243 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.294df737cd074d60:core.Destroy.Core", scope: null, file: !225, line: 14, type: !22, spFlags: DISPFlagDefinition, unit: !226, retainedNodes: !246)
// CHECK:STDOUT: !244 = !DILocalVariable(arg: 1, scope: !243, type: !20)
// CHECK:STDOUT: !245 = !DILocation(line: 14, column: 3, scope: !243)
// CHECK:STDOUT: !246 = !{!244}
// CHECK:STDOUT: !247 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.caed8a96bf328d09:core.Destroy.Core", scope: null, file: !225, line: 14, type: !22, spFlags: DISPFlagDefinition, unit: !226, retainedNodes: !250)
// CHECK:STDOUT: !248 = !DILocalVariable(arg: 1, scope: !247, type: !20)
// CHECK:STDOUT: !249 = !DILocation(line: 14, column: 3, scope: !247)
// CHECK:STDOUT: !250 = !{!248}
// CHECK:STDOUT: !251 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.b25525d21a9bd472:core.Destroy.Core", scope: null, file: !225, line: 14, type: !22, spFlags: DISPFlagDefinition, unit: !226, retainedNodes: !254)
// CHECK:STDOUT: !252 = !DILocalVariable(arg: 1, scope: !251, type: !20)
// CHECK:STDOUT: !253 = !DILocation(line: 14, column: 3, scope: !251)
// CHECK:STDOUT: !254 = !{!252}
// CHECK:STDOUT: !255 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.acf75410c7781b1f", scope: null, file: !14, line: 24, type: !22, spFlags: DISPFlagDefinition, unit: !226, retainedNodes: !261)
// CHECK:STDOUT: !256 = !DILocalVariable(arg: 1, scope: !255, type: !20)
// CHECK:STDOUT: !259 = !DILocation(line: 26, column: 5, scope: !255)
// CHECK:STDOUT: !260 = !DILocation(line: 24, column: 3, scope: !255)
// CHECK:STDOUT: !261 = !{!256}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'generic_noop.carbon'
// CHECK:STDOUT: source_filename = "generic_noop.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!263}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !262 = !DIFile(filename: "generic_noop.carbon", directory: "")
// CHECK:STDOUT: !263 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !262, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'generic_use.carbon'
// CHECK:STDOUT: source_filename = "generic_use.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @"_CBase.Main.$vtable.671bb364174176ed" = unnamed_addr constant [1 x i32] [i32 trunc (i64 sub (i64 ptrtoint (ptr @_CF.Base.Main.671bb364174176ed to i64), i64 ptrtoint (ptr @"_CBase.Main.$vtable.671bb364174176ed" to i64)) to i32)]
// CHECK:STDOUT: @"_CBase.Main.$vtable.3feab98ae4e0004e" = unnamed_addr constant [1 x i32] [i32 trunc (i64 sub (i64 ptrtoint (ptr @_CF.Base.Main.3feab98ae4e0004e to i64), i64 ptrtoint (ptr @"_CBase.Main.$vtable.3feab98ae4e0004e" to i64)) to i32)]
// CHECK:STDOUT: @Base.val.11f.loc16_3.1 = internal constant { ptr } { ptr @"_CBase.Main.$vtable.671bb364174176ed" }
// CHECK:STDOUT: @Base.val.bc5.loc17_3.1 = internal constant { ptr } { ptr @"_CBase.Main.$vtable.3feab98ae4e0004e" }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !266 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var.loc16 = alloca { ptr }, align 8, !dbg !267
// CHECK:STDOUT:   %_.var.loc17 = alloca { ptr }, align 8, !dbg !270
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc16), !dbg !267
// CHECK:STDOUT:   %.loc16_22.2.vptr = getelementptr inbounds nuw { ptr }, ptr %_.var.loc16, i32 0, i32 0, !dbg !268
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 %_.var.loc16, ptr align 8 @Base.val.11f.loc16_3.1, i64 8, i1 false), !dbg !267
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc17), !dbg !270
// CHECK:STDOUT:   %.loc17_22.2.vptr = getelementptr inbounds nuw { ptr }, ptr %_.var.loc17, i32 0, i32 0, !dbg !271
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 %_.var.loc17, ptr align 8 @Base.val.bc5.loc17_3.1, i64 8, i1 false), !dbg !270
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.0930418c28f36a71(ptr %_.var.loc17), !dbg !270
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.36ff02928fac3234(ptr %_.var.loc16), !dbg !267
// CHECK:STDOUT:   ret void, !dbg !273
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.13f86630f401913f:core.Destroy.Core"(ptr %self) #0 !dbg !274 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !276
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.f2f12ab22a0a6b87:core.Destroy.Core"(ptr %self) #0 !dbg !278 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !280
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.8578661a61c8bbbf:core.Destroy.Core"(ptr %self) #0 !dbg !282 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !284
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.95528a86e1127e2d:core.Destroy.Core"(ptr %self) #0 !dbg !286 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !288
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.87d9bc64fa8fdf45:core.Destroy.Core"(ptr %self) #0 !dbg !290 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !292
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.4c80fb5fab12acfa:core.Destroy.Core"(ptr %self) #0 !dbg !294 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !296
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Base.Main.671bb364174176ed(ptr %self) #0 !dbg !298 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca {}, align 1, !dbg !301
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !301
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.1fd451408814426d(ptr %_.var), !dbg !301
// CHECK:STDOUT:   ret void, !dbg !303
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Base.Main.3feab98ae4e0004e(ptr %self) #0 !dbg !305 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca { {} }, align 1, !dbg !308
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !308
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.cfcd411958ab8803(ptr %_.var), !dbg !308
// CHECK:STDOUT:   ret void, !dbg !310
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.0930418c28f36a71(ptr %self) #0 !dbg !312 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.87d9bc64fa8fdf45:core.Destroy.Core"(ptr %self), !dbg !316
// CHECK:STDOUT:   ret void, !dbg !317
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.36ff02928fac3234(ptr %self) #0 !dbg !319 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.4c80fb5fab12acfa:core.Destroy.Core"(ptr %self), !dbg !323
// CHECK:STDOUT:   ret void, !dbg !324
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.1fd451408814426d(ptr %self) #0 !dbg !326 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.13f86630f401913f:core.Destroy.Core"(ptr %self), !dbg !330
// CHECK:STDOUT:   ret void, !dbg !331
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.cfcd411958ab8803(ptr %self) #0 !dbg !333 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.8578661a61c8bbbf:core.Destroy.Core"(ptr %self), !dbg !337
// CHECK:STDOUT:   ret void, !dbg !338
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!265}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !14 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !20 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !21 = !{null, !20}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !40 = !{null}
// CHECK:STDOUT: !41 = !DISubroutineType(types: !40)
// CHECK:STDOUT: !264 = !DIFile(filename: "generic_use.carbon", directory: "")
// CHECK:STDOUT: !265 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !264, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !266 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !264, line: 15, type: !41, spFlags: DISPFlagDefinition, unit: !265)
// CHECK:STDOUT: !267 = !DILocation(line: 16, column: 3, scope: !266)
// CHECK:STDOUT: !268 = !DILocation(line: 16, column: 21, scope: !266)
// CHECK:STDOUT: !270 = !DILocation(line: 17, column: 3, scope: !266)
// CHECK:STDOUT: !271 = !DILocation(line: 17, column: 21, scope: !266)
// CHECK:STDOUT: !273 = !DILocation(line: 15, column: 1, scope: !266)
// CHECK:STDOUT: !274 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.13f86630f401913f:core.Destroy.Core", scope: null, file: !264, line: 16, type: !22, spFlags: DISPFlagDefinition, unit: !265, retainedNodes: !277)
// CHECK:STDOUT: !275 = !DILocalVariable(arg: 1, scope: !274, type: !20)
// CHECK:STDOUT: !276 = !DILocation(line: 16, column: 10, scope: !274)
// CHECK:STDOUT: !277 = !{!275}
// CHECK:STDOUT: !278 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.f2f12ab22a0a6b87:core.Destroy.Core", scope: null, file: !264, line: 17, type: !22, spFlags: DISPFlagDefinition, unit: !265, retainedNodes: !281)
// CHECK:STDOUT: !279 = !DILocalVariable(arg: 1, scope: !278, type: !20)
// CHECK:STDOUT: !280 = !DILocation(line: 17, column: 10, scope: !278)
// CHECK:STDOUT: !281 = !{!279}
// CHECK:STDOUT: !282 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.8578661a61c8bbbf:core.Destroy.Core", scope: null, file: !264, line: 17, type: !22, spFlags: DISPFlagDefinition, unit: !265, retainedNodes: !285)
// CHECK:STDOUT: !283 = !DILocalVariable(arg: 1, scope: !282, type: !20)
// CHECK:STDOUT: !284 = !DILocation(line: 17, column: 10, scope: !282)
// CHECK:STDOUT: !285 = !{!283}
// CHECK:STDOUT: !286 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.95528a86e1127e2d:core.Destroy.Core", scope: null, file: !264, line: 17, type: !22, spFlags: DISPFlagDefinition, unit: !265, retainedNodes: !289)
// CHECK:STDOUT: !287 = !DILocalVariable(arg: 1, scope: !286, type: !20)
// CHECK:STDOUT: !288 = !DILocation(line: 17, column: 3, scope: !286)
// CHECK:STDOUT: !289 = !{!287}
// CHECK:STDOUT: !290 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.87d9bc64fa8fdf45:core.Destroy.Core", scope: null, file: !264, line: 17, type: !22, spFlags: DISPFlagDefinition, unit: !265, retainedNodes: !293)
// CHECK:STDOUT: !291 = !DILocalVariable(arg: 1, scope: !290, type: !20)
// CHECK:STDOUT: !292 = !DILocation(line: 17, column: 3, scope: !290)
// CHECK:STDOUT: !293 = !{!291}
// CHECK:STDOUT: !294 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.4c80fb5fab12acfa:core.Destroy.Core", scope: null, file: !264, line: 16, type: !22, spFlags: DISPFlagDefinition, unit: !265, retainedNodes: !297)
// CHECK:STDOUT: !295 = !DILocalVariable(arg: 1, scope: !294, type: !20)
// CHECK:STDOUT: !296 = !DILocation(line: 16, column: 3, scope: !294)
// CHECK:STDOUT: !297 = !{!295}
// CHECK:STDOUT: !298 = distinct !DISubprogram(name: "F", linkageName: "_CF.Base.Main.671bb364174176ed", scope: null, file: !264, line: 5, type: !22, spFlags: DISPFlagDefinition, unit: !265, retainedNodes: !304)
// CHECK:STDOUT: !299 = !DILocalVariable(arg: 1, scope: !298, type: !20)
// CHECK:STDOUT: !301 = !DILocation(line: 6, column: 5, scope: !298)
// CHECK:STDOUT: !303 = !DILocation(line: 5, column: 3, scope: !298)
// CHECK:STDOUT: !304 = !{!299}
// CHECK:STDOUT: !305 = distinct !DISubprogram(name: "F", linkageName: "_CF.Base.Main.3feab98ae4e0004e", scope: null, file: !264, line: 5, type: !22, spFlags: DISPFlagDefinition, unit: !265, retainedNodes: !311)
// CHECK:STDOUT: !306 = !DILocalVariable(arg: 1, scope: !305, type: !20)
// CHECK:STDOUT: !308 = !DILocation(line: 6, column: 5, scope: !305)
// CHECK:STDOUT: !310 = !DILocation(line: 5, column: 3, scope: !305)
// CHECK:STDOUT: !311 = !{!306}
// CHECK:STDOUT: !312 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.0930418c28f36a71", scope: null, file: !14, line: 24, type: !22, spFlags: DISPFlagDefinition, unit: !265, retainedNodes: !318)
// CHECK:STDOUT: !313 = !DILocalVariable(arg: 1, scope: !312, type: !20)
// CHECK:STDOUT: !316 = !DILocation(line: 26, column: 5, scope: !312)
// CHECK:STDOUT: !317 = !DILocation(line: 24, column: 3, scope: !312)
// CHECK:STDOUT: !318 = !{!313}
// CHECK:STDOUT: !319 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.36ff02928fac3234", scope: null, file: !14, line: 24, type: !22, spFlags: DISPFlagDefinition, unit: !265, retainedNodes: !325)
// CHECK:STDOUT: !320 = !DILocalVariable(arg: 1, scope: !319, type: !20)
// CHECK:STDOUT: !323 = !DILocation(line: 26, column: 5, scope: !319)
// CHECK:STDOUT: !324 = !DILocation(line: 24, column: 3, scope: !319)
// CHECK:STDOUT: !325 = !{!320}
// CHECK:STDOUT: !326 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.1fd451408814426d", scope: null, file: !14, line: 24, type: !22, spFlags: DISPFlagDefinition, unit: !265, retainedNodes: !332)
// CHECK:STDOUT: !327 = !DILocalVariable(arg: 1, scope: !326, type: !20)
// CHECK:STDOUT: !330 = !DILocation(line: 26, column: 5, scope: !326)
// CHECK:STDOUT: !331 = !DILocation(line: 24, column: 3, scope: !326)
// CHECK:STDOUT: !332 = !{!327}
// CHECK:STDOUT: !333 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.cfcd411958ab8803", scope: null, file: !14, line: 24, type: !22, spFlags: DISPFlagDefinition, unit: !265, retainedNodes: !339)
// CHECK:STDOUT: !334 = !DILocalVariable(arg: 1, scope: !333, type: !20)
// CHECK:STDOUT: !337 = !DILocation(line: 26, column: 5, scope: !333)
// CHECK:STDOUT: !338 = !DILocation(line: 24, column: 3, scope: !333)
// CHECK:STDOUT: !339 = !{!334}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'generic_base.carbon'
// CHECK:STDOUT: source_filename = "generic_base.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @"_CDerived.Main.$vtable" = unnamed_addr constant [1 x i32] [i32 trunc (i64 sub (i64 ptrtoint (ptr @_CF.Base.Main.6d6a4a2512463e59 to i64), i64 ptrtoint (ptr @"_CDerived.Main.$vtable" to i64)) to i32)]
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMake.Main() #0 !dbg !342 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca { { ptr } }, align 8, !dbg !343
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !343
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.acf75410c7781b1f(ptr %_.var), !dbg !343
// CHECK:STDOUT:   ret void, !dbg !345
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.95528a86e1127e2d:core.Destroy.Core"(ptr %self) #0 !dbg !346 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !348
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.5b23b4f14feb87e1:core.Destroy.Core"(ptr %self) #0 !dbg !350 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !352
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.2f190654c863951f:core.Destroy.Core"(ptr %self) #0 !dbg !354 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !356
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.b25525d21a9bd472:core.Destroy.Core"(ptr %self) #0 !dbg !358 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !360
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Base.Main.6d6a4a2512463e59(ptr %self) #0 !dbg !362 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !365
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.acf75410c7781b1f(ptr %self) #0 !dbg !367 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.b25525d21a9bd472:core.Destroy.Core"(ptr %self), !dbg !371
// CHECK:STDOUT:   ret void, !dbg !372
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!341}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !14 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !20 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !21 = !{null, !20}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !40 = !{null}
// CHECK:STDOUT: !41 = !DISubroutineType(types: !40)
// CHECK:STDOUT: !340 = !DIFile(filename: "generic_base.carbon", directory: "")
// CHECK:STDOUT: !341 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !340, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !342 = distinct !DISubprogram(name: "Make", linkageName: "_CMake.Main", scope: null, file: !340, line: 14, type: !41, spFlags: DISPFlagDefinition, unit: !341)
// CHECK:STDOUT: !343 = !DILocation(line: 15, column: 3, scope: !342)
// CHECK:STDOUT: !345 = !DILocation(line: 14, column: 1, scope: !342)
// CHECK:STDOUT: !346 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.95528a86e1127e2d:core.Destroy.Core", scope: null, file: !340, line: 15, type: !22, spFlags: DISPFlagDefinition, unit: !341, retainedNodes: !349)
// CHECK:STDOUT: !347 = !DILocalVariable(arg: 1, scope: !346, type: !20)
// CHECK:STDOUT: !348 = !DILocation(line: 15, column: 3, scope: !346)
// CHECK:STDOUT: !349 = !{!347}
// CHECK:STDOUT: !350 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.5b23b4f14feb87e1:core.Destroy.Core", scope: null, file: !340, line: 15, type: !22, spFlags: DISPFlagDefinition, unit: !341, retainedNodes: !353)
// CHECK:STDOUT: !351 = !DILocalVariable(arg: 1, scope: !350, type: !20)
// CHECK:STDOUT: !352 = !DILocation(line: 15, column: 3, scope: !350)
// CHECK:STDOUT: !353 = !{!351}
// CHECK:STDOUT: !354 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.2f190654c863951f:core.Destroy.Core", scope: null, file: !340, line: 15, type: !22, spFlags: DISPFlagDefinition, unit: !341, retainedNodes: !357)
// CHECK:STDOUT: !355 = !DILocalVariable(arg: 1, scope: !354, type: !20)
// CHECK:STDOUT: !356 = !DILocation(line: 15, column: 3, scope: !354)
// CHECK:STDOUT: !357 = !{!355}
// CHECK:STDOUT: !358 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.b25525d21a9bd472:core.Destroy.Core", scope: null, file: !340, line: 15, type: !22, spFlags: DISPFlagDefinition, unit: !341, retainedNodes: !361)
// CHECK:STDOUT: !359 = !DILocalVariable(arg: 1, scope: !358, type: !20)
// CHECK:STDOUT: !360 = !DILocation(line: 15, column: 3, scope: !358)
// CHECK:STDOUT: !361 = !{!359}
// CHECK:STDOUT: !362 = distinct !DISubprogram(name: "F", linkageName: "_CF.Base.Main.6d6a4a2512463e59", scope: null, file: !340, line: 5, type: !22, spFlags: DISPFlagDefinition, unit: !341, retainedNodes: !366)
// CHECK:STDOUT: !363 = !DILocalVariable(arg: 1, scope: !362, type: !20)
// CHECK:STDOUT: !365 = !DILocation(line: 5, column: 3, scope: !362)
// CHECK:STDOUT: !366 = !{!363}
// CHECK:STDOUT: !367 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.acf75410c7781b1f", scope: null, file: !14, line: 24, type: !22, spFlags: DISPFlagDefinition, unit: !341, retainedNodes: !373)
// CHECK:STDOUT: !368 = !DILocalVariable(arg: 1, scope: !367, type: !20)
// CHECK:STDOUT: !371 = !DILocation(line: 26, column: 5, scope: !367)
// CHECK:STDOUT: !372 = !DILocation(line: 24, column: 3, scope: !367)
// CHECK:STDOUT: !373 = !{!368}
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
// CHECK:STDOUT: define void @_CF.Derived.Main(ptr sret({}) %return, ptr %self, ptr %n) #0 !dbg !380 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %return, ptr align 1 @From.val.loc18_14, i64 0, i1 false), !dbg !387
// CHECK:STDOUT:   ret void, !dbg !387
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_CF:thunk:Base.Main:Derived.Main"(ptr sret({}) %return, ptr %self, ptr %n) #1 !dbg !389 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc17_52.2.temp = alloca {}, align 1, !dbg !393
// CHECK:STDOUT:   %.5.temp = alloca {}, align 1, !dbg !392
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc17_52.2.temp), !dbg !393
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.5.temp), !dbg !392
// CHECK:STDOUT:   call void @"_CConvert.From.Main:ImplicitAs.a3bb440f6426a7d6.Core"(ptr %.5.temp, ptr %n), !dbg !392
// CHECK:STDOUT:   call void @_CF.Derived.Main(ptr %.loc17_52.2.temp, ptr %self, ptr %.5.temp), !dbg !393
// CHECK:STDOUT:   call void @"_CConvert.From.Main:ImplicitAs.a3bb440f6426a7d6.Core"(ptr %return, ptr %.loc17_52.2.temp), !dbg !393
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.3cdbb86f294f2bff(ptr %.loc17_52.2.temp), !dbg !393
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.3648669c9dbec3f4(ptr %.5.temp), !dbg !392
// CHECK:STDOUT:   ret void, !dbg !393
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.c2a743abd057cb33:core.Destroy.Core"(ptr %self) #0 !dbg !395 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !397
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.7ef2e8c2f12d163a:core.Destroy.Core"(ptr %self) #0 !dbg !399 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !401
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CUse.Main() #0 !dbg !403 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %d.var = alloca { { ptr } }, align 8, !dbg !404
// CHECK:STDOUT:   %.loc25_10.1.temp = alloca {}, align 1, !dbg !412
// CHECK:STDOUT:   %.loc25_9.2.temp = alloca {}, align 1, !dbg !413
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %d.var), !dbg !404
// CHECK:STDOUT:   %.loc23_31.2.base = getelementptr inbounds nuw { { ptr } }, ptr %d.var, i32 0, i32 0, !dbg !405
// CHECK:STDOUT:   %.loc23_30.2.vptr = getelementptr inbounds nuw { ptr }, ptr %.loc23_31.2.base, i32 0, i32 0, !dbg !406
// CHECK:STDOUT:   %.loc23_31.6.base = getelementptr inbounds nuw { { ptr } }, ptr %d.var, i32 0, i32 0, !dbg !405
// CHECK:STDOUT:   %.loc23_31.7.vptr = getelementptr inbounds nuw { ptr }, ptr %.loc23_31.6.base, i32 0, i32 0, !dbg !405
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 %d.var, ptr align 8 @Derived.val.loc23_31.5, i64 8, i1 false), !dbg !405
// CHECK:STDOUT:   store ptr @"_CDerived.Main.$vtable", ptr %.loc23_31.7.vptr, align 8, !dbg !405
// CHECK:STDOUT:   %.loc24_18.2.base = getelementptr inbounds nuw { { ptr } }, ptr %d.var, i32 0, i32 0, !dbg !409
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc25_10.1.temp), !dbg !412
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc25_9.2.temp), !dbg !413
// CHECK:STDOUT:   %Base.F.call.vtable = load ptr, ptr %.loc25_10.1.temp, align 8, !dbg !412
// CHECK:STDOUT:   %Base.F.call = call ptr @llvm.load.relative.i32(ptr %Base.F.call.vtable, i32 0), !dbg !412
// CHECK:STDOUT:   call void %Base.F.call(ptr %.loc25_10.1.temp, ptr %.loc24_18.2.base, ptr @From.val.loc18_14), !dbg !412
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.3648669c9dbec3f4(ptr %.loc25_10.1.temp), !dbg !412
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.3cdbb86f294f2bff(ptr @From.val), !dbg !413
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.acf75410c7781b1f(ptr %d.var), !dbg !404
// CHECK:STDOUT:   ret void, !dbg !414
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.95528a86e1127e2d:core.Destroy.Core"(ptr %self) #0 !dbg !415 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !417
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.294df737cd074d60:core.Destroy.Core"(ptr %self) #0 !dbg !419 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !421
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.caed8a96bf328d09:core.Destroy.Core"(ptr %self) #0 !dbg !423 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !425
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.b25525d21a9bd472:core.Destroy.Core"(ptr %self) #0 !dbg !427 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !429
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.3cdbb86f294f2bff(ptr %self) #0 !dbg !431 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.c2a743abd057cb33:core.Destroy.Core"(ptr %self), !dbg !435
// CHECK:STDOUT:   ret void, !dbg !436
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.3648669c9dbec3f4(ptr %self) #0 !dbg !438 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.7ef2e8c2f12d163a:core.Destroy.Core"(ptr %self), !dbg !442
// CHECK:STDOUT:   ret void, !dbg !443
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: read)
// CHECK:STDOUT: declare ptr @llvm.load.relative.i32(ptr, i32) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.acf75410c7781b1f(ptr %self) #0 !dbg !445 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.b25525d21a9bd472:core.Destroy.Core"(ptr %self), !dbg !449
// CHECK:STDOUT:   ret void, !dbg !450
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 4, 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Destroy.Core.3cdbb86f294f2bff, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Destroy.Core.3648669c9dbec3f4, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline nounwind }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #3 = { nocallback nofree nosync nounwind willreturn memory(argmem: read) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!375}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !14 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !20 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !21 = !{null, !20}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !40 = !{null}
// CHECK:STDOUT: !41 = !DISubroutineType(types: !40)
// CHECK:STDOUT: !374 = !DIFile(filename: "thunk.carbon", directory: "")
// CHECK:STDOUT: !375 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !374, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !378 = !{!20, !20, !20}
// CHECK:STDOUT: !379 = !DISubroutineType(types: !378)
// CHECK:STDOUT: !380 = distinct !DISubprogram(name: "F", linkageName: "_CF.Derived.Main", scope: null, file: !374, line: 17, type: !379, spFlags: DISPFlagDefinition, unit: !375, retainedNodes: !388)
// CHECK:STDOUT: !381 = !DILocalVariable(arg: 1, scope: !380, type: !20)
// CHECK:STDOUT: !382 = !DILocalVariable(arg: 2, scope: !380, type: !20)
// CHECK:STDOUT: !387 = !DILocation(line: 18, column: 5, scope: !380)
// CHECK:STDOUT: !388 = !{!381, !382}
// CHECK:STDOUT: !389 = distinct !DISubprogram(name: "F", linkageName: "_CF:thunk:Base.Main:Derived.Main", scope: null, file: !374, line: 17, type: !379, spFlags: DISPFlagDefinition, unit: !375, retainedNodes: !394)
// CHECK:STDOUT: !390 = !DILocalVariable(arg: 1, scope: !389, type: !20)
// CHECK:STDOUT: !391 = !DILocalVariable(arg: 2, scope: !389, type: !20)
// CHECK:STDOUT: !392 = !DILocation(line: 0, scope: !389)
// CHECK:STDOUT: !393 = !DILocation(line: 17, column: 3, scope: !389)
// CHECK:STDOUT: !394 = !{!390, !391}
// CHECK:STDOUT: !395 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.c2a743abd057cb33:core.Destroy.Core", scope: null, file: !374, line: 17, type: !22, spFlags: DISPFlagDefinition, unit: !375, retainedNodes: !398)
// CHECK:STDOUT: !396 = !DILocalVariable(arg: 1, scope: !395, type: !20)
// CHECK:STDOUT: !397 = !DILocation(line: 17, column: 3, scope: !395)
// CHECK:STDOUT: !398 = !{!396}
// CHECK:STDOUT: !399 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.7ef2e8c2f12d163a:core.Destroy.Core", scope: null, file: !374, type: !22, spFlags: DISPFlagDefinition, unit: !375, retainedNodes: !402)
// CHECK:STDOUT: !400 = !DILocalVariable(arg: 1, scope: !399, type: !20)
// CHECK:STDOUT: !401 = !DILocation(line: 0, scope: !399)
// CHECK:STDOUT: !402 = !{!400}
// CHECK:STDOUT: !403 = distinct !DISubprogram(name: "Use", linkageName: "_CUse.Main", scope: null, file: !374, line: 22, type: !41, spFlags: DISPFlagDefinition, unit: !375)
// CHECK:STDOUT: !404 = !DILocation(line: 23, column: 3, scope: !403)
// CHECK:STDOUT: !405 = !DILocation(line: 23, column: 20, scope: !403)
// CHECK:STDOUT: !406 = !DILocation(line: 23, column: 29, scope: !403)
// CHECK:STDOUT: !409 = !DILocation(line: 24, column: 18, scope: !403)
// CHECK:STDOUT: !412 = !DILocation(line: 25, column: 3, scope: !403)
// CHECK:STDOUT: !413 = !DILocation(line: 25, column: 8, scope: !403)
// CHECK:STDOUT: !414 = !DILocation(line: 22, column: 1, scope: !403)
// CHECK:STDOUT: !415 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.95528a86e1127e2d:core.Destroy.Core", scope: null, file: !374, line: 23, type: !22, spFlags: DISPFlagDefinition, unit: !375, retainedNodes: !418)
// CHECK:STDOUT: !416 = !DILocalVariable(arg: 1, scope: !415, type: !20)
// CHECK:STDOUT: !417 = !DILocation(line: 23, column: 3, scope: !415)
// CHECK:STDOUT: !418 = !{!416}
// CHECK:STDOUT: !419 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.294df737cd074d60:core.Destroy.Core", scope: null, file: !374, line: 23, type: !22, spFlags: DISPFlagDefinition, unit: !375, retainedNodes: !422)
// CHECK:STDOUT: !420 = !DILocalVariable(arg: 1, scope: !419, type: !20)
// CHECK:STDOUT: !421 = !DILocation(line: 23, column: 3, scope: !419)
// CHECK:STDOUT: !422 = !{!420}
// CHECK:STDOUT: !423 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.caed8a96bf328d09:core.Destroy.Core", scope: null, file: !374, line: 23, type: !22, spFlags: DISPFlagDefinition, unit: !375, retainedNodes: !426)
// CHECK:STDOUT: !424 = !DILocalVariable(arg: 1, scope: !423, type: !20)
// CHECK:STDOUT: !425 = !DILocation(line: 23, column: 3, scope: !423)
// CHECK:STDOUT: !426 = !{!424}
// CHECK:STDOUT: !427 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.b25525d21a9bd472:core.Destroy.Core", scope: null, file: !374, line: 23, type: !22, spFlags: DISPFlagDefinition, unit: !375, retainedNodes: !430)
// CHECK:STDOUT: !428 = !DILocalVariable(arg: 1, scope: !427, type: !20)
// CHECK:STDOUT: !429 = !DILocation(line: 23, column: 3, scope: !427)
// CHECK:STDOUT: !430 = !{!428}
// CHECK:STDOUT: !431 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.3cdbb86f294f2bff", scope: null, file: !14, line: 24, type: !22, spFlags: DISPFlagDefinition, unit: !375, retainedNodes: !437)
// CHECK:STDOUT: !432 = !DILocalVariable(arg: 1, scope: !431, type: !20)
// CHECK:STDOUT: !435 = !DILocation(line: 26, column: 5, scope: !431)
// CHECK:STDOUT: !436 = !DILocation(line: 24, column: 3, scope: !431)
// CHECK:STDOUT: !437 = !{!432}
// CHECK:STDOUT: !438 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.3648669c9dbec3f4", scope: null, file: !14, line: 24, type: !22, spFlags: DISPFlagDefinition, unit: !375, retainedNodes: !444)
// CHECK:STDOUT: !439 = !DILocalVariable(arg: 1, scope: !438, type: !20)
// CHECK:STDOUT: !442 = !DILocation(line: 26, column: 5, scope: !438)
// CHECK:STDOUT: !443 = !DILocation(line: 24, column: 3, scope: !438)
// CHECK:STDOUT: !444 = !{!439}
// CHECK:STDOUT: !445 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.acf75410c7781b1f", scope: null, file: !14, line: 24, type: !22, spFlags: DISPFlagDefinition, unit: !375, retainedNodes: !451)
// CHECK:STDOUT: !446 = !DILocalVariable(arg: 1, scope: !445, type: !20)
// CHECK:STDOUT: !449 = !DILocation(line: 26, column: 5, scope: !445)
// CHECK:STDOUT: !450 = !DILocation(line: 24, column: 3, scope: !445)
// CHECK:STDOUT: !451 = !{!446}
// CHECK:STDOUT:
