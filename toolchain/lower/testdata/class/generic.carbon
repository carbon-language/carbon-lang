// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/class/generic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/class/generic.carbon

// --- classes.carbon

package Classes;

base class Base(T:! type) {
  var b: T;
}

class Derived(T:! type) {
  extend base: Base(T);
  var d: T;
}

class Adapter(T:! type) {
  extend adapt Derived(T);
}

// --- create.carbon

package Create;

import Classes;

fn CreateDerived() -> Classes.Derived(i32) {
  return {.base = {.b = 1}, .d = 2};
}

fn CreateAdapter() -> Classes.Adapter(i32) {
  return CreateDerived() as Classes.Adapter(i32);
}

// --- create_generic.carbon

library "[[@TEST_NAME]]";

class A(T:! type) {
  var x: T;
  var y: T;
}

fn Make[T:! Core.Copy](x: T, y: T) -> A(T) {
  return {.x = x, .y = y};
}

fn Ints() -> A(i32) {
  return Make(1 as i32, 2 as i32);
}

fn Empty() -> A(()) {
  return Make((), ());
}

fn Tuples() -> A((i32, i32)) {
  let x: (i32, i32) = (1, 2);
  return Make(x, x);
}

// --- access.carbon

library "[[@TEST_NAME]]";

class C(T:! Core.Copy) {
  fn GetBool[self: Self]() -> bool {
    return self.v;
  }
  fn GetT[self: Self]() -> T {
    return self.w;
  }
  var v: bool;
  var w: T;
}

fn AccessBool() -> bool {
  var c: C(i32) = {.v = true, .w = 0};
  return c.GetBool();
}

fn AccessInt() -> i32 {
  var c: C(i32) = {.v = true, .w = 0};
  return c.GetT();
}

fn AccessEmpty() -> () {
  var c: C(()) = {.v = true, .w = ()};
  return c.GetT();
}

fn AccessTuple() -> (i32, i32, i32) {
  var c: C((i32, i32, i32)) = {.v = true, .w = (1, 2, 3)};
  return c.GetT();
}

// CHECK:STDOUT: ; ModuleID = 'classes.carbon'
// CHECK:STDOUT: source_filename = "classes.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "classes.carbon", directory: "")
// CHECK:STDOUT: ; ModuleID = 'create.carbon'
// CHECK:STDOUT: source_filename = "create.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @Derived.val.loc7_36 = internal constant { { i32 }, i32 } { { i32 } { i32 1 }, i32 2 }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CCreateDerived.Create(ptr sret({ { i32 }, i32 }) %return) !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_35.2.base = getelementptr inbounds nuw { { i32 }, i32 }, ptr %return, i32 0, i32 0, !dbg !7
// CHECK:STDOUT:   %.loc7_26.3.b = getelementptr inbounds nuw { i32 }, ptr %.loc7_35.2.base, i32 0, i32 0, !dbg !8
// CHECK:STDOUT:   %.loc7_35.5.d = getelementptr inbounds nuw { { i32 }, i32 }, ptr %return, i32 0, i32 1, !dbg !7
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %return, ptr align 4 @Derived.val.loc7_36, i64 8, i1 false), !dbg !9
// CHECK:STDOUT:   ret void, !dbg !9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CCreateAdapter.Create(ptr sret({ { i32 }, i32 }) %return) !dbg !10 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CCreateDerived.Create(ptr %return), !dbg !11
// CHECK:STDOUT:   ret void, !dbg !12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #0
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "create.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "CreateDerived", linkageName: "_CCreateDerived.Create", scope: null, file: !3, line: 6, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 7, column: 10, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 7, column: 19, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 7, column: 3, scope: !4)
// CHECK:STDOUT: !10 = distinct !DISubprogram(name: "CreateAdapter", linkageName: "_CCreateAdapter.Create", scope: null, file: !3, line: 10, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !11 = !DILocation(line: 11, column: 10, scope: !10)
// CHECK:STDOUT: !12 = !DILocation(line: 11, column: 3, scope: !10)
// CHECK:STDOUT: ; ModuleID = 'create_generic.carbon'
// CHECK:STDOUT: source_filename = "create_generic.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @tuple.loc22_28.6 = internal constant { i32, i32 } { i32 1, i32 2 }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CInts.Main(ptr sret({ i32, i32 }) %return) !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CMake.Main.de631560529e9861(ptr %return, i32 1, i32 2), !dbg !7
// CHECK:STDOUT:   ret void, !dbg !8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CEmpty.Main(ptr sret({ {}, {} }) %return) !dbg !9 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CMake.Main.cf13cead63317d44(ptr %return), !dbg !10
// CHECK:STDOUT:   ret void, !dbg !11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CTuples.Main(ptr sret({ { i32, i32 }, { i32, i32 } }) %return) !dbg !12 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CMake.Main.54ec9394d5df5464(ptr %return, ptr @tuple.loc22_28.6, ptr @tuple.loc22_28.6), !dbg !13
// CHECK:STDOUT:   ret void, !dbg !14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr void @_CMake.Main.de631560529e9861(ptr sret({ i32, i32 }) %return, i32 %x, i32 %y) !dbg !15 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc10_25.2.x = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 0, !dbg !16
// CHECK:STDOUT:   store i32 %x, ptr %.loc10_25.2.x, align 4, !dbg !16
// CHECK:STDOUT:   %.loc10_25.4.y = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 1, !dbg !16
// CHECK:STDOUT:   store i32 %y, ptr %.loc10_25.4.y, align 4, !dbg !16
// CHECK:STDOUT:   ret void, !dbg !17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr void @_CMake.Main.cf13cead63317d44(ptr sret({ {}, {} }) %return) !dbg !18 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc10_25.2.x = getelementptr inbounds nuw { {}, {} }, ptr %return, i32 0, i32 0, !dbg !19
// CHECK:STDOUT:   %.loc10_25.4.y = getelementptr inbounds nuw { {}, {} }, ptr %return, i32 0, i32 1, !dbg !19
// CHECK:STDOUT:   ret void, !dbg !20
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr void @_CMake.Main.54ec9394d5df5464(ptr sret({ { i32, i32 }, { i32, i32 } }) %return, ptr %x, ptr %y) !dbg !21 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc10_25.2.x = getelementptr inbounds nuw { { i32, i32 }, { i32, i32 } }, ptr %return, i32 0, i32 0, !dbg !22
// CHECK:STDOUT:   call void @"_COp.0f13b78cd2788a9b:Copy.Core.58c6957df82926a9"(ptr %.loc10_25.2.x, ptr %x), !dbg !23
// CHECK:STDOUT:   %.loc10_25.4.y = getelementptr inbounds nuw { { i32, i32 }, { i32, i32 } }, ptr %return, i32 0, i32 1, !dbg !22
// CHECK:STDOUT:   call void @"_COp.0f13b78cd2788a9b:Copy.Core.58c6957df82926a9"(ptr %.loc10_25.4.y, ptr %y), !dbg !24
// CHECK:STDOUT:   ret void, !dbg !25
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr void @"_COp.0f13b78cd2788a9b:Copy.Core.58c6957df82926a9"(ptr sret({ i32, i32 }) %return, ptr %self) !dbg !26 {
// CHECK:STDOUT:   %tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 0, !dbg !28
// CHECK:STDOUT:   %tuple.elem.load = load i32, ptr %tuple.elem, align 4, !dbg !28
// CHECK:STDOUT:   %tuple.elem1 = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 0, !dbg !29
// CHECK:STDOUT:   %tuple.elem2 = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 1, !dbg !30
// CHECK:STDOUT:   %tuple.elem.load3 = load i32, ptr %tuple.elem2, align 4, !dbg !30
// CHECK:STDOUT:   %tuple.elem4 = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 1, !dbg !29
// CHECK:STDOUT:   store i32 %tuple.elem.load, ptr %tuple.elem1, align 4, !dbg !29
// CHECK:STDOUT:   store i32 %tuple.elem.load3, ptr %tuple.elem4, align 4, !dbg !29
// CHECK:STDOUT:   ret void, !dbg !31
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @"_COp.0f13b78cd2788a9b:Copy.Core.58c6957df82926a9", { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "create_generic.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Ints", linkageName: "_CInts.Main", scope: null, file: !3, line: 13, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 14, column: 10, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 14, column: 3, scope: !4)
// CHECK:STDOUT: !9 = distinct !DISubprogram(name: "Empty", linkageName: "_CEmpty.Main", scope: null, file: !3, line: 17, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !10 = !DILocation(line: 18, column: 10, scope: !9)
// CHECK:STDOUT: !11 = !DILocation(line: 18, column: 3, scope: !9)
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "Tuples", linkageName: "_CTuples.Main", scope: null, file: !3, line: 21, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !13 = !DILocation(line: 23, column: 10, scope: !12)
// CHECK:STDOUT: !14 = !DILocation(line: 23, column: 3, scope: !12)
// CHECK:STDOUT: !15 = distinct !DISubprogram(name: "Make", linkageName: "_CMake.Main.de631560529e9861", scope: null, file: !3, line: 9, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !16 = !DILocation(line: 10, column: 10, scope: !15)
// CHECK:STDOUT: !17 = !DILocation(line: 10, column: 3, scope: !15)
// CHECK:STDOUT: !18 = distinct !DISubprogram(name: "Make", linkageName: "_CMake.Main.cf13cead63317d44", scope: null, file: !3, line: 9, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !19 = !DILocation(line: 10, column: 10, scope: !18)
// CHECK:STDOUT: !20 = !DILocation(line: 10, column: 3, scope: !18)
// CHECK:STDOUT: !21 = distinct !DISubprogram(name: "Make", linkageName: "_CMake.Main.54ec9394d5df5464", scope: null, file: !3, line: 9, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !22 = !DILocation(line: 10, column: 10, scope: !21)
// CHECK:STDOUT: !23 = !DILocation(line: 10, column: 16, scope: !21)
// CHECK:STDOUT: !24 = !DILocation(line: 10, column: 24, scope: !21)
// CHECK:STDOUT: !25 = !DILocation(line: 10, column: 3, scope: !21)
// CHECK:STDOUT: !26 = distinct !DISubprogram(name: "Op", linkageName: "_COp.0f13b78cd2788a9b:Copy.Core.58c6957df82926a9", scope: null, file: !27, line: 48, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !27 = !DIFile(filename: "min_prelude/parts/copy.carbon", directory: "")
// CHECK:STDOUT: !28 = !DILocation(line: 49, column: 13, scope: !26)
// CHECK:STDOUT: !29 = !DILocation(line: 49, column: 12, scope: !26)
// CHECK:STDOUT: !30 = !DILocation(line: 49, column: 26, scope: !26)
// CHECK:STDOUT: !31 = !DILocation(line: 49, column: 5, scope: !26)
// CHECK:STDOUT: ; ModuleID = 'access.carbon'
// CHECK:STDOUT: source_filename = "access.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @C.val.bc7.loc16_3 = internal constant { i1, i32 } { i1 true, i32 0 }
// CHECK:STDOUT: @C.val.f52.loc26_3 = internal constant { i1, {} } { i1 true, {} zeroinitializer }
// CHECK:STDOUT: @C.val.f32.loc31_3 = internal constant { i1, { i32, i32, i32 } } { i1 true, { i32, i32, i32 } { i32 1, i32 2, i32 3 } }
// CHECK:STDOUT:
// CHECK:STDOUT: define i1 @_CAccessBool.Main() !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c.var = alloca { i1, i32 }, align 8, !dbg !7
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !7
// CHECK:STDOUT:   %.loc16_37.2.v = getelementptr inbounds nuw { i1, i32 }, ptr %c.var, i32 0, i32 0, !dbg !8
// CHECK:STDOUT:   %.loc16_37.5.w = getelementptr inbounds nuw { i1, i32 }, ptr %c.var, i32 0, i32 1, !dbg !8
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %c.var, ptr align 4 @C.val.bc7.loc16_3, i64 8, i1 false), !dbg !7
// CHECK:STDOUT:   %C.GetBool.call = call i1 @_CGetBool.C.Main.de631560529e9861(ptr %c.var), !dbg !9
// CHECK:STDOUT:   ret i1 %C.GetBool.call, !dbg !10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @_CAccessInt.Main() !dbg !11 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c.var = alloca { i1, i32 }, align 8, !dbg !12
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !12
// CHECK:STDOUT:   %.loc21_37.2.v = getelementptr inbounds nuw { i1, i32 }, ptr %c.var, i32 0, i32 0, !dbg !13
// CHECK:STDOUT:   %.loc21_37.5.w = getelementptr inbounds nuw { i1, i32 }, ptr %c.var, i32 0, i32 1, !dbg !13
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %c.var, ptr align 4 @C.val.bc7.loc16_3, i64 8, i1 false), !dbg !12
// CHECK:STDOUT:   %C.GetT.call = call i32 @_CGetT.C.Main.de631560529e9861(ptr %c.var), !dbg !14
// CHECK:STDOUT:   ret i32 %C.GetT.call, !dbg !15
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CAccessEmpty.Main() !dbg !16 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c.var = alloca { i1, {} }, align 8, !dbg !17
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !17
// CHECK:STDOUT:   %.loc26_37.2.v = getelementptr inbounds nuw { i1, {} }, ptr %c.var, i32 0, i32 0, !dbg !18
// CHECK:STDOUT:   %.loc26_37.4.w = getelementptr inbounds nuw { i1, {} }, ptr %c.var, i32 0, i32 1, !dbg !18
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %c.var, ptr align 1 @C.val.f52.loc26_3, i64 1, i1 false), !dbg !17
// CHECK:STDOUT:   call void @_CGetT.C.Main.cf13cead63317d44(ptr %c.var), !dbg !19
// CHECK:STDOUT:   ret void, !dbg !20
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CAccessTuple.Main(ptr sret({ i32, i32, i32 }) %return) !dbg !21 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c.var = alloca { i1, { i32, i32, i32 } }, align 8, !dbg !22
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !22
// CHECK:STDOUT:   %.loc31_57.2.v = getelementptr inbounds nuw { i1, { i32, i32, i32 } }, ptr %c.var, i32 0, i32 0, !dbg !23
// CHECK:STDOUT:   %.loc31_57.4.w = getelementptr inbounds nuw { i1, { i32, i32, i32 } }, ptr %c.var, i32 0, i32 1, !dbg !23
// CHECK:STDOUT:   %tuple.elem0.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %.loc31_57.4.w, i32 0, i32 0, !dbg !24
// CHECK:STDOUT:   %tuple.elem1.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %.loc31_57.4.w, i32 0, i32 1, !dbg !24
// CHECK:STDOUT:   %tuple.elem2.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %.loc31_57.4.w, i32 0, i32 2, !dbg !24
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %c.var, ptr align 4 @C.val.f32.loc31_3, i64 16, i1 false), !dbg !22
// CHECK:STDOUT:   call void @_CGetT.C.Main.7a4734f1d45ccfe4(ptr %return, ptr %c.var), !dbg !25
// CHECK:STDOUT:   ret void, !dbg !26
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #0
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr i1 @_CGetBool.C.Main.de631560529e9861(ptr %self) !dbg !27 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc6_16.1.v = getelementptr inbounds nuw { i1, i32 }, ptr %self, i32 0, i32 0, !dbg !28
// CHECK:STDOUT:   %.loc6_16.2 = load i8, ptr %.loc6_16.1.v, align 1, !dbg !28
// CHECK:STDOUT:   %.loc6_16.21 = trunc i8 %.loc6_16.2 to i1, !dbg !28
// CHECK:STDOUT:   ret i1 %.loc6_16.21, !dbg !29
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr i32 @_CGetT.C.Main.de631560529e9861(ptr %self) !dbg !30 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc9_16.1.w = getelementptr inbounds nuw { i1, i32 }, ptr %self, i32 0, i32 1, !dbg !31
// CHECK:STDOUT:   %.loc9_16.2 = load i32, ptr %.loc9_16.1.w, align 4, !dbg !31
// CHECK:STDOUT:   ret i32 %.loc9_16.2, !dbg !32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr void @_CGetT.C.Main.cf13cead63317d44(ptr %self) !dbg !33 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc9_16.1.w = getelementptr inbounds nuw { i1, {} }, ptr %self, i32 0, i32 1, !dbg !34
// CHECK:STDOUT:   ret void, !dbg !35
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr void @_CGetT.C.Main.7a4734f1d45ccfe4(ptr sret({ i32, i32, i32 }) %return, ptr %self) !dbg !36 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc9_16.1.w = getelementptr inbounds nuw { i1, { i32, i32, i32 } }, ptr %self, i32 0, i32 1, !dbg !37
// CHECK:STDOUT:   call void @"_COp.95e85d56bfcba7ff:Copy.Core.03f4bad6da36de16"(ptr %return, ptr %.loc9_16.1.w), !dbg !37
// CHECK:STDOUT:   ret void, !dbg !38
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr void @"_COp.95e85d56bfcba7ff:Copy.Core.03f4bad6da36de16"(ptr sret({ i32, i32, i32 }) %return, ptr %self) !dbg !39 {
// CHECK:STDOUT:   %tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %self, i32 0, i32 0, !dbg !41
// CHECK:STDOUT:   %tuple.elem.load = load i32, ptr %tuple.elem, align 4, !dbg !41
// CHECK:STDOUT:   %tuple.elem1 = getelementptr inbounds nuw { i32, i32, i32 }, ptr %return, i32 0, i32 0, !dbg !42
// CHECK:STDOUT:   %tuple.elem2 = getelementptr inbounds nuw { i32, i32, i32 }, ptr %self, i32 0, i32 1, !dbg !43
// CHECK:STDOUT:   %tuple.elem.load3 = load i32, ptr %tuple.elem2, align 4, !dbg !43
// CHECK:STDOUT:   %tuple.elem4 = getelementptr inbounds nuw { i32, i32, i32 }, ptr %return, i32 0, i32 1, !dbg !42
// CHECK:STDOUT:   %tuple.elem5 = getelementptr inbounds nuw { i32, i32, i32 }, ptr %self, i32 0, i32 2, !dbg !44
// CHECK:STDOUT:   %tuple.elem.load6 = load i32, ptr %tuple.elem5, align 4, !dbg !44
// CHECK:STDOUT:   %tuple.elem7 = getelementptr inbounds nuw { i32, i32, i32 }, ptr %return, i32 0, i32 2, !dbg !42
// CHECK:STDOUT:   store i32 %tuple.elem.load, ptr %tuple.elem1, align 4, !dbg !42
// CHECK:STDOUT:   store i32 %tuple.elem.load3, ptr %tuple.elem4, align 4, !dbg !42
// CHECK:STDOUT:   store i32 %tuple.elem.load6, ptr %tuple.elem7, align 4, !dbg !42
// CHECK:STDOUT:   ret void, !dbg !45
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 3, 2, 1, 0 }
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
// CHECK:STDOUT: !3 = !DIFile(filename: "access.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "AccessBool", linkageName: "_CAccessBool.Main", scope: null, file: !3, line: 15, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 16, column: 3, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 16, column: 19, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 17, column: 10, scope: !4)
// CHECK:STDOUT: !10 = !DILocation(line: 17, column: 3, scope: !4)
// CHECK:STDOUT: !11 = distinct !DISubprogram(name: "AccessInt", linkageName: "_CAccessInt.Main", scope: null, file: !3, line: 20, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !12 = !DILocation(line: 21, column: 3, scope: !11)
// CHECK:STDOUT: !13 = !DILocation(line: 21, column: 19, scope: !11)
// CHECK:STDOUT: !14 = !DILocation(line: 22, column: 10, scope: !11)
// CHECK:STDOUT: !15 = !DILocation(line: 22, column: 3, scope: !11)
// CHECK:STDOUT: !16 = distinct !DISubprogram(name: "AccessEmpty", linkageName: "_CAccessEmpty.Main", scope: null, file: !3, line: 25, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !17 = !DILocation(line: 26, column: 3, scope: !16)
// CHECK:STDOUT: !18 = !DILocation(line: 26, column: 18, scope: !16)
// CHECK:STDOUT: !19 = !DILocation(line: 27, column: 10, scope: !16)
// CHECK:STDOUT: !20 = !DILocation(line: 27, column: 3, scope: !16)
// CHECK:STDOUT: !21 = distinct !DISubprogram(name: "AccessTuple", linkageName: "_CAccessTuple.Main", scope: null, file: !3, line: 30, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !22 = !DILocation(line: 31, column: 3, scope: !21)
// CHECK:STDOUT: !23 = !DILocation(line: 31, column: 31, scope: !21)
// CHECK:STDOUT: !24 = !DILocation(line: 31, column: 48, scope: !21)
// CHECK:STDOUT: !25 = !DILocation(line: 32, column: 10, scope: !21)
// CHECK:STDOUT: !26 = !DILocation(line: 32, column: 3, scope: !21)
// CHECK:STDOUT: !27 = distinct !DISubprogram(name: "GetBool", linkageName: "_CGetBool.C.Main.de631560529e9861", scope: null, file: !3, line: 5, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !28 = !DILocation(line: 6, column: 12, scope: !27)
// CHECK:STDOUT: !29 = !DILocation(line: 6, column: 5, scope: !27)
// CHECK:STDOUT: !30 = distinct !DISubprogram(name: "GetT", linkageName: "_CGetT.C.Main.de631560529e9861", scope: null, file: !3, line: 8, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !31 = !DILocation(line: 9, column: 12, scope: !30)
// CHECK:STDOUT: !32 = !DILocation(line: 9, column: 5, scope: !30)
// CHECK:STDOUT: !33 = distinct !DISubprogram(name: "GetT", linkageName: "_CGetT.C.Main.cf13cead63317d44", scope: null, file: !3, line: 8, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !34 = !DILocation(line: 9, column: 12, scope: !33)
// CHECK:STDOUT: !35 = !DILocation(line: 9, column: 5, scope: !33)
// CHECK:STDOUT: !36 = distinct !DISubprogram(name: "GetT", linkageName: "_CGetT.C.Main.7a4734f1d45ccfe4", scope: null, file: !3, line: 8, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !37 = !DILocation(line: 9, column: 12, scope: !36)
// CHECK:STDOUT: !38 = !DILocation(line: 9, column: 5, scope: !36)
// CHECK:STDOUT: !39 = distinct !DISubprogram(name: "Op", linkageName: "_COp.95e85d56bfcba7ff:Copy.Core.03f4bad6da36de16", scope: null, file: !40, line: 54, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !40 = !DIFile(filename: "min_prelude/parts/copy.carbon", directory: "")
// CHECK:STDOUT: !41 = !DILocation(line: 55, column: 13, scope: !39)
// CHECK:STDOUT: !42 = !DILocation(line: 55, column: 12, scope: !39)
// CHECK:STDOUT: !43 = !DILocation(line: 55, column: 26, scope: !39)
// CHECK:STDOUT: !44 = !DILocation(line: 55, column: 39, scope: !39)
// CHECK:STDOUT: !45 = !DILocation(line: 55, column: 5, scope: !39)
