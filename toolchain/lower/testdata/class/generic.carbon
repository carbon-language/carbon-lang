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

base class Base(T: type) {
  var b: T;
}

class Derived(T: type) {
  extend base: Base(T);
  var d: T;
}

class Adapter(T: type) {
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

class A(T: type) {
  var x: T;
  var y: T;
}

fn Make[T: Core.Copy](x: T, y: T) -> A(T) {
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

class C(T: Core.Copy) {
  fn GetBool(self) -> bool {
    return self.v;
  }
  fn GetT(self) -> T {
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

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'classes.carbon'
// CHECK:STDOUT: source_filename = "classes.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!35}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !34 = !DIFile(filename: "classes.carbon", directory: "")
// CHECK:STDOUT: !35 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !34, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'create.carbon'
// CHECK:STDOUT: source_filename = "create.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @Derived.val.loc7_36 = internal constant { { i32 }, i32 } { { i32 } { i32 1 }, i32 2 }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCreateDerived.Create(ptr sret({ { i32 }, i32 }) %return) #0 !dbg !42 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_35.2.base = getelementptr inbounds nuw { { i32 }, i32 }, ptr %return, i32 0, i32 0, !dbg !45
// CHECK:STDOUT:   %.loc7_26.3.b = getelementptr inbounds nuw { i32 }, ptr %.loc7_35.2.base, i32 0, i32 0, !dbg !46
// CHECK:STDOUT:   %.loc7_35.6.d = getelementptr inbounds nuw { { i32 }, i32 }, ptr %return, i32 0, i32 1, !dbg !45
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %return, ptr align 4 @Derived.val.loc7_36, i64 8, i1 false), !dbg !47
// CHECK:STDOUT:   ret void, !dbg !47
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCreateAdapter.Create(ptr sret({ { i32 }, i32 }) %return) #0 !dbg !48 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CCreateDerived.Create(ptr %return), !dbg !51
// CHECK:STDOUT:   ret void, !dbg !52
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!37}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !36 = !DIFile(filename: "create.carbon", directory: "")
// CHECK:STDOUT: !37 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !36, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !39 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !40 = !{!39}
// CHECK:STDOUT: !41 = !DISubroutineType(types: !40)
// CHECK:STDOUT: !42 = distinct !DISubprogram(name: "CreateDerived", linkageName: "_CCreateDerived.Create", scope: null, file: !36, line: 6, type: !41, spFlags: DISPFlagDefinition, unit: !37)
// CHECK:STDOUT: !45 = !DILocation(line: 7, column: 10, scope: !42)
// CHECK:STDOUT: !46 = !DILocation(line: 7, column: 19, scope: !42)
// CHECK:STDOUT: !47 = !DILocation(line: 7, column: 3, scope: !42)
// CHECK:STDOUT: !48 = distinct !DISubprogram(name: "CreateAdapter", linkageName: "_CCreateAdapter.Create", scope: null, file: !36, line: 10, type: !41, spFlags: DISPFlagDefinition, unit: !37)
// CHECK:STDOUT: !51 = !DILocation(line: 11, column: 10, scope: !48)
// CHECK:STDOUT: !52 = !DILocation(line: 11, column: 3, scope: !48)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'create_generic.carbon'
// CHECK:STDOUT: source_filename = "create_generic.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @tuple.205.loc22_28.6 = internal constant { i32, i32 } { i32 1, i32 2 }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CInts.Main(ptr sret({ i32, i32 }) %return) #0 !dbg !55 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CMake.Main.2aef1d52a6e1f957(ptr %return, i32 1, i32 2), !dbg !58
// CHECK:STDOUT:   ret void, !dbg !61
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CEmpty.Main(ptr sret({ {}, {} }) %return) #0 !dbg !62 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CMake.Main.22bf4b2f6c4322a2(ptr %return), !dbg !65
// CHECK:STDOUT:   ret void, !dbg !66
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTuples.Main(ptr sret({ { i32, i32 }, { i32, i32 } }) %return) #0 !dbg !67 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CMake.Main.9256efb2ae218f22(ptr %return, ptr @tuple.205.loc22_28.6, ptr @tuple.205.loc22_28.6), !dbg !74
// CHECK:STDOUT:   ret void, !dbg !77
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CMake.Main.2aef1d52a6e1f957(ptr sret({ i32, i32 }) %return, i32 %x, i32 %y) #0 !dbg !78 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc10_25.2.x = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 0, !dbg !87
// CHECK:STDOUT:   store i32 %x, ptr %.loc10_25.2.x, align 4, !dbg !87
// CHECK:STDOUT:   %.loc10_25.4.y = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 1, !dbg !87
// CHECK:STDOUT:   store i32 %y, ptr %.loc10_25.4.y, align 4, !dbg !87
// CHECK:STDOUT:   ret void, !dbg !88
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CMake.Main.22bf4b2f6c4322a2(ptr sret({ {}, {} }) %return) #0 !dbg !90 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc10_25.2.x = getelementptr inbounds nuw { {}, {} }, ptr %return, i32 0, i32 0, !dbg !97
// CHECK:STDOUT:   %.loc10_25.4.y = getelementptr inbounds nuw { {}, {} }, ptr %return, i32 0, i32 1, !dbg !97
// CHECK:STDOUT:   ret void, !dbg !98
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CMake.Main.9256efb2ae218f22(ptr sret({ { i32, i32 }, { i32, i32 } }) %return, ptr %x, ptr %y) #0 !dbg !99 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc10_25.2.x = getelementptr inbounds nuw { { i32, i32 }, { i32, i32 } }, ptr %return, i32 0, i32 0, !dbg !108
// CHECK:STDOUT:   call void @"_COp.4e4edddc7580c79b:Copy.Core.166c3bb67ee40138"(ptr %.loc10_25.2.x, ptr %x), !dbg !106
// CHECK:STDOUT:   %.loc10_25.4.y = getelementptr inbounds nuw { { i32, i32 }, { i32, i32 } }, ptr %return, i32 0, i32 1, !dbg !108
// CHECK:STDOUT:   call void @"_COp.4e4edddc7580c79b:Copy.Core.166c3bb67ee40138"(ptr %.loc10_25.4.y, ptr %y), !dbg !107
// CHECK:STDOUT:   ret void, !dbg !111
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp.4e4edddc7580c79b:Copy.Core.166c3bb67ee40138"(ptr sret({ i32, i32 }) %return, ptr %self) #0 !dbg !113 {
// CHECK:STDOUT:   %tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 0, !dbg !118
// CHECK:STDOUT:   %tuple.elem.load = load i32, ptr %tuple.elem, align 4, !dbg !118
// CHECK:STDOUT:   %tuple.elem1 = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 0, !dbg !119
// CHECK:STDOUT:   %tuple.elem2 = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 1, !dbg !120
// CHECK:STDOUT:   %tuple.elem.load3 = load i32, ptr %tuple.elem2, align 4, !dbg !120
// CHECK:STDOUT:   %tuple.elem4 = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 1, !dbg !119
// CHECK:STDOUT:   store i32 %tuple.elem.load, ptr %tuple.elem1, align 4, !dbg !119
// CHECK:STDOUT:   store i32 %tuple.elem.load3, ptr %tuple.elem4, align 4, !dbg !119
// CHECK:STDOUT:   ret void, !dbg !121
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @"_COp.4e4edddc7580c79b:Copy.Core.166c3bb67ee40138", { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!54}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !11 = !DIFile(filename: "min_prelude/parts/copy.carbon", directory: "")
// CHECK:STDOUT: !38 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !39 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !40 = !{!39}
// CHECK:STDOUT: !41 = !DISubroutineType(types: !40)
// CHECK:STDOUT: !53 = !DIFile(filename: "create_generic.carbon", directory: "")
// CHECK:STDOUT: !54 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !53, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !55 = distinct !DISubprogram(name: "Ints", linkageName: "_CInts.Main", scope: null, file: !53, line: 13, type: !41, spFlags: DISPFlagDefinition, unit: !54)
// CHECK:STDOUT: !58 = !DILocation(line: 14, column: 10, scope: !55)
// CHECK:STDOUT: !59 = !{!39, !38, !38}
// CHECK:STDOUT: !60 = !DISubroutineType(types: !59)
// CHECK:STDOUT: !61 = !DILocation(line: 14, column: 3, scope: !55)
// CHECK:STDOUT: !62 = distinct !DISubprogram(name: "Empty", linkageName: "_CEmpty.Main", scope: null, file: !53, line: 17, type: !41, spFlags: DISPFlagDefinition, unit: !54)
// CHECK:STDOUT: !65 = !DILocation(line: 18, column: 10, scope: !62)
// CHECK:STDOUT: !66 = !DILocation(line: 18, column: 3, scope: !62)
// CHECK:STDOUT: !67 = distinct !DISubprogram(name: "Tuples", linkageName: "_CTuples.Main", scope: null, file: !53, line: 21, type: !41, spFlags: DISPFlagDefinition, unit: !54)
// CHECK:STDOUT: !74 = !DILocation(line: 23, column: 10, scope: !67)
// CHECK:STDOUT: !75 = !{!39, !39, !39}
// CHECK:STDOUT: !76 = !DISubroutineType(types: !75)
// CHECK:STDOUT: !77 = !DILocation(line: 23, column: 3, scope: !67)
// CHECK:STDOUT: !78 = distinct !DISubprogram(name: "Make", linkageName: "_CMake.Main.2aef1d52a6e1f957", scope: null, file: !53, line: 9, type: !60, spFlags: DISPFlagDefinition, unit: !54, retainedNodes: !89)
// CHECK:STDOUT: !79 = !DILocalVariable(arg: 1, scope: !78, type: !38)
// CHECK:STDOUT: !80 = !DILocalVariable(arg: 2, scope: !78, type: !38)
// CHECK:STDOUT: !87 = !DILocation(line: 10, column: 10, scope: !78)
// CHECK:STDOUT: !88 = !DILocation(line: 10, column: 3, scope: !78)
// CHECK:STDOUT: !89 = !{!79, !80}
// CHECK:STDOUT: !90 = distinct !DISubprogram(name: "Make", linkageName: "_CMake.Main.22bf4b2f6c4322a2", scope: null, file: !53, line: 9, type: !41, spFlags: DISPFlagDefinition, unit: !54)
// CHECK:STDOUT: !97 = !DILocation(line: 10, column: 10, scope: !90)
// CHECK:STDOUT: !98 = !DILocation(line: 10, column: 3, scope: !90)
// CHECK:STDOUT: !99 = distinct !DISubprogram(name: "Make", linkageName: "_CMake.Main.9256efb2ae218f22", scope: null, file: !53, line: 9, type: !76, spFlags: DISPFlagDefinition, unit: !54, retainedNodes: !112)
// CHECK:STDOUT: !100 = !DILocalVariable(arg: 1, scope: !99, type: !39)
// CHECK:STDOUT: !101 = !DILocalVariable(arg: 2, scope: !99, type: !39)
// CHECK:STDOUT: !106 = !DILocation(line: 10, column: 16, scope: !99)
// CHECK:STDOUT: !107 = !DILocation(line: 10, column: 24, scope: !99)
// CHECK:STDOUT: !108 = !DILocation(line: 10, column: 10, scope: !99)
// CHECK:STDOUT: !109 = !{!39, !39}
// CHECK:STDOUT: !110 = !DISubroutineType(types: !109)
// CHECK:STDOUT: !111 = !DILocation(line: 10, column: 3, scope: !99)
// CHECK:STDOUT: !112 = !{!100, !101}
// CHECK:STDOUT: !113 = distinct !DISubprogram(name: "Op", linkageName: "_COp.4e4edddc7580c79b:Copy.Core.166c3bb67ee40138", scope: null, file: !11, line: 52, type: !110, spFlags: DISPFlagDefinition, unit: !54, retainedNodes: !122)
// CHECK:STDOUT: !114 = !DILocalVariable(arg: 1, scope: !113, type: !39)
// CHECK:STDOUT: !118 = !DILocation(line: 53, column: 13, scope: !113)
// CHECK:STDOUT: !119 = !DILocation(line: 53, column: 12, scope: !113)
// CHECK:STDOUT: !120 = !DILocation(line: 53, column: 26, scope: !113)
// CHECK:STDOUT: !121 = !DILocation(line: 53, column: 5, scope: !113)
// CHECK:STDOUT: !122 = !{!114}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'access.carbon'
// CHECK:STDOUT: source_filename = "access.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @C.val.06d.loc16_3.1 = internal constant { i1, i32 } { i1 true, i32 0 }
// CHECK:STDOUT: @C.val.722.loc26_3.1 = internal constant { i1, {} } { i1 true, {} zeroinitializer }
// CHECK:STDOUT: @C.val.97f.loc31_3.1 = internal constant { i1, { i32, i32, i32 } } { i1 true, { i32, i32, i32 } { i32 1, i32 2, i32 3 } }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CAccessBool.Main() #0 !dbg !131 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c.var = alloca { i1, i32 }, align 4, !dbg !134
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !134
// CHECK:STDOUT:   %.loc16_37.2.v = getelementptr inbounds nuw { i1, i32 }, ptr %c.var, i32 0, i32 0, !dbg !135
// CHECK:STDOUT:   %.loc16_37.5.w = getelementptr inbounds nuw { i1, i32 }, ptr %c.var, i32 0, i32 1, !dbg !135
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %c.var, ptr align 4 @C.val.06d.loc16_3.1, i64 8, i1 false), !dbg !134
// CHECK:STDOUT:   %C.GetBool.call = call i1 @_CGetBool.C.Main.2aef1d52a6e1f957(ptr %c.var), !dbg !137
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.ee380be3044985a2(ptr %c.var), !dbg !134
// CHECK:STDOUT:   ret i1 %C.GetBool.call, !dbg !138
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.7dca9c77f1689903:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !139 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc16_3.1.w = getelementptr inbounds nuw { i1, i32 }, ptr %self, i32 0, i32 1, !dbg !141
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.0aefc5da911f5051(ptr %.loc16_3.1.w), !dbg !141
// CHECK:STDOUT:   %.loc16_3.4.v = getelementptr inbounds nuw { i1, i32 }, ptr %self, i32 0, i32 0, !dbg !141
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.c96afdcab9de049c(ptr %.loc16_3.4.v), !dbg !141
// CHECK:STDOUT:   ret void, !dbg !141
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.b78480f348383035:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !145 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc16_3.1.w = getelementptr inbounds nuw { i1, i32 }, ptr %self, i32 0, i32 1, !dbg !147
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.0aefc5da911f5051(ptr %.loc16_3.1.w), !dbg !147
// CHECK:STDOUT:   %.loc16_3.4.v = getelementptr inbounds nuw { i1, i32 }, ptr %self, i32 0, i32 0, !dbg !147
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.c96afdcab9de049c(ptr %.loc16_3.4.v), !dbg !147
// CHECK:STDOUT:   ret void, !dbg !147
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CAccessInt.Main() #0 !dbg !149 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c.var = alloca { i1, i32 }, align 4, !dbg !152
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !152
// CHECK:STDOUT:   %.loc21_37.2.v = getelementptr inbounds nuw { i1, i32 }, ptr %c.var, i32 0, i32 0, !dbg !153
// CHECK:STDOUT:   %.loc21_37.5.w = getelementptr inbounds nuw { i1, i32 }, ptr %c.var, i32 0, i32 1, !dbg !153
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %c.var, ptr align 4 @C.val.06d.loc16_3.1, i64 8, i1 false), !dbg !152
// CHECK:STDOUT:   %C.GetT.call = call i32 @_CGetT.C.Main.2aef1d52a6e1f957(ptr %c.var), !dbg !155
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.ee380be3044985a2(ptr %c.var), !dbg !152
// CHECK:STDOUT:   ret i32 %C.GetT.call, !dbg !158
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CAccessEmpty.Main() #0 !dbg !159 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c.var = alloca { i1, {} }, align 1, !dbg !162
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !162
// CHECK:STDOUT:   %.loc26_37.2.v = getelementptr inbounds nuw { i1, {} }, ptr %c.var, i32 0, i32 0, !dbg !163
// CHECK:STDOUT:   %.loc26_37.4.w = getelementptr inbounds nuw { i1, {} }, ptr %c.var, i32 0, i32 1, !dbg !163
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %c.var, ptr align 1 @C.val.722.loc26_3.1, i64 1, i1 false), !dbg !162
// CHECK:STDOUT:   call void @_CGetT.C.Main.22bf4b2f6c4322a2(ptr %c.var), !dbg !165
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.5bbb7f63d621ec71(ptr %c.var), !dbg !162
// CHECK:STDOUT:   ret void, !dbg !166
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.82468d70c8d12e86:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !167 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc26_3.1.w = getelementptr inbounds nuw { i1, {} }, ptr %self, i32 0, i32 1, !dbg !169
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.bd8971525d62ef81(ptr %.loc26_3.1.w), !dbg !169
// CHECK:STDOUT:   %.loc26_3.4.v = getelementptr inbounds nuw { i1, {} }, ptr %self, i32 0, i32 0, !dbg !169
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.c96afdcab9de049c(ptr %.loc26_3.4.v), !dbg !169
// CHECK:STDOUT:   ret void, !dbg !169
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.7fca2852aa011a1c:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !171 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc26_3.1.w = getelementptr inbounds nuw { i1, {} }, ptr %self, i32 0, i32 1, !dbg !173
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.bd8971525d62ef81(ptr %.loc26_3.1.w), !dbg !173
// CHECK:STDOUT:   %.loc26_3.4.v = getelementptr inbounds nuw { i1, {} }, ptr %self, i32 0, i32 0, !dbg !173
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.c96afdcab9de049c(ptr %.loc26_3.4.v), !dbg !173
// CHECK:STDOUT:   ret void, !dbg !173
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CAccessTuple.Main(ptr sret({ i32, i32, i32 }) %return) #0 !dbg !175 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c.var = alloca { i1, { i32, i32, i32 } }, align 4, !dbg !178
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !178
// CHECK:STDOUT:   %.loc31_57.2.v = getelementptr inbounds nuw { i1, { i32, i32, i32 } }, ptr %c.var, i32 0, i32 0, !dbg !179
// CHECK:STDOUT:   %.loc31_57.4.w = getelementptr inbounds nuw { i1, { i32, i32, i32 } }, ptr %c.var, i32 0, i32 1, !dbg !179
// CHECK:STDOUT:   %tuple.elem0.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %.loc31_57.4.w, i32 0, i32 0, !dbg !180
// CHECK:STDOUT:   %tuple.elem1.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %.loc31_57.4.w, i32 0, i32 1, !dbg !180
// CHECK:STDOUT:   %tuple.elem2.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %.loc31_57.4.w, i32 0, i32 2, !dbg !180
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %c.var, ptr align 4 @C.val.97f.loc31_3.1, i64 16, i1 false), !dbg !178
// CHECK:STDOUT:   call void @_CGetT.C.Main.005fc2ec498ae6bf(ptr %return, ptr %c.var), !dbg !182
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.7b776eedfd84a6ac(ptr %c.var), !dbg !178
// CHECK:STDOUT:   ret void, !dbg !183
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.9f53a29b4522c35d:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !184 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %tuple.elem2.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %self, i32 0, i32 2, !dbg !186
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.0aefc5da911f5051(ptr %tuple.elem2.tuple.elem), !dbg !186
// CHECK:STDOUT:   %tuple.elem1.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %self, i32 0, i32 1, !dbg !186
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.0aefc5da911f5051(ptr %tuple.elem1.tuple.elem), !dbg !186
// CHECK:STDOUT:   %tuple.elem0.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %self, i32 0, i32 0, !dbg !186
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.0aefc5da911f5051(ptr %tuple.elem0.tuple.elem), !dbg !186
// CHECK:STDOUT:   ret void, !dbg !186
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.f0367c4198d74906:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !188 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc31_3.1.w = getelementptr inbounds nuw { i1, { i32, i32, i32 } }, ptr %self, i32 0, i32 1, !dbg !190
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.e9f152c5b73f0cbf(ptr %.loc31_3.1.w), !dbg !190
// CHECK:STDOUT:   %.loc31_3.4.v = getelementptr inbounds nuw { i1, { i32, i32, i32 } }, ptr %self, i32 0, i32 0, !dbg !190
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.c96afdcab9de049c(ptr %.loc31_3.4.v), !dbg !190
// CHECK:STDOUT:   ret void, !dbg !190
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.7a2f92bd9444b967:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !192 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc31_3.1.w = getelementptr inbounds nuw { i1, { i32, i32, i32 } }, ptr %self, i32 0, i32 1, !dbg !194
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.e9f152c5b73f0cbf(ptr %.loc31_3.1.w), !dbg !194
// CHECK:STDOUT:   %.loc31_3.4.v = getelementptr inbounds nuw { i1, { i32, i32, i32 } }, ptr %self, i32 0, i32 0, !dbg !194
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.c96afdcab9de049c(ptr %.loc31_3.4.v), !dbg !194
// CHECK:STDOUT:   ret void, !dbg !194
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @_CGetBool.C.Main.2aef1d52a6e1f957(ptr %self) #0 !dbg !196 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc6_16.1.v = getelementptr inbounds nuw { i1, i32 }, ptr %self, i32 0, i32 0, !dbg !201
// CHECK:STDOUT:   %.loc6_16.2 = load i8, ptr %.loc6_16.1.v, align 1, !dbg !201
// CHECK:STDOUT:   %.loc6_16.21 = trunc i8 %.loc6_16.2 to i1, !dbg !201
// CHECK:STDOUT:   ret i1 %.loc6_16.21, !dbg !202
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.ee380be3044985a2(ptr %t) #0 !dbg !204 {
// CHECK:STDOUT:   call void @"_COp.b78480f348383035:core.SubobjectDestroy.Core"(ptr %t), !dbg !208
// CHECK:STDOUT:   ret void, !dbg !209
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.0aefc5da911f5051(ptr %t) #0 !dbg !211 {
// CHECK:STDOUT:   ret void, !dbg !216
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.c96afdcab9de049c(ptr %t) #0 !dbg !218 {
// CHECK:STDOUT:   ret void, !dbg !223
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CGetT.C.Main.2aef1d52a6e1f957(ptr %self) #0 !dbg !225 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc9_16.1.w = getelementptr inbounds nuw { i1, i32 }, ptr %self, i32 0, i32 1, !dbg !230
// CHECK:STDOUT:   %.loc9_16.2 = load i32, ptr %.loc9_16.1.w, align 4, !dbg !230
// CHECK:STDOUT:   ret i32 %.loc9_16.2, !dbg !231
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CGetT.C.Main.22bf4b2f6c4322a2(ptr %self) #0 !dbg !233 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc9_16.1.w = getelementptr inbounds nuw { i1, {} }, ptr %self, i32 0, i32 1, !dbg !238
// CHECK:STDOUT:   ret void, !dbg !239
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.5bbb7f63d621ec71(ptr %t) #0 !dbg !241 {
// CHECK:STDOUT:   call void @"_COp.7fca2852aa011a1c:core.SubobjectDestroy.Core"(ptr %t), !dbg !245
// CHECK:STDOUT:   ret void, !dbg !246
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.bd8971525d62ef81(ptr %t) #0 !dbg !248 {
// CHECK:STDOUT:   ret void, !dbg !253
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CGetT.C.Main.005fc2ec498ae6bf(ptr sret({ i32, i32, i32 }) %return, ptr %self) #0 !dbg !255 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc9_16.1.w = getelementptr inbounds nuw { i1, { i32, i32, i32 } }, ptr %self, i32 0, i32 1, !dbg !260
// CHECK:STDOUT:   call void @"_COp.6753b13effab4291:Copy.Core.07edd0203c2a63a0"(ptr %return, ptr %.loc9_16.1.w), !dbg !260
// CHECK:STDOUT:   ret void, !dbg !261
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.7b776eedfd84a6ac(ptr %t) #0 !dbg !263 {
// CHECK:STDOUT:   call void @"_COp.7a2f92bd9444b967:core.SubobjectDestroy.Core"(ptr %t), !dbg !267
// CHECK:STDOUT:   ret void, !dbg !268
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.e9f152c5b73f0cbf(ptr %t) #0 !dbg !270 {
// CHECK:STDOUT:   call void @"_COp.9f53a29b4522c35d:core.SubobjectDestroy.Core"(ptr %t), !dbg !274
// CHECK:STDOUT:   ret void, !dbg !275
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp.6753b13effab4291:Copy.Core.07edd0203c2a63a0"(ptr sret({ i32, i32, i32 }) %return, ptr %self) #0 !dbg !277 {
// CHECK:STDOUT:   %tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %self, i32 0, i32 0, !dbg !282
// CHECK:STDOUT:   %tuple.elem.load = load i32, ptr %tuple.elem, align 4, !dbg !282
// CHECK:STDOUT:   %tuple.elem1 = getelementptr inbounds nuw { i32, i32, i32 }, ptr %return, i32 0, i32 0, !dbg !283
// CHECK:STDOUT:   %tuple.elem2 = getelementptr inbounds nuw { i32, i32, i32 }, ptr %self, i32 0, i32 1, !dbg !284
// CHECK:STDOUT:   %tuple.elem.load3 = load i32, ptr %tuple.elem2, align 4, !dbg !284
// CHECK:STDOUT:   %tuple.elem4 = getelementptr inbounds nuw { i32, i32, i32 }, ptr %return, i32 0, i32 1, !dbg !283
// CHECK:STDOUT:   %tuple.elem5 = getelementptr inbounds nuw { i32, i32, i32 }, ptr %self, i32 0, i32 2, !dbg !285
// CHECK:STDOUT:   %tuple.elem.load6 = load i32, ptr %tuple.elem5, align 4, !dbg !285
// CHECK:STDOUT:   %tuple.elem7 = getelementptr inbounds nuw { i32, i32, i32 }, ptr %return, i32 0, i32 2, !dbg !283
// CHECK:STDOUT:   store i32 %tuple.elem.load, ptr %tuple.elem1, align 4, !dbg !283
// CHECK:STDOUT:   store i32 %tuple.elem.load3, ptr %tuple.elem4, align 4, !dbg !283
// CHECK:STDOUT:   store i32 %tuple.elem.load6, ptr %tuple.elem7, align 4, !dbg !283
// CHECK:STDOUT:   ret void, !dbg !286
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.ee380be3044985a2, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.0aefc5da911f5051, { 4, 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.c96afdcab9de049c, { 5, 4, 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.bd8971525d62ef81, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.e9f152c5b73f0cbf, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!124}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !11 = !DIFile(filename: "min_prelude/parts/copy.carbon", directory: "")
// CHECK:STDOUT: !28 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !38 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !39 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !40 = !{!39}
// CHECK:STDOUT: !41 = !DISubroutineType(types: !40)
// CHECK:STDOUT: !109 = !{!39, !39}
// CHECK:STDOUT: !110 = !DISubroutineType(types: !109)
// CHECK:STDOUT: !123 = !DIFile(filename: "access.carbon", directory: "")
// CHECK:STDOUT: !124 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !123, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !125 = !{null, !39}
// CHECK:STDOUT: !126 = !DISubroutineType(types: !125)
// CHECK:STDOUT: !127 = !{!38}
// CHECK:STDOUT: !128 = !DISubroutineType(types: !127)
// CHECK:STDOUT: !129 = !{null}
// CHECK:STDOUT: !130 = !DISubroutineType(types: !129)
// CHECK:STDOUT: !131 = distinct !DISubprogram(name: "AccessBool", linkageName: "_CAccessBool.Main", scope: null, file: !123, line: 15, type: !41, spFlags: DISPFlagDefinition, unit: !124)
// CHECK:STDOUT: !134 = !DILocation(line: 16, column: 3, scope: !131)
// CHECK:STDOUT: !135 = !DILocation(line: 16, column: 19, scope: !131)
// CHECK:STDOUT: !137 = !DILocation(line: 17, column: 10, scope: !131)
// CHECK:STDOUT: !138 = !DILocation(line: 17, column: 3, scope: !131)
// CHECK:STDOUT: !139 = distinct !DISubprogram(name: "Op", linkageName: "_COp.7dca9c77f1689903:core.SubobjectDestroy.Core", scope: null, file: !123, line: 16, type: !126, spFlags: DISPFlagDefinition, unit: !124, retainedNodes: !144)
// CHECK:STDOUT: !140 = !DILocalVariable(arg: 1, scope: !139, type: !39)
// CHECK:STDOUT: !141 = !DILocation(line: 16, column: 3, scope: !139)
// CHECK:STDOUT: !142 = !{null, !38}
// CHECK:STDOUT: !143 = !DISubroutineType(types: !142)
// CHECK:STDOUT: !144 = !{!140}
// CHECK:STDOUT: !145 = distinct !DISubprogram(name: "Op", linkageName: "_COp.b78480f348383035:core.SubobjectDestroy.Core", scope: null, file: !123, line: 16, type: !126, spFlags: DISPFlagDefinition, unit: !124, retainedNodes: !148)
// CHECK:STDOUT: !146 = !DILocalVariable(arg: 1, scope: !145, type: !39)
// CHECK:STDOUT: !147 = !DILocation(line: 16, column: 3, scope: !145)
// CHECK:STDOUT: !148 = !{!146}
// CHECK:STDOUT: !149 = distinct !DISubprogram(name: "AccessInt", linkageName: "_CAccessInt.Main", scope: null, file: !123, line: 20, type: !128, spFlags: DISPFlagDefinition, unit: !124)
// CHECK:STDOUT: !152 = !DILocation(line: 21, column: 3, scope: !149)
// CHECK:STDOUT: !153 = !DILocation(line: 21, column: 19, scope: !149)
// CHECK:STDOUT: !155 = !DILocation(line: 22, column: 10, scope: !149)
// CHECK:STDOUT: !156 = !{!38, !39}
// CHECK:STDOUT: !157 = !DISubroutineType(types: !156)
// CHECK:STDOUT: !158 = !DILocation(line: 22, column: 3, scope: !149)
// CHECK:STDOUT: !159 = distinct !DISubprogram(name: "AccessEmpty", linkageName: "_CAccessEmpty.Main", scope: null, file: !123, line: 25, type: !130, spFlags: DISPFlagDefinition, unit: !124)
// CHECK:STDOUT: !162 = !DILocation(line: 26, column: 3, scope: !159)
// CHECK:STDOUT: !163 = !DILocation(line: 26, column: 18, scope: !159)
// CHECK:STDOUT: !165 = !DILocation(line: 27, column: 10, scope: !159)
// CHECK:STDOUT: !166 = !DILocation(line: 27, column: 3, scope: !159)
// CHECK:STDOUT: !167 = distinct !DISubprogram(name: "Op", linkageName: "_COp.82468d70c8d12e86:core.SubobjectDestroy.Core", scope: null, file: !123, line: 26, type: !126, spFlags: DISPFlagDefinition, unit: !124, retainedNodes: !170)
// CHECK:STDOUT: !168 = !DILocalVariable(arg: 1, scope: !167, type: !39)
// CHECK:STDOUT: !169 = !DILocation(line: 26, column: 3, scope: !167)
// CHECK:STDOUT: !170 = !{!168}
// CHECK:STDOUT: !171 = distinct !DISubprogram(name: "Op", linkageName: "_COp.7fca2852aa011a1c:core.SubobjectDestroy.Core", scope: null, file: !123, line: 26, type: !126, spFlags: DISPFlagDefinition, unit: !124, retainedNodes: !174)
// CHECK:STDOUT: !172 = !DILocalVariable(arg: 1, scope: !171, type: !39)
// CHECK:STDOUT: !173 = !DILocation(line: 26, column: 3, scope: !171)
// CHECK:STDOUT: !174 = !{!172}
// CHECK:STDOUT: !175 = distinct !DISubprogram(name: "AccessTuple", linkageName: "_CAccessTuple.Main", scope: null, file: !123, line: 30, type: !41, spFlags: DISPFlagDefinition, unit: !124)
// CHECK:STDOUT: !178 = !DILocation(line: 31, column: 3, scope: !175)
// CHECK:STDOUT: !179 = !DILocation(line: 31, column: 31, scope: !175)
// CHECK:STDOUT: !180 = !DILocation(line: 31, column: 48, scope: !175)
// CHECK:STDOUT: !182 = !DILocation(line: 32, column: 10, scope: !175)
// CHECK:STDOUT: !183 = !DILocation(line: 32, column: 3, scope: !175)
// CHECK:STDOUT: !184 = distinct !DISubprogram(name: "Op", linkageName: "_COp.9f53a29b4522c35d:core.SubobjectDestroy.Core", scope: null, file: !123, line: 31, type: !126, spFlags: DISPFlagDefinition, unit: !124, retainedNodes: !187)
// CHECK:STDOUT: !185 = !DILocalVariable(arg: 1, scope: !184, type: !39)
// CHECK:STDOUT: !186 = !DILocation(line: 31, column: 3, scope: !184)
// CHECK:STDOUT: !187 = !{!185}
// CHECK:STDOUT: !188 = distinct !DISubprogram(name: "Op", linkageName: "_COp.f0367c4198d74906:core.SubobjectDestroy.Core", scope: null, file: !123, line: 31, type: !126, spFlags: DISPFlagDefinition, unit: !124, retainedNodes: !191)
// CHECK:STDOUT: !189 = !DILocalVariable(arg: 1, scope: !188, type: !39)
// CHECK:STDOUT: !190 = !DILocation(line: 31, column: 3, scope: !188)
// CHECK:STDOUT: !191 = !{!189}
// CHECK:STDOUT: !192 = distinct !DISubprogram(name: "Op", linkageName: "_COp.7a2f92bd9444b967:core.SubobjectDestroy.Core", scope: null, file: !123, line: 31, type: !126, spFlags: DISPFlagDefinition, unit: !124, retainedNodes: !195)
// CHECK:STDOUT: !193 = !DILocalVariable(arg: 1, scope: !192, type: !39)
// CHECK:STDOUT: !194 = !DILocation(line: 31, column: 3, scope: !192)
// CHECK:STDOUT: !195 = !{!193}
// CHECK:STDOUT: !196 = distinct !DISubprogram(name: "GetBool", linkageName: "_CGetBool.C.Main.2aef1d52a6e1f957", scope: null, file: !123, line: 5, type: !110, spFlags: DISPFlagDefinition, unit: !124, retainedNodes: !203)
// CHECK:STDOUT: !197 = !DILocalVariable(arg: 1, scope: !196, type: !39)
// CHECK:STDOUT: !201 = !DILocation(line: 6, column: 12, scope: !196)
// CHECK:STDOUT: !202 = !DILocation(line: 6, column: 5, scope: !196)
// CHECK:STDOUT: !203 = !{!197}
// CHECK:STDOUT: !204 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.ee380be3044985a2", scope: null, file: !28, line: 26, type: !126, spFlags: DISPFlagDefinition, unit: !124, retainedNodes: !210)
// CHECK:STDOUT: !205 = !DILocalVariable(arg: 1, scope: !204, type: !39)
// CHECK:STDOUT: !208 = !DILocation(line: 28, column: 5, scope: !204)
// CHECK:STDOUT: !209 = !DILocation(line: 26, column: 1, scope: !204)
// CHECK:STDOUT: !210 = !{!205}
// CHECK:STDOUT: !211 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.0aefc5da911f5051", scope: null, file: !28, line: 26, type: !143, spFlags: DISPFlagDefinition, unit: !124, retainedNodes: !217)
// CHECK:STDOUT: !212 = !DILocalVariable(arg: 1, scope: !211, type: !38)
// CHECK:STDOUT: !216 = !DILocation(line: 26, column: 1, scope: !211)
// CHECK:STDOUT: !217 = !{!212}
// CHECK:STDOUT: !218 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.c96afdcab9de049c", scope: null, file: !28, line: 26, type: !126, spFlags: DISPFlagDefinition, unit: !124, retainedNodes: !224)
// CHECK:STDOUT: !219 = !DILocalVariable(arg: 1, scope: !218, type: !39)
// CHECK:STDOUT: !223 = !DILocation(line: 26, column: 1, scope: !218)
// CHECK:STDOUT: !224 = !{!219}
// CHECK:STDOUT: !225 = distinct !DISubprogram(name: "GetT", linkageName: "_CGetT.C.Main.2aef1d52a6e1f957", scope: null, file: !123, line: 8, type: !157, spFlags: DISPFlagDefinition, unit: !124, retainedNodes: !232)
// CHECK:STDOUT: !226 = !DILocalVariable(arg: 1, scope: !225, type: !39)
// CHECK:STDOUT: !230 = !DILocation(line: 9, column: 12, scope: !225)
// CHECK:STDOUT: !231 = !DILocation(line: 9, column: 5, scope: !225)
// CHECK:STDOUT: !232 = !{!226}
// CHECK:STDOUT: !233 = distinct !DISubprogram(name: "GetT", linkageName: "_CGetT.C.Main.22bf4b2f6c4322a2", scope: null, file: !123, line: 8, type: !126, spFlags: DISPFlagDefinition, unit: !124, retainedNodes: !240)
// CHECK:STDOUT: !234 = !DILocalVariable(arg: 1, scope: !233, type: !39)
// CHECK:STDOUT: !238 = !DILocation(line: 9, column: 12, scope: !233)
// CHECK:STDOUT: !239 = !DILocation(line: 9, column: 5, scope: !233)
// CHECK:STDOUT: !240 = !{!234}
// CHECK:STDOUT: !241 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.5bbb7f63d621ec71", scope: null, file: !28, line: 26, type: !126, spFlags: DISPFlagDefinition, unit: !124, retainedNodes: !247)
// CHECK:STDOUT: !242 = !DILocalVariable(arg: 1, scope: !241, type: !39)
// CHECK:STDOUT: !245 = !DILocation(line: 28, column: 5, scope: !241)
// CHECK:STDOUT: !246 = !DILocation(line: 26, column: 1, scope: !241)
// CHECK:STDOUT: !247 = !{!242}
// CHECK:STDOUT: !248 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.bd8971525d62ef81", scope: null, file: !28, line: 26, type: !126, spFlags: DISPFlagDefinition, unit: !124, retainedNodes: !254)
// CHECK:STDOUT: !249 = !DILocalVariable(arg: 1, scope: !248, type: !39)
// CHECK:STDOUT: !253 = !DILocation(line: 26, column: 1, scope: !248)
// CHECK:STDOUT: !254 = !{!249}
// CHECK:STDOUT: !255 = distinct !DISubprogram(name: "GetT", linkageName: "_CGetT.C.Main.005fc2ec498ae6bf", scope: null, file: !123, line: 8, type: !110, spFlags: DISPFlagDefinition, unit: !124, retainedNodes: !262)
// CHECK:STDOUT: !256 = !DILocalVariable(arg: 1, scope: !255, type: !39)
// CHECK:STDOUT: !260 = !DILocation(line: 9, column: 12, scope: !255)
// CHECK:STDOUT: !261 = !DILocation(line: 9, column: 5, scope: !255)
// CHECK:STDOUT: !262 = !{!256}
// CHECK:STDOUT: !263 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.7b776eedfd84a6ac", scope: null, file: !28, line: 26, type: !126, spFlags: DISPFlagDefinition, unit: !124, retainedNodes: !269)
// CHECK:STDOUT: !264 = !DILocalVariable(arg: 1, scope: !263, type: !39)
// CHECK:STDOUT: !267 = !DILocation(line: 28, column: 5, scope: !263)
// CHECK:STDOUT: !268 = !DILocation(line: 26, column: 1, scope: !263)
// CHECK:STDOUT: !269 = !{!264}
// CHECK:STDOUT: !270 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.e9f152c5b73f0cbf", scope: null, file: !28, line: 26, type: !126, spFlags: DISPFlagDefinition, unit: !124, retainedNodes: !276)
// CHECK:STDOUT: !271 = !DILocalVariable(arg: 1, scope: !270, type: !39)
// CHECK:STDOUT: !274 = !DILocation(line: 28, column: 5, scope: !270)
// CHECK:STDOUT: !275 = !DILocation(line: 26, column: 1, scope: !270)
// CHECK:STDOUT: !276 = !{!271}
// CHECK:STDOUT: !277 = distinct !DISubprogram(name: "Op", linkageName: "_COp.6753b13effab4291:Copy.Core.07edd0203c2a63a0", scope: null, file: !11, line: 58, type: !110, spFlags: DISPFlagDefinition, unit: !124, retainedNodes: !287)
// CHECK:STDOUT: !278 = !DILocalVariable(arg: 1, scope: !277, type: !39)
// CHECK:STDOUT: !282 = !DILocation(line: 59, column: 13, scope: !277)
// CHECK:STDOUT: !283 = !DILocation(line: 59, column: 12, scope: !277)
// CHECK:STDOUT: !284 = !DILocation(line: 59, column: 26, scope: !277)
// CHECK:STDOUT: !285 = !DILocation(line: 59, column: 39, scope: !277)
// CHECK:STDOUT: !286 = !DILocation(line: 59, column: 5, scope: !277)
// CHECK:STDOUT: !287 = !{!278}
// CHECK:STDOUT:
