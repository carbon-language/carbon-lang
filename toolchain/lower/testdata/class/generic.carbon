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
// CHECK:STDOUT: !llvm.dbg.cu = !{!33}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !32 = !DIFile(filename: "classes.carbon", directory: "")
// CHECK:STDOUT: !33 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !32, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'create.carbon'
// CHECK:STDOUT: source_filename = "create.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @Derived.val.loc7_36 = internal constant { { i32 }, i32 } { { i32 } { i32 1 }, i32 2 }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCreateDerived.Create(ptr sret({ { i32 }, i32 }) %return) #0 !dbg !40 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_35.2.base = getelementptr inbounds nuw { { i32 }, i32 }, ptr %return, i32 0, i32 0, !dbg !43
// CHECK:STDOUT:   %.loc7_26.3.b = getelementptr inbounds nuw { i32 }, ptr %.loc7_35.2.base, i32 0, i32 0, !dbg !44
// CHECK:STDOUT:   %.loc7_35.6.d = getelementptr inbounds nuw { { i32 }, i32 }, ptr %return, i32 0, i32 1, !dbg !43
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %return, ptr align 4 @Derived.val.loc7_36, i64 8, i1 false), !dbg !45
// CHECK:STDOUT:   ret void, !dbg !45
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCreateAdapter.Create(ptr sret({ { i32 }, i32 }) %return) #0 !dbg !46 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CCreateDerived.Create(ptr %return), !dbg !49
// CHECK:STDOUT:   ret void, !dbg !50
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!35}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !34 = !DIFile(filename: "create.carbon", directory: "")
// CHECK:STDOUT: !35 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !34, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !37 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !38 = !{!37}
// CHECK:STDOUT: !39 = !DISubroutineType(types: !38)
// CHECK:STDOUT: !40 = distinct !DISubprogram(name: "CreateDerived", linkageName: "_CCreateDerived.Create", scope: null, file: !34, line: 6, type: !39, spFlags: DISPFlagDefinition, unit: !35)
// CHECK:STDOUT: !43 = !DILocation(line: 7, column: 10, scope: !40)
// CHECK:STDOUT: !44 = !DILocation(line: 7, column: 19, scope: !40)
// CHECK:STDOUT: !45 = !DILocation(line: 7, column: 3, scope: !40)
// CHECK:STDOUT: !46 = distinct !DISubprogram(name: "CreateAdapter", linkageName: "_CCreateAdapter.Create", scope: null, file: !34, line: 10, type: !39, spFlags: DISPFlagDefinition, unit: !35)
// CHECK:STDOUT: !49 = !DILocation(line: 11, column: 10, scope: !46)
// CHECK:STDOUT: !50 = !DILocation(line: 11, column: 3, scope: !46)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'create_generic.carbon'
// CHECK:STDOUT: source_filename = "create_generic.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @tuple.102.loc22_28.6 = internal constant { i32, i32 } { i32 1, i32 2 }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CInts.Main(ptr sret({ i32, i32 }) %return) #0 !dbg !53 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CMake.Main.4d6ceefeb7f772f4(ptr %return, i32 1, i32 2), !dbg !56
// CHECK:STDOUT:   ret void, !dbg !59
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CEmpty.Main(ptr sret({ {}, {} }) %return) #0 !dbg !60 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CMake.Main.22bf4b2f6c4322a2(ptr %return), !dbg !63
// CHECK:STDOUT:   ret void, !dbg !64
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTuples.Main(ptr sret({ { i32, i32 }, { i32, i32 } }) %return) #0 !dbg !65 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CMake.Main.878c3517da47a752(ptr %return, ptr @tuple.102.loc22_28.6, ptr @tuple.102.loc22_28.6), !dbg !72
// CHECK:STDOUT:   ret void, !dbg !75
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CMake.Main.4d6ceefeb7f772f4(ptr sret({ i32, i32 }) %return, i32 %x, i32 %y) #0 !dbg !76 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc10_25.2.x = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 0, !dbg !85
// CHECK:STDOUT:   store i32 %x, ptr %.loc10_25.2.x, align 4, !dbg !85
// CHECK:STDOUT:   %.loc10_25.4.y = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 1, !dbg !85
// CHECK:STDOUT:   store i32 %y, ptr %.loc10_25.4.y, align 4, !dbg !85
// CHECK:STDOUT:   ret void, !dbg !86
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CMake.Main.22bf4b2f6c4322a2(ptr sret({ {}, {} }) %return) #0 !dbg !88 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc10_25.2.x = getelementptr inbounds nuw { {}, {} }, ptr %return, i32 0, i32 0, !dbg !95
// CHECK:STDOUT:   %.loc10_25.4.y = getelementptr inbounds nuw { {}, {} }, ptr %return, i32 0, i32 1, !dbg !95
// CHECK:STDOUT:   ret void, !dbg !96
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CMake.Main.878c3517da47a752(ptr sret({ { i32, i32 }, { i32, i32 } }) %return, ptr %x, ptr %y) #0 !dbg !97 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc10_25.2.x = getelementptr inbounds nuw { { i32, i32 }, { i32, i32 } }, ptr %return, i32 0, i32 0, !dbg !106
// CHECK:STDOUT:   call void @"_COp.4e4edddc7580c79b:Copy.Core.8608f3c6f62d80cb"(ptr %.loc10_25.2.x, ptr %x), !dbg !104
// CHECK:STDOUT:   %.loc10_25.4.y = getelementptr inbounds nuw { { i32, i32 }, { i32, i32 } }, ptr %return, i32 0, i32 1, !dbg !106
// CHECK:STDOUT:   call void @"_COp.4e4edddc7580c79b:Copy.Core.8608f3c6f62d80cb"(ptr %.loc10_25.4.y, ptr %y), !dbg !105
// CHECK:STDOUT:   ret void, !dbg !109
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp.4e4edddc7580c79b:Copy.Core.8608f3c6f62d80cb"(ptr sret({ i32, i32 }) %return, ptr %self) #0 !dbg !111 {
// CHECK:STDOUT:   %tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 0, !dbg !116
// CHECK:STDOUT:   %tuple.elem.load = load i32, ptr %tuple.elem, align 4, !dbg !116
// CHECK:STDOUT:   %tuple.elem1 = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 0, !dbg !117
// CHECK:STDOUT:   %tuple.elem2 = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 1, !dbg !118
// CHECK:STDOUT:   %tuple.elem.load3 = load i32, ptr %tuple.elem2, align 4, !dbg !118
// CHECK:STDOUT:   %tuple.elem4 = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 1, !dbg !117
// CHECK:STDOUT:   store i32 %tuple.elem.load, ptr %tuple.elem1, align 4, !dbg !117
// CHECK:STDOUT:   store i32 %tuple.elem.load3, ptr %tuple.elem4, align 4, !dbg !117
// CHECK:STDOUT:   ret void, !dbg !119
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @"_COp.4e4edddc7580c79b:Copy.Core.8608f3c6f62d80cb", { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!52}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !9 = !DIFile(filename: "min_prelude/parts/copy.carbon", directory: "")
// CHECK:STDOUT: !36 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !37 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !38 = !{!37}
// CHECK:STDOUT: !39 = !DISubroutineType(types: !38)
// CHECK:STDOUT: !51 = !DIFile(filename: "create_generic.carbon", directory: "")
// CHECK:STDOUT: !52 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !51, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !53 = distinct !DISubprogram(name: "Ints", linkageName: "_CInts.Main", scope: null, file: !51, line: 13, type: !39, spFlags: DISPFlagDefinition, unit: !52)
// CHECK:STDOUT: !56 = !DILocation(line: 14, column: 10, scope: !53)
// CHECK:STDOUT: !57 = !{!37, !36, !36}
// CHECK:STDOUT: !58 = !DISubroutineType(types: !57)
// CHECK:STDOUT: !59 = !DILocation(line: 14, column: 3, scope: !53)
// CHECK:STDOUT: !60 = distinct !DISubprogram(name: "Empty", linkageName: "_CEmpty.Main", scope: null, file: !51, line: 17, type: !39, spFlags: DISPFlagDefinition, unit: !52)
// CHECK:STDOUT: !63 = !DILocation(line: 18, column: 10, scope: !60)
// CHECK:STDOUT: !64 = !DILocation(line: 18, column: 3, scope: !60)
// CHECK:STDOUT: !65 = distinct !DISubprogram(name: "Tuples", linkageName: "_CTuples.Main", scope: null, file: !51, line: 21, type: !39, spFlags: DISPFlagDefinition, unit: !52)
// CHECK:STDOUT: !72 = !DILocation(line: 23, column: 10, scope: !65)
// CHECK:STDOUT: !73 = !{!37, !37, !37}
// CHECK:STDOUT: !74 = !DISubroutineType(types: !73)
// CHECK:STDOUT: !75 = !DILocation(line: 23, column: 3, scope: !65)
// CHECK:STDOUT: !76 = distinct !DISubprogram(name: "Make", linkageName: "_CMake.Main.4d6ceefeb7f772f4", scope: null, file: !51, line: 9, type: !58, spFlags: DISPFlagDefinition, unit: !52, retainedNodes: !87)
// CHECK:STDOUT: !77 = !DILocalVariable(arg: 1, scope: !76, type: !36)
// CHECK:STDOUT: !78 = !DILocalVariable(arg: 2, scope: !76, type: !36)
// CHECK:STDOUT: !85 = !DILocation(line: 10, column: 10, scope: !76)
// CHECK:STDOUT: !86 = !DILocation(line: 10, column: 3, scope: !76)
// CHECK:STDOUT: !87 = !{!77, !78}
// CHECK:STDOUT: !88 = distinct !DISubprogram(name: "Make", linkageName: "_CMake.Main.22bf4b2f6c4322a2", scope: null, file: !51, line: 9, type: !39, spFlags: DISPFlagDefinition, unit: !52)
// CHECK:STDOUT: !95 = !DILocation(line: 10, column: 10, scope: !88)
// CHECK:STDOUT: !96 = !DILocation(line: 10, column: 3, scope: !88)
// CHECK:STDOUT: !97 = distinct !DISubprogram(name: "Make", linkageName: "_CMake.Main.878c3517da47a752", scope: null, file: !51, line: 9, type: !74, spFlags: DISPFlagDefinition, unit: !52, retainedNodes: !110)
// CHECK:STDOUT: !98 = !DILocalVariable(arg: 1, scope: !97, type: !37)
// CHECK:STDOUT: !99 = !DILocalVariable(arg: 2, scope: !97, type: !37)
// CHECK:STDOUT: !104 = !DILocation(line: 10, column: 16, scope: !97)
// CHECK:STDOUT: !105 = !DILocation(line: 10, column: 24, scope: !97)
// CHECK:STDOUT: !106 = !DILocation(line: 10, column: 10, scope: !97)
// CHECK:STDOUT: !107 = !{!37, !37}
// CHECK:STDOUT: !108 = !DISubroutineType(types: !107)
// CHECK:STDOUT: !109 = !DILocation(line: 10, column: 3, scope: !97)
// CHECK:STDOUT: !110 = !{!98, !99}
// CHECK:STDOUT: !111 = distinct !DISubprogram(name: "Op", linkageName: "_COp.4e4edddc7580c79b:Copy.Core.8608f3c6f62d80cb", scope: null, file: !9, line: 52, type: !108, spFlags: DISPFlagDefinition, unit: !52, retainedNodes: !120)
// CHECK:STDOUT: !112 = !DILocalVariable(arg: 1, scope: !111, type: !37)
// CHECK:STDOUT: !116 = !DILocation(line: 53, column: 13, scope: !111)
// CHECK:STDOUT: !117 = !DILocation(line: 53, column: 12, scope: !111)
// CHECK:STDOUT: !118 = !DILocation(line: 53, column: 26, scope: !111)
// CHECK:STDOUT: !119 = !DILocation(line: 53, column: 5, scope: !111)
// CHECK:STDOUT: !120 = !{!112}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'access.carbon'
// CHECK:STDOUT: source_filename = "access.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @C.val.29b.loc16_3 = internal constant { i1, i32 } { i1 true, i32 0 }
// CHECK:STDOUT: @C.val.0ea.loc26_3 = internal constant { i1, {} } { i1 true, {} zeroinitializer }
// CHECK:STDOUT: @C.val.727.loc31_3 = internal constant { i1, { i32, i32, i32 } } { i1 true, { i32, i32, i32 } { i32 1, i32 2, i32 3 } }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CAccessBool.Main() #0 !dbg !131 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c.var = alloca { i1, i32 }, align 4, !dbg !134
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !134
// CHECK:STDOUT:   %.loc16_37.2.v = getelementptr inbounds nuw { i1, i32 }, ptr %c.var, i32 0, i32 0, !dbg !135
// CHECK:STDOUT:   %.loc16_37.5.w = getelementptr inbounds nuw { i1, i32 }, ptr %c.var, i32 0, i32 1, !dbg !135
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %c.var, ptr align 4 @C.val.29b.loc16_3, i64 8, i1 false), !dbg !134
// CHECK:STDOUT:   %C.GetBool.call = call i1 @_CGetBool.C.Main.4d6ceefeb7f772f4(ptr %c.var), !dbg !137
// CHECK:STDOUT:   call void @"_COp.a0d0786a20116357:core.Destroy.Core"(ptr %c.var), !dbg !134
// CHECK:STDOUT:   ret i1 %C.GetBool.call, !dbg !138
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.70db2d92ffddbef7:core.Destroy.Core"(ptr %self) #0 !dbg !139 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !141
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.e72c587127f1eaf2:core.Destroy.Core"(ptr %self) #0 !dbg !143 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !145
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.a0d0786a20116357:core.Destroy.Core"(ptr %self) #0 !dbg !147 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !149
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CAccessInt.Main() #0 !dbg !151 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c.var = alloca { i1, i32 }, align 4, !dbg !154
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !154
// CHECK:STDOUT:   %.loc21_37.2.v = getelementptr inbounds nuw { i1, i32 }, ptr %c.var, i32 0, i32 0, !dbg !155
// CHECK:STDOUT:   %.loc21_37.5.w = getelementptr inbounds nuw { i1, i32 }, ptr %c.var, i32 0, i32 1, !dbg !155
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %c.var, ptr align 4 @C.val.29b.loc16_3, i64 8, i1 false), !dbg !154
// CHECK:STDOUT:   %C.GetT.call = call i32 @_CGetT.C.Main.4d6ceefeb7f772f4(ptr %c.var), !dbg !157
// CHECK:STDOUT:   call void @"_COp.a0d0786a20116357:core.Destroy.Core"(ptr %c.var), !dbg !154
// CHECK:STDOUT:   ret i32 %C.GetT.call, !dbg !160
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CAccessEmpty.Main() #0 !dbg !161 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c.var = alloca { i1, {} }, align 1, !dbg !164
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !164
// CHECK:STDOUT:   %.loc26_37.2.v = getelementptr inbounds nuw { i1, {} }, ptr %c.var, i32 0, i32 0, !dbg !165
// CHECK:STDOUT:   %.loc26_37.4.w = getelementptr inbounds nuw { i1, {} }, ptr %c.var, i32 0, i32 1, !dbg !165
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %c.var, ptr align 1 @C.val.0ea.loc26_3, i64 1, i1 false), !dbg !164
// CHECK:STDOUT:   call void @_CGetT.C.Main.22bf4b2f6c4322a2(ptr %c.var), !dbg !167
// CHECK:STDOUT:   call void @"_COp.b215ecea30cf644d:core.Destroy.Core"(ptr %c.var), !dbg !164
// CHECK:STDOUT:   ret void, !dbg !168
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.e76d4e9768eb659e:core.Destroy.Core"(ptr %self) #0 !dbg !169 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !171
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.b215ecea30cf644d:core.Destroy.Core"(ptr %self) #0 !dbg !173 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !175
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CAccessTuple.Main(ptr sret({ i32, i32, i32 }) %return) #0 !dbg !177 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c.var = alloca { i1, { i32, i32, i32 } }, align 4, !dbg !180
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !180
// CHECK:STDOUT:   %.loc31_57.2.v = getelementptr inbounds nuw { i1, { i32, i32, i32 } }, ptr %c.var, i32 0, i32 0, !dbg !181
// CHECK:STDOUT:   %.loc31_57.4.w = getelementptr inbounds nuw { i1, { i32, i32, i32 } }, ptr %c.var, i32 0, i32 1, !dbg !181
// CHECK:STDOUT:   %tuple.elem0.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %.loc31_57.4.w, i32 0, i32 0, !dbg !182
// CHECK:STDOUT:   %tuple.elem1.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %.loc31_57.4.w, i32 0, i32 1, !dbg !182
// CHECK:STDOUT:   %tuple.elem2.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %.loc31_57.4.w, i32 0, i32 2, !dbg !182
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %c.var, ptr align 4 @C.val.727.loc31_3, i64 16, i1 false), !dbg !180
// CHECK:STDOUT:   call void @_CGetT.C.Main.03220e6718a91584(ptr %return, ptr %c.var), !dbg !184
// CHECK:STDOUT:   call void @"_COp.d1fe91796c0f0ec3:core.Destroy.Core"(ptr %c.var), !dbg !180
// CHECK:STDOUT:   ret void, !dbg !185
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.891f32d238880255:core.Destroy.Core"(ptr %self) #0 !dbg !186 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !188
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.f23404d5ba6ed9a1:core.Destroy.Core"(ptr %self) #0 !dbg !190 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !192
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.d1fe91796c0f0ec3:core.Destroy.Core"(ptr %self) #0 !dbg !194 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !196
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @_CGetBool.C.Main.4d6ceefeb7f772f4(ptr %self) #0 !dbg !198 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc6_16.1.v = getelementptr inbounds nuw { i1, i32 }, ptr %self, i32 0, i32 0, !dbg !203
// CHECK:STDOUT:   %.loc6_16.2 = load i8, ptr %.loc6_16.1.v, align 1, !dbg !203
// CHECK:STDOUT:   %.loc6_16.21 = trunc i8 %.loc6_16.2 to i1, !dbg !203
// CHECK:STDOUT:   ret i1 %.loc6_16.21, !dbg !204
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CGetT.C.Main.4d6ceefeb7f772f4(ptr %self) #0 !dbg !206 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc9_16.1.w = getelementptr inbounds nuw { i1, i32 }, ptr %self, i32 0, i32 1, !dbg !211
// CHECK:STDOUT:   %.loc9_16.2 = load i32, ptr %.loc9_16.1.w, align 4, !dbg !211
// CHECK:STDOUT:   ret i32 %.loc9_16.2, !dbg !212
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CGetT.C.Main.22bf4b2f6c4322a2(ptr %self) #0 !dbg !214 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc9_16.1.w = getelementptr inbounds nuw { i1, {} }, ptr %self, i32 0, i32 1, !dbg !219
// CHECK:STDOUT:   ret void, !dbg !220
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CGetT.C.Main.03220e6718a91584(ptr sret({ i32, i32, i32 }) %return, ptr %self) #0 !dbg !222 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc9_16.1.w = getelementptr inbounds nuw { i1, { i32, i32, i32 } }, ptr %self, i32 0, i32 1, !dbg !227
// CHECK:STDOUT:   call void @"_COp.6753b13effab4291:Copy.Core.c9f31a0e20c75302"(ptr %return, ptr %.loc9_16.1.w), !dbg !227
// CHECK:STDOUT:   ret void, !dbg !228
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp.6753b13effab4291:Copy.Core.c9f31a0e20c75302"(ptr sret({ i32, i32, i32 }) %return, ptr %self) #0 !dbg !230 {
// CHECK:STDOUT:   %tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %self, i32 0, i32 0, !dbg !235
// CHECK:STDOUT:   %tuple.elem.load = load i32, ptr %tuple.elem, align 4, !dbg !235
// CHECK:STDOUT:   %tuple.elem1 = getelementptr inbounds nuw { i32, i32, i32 }, ptr %return, i32 0, i32 0, !dbg !236
// CHECK:STDOUT:   %tuple.elem2 = getelementptr inbounds nuw { i32, i32, i32 }, ptr %self, i32 0, i32 1, !dbg !237
// CHECK:STDOUT:   %tuple.elem.load3 = load i32, ptr %tuple.elem2, align 4, !dbg !237
// CHECK:STDOUT:   %tuple.elem4 = getelementptr inbounds nuw { i32, i32, i32 }, ptr %return, i32 0, i32 1, !dbg !236
// CHECK:STDOUT:   %tuple.elem5 = getelementptr inbounds nuw { i32, i32, i32 }, ptr %self, i32 0, i32 2, !dbg !238
// CHECK:STDOUT:   %tuple.elem.load6 = load i32, ptr %tuple.elem5, align 4, !dbg !238
// CHECK:STDOUT:   %tuple.elem7 = getelementptr inbounds nuw { i32, i32, i32 }, ptr %return, i32 0, i32 2, !dbg !236
// CHECK:STDOUT:   store i32 %tuple.elem.load, ptr %tuple.elem1, align 4, !dbg !236
// CHECK:STDOUT:   store i32 %tuple.elem.load3, ptr %tuple.elem4, align 4, !dbg !236
// CHECK:STDOUT:   store i32 %tuple.elem.load6, ptr %tuple.elem7, align 4, !dbg !236
// CHECK:STDOUT:   ret void, !dbg !239
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 3, 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!122}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !9 = !DIFile(filename: "min_prelude/parts/copy.carbon", directory: "")
// CHECK:STDOUT: !36 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !37 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !38 = !{!37}
// CHECK:STDOUT: !39 = !DISubroutineType(types: !38)
// CHECK:STDOUT: !107 = !{!37, !37}
// CHECK:STDOUT: !108 = !DISubroutineType(types: !107)
// CHECK:STDOUT: !121 = !DIFile(filename: "access.carbon", directory: "")
// CHECK:STDOUT: !122 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !121, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !123 = !{null, !36}
// CHECK:STDOUT: !124 = !DISubroutineType(types: !123)
// CHECK:STDOUT: !125 = !{null, !37}
// CHECK:STDOUT: !126 = !DISubroutineType(types: !125)
// CHECK:STDOUT: !127 = !{!36}
// CHECK:STDOUT: !128 = !DISubroutineType(types: !127)
// CHECK:STDOUT: !129 = !{null}
// CHECK:STDOUT: !130 = !DISubroutineType(types: !129)
// CHECK:STDOUT: !131 = distinct !DISubprogram(name: "AccessBool", linkageName: "_CAccessBool.Main", scope: null, file: !121, line: 15, type: !39, spFlags: DISPFlagDefinition, unit: !122)
// CHECK:STDOUT: !134 = !DILocation(line: 16, column: 3, scope: !131)
// CHECK:STDOUT: !135 = !DILocation(line: 16, column: 19, scope: !131)
// CHECK:STDOUT: !137 = !DILocation(line: 17, column: 10, scope: !131)
// CHECK:STDOUT: !138 = !DILocation(line: 17, column: 3, scope: !131)
// CHECK:STDOUT: !139 = distinct !DISubprogram(name: "Op", linkageName: "_COp.70db2d92ffddbef7:core.Destroy.Core", scope: null, file: !121, line: 16, type: !124, spFlags: DISPFlagDefinition, unit: !122, retainedNodes: !142)
// CHECK:STDOUT: !140 = !DILocalVariable(arg: 1, scope: !139, type: !36)
// CHECK:STDOUT: !141 = !DILocation(line: 16, column: 3, scope: !139)
// CHECK:STDOUT: !142 = !{!140}
// CHECK:STDOUT: !143 = distinct !DISubprogram(name: "Op", linkageName: "_COp.e72c587127f1eaf2:core.Destroy.Core", scope: null, file: !121, line: 16, type: !126, spFlags: DISPFlagDefinition, unit: !122, retainedNodes: !146)
// CHECK:STDOUT: !144 = !DILocalVariable(arg: 1, scope: !143, type: !37)
// CHECK:STDOUT: !145 = !DILocation(line: 16, column: 3, scope: !143)
// CHECK:STDOUT: !146 = !{!144}
// CHECK:STDOUT: !147 = distinct !DISubprogram(name: "Op", linkageName: "_COp.a0d0786a20116357:core.Destroy.Core", scope: null, file: !121, line: 16, type: !126, spFlags: DISPFlagDefinition, unit: !122, retainedNodes: !150)
// CHECK:STDOUT: !148 = !DILocalVariable(arg: 1, scope: !147, type: !37)
// CHECK:STDOUT: !149 = !DILocation(line: 16, column: 3, scope: !147)
// CHECK:STDOUT: !150 = !{!148}
// CHECK:STDOUT: !151 = distinct !DISubprogram(name: "AccessInt", linkageName: "_CAccessInt.Main", scope: null, file: !121, line: 20, type: !128, spFlags: DISPFlagDefinition, unit: !122)
// CHECK:STDOUT: !154 = !DILocation(line: 21, column: 3, scope: !151)
// CHECK:STDOUT: !155 = !DILocation(line: 21, column: 19, scope: !151)
// CHECK:STDOUT: !157 = !DILocation(line: 22, column: 10, scope: !151)
// CHECK:STDOUT: !158 = !{!36, !37}
// CHECK:STDOUT: !159 = !DISubroutineType(types: !158)
// CHECK:STDOUT: !160 = !DILocation(line: 22, column: 3, scope: !151)
// CHECK:STDOUT: !161 = distinct !DISubprogram(name: "AccessEmpty", linkageName: "_CAccessEmpty.Main", scope: null, file: !121, line: 25, type: !130, spFlags: DISPFlagDefinition, unit: !122)
// CHECK:STDOUT: !164 = !DILocation(line: 26, column: 3, scope: !161)
// CHECK:STDOUT: !165 = !DILocation(line: 26, column: 18, scope: !161)
// CHECK:STDOUT: !167 = !DILocation(line: 27, column: 10, scope: !161)
// CHECK:STDOUT: !168 = !DILocation(line: 27, column: 3, scope: !161)
// CHECK:STDOUT: !169 = distinct !DISubprogram(name: "Op", linkageName: "_COp.e76d4e9768eb659e:core.Destroy.Core", scope: null, file: !121, line: 26, type: !126, spFlags: DISPFlagDefinition, unit: !122, retainedNodes: !172)
// CHECK:STDOUT: !170 = !DILocalVariable(arg: 1, scope: !169, type: !37)
// CHECK:STDOUT: !171 = !DILocation(line: 26, column: 3, scope: !169)
// CHECK:STDOUT: !172 = !{!170}
// CHECK:STDOUT: !173 = distinct !DISubprogram(name: "Op", linkageName: "_COp.b215ecea30cf644d:core.Destroy.Core", scope: null, file: !121, line: 26, type: !126, spFlags: DISPFlagDefinition, unit: !122, retainedNodes: !176)
// CHECK:STDOUT: !174 = !DILocalVariable(arg: 1, scope: !173, type: !37)
// CHECK:STDOUT: !175 = !DILocation(line: 26, column: 3, scope: !173)
// CHECK:STDOUT: !176 = !{!174}
// CHECK:STDOUT: !177 = distinct !DISubprogram(name: "AccessTuple", linkageName: "_CAccessTuple.Main", scope: null, file: !121, line: 30, type: !39, spFlags: DISPFlagDefinition, unit: !122)
// CHECK:STDOUT: !180 = !DILocation(line: 31, column: 3, scope: !177)
// CHECK:STDOUT: !181 = !DILocation(line: 31, column: 31, scope: !177)
// CHECK:STDOUT: !182 = !DILocation(line: 31, column: 48, scope: !177)
// CHECK:STDOUT: !184 = !DILocation(line: 32, column: 10, scope: !177)
// CHECK:STDOUT: !185 = !DILocation(line: 32, column: 3, scope: !177)
// CHECK:STDOUT: !186 = distinct !DISubprogram(name: "Op", linkageName: "_COp.891f32d238880255:core.Destroy.Core", scope: null, file: !121, line: 31, type: !126, spFlags: DISPFlagDefinition, unit: !122, retainedNodes: !189)
// CHECK:STDOUT: !187 = !DILocalVariable(arg: 1, scope: !186, type: !37)
// CHECK:STDOUT: !188 = !DILocation(line: 31, column: 3, scope: !186)
// CHECK:STDOUT: !189 = !{!187}
// CHECK:STDOUT: !190 = distinct !DISubprogram(name: "Op", linkageName: "_COp.f23404d5ba6ed9a1:core.Destroy.Core", scope: null, file: !121, line: 31, type: !126, spFlags: DISPFlagDefinition, unit: !122, retainedNodes: !193)
// CHECK:STDOUT: !191 = !DILocalVariable(arg: 1, scope: !190, type: !37)
// CHECK:STDOUT: !192 = !DILocation(line: 31, column: 3, scope: !190)
// CHECK:STDOUT: !193 = !{!191}
// CHECK:STDOUT: !194 = distinct !DISubprogram(name: "Op", linkageName: "_COp.d1fe91796c0f0ec3:core.Destroy.Core", scope: null, file: !121, line: 31, type: !126, spFlags: DISPFlagDefinition, unit: !122, retainedNodes: !197)
// CHECK:STDOUT: !195 = !DILocalVariable(arg: 1, scope: !194, type: !37)
// CHECK:STDOUT: !196 = !DILocation(line: 31, column: 3, scope: !194)
// CHECK:STDOUT: !197 = !{!195}
// CHECK:STDOUT: !198 = distinct !DISubprogram(name: "GetBool", linkageName: "_CGetBool.C.Main.4d6ceefeb7f772f4", scope: null, file: !121, line: 5, type: !108, spFlags: DISPFlagDefinition, unit: !122, retainedNodes: !205)
// CHECK:STDOUT: !199 = !DILocalVariable(arg: 1, scope: !198, type: !37)
// CHECK:STDOUT: !203 = !DILocation(line: 6, column: 12, scope: !198)
// CHECK:STDOUT: !204 = !DILocation(line: 6, column: 5, scope: !198)
// CHECK:STDOUT: !205 = !{!199}
// CHECK:STDOUT: !206 = distinct !DISubprogram(name: "GetT", linkageName: "_CGetT.C.Main.4d6ceefeb7f772f4", scope: null, file: !121, line: 8, type: !159, spFlags: DISPFlagDefinition, unit: !122, retainedNodes: !213)
// CHECK:STDOUT: !207 = !DILocalVariable(arg: 1, scope: !206, type: !37)
// CHECK:STDOUT: !211 = !DILocation(line: 9, column: 12, scope: !206)
// CHECK:STDOUT: !212 = !DILocation(line: 9, column: 5, scope: !206)
// CHECK:STDOUT: !213 = !{!207}
// CHECK:STDOUT: !214 = distinct !DISubprogram(name: "GetT", linkageName: "_CGetT.C.Main.22bf4b2f6c4322a2", scope: null, file: !121, line: 8, type: !126, spFlags: DISPFlagDefinition, unit: !122, retainedNodes: !221)
// CHECK:STDOUT: !215 = !DILocalVariable(arg: 1, scope: !214, type: !37)
// CHECK:STDOUT: !219 = !DILocation(line: 9, column: 12, scope: !214)
// CHECK:STDOUT: !220 = !DILocation(line: 9, column: 5, scope: !214)
// CHECK:STDOUT: !221 = !{!215}
// CHECK:STDOUT: !222 = distinct !DISubprogram(name: "GetT", linkageName: "_CGetT.C.Main.03220e6718a91584", scope: null, file: !121, line: 8, type: !108, spFlags: DISPFlagDefinition, unit: !122, retainedNodes: !229)
// CHECK:STDOUT: !223 = !DILocalVariable(arg: 1, scope: !222, type: !37)
// CHECK:STDOUT: !227 = !DILocation(line: 9, column: 12, scope: !222)
// CHECK:STDOUT: !228 = !DILocation(line: 9, column: 5, scope: !222)
// CHECK:STDOUT: !229 = !{!223}
// CHECK:STDOUT: !230 = distinct !DISubprogram(name: "Op", linkageName: "_COp.6753b13effab4291:Copy.Core.c9f31a0e20c75302", scope: null, file: !9, line: 58, type: !108, spFlags: DISPFlagDefinition, unit: !122, retainedNodes: !240)
// CHECK:STDOUT: !231 = !DILocalVariable(arg: 1, scope: !230, type: !37)
// CHECK:STDOUT: !235 = !DILocation(line: 59, column: 13, scope: !230)
// CHECK:STDOUT: !236 = !DILocation(line: 59, column: 12, scope: !230)
// CHECK:STDOUT: !237 = !DILocation(line: 59, column: 26, scope: !230)
// CHECK:STDOUT: !238 = !DILocation(line: 59, column: 39, scope: !230)
// CHECK:STDOUT: !239 = !DILocation(line: 59, column: 5, scope: !230)
// CHECK:STDOUT: !240 = !{!231}
// CHECK:STDOUT:
