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
// CHECK:STDOUT: @tuple.102.loc22_28.6 = internal constant { i32, i32 } { i32 1, i32 2 }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CInts.Main(ptr sret({ i32, i32 }) %return) #0 !dbg !55 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CMake.Main.4d6ceefeb7f772f4(ptr %return, i32 1, i32 2), !dbg !58
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
// CHECK:STDOUT:   call void @_CMake.Main.878c3517da47a752(ptr %return, ptr @tuple.102.loc22_28.6, ptr @tuple.102.loc22_28.6), !dbg !74
// CHECK:STDOUT:   ret void, !dbg !77
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CMake.Main.4d6ceefeb7f772f4(ptr sret({ i32, i32 }) %return, i32 %x, i32 %y) #0 !dbg !78 {
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
// CHECK:STDOUT: define linkonce_odr void @_CMake.Main.878c3517da47a752(ptr sret({ { i32, i32 }, { i32, i32 } }) %return, ptr %x, ptr %y) #0 !dbg !99 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc10_25.2.x = getelementptr inbounds nuw { { i32, i32 }, { i32, i32 } }, ptr %return, i32 0, i32 0, !dbg !108
// CHECK:STDOUT:   call void @"_COp.4e4edddc7580c79b:Copy.Core.8608f3c6f62d80cb"(ptr %.loc10_25.2.x, ptr %x), !dbg !106
// CHECK:STDOUT:   %.loc10_25.4.y = getelementptr inbounds nuw { { i32, i32 }, { i32, i32 } }, ptr %return, i32 0, i32 1, !dbg !108
// CHECK:STDOUT:   call void @"_COp.4e4edddc7580c79b:Copy.Core.8608f3c6f62d80cb"(ptr %.loc10_25.4.y, ptr %y), !dbg !107
// CHECK:STDOUT:   ret void, !dbg !111
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp.4e4edddc7580c79b:Copy.Core.8608f3c6f62d80cb"(ptr sret({ i32, i32 }) %return, ptr %self) #0 !dbg !113 {
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
// CHECK:STDOUT: uselistorder ptr @"_COp.4e4edddc7580c79b:Copy.Core.8608f3c6f62d80cb", { 1, 0 }
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
// CHECK:STDOUT: !78 = distinct !DISubprogram(name: "Make", linkageName: "_CMake.Main.4d6ceefeb7f772f4", scope: null, file: !53, line: 9, type: !60, spFlags: DISPFlagDefinition, unit: !54, retainedNodes: !89)
// CHECK:STDOUT: !79 = !DILocalVariable(arg: 1, scope: !78, type: !38)
// CHECK:STDOUT: !80 = !DILocalVariable(arg: 2, scope: !78, type: !38)
// CHECK:STDOUT: !87 = !DILocation(line: 10, column: 10, scope: !78)
// CHECK:STDOUT: !88 = !DILocation(line: 10, column: 3, scope: !78)
// CHECK:STDOUT: !89 = !{!79, !80}
// CHECK:STDOUT: !90 = distinct !DISubprogram(name: "Make", linkageName: "_CMake.Main.22bf4b2f6c4322a2", scope: null, file: !53, line: 9, type: !41, spFlags: DISPFlagDefinition, unit: !54)
// CHECK:STDOUT: !97 = !DILocation(line: 10, column: 10, scope: !90)
// CHECK:STDOUT: !98 = !DILocation(line: 10, column: 3, scope: !90)
// CHECK:STDOUT: !99 = distinct !DISubprogram(name: "Make", linkageName: "_CMake.Main.878c3517da47a752", scope: null, file: !53, line: 9, type: !76, spFlags: DISPFlagDefinition, unit: !54, retainedNodes: !112)
// CHECK:STDOUT: !100 = !DILocalVariable(arg: 1, scope: !99, type: !39)
// CHECK:STDOUT: !101 = !DILocalVariable(arg: 2, scope: !99, type: !39)
// CHECK:STDOUT: !106 = !DILocation(line: 10, column: 16, scope: !99)
// CHECK:STDOUT: !107 = !DILocation(line: 10, column: 24, scope: !99)
// CHECK:STDOUT: !108 = !DILocation(line: 10, column: 10, scope: !99)
// CHECK:STDOUT: !109 = !{!39, !39}
// CHECK:STDOUT: !110 = !DISubroutineType(types: !109)
// CHECK:STDOUT: !111 = !DILocation(line: 10, column: 3, scope: !99)
// CHECK:STDOUT: !112 = !{!100, !101}
// CHECK:STDOUT: !113 = distinct !DISubprogram(name: "Op", linkageName: "_COp.4e4edddc7580c79b:Copy.Core.8608f3c6f62d80cb", scope: null, file: !11, line: 52, type: !110, spFlags: DISPFlagDefinition, unit: !54, retainedNodes: !122)
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
// CHECK:STDOUT: @C.val.29b.loc16_3 = internal constant { i1, i32 } { i1 true, i32 0 }
// CHECK:STDOUT: @C.val.0ea.loc26_3 = internal constant { i1, {} } { i1 true, {} zeroinitializer }
// CHECK:STDOUT: @C.val.727.loc31_3 = internal constant { i1, { i32, i32, i32 } } { i1 true, { i32, i32, i32 } { i32 1, i32 2, i32 3 } }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CAccessBool.Main() #0 !dbg !133 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c.var = alloca { i1, i32 }, align 4, !dbg !136
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !136
// CHECK:STDOUT:   %.loc16_37.2.v = getelementptr inbounds nuw { i1, i32 }, ptr %c.var, i32 0, i32 0, !dbg !137
// CHECK:STDOUT:   %.loc16_37.5.w = getelementptr inbounds nuw { i1, i32 }, ptr %c.var, i32 0, i32 1, !dbg !137
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %c.var, ptr align 4 @C.val.29b.loc16_3, i64 8, i1 false), !dbg !136
// CHECK:STDOUT:   %C.GetBool.call = call i1 @_CGetBool.C.Main.4d6ceefeb7f772f4(ptr %c.var), !dbg !139
// CHECK:STDOUT:   call void @"_CSelfDestruct.d012f76a41f764ee:core.Destroy.Core"(ptr %c.var), !dbg !136
// CHECK:STDOUT:   ret i1 %C.GetBool.call, !dbg !140
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.48e0adf639fe7af4:core.Destroy.Core"(ptr %self) #0 !dbg !141 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !143
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !145 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !147
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.861652b2c489f7cc:core.Destroy.Core"(ptr %self) #0 !dbg !149 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc16_3.1.w = getelementptr inbounds nuw { i1, i32 }, ptr %self, i32 0, i32 1, !dbg !151
// CHECK:STDOUT:   call void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %.loc16_3.1.w), !dbg !151
// CHECK:STDOUT:   %.loc16_3.2.v = getelementptr inbounds nuw { i1, i32 }, ptr %self, i32 0, i32 0, !dbg !151
// CHECK:STDOUT:   call void @"_CSelfDestruct.48e0adf639fe7af4:core.Destroy.Core"(ptr %.loc16_3.2.v), !dbg !151
// CHECK:STDOUT:   ret void, !dbg !151
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.861652b2c489f7cc:core.Destroy.Core"(ptr %self) #0 !dbg !153 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.861652b2c489f7cc:core.Destroy.Core"(ptr %self), !dbg !155
// CHECK:STDOUT:   ret void, !dbg !155
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.d012f76a41f764ee:core.Destroy.Core"(ptr %self) #0 !dbg !157 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !159
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.d012f76a41f764ee:core.Destroy.Core"(ptr %self) #0 !dbg !161 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.d012f76a41f764ee:core.Destroy.Core"(ptr %self), !dbg !163
// CHECK:STDOUT:   ret void, !dbg !163
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CAccessInt.Main() #0 !dbg !165 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c.var = alloca { i1, i32 }, align 4, !dbg !168
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !168
// CHECK:STDOUT:   %.loc21_37.2.v = getelementptr inbounds nuw { i1, i32 }, ptr %c.var, i32 0, i32 0, !dbg !169
// CHECK:STDOUT:   %.loc21_37.5.w = getelementptr inbounds nuw { i1, i32 }, ptr %c.var, i32 0, i32 1, !dbg !169
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %c.var, ptr align 4 @C.val.29b.loc16_3, i64 8, i1 false), !dbg !168
// CHECK:STDOUT:   %C.GetT.call = call i32 @_CGetT.C.Main.4d6ceefeb7f772f4(ptr %c.var), !dbg !171
// CHECK:STDOUT:   call void @"_CSelfDestruct.d012f76a41f764ee:core.Destroy.Core"(ptr %c.var), !dbg !168
// CHECK:STDOUT:   ret i32 %C.GetT.call, !dbg !174
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CAccessEmpty.Main() #0 !dbg !175 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c.var = alloca { i1, {} }, align 1, !dbg !178
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !178
// CHECK:STDOUT:   %.loc26_37.2.v = getelementptr inbounds nuw { i1, {} }, ptr %c.var, i32 0, i32 0, !dbg !179
// CHECK:STDOUT:   %.loc26_37.4.w = getelementptr inbounds nuw { i1, {} }, ptr %c.var, i32 0, i32 1, !dbg !179
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %c.var, ptr align 1 @C.val.0ea.loc26_3, i64 1, i1 false), !dbg !178
// CHECK:STDOUT:   call void @_CGetT.C.Main.22bf4b2f6c4322a2(ptr %c.var), !dbg !181
// CHECK:STDOUT:   call void @"_CSelfDestruct.0477b985a5aa27cc:core.Destroy.Core"(ptr %c.var), !dbg !178
// CHECK:STDOUT:   ret void, !dbg !182
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.61ea2aba74ab3bf1:core.Destroy.Core"(ptr %self) #0 !dbg !183 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !185
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.82468d70c8d12e86:core.Destroy.Core"(ptr %self) #0 !dbg !187 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc26_3.1.w = getelementptr inbounds nuw { i1, {} }, ptr %self, i32 0, i32 1, !dbg !189
// CHECK:STDOUT:   call void @"_CSelfDestruct.61ea2aba74ab3bf1:core.Destroy.Core"(ptr %.loc26_3.1.w), !dbg !189
// CHECK:STDOUT:   %.loc26_3.2.v = getelementptr inbounds nuw { i1, {} }, ptr %self, i32 0, i32 0, !dbg !189
// CHECK:STDOUT:   call void @"_CSelfDestruct.48e0adf639fe7af4:core.Destroy.Core"(ptr %.loc26_3.2.v), !dbg !189
// CHECK:STDOUT:   ret void, !dbg !189
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.82468d70c8d12e86:core.Destroy.Core"(ptr %self) #0 !dbg !191 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.82468d70c8d12e86:core.Destroy.Core"(ptr %self), !dbg !193
// CHECK:STDOUT:   ret void, !dbg !193
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.0477b985a5aa27cc:core.Destroy.Core"(ptr %self) #0 !dbg !195 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !197
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.0477b985a5aa27cc:core.Destroy.Core"(ptr %self) #0 !dbg !199 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.0477b985a5aa27cc:core.Destroy.Core"(ptr %self), !dbg !201
// CHECK:STDOUT:   ret void, !dbg !201
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CAccessTuple.Main(ptr sret({ i32, i32, i32 }) %return) #0 !dbg !203 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c.var = alloca { i1, { i32, i32, i32 } }, align 4, !dbg !206
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !206
// CHECK:STDOUT:   %.loc31_57.2.v = getelementptr inbounds nuw { i1, { i32, i32, i32 } }, ptr %c.var, i32 0, i32 0, !dbg !207
// CHECK:STDOUT:   %.loc31_57.4.w = getelementptr inbounds nuw { i1, { i32, i32, i32 } }, ptr %c.var, i32 0, i32 1, !dbg !207
// CHECK:STDOUT:   %tuple.elem0.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %.loc31_57.4.w, i32 0, i32 0, !dbg !208
// CHECK:STDOUT:   %tuple.elem1.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %.loc31_57.4.w, i32 0, i32 1, !dbg !208
// CHECK:STDOUT:   %tuple.elem2.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %.loc31_57.4.w, i32 0, i32 2, !dbg !208
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %c.var, ptr align 4 @C.val.727.loc31_3, i64 16, i1 false), !dbg !206
// CHECK:STDOUT:   call void @_CGetT.C.Main.03220e6718a91584(ptr %return, ptr %c.var), !dbg !210
// CHECK:STDOUT:   call void @"_CSelfDestruct.3cffcf67f1271f67:core.Destroy.Core"(ptr %c.var), !dbg !206
// CHECK:STDOUT:   ret void, !dbg !211
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.555b1314256372b8:core.Destroy.Core"(ptr %self) #0 !dbg !212 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %tuple.elem2.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %self, i32 0, i32 2, !dbg !214
// CHECK:STDOUT:   call void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %tuple.elem2.tuple.elem), !dbg !214
// CHECK:STDOUT:   %tuple.elem1.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %self, i32 0, i32 1, !dbg !214
// CHECK:STDOUT:   call void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %tuple.elem1.tuple.elem), !dbg !214
// CHECK:STDOUT:   %tuple.elem0.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %self, i32 0, i32 0, !dbg !214
// CHECK:STDOUT:   call void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %tuple.elem0.tuple.elem), !dbg !214
// CHECK:STDOUT:   ret void, !dbg !214
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.555b1314256372b8:core.Destroy.Core"(ptr %self) #0 !dbg !216 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.555b1314256372b8:core.Destroy.Core"(ptr %self), !dbg !218
// CHECK:STDOUT:   ret void, !dbg !218
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.1ada7ed46fb16378:core.Destroy.Core"(ptr %self) #0 !dbg !220 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc31_3.1.w = getelementptr inbounds nuw { i1, { i32, i32, i32 } }, ptr %self, i32 0, i32 1, !dbg !222
// CHECK:STDOUT:   call void @"_CSelfDestruct.555b1314256372b8:core.Destroy.Core"(ptr %.loc31_3.1.w), !dbg !222
// CHECK:STDOUT:   %.loc31_3.2.v = getelementptr inbounds nuw { i1, { i32, i32, i32 } }, ptr %self, i32 0, i32 0, !dbg !222
// CHECK:STDOUT:   call void @"_CSelfDestruct.48e0adf639fe7af4:core.Destroy.Core"(ptr %.loc31_3.2.v), !dbg !222
// CHECK:STDOUT:   ret void, !dbg !222
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.1ada7ed46fb16378:core.Destroy.Core"(ptr %self) #0 !dbg !224 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.1ada7ed46fb16378:core.Destroy.Core"(ptr %self), !dbg !226
// CHECK:STDOUT:   ret void, !dbg !226
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.3cffcf67f1271f67:core.Destroy.Core"(ptr %self) #0 !dbg !228 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !230
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.3cffcf67f1271f67:core.Destroy.Core"(ptr %self) #0 !dbg !232 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.3cffcf67f1271f67:core.Destroy.Core"(ptr %self), !dbg !234
// CHECK:STDOUT:   ret void, !dbg !234
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @_CGetBool.C.Main.4d6ceefeb7f772f4(ptr %self) #0 !dbg !236 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc6_16.1.v = getelementptr inbounds nuw { i1, i32 }, ptr %self, i32 0, i32 0, !dbg !241
// CHECK:STDOUT:   %.loc6_16.2 = load i8, ptr %.loc6_16.1.v, align 1, !dbg !241
// CHECK:STDOUT:   %.loc6_16.21 = trunc i8 %.loc6_16.2 to i1, !dbg !241
// CHECK:STDOUT:   ret i1 %.loc6_16.21, !dbg !242
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CGetT.C.Main.4d6ceefeb7f772f4(ptr %self) #0 !dbg !244 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc9_16.1.w = getelementptr inbounds nuw { i1, i32 }, ptr %self, i32 0, i32 1, !dbg !249
// CHECK:STDOUT:   %.loc9_16.2 = load i32, ptr %.loc9_16.1.w, align 4, !dbg !249
// CHECK:STDOUT:   ret i32 %.loc9_16.2, !dbg !250
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CGetT.C.Main.22bf4b2f6c4322a2(ptr %self) #0 !dbg !252 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc9_16.1.w = getelementptr inbounds nuw { i1, {} }, ptr %self, i32 0, i32 1, !dbg !257
// CHECK:STDOUT:   ret void, !dbg !258
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CGetT.C.Main.03220e6718a91584(ptr sret({ i32, i32, i32 }) %return, ptr %self) #0 !dbg !260 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc9_16.1.w = getelementptr inbounds nuw { i1, { i32, i32, i32 } }, ptr %self, i32 0, i32 1, !dbg !265
// CHECK:STDOUT:   call void @"_COp.6753b13effab4291:Copy.Core.c9f31a0e20c75302"(ptr %return, ptr %.loc9_16.1.w), !dbg !265
// CHECK:STDOUT:   ret void, !dbg !266
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp.6753b13effab4291:Copy.Core.c9f31a0e20c75302"(ptr sret({ i32, i32, i32 }) %return, ptr %self) #0 !dbg !268 {
// CHECK:STDOUT:   %tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %self, i32 0, i32 0, !dbg !273
// CHECK:STDOUT:   %tuple.elem.load = load i32, ptr %tuple.elem, align 4, !dbg !273
// CHECK:STDOUT:   %tuple.elem1 = getelementptr inbounds nuw { i32, i32, i32 }, ptr %return, i32 0, i32 0, !dbg !274
// CHECK:STDOUT:   %tuple.elem2 = getelementptr inbounds nuw { i32, i32, i32 }, ptr %self, i32 0, i32 1, !dbg !275
// CHECK:STDOUT:   %tuple.elem.load3 = load i32, ptr %tuple.elem2, align 4, !dbg !275
// CHECK:STDOUT:   %tuple.elem4 = getelementptr inbounds nuw { i32, i32, i32 }, ptr %return, i32 0, i32 1, !dbg !274
// CHECK:STDOUT:   %tuple.elem5 = getelementptr inbounds nuw { i32, i32, i32 }, ptr %self, i32 0, i32 2, !dbg !276
// CHECK:STDOUT:   %tuple.elem.load6 = load i32, ptr %tuple.elem5, align 4, !dbg !276
// CHECK:STDOUT:   %tuple.elem7 = getelementptr inbounds nuw { i32, i32, i32 }, ptr %return, i32 0, i32 2, !dbg !274
// CHECK:STDOUT:   store i32 %tuple.elem.load, ptr %tuple.elem1, align 4, !dbg !274
// CHECK:STDOUT:   store i32 %tuple.elem.load3, ptr %tuple.elem4, align 4, !dbg !274
// CHECK:STDOUT:   store i32 %tuple.elem.load6, ptr %tuple.elem7, align 4, !dbg !274
// CHECK:STDOUT:   ret void, !dbg !277
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 3, 2, 1, 0 }
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
// CHECK:STDOUT: !127 = !{null, !38}
// CHECK:STDOUT: !128 = !DISubroutineType(types: !127)
// CHECK:STDOUT: !129 = !{!38}
// CHECK:STDOUT: !130 = !DISubroutineType(types: !129)
// CHECK:STDOUT: !131 = !{null}
// CHECK:STDOUT: !132 = !DISubroutineType(types: !131)
// CHECK:STDOUT: !133 = distinct !DISubprogram(name: "AccessBool", linkageName: "_CAccessBool.Main", scope: null, file: !123, line: 15, type: !41, spFlags: DISPFlagDefinition, unit: !124)
// CHECK:STDOUT: !136 = !DILocation(line: 16, column: 3, scope: !133)
// CHECK:STDOUT: !137 = !DILocation(line: 16, column: 19, scope: !133)
// CHECK:STDOUT: !139 = !DILocation(line: 17, column: 10, scope: !133)
// CHECK:STDOUT: !140 = !DILocation(line: 17, column: 3, scope: !133)
// CHECK:STDOUT: !141 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.48e0adf639fe7af4:core.Destroy.Core", scope: null, file: !123, line: 16, type: !126, spFlags: DISPFlagDefinition, unit: !124, retainedNodes: !144)
// CHECK:STDOUT: !142 = !DILocalVariable(arg: 1, scope: !141, type: !39)
// CHECK:STDOUT: !143 = !DILocation(line: 16, column: 3, scope: !141)
// CHECK:STDOUT: !144 = !{!142}
// CHECK:STDOUT: !145 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.f58524831c37807e:core.Destroy.Core", scope: null, file: !123, line: 16, type: !128, spFlags: DISPFlagDefinition, unit: !124, retainedNodes: !148)
// CHECK:STDOUT: !146 = !DILocalVariable(arg: 1, scope: !145, type: !38)
// CHECK:STDOUT: !147 = !DILocation(line: 16, column: 3, scope: !145)
// CHECK:STDOUT: !148 = !{!146}
// CHECK:STDOUT: !149 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.861652b2c489f7cc:core.Destroy.Core", scope: null, file: !123, line: 16, type: !126, spFlags: DISPFlagDefinition, unit: !124, retainedNodes: !152)
// CHECK:STDOUT: !150 = !DILocalVariable(arg: 1, scope: !149, type: !39)
// CHECK:STDOUT: !151 = !DILocation(line: 16, column: 3, scope: !149)
// CHECK:STDOUT: !152 = !{!150}
// CHECK:STDOUT: !153 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.861652b2c489f7cc:core.Destroy.Core", scope: null, file: !123, line: 16, type: !126, spFlags: DISPFlagDefinition, unit: !124, retainedNodes: !156)
// CHECK:STDOUT: !154 = !DILocalVariable(arg: 1, scope: !153, type: !39)
// CHECK:STDOUT: !155 = !DILocation(line: 16, column: 3, scope: !153)
// CHECK:STDOUT: !156 = !{!154}
// CHECK:STDOUT: !157 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.d012f76a41f764ee:core.Destroy.Core", scope: null, file: !123, line: 16, type: !126, spFlags: DISPFlagDefinition, unit: !124, retainedNodes: !160)
// CHECK:STDOUT: !158 = !DILocalVariable(arg: 1, scope: !157, type: !39)
// CHECK:STDOUT: !159 = !DILocation(line: 16, column: 3, scope: !157)
// CHECK:STDOUT: !160 = !{!158}
// CHECK:STDOUT: !161 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.d012f76a41f764ee:core.Destroy.Core", scope: null, file: !123, line: 16, type: !126, spFlags: DISPFlagDefinition, unit: !124, retainedNodes: !164)
// CHECK:STDOUT: !162 = !DILocalVariable(arg: 1, scope: !161, type: !39)
// CHECK:STDOUT: !163 = !DILocation(line: 16, column: 3, scope: !161)
// CHECK:STDOUT: !164 = !{!162}
// CHECK:STDOUT: !165 = distinct !DISubprogram(name: "AccessInt", linkageName: "_CAccessInt.Main", scope: null, file: !123, line: 20, type: !130, spFlags: DISPFlagDefinition, unit: !124)
// CHECK:STDOUT: !168 = !DILocation(line: 21, column: 3, scope: !165)
// CHECK:STDOUT: !169 = !DILocation(line: 21, column: 19, scope: !165)
// CHECK:STDOUT: !171 = !DILocation(line: 22, column: 10, scope: !165)
// CHECK:STDOUT: !172 = !{!38, !39}
// CHECK:STDOUT: !173 = !DISubroutineType(types: !172)
// CHECK:STDOUT: !174 = !DILocation(line: 22, column: 3, scope: !165)
// CHECK:STDOUT: !175 = distinct !DISubprogram(name: "AccessEmpty", linkageName: "_CAccessEmpty.Main", scope: null, file: !123, line: 25, type: !132, spFlags: DISPFlagDefinition, unit: !124)
// CHECK:STDOUT: !178 = !DILocation(line: 26, column: 3, scope: !175)
// CHECK:STDOUT: !179 = !DILocation(line: 26, column: 18, scope: !175)
// CHECK:STDOUT: !181 = !DILocation(line: 27, column: 10, scope: !175)
// CHECK:STDOUT: !182 = !DILocation(line: 27, column: 3, scope: !175)
// CHECK:STDOUT: !183 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.61ea2aba74ab3bf1:core.Destroy.Core", scope: null, file: !123, line: 26, type: !126, spFlags: DISPFlagDefinition, unit: !124, retainedNodes: !186)
// CHECK:STDOUT: !184 = !DILocalVariable(arg: 1, scope: !183, type: !39)
// CHECK:STDOUT: !185 = !DILocation(line: 26, column: 3, scope: !183)
// CHECK:STDOUT: !186 = !{!184}
// CHECK:STDOUT: !187 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.82468d70c8d12e86:core.Destroy.Core", scope: null, file: !123, line: 26, type: !126, spFlags: DISPFlagDefinition, unit: !124, retainedNodes: !190)
// CHECK:STDOUT: !188 = !DILocalVariable(arg: 1, scope: !187, type: !39)
// CHECK:STDOUT: !189 = !DILocation(line: 26, column: 3, scope: !187)
// CHECK:STDOUT: !190 = !{!188}
// CHECK:STDOUT: !191 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.82468d70c8d12e86:core.Destroy.Core", scope: null, file: !123, line: 26, type: !126, spFlags: DISPFlagDefinition, unit: !124, retainedNodes: !194)
// CHECK:STDOUT: !192 = !DILocalVariable(arg: 1, scope: !191, type: !39)
// CHECK:STDOUT: !193 = !DILocation(line: 26, column: 3, scope: !191)
// CHECK:STDOUT: !194 = !{!192}
// CHECK:STDOUT: !195 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.0477b985a5aa27cc:core.Destroy.Core", scope: null, file: !123, line: 26, type: !126, spFlags: DISPFlagDefinition, unit: !124, retainedNodes: !198)
// CHECK:STDOUT: !196 = !DILocalVariable(arg: 1, scope: !195, type: !39)
// CHECK:STDOUT: !197 = !DILocation(line: 26, column: 3, scope: !195)
// CHECK:STDOUT: !198 = !{!196}
// CHECK:STDOUT: !199 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.0477b985a5aa27cc:core.Destroy.Core", scope: null, file: !123, line: 26, type: !126, spFlags: DISPFlagDefinition, unit: !124, retainedNodes: !202)
// CHECK:STDOUT: !200 = !DILocalVariable(arg: 1, scope: !199, type: !39)
// CHECK:STDOUT: !201 = !DILocation(line: 26, column: 3, scope: !199)
// CHECK:STDOUT: !202 = !{!200}
// CHECK:STDOUT: !203 = distinct !DISubprogram(name: "AccessTuple", linkageName: "_CAccessTuple.Main", scope: null, file: !123, line: 30, type: !41, spFlags: DISPFlagDefinition, unit: !124)
// CHECK:STDOUT: !206 = !DILocation(line: 31, column: 3, scope: !203)
// CHECK:STDOUT: !207 = !DILocation(line: 31, column: 31, scope: !203)
// CHECK:STDOUT: !208 = !DILocation(line: 31, column: 48, scope: !203)
// CHECK:STDOUT: !210 = !DILocation(line: 32, column: 10, scope: !203)
// CHECK:STDOUT: !211 = !DILocation(line: 32, column: 3, scope: !203)
// CHECK:STDOUT: !212 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.555b1314256372b8:core.Destroy.Core", scope: null, file: !123, line: 31, type: !126, spFlags: DISPFlagDefinition, unit: !124, retainedNodes: !215)
// CHECK:STDOUT: !213 = !DILocalVariable(arg: 1, scope: !212, type: !39)
// CHECK:STDOUT: !214 = !DILocation(line: 31, column: 3, scope: !212)
// CHECK:STDOUT: !215 = !{!213}
// CHECK:STDOUT: !216 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.555b1314256372b8:core.Destroy.Core", scope: null, file: !123, line: 31, type: !126, spFlags: DISPFlagDefinition, unit: !124, retainedNodes: !219)
// CHECK:STDOUT: !217 = !DILocalVariable(arg: 1, scope: !216, type: !39)
// CHECK:STDOUT: !218 = !DILocation(line: 31, column: 3, scope: !216)
// CHECK:STDOUT: !219 = !{!217}
// CHECK:STDOUT: !220 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.1ada7ed46fb16378:core.Destroy.Core", scope: null, file: !123, line: 31, type: !126, spFlags: DISPFlagDefinition, unit: !124, retainedNodes: !223)
// CHECK:STDOUT: !221 = !DILocalVariable(arg: 1, scope: !220, type: !39)
// CHECK:STDOUT: !222 = !DILocation(line: 31, column: 3, scope: !220)
// CHECK:STDOUT: !223 = !{!221}
// CHECK:STDOUT: !224 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.1ada7ed46fb16378:core.Destroy.Core", scope: null, file: !123, line: 31, type: !126, spFlags: DISPFlagDefinition, unit: !124, retainedNodes: !227)
// CHECK:STDOUT: !225 = !DILocalVariable(arg: 1, scope: !224, type: !39)
// CHECK:STDOUT: !226 = !DILocation(line: 31, column: 3, scope: !224)
// CHECK:STDOUT: !227 = !{!225}
// CHECK:STDOUT: !228 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.3cffcf67f1271f67:core.Destroy.Core", scope: null, file: !123, line: 31, type: !126, spFlags: DISPFlagDefinition, unit: !124, retainedNodes: !231)
// CHECK:STDOUT: !229 = !DILocalVariable(arg: 1, scope: !228, type: !39)
// CHECK:STDOUT: !230 = !DILocation(line: 31, column: 3, scope: !228)
// CHECK:STDOUT: !231 = !{!229}
// CHECK:STDOUT: !232 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.3cffcf67f1271f67:core.Destroy.Core", scope: null, file: !123, line: 31, type: !126, spFlags: DISPFlagDefinition, unit: !124, retainedNodes: !235)
// CHECK:STDOUT: !233 = !DILocalVariable(arg: 1, scope: !232, type: !39)
// CHECK:STDOUT: !234 = !DILocation(line: 31, column: 3, scope: !232)
// CHECK:STDOUT: !235 = !{!233}
// CHECK:STDOUT: !236 = distinct !DISubprogram(name: "GetBool", linkageName: "_CGetBool.C.Main.4d6ceefeb7f772f4", scope: null, file: !123, line: 5, type: !110, spFlags: DISPFlagDefinition, unit: !124, retainedNodes: !243)
// CHECK:STDOUT: !237 = !DILocalVariable(arg: 1, scope: !236, type: !39)
// CHECK:STDOUT: !241 = !DILocation(line: 6, column: 12, scope: !236)
// CHECK:STDOUT: !242 = !DILocation(line: 6, column: 5, scope: !236)
// CHECK:STDOUT: !243 = !{!237}
// CHECK:STDOUT: !244 = distinct !DISubprogram(name: "GetT", linkageName: "_CGetT.C.Main.4d6ceefeb7f772f4", scope: null, file: !123, line: 8, type: !173, spFlags: DISPFlagDefinition, unit: !124, retainedNodes: !251)
// CHECK:STDOUT: !245 = !DILocalVariable(arg: 1, scope: !244, type: !39)
// CHECK:STDOUT: !249 = !DILocation(line: 9, column: 12, scope: !244)
// CHECK:STDOUT: !250 = !DILocation(line: 9, column: 5, scope: !244)
// CHECK:STDOUT: !251 = !{!245}
// CHECK:STDOUT: !252 = distinct !DISubprogram(name: "GetT", linkageName: "_CGetT.C.Main.22bf4b2f6c4322a2", scope: null, file: !123, line: 8, type: !126, spFlags: DISPFlagDefinition, unit: !124, retainedNodes: !259)
// CHECK:STDOUT: !253 = !DILocalVariable(arg: 1, scope: !252, type: !39)
// CHECK:STDOUT: !257 = !DILocation(line: 9, column: 12, scope: !252)
// CHECK:STDOUT: !258 = !DILocation(line: 9, column: 5, scope: !252)
// CHECK:STDOUT: !259 = !{!253}
// CHECK:STDOUT: !260 = distinct !DISubprogram(name: "GetT", linkageName: "_CGetT.C.Main.03220e6718a91584", scope: null, file: !123, line: 8, type: !110, spFlags: DISPFlagDefinition, unit: !124, retainedNodes: !267)
// CHECK:STDOUT: !261 = !DILocalVariable(arg: 1, scope: !260, type: !39)
// CHECK:STDOUT: !265 = !DILocation(line: 9, column: 12, scope: !260)
// CHECK:STDOUT: !266 = !DILocation(line: 9, column: 5, scope: !260)
// CHECK:STDOUT: !267 = !{!261}
// CHECK:STDOUT: !268 = distinct !DISubprogram(name: "Op", linkageName: "_COp.6753b13effab4291:Copy.Core.c9f31a0e20c75302", scope: null, file: !11, line: 58, type: !110, spFlags: DISPFlagDefinition, unit: !124, retainedNodes: !278)
// CHECK:STDOUT: !269 = !DILocalVariable(arg: 1, scope: !268, type: !39)
// CHECK:STDOUT: !273 = !DILocation(line: 59, column: 13, scope: !268)
// CHECK:STDOUT: !274 = !DILocation(line: 59, column: 12, scope: !268)
// CHECK:STDOUT: !275 = !DILocation(line: 59, column: 26, scope: !268)
// CHECK:STDOUT: !276 = !DILocation(line: 59, column: 39, scope: !268)
// CHECK:STDOUT: !277 = !DILocation(line: 59, column: 5, scope: !268)
// CHECK:STDOUT: !278 = !{!269}
// CHECK:STDOUT:
