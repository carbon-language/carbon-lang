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
// CHECK:STDOUT: @tuple.380.loc22_28.6 = internal constant { i32, i32 } { i32 1, i32 2 }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CInts.Main(ptr sret({ i32, i32 }) %return) #0 !dbg !55 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CMake.Main.2fad9574e050607e(ptr %return, i32 1, i32 2), !dbg !58
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
// CHECK:STDOUT:   call void @_CMake.Main.bcff5b0af1b04500(ptr %return, ptr @tuple.380.loc22_28.6, ptr @tuple.380.loc22_28.6), !dbg !74
// CHECK:STDOUT:   ret void, !dbg !77
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CMake.Main.2fad9574e050607e(ptr sret({ i32, i32 }) %return, i32 %x, i32 %y) #0 !dbg !78 {
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
// CHECK:STDOUT: define linkonce_odr void @_CMake.Main.bcff5b0af1b04500(ptr sret({ { i32, i32 }, { i32, i32 } }) %return, ptr %x, ptr %y) #0 !dbg !99 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc10_25.2.x = getelementptr inbounds nuw { { i32, i32 }, { i32, i32 } }, ptr %return, i32 0, i32 0, !dbg !108
// CHECK:STDOUT:   call void @"_COp.4e4edddc7580c79b:Copy.Core.3eda18978e8f48e1"(ptr %.loc10_25.2.x, ptr %x), !dbg !106
// CHECK:STDOUT:   %.loc10_25.4.y = getelementptr inbounds nuw { { i32, i32 }, { i32, i32 } }, ptr %return, i32 0, i32 1, !dbg !108
// CHECK:STDOUT:   call void @"_COp.4e4edddc7580c79b:Copy.Core.3eda18978e8f48e1"(ptr %.loc10_25.4.y, ptr %y), !dbg !107
// CHECK:STDOUT:   ret void, !dbg !111
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp.4e4edddc7580c79b:Copy.Core.3eda18978e8f48e1"(ptr sret({ i32, i32 }) %return, ptr %self) #0 !dbg !113 {
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
// CHECK:STDOUT: uselistorder ptr @"_COp.4e4edddc7580c79b:Copy.Core.3eda18978e8f48e1", { 1, 0 }
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
// CHECK:STDOUT: !78 = distinct !DISubprogram(name: "Make", linkageName: "_CMake.Main.2fad9574e050607e", scope: null, file: !53, line: 9, type: !60, spFlags: DISPFlagDefinition, unit: !54, retainedNodes: !89)
// CHECK:STDOUT: !79 = !DILocalVariable(arg: 1, scope: !78, type: !38)
// CHECK:STDOUT: !80 = !DILocalVariable(arg: 2, scope: !78, type: !38)
// CHECK:STDOUT: !87 = !DILocation(line: 10, column: 10, scope: !78)
// CHECK:STDOUT: !88 = !DILocation(line: 10, column: 3, scope: !78)
// CHECK:STDOUT: !89 = !{!79, !80}
// CHECK:STDOUT: !90 = distinct !DISubprogram(name: "Make", linkageName: "_CMake.Main.22bf4b2f6c4322a2", scope: null, file: !53, line: 9, type: !41, spFlags: DISPFlagDefinition, unit: !54)
// CHECK:STDOUT: !97 = !DILocation(line: 10, column: 10, scope: !90)
// CHECK:STDOUT: !98 = !DILocation(line: 10, column: 3, scope: !90)
// CHECK:STDOUT: !99 = distinct !DISubprogram(name: "Make", linkageName: "_CMake.Main.bcff5b0af1b04500", scope: null, file: !53, line: 9, type: !76, spFlags: DISPFlagDefinition, unit: !54, retainedNodes: !112)
// CHECK:STDOUT: !100 = !DILocalVariable(arg: 1, scope: !99, type: !39)
// CHECK:STDOUT: !101 = !DILocalVariable(arg: 2, scope: !99, type: !39)
// CHECK:STDOUT: !106 = !DILocation(line: 10, column: 16, scope: !99)
// CHECK:STDOUT: !107 = !DILocation(line: 10, column: 24, scope: !99)
// CHECK:STDOUT: !108 = !DILocation(line: 10, column: 10, scope: !99)
// CHECK:STDOUT: !109 = !{!39, !39}
// CHECK:STDOUT: !110 = !DISubroutineType(types: !109)
// CHECK:STDOUT: !111 = !DILocation(line: 10, column: 3, scope: !99)
// CHECK:STDOUT: !112 = !{!100, !101}
// CHECK:STDOUT: !113 = distinct !DISubprogram(name: "Op", linkageName: "_COp.4e4edddc7580c79b:Copy.Core.3eda18978e8f48e1", scope: null, file: !11, line: 52, type: !110, spFlags: DISPFlagDefinition, unit: !54, retainedNodes: !122)
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
// CHECK:STDOUT: @C.val.b50.loc16_3.1 = internal constant { i1, i32 } { i1 true, i32 0 }
// CHECK:STDOUT: @C.val.988.loc26_3.1 = internal constant { i1, {} } { i1 true, {} zeroinitializer }
// CHECK:STDOUT: @C.val.519.loc31_3.1 = internal constant { i1, { i32, i32, i32 } } { i1 true, { i32, i32, i32 } { i32 1, i32 2, i32 3 } }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CAccessBool.Main() #0 !dbg !129 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c.var = alloca { i1, i32 }, align 4, !dbg !132
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !132
// CHECK:STDOUT:   %.loc16_37.2.v = getelementptr inbounds nuw { i1, i32 }, ptr %c.var, i32 0, i32 0, !dbg !133
// CHECK:STDOUT:   %.loc16_37.5.w = getelementptr inbounds nuw { i1, i32 }, ptr %c.var, i32 0, i32 1, !dbg !133
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %c.var, ptr align 4 @C.val.b50.loc16_3.1, i64 8, i1 false), !dbg !132
// CHECK:STDOUT:   %C.GetBool.call = call i1 @_CGetBool.C.Main.2fad9574e050607e(ptr %c.var), !dbg !135
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.ab3d3663b9d81a60(ptr %c.var), !dbg !132
// CHECK:STDOUT:   ret i1 %C.GetBool.call, !dbg !138
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CAccessInt.Main() #0 !dbg !139 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c.var = alloca { i1, i32 }, align 4, !dbg !142
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !142
// CHECK:STDOUT:   %.loc21_37.2.v = getelementptr inbounds nuw { i1, i32 }, ptr %c.var, i32 0, i32 0, !dbg !143
// CHECK:STDOUT:   %.loc21_37.5.w = getelementptr inbounds nuw { i1, i32 }, ptr %c.var, i32 0, i32 1, !dbg !143
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %c.var, ptr align 4 @C.val.b50.loc16_3.1, i64 8, i1 false), !dbg !142
// CHECK:STDOUT:   %C.GetT.call = call i32 @_CGetT.C.Main.2fad9574e050607e(ptr %c.var), !dbg !145
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.ab3d3663b9d81a60(ptr %c.var), !dbg !142
// CHECK:STDOUT:   ret i32 %C.GetT.call, !dbg !148
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CAccessEmpty.Main() #0 !dbg !149 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c.var = alloca { i1, {} }, align 1, !dbg !152
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !152
// CHECK:STDOUT:   %.loc26_37.2.v = getelementptr inbounds nuw { i1, {} }, ptr %c.var, i32 0, i32 0, !dbg !153
// CHECK:STDOUT:   %.loc26_37.4.w = getelementptr inbounds nuw { i1, {} }, ptr %c.var, i32 0, i32 1, !dbg !153
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %c.var, ptr align 1 @C.val.988.loc26_3.1, i64 1, i1 false), !dbg !152
// CHECK:STDOUT:   call void @_CGetT.C.Main.22bf4b2f6c4322a2(ptr %c.var), !dbg !155
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.3900ebc9e1b9b76c(ptr %c.var), !dbg !152
// CHECK:STDOUT:   ret void, !dbg !156
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CAccessTuple.Main(ptr sret({ i32, i32, i32 }) %return) #0 !dbg !157 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c.var = alloca { i1, { i32, i32, i32 } }, align 4, !dbg !160
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !160
// CHECK:STDOUT:   %.loc31_57.2.v = getelementptr inbounds nuw { i1, { i32, i32, i32 } }, ptr %c.var, i32 0, i32 0, !dbg !161
// CHECK:STDOUT:   %.loc31_57.4.w = getelementptr inbounds nuw { i1, { i32, i32, i32 } }, ptr %c.var, i32 0, i32 1, !dbg !161
// CHECK:STDOUT:   %tuple.elem0.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %.loc31_57.4.w, i32 0, i32 0, !dbg !162
// CHECK:STDOUT:   %tuple.elem1.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %.loc31_57.4.w, i32 0, i32 1, !dbg !162
// CHECK:STDOUT:   %tuple.elem2.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %.loc31_57.4.w, i32 0, i32 2, !dbg !162
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %c.var, ptr align 4 @C.val.519.loc31_3.1, i64 16, i1 false), !dbg !160
// CHECK:STDOUT:   call void @_CGetT.C.Main.2d2ea40c1a2cefdc(ptr %return, ptr %c.var), !dbg !164
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.771b959080479bc3(ptr %c.var), !dbg !160
// CHECK:STDOUT:   ret void, !dbg !165
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @_CGetBool.C.Main.2fad9574e050607e(ptr %self) #0 !dbg !166 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc6_16.1.v = getelementptr inbounds nuw { i1, i32 }, ptr %self, i32 0, i32 0, !dbg !171
// CHECK:STDOUT:   %.loc6_16.2 = load i8, ptr %.loc6_16.1.v, align 1, !dbg !171
// CHECK:STDOUT:   %.loc6_16.21 = trunc i8 %.loc6_16.2 to i1, !dbg !171
// CHECK:STDOUT:   ret i1 %.loc6_16.21, !dbg !172
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.ab3d3663b9d81a60(ptr %t) #0 !dbg !174 {
// CHECK:STDOUT:   ret void, !dbg !179
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CGetT.C.Main.2fad9574e050607e(ptr %self) #0 !dbg !181 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc9_16.1.w = getelementptr inbounds nuw { i1, i32 }, ptr %self, i32 0, i32 1, !dbg !186
// CHECK:STDOUT:   %.loc9_16.2 = load i32, ptr %.loc9_16.1.w, align 4, !dbg !186
// CHECK:STDOUT:   ret i32 %.loc9_16.2, !dbg !187
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CGetT.C.Main.22bf4b2f6c4322a2(ptr %self) #0 !dbg !189 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc9_16.1.w = getelementptr inbounds nuw { i1, {} }, ptr %self, i32 0, i32 1, !dbg !194
// CHECK:STDOUT:   ret void, !dbg !195
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.3900ebc9e1b9b76c(ptr %t) #0 !dbg !197 {
// CHECK:STDOUT:   ret void, !dbg !202
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CGetT.C.Main.2d2ea40c1a2cefdc(ptr sret({ i32, i32, i32 }) %return, ptr %self) #0 !dbg !204 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc9_16.1.w = getelementptr inbounds nuw { i1, { i32, i32, i32 } }, ptr %self, i32 0, i32 1, !dbg !209
// CHECK:STDOUT:   call void @"_COp.6753b13effab4291:Copy.Core.dc98eaa6ba6c4450"(ptr %return, ptr %.loc9_16.1.w), !dbg !209
// CHECK:STDOUT:   ret void, !dbg !210
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.771b959080479bc3(ptr %t) #0 !dbg !212 {
// CHECK:STDOUT:   ret void, !dbg !217
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp.6753b13effab4291:Copy.Core.dc98eaa6ba6c4450"(ptr sret({ i32, i32, i32 }) %return, ptr %self) #0 !dbg !219 {
// CHECK:STDOUT:   %tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %self, i32 0, i32 0, !dbg !224
// CHECK:STDOUT:   %tuple.elem.load = load i32, ptr %tuple.elem, align 4, !dbg !224
// CHECK:STDOUT:   %tuple.elem1 = getelementptr inbounds nuw { i32, i32, i32 }, ptr %return, i32 0, i32 0, !dbg !225
// CHECK:STDOUT:   %tuple.elem2 = getelementptr inbounds nuw { i32, i32, i32 }, ptr %self, i32 0, i32 1, !dbg !226
// CHECK:STDOUT:   %tuple.elem.load3 = load i32, ptr %tuple.elem2, align 4, !dbg !226
// CHECK:STDOUT:   %tuple.elem4 = getelementptr inbounds nuw { i32, i32, i32 }, ptr %return, i32 0, i32 1, !dbg !225
// CHECK:STDOUT:   %tuple.elem5 = getelementptr inbounds nuw { i32, i32, i32 }, ptr %self, i32 0, i32 2, !dbg !227
// CHECK:STDOUT:   %tuple.elem.load6 = load i32, ptr %tuple.elem5, align 4, !dbg !227
// CHECK:STDOUT:   %tuple.elem7 = getelementptr inbounds nuw { i32, i32, i32 }, ptr %return, i32 0, i32 2, !dbg !225
// CHECK:STDOUT:   store i32 %tuple.elem.load, ptr %tuple.elem1, align 4, !dbg !225
// CHECK:STDOUT:   store i32 %tuple.elem.load3, ptr %tuple.elem4, align 4, !dbg !225
// CHECK:STDOUT:   store i32 %tuple.elem.load6, ptr %tuple.elem7, align 4, !dbg !225
// CHECK:STDOUT:   ret void, !dbg !228
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.ab3d3663b9d81a60, { 1, 0 }
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
// CHECK:STDOUT: !26 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !38 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !39 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !40 = !{!39}
// CHECK:STDOUT: !41 = !DISubroutineType(types: !40)
// CHECK:STDOUT: !109 = !{!39, !39}
// CHECK:STDOUT: !110 = !DISubroutineType(types: !109)
// CHECK:STDOUT: !123 = !DIFile(filename: "access.carbon", directory: "")
// CHECK:STDOUT: !124 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !123, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !125 = !{!38}
// CHECK:STDOUT: !126 = !DISubroutineType(types: !125)
// CHECK:STDOUT: !127 = !{null}
// CHECK:STDOUT: !128 = !DISubroutineType(types: !127)
// CHECK:STDOUT: !129 = distinct !DISubprogram(name: "AccessBool", linkageName: "_CAccessBool.Main", scope: null, file: !123, line: 15, type: !41, spFlags: DISPFlagDefinition, unit: !124)
// CHECK:STDOUT: !132 = !DILocation(line: 16, column: 3, scope: !129)
// CHECK:STDOUT: !133 = !DILocation(line: 16, column: 19, scope: !129)
// CHECK:STDOUT: !135 = !DILocation(line: 17, column: 10, scope: !129)
// CHECK:STDOUT: !136 = !{null, !39}
// CHECK:STDOUT: !137 = !DISubroutineType(types: !136)
// CHECK:STDOUT: !138 = !DILocation(line: 17, column: 3, scope: !129)
// CHECK:STDOUT: !139 = distinct !DISubprogram(name: "AccessInt", linkageName: "_CAccessInt.Main", scope: null, file: !123, line: 20, type: !126, spFlags: DISPFlagDefinition, unit: !124)
// CHECK:STDOUT: !142 = !DILocation(line: 21, column: 3, scope: !139)
// CHECK:STDOUT: !143 = !DILocation(line: 21, column: 19, scope: !139)
// CHECK:STDOUT: !145 = !DILocation(line: 22, column: 10, scope: !139)
// CHECK:STDOUT: !146 = !{!38, !39}
// CHECK:STDOUT: !147 = !DISubroutineType(types: !146)
// CHECK:STDOUT: !148 = !DILocation(line: 22, column: 3, scope: !139)
// CHECK:STDOUT: !149 = distinct !DISubprogram(name: "AccessEmpty", linkageName: "_CAccessEmpty.Main", scope: null, file: !123, line: 25, type: !128, spFlags: DISPFlagDefinition, unit: !124)
// CHECK:STDOUT: !152 = !DILocation(line: 26, column: 3, scope: !149)
// CHECK:STDOUT: !153 = !DILocation(line: 26, column: 18, scope: !149)
// CHECK:STDOUT: !155 = !DILocation(line: 27, column: 10, scope: !149)
// CHECK:STDOUT: !156 = !DILocation(line: 27, column: 3, scope: !149)
// CHECK:STDOUT: !157 = distinct !DISubprogram(name: "AccessTuple", linkageName: "_CAccessTuple.Main", scope: null, file: !123, line: 30, type: !41, spFlags: DISPFlagDefinition, unit: !124)
// CHECK:STDOUT: !160 = !DILocation(line: 31, column: 3, scope: !157)
// CHECK:STDOUT: !161 = !DILocation(line: 31, column: 31, scope: !157)
// CHECK:STDOUT: !162 = !DILocation(line: 31, column: 48, scope: !157)
// CHECK:STDOUT: !164 = !DILocation(line: 32, column: 10, scope: !157)
// CHECK:STDOUT: !165 = !DILocation(line: 32, column: 3, scope: !157)
// CHECK:STDOUT: !166 = distinct !DISubprogram(name: "GetBool", linkageName: "_CGetBool.C.Main.2fad9574e050607e", scope: null, file: !123, line: 5, type: !110, spFlags: DISPFlagDefinition, unit: !124, retainedNodes: !173)
// CHECK:STDOUT: !167 = !DILocalVariable(arg: 1, scope: !166, type: !39)
// CHECK:STDOUT: !171 = !DILocation(line: 6, column: 12, scope: !166)
// CHECK:STDOUT: !172 = !DILocation(line: 6, column: 5, scope: !166)
// CHECK:STDOUT: !173 = !{!167}
// CHECK:STDOUT: !174 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.ab3d3663b9d81a60", scope: null, file: !26, line: 32, type: !137, spFlags: DISPFlagDefinition, unit: !124, retainedNodes: !180)
// CHECK:STDOUT: !175 = !DILocalVariable(arg: 1, scope: !174, type: !39)
// CHECK:STDOUT: !179 = !DILocation(line: 32, column: 1, scope: !174)
// CHECK:STDOUT: !180 = !{!175}
// CHECK:STDOUT: !181 = distinct !DISubprogram(name: "GetT", linkageName: "_CGetT.C.Main.2fad9574e050607e", scope: null, file: !123, line: 8, type: !147, spFlags: DISPFlagDefinition, unit: !124, retainedNodes: !188)
// CHECK:STDOUT: !182 = !DILocalVariable(arg: 1, scope: !181, type: !39)
// CHECK:STDOUT: !186 = !DILocation(line: 9, column: 12, scope: !181)
// CHECK:STDOUT: !187 = !DILocation(line: 9, column: 5, scope: !181)
// CHECK:STDOUT: !188 = !{!182}
// CHECK:STDOUT: !189 = distinct !DISubprogram(name: "GetT", linkageName: "_CGetT.C.Main.22bf4b2f6c4322a2", scope: null, file: !123, line: 8, type: !137, spFlags: DISPFlagDefinition, unit: !124, retainedNodes: !196)
// CHECK:STDOUT: !190 = !DILocalVariable(arg: 1, scope: !189, type: !39)
// CHECK:STDOUT: !194 = !DILocation(line: 9, column: 12, scope: !189)
// CHECK:STDOUT: !195 = !DILocation(line: 9, column: 5, scope: !189)
// CHECK:STDOUT: !196 = !{!190}
// CHECK:STDOUT: !197 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.3900ebc9e1b9b76c", scope: null, file: !26, line: 32, type: !137, spFlags: DISPFlagDefinition, unit: !124, retainedNodes: !203)
// CHECK:STDOUT: !198 = !DILocalVariable(arg: 1, scope: !197, type: !39)
// CHECK:STDOUT: !202 = !DILocation(line: 32, column: 1, scope: !197)
// CHECK:STDOUT: !203 = !{!198}
// CHECK:STDOUT: !204 = distinct !DISubprogram(name: "GetT", linkageName: "_CGetT.C.Main.2d2ea40c1a2cefdc", scope: null, file: !123, line: 8, type: !110, spFlags: DISPFlagDefinition, unit: !124, retainedNodes: !211)
// CHECK:STDOUT: !205 = !DILocalVariable(arg: 1, scope: !204, type: !39)
// CHECK:STDOUT: !209 = !DILocation(line: 9, column: 12, scope: !204)
// CHECK:STDOUT: !210 = !DILocation(line: 9, column: 5, scope: !204)
// CHECK:STDOUT: !211 = !{!205}
// CHECK:STDOUT: !212 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.771b959080479bc3", scope: null, file: !26, line: 32, type: !137, spFlags: DISPFlagDefinition, unit: !124, retainedNodes: !218)
// CHECK:STDOUT: !213 = !DILocalVariable(arg: 1, scope: !212, type: !39)
// CHECK:STDOUT: !217 = !DILocation(line: 32, column: 1, scope: !212)
// CHECK:STDOUT: !218 = !{!213}
// CHECK:STDOUT: !219 = distinct !DISubprogram(name: "Op", linkageName: "_COp.6753b13effab4291:Copy.Core.dc98eaa6ba6c4450", scope: null, file: !11, line: 58, type: !110, spFlags: DISPFlagDefinition, unit: !124, retainedNodes: !229)
// CHECK:STDOUT: !220 = !DILocalVariable(arg: 1, scope: !219, type: !39)
// CHECK:STDOUT: !224 = !DILocation(line: 59, column: 13, scope: !219)
// CHECK:STDOUT: !225 = !DILocation(line: 59, column: 12, scope: !219)
// CHECK:STDOUT: !226 = !DILocation(line: 59, column: 26, scope: !219)
// CHECK:STDOUT: !227 = !DILocation(line: 59, column: 39, scope: !219)
// CHECK:STDOUT: !228 = !DILocation(line: 59, column: 5, scope: !219)
// CHECK:STDOUT: !229 = !{!220}
// CHECK:STDOUT:
