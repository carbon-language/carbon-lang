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
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "classes.carbon", directory: "")
// CHECK:STDOUT: ; ModuleID = 'create.carbon'
// CHECK:STDOUT: source_filename = "create.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @Derived.val.loc7_36 = internal constant { { i32 }, i32 } { { i32 } { i32 1 }, i32 2 }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCreateDerived.Create(ptr sret({ { i32 }, i32 }) %return) #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_35.2.base = getelementptr inbounds nuw { { i32 }, i32 }, ptr %return, i32 0, i32 0, !dbg !8
// CHECK:STDOUT:   %.loc7_26.3.b = getelementptr inbounds nuw { i32 }, ptr %.loc7_35.2.base, i32 0, i32 0, !dbg !9
// CHECK:STDOUT:   %.loc7_35.5.d = getelementptr inbounds nuw { { i32 }, i32 }, ptr %return, i32 0, i32 1, !dbg !8
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %return, ptr align 4 @Derived.val.loc7_36, i64 8, i1 false), !dbg !10
// CHECK:STDOUT:   ret void, !dbg !10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCreateAdapter.Create(ptr sret({ { i32 }, i32 }) %return) #0 !dbg !11 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CCreateDerived.Create(ptr %return), !dbg !12
// CHECK:STDOUT:   ret void, !dbg !13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "create.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "CreateDerived", linkageName: "_CCreateDerived.Create", scope: null, file: !3, line: 6, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{!7}
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 8)
// CHECK:STDOUT: !8 = !DILocation(line: 7, column: 10, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 7, column: 19, scope: !4)
// CHECK:STDOUT: !10 = !DILocation(line: 7, column: 3, scope: !4)
// CHECK:STDOUT: !11 = distinct !DISubprogram(name: "CreateAdapter", linkageName: "_CCreateAdapter.Create", scope: null, file: !3, line: 10, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !12 = !DILocation(line: 11, column: 10, scope: !11)
// CHECK:STDOUT: !13 = !DILocation(line: 11, column: 3, scope: !11)
// CHECK:STDOUT: ; ModuleID = 'create_generic.carbon'
// CHECK:STDOUT: source_filename = "create_generic.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @tuple.21c.loc22_28.6 = internal constant { i32, i32 } { i32 1, i32 2 }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CInts.Main(ptr sret({ i32, i32 }) %return) #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CMake.Main.64ccbb8e5d9a0b8e(ptr %return, i32 1, i32 2), !dbg !8
// CHECK:STDOUT:   ret void, !dbg !9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CEmpty.Main(ptr sret({ {}, {} }) %return) #0 !dbg !10 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CMake.Main.335f6bfc3bd91486(ptr %return), !dbg !11
// CHECK:STDOUT:   ret void, !dbg !12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTuples.Main(ptr sret({ { i32, i32 }, { i32, i32 } }) %return) #0 !dbg !13 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CMake.Main.7af4aab1086ddeeb(ptr %return, ptr @tuple.21c.loc22_28.6, ptr @tuple.21c.loc22_28.6), !dbg !14
// CHECK:STDOUT:   ret void, !dbg !15
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CMake.Main.64ccbb8e5d9a0b8e(ptr sret({ i32, i32 }) %return, i32 %x, i32 %y) #0 !dbg !16 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc10_25.2.x = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 0, !dbg !23
// CHECK:STDOUT:   store i32 %x, ptr %.loc10_25.2.x, align 4, !dbg !23
// CHECK:STDOUT:   %.loc10_25.4.y = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 1, !dbg !23
// CHECK:STDOUT:   store i32 %y, ptr %.loc10_25.4.y, align 4, !dbg !23
// CHECK:STDOUT:   ret void, !dbg !24
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CMake.Main.335f6bfc3bd91486(ptr sret({ {}, {} }) %return) #0 !dbg !25 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc10_25.2.x = getelementptr inbounds nuw { {}, {} }, ptr %return, i32 0, i32 0, !dbg !26
// CHECK:STDOUT:   %.loc10_25.4.y = getelementptr inbounds nuw { {}, {} }, ptr %return, i32 0, i32 1, !dbg !26
// CHECK:STDOUT:   ret void, !dbg !27
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CMake.Main.7af4aab1086ddeeb(ptr sret({ { i32, i32 }, { i32, i32 } }) %return, ptr %x, ptr %y) #0 !dbg !28 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc10_25.2.x = getelementptr inbounds nuw { { i32, i32 }, { i32, i32 } }, ptr %return, i32 0, i32 0, !dbg !34
// CHECK:STDOUT:   call void @"_COp.e4daa70d026fdea2:Copy.Core.3e36085cb869f630"(ptr %.loc10_25.2.x, ptr %x), !dbg !35
// CHECK:STDOUT:   %.loc10_25.4.y = getelementptr inbounds nuw { { i32, i32 }, { i32, i32 } }, ptr %return, i32 0, i32 1, !dbg !34
// CHECK:STDOUT:   call void @"_COp.e4daa70d026fdea2:Copy.Core.3e36085cb869f630"(ptr %.loc10_25.4.y, ptr %y), !dbg !36
// CHECK:STDOUT:   ret void, !dbg !37
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp.e4daa70d026fdea2:Copy.Core.3e36085cb869f630"(ptr sret({ i32, i32 }) %return, ptr %self) #0 !dbg !38 {
// CHECK:STDOUT:   %tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 0, !dbg !44
// CHECK:STDOUT:   %tuple.elem.load = load i32, ptr %tuple.elem, align 4, !dbg !44
// CHECK:STDOUT:   %tuple.elem1 = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 0, !dbg !45
// CHECK:STDOUT:   %tuple.elem2 = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 1, !dbg !46
// CHECK:STDOUT:   %tuple.elem.load3 = load i32, ptr %tuple.elem2, align 4, !dbg !46
// CHECK:STDOUT:   %tuple.elem4 = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 1, !dbg !45
// CHECK:STDOUT:   store i32 %tuple.elem.load, ptr %tuple.elem1, align 4, !dbg !45
// CHECK:STDOUT:   store i32 %tuple.elem.load3, ptr %tuple.elem4, align 4, !dbg !45
// CHECK:STDOUT:   ret void, !dbg !47
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @"_COp.e4daa70d026fdea2:Copy.Core.3e36085cb869f630", { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "create_generic.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Ints", linkageName: "_CInts.Main", scope: null, file: !3, line: 13, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{!7}
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 8)
// CHECK:STDOUT: !8 = !DILocation(line: 14, column: 10, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 14, column: 3, scope: !4)
// CHECK:STDOUT: !10 = distinct !DISubprogram(name: "Empty", linkageName: "_CEmpty.Main", scope: null, file: !3, line: 17, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !11 = !DILocation(line: 18, column: 10, scope: !10)
// CHECK:STDOUT: !12 = !DILocation(line: 18, column: 3, scope: !10)
// CHECK:STDOUT: !13 = distinct !DISubprogram(name: "Tuples", linkageName: "_CTuples.Main", scope: null, file: !3, line: 21, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !14 = !DILocation(line: 23, column: 10, scope: !13)
// CHECK:STDOUT: !15 = !DILocation(line: 23, column: 3, scope: !13)
// CHECK:STDOUT: !16 = distinct !DISubprogram(name: "Make", linkageName: "_CMake.Main.64ccbb8e5d9a0b8e", scope: null, file: !3, line: 9, type: !17, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !20)
// CHECK:STDOUT: !17 = !DISubroutineType(types: !18)
// CHECK:STDOUT: !18 = !{!7, !19, !19}
// CHECK:STDOUT: !19 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !20 = !{!21, !22}
// CHECK:STDOUT: !21 = !DILocalVariable(arg: 1, scope: !16, type: !19)
// CHECK:STDOUT: !22 = !DILocalVariable(arg: 2, scope: !16, type: !19)
// CHECK:STDOUT: !23 = !DILocation(line: 10, column: 10, scope: !16)
// CHECK:STDOUT: !24 = !DILocation(line: 10, column: 3, scope: !16)
// CHECK:STDOUT: !25 = distinct !DISubprogram(name: "Make", linkageName: "_CMake.Main.335f6bfc3bd91486", scope: null, file: !3, line: 9, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !26 = !DILocation(line: 10, column: 10, scope: !25)
// CHECK:STDOUT: !27 = !DILocation(line: 10, column: 3, scope: !25)
// CHECK:STDOUT: !28 = distinct !DISubprogram(name: "Make", linkageName: "_CMake.Main.7af4aab1086ddeeb", scope: null, file: !3, line: 9, type: !29, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !31)
// CHECK:STDOUT: !29 = !DISubroutineType(types: !30)
// CHECK:STDOUT: !30 = !{!7, !7, !7}
// CHECK:STDOUT: !31 = !{!32, !33}
// CHECK:STDOUT: !32 = !DILocalVariable(arg: 1, scope: !28, type: !7)
// CHECK:STDOUT: !33 = !DILocalVariable(arg: 2, scope: !28, type: !7)
// CHECK:STDOUT: !34 = !DILocation(line: 10, column: 10, scope: !28)
// CHECK:STDOUT: !35 = !DILocation(line: 10, column: 16, scope: !28)
// CHECK:STDOUT: !36 = !DILocation(line: 10, column: 24, scope: !28)
// CHECK:STDOUT: !37 = !DILocation(line: 10, column: 3, scope: !28)
// CHECK:STDOUT: !38 = distinct !DISubprogram(name: "Op", linkageName: "_COp.e4daa70d026fdea2:Copy.Core.3e36085cb869f630", scope: null, file: !39, line: 48, type: !40, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !42)
// CHECK:STDOUT: !39 = !DIFile(filename: "min_prelude/parts/copy.carbon", directory: "")
// CHECK:STDOUT: !40 = !DISubroutineType(types: !41)
// CHECK:STDOUT: !41 = !{!7, !7}
// CHECK:STDOUT: !42 = !{!43}
// CHECK:STDOUT: !43 = !DILocalVariable(arg: 1, scope: !38, type: !7)
// CHECK:STDOUT: !44 = !DILocation(line: 49, column: 13, scope: !38)
// CHECK:STDOUT: !45 = !DILocation(line: 49, column: 12, scope: !38)
// CHECK:STDOUT: !46 = !DILocation(line: 49, column: 26, scope: !38)
// CHECK:STDOUT: !47 = !DILocation(line: 49, column: 5, scope: !38)
// CHECK:STDOUT: ; ModuleID = 'access.carbon'
// CHECK:STDOUT: source_filename = "access.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @C.val.310.loc16_3 = internal constant { i1, i32 } { i1 true, i32 0 }
// CHECK:STDOUT: @C.val.8d5.loc26_3 = internal constant { i1, {} } { i1 true, {} zeroinitializer }
// CHECK:STDOUT: @C.val.76f.loc31_3 = internal constant { i1, { i32, i32, i32 } } { i1 true, { i32, i32, i32 } { i32 1, i32 2, i32 3 } }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CAccessBool.Main() #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c.var = alloca { i1, i32 }, align 8, !dbg !8
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !8
// CHECK:STDOUT:   %.loc16_37.2.v = getelementptr inbounds nuw { i1, i32 }, ptr %c.var, i32 0, i32 0, !dbg !9
// CHECK:STDOUT:   %.loc16_37.5.w = getelementptr inbounds nuw { i1, i32 }, ptr %c.var, i32 0, i32 1, !dbg !9
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %c.var, ptr align 4 @C.val.310.loc16_3, i64 8, i1 false), !dbg !8
// CHECK:STDOUT:   %C.GetBool.call = call i1 @_CGetBool.C.Main.64ccbb8e5d9a0b8e(ptr %c.var), !dbg !10
// CHECK:STDOUT:   ret i1 %C.GetBool.call, !dbg !11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CAccessInt.Main() #0 !dbg !12 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c.var = alloca { i1, i32 }, align 8, !dbg !16
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !16
// CHECK:STDOUT:   %.loc21_37.2.v = getelementptr inbounds nuw { i1, i32 }, ptr %c.var, i32 0, i32 0, !dbg !17
// CHECK:STDOUT:   %.loc21_37.5.w = getelementptr inbounds nuw { i1, i32 }, ptr %c.var, i32 0, i32 1, !dbg !17
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %c.var, ptr align 4 @C.val.310.loc16_3, i64 8, i1 false), !dbg !16
// CHECK:STDOUT:   %C.GetT.call = call i32 @_CGetT.C.Main.64ccbb8e5d9a0b8e(ptr %c.var), !dbg !18
// CHECK:STDOUT:   ret i32 %C.GetT.call, !dbg !19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CAccessEmpty.Main() #0 !dbg !20 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c.var = alloca { i1, {} }, align 8, !dbg !23
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !23
// CHECK:STDOUT:   %.loc26_37.2.v = getelementptr inbounds nuw { i1, {} }, ptr %c.var, i32 0, i32 0, !dbg !24
// CHECK:STDOUT:   %.loc26_37.4.w = getelementptr inbounds nuw { i1, {} }, ptr %c.var, i32 0, i32 1, !dbg !24
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %c.var, ptr align 1 @C.val.8d5.loc26_3, i64 1, i1 false), !dbg !23
// CHECK:STDOUT:   call void @_CGetT.C.Main.335f6bfc3bd91486(ptr %c.var), !dbg !25
// CHECK:STDOUT:   ret void, !dbg !26
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CAccessTuple.Main(ptr sret({ i32, i32, i32 }) %return) #0 !dbg !27 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c.var = alloca { i1, { i32, i32, i32 } }, align 8, !dbg !28
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !28
// CHECK:STDOUT:   %.loc31_57.2.v = getelementptr inbounds nuw { i1, { i32, i32, i32 } }, ptr %c.var, i32 0, i32 0, !dbg !29
// CHECK:STDOUT:   %.loc31_57.4.w = getelementptr inbounds nuw { i1, { i32, i32, i32 } }, ptr %c.var, i32 0, i32 1, !dbg !29
// CHECK:STDOUT:   %tuple.elem0.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %.loc31_57.4.w, i32 0, i32 0, !dbg !30
// CHECK:STDOUT:   %tuple.elem1.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %.loc31_57.4.w, i32 0, i32 1, !dbg !30
// CHECK:STDOUT:   %tuple.elem2.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %.loc31_57.4.w, i32 0, i32 2, !dbg !30
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %c.var, ptr align 4 @C.val.76f.loc31_3, i64 16, i1 false), !dbg !28
// CHECK:STDOUT:   call void @_CGetT.C.Main.3ae0849a77989a11(ptr %return, ptr %c.var), !dbg !31
// CHECK:STDOUT:   ret void, !dbg !32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @_CGetBool.C.Main.64ccbb8e5d9a0b8e(ptr %self) #0 !dbg !33 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc6_16.1.v = getelementptr inbounds nuw { i1, i32 }, ptr %self, i32 0, i32 0, !dbg !38
// CHECK:STDOUT:   %.loc6_16.2 = load i8, ptr %.loc6_16.1.v, align 1, !dbg !38
// CHECK:STDOUT:   %.loc6_16.21 = trunc i8 %.loc6_16.2 to i1, !dbg !38
// CHECK:STDOUT:   ret i1 %.loc6_16.21, !dbg !39
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CGetT.C.Main.64ccbb8e5d9a0b8e(ptr %self) #0 !dbg !40 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc9_16.1.w = getelementptr inbounds nuw { i1, i32 }, ptr %self, i32 0, i32 1, !dbg !45
// CHECK:STDOUT:   %.loc9_16.2 = load i32, ptr %.loc9_16.1.w, align 4, !dbg !45
// CHECK:STDOUT:   ret i32 %.loc9_16.2, !dbg !46
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CGetT.C.Main.335f6bfc3bd91486(ptr %self) #0 !dbg !47 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc9_16.1.w = getelementptr inbounds nuw { i1, {} }, ptr %self, i32 0, i32 1, !dbg !52
// CHECK:STDOUT:   ret void, !dbg !53
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CGetT.C.Main.3ae0849a77989a11(ptr sret({ i32, i32, i32 }) %return, ptr %self) #0 !dbg !54 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc9_16.1.w = getelementptr inbounds nuw { i1, { i32, i32, i32 } }, ptr %self, i32 0, i32 1, !dbg !57
// CHECK:STDOUT:   call void @"_COp.a02bcb1b1383357d:Copy.Core.975a30a52004fa2c"(ptr %return, ptr %.loc9_16.1.w), !dbg !57
// CHECK:STDOUT:   ret void, !dbg !58
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp.a02bcb1b1383357d:Copy.Core.975a30a52004fa2c"(ptr sret({ i32, i32, i32 }) %return, ptr %self) #0 !dbg !59 {
// CHECK:STDOUT:   %tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %self, i32 0, i32 0, !dbg !63
// CHECK:STDOUT:   %tuple.elem.load = load i32, ptr %tuple.elem, align 4, !dbg !63
// CHECK:STDOUT:   %tuple.elem1 = getelementptr inbounds nuw { i32, i32, i32 }, ptr %return, i32 0, i32 0, !dbg !64
// CHECK:STDOUT:   %tuple.elem2 = getelementptr inbounds nuw { i32, i32, i32 }, ptr %self, i32 0, i32 1, !dbg !65
// CHECK:STDOUT:   %tuple.elem.load3 = load i32, ptr %tuple.elem2, align 4, !dbg !65
// CHECK:STDOUT:   %tuple.elem4 = getelementptr inbounds nuw { i32, i32, i32 }, ptr %return, i32 0, i32 1, !dbg !64
// CHECK:STDOUT:   %tuple.elem5 = getelementptr inbounds nuw { i32, i32, i32 }, ptr %self, i32 0, i32 2, !dbg !66
// CHECK:STDOUT:   %tuple.elem.load6 = load i32, ptr %tuple.elem5, align 4, !dbg !66
// CHECK:STDOUT:   %tuple.elem7 = getelementptr inbounds nuw { i32, i32, i32 }, ptr %return, i32 0, i32 2, !dbg !64
// CHECK:STDOUT:   store i32 %tuple.elem.load, ptr %tuple.elem1, align 4, !dbg !64
// CHECK:STDOUT:   store i32 %tuple.elem.load3, ptr %tuple.elem4, align 4, !dbg !64
// CHECK:STDOUT:   store i32 %tuple.elem.load6, ptr %tuple.elem7, align 4, !dbg !64
// CHECK:STDOUT:   ret void, !dbg !67
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 3, 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "access.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "AccessBool", linkageName: "_CAccessBool.Main", scope: null, file: !3, line: 15, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{!7}
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 8)
// CHECK:STDOUT: !8 = !DILocation(line: 16, column: 3, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 16, column: 19, scope: !4)
// CHECK:STDOUT: !10 = !DILocation(line: 17, column: 10, scope: !4)
// CHECK:STDOUT: !11 = !DILocation(line: 17, column: 3, scope: !4)
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "AccessInt", linkageName: "_CAccessInt.Main", scope: null, file: !3, line: 20, type: !13, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !13 = !DISubroutineType(types: !14)
// CHECK:STDOUT: !14 = !{!15}
// CHECK:STDOUT: !15 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !16 = !DILocation(line: 21, column: 3, scope: !12)
// CHECK:STDOUT: !17 = !DILocation(line: 21, column: 19, scope: !12)
// CHECK:STDOUT: !18 = !DILocation(line: 22, column: 10, scope: !12)
// CHECK:STDOUT: !19 = !DILocation(line: 22, column: 3, scope: !12)
// CHECK:STDOUT: !20 = distinct !DISubprogram(name: "AccessEmpty", linkageName: "_CAccessEmpty.Main", scope: null, file: !3, line: 25, type: !21, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !21 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !22 = !{null}
// CHECK:STDOUT: !23 = !DILocation(line: 26, column: 3, scope: !20)
// CHECK:STDOUT: !24 = !DILocation(line: 26, column: 18, scope: !20)
// CHECK:STDOUT: !25 = !DILocation(line: 27, column: 10, scope: !20)
// CHECK:STDOUT: !26 = !DILocation(line: 27, column: 3, scope: !20)
// CHECK:STDOUT: !27 = distinct !DISubprogram(name: "AccessTuple", linkageName: "_CAccessTuple.Main", scope: null, file: !3, line: 30, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !28 = !DILocation(line: 31, column: 3, scope: !27)
// CHECK:STDOUT: !29 = !DILocation(line: 31, column: 31, scope: !27)
// CHECK:STDOUT: !30 = !DILocation(line: 31, column: 48, scope: !27)
// CHECK:STDOUT: !31 = !DILocation(line: 32, column: 10, scope: !27)
// CHECK:STDOUT: !32 = !DILocation(line: 32, column: 3, scope: !27)
// CHECK:STDOUT: !33 = distinct !DISubprogram(name: "GetBool", linkageName: "_CGetBool.C.Main.64ccbb8e5d9a0b8e", scope: null, file: !3, line: 5, type: !34, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !36)
// CHECK:STDOUT: !34 = !DISubroutineType(types: !35)
// CHECK:STDOUT: !35 = !{!7, !7}
// CHECK:STDOUT: !36 = !{!37}
// CHECK:STDOUT: !37 = !DILocalVariable(arg: 1, scope: !33, type: !7)
// CHECK:STDOUT: !38 = !DILocation(line: 6, column: 12, scope: !33)
// CHECK:STDOUT: !39 = !DILocation(line: 6, column: 5, scope: !33)
// CHECK:STDOUT: !40 = distinct !DISubprogram(name: "GetT", linkageName: "_CGetT.C.Main.64ccbb8e5d9a0b8e", scope: null, file: !3, line: 8, type: !41, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !43)
// CHECK:STDOUT: !41 = !DISubroutineType(types: !42)
// CHECK:STDOUT: !42 = !{!15, !7}
// CHECK:STDOUT: !43 = !{!44}
// CHECK:STDOUT: !44 = !DILocalVariable(arg: 1, scope: !40, type: !7)
// CHECK:STDOUT: !45 = !DILocation(line: 9, column: 12, scope: !40)
// CHECK:STDOUT: !46 = !DILocation(line: 9, column: 5, scope: !40)
// CHECK:STDOUT: !47 = distinct !DISubprogram(name: "GetT", linkageName: "_CGetT.C.Main.335f6bfc3bd91486", scope: null, file: !3, line: 8, type: !48, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !50)
// CHECK:STDOUT: !48 = !DISubroutineType(types: !49)
// CHECK:STDOUT: !49 = !{null, !7}
// CHECK:STDOUT: !50 = !{!51}
// CHECK:STDOUT: !51 = !DILocalVariable(arg: 1, scope: !47, type: !7)
// CHECK:STDOUT: !52 = !DILocation(line: 9, column: 12, scope: !47)
// CHECK:STDOUT: !53 = !DILocation(line: 9, column: 5, scope: !47)
// CHECK:STDOUT: !54 = distinct !DISubprogram(name: "GetT", linkageName: "_CGetT.C.Main.3ae0849a77989a11", scope: null, file: !3, line: 8, type: !34, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !55)
// CHECK:STDOUT: !55 = !{!56}
// CHECK:STDOUT: !56 = !DILocalVariable(arg: 1, scope: !54, type: !7)
// CHECK:STDOUT: !57 = !DILocation(line: 9, column: 12, scope: !54)
// CHECK:STDOUT: !58 = !DILocation(line: 9, column: 5, scope: !54)
// CHECK:STDOUT: !59 = distinct !DISubprogram(name: "Op", linkageName: "_COp.a02bcb1b1383357d:Copy.Core.975a30a52004fa2c", scope: null, file: !60, line: 54, type: !34, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !61)
// CHECK:STDOUT: !60 = !DIFile(filename: "min_prelude/parts/copy.carbon", directory: "")
// CHECK:STDOUT: !61 = !{!62}
// CHECK:STDOUT: !62 = !DILocalVariable(arg: 1, scope: !59, type: !7)
// CHECK:STDOUT: !63 = !DILocation(line: 55, column: 13, scope: !59)
// CHECK:STDOUT: !64 = !DILocation(line: 55, column: 12, scope: !59)
// CHECK:STDOUT: !65 = !DILocation(line: 55, column: 26, scope: !59)
// CHECK:STDOUT: !66 = !DILocation(line: 55, column: 39, scope: !59)
// CHECK:STDOUT: !67 = !DILocation(line: 55, column: 5, scope: !59)
