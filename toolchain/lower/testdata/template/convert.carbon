// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/template/convert.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/template/convert.carbon

// --- convert.carbon

library "[[@TEST_NAME]]";

fn F[template T: type](x: T) -> i32 {
  let n: i32 = x;
  return n;
}

fn Test1(n: i32) -> i32 {
  return F(n);
}

class C {
  var n: i32;
  impl as Core.ImplicitAs(i32) {
    fn Convert(self) -> i32 { return self.n; }
  }
}

fn Test2(c: C) -> i32 {
  return F(c);
}

// --- initialize_dependent_target_from_literal.carbon

library "[[@TEST_NAME]]";

fn F(template T: type) {
  var unused w: T = 0;
}

fn G() {
  F(i32);
}

// --- initialize_primitive.carbon

library "[[@TEST_NAME]]";

fn F[template T: type](x: T) {
  var unused w: i32 = x;
}

fn G() {
  F(0 as i32);
}

// --- initialize_from_tuple.carbon

library "[[@TEST_NAME]]";

class X {}
fn Make() -> X;

fn F(template T: type) {
  var unused w: T = (Make(), Make(), Make());
}

class Y {}

impl (X, X, X) as Core.ImplicitAs(Y) {
  fn Convert(unused self: (X, X, X)) -> Y { return {}; }
}

fn G() {
  F((X, X, X));

  F(array(X, 3));

  F(Y);
}

// --- initialize_from_nested_literal.carbon

library "[[@TEST_NAME]]";

class X {}
fn Make() -> X;

//@dump-sem-ir-begin
fn F(template T: type) {
  var unused w: T = ({.a = Make(), .b = Make()}, Make());
}
//@dump-sem-ir-end

class Y {
  impl X as Core.ImplicitAs(Y) {
    fn Convert(self: X) -> Y;
  }
}

class Z {
  var a: X;
  var b: Y;
}

fn G() {
  F(({.a: X, .b: X}, X));

  F(({.a: Y, .b: Y}, Y));

  F((Z, Y));
}

// --- as.carbon

library "[[@TEST_NAME]]";

fn AsI32[template T: type](x: T) -> i32 {
  return x as i32;
}

fn Test1(n: i32) {
  let _: i32 = AsI32(n);
}

class C {
  var n: i32;
  impl as Core.As(i32) {
    fn Convert(self) -> i32 { return self.n; }
  }
}

fn Test2(c: C) {
  let _: i32 = AsI32(c);
}

// --- as_type.carbon

library "[[@TEST_NAME]]";

fn AsType[template T: type](x: T, template U: type) {
  let _: U = x as U;
}

fn Test1(n: i32) {
  AsType(n, i32);
}

class C {
  var n: i32;
  impl as Core.As(i32) {
    fn Convert(self) -> i32 { return self.n; }
  }
}

fn Test2(c: C) {
  AsType(c, i32);
}

// --- initialize_from_template_method_call.carbon

library "[[@TEST_NAME]]";

class B {
  fn MakeB() -> B {
    return {};
  }
}

class A {
  var field: B;
}

fn MakeA(template T: type) -> A {
   return {
     .field = T.MakeB(),
   };
}

fn Run() {
  MakeA(B);
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'convert.carbon'
// CHECK:STDOUT: source_filename = "convert.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CTest1.Main(i32 %n) #0 !dbg !42 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %F.call = call i32 @_CF.Main.55456748a8a2eb70(i32 %n), !dbg !48
// CHECK:STDOUT:   ret i32 %F.call, !dbg !49
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @"_CConvert.C.Main:ImplicitAs.55456748a8a2eb70.Core"(ptr %self) #0 !dbg !51 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc16_42.1.n = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !56
// CHECK:STDOUT:   %.loc16_42.2 = load i32, ptr %.loc16_42.1.n, align 4, !dbg !56
// CHECK:STDOUT:   ret i32 %.loc16_42.2, !dbg !57
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CTest2.Main(ptr %c) #0 !dbg !59 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %F.call = call i32 @_CF.Main.9522ff908e97e267(ptr %c), !dbg !65
// CHECK:STDOUT:   ret i32 %F.call, !dbg !66
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CF.Main.55456748a8a2eb70(i32 %x) #0 !dbg !68 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %x, !dbg !77
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CF.Main.9522ff908e97e267(ptr %x) #0 !dbg !79 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.call = call i32 @"_CConvert.C.Main:ImplicitAs.55456748a8a2eb70.Core"(ptr %x), !dbg !84
// CHECK:STDOUT:   ret i32 %C.as.ImplicitAs.impl.Convert.call, !dbg !88
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!35}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !34 = !DIFile(filename: "convert.carbon", directory: "")
// CHECK:STDOUT: !35 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !34, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !36 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !37 = !{!36, !36}
// CHECK:STDOUT: !38 = !DISubroutineType(types: !37)
// CHECK:STDOUT: !39 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !40 = !{!36, !39}
// CHECK:STDOUT: !41 = !DISubroutineType(types: !40)
// CHECK:STDOUT: !42 = distinct !DISubprogram(name: "Test1", linkageName: "_CTest1.Main", scope: null, file: !34, line: 9, type: !38, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !50)
// CHECK:STDOUT: !43 = !DILocalVariable(arg: 1, scope: !42, type: !36)
// CHECK:STDOUT: !48 = !DILocation(line: 10, column: 10, scope: !42)
// CHECK:STDOUT: !49 = !DILocation(line: 10, column: 3, scope: !42)
// CHECK:STDOUT: !50 = !{!43}
// CHECK:STDOUT: !51 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.C.Main:ImplicitAs.55456748a8a2eb70.Core", scope: null, file: !34, line: 16, type: !41, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !58)
// CHECK:STDOUT: !52 = !DILocalVariable(arg: 1, scope: !51, type: !39)
// CHECK:STDOUT: !56 = !DILocation(line: 16, column: 38, scope: !51)
// CHECK:STDOUT: !57 = !DILocation(line: 16, column: 31, scope: !51)
// CHECK:STDOUT: !58 = !{!52}
// CHECK:STDOUT: !59 = distinct !DISubprogram(name: "Test2", linkageName: "_CTest2.Main", scope: null, file: !34, line: 20, type: !41, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !67)
// CHECK:STDOUT: !60 = !DILocalVariable(arg: 1, scope: !59, type: !39)
// CHECK:STDOUT: !65 = !DILocation(line: 21, column: 10, scope: !59)
// CHECK:STDOUT: !66 = !DILocation(line: 21, column: 3, scope: !59)
// CHECK:STDOUT: !67 = !{!60}
// CHECK:STDOUT: !68 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.55456748a8a2eb70", scope: null, file: !34, line: 4, type: !38, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !78)
// CHECK:STDOUT: !69 = !DILocalVariable(arg: 1, scope: !68, type: !36)
// CHECK:STDOUT: !77 = !DILocation(line: 6, column: 3, scope: !68)
// CHECK:STDOUT: !78 = !{!69}
// CHECK:STDOUT: !79 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.9522ff908e97e267", scope: null, file: !34, line: 4, type: !41, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !89)
// CHECK:STDOUT: !80 = !DILocalVariable(arg: 1, scope: !79, type: !39)
// CHECK:STDOUT: !84 = !DILocation(line: 5, column: 16, scope: !79)
// CHECK:STDOUT: !88 = !DILocation(line: 6, column: 3, scope: !79)
// CHECK:STDOUT: !89 = !{!80}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'initialize_dependent_target_from_literal.carbon'
// CHECK:STDOUT: source_filename = "initialize_dependent_target_from_literal.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.Main() #0 !dbg !94 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Main.55456748a8a2eb70(), !dbg !95
// CHECK:STDOUT:   ret void, !dbg !96
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.55456748a8a2eb70() #0 !dbg !97 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %w.var = alloca i32, align 4, !dbg !98
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %w.var), !dbg !98
// CHECK:STDOUT:   store i32 0, ptr %w.var, align 4, !dbg !98
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.dc5658e0ac74b2aa(ptr %w.var), !dbg !98
// CHECK:STDOUT:   ret void, !dbg !102
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.dc5658e0ac74b2aa(ptr %t) #0 !dbg !103 {
// CHECK:STDOUT:   ret void, !dbg !108
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!91}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !26 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !36 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !90 = !DIFile(filename: "initialize_dependent_target_from_literal.carbon", directory: "")
// CHECK:STDOUT: !91 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !90, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !92 = !{null}
// CHECK:STDOUT: !93 = !DISubroutineType(types: !92)
// CHECK:STDOUT: !94 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !90, line: 8, type: !93, spFlags: DISPFlagDefinition, unit: !91)
// CHECK:STDOUT: !95 = !DILocation(line: 9, column: 3, scope: !94)
// CHECK:STDOUT: !96 = !DILocation(line: 8, column: 1, scope: !94)
// CHECK:STDOUT: !97 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.55456748a8a2eb70", scope: null, file: !90, line: 4, type: !93, spFlags: DISPFlagDefinition, unit: !91)
// CHECK:STDOUT: !98 = !DILocation(line: 5, column: 3, scope: !97)
// CHECK:STDOUT: !100 = !{null, !36}
// CHECK:STDOUT: !101 = !DISubroutineType(types: !100)
// CHECK:STDOUT: !102 = !DILocation(line: 4, column: 1, scope: !97)
// CHECK:STDOUT: !103 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.dc5658e0ac74b2aa", scope: null, file: !26, line: 30, type: !101, spFlags: DISPFlagDefinition, unit: !91, retainedNodes: !109)
// CHECK:STDOUT: !104 = !DILocalVariable(arg: 1, scope: !103, type: !36)
// CHECK:STDOUT: !108 = !DILocation(line: 30, column: 1, scope: !103)
// CHECK:STDOUT: !109 = !{!104}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'initialize_primitive.carbon'
// CHECK:STDOUT: source_filename = "initialize_primitive.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.Main() #0 !dbg !112 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Main.55456748a8a2eb70(i32 0), !dbg !113
// CHECK:STDOUT:   ret void, !dbg !114
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.55456748a8a2eb70(i32 %x) #0 !dbg !115 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %w.var = alloca i32, align 4, !dbg !118
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %w.var), !dbg !118
// CHECK:STDOUT:   store i32 %x, ptr %w.var, align 4, !dbg !118
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.dc5658e0ac74b2aa(ptr %w.var), !dbg !118
// CHECK:STDOUT:   ret void, !dbg !121
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.dc5658e0ac74b2aa(ptr %t) #0 !dbg !123 {
// CHECK:STDOUT:   ret void, !dbg !128
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!111}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !26 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !36 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !92 = !{null}
// CHECK:STDOUT: !93 = !DISubroutineType(types: !92)
// CHECK:STDOUT: !100 = !{null, !36}
// CHECK:STDOUT: !101 = !DISubroutineType(types: !100)
// CHECK:STDOUT: !110 = !DIFile(filename: "initialize_primitive.carbon", directory: "")
// CHECK:STDOUT: !111 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !110, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !112 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !110, line: 8, type: !93, spFlags: DISPFlagDefinition, unit: !111)
// CHECK:STDOUT: !113 = !DILocation(line: 9, column: 3, scope: !112)
// CHECK:STDOUT: !114 = !DILocation(line: 8, column: 1, scope: !112)
// CHECK:STDOUT: !115 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.55456748a8a2eb70", scope: null, file: !110, line: 4, type: !101, spFlags: DISPFlagDefinition, unit: !111, retainedNodes: !122)
// CHECK:STDOUT: !116 = !DILocalVariable(arg: 1, scope: !115, type: !36)
// CHECK:STDOUT: !118 = !DILocation(line: 5, column: 3, scope: !115)
// CHECK:STDOUT: !121 = !DILocation(line: 4, column: 1, scope: !115)
// CHECK:STDOUT: !122 = !{!116}
// CHECK:STDOUT: !123 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.dc5658e0ac74b2aa", scope: null, file: !26, line: 30, type: !101, spFlags: DISPFlagDefinition, unit: !111, retainedNodes: !129)
// CHECK:STDOUT: !124 = !DILocalVariable(arg: 1, scope: !123, type: !36)
// CHECK:STDOUT: !128 = !DILocation(line: 30, column: 1, scope: !123)
// CHECK:STDOUT: !129 = !{!124}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'initialize_from_tuple.carbon'
// CHECK:STDOUT: source_filename = "initialize_from_tuple.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @Y.val.loc14_54 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CMake.Main(ptr sret({}))
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_CConvert.92e03c783d6af646:ImplicitAs.be297cc7300419fe.Core"(ptr sret({}) %return, ptr %self) #0 !dbg !138 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %return, ptr align 1 @Y.val.loc14_54, i64 0, i1 false), !dbg !143
// CHECK:STDOUT:   ret void, !dbg !143
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.Main() #0 !dbg !145 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Main.fb3d597a6b1c4ee1(), !dbg !146
// CHECK:STDOUT:   call void @_CF.Main.689d8af583927bab(), !dbg !147
// CHECK:STDOUT:   call void @_CF.Main.be297cc7300419fe(), !dbg !148
// CHECK:STDOUT:   ret void, !dbg !149
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.92e03c783d6af646:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !150 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %tuple.elem2.tuple.elem = getelementptr inbounds nuw { {}, {}, {} }, ptr %self, i32 0, i32 2, !dbg !152
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.30ecf861811bc6e0(ptr %tuple.elem2.tuple.elem), !dbg !152
// CHECK:STDOUT:   %tuple.elem1.tuple.elem = getelementptr inbounds nuw { {}, {}, {} }, ptr %self, i32 0, i32 1, !dbg !152
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.30ecf861811bc6e0(ptr %tuple.elem1.tuple.elem), !dbg !152
// CHECK:STDOUT:   %tuple.elem0.tuple.elem = getelementptr inbounds nuw { {}, {}, {} }, ptr %self, i32 0, i32 0, !dbg !152
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.30ecf861811bc6e0(ptr %tuple.elem0.tuple.elem), !dbg !152
// CHECK:STDOUT:   ret void, !dbg !152
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.2ee5d57da69a520f:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !154 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc8_3.1.array.index = getelementptr inbounds [3 x {}], ptr %self, i32 0, i64 2, !dbg !156
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.30ecf861811bc6e0(ptr %.loc8_3.1.array.index), !dbg !156
// CHECK:STDOUT:   %.loc8_3.4.array.index = getelementptr inbounds [3 x {}], ptr %self, i32 0, i64 1, !dbg !156
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.30ecf861811bc6e0(ptr %.loc8_3.4.array.index), !dbg !156
// CHECK:STDOUT:   %.loc8_3.7.array.index = getelementptr inbounds [3 x {}], ptr %self, i32 0, i64 0, !dbg !156
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.30ecf861811bc6e0(ptr %.loc8_3.7.array.index), !dbg !156
// CHECK:STDOUT:   ret void, !dbg !156
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.fb3d597a6b1c4ee1() #0 !dbg !158 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %w.var = alloca { {}, {}, {} }, align 1, !dbg !159
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %w.var), !dbg !159
// CHECK:STDOUT:   %tuple.elem0.tuple.elem = getelementptr inbounds nuw { {}, {}, {} }, ptr %w.var, i32 0, i32 0, !dbg !160
// CHECK:STDOUT:   call void @_CMake.Main(ptr %tuple.elem0.tuple.elem), !dbg !161
// CHECK:STDOUT:   %tuple.elem1.tuple.elem = getelementptr inbounds nuw { {}, {}, {} }, ptr %w.var, i32 0, i32 1, !dbg !160
// CHECK:STDOUT:   call void @_CMake.Main(ptr %tuple.elem1.tuple.elem), !dbg !162
// CHECK:STDOUT:   %tuple.elem2.tuple.elem = getelementptr inbounds nuw { {}, {}, {} }, ptr %w.var, i32 0, i32 2, !dbg !160
// CHECK:STDOUT:   call void @_CMake.Main(ptr %tuple.elem2.tuple.elem), !dbg !163
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.6add1b2daa1190e3(ptr %w.var), !dbg !159
// CHECK:STDOUT:   ret void, !dbg !165
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.689d8af583927bab() #0 !dbg !166 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %w.var = alloca [3 x {}], align 1, !dbg !167
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %w.var), !dbg !167
// CHECK:STDOUT:   %.b44.array.index = getelementptr inbounds [3 x {}], ptr %w.var, i32 0, i64 0, !dbg !168
// CHECK:STDOUT:   call void @_CMake.Main(ptr %.b44.array.index), !dbg !169
// CHECK:STDOUT:   %.5b6.array.index = getelementptr inbounds [3 x {}], ptr %w.var, i32 0, i64 1, !dbg !168
// CHECK:STDOUT:   call void @_CMake.Main(ptr %.5b6.array.index), !dbg !170
// CHECK:STDOUT:   %.97d.array.index = getelementptr inbounds [3 x {}], ptr %w.var, i32 0, i64 2, !dbg !168
// CHECK:STDOUT:   call void @_CMake.Main(ptr %.97d.array.index), !dbg !171
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.a8283838dc120d4f(ptr %w.var), !dbg !167
// CHECK:STDOUT:   ret void, !dbg !173
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.be297cc7300419fe() #0 !dbg !174 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %w.var = alloca {}, align 1, !dbg !175
// CHECK:STDOUT:   %.7a56fd.1.temp = alloca {}, align 1, !dbg !175
// CHECK:STDOUT:   %.7a56fd.2.temp = alloca {}, align 1, !dbg !175
// CHECK:STDOUT:   %.7a56fd.3.temp = alloca {}, align 1, !dbg !175
// CHECK:STDOUT:   %tuple.9aa = alloca { ptr, ptr, ptr }, align 8, !dbg !179
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %w.var), !dbg !175
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.7a56fd.1.temp), !dbg !175
// CHECK:STDOUT:   call void @_CMake.Main(ptr %.7a56fd.1.temp), !dbg !176
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.7a56fd.2.temp), !dbg !175
// CHECK:STDOUT:   call void @_CMake.Main(ptr %.7a56fd.2.temp), !dbg !177
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.7a56fd.3.temp), !dbg !175
// CHECK:STDOUT:   call void @_CMake.Main(ptr %.7a56fd.3.temp), !dbg !178
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %tuple.9aa), !dbg !179
// CHECK:STDOUT:   %tuple.9aa1 = getelementptr inbounds nuw { ptr, ptr, ptr }, ptr %tuple.9aa, i32 0, i32 0, !dbg !179
// CHECK:STDOUT:   store ptr %.7a56fd.1.temp, ptr %tuple.9aa1, align 8, !dbg !179
// CHECK:STDOUT:   %tuple.9aa2 = getelementptr inbounds nuw { ptr, ptr, ptr }, ptr %tuple.9aa, i32 0, i32 1, !dbg !179
// CHECK:STDOUT:   store ptr %.7a56fd.2.temp, ptr %tuple.9aa2, align 8, !dbg !179
// CHECK:STDOUT:   %tuple.9aa3 = getelementptr inbounds nuw { ptr, ptr, ptr }, ptr %tuple.9aa, i32 0, i32 2, !dbg !179
// CHECK:STDOUT:   store ptr %.7a56fd.3.temp, ptr %tuple.9aa3, align 8, !dbg !179
// CHECK:STDOUT:   call void @"_CConvert.92e03c783d6af646:ImplicitAs.be297cc7300419fe.Core"(ptr %w.var, ptr %tuple.9aa), !dbg !175
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.3773d92a34eaed6b(ptr %w.var), !dbg !175
// CHECK:STDOUT:   ret void, !dbg !181
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.30ecf861811bc6e0(ptr %t) #0 !dbg !182 {
// CHECK:STDOUT:   ret void, !dbg !187
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.6add1b2daa1190e3(ptr %t) #0 !dbg !189 {
// CHECK:STDOUT:   call void @"_COp.92e03c783d6af646:core.SubobjectDestroy.Core"(ptr %t), !dbg !193
// CHECK:STDOUT:   ret void, !dbg !194
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.a8283838dc120d4f(ptr %t) #0 !dbg !196 {
// CHECK:STDOUT:   call void @"_COp.2ee5d57da69a520f:core.SubobjectDestroy.Core"(ptr %t), !dbg !200
// CHECK:STDOUT:   ret void, !dbg !201
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.3773d92a34eaed6b(ptr %t) #0 !dbg !203 {
// CHECK:STDOUT:   ret void, !dbg !208
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.30ecf861811bc6e0, { 5, 4, 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 6, 5, 4, 3, 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!131}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !26 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !39 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !92 = !{null}
// CHECK:STDOUT: !93 = !DISubroutineType(types: !92)
// CHECK:STDOUT: !130 = !DIFile(filename: "initialize_from_tuple.carbon", directory: "")
// CHECK:STDOUT: !131 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !130, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !134 = !{!39, !39}
// CHECK:STDOUT: !135 = !DISubroutineType(types: !134)
// CHECK:STDOUT: !136 = !{null, !39}
// CHECK:STDOUT: !137 = !DISubroutineType(types: !136)
// CHECK:STDOUT: !138 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.92e03c783d6af646:ImplicitAs.be297cc7300419fe.Core", scope: null, file: !130, line: 14, type: !135, spFlags: DISPFlagDefinition, unit: !131, retainedNodes: !144)
// CHECK:STDOUT: !139 = !DILocalVariable(arg: 1, scope: !138, type: !39)
// CHECK:STDOUT: !143 = !DILocation(line: 14, column: 45, scope: !138)
// CHECK:STDOUT: !144 = !{!139}
// CHECK:STDOUT: !145 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !130, line: 17, type: !93, spFlags: DISPFlagDefinition, unit: !131)
// CHECK:STDOUT: !146 = !DILocation(line: 18, column: 3, scope: !145)
// CHECK:STDOUT: !147 = !DILocation(line: 20, column: 3, scope: !145)
// CHECK:STDOUT: !148 = !DILocation(line: 22, column: 3, scope: !145)
// CHECK:STDOUT: !149 = !DILocation(line: 17, column: 1, scope: !145)
// CHECK:STDOUT: !150 = distinct !DISubprogram(name: "Op", linkageName: "_COp.92e03c783d6af646:core.SubobjectDestroy.Core", scope: null, file: !130, line: 8, type: !137, spFlags: DISPFlagDefinition, unit: !131, retainedNodes: !153)
// CHECK:STDOUT: !151 = !DILocalVariable(arg: 1, scope: !150, type: !39)
// CHECK:STDOUT: !152 = !DILocation(line: 8, column: 3, scope: !150)
// CHECK:STDOUT: !153 = !{!151}
// CHECK:STDOUT: !154 = distinct !DISubprogram(name: "Op", linkageName: "_COp.2ee5d57da69a520f:core.SubobjectDestroy.Core", scope: null, file: !130, line: 8, type: !137, spFlags: DISPFlagDefinition, unit: !131, retainedNodes: !157)
// CHECK:STDOUT: !155 = !DILocalVariable(arg: 1, scope: !154, type: !39)
// CHECK:STDOUT: !156 = !DILocation(line: 8, column: 3, scope: !154)
// CHECK:STDOUT: !157 = !{!155}
// CHECK:STDOUT: !158 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.fb3d597a6b1c4ee1", scope: null, file: !130, line: 7, type: !93, spFlags: DISPFlagDefinition, unit: !131)
// CHECK:STDOUT: !159 = !DILocation(line: 8, column: 3, scope: !158)
// CHECK:STDOUT: !160 = !DILocation(line: 8, column: 21, scope: !158)
// CHECK:STDOUT: !161 = !DILocation(line: 8, column: 22, scope: !158)
// CHECK:STDOUT: !162 = !DILocation(line: 8, column: 30, scope: !158)
// CHECK:STDOUT: !163 = !DILocation(line: 8, column: 38, scope: !158)
// CHECK:STDOUT: !165 = !DILocation(line: 7, column: 1, scope: !158)
// CHECK:STDOUT: !166 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.689d8af583927bab", scope: null, file: !130, line: 7, type: !93, spFlags: DISPFlagDefinition, unit: !131)
// CHECK:STDOUT: !167 = !DILocation(line: 8, column: 3, scope: !166)
// CHECK:STDOUT: !168 = !DILocation(line: 8, column: 21, scope: !166)
// CHECK:STDOUT: !169 = !DILocation(line: 8, column: 22, scope: !166)
// CHECK:STDOUT: !170 = !DILocation(line: 8, column: 30, scope: !166)
// CHECK:STDOUT: !171 = !DILocation(line: 8, column: 38, scope: !166)
// CHECK:STDOUT: !173 = !DILocation(line: 7, column: 1, scope: !166)
// CHECK:STDOUT: !174 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.be297cc7300419fe", scope: null, file: !130, line: 7, type: !93, spFlags: DISPFlagDefinition, unit: !131)
// CHECK:STDOUT: !175 = !DILocation(line: 8, column: 3, scope: !174)
// CHECK:STDOUT: !176 = !DILocation(line: 8, column: 22, scope: !174)
// CHECK:STDOUT: !177 = !DILocation(line: 8, column: 30, scope: !174)
// CHECK:STDOUT: !178 = !DILocation(line: 8, column: 38, scope: !174)
// CHECK:STDOUT: !179 = !DILocation(line: 8, column: 21, scope: !174)
// CHECK:STDOUT: !181 = !DILocation(line: 7, column: 1, scope: !174)
// CHECK:STDOUT: !182 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.30ecf861811bc6e0", scope: null, file: !26, line: 30, type: !137, spFlags: DISPFlagDefinition, unit: !131, retainedNodes: !188)
// CHECK:STDOUT: !183 = !DILocalVariable(arg: 1, scope: !182, type: !39)
// CHECK:STDOUT: !187 = !DILocation(line: 30, column: 1, scope: !182)
// CHECK:STDOUT: !188 = !{!183}
// CHECK:STDOUT: !189 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.6add1b2daa1190e3", scope: null, file: !26, line: 30, type: !137, spFlags: DISPFlagDefinition, unit: !131, retainedNodes: !195)
// CHECK:STDOUT: !190 = !DILocalVariable(arg: 1, scope: !189, type: !39)
// CHECK:STDOUT: !193 = !DILocation(line: 32, column: 5, scope: !189)
// CHECK:STDOUT: !194 = !DILocation(line: 30, column: 1, scope: !189)
// CHECK:STDOUT: !195 = !{!190}
// CHECK:STDOUT: !196 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.a8283838dc120d4f", scope: null, file: !26, line: 30, type: !137, spFlags: DISPFlagDefinition, unit: !131, retainedNodes: !202)
// CHECK:STDOUT: !197 = !DILocalVariable(arg: 1, scope: !196, type: !39)
// CHECK:STDOUT: !200 = !DILocation(line: 32, column: 5, scope: !196)
// CHECK:STDOUT: !201 = !DILocation(line: 30, column: 1, scope: !196)
// CHECK:STDOUT: !202 = !{!197}
// CHECK:STDOUT: !203 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.3773d92a34eaed6b", scope: null, file: !26, line: 30, type: !137, spFlags: DISPFlagDefinition, unit: !131, retainedNodes: !209)
// CHECK:STDOUT: !204 = !DILocalVariable(arg: 1, scope: !203, type: !39)
// CHECK:STDOUT: !208 = !DILocation(line: 30, column: 1, scope: !203)
// CHECK:STDOUT: !209 = !{!204}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'initialize_from_nested_literal.carbon'
// CHECK:STDOUT: source_filename = "initialize_from_nested_literal.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CMake.Main(ptr sret({}))
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_CConvert.X.Main:ImplicitAs.be297cc7300419fe.Core"(ptr sret({}), ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.Main() #0 !dbg !212 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Main.75a9ebb2041c173b(), !dbg !213
// CHECK:STDOUT:   call void @_CF.Main.f5d9b693014ce43e(), !dbg !214
// CHECK:STDOUT:   call void @_CF.Main.84270bb113cb754b(), !dbg !215
// CHECK:STDOUT:   ret void, !dbg !216
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.f38e1c3ccdfe019b:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !217 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc9_3.1.b = getelementptr inbounds nuw { {}, {} }, ptr %self, i32 0, i32 1, !dbg !219
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.30ecf861811bc6e0(ptr %.loc9_3.1.b), !dbg !219
// CHECK:STDOUT:   %.loc9_3.4.a = getelementptr inbounds nuw { {}, {} }, ptr %self, i32 0, i32 0, !dbg !219
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.30ecf861811bc6e0(ptr %.loc9_3.4.a), !dbg !219
// CHECK:STDOUT:   ret void, !dbg !219
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.4c2c335412ec52fa:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !221 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %tuple.elem1.tuple.elem = getelementptr inbounds nuw { { {}, {} }, {} }, ptr %self, i32 0, i32 1, !dbg !223
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.30ecf861811bc6e0(ptr %tuple.elem1.tuple.elem), !dbg !223
// CHECK:STDOUT:   %tuple.elem0.tuple.elem = getelementptr inbounds nuw { { {}, {} }, {} }, ptr %self, i32 0, i32 0, !dbg !223
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.ea0a244f18930ba0(ptr %tuple.elem0.tuple.elem), !dbg !223
// CHECK:STDOUT:   ret void, !dbg !223
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.0950f8aecfaebf0b:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !225 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc9_3.1.b = getelementptr inbounds nuw { {}, {} }, ptr %self, i32 0, i32 1, !dbg !227
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.3773d92a34eaed6b(ptr %.loc9_3.1.b), !dbg !227
// CHECK:STDOUT:   %.loc9_3.4.a = getelementptr inbounds nuw { {}, {} }, ptr %self, i32 0, i32 0, !dbg !227
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.3773d92a34eaed6b(ptr %.loc9_3.4.a), !dbg !227
// CHECK:STDOUT:   ret void, !dbg !227
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.2f43e328c637647f:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !229 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %tuple.elem1.tuple.elem = getelementptr inbounds nuw { { {}, {} }, {} }, ptr %self, i32 0, i32 1, !dbg !231
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.3773d92a34eaed6b(ptr %tuple.elem1.tuple.elem), !dbg !231
// CHECK:STDOUT:   %tuple.elem0.tuple.elem = getelementptr inbounds nuw { { {}, {} }, {} }, ptr %self, i32 0, i32 0, !dbg !231
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.db6ed9edf4091b22(ptr %tuple.elem0.tuple.elem), !dbg !231
// CHECK:STDOUT:   ret void, !dbg !231
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.2970be6ba4c2615c:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !233 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc9_3.1.b = getelementptr inbounds nuw { {}, {} }, ptr %self, i32 0, i32 1, !dbg !235
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.3773d92a34eaed6b(ptr %.loc9_3.1.b), !dbg !235
// CHECK:STDOUT:   %.loc9_3.4.a = getelementptr inbounds nuw { {}, {} }, ptr %self, i32 0, i32 0, !dbg !235
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.30ecf861811bc6e0(ptr %.loc9_3.4.a), !dbg !235
// CHECK:STDOUT:   ret void, !dbg !235
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.680f7fa3ef86df7a:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !237 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc9_3.1.b = getelementptr inbounds nuw { {}, {} }, ptr %self, i32 0, i32 1, !dbg !239
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.3773d92a34eaed6b(ptr %.loc9_3.1.b), !dbg !239
// CHECK:STDOUT:   %.loc9_3.4.a = getelementptr inbounds nuw { {}, {} }, ptr %self, i32 0, i32 0, !dbg !239
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.30ecf861811bc6e0(ptr %.loc9_3.4.a), !dbg !239
// CHECK:STDOUT:   ret void, !dbg !239
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.bb4067704e9a198c:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !241 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %tuple.elem1.tuple.elem = getelementptr inbounds nuw { { {}, {} }, {} }, ptr %self, i32 0, i32 1, !dbg !243
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.3773d92a34eaed6b(ptr %tuple.elem1.tuple.elem), !dbg !243
// CHECK:STDOUT:   %tuple.elem0.tuple.elem = getelementptr inbounds nuw { { {}, {} }, {} }, ptr %self, i32 0, i32 0, !dbg !243
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.59962c5f89868b89(ptr %tuple.elem0.tuple.elem), !dbg !243
// CHECK:STDOUT:   ret void, !dbg !243
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.75a9ebb2041c173b() #0 !dbg !245 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %w.var = alloca { { {}, {} }, {} }, align 1, !dbg !246
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %w.var), !dbg !246
// CHECK:STDOUT:   %tuple.elem0.16b.tuple.elem = getelementptr inbounds nuw { { {}, {} }, {} }, ptr %w.var, i32 0, i32 0, !dbg !248
// CHECK:STDOUT:   %.ae2.a = getelementptr inbounds nuw { {}, {} }, ptr %tuple.elem0.16b.tuple.elem, i32 0, i32 0, !dbg !247
// CHECK:STDOUT:   call void @_CMake.Main(ptr %.ae2.a), !dbg !249
// CHECK:STDOUT:   %.ffd.b = getelementptr inbounds nuw { {}, {} }, ptr %tuple.elem0.16b.tuple.elem, i32 0, i32 1, !dbg !247
// CHECK:STDOUT:   call void @_CMake.Main(ptr %.ffd.b), !dbg !250
// CHECK:STDOUT:   %tuple.elem1.b82.tuple.elem = getelementptr inbounds nuw { { {}, {} }, {} }, ptr %w.var, i32 0, i32 1, !dbg !248
// CHECK:STDOUT:   call void @_CMake.Main(ptr %tuple.elem1.b82.tuple.elem), !dbg !251
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f5aa614245b4c289(ptr %w.var), !dbg !246
// CHECK:STDOUT:   ret void, !dbg !253
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.f5d9b693014ce43e() #0 !dbg !254 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %w.var = alloca { { {}, {} }, {} }, align 1, !dbg !255
// CHECK:STDOUT:   %.7a56fd.1.temp = alloca {}, align 1, !dbg !255
// CHECK:STDOUT:   %.7a56fd.2.temp = alloca {}, align 1, !dbg !255
// CHECK:STDOUT:   %.7a56fd.3.temp = alloca {}, align 1, !dbg !255
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %w.var), !dbg !255
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.7a56fd.1.temp), !dbg !255
// CHECK:STDOUT:   call void @_CMake.Main(ptr %.7a56fd.1.temp), !dbg !256
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.7a56fd.2.temp), !dbg !255
// CHECK:STDOUT:   call void @_CMake.Main(ptr %.7a56fd.2.temp), !dbg !257
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.7a56fd.3.temp), !dbg !255
// CHECK:STDOUT:   call void @_CMake.Main(ptr %.7a56fd.3.temp), !dbg !259
// CHECK:STDOUT:   %tuple.elem0.c3c.tuple.elem = getelementptr inbounds nuw { { {}, {} }, {} }, ptr %w.var, i32 0, i32 0, !dbg !260
// CHECK:STDOUT:   %.ce4.a = getelementptr inbounds nuw { {}, {} }, ptr %tuple.elem0.c3c.tuple.elem, i32 0, i32 0, !dbg !258
// CHECK:STDOUT:   call void @"_CConvert.X.Main:ImplicitAs.be297cc7300419fe.Core"(ptr %.ce4.a, ptr %.7a56fd.1.temp), !dbg !258
// CHECK:STDOUT:   %.0ae.b = getelementptr inbounds nuw { {}, {} }, ptr %tuple.elem0.c3c.tuple.elem, i32 0, i32 1, !dbg !258
// CHECK:STDOUT:   call void @"_CConvert.X.Main:ImplicitAs.be297cc7300419fe.Core"(ptr %.0ae.b, ptr %.7a56fd.2.temp), !dbg !258
// CHECK:STDOUT:   %tuple.elem1.1e9.tuple.elem = getelementptr inbounds nuw { { {}, {} }, {} }, ptr %w.var, i32 0, i32 1, !dbg !260
// CHECK:STDOUT:   call void @"_CConvert.X.Main:ImplicitAs.be297cc7300419fe.Core"(ptr %tuple.elem1.1e9.tuple.elem, ptr %.7a56fd.3.temp), !dbg !260
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.ac60fe411ae2f070(ptr %w.var), !dbg !255
// CHECK:STDOUT:   ret void, !dbg !262
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.84270bb113cb754b() #0 !dbg !263 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %w.var = alloca { { {}, {} }, {} }, align 1, !dbg !264
// CHECK:STDOUT:   %.7a56fd.4.temp = alloca {}, align 1, !dbg !264
// CHECK:STDOUT:   %.7a56fd.5.temp = alloca {}, align 1, !dbg !264
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %w.var), !dbg !264
// CHECK:STDOUT:   %tuple.elem0.297.tuple.elem = getelementptr inbounds nuw { { {}, {} }, {} }, ptr %w.var, i32 0, i32 0, !dbg !266
// CHECK:STDOUT:   %.014.a = getelementptr inbounds nuw { {}, {} }, ptr %tuple.elem0.297.tuple.elem, i32 0, i32 0, !dbg !265
// CHECK:STDOUT:   call void @_CMake.Main(ptr %.014.a), !dbg !267
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.7a56fd.4.temp), !dbg !264
// CHECK:STDOUT:   call void @_CMake.Main(ptr %.7a56fd.4.temp), !dbg !268
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.7a56fd.5.temp), !dbg !264
// CHECK:STDOUT:   call void @_CMake.Main(ptr %.7a56fd.5.temp), !dbg !269
// CHECK:STDOUT:   %.94f.b = getelementptr inbounds nuw { {}, {} }, ptr %tuple.elem0.297.tuple.elem, i32 0, i32 1, !dbg !265
// CHECK:STDOUT:   call void @"_CConvert.X.Main:ImplicitAs.be297cc7300419fe.Core"(ptr %.94f.b, ptr %.7a56fd.4.temp), !dbg !265
// CHECK:STDOUT:   %tuple.elem1.76f.tuple.elem = getelementptr inbounds nuw { { {}, {} }, {} }, ptr %w.var, i32 0, i32 1, !dbg !266
// CHECK:STDOUT:   call void @"_CConvert.X.Main:ImplicitAs.be297cc7300419fe.Core"(ptr %tuple.elem1.76f.tuple.elem, ptr %.7a56fd.5.temp), !dbg !266
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f34cebc0ce9f922d(ptr %w.var), !dbg !264
// CHECK:STDOUT:   ret void, !dbg !271
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.30ecf861811bc6e0(ptr %t) #0 !dbg !272 {
// CHECK:STDOUT:   ret void, !dbg !277
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.ea0a244f18930ba0(ptr %t) #0 !dbg !279 {
// CHECK:STDOUT:   call void @"_COp.f38e1c3ccdfe019b:core.SubobjectDestroy.Core"(ptr %t), !dbg !283
// CHECK:STDOUT:   ret void, !dbg !284
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.3773d92a34eaed6b(ptr %t) #0 !dbg !286 {
// CHECK:STDOUT:   ret void, !dbg !291
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.db6ed9edf4091b22(ptr %t) #0 !dbg !293 {
// CHECK:STDOUT:   call void @"_COp.0950f8aecfaebf0b:core.SubobjectDestroy.Core"(ptr %t), !dbg !297
// CHECK:STDOUT:   ret void, !dbg !298
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.59962c5f89868b89(ptr %t) #0 !dbg !300 {
// CHECK:STDOUT:   call void @"_COp.680f7fa3ef86df7a:core.SubobjectDestroy.Core"(ptr %t), !dbg !304
// CHECK:STDOUT:   ret void, !dbg !305
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.f5aa614245b4c289(ptr %t) #0 !dbg !307 {
// CHECK:STDOUT:   call void @"_COp.4c2c335412ec52fa:core.SubobjectDestroy.Core"(ptr %t), !dbg !311
// CHECK:STDOUT:   ret void, !dbg !312
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.ac60fe411ae2f070(ptr %t) #0 !dbg !314 {
// CHECK:STDOUT:   call void @"_COp.2f43e328c637647f:core.SubobjectDestroy.Core"(ptr %t), !dbg !318
// CHECK:STDOUT:   ret void, !dbg !319
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.f34cebc0ce9f922d(ptr %t) #0 !dbg !321 {
// CHECK:STDOUT:   call void @"_COp.bb4067704e9a198c:core.SubobjectDestroy.Core"(ptr %t), !dbg !325
// CHECK:STDOUT:   ret void, !dbg !326
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.30ecf861811bc6e0, { 4, 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.3773d92a34eaed6b, { 5, 4, 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 7, 6, 5, 4, 3, 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!211}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !26 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !39 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !92 = !{null}
// CHECK:STDOUT: !93 = !DISubroutineType(types: !92)
// CHECK:STDOUT: !136 = !{null, !39}
// CHECK:STDOUT: !137 = !DISubroutineType(types: !136)
// CHECK:STDOUT: !210 = !DIFile(filename: "initialize_from_nested_literal.carbon", directory: "")
// CHECK:STDOUT: !211 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !210, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !212 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !210, line: 24, type: !93, spFlags: DISPFlagDefinition, unit: !211)
// CHECK:STDOUT: !213 = !DILocation(line: 25, column: 3, scope: !212)
// CHECK:STDOUT: !214 = !DILocation(line: 27, column: 3, scope: !212)
// CHECK:STDOUT: !215 = !DILocation(line: 29, column: 3, scope: !212)
// CHECK:STDOUT: !216 = !DILocation(line: 24, column: 1, scope: !212)
// CHECK:STDOUT: !217 = distinct !DISubprogram(name: "Op", linkageName: "_COp.f38e1c3ccdfe019b:core.SubobjectDestroy.Core", scope: null, file: !210, line: 9, type: !137, spFlags: DISPFlagDefinition, unit: !211, retainedNodes: !220)
// CHECK:STDOUT: !218 = !DILocalVariable(arg: 1, scope: !217, type: !39)
// CHECK:STDOUT: !219 = !DILocation(line: 9, column: 3, scope: !217)
// CHECK:STDOUT: !220 = !{!218}
// CHECK:STDOUT: !221 = distinct !DISubprogram(name: "Op", linkageName: "_COp.4c2c335412ec52fa:core.SubobjectDestroy.Core", scope: null, file: !210, line: 9, type: !137, spFlags: DISPFlagDefinition, unit: !211, retainedNodes: !224)
// CHECK:STDOUT: !222 = !DILocalVariable(arg: 1, scope: !221, type: !39)
// CHECK:STDOUT: !223 = !DILocation(line: 9, column: 3, scope: !221)
// CHECK:STDOUT: !224 = !{!222}
// CHECK:STDOUT: !225 = distinct !DISubprogram(name: "Op", linkageName: "_COp.0950f8aecfaebf0b:core.SubobjectDestroy.Core", scope: null, file: !210, line: 9, type: !137, spFlags: DISPFlagDefinition, unit: !211, retainedNodes: !228)
// CHECK:STDOUT: !226 = !DILocalVariable(arg: 1, scope: !225, type: !39)
// CHECK:STDOUT: !227 = !DILocation(line: 9, column: 3, scope: !225)
// CHECK:STDOUT: !228 = !{!226}
// CHECK:STDOUT: !229 = distinct !DISubprogram(name: "Op", linkageName: "_COp.2f43e328c637647f:core.SubobjectDestroy.Core", scope: null, file: !210, line: 9, type: !137, spFlags: DISPFlagDefinition, unit: !211, retainedNodes: !232)
// CHECK:STDOUT: !230 = !DILocalVariable(arg: 1, scope: !229, type: !39)
// CHECK:STDOUT: !231 = !DILocation(line: 9, column: 3, scope: !229)
// CHECK:STDOUT: !232 = !{!230}
// CHECK:STDOUT: !233 = distinct !DISubprogram(name: "Op", linkageName: "_COp.2970be6ba4c2615c:core.SubobjectDestroy.Core", scope: null, file: !210, line: 9, type: !137, spFlags: DISPFlagDefinition, unit: !211, retainedNodes: !236)
// CHECK:STDOUT: !234 = !DILocalVariable(arg: 1, scope: !233, type: !39)
// CHECK:STDOUT: !235 = !DILocation(line: 9, column: 3, scope: !233)
// CHECK:STDOUT: !236 = !{!234}
// CHECK:STDOUT: !237 = distinct !DISubprogram(name: "Op", linkageName: "_COp.680f7fa3ef86df7a:core.SubobjectDestroy.Core", scope: null, file: !210, line: 9, type: !137, spFlags: DISPFlagDefinition, unit: !211, retainedNodes: !240)
// CHECK:STDOUT: !238 = !DILocalVariable(arg: 1, scope: !237, type: !39)
// CHECK:STDOUT: !239 = !DILocation(line: 9, column: 3, scope: !237)
// CHECK:STDOUT: !240 = !{!238}
// CHECK:STDOUT: !241 = distinct !DISubprogram(name: "Op", linkageName: "_COp.bb4067704e9a198c:core.SubobjectDestroy.Core", scope: null, file: !210, line: 9, type: !137, spFlags: DISPFlagDefinition, unit: !211, retainedNodes: !244)
// CHECK:STDOUT: !242 = !DILocalVariable(arg: 1, scope: !241, type: !39)
// CHECK:STDOUT: !243 = !DILocation(line: 9, column: 3, scope: !241)
// CHECK:STDOUT: !244 = !{!242}
// CHECK:STDOUT: !245 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.75a9ebb2041c173b", scope: null, file: !210, line: 8, type: !93, spFlags: DISPFlagDefinition, unit: !211)
// CHECK:STDOUT: !246 = !DILocation(line: 9, column: 3, scope: !245)
// CHECK:STDOUT: !247 = !DILocation(line: 9, column: 22, scope: !245)
// CHECK:STDOUT: !248 = !DILocation(line: 9, column: 21, scope: !245)
// CHECK:STDOUT: !249 = !DILocation(line: 9, column: 28, scope: !245)
// CHECK:STDOUT: !250 = !DILocation(line: 9, column: 41, scope: !245)
// CHECK:STDOUT: !251 = !DILocation(line: 9, column: 50, scope: !245)
// CHECK:STDOUT: !253 = !DILocation(line: 8, column: 1, scope: !245)
// CHECK:STDOUT: !254 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.f5d9b693014ce43e", scope: null, file: !210, line: 8, type: !93, spFlags: DISPFlagDefinition, unit: !211)
// CHECK:STDOUT: !255 = !DILocation(line: 9, column: 3, scope: !254)
// CHECK:STDOUT: !256 = !DILocation(line: 9, column: 28, scope: !254)
// CHECK:STDOUT: !257 = !DILocation(line: 9, column: 41, scope: !254)
// CHECK:STDOUT: !258 = !DILocation(line: 9, column: 22, scope: !254)
// CHECK:STDOUT: !259 = !DILocation(line: 9, column: 50, scope: !254)
// CHECK:STDOUT: !260 = !DILocation(line: 9, column: 21, scope: !254)
// CHECK:STDOUT: !262 = !DILocation(line: 8, column: 1, scope: !254)
// CHECK:STDOUT: !263 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.84270bb113cb754b", scope: null, file: !210, line: 8, type: !93, spFlags: DISPFlagDefinition, unit: !211)
// CHECK:STDOUT: !264 = !DILocation(line: 9, column: 3, scope: !263)
// CHECK:STDOUT: !265 = !DILocation(line: 9, column: 22, scope: !263)
// CHECK:STDOUT: !266 = !DILocation(line: 9, column: 21, scope: !263)
// CHECK:STDOUT: !267 = !DILocation(line: 9, column: 28, scope: !263)
// CHECK:STDOUT: !268 = !DILocation(line: 9, column: 41, scope: !263)
// CHECK:STDOUT: !269 = !DILocation(line: 9, column: 50, scope: !263)
// CHECK:STDOUT: !271 = !DILocation(line: 8, column: 1, scope: !263)
// CHECK:STDOUT: !272 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.30ecf861811bc6e0", scope: null, file: !26, line: 30, type: !137, spFlags: DISPFlagDefinition, unit: !211, retainedNodes: !278)
// CHECK:STDOUT: !273 = !DILocalVariable(arg: 1, scope: !272, type: !39)
// CHECK:STDOUT: !277 = !DILocation(line: 30, column: 1, scope: !272)
// CHECK:STDOUT: !278 = !{!273}
// CHECK:STDOUT: !279 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.ea0a244f18930ba0", scope: null, file: !26, line: 30, type: !137, spFlags: DISPFlagDefinition, unit: !211, retainedNodes: !285)
// CHECK:STDOUT: !280 = !DILocalVariable(arg: 1, scope: !279, type: !39)
// CHECK:STDOUT: !283 = !DILocation(line: 32, column: 5, scope: !279)
// CHECK:STDOUT: !284 = !DILocation(line: 30, column: 1, scope: !279)
// CHECK:STDOUT: !285 = !{!280}
// CHECK:STDOUT: !286 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.3773d92a34eaed6b", scope: null, file: !26, line: 30, type: !137, spFlags: DISPFlagDefinition, unit: !211, retainedNodes: !292)
// CHECK:STDOUT: !287 = !DILocalVariable(arg: 1, scope: !286, type: !39)
// CHECK:STDOUT: !291 = !DILocation(line: 30, column: 1, scope: !286)
// CHECK:STDOUT: !292 = !{!287}
// CHECK:STDOUT: !293 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.db6ed9edf4091b22", scope: null, file: !26, line: 30, type: !137, spFlags: DISPFlagDefinition, unit: !211, retainedNodes: !299)
// CHECK:STDOUT: !294 = !DILocalVariable(arg: 1, scope: !293, type: !39)
// CHECK:STDOUT: !297 = !DILocation(line: 32, column: 5, scope: !293)
// CHECK:STDOUT: !298 = !DILocation(line: 30, column: 1, scope: !293)
// CHECK:STDOUT: !299 = !{!294}
// CHECK:STDOUT: !300 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.59962c5f89868b89", scope: null, file: !26, line: 30, type: !137, spFlags: DISPFlagDefinition, unit: !211, retainedNodes: !306)
// CHECK:STDOUT: !301 = !DILocalVariable(arg: 1, scope: !300, type: !39)
// CHECK:STDOUT: !304 = !DILocation(line: 32, column: 5, scope: !300)
// CHECK:STDOUT: !305 = !DILocation(line: 30, column: 1, scope: !300)
// CHECK:STDOUT: !306 = !{!301}
// CHECK:STDOUT: !307 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.f5aa614245b4c289", scope: null, file: !26, line: 30, type: !137, spFlags: DISPFlagDefinition, unit: !211, retainedNodes: !313)
// CHECK:STDOUT: !308 = !DILocalVariable(arg: 1, scope: !307, type: !39)
// CHECK:STDOUT: !311 = !DILocation(line: 32, column: 5, scope: !307)
// CHECK:STDOUT: !312 = !DILocation(line: 30, column: 1, scope: !307)
// CHECK:STDOUT: !313 = !{!308}
// CHECK:STDOUT: !314 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.ac60fe411ae2f070", scope: null, file: !26, line: 30, type: !137, spFlags: DISPFlagDefinition, unit: !211, retainedNodes: !320)
// CHECK:STDOUT: !315 = !DILocalVariable(arg: 1, scope: !314, type: !39)
// CHECK:STDOUT: !318 = !DILocation(line: 32, column: 5, scope: !314)
// CHECK:STDOUT: !319 = !DILocation(line: 30, column: 1, scope: !314)
// CHECK:STDOUT: !320 = !{!315}
// CHECK:STDOUT: !321 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.f34cebc0ce9f922d", scope: null, file: !26, line: 30, type: !137, spFlags: DISPFlagDefinition, unit: !211, retainedNodes: !327)
// CHECK:STDOUT: !322 = !DILocalVariable(arg: 1, scope: !321, type: !39)
// CHECK:STDOUT: !325 = !DILocation(line: 32, column: 5, scope: !321)
// CHECK:STDOUT: !326 = !DILocation(line: 30, column: 1, scope: !321)
// CHECK:STDOUT: !327 = !{!322}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'as.carbon'
// CHECK:STDOUT: source_filename = "as.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTest1.Main(i32 %n) #0 !dbg !330 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %AsI32.call = call i32 @_CAsI32.Main.55456748a8a2eb70(i32 %n), !dbg !334
// CHECK:STDOUT:   ret void, !dbg !337
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @"_CConvert.C.Main:As.55456748a8a2eb70.Core"(ptr %self) #0 !dbg !339 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc15_42.1.n = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !344
// CHECK:STDOUT:   %.loc15_42.2 = load i32, ptr %.loc15_42.1.n, align 4, !dbg !344
// CHECK:STDOUT:   ret i32 %.loc15_42.2, !dbg !345
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTest2.Main(ptr %c) #0 !dbg !347 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %AsI32.call = call i32 @_CAsI32.Main.9522ff908e97e267(ptr %c), !dbg !351
// CHECK:STDOUT:   ret void, !dbg !354
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CAsI32.Main.55456748a8a2eb70(i32 %x) #0 !dbg !356 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %x, !dbg !362
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CAsI32.Main.9522ff908e97e267(ptr %x) #0 !dbg !364 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %C.as.As.impl.Convert.call = call i32 @"_CConvert.C.Main:As.55456748a8a2eb70.Core"(ptr %x), !dbg !369
// CHECK:STDOUT:   ret i32 %C.as.As.impl.Convert.call, !dbg !370
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!329}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !36 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !37 = !{!36, !36}
// CHECK:STDOUT: !38 = !DISubroutineType(types: !37)
// CHECK:STDOUT: !39 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !40 = !{!36, !39}
// CHECK:STDOUT: !41 = !DISubroutineType(types: !40)
// CHECK:STDOUT: !100 = !{null, !36}
// CHECK:STDOUT: !101 = !DISubroutineType(types: !100)
// CHECK:STDOUT: !136 = !{null, !39}
// CHECK:STDOUT: !137 = !DISubroutineType(types: !136)
// CHECK:STDOUT: !328 = !DIFile(filename: "as.carbon", directory: "")
// CHECK:STDOUT: !329 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !328, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !330 = distinct !DISubprogram(name: "Test1", linkageName: "_CTest1.Main", scope: null, file: !328, line: 8, type: !101, spFlags: DISPFlagDefinition, unit: !329, retainedNodes: !338)
// CHECK:STDOUT: !331 = !DILocalVariable(arg: 1, scope: !330, type: !36)
// CHECK:STDOUT: !334 = !DILocation(line: 9, column: 16, scope: !330)
// CHECK:STDOUT: !337 = !DILocation(line: 8, column: 1, scope: !330)
// CHECK:STDOUT: !338 = !{!331}
// CHECK:STDOUT: !339 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.C.Main:As.55456748a8a2eb70.Core", scope: null, file: !328, line: 15, type: !41, spFlags: DISPFlagDefinition, unit: !329, retainedNodes: !346)
// CHECK:STDOUT: !340 = !DILocalVariable(arg: 1, scope: !339, type: !39)
// CHECK:STDOUT: !344 = !DILocation(line: 15, column: 38, scope: !339)
// CHECK:STDOUT: !345 = !DILocation(line: 15, column: 31, scope: !339)
// CHECK:STDOUT: !346 = !{!340}
// CHECK:STDOUT: !347 = distinct !DISubprogram(name: "Test2", linkageName: "_CTest2.Main", scope: null, file: !328, line: 19, type: !137, spFlags: DISPFlagDefinition, unit: !329, retainedNodes: !355)
// CHECK:STDOUT: !348 = !DILocalVariable(arg: 1, scope: !347, type: !39)
// CHECK:STDOUT: !351 = !DILocation(line: 20, column: 16, scope: !347)
// CHECK:STDOUT: !354 = !DILocation(line: 19, column: 1, scope: !347)
// CHECK:STDOUT: !355 = !{!348}
// CHECK:STDOUT: !356 = distinct !DISubprogram(name: "AsI32", linkageName: "_CAsI32.Main.55456748a8a2eb70", scope: null, file: !328, line: 4, type: !38, spFlags: DISPFlagDefinition, unit: !329, retainedNodes: !363)
// CHECK:STDOUT: !357 = !DILocalVariable(arg: 1, scope: !356, type: !36)
// CHECK:STDOUT: !362 = !DILocation(line: 5, column: 3, scope: !356)
// CHECK:STDOUT: !363 = !{!357}
// CHECK:STDOUT: !364 = distinct !DISubprogram(name: "AsI32", linkageName: "_CAsI32.Main.9522ff908e97e267", scope: null, file: !328, line: 4, type: !41, spFlags: DISPFlagDefinition, unit: !329, retainedNodes: !371)
// CHECK:STDOUT: !365 = !DILocalVariable(arg: 1, scope: !364, type: !39)
// CHECK:STDOUT: !369 = !DILocation(line: 5, column: 10, scope: !364)
// CHECK:STDOUT: !370 = !DILocation(line: 5, column: 3, scope: !364)
// CHECK:STDOUT: !371 = !{!365}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'as_type.carbon'
// CHECK:STDOUT: source_filename = "as_type.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTest1.Main(i32 %n) #0 !dbg !374 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CAsType.Main.0f3e15a21cc8e963(i32 %n), !dbg !378
// CHECK:STDOUT:   ret void, !dbg !379
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @"_CConvert.C.Main:As.55456748a8a2eb70.Core"(ptr %self) #0 !dbg !381 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc15_42.1.n = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !386
// CHECK:STDOUT:   %.loc15_42.2 = load i32, ptr %.loc15_42.1.n, align 4, !dbg !386
// CHECK:STDOUT:   ret i32 %.loc15_42.2, !dbg !387
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTest2.Main(ptr %c) #0 !dbg !389 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CAsType.Main.9f9f842f4c8e7f62(ptr %c), !dbg !393
// CHECK:STDOUT:   ret void, !dbg !394
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CAsType.Main.0f3e15a21cc8e963(i32 %x) #0 !dbg !396 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !402
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CAsType.Main.9f9f842f4c8e7f62(ptr %x) #0 !dbg !404 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %C.as.As.impl.Convert.call = call i32 @"_CConvert.C.Main:As.55456748a8a2eb70.Core"(ptr %x), !dbg !407
// CHECK:STDOUT:   ret void, !dbg !410
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!373}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !36 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !39 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !40 = !{!36, !39}
// CHECK:STDOUT: !41 = !DISubroutineType(types: !40)
// CHECK:STDOUT: !100 = !{null, !36}
// CHECK:STDOUT: !101 = !DISubroutineType(types: !100)
// CHECK:STDOUT: !136 = !{null, !39}
// CHECK:STDOUT: !137 = !DISubroutineType(types: !136)
// CHECK:STDOUT: !372 = !DIFile(filename: "as_type.carbon", directory: "")
// CHECK:STDOUT: !373 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !372, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !374 = distinct !DISubprogram(name: "Test1", linkageName: "_CTest1.Main", scope: null, file: !372, line: 8, type: !101, spFlags: DISPFlagDefinition, unit: !373, retainedNodes: !380)
// CHECK:STDOUT: !375 = !DILocalVariable(arg: 1, scope: !374, type: !36)
// CHECK:STDOUT: !378 = !DILocation(line: 9, column: 3, scope: !374)
// CHECK:STDOUT: !379 = !DILocation(line: 8, column: 1, scope: !374)
// CHECK:STDOUT: !380 = !{!375}
// CHECK:STDOUT: !381 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.C.Main:As.55456748a8a2eb70.Core", scope: null, file: !372, line: 15, type: !41, spFlags: DISPFlagDefinition, unit: !373, retainedNodes: !388)
// CHECK:STDOUT: !382 = !DILocalVariable(arg: 1, scope: !381, type: !39)
// CHECK:STDOUT: !386 = !DILocation(line: 15, column: 38, scope: !381)
// CHECK:STDOUT: !387 = !DILocation(line: 15, column: 31, scope: !381)
// CHECK:STDOUT: !388 = !{!382}
// CHECK:STDOUT: !389 = distinct !DISubprogram(name: "Test2", linkageName: "_CTest2.Main", scope: null, file: !372, line: 19, type: !137, spFlags: DISPFlagDefinition, unit: !373, retainedNodes: !395)
// CHECK:STDOUT: !390 = !DILocalVariable(arg: 1, scope: !389, type: !39)
// CHECK:STDOUT: !393 = !DILocation(line: 20, column: 3, scope: !389)
// CHECK:STDOUT: !394 = !DILocation(line: 19, column: 1, scope: !389)
// CHECK:STDOUT: !395 = !{!390}
// CHECK:STDOUT: !396 = distinct !DISubprogram(name: "AsType", linkageName: "_CAsType.Main.0f3e15a21cc8e963", scope: null, file: !372, line: 4, type: !101, spFlags: DISPFlagDefinition, unit: !373, retainedNodes: !403)
// CHECK:STDOUT: !397 = !DILocalVariable(arg: 1, scope: !396, type: !36)
// CHECK:STDOUT: !402 = !DILocation(line: 4, column: 1, scope: !396)
// CHECK:STDOUT: !403 = !{!397}
// CHECK:STDOUT: !404 = distinct !DISubprogram(name: "AsType", linkageName: "_CAsType.Main.9f9f842f4c8e7f62", scope: null, file: !372, line: 4, type: !137, spFlags: DISPFlagDefinition, unit: !373, retainedNodes: !411)
// CHECK:STDOUT: !405 = !DILocalVariable(arg: 1, scope: !404, type: !39)
// CHECK:STDOUT: !407 = !DILocation(line: 5, column: 14, scope: !404)
// CHECK:STDOUT: !410 = !DILocation(line: 4, column: 1, scope: !404)
// CHECK:STDOUT: !411 = !{!405}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'initialize_from_template_method_call.carbon'
// CHECK:STDOUT: source_filename = "initialize_from_template_method_call.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @B.val.loc6_14 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMakeB.B.Main(ptr sret({}) %return) #0 !dbg !416 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %return, ptr align 1 @B.val.loc6_14, i64 0, i1 false), !dbg !419
// CHECK:STDOUT:   ret void, !dbg !419
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @main() #0 !dbg !420 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc21_10.1.temp = alloca { {} }, align 1, !dbg !421
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc21_10.1.temp), !dbg !421
// CHECK:STDOUT:   call void @_CMakeA.Main.a6a60dfee2d7d273(ptr %.loc21_10.1.temp), !dbg !421
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.e517637b3b58cb2c(ptr %.loc21_10.1.temp), !dbg !421
// CHECK:STDOUT:   ret i32 0, !dbg !422
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.8fbe0c6a865381a4:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !423 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc21_10.1.field = getelementptr inbounds nuw { {} }, ptr %self, i32 0, i32 0, !dbg !425
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.b963e75af1c16acd(ptr %.loc21_10.1.field), !dbg !425
// CHECK:STDOUT:   ret void, !dbg !425
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.5a23cc09585c55d2:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !427 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc21_10.1.field = getelementptr inbounds nuw { {} }, ptr %self, i32 0, i32 0, !dbg !429
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.b963e75af1c16acd(ptr %.loc21_10.1.field), !dbg !429
// CHECK:STDOUT:   ret void, !dbg !429
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CMakeA.Main.a6a60dfee2d7d273(ptr sret({ {} }) %return) #0 !dbg !431 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.f4aa6f.1.field = getelementptr inbounds nuw { {} }, ptr %return, i32 0, i32 0, !dbg !435
// CHECK:STDOUT:   call void @_CMakeB.B.Main(ptr %.f4aa6f.1.field), !dbg !434
// CHECK:STDOUT:   ret void, !dbg !436
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.e517637b3b58cb2c(ptr %t) #0 !dbg !437 {
// CHECK:STDOUT:   call void @"_COp.5a23cc09585c55d2:core.SubobjectDestroy.Core"(ptr %t), !dbg !441
// CHECK:STDOUT:   ret void, !dbg !442
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.b963e75af1c16acd(ptr %t) #0 !dbg !444 {
// CHECK:STDOUT:   ret void, !dbg !449
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.b963e75af1c16acd, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!413}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !26 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !36 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !39 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !132 = !{!39}
// CHECK:STDOUT: !133 = !DISubroutineType(types: !132)
// CHECK:STDOUT: !136 = !{null, !39}
// CHECK:STDOUT: !137 = !DISubroutineType(types: !136)
// CHECK:STDOUT: !412 = !DIFile(filename: "initialize_from_template_method_call.carbon", directory: "")
// CHECK:STDOUT: !413 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !412, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !414 = !{!36}
// CHECK:STDOUT: !415 = !DISubroutineType(types: !414)
// CHECK:STDOUT: !416 = distinct !DISubprogram(name: "MakeB", linkageName: "_CMakeB.B.Main", scope: null, file: !412, line: 5, type: !133, spFlags: DISPFlagDefinition, unit: !413)
// CHECK:STDOUT: !419 = !DILocation(line: 6, column: 5, scope: !416)
// CHECK:STDOUT: !420 = distinct !DISubprogram(name: "Run", linkageName: "main", scope: null, file: !412, line: 20, type: !415, spFlags: DISPFlagDefinition, unit: !413)
// CHECK:STDOUT: !421 = !DILocation(line: 21, column: 3, scope: !420)
// CHECK:STDOUT: !422 = !DILocation(line: 20, column: 1, scope: !420)
// CHECK:STDOUT: !423 = distinct !DISubprogram(name: "Op", linkageName: "_COp.8fbe0c6a865381a4:core.SubobjectDestroy.Core", scope: null, file: !412, line: 21, type: !137, spFlags: DISPFlagDefinition, unit: !413, retainedNodes: !426)
// CHECK:STDOUT: !424 = !DILocalVariable(arg: 1, scope: !423, type: !39)
// CHECK:STDOUT: !425 = !DILocation(line: 21, column: 3, scope: !423)
// CHECK:STDOUT: !426 = !{!424}
// CHECK:STDOUT: !427 = distinct !DISubprogram(name: "Op", linkageName: "_COp.5a23cc09585c55d2:core.SubobjectDestroy.Core", scope: null, file: !412, line: 21, type: !137, spFlags: DISPFlagDefinition, unit: !413, retainedNodes: !430)
// CHECK:STDOUT: !428 = !DILocalVariable(arg: 1, scope: !427, type: !39)
// CHECK:STDOUT: !429 = !DILocation(line: 21, column: 3, scope: !427)
// CHECK:STDOUT: !430 = !{!428}
// CHECK:STDOUT: !431 = distinct !DISubprogram(name: "MakeA", linkageName: "_CMakeA.Main.a6a60dfee2d7d273", scope: null, file: !412, line: 14, type: !133, spFlags: DISPFlagDefinition, unit: !413)
// CHECK:STDOUT: !434 = !DILocation(line: 16, column: 15, scope: !431)
// CHECK:STDOUT: !435 = !DILocation(line: 15, column: 11, scope: !431)
// CHECK:STDOUT: !436 = !DILocation(line: 15, column: 4, scope: !431)
// CHECK:STDOUT: !437 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.e517637b3b58cb2c", scope: null, file: !26, line: 30, type: !137, spFlags: DISPFlagDefinition, unit: !413, retainedNodes: !443)
// CHECK:STDOUT: !438 = !DILocalVariable(arg: 1, scope: !437, type: !39)
// CHECK:STDOUT: !441 = !DILocation(line: 32, column: 5, scope: !437)
// CHECK:STDOUT: !442 = !DILocation(line: 30, column: 1, scope: !437)
// CHECK:STDOUT: !443 = !{!438}
// CHECK:STDOUT: !444 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.b963e75af1c16acd", scope: null, file: !26, line: 30, type: !137, spFlags: DISPFlagDefinition, unit: !413, retainedNodes: !450)
// CHECK:STDOUT: !445 = !DILocalVariable(arg: 1, scope: !444, type: !39)
// CHECK:STDOUT: !449 = !DILocation(line: 30, column: 1, scope: !444)
// CHECK:STDOUT: !450 = !{!445}
// CHECK:STDOUT:
