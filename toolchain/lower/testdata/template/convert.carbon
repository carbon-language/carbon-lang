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
// CHECK:STDOUT:   %F.call = call i32 @_CF.Main.84588f41d61dafba(i32 %n), !dbg !48
// CHECK:STDOUT:   ret i32 %F.call, !dbg !49
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @"_CConvert.C.Main:ImplicitAs.84588f41d61dafba.Core"(ptr %self) #0 !dbg !51 {
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
// CHECK:STDOUT: define linkonce_odr i32 @_CF.Main.84588f41d61dafba(i32 %x) #0 !dbg !68 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %x, !dbg !77
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CF.Main.9522ff908e97e267(ptr %x) #0 !dbg !79 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.call = call i32 @"_CConvert.C.Main:ImplicitAs.84588f41d61dafba.Core"(ptr %x), !dbg !84
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
// CHECK:STDOUT: !51 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.C.Main:ImplicitAs.84588f41d61dafba.Core", scope: null, file: !34, line: 16, type: !41, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !58)
// CHECK:STDOUT: !52 = !DILocalVariable(arg: 1, scope: !51, type: !39)
// CHECK:STDOUT: !56 = !DILocation(line: 16, column: 38, scope: !51)
// CHECK:STDOUT: !57 = !DILocation(line: 16, column: 31, scope: !51)
// CHECK:STDOUT: !58 = !{!52}
// CHECK:STDOUT: !59 = distinct !DISubprogram(name: "Test2", linkageName: "_CTest2.Main", scope: null, file: !34, line: 20, type: !41, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !67)
// CHECK:STDOUT: !60 = !DILocalVariable(arg: 1, scope: !59, type: !39)
// CHECK:STDOUT: !65 = !DILocation(line: 21, column: 10, scope: !59)
// CHECK:STDOUT: !66 = !DILocation(line: 21, column: 3, scope: !59)
// CHECK:STDOUT: !67 = !{!60}
// CHECK:STDOUT: !68 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.84588f41d61dafba", scope: null, file: !34, line: 4, type: !38, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !78)
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
// CHECK:STDOUT: define void @_CG.Main() #0 !dbg !96 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Main.84588f41d61dafba(), !dbg !97
// CHECK:STDOUT:   ret void, !dbg !98
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !99 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !101
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.84588f41d61dafba() #0 !dbg !103 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %w.var = alloca i32, align 4, !dbg !104
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %w.var), !dbg !104
// CHECK:STDOUT:   store i32 0, ptr %w.var, align 4, !dbg !104
// CHECK:STDOUT:   call void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %w.var), !dbg !104
// CHECK:STDOUT:   ret void, !dbg !106
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!91}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !36 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !90 = !DIFile(filename: "initialize_dependent_target_from_literal.carbon", directory: "")
// CHECK:STDOUT: !91 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !90, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !92 = !{null}
// CHECK:STDOUT: !93 = !DISubroutineType(types: !92)
// CHECK:STDOUT: !94 = !{null, !36}
// CHECK:STDOUT: !95 = !DISubroutineType(types: !94)
// CHECK:STDOUT: !96 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !90, line: 8, type: !93, spFlags: DISPFlagDefinition, unit: !91)
// CHECK:STDOUT: !97 = !DILocation(line: 9, column: 3, scope: !96)
// CHECK:STDOUT: !98 = !DILocation(line: 8, column: 1, scope: !96)
// CHECK:STDOUT: !99 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.f58524831c37807e:core.Destroy.Core", scope: null, file: !90, line: 5, type: !95, spFlags: DISPFlagDefinition, unit: !91, retainedNodes: !102)
// CHECK:STDOUT: !100 = !DILocalVariable(arg: 1, scope: !99, type: !36)
// CHECK:STDOUT: !101 = !DILocation(line: 5, column: 3, scope: !99)
// CHECK:STDOUT: !102 = !{!100}
// CHECK:STDOUT: !103 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.84588f41d61dafba", scope: null, file: !90, line: 4, type: !93, spFlags: DISPFlagDefinition, unit: !91)
// CHECK:STDOUT: !104 = !DILocation(line: 5, column: 3, scope: !103)
// CHECK:STDOUT: !106 = !DILocation(line: 4, column: 1, scope: !103)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'initialize_primitive.carbon'
// CHECK:STDOUT: source_filename = "initialize_primitive.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !109 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !111
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.Main() #0 !dbg !113 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Main.84588f41d61dafba(i32 0), !dbg !114
// CHECK:STDOUT:   ret void, !dbg !115
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.84588f41d61dafba(i32 %x) #0 !dbg !116 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %w.var = alloca i32, align 4, !dbg !119
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %w.var), !dbg !119
// CHECK:STDOUT:   store i32 %x, ptr %w.var, align 4, !dbg !119
// CHECK:STDOUT:   call void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %w.var), !dbg !119
// CHECK:STDOUT:   ret void, !dbg !122
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!108}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !36 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !92 = !{null}
// CHECK:STDOUT: !93 = !DISubroutineType(types: !92)
// CHECK:STDOUT: !94 = !{null, !36}
// CHECK:STDOUT: !95 = !DISubroutineType(types: !94)
// CHECK:STDOUT: !107 = !DIFile(filename: "initialize_primitive.carbon", directory: "")
// CHECK:STDOUT: !108 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !107, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !109 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.f58524831c37807e:core.Destroy.Core", scope: null, file: !107, line: 5, type: !95, spFlags: DISPFlagDefinition, unit: !108, retainedNodes: !112)
// CHECK:STDOUT: !110 = !DILocalVariable(arg: 1, scope: !109, type: !36)
// CHECK:STDOUT: !111 = !DILocation(line: 5, column: 3, scope: !109)
// CHECK:STDOUT: !112 = !{!110}
// CHECK:STDOUT: !113 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !107, line: 8, type: !93, spFlags: DISPFlagDefinition, unit: !108)
// CHECK:STDOUT: !114 = !DILocation(line: 9, column: 3, scope: !113)
// CHECK:STDOUT: !115 = !DILocation(line: 8, column: 1, scope: !113)
// CHECK:STDOUT: !116 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.84588f41d61dafba", scope: null, file: !107, line: 4, type: !95, spFlags: DISPFlagDefinition, unit: !108, retainedNodes: !123)
// CHECK:STDOUT: !117 = !DILocalVariable(arg: 1, scope: !116, type: !36)
// CHECK:STDOUT: !119 = !DILocation(line: 5, column: 3, scope: !116)
// CHECK:STDOUT: !122 = !DILocation(line: 4, column: 1, scope: !116)
// CHECK:STDOUT: !123 = !{!117}
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
// CHECK:STDOUT: define void @"_CConvert.92e03c783d6af646:ImplicitAs.be297cc7300419fe.Core"(ptr sret({}) %return, ptr %self) #0 !dbg !132 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %return, ptr align 1 @Y.val.loc14_54, i64 0, i1 false), !dbg !137
// CHECK:STDOUT:   ret void, !dbg !137
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.Main() #0 !dbg !139 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Main.fb3d597a6b1c4ee1(), !dbg !140
// CHECK:STDOUT:   call void @_CF.Main.2eaa3ac913700940(), !dbg !141
// CHECK:STDOUT:   call void @_CF.Main.be297cc7300419fe(), !dbg !142
// CHECK:STDOUT:   ret void, !dbg !143
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.cb8adced19392b99:core.Destroy.Core"(ptr %self) #0 !dbg !144 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !146
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.8b0bf151036f045a:core.Destroy.Core"(ptr %self) #0 !dbg !148 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !150
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.92e03c783d6af646:core.Destroy.Core"(ptr %self) #0 !dbg !152 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %tuple.elem2.tuple.elem = getelementptr inbounds nuw { {}, {}, {} }, ptr %self, i32 0, i32 2, !dbg !154
// CHECK:STDOUT:   call void @"_CSelfDestruct.8b0bf151036f045a:core.Destroy.Core"(ptr %tuple.elem2.tuple.elem), !dbg !154
// CHECK:STDOUT:   %tuple.elem1.tuple.elem = getelementptr inbounds nuw { {}, {}, {} }, ptr %self, i32 0, i32 1, !dbg !154
// CHECK:STDOUT:   call void @"_CSelfDestruct.8b0bf151036f045a:core.Destroy.Core"(ptr %tuple.elem1.tuple.elem), !dbg !154
// CHECK:STDOUT:   %tuple.elem0.tuple.elem = getelementptr inbounds nuw { {}, {}, {} }, ptr %self, i32 0, i32 0, !dbg !154
// CHECK:STDOUT:   call void @"_CSelfDestruct.8b0bf151036f045a:core.Destroy.Core"(ptr %tuple.elem0.tuple.elem), !dbg !154
// CHECK:STDOUT:   ret void, !dbg !154
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.92e03c783d6af646:core.Destroy.Core"(ptr %self) #0 !dbg !156 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.92e03c783d6af646:core.Destroy.Core"(ptr %self), !dbg !158
// CHECK:STDOUT:   ret void, !dbg !158
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.452a47fbe5382294:core.Destroy.Core"(ptr %self) #0 !dbg !160 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc8_3.1.array.index = getelementptr inbounds [3 x {}], ptr %self, i32 0, i64 2, !dbg !162
// CHECK:STDOUT:   call void @"_CSelfDestruct.8b0bf151036f045a:core.Destroy.Core"(ptr %.loc8_3.1.array.index), !dbg !162
// CHECK:STDOUT:   %.loc8_3.2.array.index = getelementptr inbounds [3 x {}], ptr %self, i32 0, i64 1, !dbg !162
// CHECK:STDOUT:   call void @"_CSelfDestruct.8b0bf151036f045a:core.Destroy.Core"(ptr %.loc8_3.2.array.index), !dbg !162
// CHECK:STDOUT:   %.loc8_3.3.array.index = getelementptr inbounds [3 x {}], ptr %self, i32 0, i64 0, !dbg !162
// CHECK:STDOUT:   call void @"_CSelfDestruct.8b0bf151036f045a:core.Destroy.Core"(ptr %.loc8_3.3.array.index), !dbg !162
// CHECK:STDOUT:   ret void, !dbg !162
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.452a47fbe5382294:core.Destroy.Core"(ptr %self) #0 !dbg !164 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.452a47fbe5382294:core.Destroy.Core"(ptr %self), !dbg !166
// CHECK:STDOUT:   ret void, !dbg !166
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.12212b84320f2fd6:core.Destroy.Core"(ptr %self) #0 !dbg !168 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !170
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.fb3d597a6b1c4ee1() #0 !dbg !172 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %w.var = alloca { {}, {}, {} }, align 1, !dbg !173
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %w.var), !dbg !173
// CHECK:STDOUT:   %tuple.elem0.tuple.elem = getelementptr inbounds nuw { {}, {}, {} }, ptr %w.var, i32 0, i32 0, !dbg !174
// CHECK:STDOUT:   call void @_CMake.Main(ptr %tuple.elem0.tuple.elem), !dbg !175
// CHECK:STDOUT:   %tuple.elem1.tuple.elem = getelementptr inbounds nuw { {}, {}, {} }, ptr %w.var, i32 0, i32 1, !dbg !174
// CHECK:STDOUT:   call void @_CMake.Main(ptr %tuple.elem1.tuple.elem), !dbg !176
// CHECK:STDOUT:   %tuple.elem2.tuple.elem = getelementptr inbounds nuw { {}, {}, {} }, ptr %w.var, i32 0, i32 2, !dbg !174
// CHECK:STDOUT:   call void @_CMake.Main(ptr %tuple.elem2.tuple.elem), !dbg !177
// CHECK:STDOUT:   call void @"_CSelfDestruct.92e03c783d6af646:core.Destroy.Core"(ptr %w.var), !dbg !173
// CHECK:STDOUT:   ret void, !dbg !179
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.2eaa3ac913700940() #0 !dbg !180 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %w.var = alloca [3 x {}], align 1, !dbg !181
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %w.var), !dbg !181
// CHECK:STDOUT:   %.5d0.array.index = getelementptr inbounds [3 x {}], ptr %w.var, i32 0, i64 0, !dbg !182
// CHECK:STDOUT:   call void @_CMake.Main(ptr %.5d0.array.index), !dbg !183
// CHECK:STDOUT:   %.502.array.index = getelementptr inbounds [3 x {}], ptr %w.var, i32 0, i64 1, !dbg !182
// CHECK:STDOUT:   call void @_CMake.Main(ptr %.502.array.index), !dbg !184
// CHECK:STDOUT:   %.5c5.array.index = getelementptr inbounds [3 x {}], ptr %w.var, i32 0, i64 2, !dbg !182
// CHECK:STDOUT:   call void @_CMake.Main(ptr %.5c5.array.index), !dbg !185
// CHECK:STDOUT:   call void @"_CSelfDestruct.452a47fbe5382294:core.Destroy.Core"(ptr %w.var), !dbg !181
// CHECK:STDOUT:   ret void, !dbg !187
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.be297cc7300419fe() #0 !dbg !188 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %w.var = alloca {}, align 1, !dbg !189
// CHECK:STDOUT:   %.7a56fd.1.temp = alloca {}, align 1, !dbg !189
// CHECK:STDOUT:   %.7a56fd.2.temp = alloca {}, align 1, !dbg !189
// CHECK:STDOUT:   %.7a56fd.3.temp = alloca {}, align 1, !dbg !189
// CHECK:STDOUT:   %tuple.9aa = alloca { ptr, ptr, ptr }, align 8, !dbg !193
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %w.var), !dbg !189
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.7a56fd.1.temp), !dbg !189
// CHECK:STDOUT:   call void @_CMake.Main(ptr %.7a56fd.1.temp), !dbg !190
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.7a56fd.2.temp), !dbg !189
// CHECK:STDOUT:   call void @_CMake.Main(ptr %.7a56fd.2.temp), !dbg !191
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.7a56fd.3.temp), !dbg !189
// CHECK:STDOUT:   call void @_CMake.Main(ptr %.7a56fd.3.temp), !dbg !192
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %tuple.9aa), !dbg !193
// CHECK:STDOUT:   %tuple.9aa1 = getelementptr inbounds nuw { ptr, ptr, ptr }, ptr %tuple.9aa, i32 0, i32 0, !dbg !193
// CHECK:STDOUT:   store ptr %.7a56fd.1.temp, ptr %tuple.9aa1, align 8, !dbg !193
// CHECK:STDOUT:   %tuple.9aa2 = getelementptr inbounds nuw { ptr, ptr, ptr }, ptr %tuple.9aa, i32 0, i32 1, !dbg !193
// CHECK:STDOUT:   store ptr %.7a56fd.2.temp, ptr %tuple.9aa2, align 8, !dbg !193
// CHECK:STDOUT:   %tuple.9aa3 = getelementptr inbounds nuw { ptr, ptr, ptr }, ptr %tuple.9aa, i32 0, i32 2, !dbg !193
// CHECK:STDOUT:   store ptr %.7a56fd.3.temp, ptr %tuple.9aa3, align 8, !dbg !193
// CHECK:STDOUT:   call void @"_CConvert.92e03c783d6af646:ImplicitAs.be297cc7300419fe.Core"(ptr %w.var, ptr %tuple.9aa), !dbg !189
// CHECK:STDOUT:   call void @"_CSelfDestruct.12212b84320f2fd6:core.Destroy.Core"(ptr %w.var), !dbg !189
// CHECK:STDOUT:   ret void, !dbg !195
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 6, 5, 4, 3, 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!125}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !39 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !92 = !{null}
// CHECK:STDOUT: !93 = !DISubroutineType(types: !92)
// CHECK:STDOUT: !124 = !DIFile(filename: "initialize_from_tuple.carbon", directory: "")
// CHECK:STDOUT: !125 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !124, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !128 = !{!39, !39}
// CHECK:STDOUT: !129 = !DISubroutineType(types: !128)
// CHECK:STDOUT: !130 = !{null, !39}
// CHECK:STDOUT: !131 = !DISubroutineType(types: !130)
// CHECK:STDOUT: !132 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.92e03c783d6af646:ImplicitAs.be297cc7300419fe.Core", scope: null, file: !124, line: 14, type: !129, spFlags: DISPFlagDefinition, unit: !125, retainedNodes: !138)
// CHECK:STDOUT: !133 = !DILocalVariable(arg: 1, scope: !132, type: !39)
// CHECK:STDOUT: !137 = !DILocation(line: 14, column: 45, scope: !132)
// CHECK:STDOUT: !138 = !{!133}
// CHECK:STDOUT: !139 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !124, line: 17, type: !93, spFlags: DISPFlagDefinition, unit: !125)
// CHECK:STDOUT: !140 = !DILocation(line: 18, column: 3, scope: !139)
// CHECK:STDOUT: !141 = !DILocation(line: 20, column: 3, scope: !139)
// CHECK:STDOUT: !142 = !DILocation(line: 22, column: 3, scope: !139)
// CHECK:STDOUT: !143 = !DILocation(line: 17, column: 1, scope: !139)
// CHECK:STDOUT: !144 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.cb8adced19392b99:core.Destroy.Core", scope: null, file: !124, line: 8, type: !131, spFlags: DISPFlagDefinition, unit: !125, retainedNodes: !147)
// CHECK:STDOUT: !145 = !DILocalVariable(arg: 1, scope: !144, type: !39)
// CHECK:STDOUT: !146 = !DILocation(line: 8, column: 3, scope: !144)
// CHECK:STDOUT: !147 = !{!145}
// CHECK:STDOUT: !148 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.8b0bf151036f045a:core.Destroy.Core", scope: null, file: !124, line: 8, type: !131, spFlags: DISPFlagDefinition, unit: !125, retainedNodes: !151)
// CHECK:STDOUT: !149 = !DILocalVariable(arg: 1, scope: !148, type: !39)
// CHECK:STDOUT: !150 = !DILocation(line: 8, column: 3, scope: !148)
// CHECK:STDOUT: !151 = !{!149}
// CHECK:STDOUT: !152 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.92e03c783d6af646:core.Destroy.Core", scope: null, file: !124, line: 8, type: !131, spFlags: DISPFlagDefinition, unit: !125, retainedNodes: !155)
// CHECK:STDOUT: !153 = !DILocalVariable(arg: 1, scope: !152, type: !39)
// CHECK:STDOUT: !154 = !DILocation(line: 8, column: 3, scope: !152)
// CHECK:STDOUT: !155 = !{!153}
// CHECK:STDOUT: !156 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.92e03c783d6af646:core.Destroy.Core", scope: null, file: !124, line: 8, type: !131, spFlags: DISPFlagDefinition, unit: !125, retainedNodes: !159)
// CHECK:STDOUT: !157 = !DILocalVariable(arg: 1, scope: !156, type: !39)
// CHECK:STDOUT: !158 = !DILocation(line: 8, column: 3, scope: !156)
// CHECK:STDOUT: !159 = !{!157}
// CHECK:STDOUT: !160 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.452a47fbe5382294:core.Destroy.Core", scope: null, file: !124, line: 8, type: !131, spFlags: DISPFlagDefinition, unit: !125, retainedNodes: !163)
// CHECK:STDOUT: !161 = !DILocalVariable(arg: 1, scope: !160, type: !39)
// CHECK:STDOUT: !162 = !DILocation(line: 8, column: 3, scope: !160)
// CHECK:STDOUT: !163 = !{!161}
// CHECK:STDOUT: !164 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.452a47fbe5382294:core.Destroy.Core", scope: null, file: !124, line: 8, type: !131, spFlags: DISPFlagDefinition, unit: !125, retainedNodes: !167)
// CHECK:STDOUT: !165 = !DILocalVariable(arg: 1, scope: !164, type: !39)
// CHECK:STDOUT: !166 = !DILocation(line: 8, column: 3, scope: !164)
// CHECK:STDOUT: !167 = !{!165}
// CHECK:STDOUT: !168 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.12212b84320f2fd6:core.Destroy.Core", scope: null, file: !124, line: 8, type: !131, spFlags: DISPFlagDefinition, unit: !125, retainedNodes: !171)
// CHECK:STDOUT: !169 = !DILocalVariable(arg: 1, scope: !168, type: !39)
// CHECK:STDOUT: !170 = !DILocation(line: 8, column: 3, scope: !168)
// CHECK:STDOUT: !171 = !{!169}
// CHECK:STDOUT: !172 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.fb3d597a6b1c4ee1", scope: null, file: !124, line: 7, type: !93, spFlags: DISPFlagDefinition, unit: !125)
// CHECK:STDOUT: !173 = !DILocation(line: 8, column: 3, scope: !172)
// CHECK:STDOUT: !174 = !DILocation(line: 8, column: 21, scope: !172)
// CHECK:STDOUT: !175 = !DILocation(line: 8, column: 22, scope: !172)
// CHECK:STDOUT: !176 = !DILocation(line: 8, column: 30, scope: !172)
// CHECK:STDOUT: !177 = !DILocation(line: 8, column: 38, scope: !172)
// CHECK:STDOUT: !179 = !DILocation(line: 7, column: 1, scope: !172)
// CHECK:STDOUT: !180 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.2eaa3ac913700940", scope: null, file: !124, line: 7, type: !93, spFlags: DISPFlagDefinition, unit: !125)
// CHECK:STDOUT: !181 = !DILocation(line: 8, column: 3, scope: !180)
// CHECK:STDOUT: !182 = !DILocation(line: 8, column: 21, scope: !180)
// CHECK:STDOUT: !183 = !DILocation(line: 8, column: 22, scope: !180)
// CHECK:STDOUT: !184 = !DILocation(line: 8, column: 30, scope: !180)
// CHECK:STDOUT: !185 = !DILocation(line: 8, column: 38, scope: !180)
// CHECK:STDOUT: !187 = !DILocation(line: 7, column: 1, scope: !180)
// CHECK:STDOUT: !188 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.be297cc7300419fe", scope: null, file: !124, line: 7, type: !93, spFlags: DISPFlagDefinition, unit: !125)
// CHECK:STDOUT: !189 = !DILocation(line: 8, column: 3, scope: !188)
// CHECK:STDOUT: !190 = !DILocation(line: 8, column: 22, scope: !188)
// CHECK:STDOUT: !191 = !DILocation(line: 8, column: 30, scope: !188)
// CHECK:STDOUT: !192 = !DILocation(line: 8, column: 38, scope: !188)
// CHECK:STDOUT: !193 = !DILocation(line: 8, column: 21, scope: !188)
// CHECK:STDOUT: !195 = !DILocation(line: 7, column: 1, scope: !188)
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
// CHECK:STDOUT: define void @_CG.Main() #0 !dbg !198 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Main.75a9ebb2041c173b(), !dbg !199
// CHECK:STDOUT:   call void @_CF.Main.f5d9b693014ce43e(), !dbg !200
// CHECK:STDOUT:   call void @_CF.Main.84270bb113cb754b(), !dbg !201
// CHECK:STDOUT:   ret void, !dbg !202
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.cb8adced19392b99:core.Destroy.Core"(ptr %self) #0 !dbg !203 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !205
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.8b0bf151036f045a:core.Destroy.Core"(ptr %self) #0 !dbg !207 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !209
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.f38e1c3ccdfe019b:core.Destroy.Core"(ptr %self) #0 !dbg !211 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc9_3.1.b = getelementptr inbounds nuw { {}, {} }, ptr %self, i32 0, i32 1, !dbg !213
// CHECK:STDOUT:   call void @"_CSelfDestruct.8b0bf151036f045a:core.Destroy.Core"(ptr %.loc9_3.1.b), !dbg !213
// CHECK:STDOUT:   %.loc9_3.2.a = getelementptr inbounds nuw { {}, {} }, ptr %self, i32 0, i32 0, !dbg !213
// CHECK:STDOUT:   call void @"_CSelfDestruct.8b0bf151036f045a:core.Destroy.Core"(ptr %.loc9_3.2.a), !dbg !213
// CHECK:STDOUT:   ret void, !dbg !213
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.f38e1c3ccdfe019b:core.Destroy.Core"(ptr %self) #0 !dbg !215 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.f38e1c3ccdfe019b:core.Destroy.Core"(ptr %self), !dbg !217
// CHECK:STDOUT:   ret void, !dbg !217
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.4c2c335412ec52fa:core.Destroy.Core"(ptr %self) #0 !dbg !219 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %tuple.elem1.tuple.elem = getelementptr inbounds nuw { { {}, {} }, {} }, ptr %self, i32 0, i32 1, !dbg !221
// CHECK:STDOUT:   call void @"_CSelfDestruct.8b0bf151036f045a:core.Destroy.Core"(ptr %tuple.elem1.tuple.elem), !dbg !221
// CHECK:STDOUT:   %tuple.elem0.tuple.elem = getelementptr inbounds nuw { { {}, {} }, {} }, ptr %self, i32 0, i32 0, !dbg !221
// CHECK:STDOUT:   call void @"_CSelfDestruct.f38e1c3ccdfe019b:core.Destroy.Core"(ptr %tuple.elem0.tuple.elem), !dbg !221
// CHECK:STDOUT:   ret void, !dbg !221
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.4c2c335412ec52fa:core.Destroy.Core"(ptr %self) #0 !dbg !223 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.4c2c335412ec52fa:core.Destroy.Core"(ptr %self), !dbg !225
// CHECK:STDOUT:   ret void, !dbg !225
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.12212b84320f2fd6:core.Destroy.Core"(ptr %self) #0 !dbg !227 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !229
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.0950f8aecfaebf0b:core.Destroy.Core"(ptr %self) #0 !dbg !231 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc9_3.1.b = getelementptr inbounds nuw { {}, {} }, ptr %self, i32 0, i32 1, !dbg !233
// CHECK:STDOUT:   call void @"_CSelfDestruct.12212b84320f2fd6:core.Destroy.Core"(ptr %.loc9_3.1.b), !dbg !233
// CHECK:STDOUT:   %.loc9_3.2.a = getelementptr inbounds nuw { {}, {} }, ptr %self, i32 0, i32 0, !dbg !233
// CHECK:STDOUT:   call void @"_CSelfDestruct.12212b84320f2fd6:core.Destroy.Core"(ptr %.loc9_3.2.a), !dbg !233
// CHECK:STDOUT:   ret void, !dbg !233
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.0950f8aecfaebf0b:core.Destroy.Core"(ptr %self) #0 !dbg !235 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.0950f8aecfaebf0b:core.Destroy.Core"(ptr %self), !dbg !237
// CHECK:STDOUT:   ret void, !dbg !237
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.2f43e328c637647f:core.Destroy.Core"(ptr %self) #0 !dbg !239 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %tuple.elem1.tuple.elem = getelementptr inbounds nuw { { {}, {} }, {} }, ptr %self, i32 0, i32 1, !dbg !241
// CHECK:STDOUT:   call void @"_CSelfDestruct.12212b84320f2fd6:core.Destroy.Core"(ptr %tuple.elem1.tuple.elem), !dbg !241
// CHECK:STDOUT:   %tuple.elem0.tuple.elem = getelementptr inbounds nuw { { {}, {} }, {} }, ptr %self, i32 0, i32 0, !dbg !241
// CHECK:STDOUT:   call void @"_CSelfDestruct.0950f8aecfaebf0b:core.Destroy.Core"(ptr %tuple.elem0.tuple.elem), !dbg !241
// CHECK:STDOUT:   ret void, !dbg !241
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.2f43e328c637647f:core.Destroy.Core"(ptr %self) #0 !dbg !243 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.2f43e328c637647f:core.Destroy.Core"(ptr %self), !dbg !245
// CHECK:STDOUT:   ret void, !dbg !245
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.2970be6ba4c2615c:core.Destroy.Core"(ptr %self) #0 !dbg !247 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc9_3.1.b = getelementptr inbounds nuw { {}, {} }, ptr %self, i32 0, i32 1, !dbg !249
// CHECK:STDOUT:   call void @"_CSelfDestruct.12212b84320f2fd6:core.Destroy.Core"(ptr %.loc9_3.1.b), !dbg !249
// CHECK:STDOUT:   %.loc9_3.2.a = getelementptr inbounds nuw { {}, {} }, ptr %self, i32 0, i32 0, !dbg !249
// CHECK:STDOUT:   call void @"_CSelfDestruct.8b0bf151036f045a:core.Destroy.Core"(ptr %.loc9_3.2.a), !dbg !249
// CHECK:STDOUT:   ret void, !dbg !249
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.2970be6ba4c2615c:core.Destroy.Core"(ptr %self) #0 !dbg !251 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.2970be6ba4c2615c:core.Destroy.Core"(ptr %self), !dbg !253
// CHECK:STDOUT:   ret void, !dbg !253
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.680f7fa3ef86df7a:core.Destroy.Core"(ptr %self) #0 !dbg !255 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc9_3.1.b = getelementptr inbounds nuw { {}, {} }, ptr %self, i32 0, i32 1, !dbg !257
// CHECK:STDOUT:   call void @"_CSelfDestruct.12212b84320f2fd6:core.Destroy.Core"(ptr %.loc9_3.1.b), !dbg !257
// CHECK:STDOUT:   %.loc9_3.2.a = getelementptr inbounds nuw { {}, {} }, ptr %self, i32 0, i32 0, !dbg !257
// CHECK:STDOUT:   call void @"_CSelfDestruct.8b0bf151036f045a:core.Destroy.Core"(ptr %.loc9_3.2.a), !dbg !257
// CHECK:STDOUT:   ret void, !dbg !257
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.680f7fa3ef86df7a:core.Destroy.Core"(ptr %self) #0 !dbg !259 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.680f7fa3ef86df7a:core.Destroy.Core"(ptr %self), !dbg !261
// CHECK:STDOUT:   ret void, !dbg !261
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.bb4067704e9a198c:core.Destroy.Core"(ptr %self) #0 !dbg !263 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %tuple.elem1.tuple.elem = getelementptr inbounds nuw { { {}, {} }, {} }, ptr %self, i32 0, i32 1, !dbg !265
// CHECK:STDOUT:   call void @"_CSelfDestruct.12212b84320f2fd6:core.Destroy.Core"(ptr %tuple.elem1.tuple.elem), !dbg !265
// CHECK:STDOUT:   %tuple.elem0.tuple.elem = getelementptr inbounds nuw { { {}, {} }, {} }, ptr %self, i32 0, i32 0, !dbg !265
// CHECK:STDOUT:   call void @"_CSelfDestruct.680f7fa3ef86df7a:core.Destroy.Core"(ptr %tuple.elem0.tuple.elem), !dbg !265
// CHECK:STDOUT:   ret void, !dbg !265
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.bb4067704e9a198c:core.Destroy.Core"(ptr %self) #0 !dbg !267 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.bb4067704e9a198c:core.Destroy.Core"(ptr %self), !dbg !269
// CHECK:STDOUT:   ret void, !dbg !269
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.75a9ebb2041c173b() #0 !dbg !271 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %w.var = alloca { { {}, {} }, {} }, align 1, !dbg !272
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %w.var), !dbg !272
// CHECK:STDOUT:   %tuple.elem0.a28.tuple.elem = getelementptr inbounds nuw { { {}, {} }, {} }, ptr %w.var, i32 0, i32 0, !dbg !274
// CHECK:STDOUT:   %.d62.a = getelementptr inbounds nuw { {}, {} }, ptr %tuple.elem0.a28.tuple.elem, i32 0, i32 0, !dbg !273
// CHECK:STDOUT:   call void @_CMake.Main(ptr %.d62.a), !dbg !275
// CHECK:STDOUT:   %.162.b = getelementptr inbounds nuw { {}, {} }, ptr %tuple.elem0.a28.tuple.elem, i32 0, i32 1, !dbg !273
// CHECK:STDOUT:   call void @_CMake.Main(ptr %.162.b), !dbg !276
// CHECK:STDOUT:   %tuple.elem1.79b.tuple.elem = getelementptr inbounds nuw { { {}, {} }, {} }, ptr %w.var, i32 0, i32 1, !dbg !274
// CHECK:STDOUT:   call void @_CMake.Main(ptr %tuple.elem1.79b.tuple.elem), !dbg !277
// CHECK:STDOUT:   call void @"_CSelfDestruct.4c2c335412ec52fa:core.Destroy.Core"(ptr %w.var), !dbg !272
// CHECK:STDOUT:   ret void, !dbg !279
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.f5d9b693014ce43e() #0 !dbg !280 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %w.var = alloca { { {}, {} }, {} }, align 1, !dbg !281
// CHECK:STDOUT:   %.7a56fd.1.temp = alloca {}, align 1, !dbg !281
// CHECK:STDOUT:   %.7a56fd.2.temp = alloca {}, align 1, !dbg !281
// CHECK:STDOUT:   %.7a56fd.3.temp = alloca {}, align 1, !dbg !281
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %w.var), !dbg !281
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.7a56fd.1.temp), !dbg !281
// CHECK:STDOUT:   call void @_CMake.Main(ptr %.7a56fd.1.temp), !dbg !282
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.7a56fd.2.temp), !dbg !281
// CHECK:STDOUT:   call void @_CMake.Main(ptr %.7a56fd.2.temp), !dbg !283
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.7a56fd.3.temp), !dbg !281
// CHECK:STDOUT:   call void @_CMake.Main(ptr %.7a56fd.3.temp), !dbg !285
// CHECK:STDOUT:   %tuple.elem0.25a.tuple.elem = getelementptr inbounds nuw { { {}, {} }, {} }, ptr %w.var, i32 0, i32 0, !dbg !286
// CHECK:STDOUT:   %.e68.a = getelementptr inbounds nuw { {}, {} }, ptr %tuple.elem0.25a.tuple.elem, i32 0, i32 0, !dbg !284
// CHECK:STDOUT:   call void @"_CConvert.X.Main:ImplicitAs.be297cc7300419fe.Core"(ptr %.e68.a, ptr %.7a56fd.1.temp), !dbg !284
// CHECK:STDOUT:   %.9f2.b = getelementptr inbounds nuw { {}, {} }, ptr %tuple.elem0.25a.tuple.elem, i32 0, i32 1, !dbg !284
// CHECK:STDOUT:   call void @"_CConvert.X.Main:ImplicitAs.be297cc7300419fe.Core"(ptr %.9f2.b, ptr %.7a56fd.2.temp), !dbg !284
// CHECK:STDOUT:   %tuple.elem1.0e1.tuple.elem = getelementptr inbounds nuw { { {}, {} }, {} }, ptr %w.var, i32 0, i32 1, !dbg !286
// CHECK:STDOUT:   call void @"_CConvert.X.Main:ImplicitAs.be297cc7300419fe.Core"(ptr %tuple.elem1.0e1.tuple.elem, ptr %.7a56fd.3.temp), !dbg !286
// CHECK:STDOUT:   call void @"_CSelfDestruct.2f43e328c637647f:core.Destroy.Core"(ptr %w.var), !dbg !281
// CHECK:STDOUT:   ret void, !dbg !288
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.84270bb113cb754b() #0 !dbg !289 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %w.var = alloca { { {}, {} }, {} }, align 1, !dbg !290
// CHECK:STDOUT:   %.7a56fd.4.temp = alloca {}, align 1, !dbg !290
// CHECK:STDOUT:   %.7a56fd.5.temp = alloca {}, align 1, !dbg !290
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %w.var), !dbg !290
// CHECK:STDOUT:   %tuple.elem0.d0d.tuple.elem = getelementptr inbounds nuw { { {}, {} }, {} }, ptr %w.var, i32 0, i32 0, !dbg !292
// CHECK:STDOUT:   %.b76.a = getelementptr inbounds nuw { {}, {} }, ptr %tuple.elem0.d0d.tuple.elem, i32 0, i32 0, !dbg !291
// CHECK:STDOUT:   call void @_CMake.Main(ptr %.b76.a), !dbg !293
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.7a56fd.4.temp), !dbg !290
// CHECK:STDOUT:   call void @_CMake.Main(ptr %.7a56fd.4.temp), !dbg !294
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.7a56fd.5.temp), !dbg !290
// CHECK:STDOUT:   call void @_CMake.Main(ptr %.7a56fd.5.temp), !dbg !295
// CHECK:STDOUT:   %.25e.b = getelementptr inbounds nuw { {}, {} }, ptr %tuple.elem0.d0d.tuple.elem, i32 0, i32 1, !dbg !291
// CHECK:STDOUT:   call void @"_CConvert.X.Main:ImplicitAs.be297cc7300419fe.Core"(ptr %.25e.b, ptr %.7a56fd.4.temp), !dbg !291
// CHECK:STDOUT:   %tuple.elem1.c76.tuple.elem = getelementptr inbounds nuw { { {}, {} }, {} }, ptr %w.var, i32 0, i32 1, !dbg !292
// CHECK:STDOUT:   call void @"_CConvert.X.Main:ImplicitAs.be297cc7300419fe.Core"(ptr %tuple.elem1.c76.tuple.elem, ptr %.7a56fd.5.temp), !dbg !292
// CHECK:STDOUT:   call void @"_CSelfDestruct.bb4067704e9a198c:core.Destroy.Core"(ptr %w.var), !dbg !290
// CHECK:STDOUT:   ret void, !dbg !297
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 7, 6, 5, 4, 3, 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!197}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !39 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !92 = !{null}
// CHECK:STDOUT: !93 = !DISubroutineType(types: !92)
// CHECK:STDOUT: !130 = !{null, !39}
// CHECK:STDOUT: !131 = !DISubroutineType(types: !130)
// CHECK:STDOUT: !196 = !DIFile(filename: "initialize_from_nested_literal.carbon", directory: "")
// CHECK:STDOUT: !197 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !196, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !198 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !196, line: 24, type: !93, spFlags: DISPFlagDefinition, unit: !197)
// CHECK:STDOUT: !199 = !DILocation(line: 25, column: 3, scope: !198)
// CHECK:STDOUT: !200 = !DILocation(line: 27, column: 3, scope: !198)
// CHECK:STDOUT: !201 = !DILocation(line: 29, column: 3, scope: !198)
// CHECK:STDOUT: !202 = !DILocation(line: 24, column: 1, scope: !198)
// CHECK:STDOUT: !203 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.cb8adced19392b99:core.Destroy.Core", scope: null, file: !196, line: 9, type: !131, spFlags: DISPFlagDefinition, unit: !197, retainedNodes: !206)
// CHECK:STDOUT: !204 = !DILocalVariable(arg: 1, scope: !203, type: !39)
// CHECK:STDOUT: !205 = !DILocation(line: 9, column: 3, scope: !203)
// CHECK:STDOUT: !206 = !{!204}
// CHECK:STDOUT: !207 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.8b0bf151036f045a:core.Destroy.Core", scope: null, file: !196, line: 9, type: !131, spFlags: DISPFlagDefinition, unit: !197, retainedNodes: !210)
// CHECK:STDOUT: !208 = !DILocalVariable(arg: 1, scope: !207, type: !39)
// CHECK:STDOUT: !209 = !DILocation(line: 9, column: 3, scope: !207)
// CHECK:STDOUT: !210 = !{!208}
// CHECK:STDOUT: !211 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.f38e1c3ccdfe019b:core.Destroy.Core", scope: null, file: !196, line: 9, type: !131, spFlags: DISPFlagDefinition, unit: !197, retainedNodes: !214)
// CHECK:STDOUT: !212 = !DILocalVariable(arg: 1, scope: !211, type: !39)
// CHECK:STDOUT: !213 = !DILocation(line: 9, column: 3, scope: !211)
// CHECK:STDOUT: !214 = !{!212}
// CHECK:STDOUT: !215 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.f38e1c3ccdfe019b:core.Destroy.Core", scope: null, file: !196, line: 9, type: !131, spFlags: DISPFlagDefinition, unit: !197, retainedNodes: !218)
// CHECK:STDOUT: !216 = !DILocalVariable(arg: 1, scope: !215, type: !39)
// CHECK:STDOUT: !217 = !DILocation(line: 9, column: 3, scope: !215)
// CHECK:STDOUT: !218 = !{!216}
// CHECK:STDOUT: !219 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.4c2c335412ec52fa:core.Destroy.Core", scope: null, file: !196, line: 9, type: !131, spFlags: DISPFlagDefinition, unit: !197, retainedNodes: !222)
// CHECK:STDOUT: !220 = !DILocalVariable(arg: 1, scope: !219, type: !39)
// CHECK:STDOUT: !221 = !DILocation(line: 9, column: 3, scope: !219)
// CHECK:STDOUT: !222 = !{!220}
// CHECK:STDOUT: !223 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.4c2c335412ec52fa:core.Destroy.Core", scope: null, file: !196, line: 9, type: !131, spFlags: DISPFlagDefinition, unit: !197, retainedNodes: !226)
// CHECK:STDOUT: !224 = !DILocalVariable(arg: 1, scope: !223, type: !39)
// CHECK:STDOUT: !225 = !DILocation(line: 9, column: 3, scope: !223)
// CHECK:STDOUT: !226 = !{!224}
// CHECK:STDOUT: !227 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.12212b84320f2fd6:core.Destroy.Core", scope: null, file: !196, line: 9, type: !131, spFlags: DISPFlagDefinition, unit: !197, retainedNodes: !230)
// CHECK:STDOUT: !228 = !DILocalVariable(arg: 1, scope: !227, type: !39)
// CHECK:STDOUT: !229 = !DILocation(line: 9, column: 3, scope: !227)
// CHECK:STDOUT: !230 = !{!228}
// CHECK:STDOUT: !231 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.0950f8aecfaebf0b:core.Destroy.Core", scope: null, file: !196, line: 9, type: !131, spFlags: DISPFlagDefinition, unit: !197, retainedNodes: !234)
// CHECK:STDOUT: !232 = !DILocalVariable(arg: 1, scope: !231, type: !39)
// CHECK:STDOUT: !233 = !DILocation(line: 9, column: 3, scope: !231)
// CHECK:STDOUT: !234 = !{!232}
// CHECK:STDOUT: !235 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.0950f8aecfaebf0b:core.Destroy.Core", scope: null, file: !196, line: 9, type: !131, spFlags: DISPFlagDefinition, unit: !197, retainedNodes: !238)
// CHECK:STDOUT: !236 = !DILocalVariable(arg: 1, scope: !235, type: !39)
// CHECK:STDOUT: !237 = !DILocation(line: 9, column: 3, scope: !235)
// CHECK:STDOUT: !238 = !{!236}
// CHECK:STDOUT: !239 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.2f43e328c637647f:core.Destroy.Core", scope: null, file: !196, line: 9, type: !131, spFlags: DISPFlagDefinition, unit: !197, retainedNodes: !242)
// CHECK:STDOUT: !240 = !DILocalVariable(arg: 1, scope: !239, type: !39)
// CHECK:STDOUT: !241 = !DILocation(line: 9, column: 3, scope: !239)
// CHECK:STDOUT: !242 = !{!240}
// CHECK:STDOUT: !243 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.2f43e328c637647f:core.Destroy.Core", scope: null, file: !196, line: 9, type: !131, spFlags: DISPFlagDefinition, unit: !197, retainedNodes: !246)
// CHECK:STDOUT: !244 = !DILocalVariable(arg: 1, scope: !243, type: !39)
// CHECK:STDOUT: !245 = !DILocation(line: 9, column: 3, scope: !243)
// CHECK:STDOUT: !246 = !{!244}
// CHECK:STDOUT: !247 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.2970be6ba4c2615c:core.Destroy.Core", scope: null, file: !196, line: 9, type: !131, spFlags: DISPFlagDefinition, unit: !197, retainedNodes: !250)
// CHECK:STDOUT: !248 = !DILocalVariable(arg: 1, scope: !247, type: !39)
// CHECK:STDOUT: !249 = !DILocation(line: 9, column: 3, scope: !247)
// CHECK:STDOUT: !250 = !{!248}
// CHECK:STDOUT: !251 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.2970be6ba4c2615c:core.Destroy.Core", scope: null, file: !196, line: 9, type: !131, spFlags: DISPFlagDefinition, unit: !197, retainedNodes: !254)
// CHECK:STDOUT: !252 = !DILocalVariable(arg: 1, scope: !251, type: !39)
// CHECK:STDOUT: !253 = !DILocation(line: 9, column: 3, scope: !251)
// CHECK:STDOUT: !254 = !{!252}
// CHECK:STDOUT: !255 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.680f7fa3ef86df7a:core.Destroy.Core", scope: null, file: !196, line: 9, type: !131, spFlags: DISPFlagDefinition, unit: !197, retainedNodes: !258)
// CHECK:STDOUT: !256 = !DILocalVariable(arg: 1, scope: !255, type: !39)
// CHECK:STDOUT: !257 = !DILocation(line: 9, column: 3, scope: !255)
// CHECK:STDOUT: !258 = !{!256}
// CHECK:STDOUT: !259 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.680f7fa3ef86df7a:core.Destroy.Core", scope: null, file: !196, line: 9, type: !131, spFlags: DISPFlagDefinition, unit: !197, retainedNodes: !262)
// CHECK:STDOUT: !260 = !DILocalVariable(arg: 1, scope: !259, type: !39)
// CHECK:STDOUT: !261 = !DILocation(line: 9, column: 3, scope: !259)
// CHECK:STDOUT: !262 = !{!260}
// CHECK:STDOUT: !263 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.bb4067704e9a198c:core.Destroy.Core", scope: null, file: !196, line: 9, type: !131, spFlags: DISPFlagDefinition, unit: !197, retainedNodes: !266)
// CHECK:STDOUT: !264 = !DILocalVariable(arg: 1, scope: !263, type: !39)
// CHECK:STDOUT: !265 = !DILocation(line: 9, column: 3, scope: !263)
// CHECK:STDOUT: !266 = !{!264}
// CHECK:STDOUT: !267 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.bb4067704e9a198c:core.Destroy.Core", scope: null, file: !196, line: 9, type: !131, spFlags: DISPFlagDefinition, unit: !197, retainedNodes: !270)
// CHECK:STDOUT: !268 = !DILocalVariable(arg: 1, scope: !267, type: !39)
// CHECK:STDOUT: !269 = !DILocation(line: 9, column: 3, scope: !267)
// CHECK:STDOUT: !270 = !{!268}
// CHECK:STDOUT: !271 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.75a9ebb2041c173b", scope: null, file: !196, line: 8, type: !93, spFlags: DISPFlagDefinition, unit: !197)
// CHECK:STDOUT: !272 = !DILocation(line: 9, column: 3, scope: !271)
// CHECK:STDOUT: !273 = !DILocation(line: 9, column: 22, scope: !271)
// CHECK:STDOUT: !274 = !DILocation(line: 9, column: 21, scope: !271)
// CHECK:STDOUT: !275 = !DILocation(line: 9, column: 28, scope: !271)
// CHECK:STDOUT: !276 = !DILocation(line: 9, column: 41, scope: !271)
// CHECK:STDOUT: !277 = !DILocation(line: 9, column: 50, scope: !271)
// CHECK:STDOUT: !279 = !DILocation(line: 8, column: 1, scope: !271)
// CHECK:STDOUT: !280 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.f5d9b693014ce43e", scope: null, file: !196, line: 8, type: !93, spFlags: DISPFlagDefinition, unit: !197)
// CHECK:STDOUT: !281 = !DILocation(line: 9, column: 3, scope: !280)
// CHECK:STDOUT: !282 = !DILocation(line: 9, column: 28, scope: !280)
// CHECK:STDOUT: !283 = !DILocation(line: 9, column: 41, scope: !280)
// CHECK:STDOUT: !284 = !DILocation(line: 9, column: 22, scope: !280)
// CHECK:STDOUT: !285 = !DILocation(line: 9, column: 50, scope: !280)
// CHECK:STDOUT: !286 = !DILocation(line: 9, column: 21, scope: !280)
// CHECK:STDOUT: !288 = !DILocation(line: 8, column: 1, scope: !280)
// CHECK:STDOUT: !289 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.84270bb113cb754b", scope: null, file: !196, line: 8, type: !93, spFlags: DISPFlagDefinition, unit: !197)
// CHECK:STDOUT: !290 = !DILocation(line: 9, column: 3, scope: !289)
// CHECK:STDOUT: !291 = !DILocation(line: 9, column: 22, scope: !289)
// CHECK:STDOUT: !292 = !DILocation(line: 9, column: 21, scope: !289)
// CHECK:STDOUT: !293 = !DILocation(line: 9, column: 28, scope: !289)
// CHECK:STDOUT: !294 = !DILocation(line: 9, column: 41, scope: !289)
// CHECK:STDOUT: !295 = !DILocation(line: 9, column: 50, scope: !289)
// CHECK:STDOUT: !297 = !DILocation(line: 8, column: 1, scope: !289)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'as.carbon'
// CHECK:STDOUT: source_filename = "as.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTest1.Main(i32 %n) #0 !dbg !300 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %AsI32.call = call i32 @_CAsI32.Main.84588f41d61dafba(i32 %n), !dbg !304
// CHECK:STDOUT:   ret void, !dbg !307
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @"_CConvert.C.Main:As.84588f41d61dafba.Core"(ptr %self) #0 !dbg !309 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc15_42.1.n = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !314
// CHECK:STDOUT:   %.loc15_42.2 = load i32, ptr %.loc15_42.1.n, align 4, !dbg !314
// CHECK:STDOUT:   ret i32 %.loc15_42.2, !dbg !315
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTest2.Main(ptr %c) #0 !dbg !317 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %AsI32.call = call i32 @_CAsI32.Main.9522ff908e97e267(ptr %c), !dbg !321
// CHECK:STDOUT:   ret void, !dbg !324
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CAsI32.Main.84588f41d61dafba(i32 %x) #0 !dbg !326 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %x, !dbg !332
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CAsI32.Main.9522ff908e97e267(ptr %x) #0 !dbg !334 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %C.as.As.impl.Convert.call = call i32 @"_CConvert.C.Main:As.84588f41d61dafba.Core"(ptr %x), !dbg !339
// CHECK:STDOUT:   ret i32 %C.as.As.impl.Convert.call, !dbg !340
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!299}
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
// CHECK:STDOUT: !94 = !{null, !36}
// CHECK:STDOUT: !95 = !DISubroutineType(types: !94)
// CHECK:STDOUT: !130 = !{null, !39}
// CHECK:STDOUT: !131 = !DISubroutineType(types: !130)
// CHECK:STDOUT: !298 = !DIFile(filename: "as.carbon", directory: "")
// CHECK:STDOUT: !299 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !298, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !300 = distinct !DISubprogram(name: "Test1", linkageName: "_CTest1.Main", scope: null, file: !298, line: 8, type: !95, spFlags: DISPFlagDefinition, unit: !299, retainedNodes: !308)
// CHECK:STDOUT: !301 = !DILocalVariable(arg: 1, scope: !300, type: !36)
// CHECK:STDOUT: !304 = !DILocation(line: 9, column: 16, scope: !300)
// CHECK:STDOUT: !307 = !DILocation(line: 8, column: 1, scope: !300)
// CHECK:STDOUT: !308 = !{!301}
// CHECK:STDOUT: !309 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.C.Main:As.84588f41d61dafba.Core", scope: null, file: !298, line: 15, type: !41, spFlags: DISPFlagDefinition, unit: !299, retainedNodes: !316)
// CHECK:STDOUT: !310 = !DILocalVariable(arg: 1, scope: !309, type: !39)
// CHECK:STDOUT: !314 = !DILocation(line: 15, column: 38, scope: !309)
// CHECK:STDOUT: !315 = !DILocation(line: 15, column: 31, scope: !309)
// CHECK:STDOUT: !316 = !{!310}
// CHECK:STDOUT: !317 = distinct !DISubprogram(name: "Test2", linkageName: "_CTest2.Main", scope: null, file: !298, line: 19, type: !131, spFlags: DISPFlagDefinition, unit: !299, retainedNodes: !325)
// CHECK:STDOUT: !318 = !DILocalVariable(arg: 1, scope: !317, type: !39)
// CHECK:STDOUT: !321 = !DILocation(line: 20, column: 16, scope: !317)
// CHECK:STDOUT: !324 = !DILocation(line: 19, column: 1, scope: !317)
// CHECK:STDOUT: !325 = !{!318}
// CHECK:STDOUT: !326 = distinct !DISubprogram(name: "AsI32", linkageName: "_CAsI32.Main.84588f41d61dafba", scope: null, file: !298, line: 4, type: !38, spFlags: DISPFlagDefinition, unit: !299, retainedNodes: !333)
// CHECK:STDOUT: !327 = !DILocalVariable(arg: 1, scope: !326, type: !36)
// CHECK:STDOUT: !332 = !DILocation(line: 5, column: 3, scope: !326)
// CHECK:STDOUT: !333 = !{!327}
// CHECK:STDOUT: !334 = distinct !DISubprogram(name: "AsI32", linkageName: "_CAsI32.Main.9522ff908e97e267", scope: null, file: !298, line: 4, type: !41, spFlags: DISPFlagDefinition, unit: !299, retainedNodes: !341)
// CHECK:STDOUT: !335 = !DILocalVariable(arg: 1, scope: !334, type: !39)
// CHECK:STDOUT: !339 = !DILocation(line: 5, column: 10, scope: !334)
// CHECK:STDOUT: !340 = !DILocation(line: 5, column: 3, scope: !334)
// CHECK:STDOUT: !341 = !{!335}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'as_type.carbon'
// CHECK:STDOUT: source_filename = "as_type.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTest1.Main(i32 %n) #0 !dbg !344 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CAsType.Main.8122a03446648385(i32 %n), !dbg !348
// CHECK:STDOUT:   ret void, !dbg !349
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @"_CConvert.C.Main:As.84588f41d61dafba.Core"(ptr %self) #0 !dbg !351 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc15_42.1.n = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !356
// CHECK:STDOUT:   %.loc15_42.2 = load i32, ptr %.loc15_42.1.n, align 4, !dbg !356
// CHECK:STDOUT:   ret i32 %.loc15_42.2, !dbg !357
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTest2.Main(ptr %c) #0 !dbg !359 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CAsType.Main.3280a9d16838b780(ptr %c), !dbg !363
// CHECK:STDOUT:   ret void, !dbg !364
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CAsType.Main.8122a03446648385(i32 %x) #0 !dbg !366 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !372
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CAsType.Main.3280a9d16838b780(ptr %x) #0 !dbg !374 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %C.as.As.impl.Convert.call = call i32 @"_CConvert.C.Main:As.84588f41d61dafba.Core"(ptr %x), !dbg !377
// CHECK:STDOUT:   ret void, !dbg !380
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!343}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !36 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !39 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !40 = !{!36, !39}
// CHECK:STDOUT: !41 = !DISubroutineType(types: !40)
// CHECK:STDOUT: !94 = !{null, !36}
// CHECK:STDOUT: !95 = !DISubroutineType(types: !94)
// CHECK:STDOUT: !130 = !{null, !39}
// CHECK:STDOUT: !131 = !DISubroutineType(types: !130)
// CHECK:STDOUT: !342 = !DIFile(filename: "as_type.carbon", directory: "")
// CHECK:STDOUT: !343 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !342, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !344 = distinct !DISubprogram(name: "Test1", linkageName: "_CTest1.Main", scope: null, file: !342, line: 8, type: !95, spFlags: DISPFlagDefinition, unit: !343, retainedNodes: !350)
// CHECK:STDOUT: !345 = !DILocalVariable(arg: 1, scope: !344, type: !36)
// CHECK:STDOUT: !348 = !DILocation(line: 9, column: 3, scope: !344)
// CHECK:STDOUT: !349 = !DILocation(line: 8, column: 1, scope: !344)
// CHECK:STDOUT: !350 = !{!345}
// CHECK:STDOUT: !351 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.C.Main:As.84588f41d61dafba.Core", scope: null, file: !342, line: 15, type: !41, spFlags: DISPFlagDefinition, unit: !343, retainedNodes: !358)
// CHECK:STDOUT: !352 = !DILocalVariable(arg: 1, scope: !351, type: !39)
// CHECK:STDOUT: !356 = !DILocation(line: 15, column: 38, scope: !351)
// CHECK:STDOUT: !357 = !DILocation(line: 15, column: 31, scope: !351)
// CHECK:STDOUT: !358 = !{!352}
// CHECK:STDOUT: !359 = distinct !DISubprogram(name: "Test2", linkageName: "_CTest2.Main", scope: null, file: !342, line: 19, type: !131, spFlags: DISPFlagDefinition, unit: !343, retainedNodes: !365)
// CHECK:STDOUT: !360 = !DILocalVariable(arg: 1, scope: !359, type: !39)
// CHECK:STDOUT: !363 = !DILocation(line: 20, column: 3, scope: !359)
// CHECK:STDOUT: !364 = !DILocation(line: 19, column: 1, scope: !359)
// CHECK:STDOUT: !365 = !{!360}
// CHECK:STDOUT: !366 = distinct !DISubprogram(name: "AsType", linkageName: "_CAsType.Main.8122a03446648385", scope: null, file: !342, line: 4, type: !95, spFlags: DISPFlagDefinition, unit: !343, retainedNodes: !373)
// CHECK:STDOUT: !367 = !DILocalVariable(arg: 1, scope: !366, type: !36)
// CHECK:STDOUT: !372 = !DILocation(line: 4, column: 1, scope: !366)
// CHECK:STDOUT: !373 = !{!367}
// CHECK:STDOUT: !374 = distinct !DISubprogram(name: "AsType", linkageName: "_CAsType.Main.3280a9d16838b780", scope: null, file: !342, line: 4, type: !131, spFlags: DISPFlagDefinition, unit: !343, retainedNodes: !381)
// CHECK:STDOUT: !375 = !DILocalVariable(arg: 1, scope: !374, type: !39)
// CHECK:STDOUT: !377 = !DILocation(line: 5, column: 14, scope: !374)
// CHECK:STDOUT: !380 = !DILocation(line: 4, column: 1, scope: !374)
// CHECK:STDOUT: !381 = !{!375}
// CHECK:STDOUT:
