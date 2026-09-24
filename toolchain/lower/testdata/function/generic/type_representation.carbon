// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/function/generic/type_representation.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/function/generic/type_representation.carbon

interface Copy {
  fn Op(self) -> Self;
}

fn F[T: Copy & Core.Destroy](a: T) -> T {
  var _: T = a.(Copy.Op)();
  return a.(Copy.Op)();
}

impl i32 as Copy {
  fn Op(self: i32) -> i32 {
    return self;
  }
}

fn F_i32(a: i32) -> i32 {
  return F(a);
}

class X {
  var a: i32;
  var b: i32;

  impl as Copy {
    fn Op(self: X) -> X {
      return {.a = self.a, .b = self.b};
    }
  }
}

fn F_X(a: X) -> X {
  return F(a);
}

impl () as Copy {
  fn Op(unused self: ()) -> () {
    return ();
  }
}

fn F_empty_tuple(a: ()) -> () {
  return F(a);
}

impl (i32, i32) as Copy {
  fn Op(self: (i32, i32)) -> (i32, i32) {
    return self;
  }
}

fn F_two_tuple(a: (i32, i32)) -> (i32, i32) {
  return F(a);
}

impl ((i32, i32), X) as Copy {
  fn Op(self: ((i32, i32), X)) -> ((i32, i32), X) {
    return (self.0.(Copy.Op)(), self.1.(Copy.Op)());
  }
}

fn F_nested_tuple(a: ((i32, i32), X)) -> ((i32, i32), X) {
  return F(a);
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'type_representation.carbon'
// CHECK:STDOUT: source_filename = "type_representation.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @"_COp.Int.be1e879c1ad406d8.Core:Copy.Main"(i32 %self) #0 !dbg !32 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %self, !dbg !38
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CF_i32.Main(i32 %a) #0 !dbg !40 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %F.call = call i32 @_CF.Main.804fd121057d1d93(i32 %a), !dbg !46
// CHECK:STDOUT:   ret i32 %F.call, !dbg !47
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.6dae29ece9a708c4:core.Destroy.Core"(ptr %self) #0 !dbg !49 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !51
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !53 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !55
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !57 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self), !dbg !59
// CHECK:STDOUT:   ret void, !dbg !59
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_COp.X.Main:Copy.Main"(ptr sret({ i32, i32 }) %return, ptr %self) #0 !dbg !61 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc38_24.1.a = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 0, !dbg !66
// CHECK:STDOUT:   %.loc38_24.2 = load i32, ptr %.loc38_24.1.a, align 4, !dbg !66
// CHECK:STDOUT:   %.loc38_37.1.b = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 1, !dbg !67
// CHECK:STDOUT:   %.loc38_37.2 = load i32, ptr %.loc38_37.1.b, align 4, !dbg !67
// CHECK:STDOUT:   %.loc38_39.2.a = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 0, !dbg !68
// CHECK:STDOUT:   store i32 %.loc38_24.2, ptr %.loc38_39.2.a, align 4, !dbg !68
// CHECK:STDOUT:   %.loc38_39.4.b = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 1, !dbg !68
// CHECK:STDOUT:   store i32 %.loc38_37.2, ptr %.loc38_39.4.b, align 4, !dbg !68
// CHECK:STDOUT:   ret void, !dbg !69
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF_X.Main(ptr sret({ i32, i32 }) %return, ptr %a) #0 !dbg !71 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Main.9d368fc2e01d20c2(ptr %return, ptr %a), !dbg !77
// CHECK:STDOUT:   ret void, !dbg !78
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.b4c762864cd9a02b:core.Destroy.Core"(ptr %self) #0 !dbg !80 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc44_13.1.a = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 0, !dbg !82
// CHECK:STDOUT:   call void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %.loc44_13.1.a), !dbg !82
// CHECK:STDOUT:   %.loc44_13.2.b = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 1, !dbg !82
// CHECK:STDOUT:   call void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %.loc44_13.2.b), !dbg !82
// CHECK:STDOUT:   ret void, !dbg !82
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.b4c762864cd9a02b:core.Destroy.Core"(ptr %self) #0 !dbg !84 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.b4c762864cd9a02b:core.Destroy.Core"(ptr %self), !dbg !86
// CHECK:STDOUT:   ret void, !dbg !86
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.8b0bf151036f045a:core.Destroy.Core"(ptr %self) #0 !dbg !88 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc44_13.1.a = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 0, !dbg !90
// CHECK:STDOUT:   call void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %.loc44_13.1.a), !dbg !90
// CHECK:STDOUT:   %.loc44_13.2.b = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 1, !dbg !90
// CHECK:STDOUT:   call void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %.loc44_13.2.b), !dbg !90
// CHECK:STDOUT:   ret void, !dbg !90
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.8b0bf151036f045a:core.Destroy.Core"(ptr %self) #0 !dbg !92 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.8b0bf151036f045a:core.Destroy.Core"(ptr %self), !dbg !94
// CHECK:STDOUT:   ret void, !dbg !94
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_COp.61ea2aba74ab3bf1:Copy.Main"() #0 !dbg !96 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !100
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF_empty_tuple.Main() #0 !dbg !101 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Main.a5d0440dfa027ffe(), !dbg !106
// CHECK:STDOUT:   ret void, !dbg !107
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.61ea2aba74ab3bf1:core.Destroy.Core"(ptr %self) #0 !dbg !108 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !110
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_COp.e554695bf635d9c1:Copy.Main"(ptr sret({ i32, i32 }) %return, ptr %self) #0 !dbg !112 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %tuple.elem0.loc59_12.1.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 0, !dbg !117
// CHECK:STDOUT:   %tuple.elem0.loc59_12.1.tuple.elem.load = load i32, ptr %tuple.elem0.loc59_12.1.tuple.elem, align 4, !dbg !117
// CHECK:STDOUT:   %tuple.elem0.loc59_12.2.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 0, !dbg !117
// CHECK:STDOUT:   store i32 %tuple.elem0.loc59_12.1.tuple.elem.load, ptr %tuple.elem0.loc59_12.2.tuple.elem, align 4, !dbg !117
// CHECK:STDOUT:   %tuple.elem1.loc59_12.1.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 1, !dbg !117
// CHECK:STDOUT:   %tuple.elem1.loc59_12.1.tuple.elem.load = load i32, ptr %tuple.elem1.loc59_12.1.tuple.elem, align 4, !dbg !117
// CHECK:STDOUT:   %tuple.elem1.loc59_12.2.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 1, !dbg !117
// CHECK:STDOUT:   store i32 %tuple.elem1.loc59_12.1.tuple.elem.load, ptr %tuple.elem1.loc59_12.2.tuple.elem, align 4, !dbg !117
// CHECK:STDOUT:   ret void, !dbg !118
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF_two_tuple.Main(ptr sret({ i32, i32 }) %return, ptr %a) #0 !dbg !120 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Main.25802053f9465fe8(ptr %return, ptr %a), !dbg !126
// CHECK:STDOUT:   ret void, !dbg !127
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.e554695bf635d9c1:core.Destroy.Core"(ptr %self) #0 !dbg !129 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !131
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.e554695bf635d9c1:core.Destroy.Core"(ptr %self) #0 !dbg !133 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.e554695bf635d9c1:core.Destroy.Core"(ptr %self), !dbg !135
// CHECK:STDOUT:   ret void, !dbg !135
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_COp.ae1e3622d7875d32:Copy.Main"(ptr sret({ { i32, i32 }, { i32, i32 } }) %return, ptr %self) #0 !dbg !137 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %tuple.elem0.loc69_17.tuple.elem = getelementptr inbounds nuw { ptr, ptr }, ptr %self, i32 0, i32 0, !dbg !142
// CHECK:STDOUT:   %tuple.elem0.loc69_17.tuple.elem.load = load ptr, ptr %tuple.elem0.loc69_17.tuple.elem, align 8, !dbg !142
// CHECK:STDOUT:   %tuple.elem0.loc69_51.tuple.elem = getelementptr inbounds nuw { { i32, i32 }, { i32, i32 } }, ptr %return, i32 0, i32 0, !dbg !143
// CHECK:STDOUT:   call void @"_COp.e554695bf635d9c1:Copy.Main"(ptr %tuple.elem0.loc69_51.tuple.elem, ptr %tuple.elem0.loc69_17.tuple.elem.load), !dbg !142
// CHECK:STDOUT:   %tuple.elem1.loc69_37.tuple.elem = getelementptr inbounds nuw { ptr, ptr }, ptr %self, i32 0, i32 1, !dbg !144
// CHECK:STDOUT:   %tuple.elem1.loc69_37.tuple.elem.load = load ptr, ptr %tuple.elem1.loc69_37.tuple.elem, align 8, !dbg !144
// CHECK:STDOUT:   %tuple.elem1.loc69_51.tuple.elem = getelementptr inbounds nuw { { i32, i32 }, { i32, i32 } }, ptr %return, i32 0, i32 1, !dbg !143
// CHECK:STDOUT:   call void @"_COp.X.Main:Copy.Main"(ptr %tuple.elem1.loc69_51.tuple.elem, ptr %tuple.elem1.loc69_37.tuple.elem.load), !dbg !144
// CHECK:STDOUT:   ret void, !dbg !145
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF_nested_tuple.Main(ptr sret({ { i32, i32 }, { i32, i32 } }) %return, ptr %a) #0 !dbg !147 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Main.f37d80e656707351(ptr %return, ptr %a), !dbg !153
// CHECK:STDOUT:   ret void, !dbg !154
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.ae1e3622d7875d32:core.Destroy.Core"(ptr %self) #0 !dbg !156 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !158
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.ae1e3622d7875d32:core.Destroy.Core"(ptr %self) #0 !dbg !160 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.ae1e3622d7875d32:core.Destroy.Core"(ptr %self), !dbg !162
// CHECK:STDOUT:   ret void, !dbg !162
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CF.Main.804fd121057d1d93(i32 %a) #0 !dbg !164 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca i32, align 4, !dbg !169
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !169
// CHECK:STDOUT:   %Copy.WithSelf.Op.call.loc18 = call i32 @"_COp.Int.be1e879c1ad406d8.Core:Copy.Main"(i32 %a), !dbg !170
// CHECK:STDOUT:   store i32 %Copy.WithSelf.Op.call.loc18, ptr %_.var, align 4, !dbg !169
// CHECK:STDOUT:   %Copy.WithSelf.Op.call.loc19 = call i32 @"_COp.Int.be1e879c1ad406d8.Core:Copy.Main"(i32 %a), !dbg !172
// CHECK:STDOUT:   call void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %_.var), !dbg !169
// CHECK:STDOUT:   ret i32 %Copy.WithSelf.Op.call.loc19, !dbg !173
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.9d368fc2e01d20c2(ptr sret({ i32, i32 }) %return, ptr %a) #0 !dbg !175 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca { i32, i32 }, align 4, !dbg !180
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !180
// CHECK:STDOUT:   call void @"_COp.X.Main:Copy.Main"(ptr %_.var, ptr %a), !dbg !181
// CHECK:STDOUT:   call void @"_COp.X.Main:Copy.Main"(ptr %return, ptr %a), !dbg !183
// CHECK:STDOUT:   call void @"_CSelfDestruct.8b0bf151036f045a:core.Destroy.Core"(ptr %_.var), !dbg !180
// CHECK:STDOUT:   ret void, !dbg !184
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.a5d0440dfa027ffe() #0 !dbg !186 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca {}, align 1, !dbg !190
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !190
// CHECK:STDOUT:   call void @"_COp.61ea2aba74ab3bf1:Copy.Main"(), !dbg !191
// CHECK:STDOUT:   call void @"_COp.61ea2aba74ab3bf1:Copy.Main"(), !dbg !193
// CHECK:STDOUT:   call void @"_CSelfDestruct.61ea2aba74ab3bf1:core.Destroy.Core"(ptr %_.var), !dbg !190
// CHECK:STDOUT:   ret void, !dbg !194
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.25802053f9465fe8(ptr sret({ i32, i32 }) %return, ptr %a) #0 !dbg !195 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca { i32, i32 }, align 4, !dbg !200
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !200
// CHECK:STDOUT:   call void @"_COp.e554695bf635d9c1:Copy.Main"(ptr %_.var, ptr %a), !dbg !201
// CHECK:STDOUT:   call void @"_COp.e554695bf635d9c1:Copy.Main"(ptr %return, ptr %a), !dbg !203
// CHECK:STDOUT:   call void @"_CSelfDestruct.e554695bf635d9c1:core.Destroy.Core"(ptr %_.var), !dbg !200
// CHECK:STDOUT:   ret void, !dbg !204
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.f37d80e656707351(ptr sret({ { i32, i32 }, { i32, i32 } }) %return, ptr %a) #0 !dbg !206 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca { { i32, i32 }, { i32, i32 } }, align 4, !dbg !211
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !211
// CHECK:STDOUT:   call void @"_COp.ae1e3622d7875d32:Copy.Main"(ptr %_.var, ptr %a), !dbg !212
// CHECK:STDOUT:   call void @"_COp.ae1e3622d7875d32:Copy.Main"(ptr %return, ptr %a), !dbg !214
// CHECK:STDOUT:   call void @"_CSelfDestruct.ae1e3622d7875d32:core.Destroy.Core"(ptr %_.var), !dbg !211
// CHECK:STDOUT:   ret void, !dbg !215
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 4, 3, 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!19}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !18 = !DIFile(filename: "type_representation.carbon", directory: "")
// CHECK:STDOUT: !19 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !18, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !{!20, !20}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !23 = !{null, !20}
// CHECK:STDOUT: !24 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !25 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !26 = !{!25, !25}
// CHECK:STDOUT: !27 = !DISubroutineType(types: !26)
// CHECK:STDOUT: !28 = !{null, !25}
// CHECK:STDOUT: !29 = !DISubroutineType(types: !28)
// CHECK:STDOUT: !30 = !{null}
// CHECK:STDOUT: !31 = !DISubroutineType(types: !30)
// CHECK:STDOUT: !32 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Int.be1e879c1ad406d8.Core:Copy.Main", scope: null, file: !18, line: 23, type: !22, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !39)
// CHECK:STDOUT: !33 = !DILocalVariable(arg: 1, scope: !32, type: !20)
// CHECK:STDOUT: !38 = !DILocation(line: 24, column: 5, scope: !32)
// CHECK:STDOUT: !39 = !{!33}
// CHECK:STDOUT: !40 = distinct !DISubprogram(name: "F_i32", linkageName: "_CF_i32.Main", scope: null, file: !18, line: 28, type: !22, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !48)
// CHECK:STDOUT: !41 = !DILocalVariable(arg: 1, scope: !40, type: !20)
// CHECK:STDOUT: !46 = !DILocation(line: 29, column: 10, scope: !40)
// CHECK:STDOUT: !47 = !DILocation(line: 29, column: 3, scope: !40)
// CHECK:STDOUT: !48 = !{!41}
// CHECK:STDOUT: !49 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.6dae29ece9a708c4:core.Destroy.Core", scope: null, file: !18, line: 29, type: !24, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !52)
// CHECK:STDOUT: !50 = !DILocalVariable(arg: 1, scope: !49, type: !20)
// CHECK:STDOUT: !51 = !DILocation(line: 29, column: 10, scope: !49)
// CHECK:STDOUT: !52 = !{!50}
// CHECK:STDOUT: !53 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core", scope: null, file: !18, line: 29, type: !24, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !56)
// CHECK:STDOUT: !54 = !DILocalVariable(arg: 1, scope: !53, type: !20)
// CHECK:STDOUT: !55 = !DILocation(line: 29, column: 10, scope: !53)
// CHECK:STDOUT: !56 = !{!54}
// CHECK:STDOUT: !57 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.f58524831c37807e:core.Destroy.Core", scope: null, file: !18, line: 29, type: !24, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !60)
// CHECK:STDOUT: !58 = !DILocalVariable(arg: 1, scope: !57, type: !20)
// CHECK:STDOUT: !59 = !DILocation(line: 29, column: 10, scope: !57)
// CHECK:STDOUT: !60 = !{!58}
// CHECK:STDOUT: !61 = distinct !DISubprogram(name: "Op", linkageName: "_COp.X.Main:Copy.Main", scope: null, file: !18, line: 37, type: !27, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !70)
// CHECK:STDOUT: !62 = !DILocalVariable(arg: 1, scope: !61, type: !25)
// CHECK:STDOUT: !66 = !DILocation(line: 38, column: 20, scope: !61)
// CHECK:STDOUT: !67 = !DILocation(line: 38, column: 33, scope: !61)
// CHECK:STDOUT: !68 = !DILocation(line: 38, column: 14, scope: !61)
// CHECK:STDOUT: !69 = !DILocation(line: 38, column: 7, scope: !61)
// CHECK:STDOUT: !70 = !{!62}
// CHECK:STDOUT: !71 = distinct !DISubprogram(name: "F_X", linkageName: "_CF_X.Main", scope: null, file: !18, line: 43, type: !27, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !79)
// CHECK:STDOUT: !72 = !DILocalVariable(arg: 1, scope: !71, type: !25)
// CHECK:STDOUT: !77 = !DILocation(line: 44, column: 10, scope: !71)
// CHECK:STDOUT: !78 = !DILocation(line: 44, column: 3, scope: !71)
// CHECK:STDOUT: !79 = !{!72}
// CHECK:STDOUT: !80 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.b4c762864cd9a02b:core.Destroy.Core", scope: null, file: !18, line: 44, type: !29, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !83)
// CHECK:STDOUT: !81 = !DILocalVariable(arg: 1, scope: !80, type: !25)
// CHECK:STDOUT: !82 = !DILocation(line: 44, column: 10, scope: !80)
// CHECK:STDOUT: !83 = !{!81}
// CHECK:STDOUT: !84 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.b4c762864cd9a02b:core.Destroy.Core", scope: null, file: !18, line: 44, type: !29, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !87)
// CHECK:STDOUT: !85 = !DILocalVariable(arg: 1, scope: !84, type: !25)
// CHECK:STDOUT: !86 = !DILocation(line: 44, column: 10, scope: !84)
// CHECK:STDOUT: !87 = !{!85}
// CHECK:STDOUT: !88 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.8b0bf151036f045a:core.Destroy.Core", scope: null, file: !18, line: 44, type: !29, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !91)
// CHECK:STDOUT: !89 = !DILocalVariable(arg: 1, scope: !88, type: !25)
// CHECK:STDOUT: !90 = !DILocation(line: 44, column: 10, scope: !88)
// CHECK:STDOUT: !91 = !{!89}
// CHECK:STDOUT: !92 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.8b0bf151036f045a:core.Destroy.Core", scope: null, file: !18, line: 44, type: !29, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !95)
// CHECK:STDOUT: !93 = !DILocalVariable(arg: 1, scope: !92, type: !25)
// CHECK:STDOUT: !94 = !DILocation(line: 44, column: 10, scope: !92)
// CHECK:STDOUT: !95 = !{!93}
// CHECK:STDOUT: !96 = distinct !DISubprogram(name: "Op", linkageName: "_COp.61ea2aba74ab3bf1:Copy.Main", scope: null, file: !18, line: 48, type: !31, spFlags: DISPFlagDefinition, unit: !19)
// CHECK:STDOUT: !100 = !DILocation(line: 49, column: 5, scope: !96)
// CHECK:STDOUT: !101 = distinct !DISubprogram(name: "F_empty_tuple", linkageName: "_CF_empty_tuple.Main", scope: null, file: !18, line: 53, type: !31, spFlags: DISPFlagDefinition, unit: !19)
// CHECK:STDOUT: !106 = !DILocation(line: 54, column: 10, scope: !101)
// CHECK:STDOUT: !107 = !DILocation(line: 54, column: 3, scope: !101)
// CHECK:STDOUT: !108 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.61ea2aba74ab3bf1:core.Destroy.Core", scope: null, file: !18, line: 54, type: !29, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !111)
// CHECK:STDOUT: !109 = !DILocalVariable(arg: 1, scope: !108, type: !25)
// CHECK:STDOUT: !110 = !DILocation(line: 54, column: 10, scope: !108)
// CHECK:STDOUT: !111 = !{!109}
// CHECK:STDOUT: !112 = distinct !DISubprogram(name: "Op", linkageName: "_COp.e554695bf635d9c1:Copy.Main", scope: null, file: !18, line: 58, type: !27, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !119)
// CHECK:STDOUT: !113 = !DILocalVariable(arg: 1, scope: !112, type: !25)
// CHECK:STDOUT: !117 = !DILocation(line: 59, column: 12, scope: !112)
// CHECK:STDOUT: !118 = !DILocation(line: 59, column: 5, scope: !112)
// CHECK:STDOUT: !119 = !{!113}
// CHECK:STDOUT: !120 = distinct !DISubprogram(name: "F_two_tuple", linkageName: "_CF_two_tuple.Main", scope: null, file: !18, line: 63, type: !27, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !128)
// CHECK:STDOUT: !121 = !DILocalVariable(arg: 1, scope: !120, type: !25)
// CHECK:STDOUT: !126 = !DILocation(line: 64, column: 10, scope: !120)
// CHECK:STDOUT: !127 = !DILocation(line: 64, column: 3, scope: !120)
// CHECK:STDOUT: !128 = !{!121}
// CHECK:STDOUT: !129 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.e554695bf635d9c1:core.Destroy.Core", scope: null, file: !18, line: 64, type: !29, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !132)
// CHECK:STDOUT: !130 = !DILocalVariable(arg: 1, scope: !129, type: !25)
// CHECK:STDOUT: !131 = !DILocation(line: 64, column: 10, scope: !129)
// CHECK:STDOUT: !132 = !{!130}
// CHECK:STDOUT: !133 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.e554695bf635d9c1:core.Destroy.Core", scope: null, file: !18, line: 64, type: !29, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !136)
// CHECK:STDOUT: !134 = !DILocalVariable(arg: 1, scope: !133, type: !25)
// CHECK:STDOUT: !135 = !DILocation(line: 64, column: 10, scope: !133)
// CHECK:STDOUT: !136 = !{!134}
// CHECK:STDOUT: !137 = distinct !DISubprogram(name: "Op", linkageName: "_COp.ae1e3622d7875d32:Copy.Main", scope: null, file: !18, line: 68, type: !27, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !146)
// CHECK:STDOUT: !138 = !DILocalVariable(arg: 1, scope: !137, type: !25)
// CHECK:STDOUT: !142 = !DILocation(line: 69, column: 13, scope: !137)
// CHECK:STDOUT: !143 = !DILocation(line: 69, column: 12, scope: !137)
// CHECK:STDOUT: !144 = !DILocation(line: 69, column: 33, scope: !137)
// CHECK:STDOUT: !145 = !DILocation(line: 69, column: 5, scope: !137)
// CHECK:STDOUT: !146 = !{!138}
// CHECK:STDOUT: !147 = distinct !DISubprogram(name: "F_nested_tuple", linkageName: "_CF_nested_tuple.Main", scope: null, file: !18, line: 73, type: !27, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !155)
// CHECK:STDOUT: !148 = !DILocalVariable(arg: 1, scope: !147, type: !25)
// CHECK:STDOUT: !153 = !DILocation(line: 74, column: 10, scope: !147)
// CHECK:STDOUT: !154 = !DILocation(line: 74, column: 3, scope: !147)
// CHECK:STDOUT: !155 = !{!148}
// CHECK:STDOUT: !156 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.ae1e3622d7875d32:core.Destroy.Core", scope: null, file: !18, line: 74, type: !29, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !159)
// CHECK:STDOUT: !157 = !DILocalVariable(arg: 1, scope: !156, type: !25)
// CHECK:STDOUT: !158 = !DILocation(line: 74, column: 10, scope: !156)
// CHECK:STDOUT: !159 = !{!157}
// CHECK:STDOUT: !160 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.ae1e3622d7875d32:core.Destroy.Core", scope: null, file: !18, line: 74, type: !29, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !163)
// CHECK:STDOUT: !161 = !DILocalVariable(arg: 1, scope: !160, type: !25)
// CHECK:STDOUT: !162 = !DILocation(line: 74, column: 10, scope: !160)
// CHECK:STDOUT: !163 = !{!161}
// CHECK:STDOUT: !164 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.804fd121057d1d93", scope: null, file: !18, line: 17, type: !22, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !174)
// CHECK:STDOUT: !165 = !DILocalVariable(arg: 1, scope: !164, type: !20)
// CHECK:STDOUT: !169 = !DILocation(line: 18, column: 3, scope: !164)
// CHECK:STDOUT: !170 = !DILocation(line: 18, column: 14, scope: !164)
// CHECK:STDOUT: !172 = !DILocation(line: 19, column: 10, scope: !164)
// CHECK:STDOUT: !173 = !DILocation(line: 19, column: 3, scope: !164)
// CHECK:STDOUT: !174 = !{!165}
// CHECK:STDOUT: !175 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.9d368fc2e01d20c2", scope: null, file: !18, line: 17, type: !27, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !185)
// CHECK:STDOUT: !176 = !DILocalVariable(arg: 1, scope: !175, type: !25)
// CHECK:STDOUT: !180 = !DILocation(line: 18, column: 3, scope: !175)
// CHECK:STDOUT: !181 = !DILocation(line: 18, column: 14, scope: !175)
// CHECK:STDOUT: !183 = !DILocation(line: 19, column: 10, scope: !175)
// CHECK:STDOUT: !184 = !DILocation(line: 19, column: 3, scope: !175)
// CHECK:STDOUT: !185 = !{!176}
// CHECK:STDOUT: !186 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.a5d0440dfa027ffe", scope: null, file: !18, line: 17, type: !31, spFlags: DISPFlagDefinition, unit: !19)
// CHECK:STDOUT: !190 = !DILocation(line: 18, column: 3, scope: !186)
// CHECK:STDOUT: !191 = !DILocation(line: 18, column: 14, scope: !186)
// CHECK:STDOUT: !193 = !DILocation(line: 19, column: 10, scope: !186)
// CHECK:STDOUT: !194 = !DILocation(line: 19, column: 3, scope: !186)
// CHECK:STDOUT: !195 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.25802053f9465fe8", scope: null, file: !18, line: 17, type: !27, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !205)
// CHECK:STDOUT: !196 = !DILocalVariable(arg: 1, scope: !195, type: !25)
// CHECK:STDOUT: !200 = !DILocation(line: 18, column: 3, scope: !195)
// CHECK:STDOUT: !201 = !DILocation(line: 18, column: 14, scope: !195)
// CHECK:STDOUT: !203 = !DILocation(line: 19, column: 10, scope: !195)
// CHECK:STDOUT: !204 = !DILocation(line: 19, column: 3, scope: !195)
// CHECK:STDOUT: !205 = !{!196}
// CHECK:STDOUT: !206 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.f37d80e656707351", scope: null, file: !18, line: 17, type: !27, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !216)
// CHECK:STDOUT: !207 = !DILocalVariable(arg: 1, scope: !206, type: !25)
// CHECK:STDOUT: !211 = !DILocation(line: 18, column: 3, scope: !206)
// CHECK:STDOUT: !212 = !DILocation(line: 18, column: 14, scope: !206)
// CHECK:STDOUT: !214 = !DILocation(line: 19, column: 10, scope: !206)
// CHECK:STDOUT: !215 = !DILocation(line: 19, column: 3, scope: !206)
// CHECK:STDOUT: !216 = !{!207}
// CHECK:STDOUT:
