// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/interface/modifiers.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/interface/modifiers.carbon

// --- default_interfaces.carbon

library "[[@TEST_NAME]]";

fn Hello() {}

interface DefaultFn {
  default fn F() { Hello(); }
}

fn World[T: type](unused t: T) {}

interface DefaultMethod {
  default fn M(self) { World(self); }
}

fn CallFn(generic d: DefaultFn) {
  d.F();
}

fn CallMethod[T: DefaultMethod](t: T) {
  t.M();
}

// --- final_interfaces.carbon

library "[[@TEST_NAME]]";

interface FinalFn {
  final fn F() {}
}

interface FinalMethod {
  final fn M(unused self) {}
}

fn CallFn(generic f: FinalFn) {
  f.F();
}

fn CallMethod[T: FinalMethod](t: T) {
  t.M();
}

// --- default_function.carbon

library "[[@TEST_NAME]]";
import library "default_interfaces";

class C {
  impl as DefaultFn {}
}

fn F() {
  CallFn(C);
}

// --- default_method.carbon

library "[[@TEST_NAME]]";
import library "default_interfaces";

class C {
  impl as DefaultMethod {}
}

fn F() {
  let c: C = {};
  CallMethod(c);
}

// --- default_function_overridden.carbon

library "[[@TEST_NAME]]";
import library "default_interfaces";

class C {
  fn Clone(unused other: C) { }

  impl as DefaultFn {
    fn F() { C.Clone({}); }
  }
}

fn F() {
  CallFn(C);
}

// --- default_method_overridden.carbon

library "[[@TEST_NAME]]";
import library "default_interfaces";

class C {
  fn Copy(unused self) { }

  impl as DefaultMethod {
    fn M(self) { self.Copy(); }
  }
}

fn F() {
  let c: C = {};
  CallMethod(c);
}

// --- final_function.carbon

library "[[@TEST_NAME]]";
import library "final_interfaces";

class C {
  impl as FinalFn {}
}

fn F() {
  CallFn(C);
}

// --- final_method.carbon

library "[[@TEST_NAME]]";
import library "final_interfaces";

class C {
  impl as FinalMethod {}
}

fn F() {
  let c: C = {};
  CallMethod(c);
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'default_interfaces.carbon'
// CHECK:STDOUT: source_filename = "default_interfaces.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CHello.Main() #0 !dbg !18 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!15}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !14 = !DIFile(filename: "default_interfaces.carbon", directory: "")
// CHECK:STDOUT: !15 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !14, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !16 = !{null}
// CHECK:STDOUT: !17 = !DISubroutineType(types: !16)
// CHECK:STDOUT: !18 = distinct !DISubprogram(name: "Hello", linkageName: "_CHello.Main", scope: null, file: !14, line: 4, type: !17, spFlags: DISPFlagDefinition, unit: !15)
// CHECK:STDOUT: !19 = !DILocation(line: 4, column: 1, scope: !18)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'final_interfaces.carbon'
// CHECK:STDOUT: source_filename = "final_interfaces.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!21}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIFile(filename: "final_interfaces.carbon", directory: "")
// CHECK:STDOUT: !21 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !20, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'default_function.carbon'
// CHECK:STDOUT: source_filename = "default_function.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !24 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CCallFn.Main.794e85173e977a8b(), !dbg !25
// CHECK:STDOUT:   ret void, !dbg !26
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CCallFn.Main.794e85173e977a8b() #0 !dbg !27 {
// CHECK:STDOUT:   call void @_CF.DefaultFn.Main.794e85173e977a8b(), !dbg !28
// CHECK:STDOUT:   ret void, !dbg !29
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CHello.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.DefaultFn.Main.794e85173e977a8b() #0 !dbg !30 {
// CHECK:STDOUT:   call void @_CHello.Main(), !dbg !31
// CHECK:STDOUT:   ret void, !dbg !32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!23}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !14 = !DIFile(filename: "default_interfaces.carbon", directory: "")
// CHECK:STDOUT: !16 = !{null}
// CHECK:STDOUT: !17 = !DISubroutineType(types: !16)
// CHECK:STDOUT: !22 = !DIFile(filename: "default_function.carbon", directory: "")
// CHECK:STDOUT: !23 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !22, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !24 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !22, line: 9, type: !17, spFlags: DISPFlagDefinition, unit: !23)
// CHECK:STDOUT: !25 = !DILocation(line: 10, column: 3, scope: !24)
// CHECK:STDOUT: !26 = !DILocation(line: 9, column: 1, scope: !24)
// CHECK:STDOUT: !27 = distinct !DISubprogram(name: "CallFn", linkageName: "_CCallFn.Main.794e85173e977a8b", scope: null, file: !14, line: 16, type: !17, spFlags: DISPFlagDefinition, unit: !23)
// CHECK:STDOUT: !28 = !DILocation(line: 17, column: 3, scope: !27)
// CHECK:STDOUT: !29 = !DILocation(line: 16, column: 1, scope: !27)
// CHECK:STDOUT: !30 = distinct !DISubprogram(name: "F", linkageName: "_CF.DefaultFn.Main.794e85173e977a8b", scope: null, file: !14, line: 7, type: !17, spFlags: DISPFlagDefinition, unit: !23)
// CHECK:STDOUT: !31 = !DILocation(line: 7, column: 20, scope: !30)
// CHECK:STDOUT: !32 = !DILocation(line: 7, column: 3, scope: !30)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'default_method.carbon'
// CHECK:STDOUT: source_filename = "default_method.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @C.val = internal constant {} zeroinitializer
// CHECK:STDOUT: @C.val.loc10_15.6 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !38 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc10_15.2.temp = alloca {}, align 1, !dbg !39
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc10_15.2.temp), !dbg !39
// CHECK:STDOUT:   call void @_CCallMethod.Main.805a971dc5455081(ptr @C.val.loc10_15.6), !dbg !43
// CHECK:STDOUT:   call void @"_CSelfDestruct.08d9e38b928ba575:core.Destroy.Core"(ptr @C.val), !dbg !39
// CHECK:STDOUT:   ret void, !dbg !44
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.cb8adced19392b99:core.Destroy.Core"(ptr %self) #0 !dbg !45 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !47
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.08d9e38b928ba575:core.Destroy.Core"(ptr %self) #0 !dbg !49 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !51
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CCallMethod.Main.805a971dc5455081(ptr %t) #0 !dbg !53 {
// CHECK:STDOUT:   call void @_CM.DefaultMethod.Main.805a971dc5455081(ptr %t), !dbg !56
// CHECK:STDOUT:   ret void, !dbg !57
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CHello.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CM.DefaultMethod.Main.805a971dc5455081(ptr %self) #0 !dbg !59 {
// CHECK:STDOUT:   call void @_CWorld.Main.9522ff908e97e267(ptr %self), !dbg !63
// CHECK:STDOUT:   ret void, !dbg !64
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CWorld.Main.9522ff908e97e267(ptr %t) #0 !dbg !66 {
// CHECK:STDOUT:   ret void, !dbg !69
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!34}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !14 = !DIFile(filename: "default_interfaces.carbon", directory: "")
// CHECK:STDOUT: !16 = !{null}
// CHECK:STDOUT: !17 = !DISubroutineType(types: !16)
// CHECK:STDOUT: !33 = !DIFile(filename: "default_method.carbon", directory: "")
// CHECK:STDOUT: !34 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !33, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !35 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !36 = !{null, !35}
// CHECK:STDOUT: !37 = !DISubroutineType(types: !36)
// CHECK:STDOUT: !38 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !33, line: 9, type: !17, spFlags: DISPFlagDefinition, unit: !34)
// CHECK:STDOUT: !39 = !DILocation(line: 10, column: 14, scope: !38)
// CHECK:STDOUT: !43 = !DILocation(line: 11, column: 3, scope: !38)
// CHECK:STDOUT: !44 = !DILocation(line: 9, column: 1, scope: !38)
// CHECK:STDOUT: !45 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.cb8adced19392b99:core.Destroy.Core", scope: null, file: !33, line: 10, type: !37, spFlags: DISPFlagDefinition, unit: !34, retainedNodes: !48)
// CHECK:STDOUT: !46 = !DILocalVariable(arg: 1, scope: !45, type: !35)
// CHECK:STDOUT: !47 = !DILocation(line: 10, column: 14, scope: !45)
// CHECK:STDOUT: !48 = !{!46}
// CHECK:STDOUT: !49 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.08d9e38b928ba575:core.Destroy.Core", scope: null, file: !33, line: 10, type: !37, spFlags: DISPFlagDefinition, unit: !34, retainedNodes: !52)
// CHECK:STDOUT: !50 = !DILocalVariable(arg: 1, scope: !49, type: !35)
// CHECK:STDOUT: !51 = !DILocation(line: 10, column: 14, scope: !49)
// CHECK:STDOUT: !52 = !{!50}
// CHECK:STDOUT: !53 = distinct !DISubprogram(name: "CallMethod", linkageName: "_CCallMethod.Main.805a971dc5455081", scope: null, file: !14, line: 20, type: !37, spFlags: DISPFlagDefinition, unit: !34, retainedNodes: !58)
// CHECK:STDOUT: !54 = !DILocalVariable(arg: 1, scope: !53, type: !35)
// CHECK:STDOUT: !56 = !DILocation(line: 21, column: 3, scope: !53)
// CHECK:STDOUT: !57 = !DILocation(line: 20, column: 1, scope: !53)
// CHECK:STDOUT: !58 = !{!54}
// CHECK:STDOUT: !59 = distinct !DISubprogram(name: "M", linkageName: "_CM.DefaultMethod.Main.805a971dc5455081", scope: null, file: !14, line: 13, type: !37, spFlags: DISPFlagDefinition, unit: !34, retainedNodes: !65)
// CHECK:STDOUT: !60 = !DILocalVariable(arg: 1, scope: !59, type: !35)
// CHECK:STDOUT: !63 = !DILocation(line: 13, column: 24, scope: !59)
// CHECK:STDOUT: !64 = !DILocation(line: 13, column: 3, scope: !59)
// CHECK:STDOUT: !65 = !{!60}
// CHECK:STDOUT: !66 = distinct !DISubprogram(name: "World", linkageName: "_CWorld.Main.9522ff908e97e267", scope: null, file: !14, line: 10, type: !37, spFlags: DISPFlagDefinition, unit: !34, retainedNodes: !70)
// CHECK:STDOUT: !67 = !DILocalVariable(arg: 1, scope: !66, type: !35)
// CHECK:STDOUT: !69 = !DILocation(line: 10, column: 1, scope: !66)
// CHECK:STDOUT: !70 = !{!67}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'default_function_overridden.carbon'
// CHECK:STDOUT: source_filename = "default_function_overridden.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @C.val = internal constant {} zeroinitializer
// CHECK:STDOUT: @C.val.loc9_23.6 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CClone.C.Main(ptr %other) #0 !dbg !73 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !76
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_CF.C.Main:DefaultFn.Main"() #0 !dbg !78 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc9_23.2.temp = alloca {}, align 1, !dbg !79
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc9_23.2.temp), !dbg !79
// CHECK:STDOUT:   call void @_CClone.C.Main(ptr @C.val.loc9_23.6), !dbg !80
// CHECK:STDOUT:   call void @"_CSelfDestruct.08d9e38b928ba575:core.Destroy.Core"(ptr @C.val), !dbg !79
// CHECK:STDOUT:   ret void, !dbg !81
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.cb8adced19392b99:core.Destroy.Core"(ptr %self) #0 !dbg !82 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !84
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.08d9e38b928ba575:core.Destroy.Core"(ptr %self) #0 !dbg !86 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !88
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !90 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CCallFn.Main.794e85173e977a8b(), !dbg !91
// CHECK:STDOUT:   ret void, !dbg !92
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CCallFn.Main.794e85173e977a8b() #0 !dbg !93 {
// CHECK:STDOUT:   call void @"_CF.C.Main:DefaultFn.Main"(), !dbg !94
// CHECK:STDOUT:   ret void, !dbg !95
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CHello.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!72}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !14 = !DIFile(filename: "default_interfaces.carbon", directory: "")
// CHECK:STDOUT: !16 = !{null}
// CHECK:STDOUT: !17 = !DISubroutineType(types: !16)
// CHECK:STDOUT: !35 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !36 = !{null, !35}
// CHECK:STDOUT: !37 = !DISubroutineType(types: !36)
// CHECK:STDOUT: !71 = !DIFile(filename: "default_function_overridden.carbon", directory: "")
// CHECK:STDOUT: !72 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !71, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !73 = distinct !DISubprogram(name: "Clone", linkageName: "_CClone.C.Main", scope: null, file: !71, line: 6, type: !37, spFlags: DISPFlagDefinition, unit: !72, retainedNodes: !77)
// CHECK:STDOUT: !74 = !DILocalVariable(arg: 1, scope: !73, type: !35)
// CHECK:STDOUT: !76 = !DILocation(line: 6, column: 3, scope: !73)
// CHECK:STDOUT: !77 = !{!74}
// CHECK:STDOUT: !78 = distinct !DISubprogram(name: "F", linkageName: "_CF.C.Main:DefaultFn.Main", scope: null, file: !71, line: 9, type: !17, spFlags: DISPFlagDefinition, unit: !72)
// CHECK:STDOUT: !79 = !DILocation(line: 9, column: 22, scope: !78)
// CHECK:STDOUT: !80 = !DILocation(line: 9, column: 14, scope: !78)
// CHECK:STDOUT: !81 = !DILocation(line: 9, column: 5, scope: !78)
// CHECK:STDOUT: !82 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.cb8adced19392b99:core.Destroy.Core", scope: null, file: !71, line: 9, type: !37, spFlags: DISPFlagDefinition, unit: !72, retainedNodes: !85)
// CHECK:STDOUT: !83 = !DILocalVariable(arg: 1, scope: !82, type: !35)
// CHECK:STDOUT: !84 = !DILocation(line: 9, column: 22, scope: !82)
// CHECK:STDOUT: !85 = !{!83}
// CHECK:STDOUT: !86 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.08d9e38b928ba575:core.Destroy.Core", scope: null, file: !71, line: 9, type: !37, spFlags: DISPFlagDefinition, unit: !72, retainedNodes: !89)
// CHECK:STDOUT: !87 = !DILocalVariable(arg: 1, scope: !86, type: !35)
// CHECK:STDOUT: !88 = !DILocation(line: 9, column: 22, scope: !86)
// CHECK:STDOUT: !89 = !{!87}
// CHECK:STDOUT: !90 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !71, line: 13, type: !17, spFlags: DISPFlagDefinition, unit: !72)
// CHECK:STDOUT: !91 = !DILocation(line: 14, column: 3, scope: !90)
// CHECK:STDOUT: !92 = !DILocation(line: 13, column: 1, scope: !90)
// CHECK:STDOUT: !93 = distinct !DISubprogram(name: "CallFn", linkageName: "_CCallFn.Main.794e85173e977a8b", scope: null, file: !14, line: 16, type: !17, spFlags: DISPFlagDefinition, unit: !72)
// CHECK:STDOUT: !94 = !DILocation(line: 17, column: 3, scope: !93)
// CHECK:STDOUT: !95 = !DILocation(line: 16, column: 1, scope: !93)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'default_method_overridden.carbon'
// CHECK:STDOUT: source_filename = "default_method_overridden.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @C.val = internal constant {} zeroinitializer
// CHECK:STDOUT: @C.val.loc14_15.6 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCopy.C.Main(ptr %self) #0 !dbg !98 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !101
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_CM.C.Main:DefaultMethod.Main"(ptr %self) #0 !dbg !103 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CCopy.C.Main(ptr %self), !dbg !106
// CHECK:STDOUT:   ret void, !dbg !107
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !109 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc14_15.2.temp = alloca {}, align 1, !dbg !110
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc14_15.2.temp), !dbg !110
// CHECK:STDOUT:   call void @_CCallMethod.Main.805a971dc5455081(ptr @C.val.loc14_15.6), !dbg !114
// CHECK:STDOUT:   call void @"_CSelfDestruct.08d9e38b928ba575:core.Destroy.Core"(ptr @C.val), !dbg !110
// CHECK:STDOUT:   ret void, !dbg !115
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.cb8adced19392b99:core.Destroy.Core"(ptr %self) #0 !dbg !116 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !118
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.08d9e38b928ba575:core.Destroy.Core"(ptr %self) #0 !dbg !120 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !122
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CCallMethod.Main.805a971dc5455081(ptr %t) #0 !dbg !124 {
// CHECK:STDOUT:   call void @"_CM.C.Main:DefaultMethod.Main"(ptr %t), !dbg !127
// CHECK:STDOUT:   ret void, !dbg !128
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CHello.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!97}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !14 = !DIFile(filename: "default_interfaces.carbon", directory: "")
// CHECK:STDOUT: !16 = !{null}
// CHECK:STDOUT: !17 = !DISubroutineType(types: !16)
// CHECK:STDOUT: !35 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !36 = !{null, !35}
// CHECK:STDOUT: !37 = !DISubroutineType(types: !36)
// CHECK:STDOUT: !96 = !DIFile(filename: "default_method_overridden.carbon", directory: "")
// CHECK:STDOUT: !97 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !96, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !98 = distinct !DISubprogram(name: "Copy", linkageName: "_CCopy.C.Main", scope: null, file: !96, line: 6, type: !37, spFlags: DISPFlagDefinition, unit: !97, retainedNodes: !102)
// CHECK:STDOUT: !99 = !DILocalVariable(arg: 1, scope: !98, type: !35)
// CHECK:STDOUT: !101 = !DILocation(line: 6, column: 3, scope: !98)
// CHECK:STDOUT: !102 = !{!99}
// CHECK:STDOUT: !103 = distinct !DISubprogram(name: "M", linkageName: "_CM.C.Main:DefaultMethod.Main", scope: null, file: !96, line: 9, type: !37, spFlags: DISPFlagDefinition, unit: !97, retainedNodes: !108)
// CHECK:STDOUT: !104 = !DILocalVariable(arg: 1, scope: !103, type: !35)
// CHECK:STDOUT: !106 = !DILocation(line: 9, column: 18, scope: !103)
// CHECK:STDOUT: !107 = !DILocation(line: 9, column: 5, scope: !103)
// CHECK:STDOUT: !108 = !{!104}
// CHECK:STDOUT: !109 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !96, line: 13, type: !17, spFlags: DISPFlagDefinition, unit: !97)
// CHECK:STDOUT: !110 = !DILocation(line: 14, column: 14, scope: !109)
// CHECK:STDOUT: !114 = !DILocation(line: 15, column: 3, scope: !109)
// CHECK:STDOUT: !115 = !DILocation(line: 13, column: 1, scope: !109)
// CHECK:STDOUT: !116 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.cb8adced19392b99:core.Destroy.Core", scope: null, file: !96, line: 14, type: !37, spFlags: DISPFlagDefinition, unit: !97, retainedNodes: !119)
// CHECK:STDOUT: !117 = !DILocalVariable(arg: 1, scope: !116, type: !35)
// CHECK:STDOUT: !118 = !DILocation(line: 14, column: 14, scope: !116)
// CHECK:STDOUT: !119 = !{!117}
// CHECK:STDOUT: !120 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.08d9e38b928ba575:core.Destroy.Core", scope: null, file: !96, line: 14, type: !37, spFlags: DISPFlagDefinition, unit: !97, retainedNodes: !123)
// CHECK:STDOUT: !121 = !DILocalVariable(arg: 1, scope: !120, type: !35)
// CHECK:STDOUT: !122 = !DILocation(line: 14, column: 14, scope: !120)
// CHECK:STDOUT: !123 = !{!121}
// CHECK:STDOUT: !124 = distinct !DISubprogram(name: "CallMethod", linkageName: "_CCallMethod.Main.805a971dc5455081", scope: null, file: !14, line: 20, type: !37, spFlags: DISPFlagDefinition, unit: !97, retainedNodes: !129)
// CHECK:STDOUT: !125 = !DILocalVariable(arg: 1, scope: !124, type: !35)
// CHECK:STDOUT: !127 = !DILocation(line: 21, column: 3, scope: !124)
// CHECK:STDOUT: !128 = !DILocation(line: 20, column: 1, scope: !124)
// CHECK:STDOUT: !129 = !{!125}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'final_function.carbon'
// CHECK:STDOUT: source_filename = "final_function.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !132 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CCallFn.Main.e342d6e8ff5d87ae(), !dbg !133
// CHECK:STDOUT:   ret void, !dbg !134
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CCallFn.Main.e342d6e8ff5d87ae() #0 !dbg !135 {
// CHECK:STDOUT:   call void @_CF.FinalFn.Main.e342d6e8ff5d87ae(), !dbg !136
// CHECK:STDOUT:   ret void, !dbg !137
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.FinalFn.Main.e342d6e8ff5d87ae() #0 !dbg !138 {
// CHECK:STDOUT:   ret void, !dbg !139
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!131}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !16 = !{null}
// CHECK:STDOUT: !17 = !DISubroutineType(types: !16)
// CHECK:STDOUT: !20 = !DIFile(filename: "final_interfaces.carbon", directory: "")
// CHECK:STDOUT: !130 = !DIFile(filename: "final_function.carbon", directory: "")
// CHECK:STDOUT: !131 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !130, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !132 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !130, line: 9, type: !17, spFlags: DISPFlagDefinition, unit: !131)
// CHECK:STDOUT: !133 = !DILocation(line: 10, column: 3, scope: !132)
// CHECK:STDOUT: !134 = !DILocation(line: 9, column: 1, scope: !132)
// CHECK:STDOUT: !135 = distinct !DISubprogram(name: "CallFn", linkageName: "_CCallFn.Main.e342d6e8ff5d87ae", scope: null, file: !20, line: 12, type: !17, spFlags: DISPFlagDefinition, unit: !131)
// CHECK:STDOUT: !136 = !DILocation(line: 13, column: 3, scope: !135)
// CHECK:STDOUT: !137 = !DILocation(line: 12, column: 1, scope: !135)
// CHECK:STDOUT: !138 = distinct !DISubprogram(name: "F", linkageName: "_CF.FinalFn.Main.e342d6e8ff5d87ae", scope: null, file: !20, line: 5, type: !17, spFlags: DISPFlagDefinition, unit: !131)
// CHECK:STDOUT: !139 = !DILocation(line: 5, column: 3, scope: !138)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'final_method.carbon'
// CHECK:STDOUT: source_filename = "final_method.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @C.val = internal constant {} zeroinitializer
// CHECK:STDOUT: @C.val.loc10_15.6 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !142 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc10_15.2.temp = alloca {}, align 1, !dbg !143
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc10_15.2.temp), !dbg !143
// CHECK:STDOUT:   call void @_CCallMethod.Main.96e0ef8a617a6362(ptr @C.val.loc10_15.6), !dbg !147
// CHECK:STDOUT:   call void @"_CSelfDestruct.08d9e38b928ba575:core.Destroy.Core"(ptr @C.val), !dbg !143
// CHECK:STDOUT:   ret void, !dbg !148
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.cb8adced19392b99:core.Destroy.Core"(ptr %self) #0 !dbg !149 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !151
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.08d9e38b928ba575:core.Destroy.Core"(ptr %self) #0 !dbg !153 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !155
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CCallMethod.Main.96e0ef8a617a6362(ptr %t) #0 !dbg !157 {
// CHECK:STDOUT:   call void @_CM.FinalMethod.Main.96e0ef8a617a6362(ptr %t), !dbg !160
// CHECK:STDOUT:   ret void, !dbg !161
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CM.FinalMethod.Main.96e0ef8a617a6362(ptr %self) #0 !dbg !163 {
// CHECK:STDOUT:   ret void, !dbg !166
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!141}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !16 = !{null}
// CHECK:STDOUT: !17 = !DISubroutineType(types: !16)
// CHECK:STDOUT: !20 = !DIFile(filename: "final_interfaces.carbon", directory: "")
// CHECK:STDOUT: !35 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !36 = !{null, !35}
// CHECK:STDOUT: !37 = !DISubroutineType(types: !36)
// CHECK:STDOUT: !140 = !DIFile(filename: "final_method.carbon", directory: "")
// CHECK:STDOUT: !141 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !140, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !142 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !140, line: 9, type: !17, spFlags: DISPFlagDefinition, unit: !141)
// CHECK:STDOUT: !143 = !DILocation(line: 10, column: 14, scope: !142)
// CHECK:STDOUT: !147 = !DILocation(line: 11, column: 3, scope: !142)
// CHECK:STDOUT: !148 = !DILocation(line: 9, column: 1, scope: !142)
// CHECK:STDOUT: !149 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.cb8adced19392b99:core.Destroy.Core", scope: null, file: !140, line: 10, type: !37, spFlags: DISPFlagDefinition, unit: !141, retainedNodes: !152)
// CHECK:STDOUT: !150 = !DILocalVariable(arg: 1, scope: !149, type: !35)
// CHECK:STDOUT: !151 = !DILocation(line: 10, column: 14, scope: !149)
// CHECK:STDOUT: !152 = !{!150}
// CHECK:STDOUT: !153 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.08d9e38b928ba575:core.Destroy.Core", scope: null, file: !140, line: 10, type: !37, spFlags: DISPFlagDefinition, unit: !141, retainedNodes: !156)
// CHECK:STDOUT: !154 = !DILocalVariable(arg: 1, scope: !153, type: !35)
// CHECK:STDOUT: !155 = !DILocation(line: 10, column: 14, scope: !153)
// CHECK:STDOUT: !156 = !{!154}
// CHECK:STDOUT: !157 = distinct !DISubprogram(name: "CallMethod", linkageName: "_CCallMethod.Main.96e0ef8a617a6362", scope: null, file: !20, line: 16, type: !37, spFlags: DISPFlagDefinition, unit: !141, retainedNodes: !162)
// CHECK:STDOUT: !158 = !DILocalVariable(arg: 1, scope: !157, type: !35)
// CHECK:STDOUT: !160 = !DILocation(line: 17, column: 3, scope: !157)
// CHECK:STDOUT: !161 = !DILocation(line: 16, column: 1, scope: !157)
// CHECK:STDOUT: !162 = !{!158}
// CHECK:STDOUT: !163 = distinct !DISubprogram(name: "M", linkageName: "_CM.FinalMethod.Main.96e0ef8a617a6362", scope: null, file: !20, line: 9, type: !37, spFlags: DISPFlagDefinition, unit: !141, retainedNodes: !167)
// CHECK:STDOUT: !164 = !DILocalVariable(arg: 1, scope: !163, type: !35)
// CHECK:STDOUT: !166 = !DILocation(line: 9, column: 3, scope: !163)
// CHECK:STDOUT: !167 = !{!164}
// CHECK:STDOUT:
