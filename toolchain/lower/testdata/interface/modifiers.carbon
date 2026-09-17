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
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.2eb69dd0c58be63c(ptr @C.val), !dbg !39
// CHECK:STDOUT:   ret void, !dbg !44
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.08d9e38b928ba575:core.Destroy.Core"(ptr %self) #0 !dbg !45 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !47
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CCallMethod.Main.805a971dc5455081(ptr %t) #0 !dbg !49 {
// CHECK:STDOUT:   call void @_CM.DefaultMethod.Main.805a971dc5455081(ptr %t), !dbg !52
// CHECK:STDOUT:   ret void, !dbg !53
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.2eb69dd0c58be63c(ptr %self) #0 !dbg !55 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.08d9e38b928ba575:core.Destroy.Core"(ptr %self), !dbg !59
// CHECK:STDOUT:   ret void, !dbg !60
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CHello.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CM.DefaultMethod.Main.805a971dc5455081(ptr %self) #0 !dbg !62 {
// CHECK:STDOUT:   call void @_CWorld.Main.9522ff908e97e267(ptr %self), !dbg !66
// CHECK:STDOUT:   ret void, !dbg !67
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CWorld.Main.9522ff908e97e267(ptr %t) #0 !dbg !69 {
// CHECK:STDOUT:   ret void, !dbg !72
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
// CHECK:STDOUT: !10 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
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
// CHECK:STDOUT: !45 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.08d9e38b928ba575:core.Destroy.Core", scope: null, file: !33, line: 10, type: !37, spFlags: DISPFlagDefinition, unit: !34, retainedNodes: !48)
// CHECK:STDOUT: !46 = !DILocalVariable(arg: 1, scope: !45, type: !35)
// CHECK:STDOUT: !47 = !DILocation(line: 10, column: 14, scope: !45)
// CHECK:STDOUT: !48 = !{!46}
// CHECK:STDOUT: !49 = distinct !DISubprogram(name: "CallMethod", linkageName: "_CCallMethod.Main.805a971dc5455081", scope: null, file: !14, line: 20, type: !37, spFlags: DISPFlagDefinition, unit: !34, retainedNodes: !54)
// CHECK:STDOUT: !50 = !DILocalVariable(arg: 1, scope: !49, type: !35)
// CHECK:STDOUT: !52 = !DILocation(line: 21, column: 3, scope: !49)
// CHECK:STDOUT: !53 = !DILocation(line: 20, column: 1, scope: !49)
// CHECK:STDOUT: !54 = !{!50}
// CHECK:STDOUT: !55 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.2eb69dd0c58be63c", scope: null, file: !10, line: 24, type: !37, spFlags: DISPFlagDefinition, unit: !34, retainedNodes: !61)
// CHECK:STDOUT: !56 = !DILocalVariable(arg: 1, scope: !55, type: !35)
// CHECK:STDOUT: !59 = !DILocation(line: 26, column: 5, scope: !55)
// CHECK:STDOUT: !60 = !DILocation(line: 24, column: 3, scope: !55)
// CHECK:STDOUT: !61 = !{!56}
// CHECK:STDOUT: !62 = distinct !DISubprogram(name: "M", linkageName: "_CM.DefaultMethod.Main.805a971dc5455081", scope: null, file: !14, line: 13, type: !37, spFlags: DISPFlagDefinition, unit: !34, retainedNodes: !68)
// CHECK:STDOUT: !63 = !DILocalVariable(arg: 1, scope: !62, type: !35)
// CHECK:STDOUT: !66 = !DILocation(line: 13, column: 24, scope: !62)
// CHECK:STDOUT: !67 = !DILocation(line: 13, column: 3, scope: !62)
// CHECK:STDOUT: !68 = !{!63}
// CHECK:STDOUT: !69 = distinct !DISubprogram(name: "World", linkageName: "_CWorld.Main.9522ff908e97e267", scope: null, file: !14, line: 10, type: !37, spFlags: DISPFlagDefinition, unit: !34, retainedNodes: !73)
// CHECK:STDOUT: !70 = !DILocalVariable(arg: 1, scope: !69, type: !35)
// CHECK:STDOUT: !72 = !DILocation(line: 10, column: 1, scope: !69)
// CHECK:STDOUT: !73 = !{!70}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'default_function_overridden.carbon'
// CHECK:STDOUT: source_filename = "default_function_overridden.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @C.val = internal constant {} zeroinitializer
// CHECK:STDOUT: @C.val.loc9_23.6 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CClone.C.Main(ptr %other) #0 !dbg !76 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !79
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_CF.C.Main:DefaultFn.Main"() #0 !dbg !81 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc9_23.2.temp = alloca {}, align 1, !dbg !82
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc9_23.2.temp), !dbg !82
// CHECK:STDOUT:   call void @_CClone.C.Main(ptr @C.val.loc9_23.6), !dbg !83
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.2eb69dd0c58be63c(ptr @C.val), !dbg !82
// CHECK:STDOUT:   ret void, !dbg !84
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.08d9e38b928ba575:core.Destroy.Core"(ptr %self) #0 !dbg !85 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !87
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !89 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CCallFn.Main.794e85173e977a8b(), !dbg !90
// CHECK:STDOUT:   ret void, !dbg !91
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.2eb69dd0c58be63c(ptr %self) #0 !dbg !92 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.08d9e38b928ba575:core.Destroy.Core"(ptr %self), !dbg !96
// CHECK:STDOUT:   ret void, !dbg !97
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CCallFn.Main.794e85173e977a8b() #0 !dbg !99 {
// CHECK:STDOUT:   call void @"_CF.C.Main:DefaultFn.Main"(), !dbg !100
// CHECK:STDOUT:   ret void, !dbg !101
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CHello.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!75}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !10 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !14 = !DIFile(filename: "default_interfaces.carbon", directory: "")
// CHECK:STDOUT: !16 = !{null}
// CHECK:STDOUT: !17 = !DISubroutineType(types: !16)
// CHECK:STDOUT: !35 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !36 = !{null, !35}
// CHECK:STDOUT: !37 = !DISubroutineType(types: !36)
// CHECK:STDOUT: !74 = !DIFile(filename: "default_function_overridden.carbon", directory: "")
// CHECK:STDOUT: !75 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !74, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !76 = distinct !DISubprogram(name: "Clone", linkageName: "_CClone.C.Main", scope: null, file: !74, line: 6, type: !37, spFlags: DISPFlagDefinition, unit: !75, retainedNodes: !80)
// CHECK:STDOUT: !77 = !DILocalVariable(arg: 1, scope: !76, type: !35)
// CHECK:STDOUT: !79 = !DILocation(line: 6, column: 3, scope: !76)
// CHECK:STDOUT: !80 = !{!77}
// CHECK:STDOUT: !81 = distinct !DISubprogram(name: "F", linkageName: "_CF.C.Main:DefaultFn.Main", scope: null, file: !74, line: 9, type: !17, spFlags: DISPFlagDefinition, unit: !75)
// CHECK:STDOUT: !82 = !DILocation(line: 9, column: 22, scope: !81)
// CHECK:STDOUT: !83 = !DILocation(line: 9, column: 14, scope: !81)
// CHECK:STDOUT: !84 = !DILocation(line: 9, column: 5, scope: !81)
// CHECK:STDOUT: !85 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.08d9e38b928ba575:core.Destroy.Core", scope: null, file: !74, line: 9, type: !37, spFlags: DISPFlagDefinition, unit: !75, retainedNodes: !88)
// CHECK:STDOUT: !86 = !DILocalVariable(arg: 1, scope: !85, type: !35)
// CHECK:STDOUT: !87 = !DILocation(line: 9, column: 22, scope: !85)
// CHECK:STDOUT: !88 = !{!86}
// CHECK:STDOUT: !89 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !74, line: 13, type: !17, spFlags: DISPFlagDefinition, unit: !75)
// CHECK:STDOUT: !90 = !DILocation(line: 14, column: 3, scope: !89)
// CHECK:STDOUT: !91 = !DILocation(line: 13, column: 1, scope: !89)
// CHECK:STDOUT: !92 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.2eb69dd0c58be63c", scope: null, file: !10, line: 24, type: !37, spFlags: DISPFlagDefinition, unit: !75, retainedNodes: !98)
// CHECK:STDOUT: !93 = !DILocalVariable(arg: 1, scope: !92, type: !35)
// CHECK:STDOUT: !96 = !DILocation(line: 26, column: 5, scope: !92)
// CHECK:STDOUT: !97 = !DILocation(line: 24, column: 3, scope: !92)
// CHECK:STDOUT: !98 = !{!93}
// CHECK:STDOUT: !99 = distinct !DISubprogram(name: "CallFn", linkageName: "_CCallFn.Main.794e85173e977a8b", scope: null, file: !14, line: 16, type: !17, spFlags: DISPFlagDefinition, unit: !75)
// CHECK:STDOUT: !100 = !DILocation(line: 17, column: 3, scope: !99)
// CHECK:STDOUT: !101 = !DILocation(line: 16, column: 1, scope: !99)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'default_method_overridden.carbon'
// CHECK:STDOUT: source_filename = "default_method_overridden.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @C.val = internal constant {} zeroinitializer
// CHECK:STDOUT: @C.val.loc14_15.6 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCopy.C.Main(ptr %self) #0 !dbg !104 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !107
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_CM.C.Main:DefaultMethod.Main"(ptr %self) #0 !dbg !109 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CCopy.C.Main(ptr %self), !dbg !112
// CHECK:STDOUT:   ret void, !dbg !113
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !115 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc14_15.2.temp = alloca {}, align 1, !dbg !116
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc14_15.2.temp), !dbg !116
// CHECK:STDOUT:   call void @_CCallMethod.Main.805a971dc5455081(ptr @C.val.loc14_15.6), !dbg !120
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.2eb69dd0c58be63c(ptr @C.val), !dbg !116
// CHECK:STDOUT:   ret void, !dbg !121
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.08d9e38b928ba575:core.Destroy.Core"(ptr %self) #0 !dbg !122 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !124
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CCallMethod.Main.805a971dc5455081(ptr %t) #0 !dbg !126 {
// CHECK:STDOUT:   call void @"_CM.C.Main:DefaultMethod.Main"(ptr %t), !dbg !129
// CHECK:STDOUT:   ret void, !dbg !130
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.2eb69dd0c58be63c(ptr %self) #0 !dbg !132 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.08d9e38b928ba575:core.Destroy.Core"(ptr %self), !dbg !136
// CHECK:STDOUT:   ret void, !dbg !137
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CHello.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!103}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !10 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !14 = !DIFile(filename: "default_interfaces.carbon", directory: "")
// CHECK:STDOUT: !16 = !{null}
// CHECK:STDOUT: !17 = !DISubroutineType(types: !16)
// CHECK:STDOUT: !35 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !36 = !{null, !35}
// CHECK:STDOUT: !37 = !DISubroutineType(types: !36)
// CHECK:STDOUT: !102 = !DIFile(filename: "default_method_overridden.carbon", directory: "")
// CHECK:STDOUT: !103 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !102, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !104 = distinct !DISubprogram(name: "Copy", linkageName: "_CCopy.C.Main", scope: null, file: !102, line: 6, type: !37, spFlags: DISPFlagDefinition, unit: !103, retainedNodes: !108)
// CHECK:STDOUT: !105 = !DILocalVariable(arg: 1, scope: !104, type: !35)
// CHECK:STDOUT: !107 = !DILocation(line: 6, column: 3, scope: !104)
// CHECK:STDOUT: !108 = !{!105}
// CHECK:STDOUT: !109 = distinct !DISubprogram(name: "M", linkageName: "_CM.C.Main:DefaultMethod.Main", scope: null, file: !102, line: 9, type: !37, spFlags: DISPFlagDefinition, unit: !103, retainedNodes: !114)
// CHECK:STDOUT: !110 = !DILocalVariable(arg: 1, scope: !109, type: !35)
// CHECK:STDOUT: !112 = !DILocation(line: 9, column: 18, scope: !109)
// CHECK:STDOUT: !113 = !DILocation(line: 9, column: 5, scope: !109)
// CHECK:STDOUT: !114 = !{!110}
// CHECK:STDOUT: !115 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !102, line: 13, type: !17, spFlags: DISPFlagDefinition, unit: !103)
// CHECK:STDOUT: !116 = !DILocation(line: 14, column: 14, scope: !115)
// CHECK:STDOUT: !120 = !DILocation(line: 15, column: 3, scope: !115)
// CHECK:STDOUT: !121 = !DILocation(line: 13, column: 1, scope: !115)
// CHECK:STDOUT: !122 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.08d9e38b928ba575:core.Destroy.Core", scope: null, file: !102, line: 14, type: !37, spFlags: DISPFlagDefinition, unit: !103, retainedNodes: !125)
// CHECK:STDOUT: !123 = !DILocalVariable(arg: 1, scope: !122, type: !35)
// CHECK:STDOUT: !124 = !DILocation(line: 14, column: 14, scope: !122)
// CHECK:STDOUT: !125 = !{!123}
// CHECK:STDOUT: !126 = distinct !DISubprogram(name: "CallMethod", linkageName: "_CCallMethod.Main.805a971dc5455081", scope: null, file: !14, line: 20, type: !37, spFlags: DISPFlagDefinition, unit: !103, retainedNodes: !131)
// CHECK:STDOUT: !127 = !DILocalVariable(arg: 1, scope: !126, type: !35)
// CHECK:STDOUT: !129 = !DILocation(line: 21, column: 3, scope: !126)
// CHECK:STDOUT: !130 = !DILocation(line: 20, column: 1, scope: !126)
// CHECK:STDOUT: !131 = !{!127}
// CHECK:STDOUT: !132 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.2eb69dd0c58be63c", scope: null, file: !10, line: 24, type: !37, spFlags: DISPFlagDefinition, unit: !103, retainedNodes: !138)
// CHECK:STDOUT: !133 = !DILocalVariable(arg: 1, scope: !132, type: !35)
// CHECK:STDOUT: !136 = !DILocation(line: 26, column: 5, scope: !132)
// CHECK:STDOUT: !137 = !DILocation(line: 24, column: 3, scope: !132)
// CHECK:STDOUT: !138 = !{!133}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'final_function.carbon'
// CHECK:STDOUT: source_filename = "final_function.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !141 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CCallFn.Main.e342d6e8ff5d87ae(), !dbg !142
// CHECK:STDOUT:   ret void, !dbg !143
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CCallFn.Main.e342d6e8ff5d87ae() #0 !dbg !144 {
// CHECK:STDOUT:   call void @_CF.FinalFn.Main.e342d6e8ff5d87ae(), !dbg !145
// CHECK:STDOUT:   ret void, !dbg !146
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.FinalFn.Main.e342d6e8ff5d87ae() #0 !dbg !147 {
// CHECK:STDOUT:   ret void, !dbg !148
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!140}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !16 = !{null}
// CHECK:STDOUT: !17 = !DISubroutineType(types: !16)
// CHECK:STDOUT: !20 = !DIFile(filename: "final_interfaces.carbon", directory: "")
// CHECK:STDOUT: !139 = !DIFile(filename: "final_function.carbon", directory: "")
// CHECK:STDOUT: !140 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !139, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !141 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !139, line: 9, type: !17, spFlags: DISPFlagDefinition, unit: !140)
// CHECK:STDOUT: !142 = !DILocation(line: 10, column: 3, scope: !141)
// CHECK:STDOUT: !143 = !DILocation(line: 9, column: 1, scope: !141)
// CHECK:STDOUT: !144 = distinct !DISubprogram(name: "CallFn", linkageName: "_CCallFn.Main.e342d6e8ff5d87ae", scope: null, file: !20, line: 12, type: !17, spFlags: DISPFlagDefinition, unit: !140)
// CHECK:STDOUT: !145 = !DILocation(line: 13, column: 3, scope: !144)
// CHECK:STDOUT: !146 = !DILocation(line: 12, column: 1, scope: !144)
// CHECK:STDOUT: !147 = distinct !DISubprogram(name: "F", linkageName: "_CF.FinalFn.Main.e342d6e8ff5d87ae", scope: null, file: !20, line: 5, type: !17, spFlags: DISPFlagDefinition, unit: !140)
// CHECK:STDOUT: !148 = !DILocation(line: 5, column: 3, scope: !147)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'final_method.carbon'
// CHECK:STDOUT: source_filename = "final_method.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @C.val = internal constant {} zeroinitializer
// CHECK:STDOUT: @C.val.loc10_15.6 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !151 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc10_15.2.temp = alloca {}, align 1, !dbg !152
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc10_15.2.temp), !dbg !152
// CHECK:STDOUT:   call void @_CCallMethod.Main.96e0ef8a617a6362(ptr @C.val.loc10_15.6), !dbg !156
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.2eb69dd0c58be63c(ptr @C.val), !dbg !152
// CHECK:STDOUT:   ret void, !dbg !157
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.08d9e38b928ba575:core.Destroy.Core"(ptr %self) #0 !dbg !158 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !160
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CCallMethod.Main.96e0ef8a617a6362(ptr %t) #0 !dbg !162 {
// CHECK:STDOUT:   call void @_CM.FinalMethod.Main.96e0ef8a617a6362(ptr %t), !dbg !165
// CHECK:STDOUT:   ret void, !dbg !166
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.2eb69dd0c58be63c(ptr %self) #0 !dbg !168 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.08d9e38b928ba575:core.Destroy.Core"(ptr %self), !dbg !172
// CHECK:STDOUT:   ret void, !dbg !173
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CM.FinalMethod.Main.96e0ef8a617a6362(ptr %self) #0 !dbg !175 {
// CHECK:STDOUT:   ret void, !dbg !178
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!150}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !10 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !16 = !{null}
// CHECK:STDOUT: !17 = !DISubroutineType(types: !16)
// CHECK:STDOUT: !20 = !DIFile(filename: "final_interfaces.carbon", directory: "")
// CHECK:STDOUT: !35 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !36 = !{null, !35}
// CHECK:STDOUT: !37 = !DISubroutineType(types: !36)
// CHECK:STDOUT: !149 = !DIFile(filename: "final_method.carbon", directory: "")
// CHECK:STDOUT: !150 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !149, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !151 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !149, line: 9, type: !17, spFlags: DISPFlagDefinition, unit: !150)
// CHECK:STDOUT: !152 = !DILocation(line: 10, column: 14, scope: !151)
// CHECK:STDOUT: !156 = !DILocation(line: 11, column: 3, scope: !151)
// CHECK:STDOUT: !157 = !DILocation(line: 9, column: 1, scope: !151)
// CHECK:STDOUT: !158 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.08d9e38b928ba575:core.Destroy.Core", scope: null, file: !149, line: 10, type: !37, spFlags: DISPFlagDefinition, unit: !150, retainedNodes: !161)
// CHECK:STDOUT: !159 = !DILocalVariable(arg: 1, scope: !158, type: !35)
// CHECK:STDOUT: !160 = !DILocation(line: 10, column: 14, scope: !158)
// CHECK:STDOUT: !161 = !{!159}
// CHECK:STDOUT: !162 = distinct !DISubprogram(name: "CallMethod", linkageName: "_CCallMethod.Main.96e0ef8a617a6362", scope: null, file: !20, line: 16, type: !37, spFlags: DISPFlagDefinition, unit: !150, retainedNodes: !167)
// CHECK:STDOUT: !163 = !DILocalVariable(arg: 1, scope: !162, type: !35)
// CHECK:STDOUT: !165 = !DILocation(line: 17, column: 3, scope: !162)
// CHECK:STDOUT: !166 = !DILocation(line: 16, column: 1, scope: !162)
// CHECK:STDOUT: !167 = !{!163}
// CHECK:STDOUT: !168 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.2eb69dd0c58be63c", scope: null, file: !10, line: 24, type: !37, spFlags: DISPFlagDefinition, unit: !150, retainedNodes: !174)
// CHECK:STDOUT: !169 = !DILocalVariable(arg: 1, scope: !168, type: !35)
// CHECK:STDOUT: !172 = !DILocation(line: 26, column: 5, scope: !168)
// CHECK:STDOUT: !173 = !DILocation(line: 24, column: 3, scope: !168)
// CHECK:STDOUT: !174 = !{!169}
// CHECK:STDOUT: !175 = distinct !DISubprogram(name: "M", linkageName: "_CM.FinalMethod.Main.96e0ef8a617a6362", scope: null, file: !20, line: 9, type: !37, spFlags: DISPFlagDefinition, unit: !150, retainedNodes: !179)
// CHECK:STDOUT: !176 = !DILocalVariable(arg: 1, scope: !175, type: !35)
// CHECK:STDOUT: !178 = !DILocation(line: 9, column: 3, scope: !175)
// CHECK:STDOUT: !179 = !{!176}
// CHECK:STDOUT:
