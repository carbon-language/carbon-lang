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
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !35 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc10_15.2.temp = alloca {}, align 1, !dbg !36
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc10_15.2.temp), !dbg !36
// CHECK:STDOUT:   call void @_CCallMethod.Main.805a971dc5455081(ptr @C.val.loc10_15.6), !dbg !40
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.166956a092acbab5(ptr @C.val), !dbg !36
// CHECK:STDOUT:   ret void, !dbg !44
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CCallMethod.Main.805a971dc5455081(ptr %t) #0 !dbg !45 {
// CHECK:STDOUT:   call void @_CM.DefaultMethod.Main.805a971dc5455081(ptr %t), !dbg !48
// CHECK:STDOUT:   ret void, !dbg !49
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.166956a092acbab5(ptr %t) #0 !dbg !51 {
// CHECK:STDOUT:   ret void, !dbg !56
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CHello.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CM.DefaultMethod.Main.805a971dc5455081(ptr %self) #0 !dbg !58 {
// CHECK:STDOUT:   call void @_CWorld.Main.9522ff908e97e267(ptr %self), !dbg !62
// CHECK:STDOUT:   ret void, !dbg !63
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CWorld.Main.9522ff908e97e267(ptr %t) #0 !dbg !65 {
// CHECK:STDOUT:   ret void, !dbg !68
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
// CHECK:STDOUT: !6 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !14 = !DIFile(filename: "default_interfaces.carbon", directory: "")
// CHECK:STDOUT: !16 = !{null}
// CHECK:STDOUT: !17 = !DISubroutineType(types: !16)
// CHECK:STDOUT: !33 = !DIFile(filename: "default_method.carbon", directory: "")
// CHECK:STDOUT: !34 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !33, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !35 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !33, line: 9, type: !17, spFlags: DISPFlagDefinition, unit: !34)
// CHECK:STDOUT: !36 = !DILocation(line: 10, column: 14, scope: !35)
// CHECK:STDOUT: !40 = !DILocation(line: 11, column: 3, scope: !35)
// CHECK:STDOUT: !41 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !42 = !{null, !41}
// CHECK:STDOUT: !43 = !DISubroutineType(types: !42)
// CHECK:STDOUT: !44 = !DILocation(line: 9, column: 1, scope: !35)
// CHECK:STDOUT: !45 = distinct !DISubprogram(name: "CallMethod", linkageName: "_CCallMethod.Main.805a971dc5455081", scope: null, file: !14, line: 20, type: !43, spFlags: DISPFlagDefinition, unit: !34, retainedNodes: !50)
// CHECK:STDOUT: !46 = !DILocalVariable(arg: 1, scope: !45, type: !41)
// CHECK:STDOUT: !48 = !DILocation(line: 21, column: 3, scope: !45)
// CHECK:STDOUT: !49 = !DILocation(line: 20, column: 1, scope: !45)
// CHECK:STDOUT: !50 = !{!46}
// CHECK:STDOUT: !51 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.166956a092acbab5", scope: null, file: !6, line: 26, type: !43, spFlags: DISPFlagDefinition, unit: !34, retainedNodes: !57)
// CHECK:STDOUT: !52 = !DILocalVariable(arg: 1, scope: !51, type: !41)
// CHECK:STDOUT: !56 = !DILocation(line: 26, column: 1, scope: !51)
// CHECK:STDOUT: !57 = !{!52}
// CHECK:STDOUT: !58 = distinct !DISubprogram(name: "M", linkageName: "_CM.DefaultMethod.Main.805a971dc5455081", scope: null, file: !14, line: 13, type: !43, spFlags: DISPFlagDefinition, unit: !34, retainedNodes: !64)
// CHECK:STDOUT: !59 = !DILocalVariable(arg: 1, scope: !58, type: !41)
// CHECK:STDOUT: !62 = !DILocation(line: 13, column: 24, scope: !58)
// CHECK:STDOUT: !63 = !DILocation(line: 13, column: 3, scope: !58)
// CHECK:STDOUT: !64 = !{!59}
// CHECK:STDOUT: !65 = distinct !DISubprogram(name: "World", linkageName: "_CWorld.Main.9522ff908e97e267", scope: null, file: !14, line: 10, type: !43, spFlags: DISPFlagDefinition, unit: !34, retainedNodes: !69)
// CHECK:STDOUT: !66 = !DILocalVariable(arg: 1, scope: !65, type: !41)
// CHECK:STDOUT: !68 = !DILocation(line: 10, column: 1, scope: !65)
// CHECK:STDOUT: !69 = !{!66}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'default_function_overridden.carbon'
// CHECK:STDOUT: source_filename = "default_function_overridden.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @C.val = internal constant {} zeroinitializer
// CHECK:STDOUT: @C.val.loc9_23.6 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CClone.C.Main(ptr %other) #0 !dbg !72 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !75
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_CF.C.Main:DefaultFn.Main"() #0 !dbg !77 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc9_23.2.temp = alloca {}, align 1, !dbg !78
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc9_23.2.temp), !dbg !78
// CHECK:STDOUT:   call void @_CClone.C.Main(ptr @C.val.loc9_23.6), !dbg !79
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.166956a092acbab5(ptr @C.val), !dbg !78
// CHECK:STDOUT:   ret void, !dbg !80
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !81 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CCallFn.Main.794e85173e977a8b(), !dbg !82
// CHECK:STDOUT:   ret void, !dbg !83
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.166956a092acbab5(ptr %t) #0 !dbg !84 {
// CHECK:STDOUT:   ret void, !dbg !89
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CCallFn.Main.794e85173e977a8b() #0 !dbg !91 {
// CHECK:STDOUT:   call void @"_CF.C.Main:DefaultFn.Main"(), !dbg !92
// CHECK:STDOUT:   ret void, !dbg !93
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CHello.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!71}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !6 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !14 = !DIFile(filename: "default_interfaces.carbon", directory: "")
// CHECK:STDOUT: !16 = !{null}
// CHECK:STDOUT: !17 = !DISubroutineType(types: !16)
// CHECK:STDOUT: !41 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !42 = !{null, !41}
// CHECK:STDOUT: !43 = !DISubroutineType(types: !42)
// CHECK:STDOUT: !70 = !DIFile(filename: "default_function_overridden.carbon", directory: "")
// CHECK:STDOUT: !71 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !70, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !72 = distinct !DISubprogram(name: "Clone", linkageName: "_CClone.C.Main", scope: null, file: !70, line: 6, type: !43, spFlags: DISPFlagDefinition, unit: !71, retainedNodes: !76)
// CHECK:STDOUT: !73 = !DILocalVariable(arg: 1, scope: !72, type: !41)
// CHECK:STDOUT: !75 = !DILocation(line: 6, column: 3, scope: !72)
// CHECK:STDOUT: !76 = !{!73}
// CHECK:STDOUT: !77 = distinct !DISubprogram(name: "F", linkageName: "_CF.C.Main:DefaultFn.Main", scope: null, file: !70, line: 9, type: !17, spFlags: DISPFlagDefinition, unit: !71)
// CHECK:STDOUT: !78 = !DILocation(line: 9, column: 22, scope: !77)
// CHECK:STDOUT: !79 = !DILocation(line: 9, column: 14, scope: !77)
// CHECK:STDOUT: !80 = !DILocation(line: 9, column: 5, scope: !77)
// CHECK:STDOUT: !81 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !70, line: 13, type: !17, spFlags: DISPFlagDefinition, unit: !71)
// CHECK:STDOUT: !82 = !DILocation(line: 14, column: 3, scope: !81)
// CHECK:STDOUT: !83 = !DILocation(line: 13, column: 1, scope: !81)
// CHECK:STDOUT: !84 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.166956a092acbab5", scope: null, file: !6, line: 26, type: !43, spFlags: DISPFlagDefinition, unit: !71, retainedNodes: !90)
// CHECK:STDOUT: !85 = !DILocalVariable(arg: 1, scope: !84, type: !41)
// CHECK:STDOUT: !89 = !DILocation(line: 26, column: 1, scope: !84)
// CHECK:STDOUT: !90 = !{!85}
// CHECK:STDOUT: !91 = distinct !DISubprogram(name: "CallFn", linkageName: "_CCallFn.Main.794e85173e977a8b", scope: null, file: !14, line: 16, type: !17, spFlags: DISPFlagDefinition, unit: !71)
// CHECK:STDOUT: !92 = !DILocation(line: 17, column: 3, scope: !91)
// CHECK:STDOUT: !93 = !DILocation(line: 16, column: 1, scope: !91)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'default_method_overridden.carbon'
// CHECK:STDOUT: source_filename = "default_method_overridden.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @C.val = internal constant {} zeroinitializer
// CHECK:STDOUT: @C.val.loc14_15.6 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCopy.C.Main(ptr %self) #0 !dbg !96 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !99
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_CM.C.Main:DefaultMethod.Main"(ptr %self) #0 !dbg !101 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CCopy.C.Main(ptr %self), !dbg !104
// CHECK:STDOUT:   ret void, !dbg !105
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !107 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc14_15.2.temp = alloca {}, align 1, !dbg !108
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc14_15.2.temp), !dbg !108
// CHECK:STDOUT:   call void @_CCallMethod.Main.805a971dc5455081(ptr @C.val.loc14_15.6), !dbg !112
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.166956a092acbab5(ptr @C.val), !dbg !108
// CHECK:STDOUT:   ret void, !dbg !113
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CCallMethod.Main.805a971dc5455081(ptr %t) #0 !dbg !114 {
// CHECK:STDOUT:   call void @"_CM.C.Main:DefaultMethod.Main"(ptr %t), !dbg !117
// CHECK:STDOUT:   ret void, !dbg !118
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.166956a092acbab5(ptr %t) #0 !dbg !120 {
// CHECK:STDOUT:   ret void, !dbg !125
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CHello.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!95}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !6 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !14 = !DIFile(filename: "default_interfaces.carbon", directory: "")
// CHECK:STDOUT: !16 = !{null}
// CHECK:STDOUT: !17 = !DISubroutineType(types: !16)
// CHECK:STDOUT: !41 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !42 = !{null, !41}
// CHECK:STDOUT: !43 = !DISubroutineType(types: !42)
// CHECK:STDOUT: !94 = !DIFile(filename: "default_method_overridden.carbon", directory: "")
// CHECK:STDOUT: !95 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !94, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !96 = distinct !DISubprogram(name: "Copy", linkageName: "_CCopy.C.Main", scope: null, file: !94, line: 6, type: !43, spFlags: DISPFlagDefinition, unit: !95, retainedNodes: !100)
// CHECK:STDOUT: !97 = !DILocalVariable(arg: 1, scope: !96, type: !41)
// CHECK:STDOUT: !99 = !DILocation(line: 6, column: 3, scope: !96)
// CHECK:STDOUT: !100 = !{!97}
// CHECK:STDOUT: !101 = distinct !DISubprogram(name: "M", linkageName: "_CM.C.Main:DefaultMethod.Main", scope: null, file: !94, line: 9, type: !43, spFlags: DISPFlagDefinition, unit: !95, retainedNodes: !106)
// CHECK:STDOUT: !102 = !DILocalVariable(arg: 1, scope: !101, type: !41)
// CHECK:STDOUT: !104 = !DILocation(line: 9, column: 18, scope: !101)
// CHECK:STDOUT: !105 = !DILocation(line: 9, column: 5, scope: !101)
// CHECK:STDOUT: !106 = !{!102}
// CHECK:STDOUT: !107 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !94, line: 13, type: !17, spFlags: DISPFlagDefinition, unit: !95)
// CHECK:STDOUT: !108 = !DILocation(line: 14, column: 14, scope: !107)
// CHECK:STDOUT: !112 = !DILocation(line: 15, column: 3, scope: !107)
// CHECK:STDOUT: !113 = !DILocation(line: 13, column: 1, scope: !107)
// CHECK:STDOUT: !114 = distinct !DISubprogram(name: "CallMethod", linkageName: "_CCallMethod.Main.805a971dc5455081", scope: null, file: !14, line: 20, type: !43, spFlags: DISPFlagDefinition, unit: !95, retainedNodes: !119)
// CHECK:STDOUT: !115 = !DILocalVariable(arg: 1, scope: !114, type: !41)
// CHECK:STDOUT: !117 = !DILocation(line: 21, column: 3, scope: !114)
// CHECK:STDOUT: !118 = !DILocation(line: 20, column: 1, scope: !114)
// CHECK:STDOUT: !119 = !{!115}
// CHECK:STDOUT: !120 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.166956a092acbab5", scope: null, file: !6, line: 26, type: !43, spFlags: DISPFlagDefinition, unit: !95, retainedNodes: !126)
// CHECK:STDOUT: !121 = !DILocalVariable(arg: 1, scope: !120, type: !41)
// CHECK:STDOUT: !125 = !DILocation(line: 26, column: 1, scope: !120)
// CHECK:STDOUT: !126 = !{!121}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'final_function.carbon'
// CHECK:STDOUT: source_filename = "final_function.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !129 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CCallFn.Main.e342d6e8ff5d87ae(), !dbg !130
// CHECK:STDOUT:   ret void, !dbg !131
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CCallFn.Main.e342d6e8ff5d87ae() #0 !dbg !132 {
// CHECK:STDOUT:   call void @_CF.FinalFn.Main.e342d6e8ff5d87ae(), !dbg !133
// CHECK:STDOUT:   ret void, !dbg !134
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.FinalFn.Main.e342d6e8ff5d87ae() #0 !dbg !135 {
// CHECK:STDOUT:   ret void, !dbg !136
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!128}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !16 = !{null}
// CHECK:STDOUT: !17 = !DISubroutineType(types: !16)
// CHECK:STDOUT: !20 = !DIFile(filename: "final_interfaces.carbon", directory: "")
// CHECK:STDOUT: !127 = !DIFile(filename: "final_function.carbon", directory: "")
// CHECK:STDOUT: !128 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !127, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !129 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !127, line: 9, type: !17, spFlags: DISPFlagDefinition, unit: !128)
// CHECK:STDOUT: !130 = !DILocation(line: 10, column: 3, scope: !129)
// CHECK:STDOUT: !131 = !DILocation(line: 9, column: 1, scope: !129)
// CHECK:STDOUT: !132 = distinct !DISubprogram(name: "CallFn", linkageName: "_CCallFn.Main.e342d6e8ff5d87ae", scope: null, file: !20, line: 12, type: !17, spFlags: DISPFlagDefinition, unit: !128)
// CHECK:STDOUT: !133 = !DILocation(line: 13, column: 3, scope: !132)
// CHECK:STDOUT: !134 = !DILocation(line: 12, column: 1, scope: !132)
// CHECK:STDOUT: !135 = distinct !DISubprogram(name: "F", linkageName: "_CF.FinalFn.Main.e342d6e8ff5d87ae", scope: null, file: !20, line: 5, type: !17, spFlags: DISPFlagDefinition, unit: !128)
// CHECK:STDOUT: !136 = !DILocation(line: 5, column: 3, scope: !135)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'final_method.carbon'
// CHECK:STDOUT: source_filename = "final_method.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @C.val = internal constant {} zeroinitializer
// CHECK:STDOUT: @C.val.loc10_15.6 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !139 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc10_15.2.temp = alloca {}, align 1, !dbg !140
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc10_15.2.temp), !dbg !140
// CHECK:STDOUT:   call void @_CCallMethod.Main.96e0ef8a617a6362(ptr @C.val.loc10_15.6), !dbg !144
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.166956a092acbab5(ptr @C.val), !dbg !140
// CHECK:STDOUT:   ret void, !dbg !145
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CCallMethod.Main.96e0ef8a617a6362(ptr %t) #0 !dbg !146 {
// CHECK:STDOUT:   call void @_CM.FinalMethod.Main.96e0ef8a617a6362(ptr %t), !dbg !149
// CHECK:STDOUT:   ret void, !dbg !150
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.166956a092acbab5(ptr %t) #0 !dbg !152 {
// CHECK:STDOUT:   ret void, !dbg !157
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CM.FinalMethod.Main.96e0ef8a617a6362(ptr %self) #0 !dbg !159 {
// CHECK:STDOUT:   ret void, !dbg !162
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!138}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !6 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !16 = !{null}
// CHECK:STDOUT: !17 = !DISubroutineType(types: !16)
// CHECK:STDOUT: !20 = !DIFile(filename: "final_interfaces.carbon", directory: "")
// CHECK:STDOUT: !41 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !42 = !{null, !41}
// CHECK:STDOUT: !43 = !DISubroutineType(types: !42)
// CHECK:STDOUT: !137 = !DIFile(filename: "final_method.carbon", directory: "")
// CHECK:STDOUT: !138 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !137, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !139 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !137, line: 9, type: !17, spFlags: DISPFlagDefinition, unit: !138)
// CHECK:STDOUT: !140 = !DILocation(line: 10, column: 14, scope: !139)
// CHECK:STDOUT: !144 = !DILocation(line: 11, column: 3, scope: !139)
// CHECK:STDOUT: !145 = !DILocation(line: 9, column: 1, scope: !139)
// CHECK:STDOUT: !146 = distinct !DISubprogram(name: "CallMethod", linkageName: "_CCallMethod.Main.96e0ef8a617a6362", scope: null, file: !20, line: 16, type: !43, spFlags: DISPFlagDefinition, unit: !138, retainedNodes: !151)
// CHECK:STDOUT: !147 = !DILocalVariable(arg: 1, scope: !146, type: !41)
// CHECK:STDOUT: !149 = !DILocation(line: 17, column: 3, scope: !146)
// CHECK:STDOUT: !150 = !DILocation(line: 16, column: 1, scope: !146)
// CHECK:STDOUT: !151 = !{!147}
// CHECK:STDOUT: !152 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.166956a092acbab5", scope: null, file: !6, line: 26, type: !43, spFlags: DISPFlagDefinition, unit: !138, retainedNodes: !158)
// CHECK:STDOUT: !153 = !DILocalVariable(arg: 1, scope: !152, type: !41)
// CHECK:STDOUT: !157 = !DILocation(line: 26, column: 1, scope: !152)
// CHECK:STDOUT: !158 = !{!153}
// CHECK:STDOUT: !159 = distinct !DISubprogram(name: "M", linkageName: "_CM.FinalMethod.Main.96e0ef8a617a6362", scope: null, file: !20, line: 9, type: !43, spFlags: DISPFlagDefinition, unit: !138, retainedNodes: !163)
// CHECK:STDOUT: !160 = !DILocalVariable(arg: 1, scope: !159, type: !41)
// CHECK:STDOUT: !162 = !DILocation(line: 9, column: 3, scope: !159)
// CHECK:STDOUT: !163 = !{!160}
// CHECK:STDOUT:
