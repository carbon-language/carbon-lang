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
// CHECK:STDOUT: define void @_CHello.Main() #0 !dbg !20 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!17}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !16 = !DIFile(filename: "default_interfaces.carbon", directory: "")
// CHECK:STDOUT: !17 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !16, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !18 = !{null}
// CHECK:STDOUT: !19 = !DISubroutineType(types: !18)
// CHECK:STDOUT: !20 = distinct !DISubprogram(name: "Hello", linkageName: "_CHello.Main", scope: null, file: !16, line: 4, type: !19, spFlags: DISPFlagDefinition, unit: !17)
// CHECK:STDOUT: !21 = !DILocation(line: 4, column: 1, scope: !20)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'final_interfaces.carbon'
// CHECK:STDOUT: source_filename = "final_interfaces.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!23}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !22 = !DIFile(filename: "final_interfaces.carbon", directory: "")
// CHECK:STDOUT: !23 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !22, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'default_function.carbon'
// CHECK:STDOUT: source_filename = "default_function.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !26 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CCallFn.Main.794e85173e977a8b(), !dbg !27
// CHECK:STDOUT:   ret void, !dbg !28
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CCallFn.Main.794e85173e977a8b() #0 !dbg !29 {
// CHECK:STDOUT:   call void @_CF.DefaultFn.Main.794e85173e977a8b(), !dbg !30
// CHECK:STDOUT:   ret void, !dbg !31
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CHello.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.DefaultFn.Main.794e85173e977a8b() #0 !dbg !32 {
// CHECK:STDOUT:   call void @_CHello.Main(), !dbg !33
// CHECK:STDOUT:   ret void, !dbg !34
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!25}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !16 = !DIFile(filename: "default_interfaces.carbon", directory: "")
// CHECK:STDOUT: !18 = !{null}
// CHECK:STDOUT: !19 = !DISubroutineType(types: !18)
// CHECK:STDOUT: !24 = !DIFile(filename: "default_function.carbon", directory: "")
// CHECK:STDOUT: !25 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !24, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !26 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !24, line: 9, type: !19, spFlags: DISPFlagDefinition, unit: !25)
// CHECK:STDOUT: !27 = !DILocation(line: 10, column: 3, scope: !26)
// CHECK:STDOUT: !28 = !DILocation(line: 9, column: 1, scope: !26)
// CHECK:STDOUT: !29 = distinct !DISubprogram(name: "CallFn", linkageName: "_CCallFn.Main.794e85173e977a8b", scope: null, file: !16, line: 16, type: !19, spFlags: DISPFlagDefinition, unit: !25)
// CHECK:STDOUT: !30 = !DILocation(line: 17, column: 3, scope: !29)
// CHECK:STDOUT: !31 = !DILocation(line: 16, column: 1, scope: !29)
// CHECK:STDOUT: !32 = distinct !DISubprogram(name: "F", linkageName: "_CF.DefaultFn.Main.794e85173e977a8b", scope: null, file: !16, line: 7, type: !19, spFlags: DISPFlagDefinition, unit: !25)
// CHECK:STDOUT: !33 = !DILocation(line: 7, column: 20, scope: !32)
// CHECK:STDOUT: !34 = !DILocation(line: 7, column: 3, scope: !32)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'default_method.carbon'
// CHECK:STDOUT: source_filename = "default_method.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @C.val = internal constant {} zeroinitializer
// CHECK:STDOUT: @C.val.loc10_15.6 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !37 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc10_15.2.temp = alloca {}, align 1, !dbg !38
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc10_15.2.temp), !dbg !38
// CHECK:STDOUT:   call void @_CCallMethod.Main.805a971dc5455081(ptr @C.val.loc10_15.6), !dbg !42
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.38e0a41f1f1b0274(ptr @C.val), !dbg !38
// CHECK:STDOUT:   ret void, !dbg !46
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CCallMethod.Main.805a971dc5455081(ptr %t) #0 !dbg !47 {
// CHECK:STDOUT:   call void @_CM.DefaultMethod.Main.805a971dc5455081(ptr %t), !dbg !50
// CHECK:STDOUT:   ret void, !dbg !51
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.38e0a41f1f1b0274(ptr %t) #0 !dbg !53 {
// CHECK:STDOUT:   ret void, !dbg !58
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CHello.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CM.DefaultMethod.Main.805a971dc5455081(ptr %self) #0 !dbg !60 {
// CHECK:STDOUT:   call void @_CWorld.Main.9522ff908e97e267(ptr %self), !dbg !64
// CHECK:STDOUT:   ret void, !dbg !65
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CWorld.Main.9522ff908e97e267(ptr %t) #0 !dbg !67 {
// CHECK:STDOUT:   ret void, !dbg !70
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!36}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !6 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !16 = !DIFile(filename: "default_interfaces.carbon", directory: "")
// CHECK:STDOUT: !18 = !{null}
// CHECK:STDOUT: !19 = !DISubroutineType(types: !18)
// CHECK:STDOUT: !35 = !DIFile(filename: "default_method.carbon", directory: "")
// CHECK:STDOUT: !36 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !35, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !37 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !35, line: 9, type: !19, spFlags: DISPFlagDefinition, unit: !36)
// CHECK:STDOUT: !38 = !DILocation(line: 10, column: 14, scope: !37)
// CHECK:STDOUT: !42 = !DILocation(line: 11, column: 3, scope: !37)
// CHECK:STDOUT: !43 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !44 = !{null, !43}
// CHECK:STDOUT: !45 = !DISubroutineType(types: !44)
// CHECK:STDOUT: !46 = !DILocation(line: 9, column: 1, scope: !37)
// CHECK:STDOUT: !47 = distinct !DISubprogram(name: "CallMethod", linkageName: "_CCallMethod.Main.805a971dc5455081", scope: null, file: !16, line: 20, type: !45, spFlags: DISPFlagDefinition, unit: !36, retainedNodes: !52)
// CHECK:STDOUT: !48 = !DILocalVariable(arg: 1, scope: !47, type: !43)
// CHECK:STDOUT: !50 = !DILocation(line: 21, column: 3, scope: !47)
// CHECK:STDOUT: !51 = !DILocation(line: 20, column: 1, scope: !47)
// CHECK:STDOUT: !52 = !{!48}
// CHECK:STDOUT: !53 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.38e0a41f1f1b0274", scope: null, file: !6, line: 30, type: !45, spFlags: DISPFlagDefinition, unit: !36, retainedNodes: !59)
// CHECK:STDOUT: !54 = !DILocalVariable(arg: 1, scope: !53, type: !43)
// CHECK:STDOUT: !58 = !DILocation(line: 30, column: 1, scope: !53)
// CHECK:STDOUT: !59 = !{!54}
// CHECK:STDOUT: !60 = distinct !DISubprogram(name: "M", linkageName: "_CM.DefaultMethod.Main.805a971dc5455081", scope: null, file: !16, line: 13, type: !45, spFlags: DISPFlagDefinition, unit: !36, retainedNodes: !66)
// CHECK:STDOUT: !61 = !DILocalVariable(arg: 1, scope: !60, type: !43)
// CHECK:STDOUT: !64 = !DILocation(line: 13, column: 24, scope: !60)
// CHECK:STDOUT: !65 = !DILocation(line: 13, column: 3, scope: !60)
// CHECK:STDOUT: !66 = !{!61}
// CHECK:STDOUT: !67 = distinct !DISubprogram(name: "World", linkageName: "_CWorld.Main.9522ff908e97e267", scope: null, file: !16, line: 10, type: !45, spFlags: DISPFlagDefinition, unit: !36, retainedNodes: !71)
// CHECK:STDOUT: !68 = !DILocalVariable(arg: 1, scope: !67, type: !43)
// CHECK:STDOUT: !70 = !DILocation(line: 10, column: 1, scope: !67)
// CHECK:STDOUT: !71 = !{!68}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'default_function_overridden.carbon'
// CHECK:STDOUT: source_filename = "default_function_overridden.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @C.val = internal constant {} zeroinitializer
// CHECK:STDOUT: @C.val.loc9_23.6 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CClone.C.Main(ptr %other) #0 !dbg !74 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !77
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_CF.C.Main:DefaultFn.Main"() #0 !dbg !79 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc9_23.2.temp = alloca {}, align 1, !dbg !80
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc9_23.2.temp), !dbg !80
// CHECK:STDOUT:   call void @_CClone.C.Main(ptr @C.val.loc9_23.6), !dbg !81
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.38e0a41f1f1b0274(ptr @C.val), !dbg !80
// CHECK:STDOUT:   ret void, !dbg !82
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !83 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CCallFn.Main.794e85173e977a8b(), !dbg !84
// CHECK:STDOUT:   ret void, !dbg !85
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.38e0a41f1f1b0274(ptr %t) #0 !dbg !86 {
// CHECK:STDOUT:   ret void, !dbg !91
// CHECK:STDOUT: }
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!73}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !6 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !16 = !DIFile(filename: "default_interfaces.carbon", directory: "")
// CHECK:STDOUT: !18 = !{null}
// CHECK:STDOUT: !19 = !DISubroutineType(types: !18)
// CHECK:STDOUT: !43 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !44 = !{null, !43}
// CHECK:STDOUT: !45 = !DISubroutineType(types: !44)
// CHECK:STDOUT: !72 = !DIFile(filename: "default_function_overridden.carbon", directory: "")
// CHECK:STDOUT: !73 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !72, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !74 = distinct !DISubprogram(name: "Clone", linkageName: "_CClone.C.Main", scope: null, file: !72, line: 6, type: !45, spFlags: DISPFlagDefinition, unit: !73, retainedNodes: !78)
// CHECK:STDOUT: !75 = !DILocalVariable(arg: 1, scope: !74, type: !43)
// CHECK:STDOUT: !77 = !DILocation(line: 6, column: 3, scope: !74)
// CHECK:STDOUT: !78 = !{!75}
// CHECK:STDOUT: !79 = distinct !DISubprogram(name: "F", linkageName: "_CF.C.Main:DefaultFn.Main", scope: null, file: !72, line: 9, type: !19, spFlags: DISPFlagDefinition, unit: !73)
// CHECK:STDOUT: !80 = !DILocation(line: 9, column: 22, scope: !79)
// CHECK:STDOUT: !81 = !DILocation(line: 9, column: 14, scope: !79)
// CHECK:STDOUT: !82 = !DILocation(line: 9, column: 5, scope: !79)
// CHECK:STDOUT: !83 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !72, line: 13, type: !19, spFlags: DISPFlagDefinition, unit: !73)
// CHECK:STDOUT: !84 = !DILocation(line: 14, column: 3, scope: !83)
// CHECK:STDOUT: !85 = !DILocation(line: 13, column: 1, scope: !83)
// CHECK:STDOUT: !86 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.38e0a41f1f1b0274", scope: null, file: !6, line: 30, type: !45, spFlags: DISPFlagDefinition, unit: !73, retainedNodes: !92)
// CHECK:STDOUT: !87 = !DILocalVariable(arg: 1, scope: !86, type: !43)
// CHECK:STDOUT: !91 = !DILocation(line: 30, column: 1, scope: !86)
// CHECK:STDOUT: !92 = !{!87}
// CHECK:STDOUT: !93 = distinct !DISubprogram(name: "CallFn", linkageName: "_CCallFn.Main.794e85173e977a8b", scope: null, file: !16, line: 16, type: !19, spFlags: DISPFlagDefinition, unit: !73)
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
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.38e0a41f1f1b0274(ptr @C.val), !dbg !110
// CHECK:STDOUT:   ret void, !dbg !115
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CCallMethod.Main.805a971dc5455081(ptr %t) #0 !dbg !116 {
// CHECK:STDOUT:   call void @"_CM.C.Main:DefaultMethod.Main"(ptr %t), !dbg !119
// CHECK:STDOUT:   ret void, !dbg !120
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.38e0a41f1f1b0274(ptr %t) #0 !dbg !122 {
// CHECK:STDOUT:   ret void, !dbg !127
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
// CHECK:STDOUT: !6 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !16 = !DIFile(filename: "default_interfaces.carbon", directory: "")
// CHECK:STDOUT: !18 = !{null}
// CHECK:STDOUT: !19 = !DISubroutineType(types: !18)
// CHECK:STDOUT: !43 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !44 = !{null, !43}
// CHECK:STDOUT: !45 = !DISubroutineType(types: !44)
// CHECK:STDOUT: !96 = !DIFile(filename: "default_method_overridden.carbon", directory: "")
// CHECK:STDOUT: !97 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !96, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !98 = distinct !DISubprogram(name: "Copy", linkageName: "_CCopy.C.Main", scope: null, file: !96, line: 6, type: !45, spFlags: DISPFlagDefinition, unit: !97, retainedNodes: !102)
// CHECK:STDOUT: !99 = !DILocalVariable(arg: 1, scope: !98, type: !43)
// CHECK:STDOUT: !101 = !DILocation(line: 6, column: 3, scope: !98)
// CHECK:STDOUT: !102 = !{!99}
// CHECK:STDOUT: !103 = distinct !DISubprogram(name: "M", linkageName: "_CM.C.Main:DefaultMethod.Main", scope: null, file: !96, line: 9, type: !45, spFlags: DISPFlagDefinition, unit: !97, retainedNodes: !108)
// CHECK:STDOUT: !104 = !DILocalVariable(arg: 1, scope: !103, type: !43)
// CHECK:STDOUT: !106 = !DILocation(line: 9, column: 18, scope: !103)
// CHECK:STDOUT: !107 = !DILocation(line: 9, column: 5, scope: !103)
// CHECK:STDOUT: !108 = !{!104}
// CHECK:STDOUT: !109 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !96, line: 13, type: !19, spFlags: DISPFlagDefinition, unit: !97)
// CHECK:STDOUT: !110 = !DILocation(line: 14, column: 14, scope: !109)
// CHECK:STDOUT: !114 = !DILocation(line: 15, column: 3, scope: !109)
// CHECK:STDOUT: !115 = !DILocation(line: 13, column: 1, scope: !109)
// CHECK:STDOUT: !116 = distinct !DISubprogram(name: "CallMethod", linkageName: "_CCallMethod.Main.805a971dc5455081", scope: null, file: !16, line: 20, type: !45, spFlags: DISPFlagDefinition, unit: !97, retainedNodes: !121)
// CHECK:STDOUT: !117 = !DILocalVariable(arg: 1, scope: !116, type: !43)
// CHECK:STDOUT: !119 = !DILocation(line: 21, column: 3, scope: !116)
// CHECK:STDOUT: !120 = !DILocation(line: 20, column: 1, scope: !116)
// CHECK:STDOUT: !121 = !{!117}
// CHECK:STDOUT: !122 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.38e0a41f1f1b0274", scope: null, file: !6, line: 30, type: !45, spFlags: DISPFlagDefinition, unit: !97, retainedNodes: !128)
// CHECK:STDOUT: !123 = !DILocalVariable(arg: 1, scope: !122, type: !43)
// CHECK:STDOUT: !127 = !DILocation(line: 30, column: 1, scope: !122)
// CHECK:STDOUT: !128 = !{!123}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'final_function.carbon'
// CHECK:STDOUT: source_filename = "final_function.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !131 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CCallFn.Main.e342d6e8ff5d87ae(), !dbg !132
// CHECK:STDOUT:   ret void, !dbg !133
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CCallFn.Main.e342d6e8ff5d87ae() #0 !dbg !134 {
// CHECK:STDOUT:   call void @_CF.FinalFn.Main.e342d6e8ff5d87ae(), !dbg !135
// CHECK:STDOUT:   ret void, !dbg !136
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.FinalFn.Main.e342d6e8ff5d87ae() #0 !dbg !137 {
// CHECK:STDOUT:   ret void, !dbg !138
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!130}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !18 = !{null}
// CHECK:STDOUT: !19 = !DISubroutineType(types: !18)
// CHECK:STDOUT: !22 = !DIFile(filename: "final_interfaces.carbon", directory: "")
// CHECK:STDOUT: !129 = !DIFile(filename: "final_function.carbon", directory: "")
// CHECK:STDOUT: !130 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !129, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !131 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !129, line: 9, type: !19, spFlags: DISPFlagDefinition, unit: !130)
// CHECK:STDOUT: !132 = !DILocation(line: 10, column: 3, scope: !131)
// CHECK:STDOUT: !133 = !DILocation(line: 9, column: 1, scope: !131)
// CHECK:STDOUT: !134 = distinct !DISubprogram(name: "CallFn", linkageName: "_CCallFn.Main.e342d6e8ff5d87ae", scope: null, file: !22, line: 12, type: !19, spFlags: DISPFlagDefinition, unit: !130)
// CHECK:STDOUT: !135 = !DILocation(line: 13, column: 3, scope: !134)
// CHECK:STDOUT: !136 = !DILocation(line: 12, column: 1, scope: !134)
// CHECK:STDOUT: !137 = distinct !DISubprogram(name: "F", linkageName: "_CF.FinalFn.Main.e342d6e8ff5d87ae", scope: null, file: !22, line: 5, type: !19, spFlags: DISPFlagDefinition, unit: !130)
// CHECK:STDOUT: !138 = !DILocation(line: 5, column: 3, scope: !137)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'final_method.carbon'
// CHECK:STDOUT: source_filename = "final_method.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @C.val = internal constant {} zeroinitializer
// CHECK:STDOUT: @C.val.loc10_15.6 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !141 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc10_15.2.temp = alloca {}, align 1, !dbg !142
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc10_15.2.temp), !dbg !142
// CHECK:STDOUT:   call void @_CCallMethod.Main.96e0ef8a617a6362(ptr @C.val.loc10_15.6), !dbg !146
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.38e0a41f1f1b0274(ptr @C.val), !dbg !142
// CHECK:STDOUT:   ret void, !dbg !147
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CCallMethod.Main.96e0ef8a617a6362(ptr %t) #0 !dbg !148 {
// CHECK:STDOUT:   call void @_CM.FinalMethod.Main.96e0ef8a617a6362(ptr %t), !dbg !151
// CHECK:STDOUT:   ret void, !dbg !152
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.38e0a41f1f1b0274(ptr %t) #0 !dbg !154 {
// CHECK:STDOUT:   ret void, !dbg !159
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CM.FinalMethod.Main.96e0ef8a617a6362(ptr %self) #0 !dbg !161 {
// CHECK:STDOUT:   ret void, !dbg !164
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!140}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !6 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !18 = !{null}
// CHECK:STDOUT: !19 = !DISubroutineType(types: !18)
// CHECK:STDOUT: !22 = !DIFile(filename: "final_interfaces.carbon", directory: "")
// CHECK:STDOUT: !43 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !44 = !{null, !43}
// CHECK:STDOUT: !45 = !DISubroutineType(types: !44)
// CHECK:STDOUT: !139 = !DIFile(filename: "final_method.carbon", directory: "")
// CHECK:STDOUT: !140 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !139, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !141 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !139, line: 9, type: !19, spFlags: DISPFlagDefinition, unit: !140)
// CHECK:STDOUT: !142 = !DILocation(line: 10, column: 14, scope: !141)
// CHECK:STDOUT: !146 = !DILocation(line: 11, column: 3, scope: !141)
// CHECK:STDOUT: !147 = !DILocation(line: 9, column: 1, scope: !141)
// CHECK:STDOUT: !148 = distinct !DISubprogram(name: "CallMethod", linkageName: "_CCallMethod.Main.96e0ef8a617a6362", scope: null, file: !22, line: 16, type: !45, spFlags: DISPFlagDefinition, unit: !140, retainedNodes: !153)
// CHECK:STDOUT: !149 = !DILocalVariable(arg: 1, scope: !148, type: !43)
// CHECK:STDOUT: !151 = !DILocation(line: 17, column: 3, scope: !148)
// CHECK:STDOUT: !152 = !DILocation(line: 16, column: 1, scope: !148)
// CHECK:STDOUT: !153 = !{!149}
// CHECK:STDOUT: !154 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.38e0a41f1f1b0274", scope: null, file: !6, line: 30, type: !45, spFlags: DISPFlagDefinition, unit: !140, retainedNodes: !160)
// CHECK:STDOUT: !155 = !DILocalVariable(arg: 1, scope: !154, type: !43)
// CHECK:STDOUT: !159 = !DILocation(line: 30, column: 1, scope: !154)
// CHECK:STDOUT: !160 = !{!155}
// CHECK:STDOUT: !161 = distinct !DISubprogram(name: "M", linkageName: "_CM.FinalMethod.Main.96e0ef8a617a6362", scope: null, file: !22, line: 9, type: !45, spFlags: DISPFlagDefinition, unit: !140, retainedNodes: !165)
// CHECK:STDOUT: !162 = !DILocalVariable(arg: 1, scope: !161, type: !43)
// CHECK:STDOUT: !164 = !DILocation(line: 9, column: 3, scope: !161)
// CHECK:STDOUT: !165 = !{!162}
// CHECK:STDOUT:
