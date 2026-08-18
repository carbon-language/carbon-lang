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
// CHECK:STDOUT: define void @_CHello.Main() #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!0}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !1 = !DIFile(filename: "default_interfaces.carbon", directory: "")
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Hello", linkageName: "_CHello.Main", scope: null, file: !1, line: 4, type: !5, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{null}
// CHECK:STDOUT: !7 = !DILocation(line: 4, column: 1, scope: !4)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'final_interfaces.carbon'
// CHECK:STDOUT: source_filename = "final_interfaces.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!0}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !1 = !DIFile(filename: "final_interfaces.carbon", directory: "")
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'default_function.carbon'
// CHECK:STDOUT: source_filename = "default_function.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CCallFn.Main.794e85173e977a8b(), !dbg !7
// CHECK:STDOUT:   ret void, !dbg !8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CCallFn.Main.794e85173e977a8b() #0 !dbg !9 {
// CHECK:STDOUT:   call void @_CF.DefaultFn.Main.794e85173e977a8b(), !dbg !11
// CHECK:STDOUT:   ret void, !dbg !12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CHello.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.DefaultFn.Main.794e85173e977a8b() #0 !dbg !13 {
// CHECK:STDOUT:   call void @_CHello.Main(), !dbg !14
// CHECK:STDOUT:   ret void, !dbg !15
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!0}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !1 = !DIFile(filename: "default_function.carbon", directory: "")
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !1, line: 9, type: !5, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{null}
// CHECK:STDOUT: !7 = !DILocation(line: 10, column: 3, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 9, column: 1, scope: !4)
// CHECK:STDOUT: !9 = distinct !DISubprogram(name: "CallFn", linkageName: "_CCallFn.Main.794e85173e977a8b", scope: null, file: !10, line: 16, type: !5, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !10 = !DIFile(filename: "default_interfaces.carbon", directory: "")
// CHECK:STDOUT: !11 = !DILocation(line: 17, column: 3, scope: !9)
// CHECK:STDOUT: !12 = !DILocation(line: 16, column: 1, scope: !9)
// CHECK:STDOUT: !13 = distinct !DISubprogram(name: "F", linkageName: "_CF.DefaultFn.Main.794e85173e977a8b", scope: null, file: !10, line: 7, type: !5, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !14 = !DILocation(line: 7, column: 20, scope: !13)
// CHECK:STDOUT: !15 = !DILocation(line: 7, column: 3, scope: !13)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'default_method.carbon'
// CHECK:STDOUT: source_filename = "default_method.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @C.val = internal constant {} zeroinitializer
// CHECK:STDOUT: @C.val.loc10_15.6 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc10_15.2.temp = alloca {}, align 1, !dbg !7
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc10_15.2.temp), !dbg !7
// CHECK:STDOUT:   call void @_CCallMethod.Main.805a971dc5455081(ptr @C.val.loc10_15.6), !dbg !8
// CHECK:STDOUT:   call void @"_COp.807eb4f2d3ed7e54:core.Destroy.Core"(ptr @C.val), !dbg !7
// CHECK:STDOUT:   ret void, !dbg !9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.807eb4f2d3ed7e54:core.Destroy.Core"(ptr %self) #0 !dbg !10 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CCallMethod.Main.805a971dc5455081(ptr %t) #0 !dbg !17 {
// CHECK:STDOUT:   call void @_CM.DefaultMethod.Main.805a971dc5455081(ptr %t), !dbg !21
// CHECK:STDOUT:   ret void, !dbg !22
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CHello.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CM.DefaultMethod.Main.805a971dc5455081(ptr %self) #0 !dbg !23 {
// CHECK:STDOUT:   call void @_CWorld.Main.9522ff908e97e267(ptr %self), !dbg !26
// CHECK:STDOUT:   ret void, !dbg !27
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CWorld.Main.9522ff908e97e267(ptr %t) #0 !dbg !28 {
// CHECK:STDOUT:   ret void, !dbg !31
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!0}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !1 = !DIFile(filename: "default_method.carbon", directory: "")
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !1, line: 9, type: !5, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{null}
// CHECK:STDOUT: !7 = !DILocation(line: 10, column: 14, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 11, column: 3, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 9, column: 1, scope: !4)
// CHECK:STDOUT: !10 = distinct !DISubprogram(name: "Op", linkageName: "_COp.807eb4f2d3ed7e54:core.Destroy.Core", scope: null, file: !1, line: 10, type: !11, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !14)
// CHECK:STDOUT: !11 = !DISubroutineType(types: !12)
// CHECK:STDOUT: !12 = !{null, !13}
// CHECK:STDOUT: !13 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !14 = !{!15}
// CHECK:STDOUT: !15 = !DILocalVariable(arg: 1, scope: !10, type: !13)
// CHECK:STDOUT: !16 = !DILocation(line: 10, column: 14, scope: !10)
// CHECK:STDOUT: !17 = distinct !DISubprogram(name: "CallMethod", linkageName: "_CCallMethod.Main.805a971dc5455081", scope: null, file: !18, line: 20, type: !11, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !19)
// CHECK:STDOUT: !18 = !DIFile(filename: "default_interfaces.carbon", directory: "")
// CHECK:STDOUT: !19 = !{!20}
// CHECK:STDOUT: !20 = !DILocalVariable(arg: 1, scope: !17, type: !13)
// CHECK:STDOUT: !21 = !DILocation(line: 21, column: 3, scope: !17)
// CHECK:STDOUT: !22 = !DILocation(line: 20, column: 1, scope: !17)
// CHECK:STDOUT: !23 = distinct !DISubprogram(name: "M", linkageName: "_CM.DefaultMethod.Main.805a971dc5455081", scope: null, file: !18, line: 13, type: !11, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !24)
// CHECK:STDOUT: !24 = !{!25}
// CHECK:STDOUT: !25 = !DILocalVariable(arg: 1, scope: !23, type: !13)
// CHECK:STDOUT: !26 = !DILocation(line: 13, column: 24, scope: !23)
// CHECK:STDOUT: !27 = !DILocation(line: 13, column: 3, scope: !23)
// CHECK:STDOUT: !28 = distinct !DISubprogram(name: "World", linkageName: "_CWorld.Main.9522ff908e97e267", scope: null, file: !18, line: 10, type: !11, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !29)
// CHECK:STDOUT: !29 = !{!30}
// CHECK:STDOUT: !30 = !DILocalVariable(arg: 1, scope: !28, type: !13)
// CHECK:STDOUT: !31 = !DILocation(line: 10, column: 1, scope: !28)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'default_function_overridden.carbon'
// CHECK:STDOUT: source_filename = "default_function_overridden.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @C.val = internal constant {} zeroinitializer
// CHECK:STDOUT: @C.val.loc9_23.6 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CClone.C.Main(ptr %other) #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_CF.C.Main:DefaultFn.Main"() #0 !dbg !11 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc9_23.2.temp = alloca {}, align 1, !dbg !14
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc9_23.2.temp), !dbg !14
// CHECK:STDOUT:   call void @_CClone.C.Main(ptr @C.val.loc9_23.6), !dbg !15
// CHECK:STDOUT:   call void @"_COp.807eb4f2d3ed7e54:core.Destroy.Core"(ptr @C.val), !dbg !14
// CHECK:STDOUT:   ret void, !dbg !16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.807eb4f2d3ed7e54:core.Destroy.Core"(ptr %self) #0 !dbg !17 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !20
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !21 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CCallFn.Main.794e85173e977a8b(), !dbg !22
// CHECK:STDOUT:   ret void, !dbg !23
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CCallFn.Main.794e85173e977a8b() #0 !dbg !24 {
// CHECK:STDOUT:   call void @"_CF.C.Main:DefaultFn.Main"(), !dbg !26
// CHECK:STDOUT:   ret void, !dbg !27
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CHello.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!0}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !1 = !DIFile(filename: "default_function_overridden.carbon", directory: "")
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Clone", linkageName: "_CClone.C.Main", scope: null, file: !1, line: 6, type: !5, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !8)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{null, !7}
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !8 = !{!9}
// CHECK:STDOUT: !9 = !DILocalVariable(arg: 1, scope: !4, type: !7)
// CHECK:STDOUT: !10 = !DILocation(line: 6, column: 3, scope: !4)
// CHECK:STDOUT: !11 = distinct !DISubprogram(name: "F", linkageName: "_CF.C.Main:DefaultFn.Main", scope: null, file: !1, line: 9, type: !12, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !12 = !DISubroutineType(types: !13)
// CHECK:STDOUT: !13 = !{null}
// CHECK:STDOUT: !14 = !DILocation(line: 9, column: 22, scope: !11)
// CHECK:STDOUT: !15 = !DILocation(line: 9, column: 14, scope: !11)
// CHECK:STDOUT: !16 = !DILocation(line: 9, column: 5, scope: !11)
// CHECK:STDOUT: !17 = distinct !DISubprogram(name: "Op", linkageName: "_COp.807eb4f2d3ed7e54:core.Destroy.Core", scope: null, file: !1, line: 9, type: !5, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !18)
// CHECK:STDOUT: !18 = !{!19}
// CHECK:STDOUT: !19 = !DILocalVariable(arg: 1, scope: !17, type: !7)
// CHECK:STDOUT: !20 = !DILocation(line: 9, column: 22, scope: !17)
// CHECK:STDOUT: !21 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !1, line: 13, type: !12, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !22 = !DILocation(line: 14, column: 3, scope: !21)
// CHECK:STDOUT: !23 = !DILocation(line: 13, column: 1, scope: !21)
// CHECK:STDOUT: !24 = distinct !DISubprogram(name: "CallFn", linkageName: "_CCallFn.Main.794e85173e977a8b", scope: null, file: !25, line: 16, type: !12, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !25 = !DIFile(filename: "default_interfaces.carbon", directory: "")
// CHECK:STDOUT: !26 = !DILocation(line: 17, column: 3, scope: !24)
// CHECK:STDOUT: !27 = !DILocation(line: 16, column: 1, scope: !24)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'default_method_overridden.carbon'
// CHECK:STDOUT: source_filename = "default_method_overridden.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @C.val = internal constant {} zeroinitializer
// CHECK:STDOUT: @C.val.loc14_15.6 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCopy.C.Main(ptr %self) #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_CM.C.Main:DefaultMethod.Main"(ptr %self) #0 !dbg !11 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CCopy.C.Main(ptr %self), !dbg !14
// CHECK:STDOUT:   ret void, !dbg !15
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !16 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc14_15.2.temp = alloca {}, align 1, !dbg !19
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc14_15.2.temp), !dbg !19
// CHECK:STDOUT:   call void @_CCallMethod.Main.805a971dc5455081(ptr @C.val.loc14_15.6), !dbg !20
// CHECK:STDOUT:   call void @"_COp.807eb4f2d3ed7e54:core.Destroy.Core"(ptr @C.val), !dbg !19
// CHECK:STDOUT:   ret void, !dbg !21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.807eb4f2d3ed7e54:core.Destroy.Core"(ptr %self) #0 !dbg !22 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !25
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CCallMethod.Main.805a971dc5455081(ptr %t) #0 !dbg !26 {
// CHECK:STDOUT:   call void @"_CM.C.Main:DefaultMethod.Main"(ptr %t), !dbg !30
// CHECK:STDOUT:   ret void, !dbg !31
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CHello.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!0}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !1 = !DIFile(filename: "default_method_overridden.carbon", directory: "")
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Copy", linkageName: "_CCopy.C.Main", scope: null, file: !1, line: 6, type: !5, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !8)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{null, !7}
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !8 = !{!9}
// CHECK:STDOUT: !9 = !DILocalVariable(arg: 1, scope: !4, type: !7)
// CHECK:STDOUT: !10 = !DILocation(line: 6, column: 3, scope: !4)
// CHECK:STDOUT: !11 = distinct !DISubprogram(name: "M", linkageName: "_CM.C.Main:DefaultMethod.Main", scope: null, file: !1, line: 9, type: !5, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !12)
// CHECK:STDOUT: !12 = !{!13}
// CHECK:STDOUT: !13 = !DILocalVariable(arg: 1, scope: !11, type: !7)
// CHECK:STDOUT: !14 = !DILocation(line: 9, column: 18, scope: !11)
// CHECK:STDOUT: !15 = !DILocation(line: 9, column: 5, scope: !11)
// CHECK:STDOUT: !16 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !1, line: 13, type: !17, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !17 = !DISubroutineType(types: !18)
// CHECK:STDOUT: !18 = !{null}
// CHECK:STDOUT: !19 = !DILocation(line: 14, column: 14, scope: !16)
// CHECK:STDOUT: !20 = !DILocation(line: 15, column: 3, scope: !16)
// CHECK:STDOUT: !21 = !DILocation(line: 13, column: 1, scope: !16)
// CHECK:STDOUT: !22 = distinct !DISubprogram(name: "Op", linkageName: "_COp.807eb4f2d3ed7e54:core.Destroy.Core", scope: null, file: !1, line: 14, type: !5, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !23)
// CHECK:STDOUT: !23 = !{!24}
// CHECK:STDOUT: !24 = !DILocalVariable(arg: 1, scope: !22, type: !7)
// CHECK:STDOUT: !25 = !DILocation(line: 14, column: 14, scope: !22)
// CHECK:STDOUT: !26 = distinct !DISubprogram(name: "CallMethod", linkageName: "_CCallMethod.Main.805a971dc5455081", scope: null, file: !27, line: 20, type: !5, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !28)
// CHECK:STDOUT: !27 = !DIFile(filename: "default_interfaces.carbon", directory: "")
// CHECK:STDOUT: !28 = !{!29}
// CHECK:STDOUT: !29 = !DILocalVariable(arg: 1, scope: !26, type: !7)
// CHECK:STDOUT: !30 = !DILocation(line: 21, column: 3, scope: !26)
// CHECK:STDOUT: !31 = !DILocation(line: 20, column: 1, scope: !26)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'final_function.carbon'
// CHECK:STDOUT: source_filename = "final_function.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CCallFn.Main.e342d6e8ff5d87ae(), !dbg !7
// CHECK:STDOUT:   ret void, !dbg !8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CCallFn.Main.e342d6e8ff5d87ae() #0 !dbg !9 {
// CHECK:STDOUT:   call void @_CF.FinalFn.Main.e342d6e8ff5d87ae(), !dbg !11
// CHECK:STDOUT:   ret void, !dbg !12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.FinalFn.Main.e342d6e8ff5d87ae() #0 !dbg !13 {
// CHECK:STDOUT:   ret void, !dbg !14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!0}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !1 = !DIFile(filename: "final_function.carbon", directory: "")
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !1, line: 9, type: !5, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{null}
// CHECK:STDOUT: !7 = !DILocation(line: 10, column: 3, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 9, column: 1, scope: !4)
// CHECK:STDOUT: !9 = distinct !DISubprogram(name: "CallFn", linkageName: "_CCallFn.Main.e342d6e8ff5d87ae", scope: null, file: !10, line: 12, type: !5, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !10 = !DIFile(filename: "final_interfaces.carbon", directory: "")
// CHECK:STDOUT: !11 = !DILocation(line: 13, column: 3, scope: !9)
// CHECK:STDOUT: !12 = !DILocation(line: 12, column: 1, scope: !9)
// CHECK:STDOUT: !13 = distinct !DISubprogram(name: "F", linkageName: "_CF.FinalFn.Main.e342d6e8ff5d87ae", scope: null, file: !10, line: 5, type: !5, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !14 = !DILocation(line: 5, column: 3, scope: !13)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'final_method.carbon'
// CHECK:STDOUT: source_filename = "final_method.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @C.val = internal constant {} zeroinitializer
// CHECK:STDOUT: @C.val.loc10_15.6 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc10_15.2.temp = alloca {}, align 1, !dbg !7
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc10_15.2.temp), !dbg !7
// CHECK:STDOUT:   call void @_CCallMethod.Main.96e0ef8a617a6362(ptr @C.val.loc10_15.6), !dbg !8
// CHECK:STDOUT:   call void @"_COp.807eb4f2d3ed7e54:core.Destroy.Core"(ptr @C.val), !dbg !7
// CHECK:STDOUT:   ret void, !dbg !9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.807eb4f2d3ed7e54:core.Destroy.Core"(ptr %self) #0 !dbg !10 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CCallMethod.Main.96e0ef8a617a6362(ptr %t) #0 !dbg !17 {
// CHECK:STDOUT:   call void @_CM.FinalMethod.Main.96e0ef8a617a6362(ptr %t), !dbg !21
// CHECK:STDOUT:   ret void, !dbg !22
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CM.FinalMethod.Main.96e0ef8a617a6362(ptr %self) #0 !dbg !23 {
// CHECK:STDOUT:   ret void, !dbg !26
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!0}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !1 = !DIFile(filename: "final_method.carbon", directory: "")
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !1, line: 9, type: !5, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{null}
// CHECK:STDOUT: !7 = !DILocation(line: 10, column: 14, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 11, column: 3, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 9, column: 1, scope: !4)
// CHECK:STDOUT: !10 = distinct !DISubprogram(name: "Op", linkageName: "_COp.807eb4f2d3ed7e54:core.Destroy.Core", scope: null, file: !1, line: 10, type: !11, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !14)
// CHECK:STDOUT: !11 = !DISubroutineType(types: !12)
// CHECK:STDOUT: !12 = !{null, !13}
// CHECK:STDOUT: !13 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !14 = !{!15}
// CHECK:STDOUT: !15 = !DILocalVariable(arg: 1, scope: !10, type: !13)
// CHECK:STDOUT: !16 = !DILocation(line: 10, column: 14, scope: !10)
// CHECK:STDOUT: !17 = distinct !DISubprogram(name: "CallMethod", linkageName: "_CCallMethod.Main.96e0ef8a617a6362", scope: null, file: !18, line: 16, type: !11, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !19)
// CHECK:STDOUT: !18 = !DIFile(filename: "final_interfaces.carbon", directory: "")
// CHECK:STDOUT: !19 = !{!20}
// CHECK:STDOUT: !20 = !DILocalVariable(arg: 1, scope: !17, type: !13)
// CHECK:STDOUT: !21 = !DILocation(line: 17, column: 3, scope: !17)
// CHECK:STDOUT: !22 = !DILocation(line: 16, column: 1, scope: !17)
// CHECK:STDOUT: !23 = distinct !DISubprogram(name: "M", linkageName: "_CM.FinalMethod.Main.96e0ef8a617a6362", scope: null, file: !18, line: 9, type: !11, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !24)
// CHECK:STDOUT: !24 = !{!25}
// CHECK:STDOUT: !25 = !DILocalVariable(arg: 1, scope: !23, type: !13)
// CHECK:STDOUT: !26 = !DILocation(line: 9, column: 3, scope: !23)
// CHECK:STDOUT:
