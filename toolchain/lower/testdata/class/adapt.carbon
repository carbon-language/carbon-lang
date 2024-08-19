// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/class/adapt.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/class/adapt.carbon

// --- adapt_class.carbon

library "adapt_class";

class PairOfInts {
  var a: i32;
  var b: i32;

  fn Make() -> Self {
    return {.a = 1, .b = 2};
  }
}

class PairAdapter {
  adapt PairOfInts;

  fn Make() -> Self {
    return PairOfInts.Make() as Self;
  }

  fn GetB[self: Self]() -> i32 {
    let pi: PairOfInts = self as PairOfInts;
    return pi.b;
  }
}

fn Use() -> i32 {
  var pa: PairAdapter = PairAdapter.Make();
  return pa.GetB();
}

// --- adapt_int.carbon

library "adapt_int";

class Int {
  adapt i32;
}

fn DoStuff(a: Int) -> Int {
  return a;
}

// CHECK:STDOUT: ; ModuleID = 'adapt_class.carbon'
// CHECK:STDOUT: source_filename = "adapt_class.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @struct.loc9_28 = internal constant { i32, i32 } { i32 1, i32 2 }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @Make(ptr sret({ i32, i32 }) %return) {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc9_27.2.a = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 0
// CHECK:STDOUT:   %.loc9_27.4.b = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 1
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %return, ptr align 4 @struct.loc9_28, i64 8, i1 false)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @Make.1(ptr sret({ i32, i32 }) %return) {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @Make(ptr %return)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @GetB(ptr %self) {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc22_14.1.b = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 1
// CHECK:STDOUT:   %.loc22_14.2 = load i32, ptr %.loc22_14.1.b, align 4
// CHECK:STDOUT:   ret i32 %.loc22_14.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @Use() {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %pa.var = alloca { i32, i32 }, align 8
// CHECK:STDOUT:   call void @Make.1(ptr %pa.var)
// CHECK:STDOUT:   %GetB.call = call i32 @GetB(ptr %pa.var)
// CHECK:STDOUT:   ret i32 %GetB.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg) #0
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "adapt_class.carbon", directory: "")
// CHECK:STDOUT: ; ModuleID = 'adapt_int.carbon'
// CHECK:STDOUT: source_filename = "adapt_int.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @DoStuff(i32 %a) {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "adapt_int.carbon", directory: "")
