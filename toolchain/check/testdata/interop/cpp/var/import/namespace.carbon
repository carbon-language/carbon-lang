// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/var/import/namespace.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/var/import/namespace.carbon

// --- namespace.h

namespace N {
class C {};
C global;
}  // namespace N

// --- import_namespace.carbon

library "[[@TEST_NAME]]";

import Cpp library "namespace.h";

fn MyF() {
  let unused local: Cpp.N.C = Cpp.N.global;
}

// --- type_in_namespace.carbon

import Cpp inline '''
namespace N {
class C {};
}

extern N::C c;
''';

//@dump-sem-ir-begin
fn F() {
  // Import `::c` and its type `N::C` without having ever mentioned `N`.
  // This used to cause a crash.
  Cpp.c;
}
//@dump-sem-ir-end

// CHECK:STDOUT: --- type_in_namespace.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %c.patt: %pattern_type = ref_binding_pattern c [concrete]
// CHECK:STDOUT:   %c.var_patt: %pattern_type = var_pattern %c.patt [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .c = %c.var
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.patt: %pattern_type = ref_binding_pattern c [concrete = constants.%c.patt]
// CHECK:STDOUT:   %c.var_patt: %pattern_type = var_pattern %c.patt [concrete = constants.%c.var_patt]
// CHECK:STDOUT:   %c.var: ref %C = var %c.var_patt [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %c.ref: ref %C = name_ref c, imports.%c.var [concrete = imports.%c.var]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
