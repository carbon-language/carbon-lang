// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/cpp_namespace.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/cpp_namespace.carbon

// --- header.h

// --- fail_duplicate_cpp_name.carbon

library "[[@TEST_NAME]]";

import Cpp library "header.h";

// CHECK:STDERR: fail_duplicate_cpp_name.carbon:[[@LINE+7]]:11: error: duplicate name `Cpp` being declared in the same scope [NameDeclDuplicate]
// CHECK:STDERR: namespace Cpp;
// CHECK:STDERR:           ^~~
// CHECK:STDERR: fail_duplicate_cpp_name.carbon:[[@LINE-5]]:1: note: name is previously declared here [NameDeclPrevious]
// CHECK:STDERR: import Cpp library "header.h";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
namespace Cpp;

// --- cpp_in_inner_namespace.carbon

library "[[@TEST_NAME]]";

import Cpp library "header.h";

//@dump-sem-ir-begin
namespace N;
namespace N.Cpp;
//@dump-sem-ir-end

// --- alias.carbon

library "[[@TEST_NAME]]";

import Cpp library "header.h";

//@dump-sem-ir-begin
alias MyCpp = Cpp;
//@dump-sem-ir-end

// --- fail_add_name_to_cpp_namespace.carbon

library "[[@TEST_NAME]]";

import Cpp library "header.h";

// CHECK:STDERR: fail_add_name_to_cpp_namespace.carbon:[[@LINE+7]]:7: error: imported packages cannot be used for declarations [QualifiedDeclOutsidePackage]
// CHECK:STDERR: class Cpp.C {};
// CHECK:STDERR:       ^~~
// CHECK:STDERR: fail_add_name_to_cpp_namespace.carbon:[[@LINE-5]]:1: note: package imported here [QualifiedDeclOutsidePackageSource]
// CHECK:STDERR: import Cpp library "header.h";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
class Cpp.C {};

// --- api_and_impl.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
import Cpp library "header.h";
//@dump-sem-ir-end

// --- api_and_impl.impl.carbon

impl library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
import Cpp library "header.h";
//@dump-sem-ir-end


// --- /usr/include/system_header.h

auto system_function() -> void;

// --- system_header.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
import Cpp library "<system_header.h>";
//@dump-sem-ir-end

fn F() {
//@dump-sem-ir-begin
  Cpp.system_function();
//@dump-sem-ir-end
}

// CHECK:STDOUT: --- cpp_in_inner_namespace.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %N: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Cpp = %Cpp
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp: <namespace> = namespace [concrete] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- alias.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %MyCpp: <namespace> = bind_alias MyCpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- api_and_impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Cpp.import_cpp = import_cpp {
// CHECK:STDOUT:     import Cpp "header.h"
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- api_and_impl.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- system_header.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %.98d: type = cpp_overload_set_type @<null name> [concrete]
// CHECK:STDOUT:   %.769: %.98d = cpp_overload_set_value @<null name> [concrete]
// CHECK:STDOUT:   %system_function.type: type = fn_type @system_function [concrete]
// CHECK:STDOUT:   %system_function: %system_function.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .system_function = %.769
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.769: %.98d = cpp_overload_set_value @<null name> [concrete = constants.%.769]
// CHECK:STDOUT:   %system_function.decl: %system_function.type = fn_decl @system_function [concrete = constants.%system_function] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Cpp.import_cpp = import_cpp {
// CHECK:STDOUT:     import Cpp "<system_header.h>"
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %system_function.ref: %.98d = name_ref system_function, imports.%.769 [concrete = constants.%.769]
// CHECK:STDOUT:   %system_function.call: init %empty_tuple.type = call imports.%system_function.decl()
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
