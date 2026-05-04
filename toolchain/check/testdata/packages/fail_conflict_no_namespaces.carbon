// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/packages/fail_conflict_no_namespaces.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/packages/fail_conflict_no_namespaces.carbon

// --- fn.carbon

package Example library "[[@TEST_NAME]]";

fn Foo();

// --- var.carbon

package Example library "[[@TEST_NAME]]";

var Foo: i32;

// --- fail_conflict.carbon

package Example library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_conflict.carbon:[[@LINE+9]]:1: in import [InImport]
// CHECK:STDERR: var.carbon:4:5: error: duplicate name `Foo` being declared in the same scope [NameDeclDuplicate]
// CHECK:STDERR: var Foo: i32;
// CHECK:STDERR:     ^~~~~~~~
// CHECK:STDERR: fail_conflict.carbon:[[@LINE+5]]:1: in import [InImport]
// CHECK:STDERR: fn.carbon:4:1: note: name is previously declared here [NameDeclPrevious]
// CHECK:STDERR: fn Foo();
// CHECK:STDERR: ^~~~~~~~~
// CHECK:STDERR:
import library "fn";
import library "var";

// CHECK:STDOUT: --- fn.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Foo.type: type = fn_type @Foo [concrete]
// CHECK:STDOUT:   %Foo: %Foo.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Foo = %Foo.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Foo.decl: %Foo.type = fn_decl @Foo [concrete = constants.%Foo] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Foo();
// CHECK:STDOUT:
// CHECK:STDOUT: --- var.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.type: type = facet_type <@DefaultOrUnformed> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.type.462: type = fn_type @T.as.DefaultOrUnformed.impl.Op, @T.as.DefaultOrUnformed.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.b80: %T.as.DefaultOrUnformed.impl.Op.type.462 = struct_value () [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness.89f: <witness> = impl_witness imports.%DefaultOrUnformed.impl_witness_table.725, @T.as.DefaultOrUnformed.impl(%i32) [concrete]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.type.2b0: type = fn_type @T.as.DefaultOrUnformed.impl.Op, @T.as.DefaultOrUnformed.impl(%i32) [concrete]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.3bd: %T.as.DefaultOrUnformed.impl.Op.type.2b0 = struct_value () [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.facet: %DefaultOrUnformed.type = facet_value %i32, (%DefaultOrUnformed.impl_witness.89f) [concrete]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.specific_fn: <specific function> = specific_function %T.as.DefaultOrUnformed.impl.Op.3bd, @T.as.DefaultOrUnformed.impl.Op(%i32) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .DefaultOrUnformed = %Core.DefaultOrUnformed
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.DefaultOrUnformed: type = import_ref Core//prelude/parts/default, DefaultOrUnformed, loaded [concrete = constants.%DefaultOrUnformed.type]
// CHECK:STDOUT:   %Core.import_ref.dba: @T.as.DefaultOrUnformed.impl.%T.as.DefaultOrUnformed.impl.Op.type (%T.as.DefaultOrUnformed.impl.Op.type.462) = import_ref Core//prelude/parts/default, loc{{\d+_\d+}}, loaded [symbolic = @T.as.DefaultOrUnformed.impl.%T.as.DefaultOrUnformed.impl.Op (constants.%T.as.DefaultOrUnformed.impl.Op.b80)]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness_table.725 = impl_witness_table (%Core.import_ref.dba), @T.as.DefaultOrUnformed.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Foo = %Foo
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %Foo.patt: %pattern_type.7ce = ref_binding_pattern Foo [concrete]
// CHECK:STDOUT:     %Foo.var_patt: %pattern_type.7ce = var_pattern %Foo.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Foo.var: ref %i32 = var %Foo.var_patt [concrete]
// CHECK:STDOUT:   %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %Foo: ref %i32 = ref_binding Foo, %Foo.var [concrete = %Foo.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %DefaultOrUnformed.facet: %DefaultOrUnformed.type = facet_value constants.%i32, (constants.%DefaultOrUnformed.impl_witness.89f) [concrete = constants.%DefaultOrUnformed.facet]
// CHECK:STDOUT:   %.loc4_13.1: %DefaultOrUnformed.type = converted constants.%i32, %DefaultOrUnformed.facet [concrete = constants.%DefaultOrUnformed.facet]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc4_13.1 [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc4_13.2: type = converted %.loc4_13.1, %as_type [concrete = constants.%i32]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.specific_fn: <specific function> = specific_function constants.%T.as.DefaultOrUnformed.impl.Op.3bd, @T.as.DefaultOrUnformed.impl.Op(constants.%i32) [concrete = constants.%T.as.DefaultOrUnformed.impl.Op.specific_fn]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.call: init %i32 = call %T.as.DefaultOrUnformed.impl.Op.specific_fn()
// CHECK:STDOUT:   assign file.%Foo.var, %T.as.DefaultOrUnformed.impl.Op.call
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_conflict.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Example.Foo = import_ref Example//fn, Foo, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Foo = imports.%Example.Foo
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT: }
// CHECK:STDOUT:
