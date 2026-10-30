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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/pointer/fail_deref_not_pointer.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/pointer/fail_deref_not_pointer.carbon

fn Deref(n: i32) {
  // CHECK:STDERR:        ╭─┤ fail_deref_not_pointer.carbon:[[@LINE+7]]:3
  // CHECK:STDERR: ->     │   *n;
  // CHECK:STDERR:        ·   ┳
  // CHECK:STDERR:        ·   ┗━━┫ error: cannot dereference operand of non-pointer type `i32`
  // CHECK:STDERR:        ·      ┃        [DerefOfNonPointer]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  *n;
  // CHECK:STDERR:        ╭─┤ fail_deref_not_pointer.carbon:[[@LINE+7]]:4
  // CHECK:STDERR: ->     │   n->foo;
  // CHECK:STDERR:        ·    ━┳
  // CHECK:STDERR:        ·     ┗━━┫ error: cannot apply `->` operator to non-pointer type `i32`
  // CHECK:STDERR:        ·        ┃        [ArrowOperatorOfNonPointer]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  n->foo;
  // CHECK:STDERR:        ╭─┤ fail_deref_not_pointer.carbon:[[@LINE+7]]:3
  // CHECK:STDERR: ->     │   *();
  // CHECK:STDERR:        ·   ┳
  // CHECK:STDERR:        ·   ┗━━┫ error: cannot dereference operand of non-pointer type `()`
  // CHECK:STDERR:        ·      ┃        [DerefOfNonPointer]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  *();
  // CHECK:STDERR:        ╭─┤ fail_deref_not_pointer.carbon:[[@LINE+7]]:5
  // CHECK:STDERR: ->     │   ()->foo;
  // CHECK:STDERR:        ·     ━┳
  // CHECK:STDERR:        ·      ┗━━┫ error: cannot apply `->` operator to non-pointer type `()`
  // CHECK:STDERR:        ·         ┃        [ArrowOperatorOfNonPointer]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  ()->foo;
  // CHECK:STDERR:        ╭─┤ fail_deref_not_pointer.carbon:[[@LINE+7]]:3
  // CHECK:STDERR: ->     │   *{};
  // CHECK:STDERR:        ·   ┳
  // CHECK:STDERR:        ·   ┗━━┫ error: cannot dereference operand of non-pointer type `{}`
  // CHECK:STDERR:        ·      ┃        [DerefOfNonPointer]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  *{};
  // CHECK:STDERR:        ╭─┤ fail_deref_not_pointer.carbon:[[@LINE+7]]:5
  // CHECK:STDERR: ->     │   {}->foo;
  // CHECK:STDERR:        ·     ━┳
  // CHECK:STDERR:        ·      ┗━━┫ error: cannot apply `->` operator to non-pointer type `{}`
  // CHECK:STDERR:        ·         ┃        [ArrowOperatorOfNonPointer]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  {}->foo;
}

// CHECK:STDOUT: --- fail_deref_not_pointer.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = parameterized_class_name_type @Int [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.d94: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %n.param_patt: %pattern_type.d94 = value_param_pattern [concrete]
// CHECK:STDOUT:   %n.patt: %pattern_type.d94 = wrapper_binding_pattern n, %n.param_patt [concrete]
// CHECK:STDOUT:   %Deref.type: type = fn_type @Deref [concrete]
// CHECK:STDOUT:   %Deref: %Deref.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Deref = %Deref.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Deref.decl: %Deref.type = fn_decl @Deref [concrete = constants.%Deref] {
// CHECK:STDOUT:     %n.param_patt: %pattern_type.d94 = value_param_pattern [concrete = constants.%n.param_patt]
// CHECK:STDOUT:     %n.patt: %pattern_type.d94 = wrapper_binding_pattern n, %n.param_patt [concrete = constants.%n.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %n.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %n: %i32 = wrapper_binding n, %n.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Deref(%n.param: %i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref.loc23: %i32 = name_ref n, %n
// CHECK:STDOUT:   %.loc23: ref <error> = deref %n.ref.loc23
// CHECK:STDOUT:   %n.ref.loc31: %i32 = name_ref n, %n
// CHECK:STDOUT:   %.loc31: ref <error> = deref %n.ref.loc31
// CHECK:STDOUT:   %foo.ref.loc31: <error> = name_ref foo, <error> [concrete = <error>]
// CHECK:STDOUT:   %.loc39_5.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %empty_tuple.loc39: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc39_5.2: %empty_tuple.type = converted %.loc39_5.1, %empty_tuple.loc39 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc39_3: ref <error> = deref %.loc39_5.2 [concrete = <error>]
// CHECK:STDOUT:   %.loc47_4.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %empty_tuple.loc47: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc47_4.2: %empty_tuple.type = converted %.loc47_4.1, %empty_tuple.loc47 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc47_5: ref <error> = deref %.loc47_4.2 [concrete = <error>]
// CHECK:STDOUT:   %foo.ref.loc47: <error> = name_ref foo, <error> [concrete = <error>]
// CHECK:STDOUT:   %.loc55_5.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %empty_struct.loc55: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc55_5.2: %empty_struct_type = converted %.loc55_5.1, %empty_struct.loc55 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc55_3: ref <error> = deref %.loc55_5.2 [concrete = <error>]
// CHECK:STDOUT:   %.loc63_4.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %empty_struct.loc63: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc63_4.2: %empty_struct_type = converted %.loc63_4.1, %empty_struct.loc63 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc63_5: ref <error> = deref %.loc63_4.2 [concrete = <error>]
// CHECK:STDOUT:   %foo.ref.loc63: <error> = name_ref foo, <error> [concrete = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
