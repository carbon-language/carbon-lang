// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/where_assignment.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/where_assignment.carbon

// --- basic.carbon
library "[[@TEST_NAME]]";

interface Z {
  let Z1: type;
}

//@dump-sem-ir-begin
impl () as Z {
  where Z1 = type;
}
//@dump-sem-ir-end

// --- basic_in_class.carbon
library "[[@TEST_NAME]]";

interface Z {
  let Z1: type;
}

class C {
  impl as Z {
    where Z1 = type;
  }
}

// --- convert.carbon
library "[[@TEST_NAME]]";

class B {}
class C {
  impl as Core.ImplicitAs(B) {
    eval fn Convert(unused self) -> B { return {} as B; }
  }
}

interface Z {
  let Z1: B;
}


impl () as Z {
  where Z1 = ({} as C);
}

// --- generic.carbon
library "[[@TEST_NAME]]";

interface Z(T: type) {
  let Z1: type;
}

//@dump-sem-ir-begin
impl () as Z(()) {
  where Z1 = type;
}
//@dump-sem-ir-end

// --- fail_todo_assoc_const_refers_to_previous.carbon
library "[[@TEST_NAME]]";

class C(T: type) {}
interface Z {
  let Z1: type;
  // TODO: It's impossible to form the interface's specific to get
  // the type of Z1 without also resolving the type of Z2 which accesses
  // Z1 before its set. Each assoc constant needs its own generic to
  // allow each one to be resolved independently.
  let Z2: C(Z1);
}

impl () as Z {
  // CHECK:STDERR:        ╭─┤ fail_todo_assoc_const_refers_to_previous.carbon:[[@LINE+11]]:3
  // CHECK:STDERR:        │   let Z2: C(Z1);
  // CHECK:STDERR:        ·             ─┬
  // CHECK:STDERR:        ·              ╰──┤ accessing member from impl before it has a defined value
  // CHECK:STDERR:        ·                 │ [ImplAccessMemberBeforeSet]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR: ->     │   where Z1 = {};
  // CHECK:STDERR:        ·   ━━━━━━━┳━━━━━━
  // CHECK:STDERR:        ·          ┗━━┫ error: unable to monomorphize specific `Z` [ResolvingSpecificHere]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  where Z1 = {};
  where Z2 = ({} as C({}));
}

// --- symbolic_type.carbon
library "[[@TEST_NAME]]";

interface Z {
  let Z1: Self;
}

impl () as Z {
  where Z1 = ();
}

// --- symbolic_impl_self_type.carbon
library "[[@TEST_NAME]]";

interface Y {}
class C(T: Y) { adapt {}; }

interface Z(T: type) {
  let Z1: T;
}

impl forall [T: Y] T as Z(C(T)) {
  where Z1 = ({} as C(T));
}

// --- fail_invalid_scope.carbon
library "[[@TEST_NAME]]";

let Z1: type = type;
// CHECK:STDERR:        ╭─┤ fail_invalid_scope.carbon:[[@LINE+8]]:1
// CHECK:STDERR: ->     │ where Z1 = type;
// CHECK:STDERR:        · ┳━━━━
// CHECK:STDERR:        · ┗━━┫ error: `where` assignment declaration found in invalid scope; must be in
// CHECK:STDERR:        ·    ┃        `impl` definition
// CHECK:STDERR:        ·    ┃        [WhereAssignmentInWrongScope]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
where Z1 = type;

// --- fail_fn_invalid_scope.carbon
library "[[@TEST_NAME]]";

let Z1: type = type;
fn F() {
  // CHECK:STDERR:        ╭─┤ fail_fn_invalid_scope.carbon:[[@LINE+8]]:3
  // CHECK:STDERR: ->     │   where Z1 = type;
  // CHECK:STDERR:        ·   ┳━━━━
  // CHECK:STDERR:        ·   ┗━━┫ error: `where` assignment declaration found in invalid scope; must be in
  // CHECK:STDERR:        ·      ┃        `impl` definition
  // CHECK:STDERR:        ·      ┃        [WhereAssignmentInWrongScope]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  where Z1 = type;
}

// --- fail_region_invalid_scope.carbon
library "[[@TEST_NAME]]";

let Z1: type = type;
fn F() {
  // If blocks have a scope that has no declaration scope inst associated with
  // them.
  if (true) {
    // CHECK:STDERR:        ╭─┤ fail_region_invalid_scope.carbon:[[@LINE+8]]:5
    // CHECK:STDERR: ->     │     where Z1 = type;
    // CHECK:STDERR:        ·     ┳━━━━
    // CHECK:STDERR:        ·     ┗━━┫ error: `where` assignment declaration found in invalid scope; must be
    // CHECK:STDERR:        ·        ┃        in `impl` definition
    // CHECK:STDERR:        ·        ┃        [WhereAssignmentInWrongScope]
    // CHECK:STDERR:        ╰────
    // CHECK:STDERR:
    where Z1 = type;
  }
}

// --- fail_class_invalid_scope.carbon
library "[[@TEST_NAME]]";

class C {
  let Z1: type = type;
  // CHECK:STDERR:        ╭─┤ fail_class_invalid_scope.carbon:[[@LINE+8]]:3
  // CHECK:STDERR: ->     │   where Z1 = type;
  // CHECK:STDERR:        ·   ┳━━━━
  // CHECK:STDERR:        ·   ┗━━┫ error: `where` assignment declaration found in invalid scope; must be in
  // CHECK:STDERR:        ·      ┃        `impl` definition
  // CHECK:STDERR:        ·      ┃        [WhereAssignmentInWrongScope]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  where Z1 = type;
}

// --- fail_name_not_assoc_const.carbon
library "[[@TEST_NAME]]";

let Y: type = type;

interface Z {
  let Z1: type;
}

impl () as Z {
  // CHECK:STDERR:        ╭─┤ fail_name_not_assoc_const.carbon:[[@LINE+7]]:3
  // CHECK:STDERR: ->     │   where Y = type;
  // CHECK:STDERR:        ·   ━━━━━━━┳━━━━━━━
  // CHECK:STDERR:        ·          ┗━━┫ error: member name `Y` not found in `Z`
  // CHECK:STDERR:        ·             ┃        [WhereAssignmentNameNotFound]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  where Y = type;
  where Z1 = type;
}

// --- fail_incomplete.carbon
library "[[@TEST_NAME]]";

interface Z;

// CHECK:STDERR:        ╭─┤ fail_incomplete.carbon:[[@LINE+11]]:1
// CHECK:STDERR:        │ interface Z;
// CHECK:STDERR:        · ──────┬─────
// CHECK:STDERR:        ·       ╰──┤ interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR:        ┆
// CHECK:STDERR: ->     │ impl () as Z {
// CHECK:STDERR:        · ━━━━━━━┳━━━━━━
// CHECK:STDERR:        ·        ┗━━┫ error: definition of impl as incomplete facet type `Z`
// CHECK:STDERR:        ·           ┃        [ImplAsIncompleteFacetTypeDefinition]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
impl () as Z {
  where Z1 = type;
}

// --- fail_runtime.carbon
library "[[@TEST_NAME]]";

var Runtime: type = type;

interface Z {
  let Z1: type;
}

impl () as Z {
  // CHECK:STDERR:        ╭─┤ fail_runtime.carbon:[[@LINE+8]]:3
  // CHECK:STDERR: ->     │   where Z1 = Runtime;
  // CHECK:STDERR:        ·   ┳━━━━━━━━━━━━━━━━━━
  // CHECK:STDERR:        ·   ┗━━┫ error: associated constant Z1 given value `Runtime` that is not constant
  // CHECK:STDERR:        ·      ┃        after conversion to `type`
  // CHECK:STDERR:        ·      ┃        [AssociatedConstantNotConstantAfterConversion]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  where Z1 = Runtime;
}

// --- fail_runtime_fn.carbon
library "[[@TEST_NAME]]";

fn Runtime() -> type;

interface Z {
  let Z1: type;
}

impl () as Z {
  // CHECK:STDERR:        ╭─┤ fail_runtime_fn.carbon:[[@LINE+8]]:3
  // CHECK:STDERR: ->     │   where Z1 = Runtime();
  // CHECK:STDERR:        ·   ┳━━━━━━━━━━━━━━━━━━━━
  // CHECK:STDERR:        ·   ┗━━┫ error: associated constant Z1 given value `Runtime()` that is not
  // CHECK:STDERR:        ·      ┃        constant after conversion to `type`
  // CHECK:STDERR:        ·      ┃        [AssociatedConstantNotConstantAfterConversion]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  where Z1 = Runtime();
}

// --- same_value.carbon
library "[[@TEST_NAME]]";

interface Z {
  let Z1: type;
}

impl () as Z {
  where Z1 = type;
  where Z1 = type;
}

// --- same_value_as_decl.carbon
library "[[@TEST_NAME]]";

interface Z {
  let Z1: type;
}

impl () as Z where .Z1 = type {
  where Z1 = type;
}

// --- fail_different_value.carbon
library "[[@TEST_NAME]]";

interface Z {
  let Z1: type;
}

impl () as Z {
  where Z1 = type;
  // CHECK:STDERR:        ╭─┤ fail_different_value.carbon:[[@LINE+8]]:3
  // CHECK:STDERR: ->     │   where Z1 = {};
  // CHECK:STDERR:        ·   ┳━━━━━━━━━━━━━
  // CHECK:STDERR:        ·   ┗━━┫ error: associated constant Z1 given value `{}` that is different from
  // CHECK:STDERR:        ·      ┃        previously given value `type`
  // CHECK:STDERR:        ·      ┃        [AssociatedConstantGivenDifferentValues]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  where Z1 = {};
}

// --- fail_different_value_from_decl.carbon
library "[[@TEST_NAME]]";

interface Z {
  let Z1: type;
}

impl () as Z where .Z1 = type {
  // CHECK:STDERR:        ╭─┤ fail_different_value_from_decl.carbon:[[@LINE+8]]:3
  // CHECK:STDERR: ->     │   where Z1 = {};
  // CHECK:STDERR:        ·   ┳━━━━━━━━━━━━━
  // CHECK:STDERR:        ·   ┗━━┫ error: associated constant Z1 given value `{}` that is different from
  // CHECK:STDERR:        ·      ┃        previously given value `type`
  // CHECK:STDERR:        ·      ┃        [AssociatedConstantGivenDifferentValues]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  where Z1 = {};
}

// --- fail_assign_fn.carbon
library "[[@TEST_NAME]]";

interface Z {
  fn F();
}

impl () as Z {
  // CHECK:STDERR:        ╭─┤ fail_assign_fn.carbon:[[@LINE+11]]:13
  // CHECK:STDERR: ->     │   where F = fn()=>();
  // CHECK:STDERR:        ·             ┳━━━━━
  // CHECK:STDERR:        ·             ┗━━┫ error: cannot implicitly convert expression of type
  // CHECK:STDERR:        ·             │  ┃        `<type of function expression>` to `<type of F>`
  // CHECK:STDERR:        ·             │  ┃        [ConversionFailure]
  // CHECK:STDERR:        ·             ╰────┤ type `<type of function expression>` does not implement
  // CHECK:STDERR:        ·                  │ interface `Core.ImplicitAs(<type of F>)`
  // CHECK:STDERR:        ·                  │ [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  where F = fn()=>();

  fn F() {}
}

// --- fail_set_through_extend_require.carbon
library "[[@TEST_NAME]]";

interface Y {
  let Y1: type;
}

interface Z {
  extend require impls Y;
}

impl () as Y where .Y1 = {} {}

impl () as Z {
  // CHECK:STDERR:        ╭─┤ fail_set_through_extend_require.carbon:[[@LINE+7]]:3
  // CHECK:STDERR: ->     │   where Y1 = ();
  // CHECK:STDERR:        ·   ━━━━━━━┳━━━━━━
  // CHECK:STDERR:        ·          ┗━━┫ error: member name `Y1` not found in `Z`
  // CHECK:STDERR:        ·             ┃        [WhereAssignmentNameNotFound]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  where Y1 = ();
}

// CHECK:STDOUT: --- basic.carbon = $basic
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %.78e: <witness> = impl_self_witness %empty_tuple.type, @Z [concrete]
// CHECK:STDOUT:   %Z.impl_witness: <witness> = impl_witness @empty_tuple.type.as.Z.impl.%Z.impl_witness_table [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   impl_decl @empty_tuple.type.as.Z.impl [concrete] {} {
// CHECK:STDOUT:     %.loc8_7.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc8_7.2: type = converted %.loc8_7.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %Z.ref: type = name_ref Z, file.%Z.decl [concrete = constants.%Z.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8: <witness> = impl_self_witness @empty_tuple.type.as.Z.impl.%.loc8_7.2, @Z [concrete = constants.%.78e]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @empty_tuple.type.as.Z.impl: %.loc8_7.2 as %Z.ref {
// CHECK:STDOUT:   %.loc9: type = type_literal type [concrete = type]
// CHECK:STDOUT:   %impl_witness_assoc_constant: type = impl_witness_assoc_constant type [concrete = type]
// CHECK:STDOUT:   %Z.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant), @empty_tuple.type.as.Z.impl [concrete]
// CHECK:STDOUT:   %Z.impl_witness: <witness> = impl_witness %Z.impl_witness_table [concrete = constants.%Z.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   extend %Z.ref
// CHECK:STDOUT:   witness = %Z.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- generic.carbon = $generic
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Z.type.c39: type = generic_interface_type @Z [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Z.generic: %Z.type.c39 = struct_value () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %Z.type.cf0: type = facet_type <@Z, @Z(%empty_tuple.type)> [concrete]
// CHECK:STDOUT:   %.87d: <witness> = impl_self_witness %empty_tuple.type, @Z, @Z(%empty_tuple.type) [concrete]
// CHECK:STDOUT:   %Z.impl_witness: <witness> = impl_witness @empty_tuple.type.as.Z.impl.%Z.impl_witness_table [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   impl_decl @empty_tuple.type.as.Z.impl [concrete] {} {
// CHECK:STDOUT:     %.loc8_7.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc8_7.2: type = converted %.loc8_7.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %Z.ref: %Z.type.c39 = name_ref Z, file.%Z.decl [concrete = constants.%Z.generic]
// CHECK:STDOUT:     %.loc8_15: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc8_16: type = converted %.loc8_15, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %Z.type: type = facet_type <@Z, @Z(constants.%empty_tuple.type)> [concrete = constants.%Z.type.cf0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8: <witness> = impl_self_witness @empty_tuple.type.as.Z.impl.%.loc8_7.2, @Z, @Z(constants.%empty_tuple.type) [concrete = constants.%.87d]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @empty_tuple.type.as.Z.impl: %.loc8_7.2 as %Z.type {
// CHECK:STDOUT:   %.loc9: type = type_literal type [concrete = type]
// CHECK:STDOUT:   %impl_witness_assoc_constant: type = impl_witness_assoc_constant type [concrete = type]
// CHECK:STDOUT:   %Z.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant), @empty_tuple.type.as.Z.impl [concrete]
// CHECK:STDOUT:   %Z.impl_witness: <witness> = impl_witness %Z.impl_witness_table [concrete = constants.%Z.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   extend %Z.type
// CHECK:STDOUT:   witness = %Z.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
