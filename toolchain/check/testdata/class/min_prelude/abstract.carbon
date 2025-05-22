// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --dump-sem-ir-ranges=only
// INCLUDE-FILE: toolchain/testing/min_prelude/destroy.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/min_prelude/abstract.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/min_prelude/abstract.carbon

// --- fail_abstract_field.carbon

library "[[@TEST_NAME]]";

abstract class Abstract {
}

class Contains {
  // CHECK:STDERR: fail_abstract_field.carbon:[[@LINE+7]]:10: error: field has abstract type `Abstract` [AbstractTypeInFieldDecl]
  // CHECK:STDERR:   var a: Abstract;
  // CHECK:STDERR:          ^~~~~~~~
  // CHECK:STDERR: fail_abstract_field.carbon:[[@LINE-7]]:1: note: class was declared abstract here [ClassAbstractHere]
  // CHECK:STDERR: abstract class Abstract {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  var a: Abstract;
}

// --- fail_abstract_var.carbon

library "[[@TEST_NAME]]";

abstract class Abstract {
}

fn Var() {
  // CHECK:STDERR: fail_abstract_var.carbon:[[@LINE+7]]:10: error: binding pattern has abstract type `Abstract` in `var` pattern [AbstractTypeInVarPattern]
  // CHECK:STDERR:   var v: Abstract;
  // CHECK:STDERR:          ^~~~~~~~
  // CHECK:STDERR: fail_abstract_var.carbon:[[@LINE-7]]:1: note: class was declared abstract here [ClassAbstractHere]
  // CHECK:STDERR: abstract class Abstract {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  var v: Abstract;
}

// --- abstract_let.carbon

library "[[@TEST_NAME]]";

abstract class Abstract {
}

fn F(a: Abstract) {
  //@dump-sem-ir-begin
  let l: Abstract = a;
  //@dump-sem-ir-end
}

// --- fail_abstract_adapter.carbon

library "[[@TEST_NAME]]";

abstract class Abstract {
}

class Adapter {
  // TODO(#4387): This should probably be valid
  // CHECK:STDERR: fail_abstract_adapter.carbon:[[@LINE+7]]:3: error: adapted type `Abstract` is an abstract type [AbstractTypeInAdaptDecl]
  // CHECK:STDERR:   adapt Abstract;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_abstract_adapter.carbon:[[@LINE-8]]:1: note: class was declared abstract here [ClassAbstractHere]
  // CHECK:STDERR: abstract class Abstract {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  adapt Abstract;
}

// --- define_and_call_abstract_param.carbon

library "[[@TEST_NAME]]";

abstract class Abstract {
}

fn Param(a: Abstract);

fn Call(p: Abstract) {
  //@dump-sem-ir-begin
  Param(p);
  //@dump-sem-ir-end
}

// --- fail_todo_return_nonabstract_derived.carbon

library "[[@TEST_NAME]]";

abstract class Abstract {
}

class Derived {
  extend base: Abstract;

  var d: {};
}

fn Make() -> Derived {
  //@dump-sem-ir-begin
  // TODO: This should be valid, and should construct an instance of `partial Abstract` as the base.
  // CHECK:STDERR: fail_todo_return_nonabstract_derived.carbon:[[@LINE+7]]:10: error: initialization of abstract type `Abstract` [AbstractTypeInInit]
  // CHECK:STDERR:   return {.base = {}, .d = {}};
  // CHECK:STDERR:          ^~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_return_nonabstract_derived.carbon:[[@LINE-15]]:1: note: class was declared abstract here [ClassAbstractHere]
  // CHECK:STDERR: abstract class Abstract {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  return {.base = {}, .d = {}};
  //@dump-sem-ir-end
}

// --- fail_return_abstract.carbon

library "[[@TEST_NAME]]";

abstract class Abstract {
}

class Derived {
  extend base: Abstract;

  var d: {};
}

fn Return(a: Abstract) -> Abstract {
  // TODO: Seems like this would be better off failing with "function returns abstract type" here instead of this \/
  // CHECK:STDERR: fail_return_abstract.carbon:[[@LINE+7]]:3: error: initialization of abstract type `Abstract` [AbstractTypeInInit]
  // CHECK:STDERR:   return a;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR: fail_return_abstract.carbon:[[@LINE-14]]:1: note: class was declared abstract here [ClassAbstractHere]
  // CHECK:STDERR: abstract class Abstract {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  return a;
}

// --- access_abstract_subobject.carbon

library "[[@TEST_NAME]]";

abstract class Abstract {
  var a: {};
}

class Derived {
  extend base: Abstract;

  var d: {};
}

fn Access(d: Derived) -> {} {
  //@dump-sem-ir-begin
  return d.base.a;
  //@dump-sem-ir-end
}

// --- todo_fail_abstract_let_temporary.carbon

library "[[@TEST_NAME]]";

abstract class Abstract {
}

fn F() {
  // TODO: This should probably require `partial`.
  let l: Abstract = {};
}

// --- fail_call_abstract_return.carbon

library "[[@TEST_NAME]]";

abstract class Abstract {
}

fn ReturnAbstract() -> Abstract;

fn CallReturnAbstract() {
  // CHECK:STDERR: fail_call_abstract_return.carbon:[[@LINE+10]]:3: error: function returns abstract type `Abstract` [AbstractTypeInFunctionReturnType]
  // CHECK:STDERR:   ReturnAbstract();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_call_abstract_return.carbon:[[@LINE-9]]:1: note: class was declared abstract here [ClassAbstractHere]
  // CHECK:STDERR: abstract class Abstract {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_call_abstract_return.carbon:[[@LINE-9]]:21: note: return type declared here [IncompleteReturnTypeHere]
  // CHECK:STDERR: fn ReturnAbstract() -> Abstract;
  // CHECK:STDERR:                     ^~~~~~~~~~~
  // CHECK:STDERR:
  ReturnAbstract();
}

// CHECK:STDOUT: --- abstract_let.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Abstract: type = class_type @Abstract [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Abstract [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%a.param: %Abstract) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %l.patt: %pattern_type = binding_pattern l [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.ref: %Abstract = name_ref a, %a
// CHECK:STDOUT:   %Abstract.ref.loc9: type = name_ref Abstract, file.%Abstract.decl [concrete = constants.%Abstract]
// CHECK:STDOUT:   %l: %Abstract = bind_name l, <error>
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- define_and_call_abstract_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Abstract: type = class_type @Abstract [concrete]
// CHECK:STDOUT:   %Param.type: type = fn_type @Param [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Param: %Param.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Call(%p.param: %Abstract) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Param.ref: %Param.type = name_ref Param, file.%Param.decl [concrete = constants.%Param]
// CHECK:STDOUT:   %p.ref: %Abstract = name_ref p, %p
// CHECK:STDOUT:   %Param.call: init %empty_tuple.type = call %Param.ref(<error>)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_return_nonabstract_derived.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %struct_type.base.d.e0f: type = struct_type {.base: %empty_struct_type, .d: %empty_struct_type} [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make() -> %return.param: %Derived {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc23_20: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc23_29: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc23_30: %struct_type.base.d.e0f = struct_literal (%.loc23_20, %.loc23_29)
// CHECK:STDOUT:   return <error> to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- access_abstract_subobject.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Abstract: type = class_type @Abstract [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %Derived.elem.513: type = unbound_element_type %Derived, %Abstract [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Access(%d.param: %Derived) -> %empty_struct_type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %d.ref: %Derived = name_ref d, %d
// CHECK:STDOUT:   %base.ref: %Derived.elem.513 = name_ref base, @Derived.%.loc9 [concrete = @Derived.%.loc9]
// CHECK:STDOUT:   %.loc16: ref %Abstract = class_element_access %d.ref, element0
// CHECK:STDOUT:   %a.ref: <error> = name_ref a, <error> [concrete = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
