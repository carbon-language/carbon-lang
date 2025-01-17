// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/declaration/no_prelude/name_poisoning.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/declaration/no_prelude/name_poisoning.carbon

// --- no_poison.carbon

library "[[@TEST_NAME]]";

class C {};

// Both N.F1 and N.F2 use N.C and not C.
namespace N;
class N.C {}
fn N.F1(x: C);
fn N.F2(x: C) { N.F1(x); }

// --- poison.carbon

library "[[@TEST_NAME]]";

class C {};

namespace N;
// Here we use C and poison N.C.
fn N.F1(x: C);

// --- fail_poison_class_without_usage.carbon
// CHECK:STDERR: fail_poison_class_without_usage.carbon: error: name used before it was declared [NameUseBeforeDecl]

library "[[@TEST_NAME]]";

class C {};

namespace N;
// Here we use C and poison N.C.
fn N.F1(x: C);

// Should fail here since C was poisoned for namespace N when it was used in N
// context without qualification.
// CHECK:STDERR: fail_poison_class_without_usage.carbon:[[@LINE+4]]:1: note: declared here [NameUseBeforeDeclNote]
// CHECK:STDERR: class N.C {}
// CHECK:STDERR: ^~~~~~~~~~~
// CHECK:STDERR:
class N.C {}

// --- fail_poison_interface_without_usage.carbon
// CHECK:STDERR: fail_poison_interface_without_usage.carbon: error: name used before it was declared [NameUseBeforeDecl]

library "[[@TEST_NAME]]";

interface I {};

namespace N;
// Here we use I and poison N.I.
fn N.F1(x: I);

// Should fail here since I was poisoned for namespace N when it was used in N
// context without qualification.
// CHECK:STDERR: fail_poison_interface_without_usage.carbon:[[@LINE+4]]:1: note: declared here [NameUseBeforeDeclNote]
// CHECK:STDERR: interface N.I {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~
// CHECK:STDERR:
interface N.I {}

// --- fail_poison_namespace_without_usage.carbon
// CHECK:STDERR: fail_poison_namespace_without_usage.carbon: error: name used before it was declared [NameUseBeforeDecl]

library "[[@TEST_NAME]]";

class C {};

namespace N;
// Here we use C and poison N.C.
fn N.F1(x: C);

// Should fail here since C was poisoned for namespace N when it was used in N
// context without qualification.
// CHECK:STDERR: fail_poison_namespace_without_usage.carbon:[[@LINE+4]]:1: note: declared here [NameUseBeforeDeclNote]
// CHECK:STDERR: namespace N.C;
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR:
namespace N.C;

// --- fail_poison_member_without_usage.carbon
// CHECK:STDERR: fail_poison_member_without_usage.carbon: error: name used before it was declared [NameUseBeforeDecl]

library "[[@TEST_NAME]]";

class C1 {};

class D {
  // Here we use C1 and poison D.C1.
  fn F1(x: C1);

  class C2 {};
  // Should fail here since C1 was poisoned for namespace class D when it was
  // used in D context without qualification.
  // CHECK:STDERR: fail_poison_member_without_usage.carbon:[[@LINE+4]]:7: note: declared here [NameUseBeforeDeclNote]
  // CHECK:STDERR:   var C1: C2;
  // CHECK:STDERR:       ^~~~~~
  // CHECK:STDERR:
  var C1: C2;
}

// --- fail_poison_function_without_usage.carbon
// CHECK:STDERR: fail_poison_function_without_usage.carbon: error: name used before it was declared [NameUseBeforeDecl]

library "[[@TEST_NAME]]";

class C {};

namespace N;
// Here we use C and poison N.C.
fn N.F1(x: C);

// Should fail here since C was poisoned for namespace N when it was used in N
// context without qualification.
// CHECK:STDERR: fail_poison_function_without_usage.carbon:[[@LINE+4]]:1: note: declared here [NameUseBeforeDeclNote]
// CHECK:STDERR: fn N.C();
// CHECK:STDERR: ^~~~~~~~~
// CHECK:STDERR:
fn N.C();

// --- fail_use_undefined_poisoned_name.carbon

library "[[@TEST_NAME]]";

class C {};

namespace N;
// Here we use C and poison N.C.
fn N.F1() -> C;

// Try to use N.C which was never defined and poisoned.
// CHECK:STDERR: fail_use_undefined_poisoned_name.carbon:[[@LINE+4]]:14: error: member name `C` not found in `N` [MemberNameNotFoundInScope]
// CHECK:STDERR: fn N.F2() -> N.C;
// CHECK:STDERR:              ^~~
// CHECK:STDERR:
fn N.F2() -> N.C;

// --- fail_poison_with_usage.carbon
// CHECK:STDERR: fail_poison_with_usage.carbon: error: name used before it was declared [NameUseBeforeDecl]

library "[[@TEST_NAME]]";

class C {};

namespace N;
// Here we use C and poison N.C.
fn N.F1(x: C);

// Should fail here since C was poisoned for namespace N when it was used in N
// context without qualification.
// CHECK:STDERR: fail_poison_with_usage.carbon:[[@LINE+4]]:1: note: declared here [NameUseBeforeDeclNote]
// CHECK:STDERR: class N.C {}
// CHECK:STDERR: ^~~~~~~~~~~
// CHECK:STDERR:
class N.C {}

// Should not fail here since both N.F2() and N.F1() input is the class C and
// not class N.C.
fn N.F2(x: C) { N.F1(x); }

// --- fail_poison_multiple_scopes.carbon
// CHECK:STDERR: fail_poison_multiple_scopes.carbon: error: name used before it was declared [NameUseBeforeDecl]

library "[[@TEST_NAME]]";

class C {};

namespace N1;
namespace N1.N2;
namespace N1.N2.N3;
class N1.N2.N3.D1 {
  interface D2 {
    class D3 {
      // Here we use C and poison:
      // * N1.C
      // * N1.N2.C
      // * N1.N2.N3.C
      // * N1.N2.N3.D1.C
      // * N1.N2.N3.D1.D2.C
      // * N1.N2.N3.D1.D2.D3.C
      fn F(x: C);

      // CHECK:STDERR: fail_poison_multiple_scopes.carbon:[[@LINE+5]]:7: note: declared here [NameUseBeforeDeclNote]
      // CHECK:STDERR:       class C {}
      // CHECK:STDERR:       ^~~~~~~~~
      // CHECK:STDERR:
      // CHECK:STDERR: fail_poison_multiple_scopes.carbon: error: name used before it was declared [NameUseBeforeDecl]
      class C {}
    }
    // CHECK:STDERR: fail_poison_multiple_scopes.carbon:[[@LINE+5]]:5: note: declared here [NameUseBeforeDeclNote]
    // CHECK:STDERR:     class C {}
    // CHECK:STDERR:     ^~~~~~~~~
    // CHECK:STDERR:
    // CHECK:STDERR: fail_poison_multiple_scopes.carbon: error: name used before it was declared [NameUseBeforeDecl]
    class C {}
  }
  // CHECK:STDERR: fail_poison_multiple_scopes.carbon:[[@LINE+5]]:3: note: declared here [NameUseBeforeDeclNote]
  // CHECK:STDERR:   class C {}
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_poison_multiple_scopes.carbon: error: name used before it was declared [NameUseBeforeDecl]
  class C {}
}

// CHECK:STDERR: fail_poison_multiple_scopes.carbon:[[@LINE+5]]:1: note: declared here [NameUseBeforeDeclNote]
// CHECK:STDERR: class N1.C {}
// CHECK:STDERR: ^~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_poison_multiple_scopes.carbon: error: name used before it was declared [NameUseBeforeDecl]
class N1.C {}

// CHECK:STDERR: fail_poison_multiple_scopes.carbon:[[@LINE+5]]:1: note: declared here [NameUseBeforeDeclNote]
// CHECK:STDERR: interface N1.N2.C {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_poison_multiple_scopes.carbon: error: name used before it was declared [NameUseBeforeDecl]
interface N1.N2.C {}

// CHECK:STDERR: fail_poison_multiple_scopes.carbon:[[@LINE+4]]:1: note: declared here [NameUseBeforeDeclNote]
// CHECK:STDERR: class N1.N2.N3.C {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
class N1.N2.N3.C {}

// --- fail_alias.carbon
// CHECK:STDERR: fail_alias.carbon: error: name used before it was declared [NameUseBeforeDecl]

library "[[@TEST_NAME]]";

class C {}

namespace N;
// CHECK:STDERR: fail_alias.carbon:[[@LINE+4]]:9: note: declared here [NameUseBeforeDeclNote]
// CHECK:STDERR: alias N.C = C;
// CHECK:STDERR:         ^
// CHECK:STDERR:
alias N.C = C;

// --- ignored_poison_in_import.carbon

library "[[@TEST_NAME]]";
import library "poison";

// This doesn't fail.
class N.C {}

// --- poison.impl.carbon

impl library "[[@TEST_NAME]]";

// TODO: This should fail since N.C was poisoned in the api.
class N.C {}

// --- using_poisoned_name_in_impl.carbon

library "[[@TEST_NAME]]";

interface C {};

namespace N;
// Here we use C and poison N.C.
fn N.F1(x: C);

class N.X {
  extend impl as C {
  }
}

// --- fail_using_poisoned_name_in_impl_outside_class.carbon

library "[[@TEST_NAME]]";

interface A {
  fn B();
}
class X {
  extend impl as A {
    fn F() { return; }
    // CHECK:STDERR: fail_using_poisoned_name_in_impl_outside_class.carbon:[[@LINE+4]]:10: error: `impl as` can only be used in a class [ImplAsOutsideClass]
    // CHECK:STDERR:     impl as B {}
    // CHECK:STDERR:          ^~
    // CHECK:STDERR:
    impl as B {}
  }
}

// --- fail_poison_when_lookup_fails.carbon

library "[[@TEST_NAME]]";

namespace N;
// CHECK:STDERR: fail_poison_when_lookup_fails.carbon:[[@LINE+5]]:11: error: name `C` not found [NameNotFound]
// CHECK:STDERR: fn N.F(x: C);
// CHECK:STDERR:           ^
// CHECK:STDERR:
// CHECK:STDERR: fail_poison_when_lookup_fails.carbon: error: name used before it was declared [NameUseBeforeDecl]
fn N.F(x: C);

// TODO: We should ideally only produce one diagnostic here.
// CHECK:STDERR: fail_poison_when_lookup_fails.carbon:[[@LINE+5]]:1: note: declared here [NameUseBeforeDeclNote]
// CHECK:STDERR: class C {}
// CHECK:STDERR: ^~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_poison_when_lookup_fails.carbon: error: name used before it was declared [NameUseBeforeDecl]
class C {}
// CHECK:STDERR: fail_poison_when_lookup_fails.carbon:[[@LINE+4]]:1: note: declared here [NameUseBeforeDeclNote]
// CHECK:STDERR: class N.C {}
// CHECK:STDERR: ^~~~~~~~~~~
// CHECK:STDERR:
class N.C {}

// --- fail_poison_with_lexical_result.carbon
// CHECK:STDERR: fail_poison_with_lexical_result.carbon: error: name used before it was declared [NameUseBeforeDecl]

library "[[@TEST_NAME]]";

fn F() {
  class A {}

  class B {
    var v: A;

    // CHECK:STDERR: fail_poison_with_lexical_result.carbon:[[@LINE+4]]:5: note: declared here [NameUseBeforeDeclNote]
    // CHECK:STDERR:     class A {}
    // CHECK:STDERR:     ^~~~~~~~~
    // CHECK:STDERR:
    class A {}
  }
}

// CHECK:STDOUT: --- no_poison.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C.f79: type = class_type @C.1 [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %C.9f4: type = class_type @C.2 [template]
// CHECK:STDOUT:   %F1.type: type = fn_type @F1 [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %F1: %F1.type = struct_value () [template]
// CHECK:STDOUT:   %F2.type: type = fn_type @F2 [template]
// CHECK:STDOUT:   %F2: %F2.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %C.decl.loc4
// CHECK:STDOUT:     .N = %N
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl.loc4: type = class_decl @C.1 [template = constants.%C.f79] {} {}
// CHECK:STDOUT:   %N: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %C.decl.loc8
// CHECK:STDOUT:     .F1 = %F1.decl
// CHECK:STDOUT:     .F2 = %F2.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl.loc8: type = class_decl @C.2 [template = constants.%C.9f4] {} {}
// CHECK:STDOUT:   %F1.decl: %F1.type = fn_decl @F1 [template = constants.%F1] {
// CHECK:STDOUT:     %x.patt: %C.9f4 = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: %C.9f4 = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %C.9f4 = value_param runtime_param0
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl.loc8 [template = constants.%C.9f4]
// CHECK:STDOUT:     %x: %C.9f4 = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F2.decl: %F2.type = fn_decl @F2 [template = constants.%F2] {
// CHECK:STDOUT:     %x.patt: %C.9f4 = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: %C.9f4 = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %C.9f4 = value_param runtime_param0
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl.loc8 [template = constants.%C.9f4]
// CHECK:STDOUT:     %x: %C.9f4 = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C.1 {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C.2 {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F1(%x.param_patt: %C.9f4);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F2(%x.param_patt: %C.9f4) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %N.ref: <namespace> = name_ref N, file.%N [template = file.%N]
// CHECK:STDOUT:   %F1.ref: %F1.type = name_ref F1, file.%F1.decl [template = constants.%F1]
// CHECK:STDOUT:   %x.ref: %C.9f4 = name_ref x, %x
// CHECK:STDOUT:   %F1.call: init %empty_tuple.type = call %F1.ref(%x.ref)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- poison.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %F1.type: type = fn_type @F1 [template]
// CHECK:STDOUT:   %F1: %F1.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .N = %N
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   %N: <namespace> = namespace [template] {
// CHECK:STDOUT:     .F1 = %F1.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F1.decl: %F1.type = fn_decl @F1 [template = constants.%F1] {
// CHECK:STDOUT:     %x.patt: %C = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: %C = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %C = value_param runtime_param0
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %x: %C = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F1(%x.param_patt: %C);
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_poison_class_without_usage.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %F1.type: type = fn_type @F1 [template]
// CHECK:STDOUT:   %F1: %F1.type = struct_value () [template]
// CHECK:STDOUT:   %.fb7: type = class_type @.1 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .N = %N
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   %N: <namespace> = namespace [template] {
// CHECK:STDOUT:     .F1 = %F1.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F1.decl: %F1.type = fn_decl @F1 [template = constants.%F1] {
// CHECK:STDOUT:     %x.patt: %C = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: %C = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %C = value_param runtime_param0
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %x: %C = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.decl: type = class_decl @.1 [template = constants.%.fb7] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @.1 {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F1(%x.param_patt: %C);
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_poison_interface_without_usage.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [template]
// CHECK:STDOUT:   %Self.826: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %F1.type: type = fn_type @F1 [template]
// CHECK:STDOUT:   %F1: %F1.type = struct_value () [template]
// CHECK:STDOUT:   %.type: type = facet_type <@.1> [template]
// CHECK:STDOUT:   %Self.d73: %.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .N = %N
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [template = constants.%I.type] {} {}
// CHECK:STDOUT:   %N: <namespace> = namespace [template] {
// CHECK:STDOUT:     .F1 = %F1.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F1.decl: %F1.type = fn_decl @F1 [template = constants.%F1] {
// CHECK:STDOUT:     %x.patt: %I.type = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: %I.type = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %I.type = value_param runtime_param0
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [template = constants.%I.type]
// CHECK:STDOUT:     %x: %I.type = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.decl: type = interface_decl @.1 [template = constants.%.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.826]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @.1 {
// CHECK:STDOUT:   %Self: %.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.d73]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F1(%x.param_patt: %I.type);
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_poison_namespace_without_usage.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %F1.type: type = fn_type @F1 [template]
// CHECK:STDOUT:   %F1: %F1.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .N = %N
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   %N: <namespace> = namespace [template] {
// CHECK:STDOUT:     .F1 = %F1.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F1.decl: %F1.type = fn_decl @F1 [template = constants.%F1] {
// CHECK:STDOUT:     %x.patt: %C = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: %C = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %C = value_param runtime_param0
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %x: %C = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc17: <namespace> = namespace [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F1(%x.param_patt: %C);
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_poison_member_without_usage.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C1: type = class_type @C1 [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %D: type = class_type @D [template]
// CHECK:STDOUT:   %F1.type: type = fn_type @F1 [template]
// CHECK:STDOUT:   %F1: %F1.type = struct_value () [template]
// CHECK:STDOUT:   %C2: type = class_type @C2 [template]
// CHECK:STDOUT:   %D.elem: type = unbound_element_type %D, %C2 [template]
// CHECK:STDOUT:   %struct_type.C1: type = struct_type {.C1: %C2} [template]
// CHECK:STDOUT:   %complete_type.ec1: <witness> = complete_type_witness %struct_type.C1 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C1 = %C1.decl
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C1.decl: type = class_decl @C1 [template = constants.%C1] {} {}
// CHECK:STDOUT:   %D.decl: type = class_decl @D [template = constants.%D] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C1 {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %F1.decl: %F1.type = fn_decl @F1 [template = constants.%F1] {
// CHECK:STDOUT:     %x.patt: %C1 = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: %C1 = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %C1 = value_param runtime_param0
// CHECK:STDOUT:     %C1.ref: type = name_ref C1, file.%C1.decl [template = constants.%C1]
// CHECK:STDOUT:     %x: %C1 = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C2.decl: type = class_decl @C2 [template = constants.%C2] {} {}
// CHECK:STDOUT:   %.loc18_9: %D.elem = field_decl C1, element0 [template]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %.loc18_3: %D.elem = var_pattern %.loc18_9
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var: ref %D.elem = var <invalid>
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.C1 [template = constants.%complete_type.ec1]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C2 {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F1(%x.param_patt: %C1);
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_poison_function_without_usage.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %F1.type: type = fn_type @F1 [template]
// CHECK:STDOUT:   %F1: %F1.type = struct_value () [template]
// CHECK:STDOUT:   %.type: type = fn_type @.1 [template]
// CHECK:STDOUT:   %.236: %.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .N = %N
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   %N: <namespace> = namespace [template] {
// CHECK:STDOUT:     .F1 = %F1.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F1.decl: %F1.type = fn_decl @F1 [template = constants.%F1] {
// CHECK:STDOUT:     %x.patt: %C = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: %C = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %C = value_param runtime_param0
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %x: %C = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.decl: %.type = fn_decl @.1 [template = constants.%.236] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F1(%x.param_patt: %C);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @.1();
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_use_undefined_poisoned_name.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %F1.type: type = fn_type @F1 [template]
// CHECK:STDOUT:   %F1: %F1.type = struct_value () [template]
// CHECK:STDOUT:   %F2.type: type = fn_type @F2 [template]
// CHECK:STDOUT:   %F2: %F2.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .N = %N
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   %N: <namespace> = namespace [template] {
// CHECK:STDOUT:     .F1 = %F1.decl
// CHECK:STDOUT:     .F2 = %F2.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F1.decl: %F1.type = fn_decl @F1 [template = constants.%F1] {
// CHECK:STDOUT:     %return.patt: %C = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %C = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %return.param: ref %C = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %C = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F2.decl: %F2.type = fn_decl @F2 [template = constants.%F2] {
// CHECK:STDOUT:     %return.patt: <error> = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: <error> = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %N.ref: <namespace> = name_ref N, file.%N [template = file.%N]
// CHECK:STDOUT:     %C.ref: <error> = name_ref C, <error> [template = <error>]
// CHECK:STDOUT:     %return.param: ref <error> = out_param runtime_param0
// CHECK:STDOUT:     %return: ref <error> = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F1() -> %C;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F2() -> <error>;
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_poison_with_usage.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %F1.type: type = fn_type @F1 [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %F1: %F1.type = struct_value () [template]
// CHECK:STDOUT:   %.fb7: type = class_type @.1 [template]
// CHECK:STDOUT:   %F2.type: type = fn_type @F2 [template]
// CHECK:STDOUT:   %F2: %F2.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .N = %N
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   %N: <namespace> = namespace [template] {
// CHECK:STDOUT:     .F1 = %F1.decl
// CHECK:STDOUT:     .F2 = %F2.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F1.decl: %F1.type = fn_decl @F1 [template = constants.%F1] {
// CHECK:STDOUT:     %x.patt: %C = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: %C = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %C = value_param runtime_param0
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %x: %C = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.decl: type = class_decl @.1 [template = constants.%.fb7] {} {}
// CHECK:STDOUT:   %F2.decl: %F2.type = fn_decl @F2 [template = constants.%F2] {
// CHECK:STDOUT:     %x.patt: %C = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: %C = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %C = value_param runtime_param0
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %x: %C = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @.1 {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F1(%x.param_patt: %C);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F2(%x.param_patt: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %N.ref: <namespace> = name_ref N, file.%N [template = file.%N]
// CHECK:STDOUT:   %F1.ref: %F1.type = name_ref F1, file.%F1.decl [template = constants.%F1]
// CHECK:STDOUT:   %x.ref: %C = name_ref x, %x
// CHECK:STDOUT:   %F1.call: init %empty_tuple.type = call %F1.ref(%x.ref)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_poison_multiple_scopes.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %D1: type = class_type @D1 [template]
// CHECK:STDOUT:   %D2.type: type = facet_type <@D2> [template]
// CHECK:STDOUT:   %Self.8cf: %D2.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %D3.b65: type = class_type @D3 [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F, @D3(%Self.8cf) [symbolic]
// CHECK:STDOUT:   %F: %F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %.c1e: type = class_type @.1 [template]
// CHECK:STDOUT:   %.9af: type = class_type @.2 [template]
// CHECK:STDOUT:   %.6ae: type = class_type @.3 [template]
// CHECK:STDOUT:   %.1c0: type = class_type @.4 [template]
// CHECK:STDOUT:   %.type: type = facet_type <@.6> [template]
// CHECK:STDOUT:   %Self.501: %.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %.4b5: type = class_type @.5 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .N1 = %N1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   %N1: <namespace> = namespace [template] {
// CHECK:STDOUT:     .N2 = %N2
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %N2: <namespace> = namespace [template] {
// CHECK:STDOUT:     .N3 = %N3
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %N3: <namespace> = namespace [template] {
// CHECK:STDOUT:     .D1 = %D1.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D1.decl: type = class_decl @D1 [template = constants.%D1] {} {}
// CHECK:STDOUT:   %.decl.loc49: type = class_decl @.4 [template = constants.%.1c0] {} {}
// CHECK:STDOUT:   %.decl.loc56: type = interface_decl @.6 [template = constants.%.type] {} {}
// CHECK:STDOUT:   %.decl.loc62: type = class_decl @.5 [template = constants.%.4b5] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @D2 {
// CHECK:STDOUT:   %Self: %D2.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.8cf]
// CHECK:STDOUT:   %D3.decl: type = class_decl @D3 [template = constants.%D3.b65] {} {}
// CHECK:STDOUT:   %.decl: type = class_decl @.2 [template = constants.%.9af] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .D3 = %D3.decl
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @.6 {
// CHECK:STDOUT:   %Self: %.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.501]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D1 {
// CHECK:STDOUT:   %D2.decl: type = interface_decl @D2 [template = constants.%D2.type] {} {}
// CHECK:STDOUT:   %.decl: type = class_decl @.3 [template = constants.%.6ae] {} {}
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @D3(@D2.%Self: %D2.type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self: %D2.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.8cf)]
// CHECK:STDOUT:   %F.type: type = fn_type @F, @D3(%Self) [symbolic = %F.type (constants.%F.type)]
// CHECK:STDOUT:   %F: @D3.%F.type (%F.type) = struct_value () [symbolic = %F (constants.%F)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %F.decl: @D3.%F.type (%F.type) = fn_decl @F [symbolic = @D3.%F (constants.%F)] {
// CHECK:STDOUT:       %x.patt: %C = binding_pattern x
// CHECK:STDOUT:       %x.param_patt: %C = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %x.param: %C = value_param runtime_param0
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:       %x: %C = bind_name x, %x.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.decl: type = class_decl @.1 [template = constants.%.c1e] {} {}
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @.1(@D2.%Self: %D2.type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @.2(@D2.%Self: %D2.type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @.3 {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @.4 {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @.5 {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(@D2.%Self: %D2.type) {
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param_patt: %C);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D3(constants.%Self.8cf) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%Self.8cf) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @.1(constants.%Self.8cf) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D3(%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @.2(constants.%Self.8cf) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_alias.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .N = %N
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   %N: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %C.ref: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:   %.loc12: type = bind_alias <invalid>, %C.decl [template = constants.%C]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- ignored_poison_in_import.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.f42 = import_ref Main//poison, C, unloaded
// CHECK:STDOUT:   %import_ref.a98: <namespace> = import_ref Main//poison, N, loaded
// CHECK:STDOUT:   %N: <namespace> = namespace %import_ref.a98, [template] {
// CHECK:STDOUT:     .F1 = %import_ref.cd4
// CHECK:STDOUT:     .C = file.%C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = imports.%import_ref.f42
// CHECK:STDOUT:     .N = imports.%N
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- poison.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.f42 = import_ref Main//poison, C, unloaded
// CHECK:STDOUT:   %import_ref.a98: <namespace> = import_ref Main//poison, N, loaded
// CHECK:STDOUT:   %N: <namespace> = namespace %import_ref.a98, [template] {
// CHECK:STDOUT:     .F1 = %import_ref.cd4
// CHECK:STDOUT:     .C = file.%C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = imports.%import_ref.f42
// CHECK:STDOUT:     .N = imports.%N
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import.loc2_6.1 = import <invalid>
// CHECK:STDOUT:   %default.import.loc2_6.2 = import <invalid>
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- using_poisoned_name_in_impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C.type: type = facet_type <@C> [template]
// CHECK:STDOUT:   %Self: %C.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %F1.type: type = fn_type @F1 [template]
// CHECK:STDOUT:   %F1: %F1.type = struct_value () [template]
// CHECK:STDOUT:   %X: type = class_type @X [template]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness () [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .N = %N
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = interface_decl @C [template = constants.%C.type] {} {}
// CHECK:STDOUT:   %N: <namespace> = namespace [template] {
// CHECK:STDOUT:     .F1 = %F1.decl
// CHECK:STDOUT:     .X = %X.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F1.decl: %F1.type = fn_decl @F1 [template = constants.%F1] {
// CHECK:STDOUT:     %x.patt: %C.type = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: %C.type = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %C.type = value_param runtime_param0
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C.type]
// CHECK:STDOUT:     %x: %C.type = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %X.decl: type = class_decl @X [template = constants.%X] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @C {
// CHECK:STDOUT:   %Self: %C.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %Self.ref as %C.ref {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = @X.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @X {
// CHECK:STDOUT:   impl_decl @impl [template] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%X [template = constants.%X]
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness () [template = constants.%impl_witness]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F1(%x.param_patt: %C.type);
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_using_poisoned_name_in_impl_outside_class.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.type: type = facet_type <@A> [template]
// CHECK:STDOUT:   %Self: %A.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %B.type: type = fn_type @B [template]
// CHECK:STDOUT:   %B: %B.type = struct_value () [template]
// CHECK:STDOUT:   %B.assoc_type: type = assoc_entity_type %A.type, %B.type [template]
// CHECK:STDOUT:   %assoc0: %B.assoc_type = assoc_entity element0, @A.%B.decl [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {}
// CHECK:STDOUT:
// CHECK:STDOUT: interface @A {
// CHECK:STDOUT:   %Self: %A.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %B.decl: %B.type = fn_decl @B [template = constants.%B] {} {}
// CHECK:STDOUT:   %assoc0: %B.assoc_type = assoc_entity element0, %B.decl [template = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .B = %assoc0
// CHECK:STDOUT:   witness = (%B.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: <unexpected>.inst26.loc8_15 as <unexpected>.inst27.loc8_18;
// CHECK:STDOUT:
// CHECK:STDOUT: class @X {
// CHECK:STDOUT:   complete_type_witness = invalid
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @B(@A.%Self: %A.type) {
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F();
// CHECK:STDOUT:
// CHECK:STDOUT: specific @B(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_poison_when_lookup_fails.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.a95: type = class_type @.1 [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %.fb7: type = class_type @.2 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .N = %N
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %N: <namespace> = namespace [template] {
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %x.patt: <error> = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: <error> = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: <error> = value_param runtime_param0
// CHECK:STDOUT:     %C.ref: <error> = name_ref C, <error> [template = <error>]
// CHECK:STDOUT:     %x: <error> = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.decl.loc18: type = class_decl @.1 [template = constants.%.a95] {} {}
// CHECK:STDOUT:   %.decl.loc23: type = class_decl @.2 [template = constants.%.fb7] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @.1 {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @.2 {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%x.param_patt: <error>);
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_poison_with_lexical_result.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %A: type = class_type @A [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %B: type = class_type @B [template]
// CHECK:STDOUT:   %B.elem: type = unbound_element_type %B, %A [template]
// CHECK:STDOUT:   %.96d: type = class_type @.1 [template]
// CHECK:STDOUT:   %struct_type.v: type = struct_type {.v: %A} [template]
// CHECK:STDOUT:   %complete_type.57e: <witness> = complete_type_witness %struct_type.v [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %.loc9_10: %B.elem = field_decl v, element0 [template]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %.loc9_5: %B.elem = var_pattern %.loc9_10
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var: ref %B.elem = var <invalid>
// CHECK:STDOUT:   %.decl: type = class_decl @.1 [template = constants.%.96d] {} {}
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.v [template = constants.%complete_type.57e]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @.1 {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.decl: type = class_decl @A [template = constants.%A] {} {}
// CHECK:STDOUT:   %B.decl: type = class_decl @B [template = constants.%B] {} {}
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
