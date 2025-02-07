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

library "[[@TEST_NAME]]";

class C {};

namespace N;
// Here we use C and poison N.C.
// CHECK:STDERR: fail_poison_class_without_usage.carbon:[[@LINE+3]]:12: error: name used before it was declared [NameUseBeforeDecl]
// CHECK:STDERR: fn N.F1(x: C);
// CHECK:STDERR:            ^
fn N.F1(x: C);

// Should fail here since C was poisoned for namespace N when it was used in N
// context without qualification.
// CHECK:STDERR: fail_poison_class_without_usage.carbon:[[@LINE+4]]:1: note: declared here [NameUseBeforeDeclNote]
// CHECK:STDERR: class N.C {}
// CHECK:STDERR: ^~~~~~~~~~~
// CHECK:STDERR:
class N.C {}

// --- fail_poison_interface_without_usage.carbon

library "[[@TEST_NAME]]";

interface I {};

namespace N;
// Here we use I and poison N.I.
// CHECK:STDERR: fail_poison_interface_without_usage.carbon:[[@LINE+3]]:12: error: name used before it was declared [NameUseBeforeDecl]
// CHECK:STDERR: fn N.F1(x: I);
// CHECK:STDERR:            ^
fn N.F1(x: I);

// Should fail here since I was poisoned for namespace N when it was used in N
// context without qualification.
// CHECK:STDERR: fail_poison_interface_without_usage.carbon:[[@LINE+4]]:1: note: declared here [NameUseBeforeDeclNote]
// CHECK:STDERR: interface N.I {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~
// CHECK:STDERR:
interface N.I {}

// --- fail_poison_namespace_without_usage.carbon

library "[[@TEST_NAME]]";

class C {};

namespace N;
// Here we use C and poison N.C.
// CHECK:STDERR: fail_poison_namespace_without_usage.carbon:[[@LINE+3]]:12: error: name used before it was declared [NameUseBeforeDecl]
// CHECK:STDERR: fn N.F1(x: C);
// CHECK:STDERR:            ^
fn N.F1(x: C);

// Should fail here since C was poisoned for namespace N when it was used in N
// context without qualification.
// CHECK:STDERR: fail_poison_namespace_without_usage.carbon:[[@LINE+4]]:1: note: declared here [NameUseBeforeDeclNote]
// CHECK:STDERR: namespace N.C;
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR:
namespace N.C;

// --- fail_poison_member_without_usage.carbon

library "[[@TEST_NAME]]";

class C1 {};

class D {
  // Here we use C1 and poison D.C1.
  // CHECK:STDERR: fail_poison_member_without_usage.carbon:[[@LINE+3]]:12: error: name used before it was declared [NameUseBeforeDecl]
  // CHECK:STDERR:   fn F1(x: C1);
  // CHECK:STDERR:            ^~
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

library "[[@TEST_NAME]]";

class C {};

namespace N;
// Here we use C and poison N.C.
// CHECK:STDERR: fail_poison_function_without_usage.carbon:[[@LINE+3]]:12: error: name used before it was declared [NameUseBeforeDecl]
// CHECK:STDERR: fn N.F1(x: C);
// CHECK:STDERR:            ^
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

library "[[@TEST_NAME]]";

class C {};

namespace N;
// Here we use C and poison N.C.
// CHECK:STDERR: fail_poison_with_usage.carbon:[[@LINE+3]]:12: error: name used before it was declared [NameUseBeforeDecl]
// CHECK:STDERR: fn N.F1(x: C);
// CHECK:STDERR:            ^
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
      // CHECK:STDERR: fail_poison_multiple_scopes.carbon:[[@LINE+3]]:15: error: name used before it was declared [NameUseBeforeDecl]
      // CHECK:STDERR:       fn F(x: C);
      // CHECK:STDERR:               ^
      fn F(x: C);

      // CHECK:STDERR: fail_poison_multiple_scopes.carbon:[[@LINE+7]]:7: note: declared here [NameUseBeforeDeclNote]
      // CHECK:STDERR:       class C {}
      // CHECK:STDERR:       ^~~~~~~~~
      // CHECK:STDERR:
      // CHECK:STDERR: fail_poison_multiple_scopes.carbon:[[@LINE-6]]:15: error: name used before it was declared [NameUseBeforeDecl]
      // CHECK:STDERR:       fn F(x: C);
      // CHECK:STDERR:               ^
      class C {}
    }
    // CHECK:STDERR: fail_poison_multiple_scopes.carbon:[[@LINE+7]]:5: note: declared here [NameUseBeforeDeclNote]
    // CHECK:STDERR:     class C {}
    // CHECK:STDERR:     ^~~~~~~~~
    // CHECK:STDERR:
    // CHECK:STDERR: fail_poison_multiple_scopes.carbon:[[@LINE-15]]:15: error: name used before it was declared [NameUseBeforeDecl]
    // CHECK:STDERR:       fn F(x: C);
    // CHECK:STDERR:               ^
    class C {}
  }
  // CHECK:STDERR: fail_poison_multiple_scopes.carbon:[[@LINE+7]]:3: note: declared here [NameUseBeforeDeclNote]
  // CHECK:STDERR:   class C {}
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_poison_multiple_scopes.carbon:[[@LINE-24]]:15: error: name used before it was declared [NameUseBeforeDecl]
  // CHECK:STDERR:       fn F(x: C);
  // CHECK:STDERR:               ^
  class C {}
}

// CHECK:STDERR: fail_poison_multiple_scopes.carbon:[[@LINE+7]]:1: note: declared here [NameUseBeforeDeclNote]
// CHECK:STDERR: class N1.C {}
// CHECK:STDERR: ^~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_poison_multiple_scopes.carbon:[[@LINE-34]]:15: error: name used before it was declared [NameUseBeforeDecl]
// CHECK:STDERR:       fn F(x: C);
// CHECK:STDERR:               ^
class N1.C {}

// CHECK:STDERR: fail_poison_multiple_scopes.carbon:[[@LINE+7]]:1: note: declared here [NameUseBeforeDeclNote]
// CHECK:STDERR: interface N1.N2.C {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_poison_multiple_scopes.carbon:[[@LINE-43]]:15: error: name used before it was declared [NameUseBeforeDecl]
// CHECK:STDERR:       fn F(x: C);
// CHECK:STDERR:               ^
interface N1.N2.C {}

// CHECK:STDERR: fail_poison_multiple_scopes.carbon:[[@LINE+4]]:1: note: declared here [NameUseBeforeDeclNote]
// CHECK:STDERR: class N1.N2.N3.C {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
class N1.N2.N3.C {}

// --- fail_alias.carbon

library "[[@TEST_NAME]]";

class C {}

namespace N;
// CHECK:STDERR: fail_alias.carbon:[[@LINE+7]]:13: error: name used before it was declared [NameUseBeforeDecl]
// CHECK:STDERR: alias N.C = C;
// CHECK:STDERR:             ^
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
// CHECK:STDERR: fail_poison_when_lookup_fails.carbon:[[@LINE+7]]:11: error: name `C` not found [NameNotFound]
// CHECK:STDERR: fn N.F(x: C);
// CHECK:STDERR:           ^
// CHECK:STDERR:
// CHECK:STDERR: fail_poison_when_lookup_fails.carbon:[[@LINE+3]]:11: error: name used before it was declared [NameUseBeforeDecl]
// CHECK:STDERR: fn N.F(x: C);
// CHECK:STDERR:           ^
fn N.F(x: C);

// TODO: We should ideally only produce one diagnostic here.
// CHECK:STDERR: fail_poison_when_lookup_fails.carbon:[[@LINE+7]]:1: note: declared here [NameUseBeforeDeclNote]
// CHECK:STDERR: class C {}
// CHECK:STDERR: ^~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_poison_when_lookup_fails.carbon:[[@LINE-7]]:11: error: name used before it was declared [NameUseBeforeDecl]
// CHECK:STDERR: fn N.F(x: C);
// CHECK:STDERR:           ^
class C {}
// CHECK:STDERR: fail_poison_when_lookup_fails.carbon:[[@LINE+4]]:1: note: declared here [NameUseBeforeDeclNote]
// CHECK:STDERR: class N.C {}
// CHECK:STDERR: ^~~~~~~~~~~
// CHECK:STDERR:
class N.C {}

// --- fail_poison_with_lexical_result.carbon

library "[[@TEST_NAME]]";

fn F() {
  class A {}

  class B {
    // CHECK:STDERR: fail_poison_with_lexical_result.carbon:[[@LINE+3]]:12: error: name used before it was declared [NameUseBeforeDecl]
    // CHECK:STDERR:     var v: A;
    // CHECK:STDERR:            ^
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
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C.f79
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C.2 {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C.9f4
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
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F1(%x.param_patt: %C);
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_poison_class_without_usage.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C.f79: type = class_type @C.1 [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %F1.type: type = fn_type @F1 [template]
// CHECK:STDOUT:   %F1: %F1.type = struct_value () [template]
// CHECK:STDOUT:   %C.9f4: type = class_type @C.2 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %C.decl.loc4
// CHECK:STDOUT:     .N = %N
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl.loc4: type = class_decl @C.1 [template = constants.%C.f79] {} {}
// CHECK:STDOUT:   %N: <namespace> = namespace [template] {
// CHECK:STDOUT:     .F1 = %F1.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F1.decl: %F1.type = fn_decl @F1 [template = constants.%F1] {
// CHECK:STDOUT:     %x.patt: %C.f79 = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: %C.f79 = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %C.f79 = value_param runtime_param0
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl.loc4 [template = constants.%C.f79]
// CHECK:STDOUT:     %x: %C.f79 = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl.loc19: type = class_decl @C.2 [template = constants.%C.9f4] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C.1 {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C.f79
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C.2 {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C.9f4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F1(%x.param_patt: %C.f79);
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_poison_interface_without_usage.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type.733: type = facet_type <@I.1> [template]
// CHECK:STDOUT:   %Self.826: %I.type.733 = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %F1.type: type = fn_type @F1 [template]
// CHECK:STDOUT:   %F1: %F1.type = struct_value () [template]
// CHECK:STDOUT:   %I.type.4da: type = facet_type <@I.2> [template]
// CHECK:STDOUT:   %Self.f85: %I.type.4da = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .I = %I.decl.loc4
// CHECK:STDOUT:     .N = %N
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl.loc4: type = interface_decl @I.1 [template = constants.%I.type.733] {} {}
// CHECK:STDOUT:   %N: <namespace> = namespace [template] {
// CHECK:STDOUT:     .F1 = %F1.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F1.decl: %F1.type = fn_decl @F1 [template = constants.%F1] {
// CHECK:STDOUT:     %x.patt: %I.type.733 = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: %I.type.733 = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %I.type.733 = value_param runtime_param0
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl.loc4 [template = constants.%I.type.733]
// CHECK:STDOUT:     %x: %I.type.733 = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl.loc19: type = interface_decl @I.2 [template = constants.%I.type.4da] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I.1 {
// CHECK:STDOUT:   %Self: %I.type.733 = bind_symbolic_name Self, 0 [symbolic = constants.%Self.826]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I.2 {
// CHECK:STDOUT:   %Self: %I.type.4da = bind_symbolic_name Self, 0 [symbolic = constants.%Self.f85]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F1(%x.param_patt: %I.type.733);
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
// CHECK:STDOUT:   %C: <namespace> = namespace [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
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
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C1
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
// CHECK:STDOUT:   %.loc20_9: %D.elem = field_decl C1, element0 [template]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %.loc20_3: %D.elem = var_pattern %.loc20_9
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var: ref %D.elem = var <none>
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.C1 [template = constants.%complete_type.ec1]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT:   .F1 = %F1.decl
// CHECK:STDOUT:   .C2 = %C2.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C2 {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F1(%x.param_patt: %C1);
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_poison_function_without_usage.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C.f79: type = class_type @C.2 [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %F1.type: type = fn_type @F1 [template]
// CHECK:STDOUT:   %F1: %F1.type = struct_value () [template]
// CHECK:STDOUT:   %C.type: type = fn_type @C.1 [template]
// CHECK:STDOUT:   %C.3f2: %C.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %C.decl.loc4
// CHECK:STDOUT:     .N = %N
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl.loc4: type = class_decl @C.2 [template = constants.%C.f79] {} {}
// CHECK:STDOUT:   %N: <namespace> = namespace [template] {
// CHECK:STDOUT:     .F1 = %F1.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F1.decl: %F1.type = fn_decl @F1 [template = constants.%F1] {
// CHECK:STDOUT:     %x.patt: %C.f79 = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: %C.f79 = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %C.f79 = value_param runtime_param0
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl.loc4 [template = constants.%C.f79]
// CHECK:STDOUT:     %x: %C.f79 = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl.loc19: %C.type = fn_decl @C.1 [template = constants.%C.3f2] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C.2 {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C.f79
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F1(%x.param_patt: %C.f79);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.1();
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
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F1() -> %C;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F2() -> <error>;
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_poison_with_usage.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C.f79: type = class_type @C.1 [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %F1.type: type = fn_type @F1 [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %F1: %F1.type = struct_value () [template]
// CHECK:STDOUT:   %C.9f4: type = class_type @C.2 [template]
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
// CHECK:STDOUT:     .F1 = %F1.decl
// CHECK:STDOUT:     .F2 = %F2.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F1.decl: %F1.type = fn_decl @F1 [template = constants.%F1] {
// CHECK:STDOUT:     %x.patt: %C.f79 = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: %C.f79 = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %C.f79 = value_param runtime_param0
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl.loc4 [template = constants.%C.f79]
// CHECK:STDOUT:     %x: %C.f79 = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl.loc19: type = class_decl @C.2 [template = constants.%C.9f4] {} {}
// CHECK:STDOUT:   %F2.decl: %F2.type = fn_decl @F2 [template = constants.%F2] {
// CHECK:STDOUT:     %x.patt: %C.f79 = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: %C.f79 = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %C.f79 = value_param runtime_param0
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl.loc4 [template = constants.%C.f79]
// CHECK:STDOUT:     %x: %C.f79 = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C.1 {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C.f79
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C.2 {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C.9f4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F1(%x.param_patt: %C.f79);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F2(%x.param_patt: %C.f79) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %N.ref: <namespace> = name_ref N, file.%N [template = file.%N]
// CHECK:STDOUT:   %F1.ref: %F1.type = name_ref F1, file.%F1.decl [template = constants.%F1]
// CHECK:STDOUT:   %x.ref: %C.f79 = name_ref x, %x
// CHECK:STDOUT:   %F1.call: init %empty_tuple.type = call %F1.ref(%x.ref)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_poison_multiple_scopes.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C.f79: type = class_type @C.1 [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %D1: type = class_type @D1 [template]
// CHECK:STDOUT:   %D2.type: type = facet_type <@D2> [template]
// CHECK:STDOUT:   %Self.8cf: %D2.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %D3.b65: type = class_type @D3 [template]
// CHECK:STDOUT:   %D3.68e: type = class_type @D3, @D3(%Self.8cf) [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F, @D3(%Self.8cf) [symbolic]
// CHECK:STDOUT:   %F: %F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %C.fef: type = class_type @C.2 [template]
// CHECK:STDOUT:   %C.5a3: type = class_type @C.2, @C.2(%Self.8cf) [symbolic]
// CHECK:STDOUT:   %C.2fa: type = class_type @C.3 [template]
// CHECK:STDOUT:   %C.4bd: type = class_type @C.3, @C.3(%Self.8cf) [symbolic]
// CHECK:STDOUT:   %C.6f6: type = class_type @C.4 [template]
// CHECK:STDOUT:   %C.0b8: type = class_type @C.5 [template]
// CHECK:STDOUT:   %C.type: type = facet_type <@C.7> [template]
// CHECK:STDOUT:   %Self.b2a: %C.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %C.6f1: type = class_type @C.6 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %C.decl.loc4
// CHECK:STDOUT:     .N1 = %N1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl.loc4: type = class_decl @C.1 [template = constants.%C.f79] {} {}
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
// CHECK:STDOUT:   %C.decl.loc59: type = class_decl @C.5 [template = constants.%C.0b8] {} {}
// CHECK:STDOUT:   %C.decl.loc68: type = interface_decl @C.7 [template = constants.%C.type] {} {}
// CHECK:STDOUT:   %C.decl.loc74: type = class_decl @C.6 [template = constants.%C.6f1] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @D2 {
// CHECK:STDOUT:   %Self: %D2.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.8cf]
// CHECK:STDOUT:   %D3.decl: type = class_decl @D3 [template = constants.%D3.b65] {} {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C.3 [template = constants.%C.2fa] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .D3 = %D3.decl
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @C.7 {
// CHECK:STDOUT:   %Self: %C.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.b2a]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C.1 {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C.f79
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D1 {
// CHECK:STDOUT:   %D2.decl: type = interface_decl @D2 [template = constants.%D2.type] {} {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C.4 [template = constants.%C.6f6] {} {}
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D1
// CHECK:STDOUT:   .D2 = %D2.decl
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
// CHECK:STDOUT:       %x.patt: %C.f79 = binding_pattern x
// CHECK:STDOUT:       %x.param_patt: %C.f79 = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %x.param: %C.f79 = value_param runtime_param0
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl.loc4 [template = constants.%C.f79]
// CHECK:STDOUT:       %x: %C.f79 = bind_name x, %x.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %C.decl: type = class_decl @C.2 [template = constants.%C.fef] {} {}
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%D3.68e
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C.2(@D2.%Self: %D2.type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.5a3
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C.3(@D2.%Self: %D2.type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.4bd
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C.4 {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C.6f6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C.5 {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C.0b8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C.6 {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C.6f1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(@D2.%Self: %D2.type) {
// CHECK:STDOUT:   fn(%x.param_patt: %C.f79);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D3(constants.%Self.8cf) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%Self.8cf) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.2(constants.%Self.8cf) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D3(%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.3(constants.%Self.8cf) {}
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
// CHECK:STDOUT:   %C: type = bind_alias C, %C.decl [template = constants.%C]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
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
// CHECK:STDOUT:   %Main.C = import_ref Main//poison, C, unloaded
// CHECK:STDOUT:   %Main.N: <namespace> = import_ref Main//poison, N, loaded
// CHECK:STDOUT:   %N: <namespace> = namespace %Main.N, [template] {
// CHECK:STDOUT:     .F1 = %Main.F1
// CHECK:STDOUT:     .C = file.%C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = imports.%Main.C
// CHECK:STDOUT:     .N = imports.%N
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
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
// CHECK:STDOUT:   %Main.C = import_ref Main//poison, C, unloaded
// CHECK:STDOUT:   %Main.N: <namespace> = import_ref Main//poison, N, loaded
// CHECK:STDOUT:   %N: <namespace> = namespace %Main.N, [template] {
// CHECK:STDOUT:     .F1 = %Main.F1
// CHECK:STDOUT:     .C = file.%C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = imports.%Main.C
// CHECK:STDOUT:     .N = imports.%N
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import.loc2_6.1 = import <none>
// CHECK:STDOUT:   %default.import.loc2_6.2 = import <none>
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
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
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%X
// CHECK:STDOUT:   extend @impl.%C.ref
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
// CHECK:STDOUT:   %A.assoc_type: type = assoc_entity_type %A.type [template]
// CHECK:STDOUT:   %assoc0: %A.assoc_type = assoc_entity element0, @A.%B.decl [template]
// CHECK:STDOUT:   %X: type = class_type @X [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {}
// CHECK:STDOUT:
// CHECK:STDOUT: interface @A {
// CHECK:STDOUT:   %Self: %A.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %B.decl: %B.type = fn_decl @B [template = constants.%B] {} {}
// CHECK:STDOUT:   %assoc0: %A.assoc_type = assoc_entity element0, %B.decl [template = constants.%assoc0]
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
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%X
// CHECK:STDOUT:   extend <unexpected>.inst27.loc8_18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @B(@A.%Self: %A.type) {
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
// CHECK:STDOUT:   %C.f79: type = class_type @C.1 [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %C.9f4: type = class_type @C.2 [template]
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
// CHECK:STDOUT:   %C.decl.loc22: type = class_decl @C.1 [template = constants.%C.f79] {} {}
// CHECK:STDOUT:   %C.decl.loc27: type = class_decl @C.2 [template = constants.%C.9f4] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C.1 {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C.f79
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C.2 {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C.9f4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%x.param_patt: <error>);
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_poison_with_lexical_result.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %A.666: type = class_type @A.1 [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %B: type = class_type @B [template]
// CHECK:STDOUT:   %B.elem: type = unbound_element_type %B, %A.666 [template]
// CHECK:STDOUT:   %A.9b6: type = class_type @A.2 [template]
// CHECK:STDOUT:   %struct_type.v: type = struct_type {.v: %A.666} [template]
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
// CHECK:STDOUT: class @A.1 {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A.666
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %.loc11_10: %B.elem = field_decl v, element0 [template]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %.loc11_5: %B.elem = var_pattern %.loc11_10
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var: ref %B.elem = var <none>
// CHECK:STDOUT:   %A.decl: type = class_decl @A.2 [template = constants.%A.9b6] {} {}
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.v [template = constants.%complete_type.57e]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT:   .v = %.loc11_10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A.2 {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A.9b6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.decl: type = class_decl @A.1 [template = constants.%A.666] {} {}
// CHECK:STDOUT:   %B.decl: type = class_decl @B [template = constants.%B] {} {}
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
