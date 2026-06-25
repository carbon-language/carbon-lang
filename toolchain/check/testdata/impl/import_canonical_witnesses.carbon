// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// This test is producing a case where interface ordering will differ between
// interfaces.carbon and use.carbon, testing that witnesses are maintained in a
// canonical order. This tests a regression where the imported ordering would
// be incorrectly used, resulting in witnesses getting swapped and an opaque
// error for the `SimpleOp` lookup because `Empty`'s table is empty.
//
// The IR is generally not significant, but the important difference is as
// follows (if these become the same, it means the test is no longer effective):
//
// SET-CHECK-SUBSET
// CHECK:STDOUT: --- interfaces.carbon
// CHECK:STDOUT:   %facet_type: type = facet_type <@SimpleOp & @Empty> [concrete]
// CHECK:STDOUT: --- use.carbon
// CHECK:STDOUT:   %facet_type: type = facet_type <@Empty & @SimpleOp> [concrete]
//
// This check does not strictly exclude extra facet types, although that should
// create different instruction names. We also anticipate diagnostics should be
// produced if there's a significant error.
//
// NOAUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/import_canonical_witnesses.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/import_canonical_witnesses.carbon


// --- interfaces.carbon
//@include-in-dumps
library "[[@TEST_NAME]]";

// An interface with a trivial function and blanket impl for pointers.
interface SimpleOp {
  fn Op(self);
}

impl forall [T:! type] T as SimpleOp {
  fn Op(unused self) {}
}

// An empty interface with blanket impl for pointers.
interface Empty {}
impl forall [T:! type] T as Empty {}

// Class with a use of `SimpleOp.Op`.
class InterfaceWrap(T:! SimpleOp & Empty) {
  fn UseSimpleOp(unused self, value:! T) {
    value.(SimpleOp.Op)();
  }
}

// Interface combining the above for an indirect `SimpleOp` use through `C`,
// particularly use an associated constant.
interface CombinedFacet {
  let T:! SimpleOp & Empty;
  fn GetInterfaceWrap(self) -> InterfaceWrap(T);
}

// A wrapper providing `CombinedFacet`.
class TypeWrap(T:! type) {
  impl as CombinedFacet where .T = T {
    fn GetInterfaceWrap(unused self) -> InterfaceWrap(T) {
      return {};
    }
  }
}

// --- use.carbon
//@include-in-dumps
library "[[@TEST_NAME]]";
import library "interfaces";

class C {}

fn F(t: TypeWrap(C), c:! C) {
  t.(CombinedFacet.GetInterfaceWrap)().UseSimpleOp(c);
}
