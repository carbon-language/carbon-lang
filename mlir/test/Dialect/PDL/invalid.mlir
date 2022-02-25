// RUN: mlir-opt %s -split-input-file -verify-diagnostics

//===----------------------------------------------------------------------===//
// pdl::ApplyNativeConstraintOp
//===----------------------------------------------------------------------===//

pdl.pattern : benefit(1) {
  %op = operation "foo.op"

  // expected-error@below {{expected at least one argument}}
  "pdl.apply_native_constraint"() {name = "foo", params = []} : () -> ()
  rewrite %op with "rewriter"
}

// -----

//===----------------------------------------------------------------------===//
// pdl::ApplyNativeRewriteOp
//===----------------------------------------------------------------------===//

pdl.pattern : benefit(1) {
  %op = operation "foo.op"
  rewrite %op {
    // expected-error@below {{expected at least one argument}}
    "pdl.apply_native_rewrite"() {name = "foo", params = []} : () -> ()
  }
}

// -----

//===----------------------------------------------------------------------===//
// pdl::AttributeOp
//===----------------------------------------------------------------------===//

pdl.pattern : benefit(1) {
  %type = type

  // expected-error@below {{expected only one of [`type`, `value`] to be set}}
  %attr = attribute : %type 10

  %op = operation "foo.op" {"attr" = %attr} -> (%type : !pdl.type)
  rewrite %op with "rewriter"
}

// -----

pdl.pattern : benefit(1) {
  %op = operation "foo.op"
  rewrite %op {
    %type = type

    // expected-error@below {{expected constant value when specified within a `pdl.rewrite`}}
    %attr = attribute : %type
  }
}

// -----

pdl.pattern : benefit(1) {
  %op = operation "foo.op"
  rewrite %op {
    // expected-error@below {{expected constant value when specified within a `pdl.rewrite`}}
    %attr = attribute
  }
}

// -----

pdl.pattern : benefit(1) {
  // expected-error@below {{expected a bindable user when defined in the matcher body of a `pdl.pattern`}}
  %unused = attribute

  %op = operation "foo.op"
  rewrite %op with "rewriter"
}

// -----

//===----------------------------------------------------------------------===//
// pdl::OperandOp
//===----------------------------------------------------------------------===//

pdl.pattern : benefit(1) {
  // expected-error@below {{expected a bindable user when defined in the matcher body of a `pdl.pattern`}}
  %unused = operand

  %op = operation "foo.op"
  rewrite %op with "rewriter"
}

// -----

//===----------------------------------------------------------------------===//
// pdl::OperandsOp
//===----------------------------------------------------------------------===//

pdl.pattern : benefit(1) {
  // expected-error@below {{expected a bindable user when defined in the matcher body of a `pdl.pattern`}}
  %unused = operands

  %op = operation "foo.op"
  rewrite %op with "rewriter"
}

// -----

//===----------------------------------------------------------------------===//
// pdl::OperationOp
//===----------------------------------------------------------------------===//

pdl.pattern : benefit(1) {
  %op = operation "foo.op"
  rewrite %op {
    // expected-error@below {{must have an operation name when nested within a `pdl.rewrite`}}
    %newOp = operation
  }
}

// -----

pdl.pattern : benefit(1) {
  // expected-error@below {{expected the same number of attribute values and attribute names, got 1 names and 0 values}}
  %op = "pdl.operation"() {
    attributeNames = ["attr"],
    operand_segment_sizes = dense<0> : vector<3xi32>
  } : () -> (!pdl.operation)
  rewrite %op with "rewriter"
}

// -----

pdl.pattern : benefit(1) {
  %op = operation "foo.op"
  rewrite %op {
    %type = type

    // expected-error@below {{op must have inferable or constrained result types when nested within `pdl.rewrite`}}
    // expected-note@below {{result type #0 was not constrained}}
    %newOp = operation "foo.op" -> (%type : !pdl.type)
  }
}

// -----

pdl.pattern : benefit(1) {
  // expected-error@below {{expected a bindable user when defined in the matcher body of a `pdl.pattern`}}
  %unused = operation "foo.op"

  %op = operation "foo.op"
  rewrite %op with "rewriter"
}

// -----

//===----------------------------------------------------------------------===//
// pdl::PatternOp
//===----------------------------------------------------------------------===//

// expected-error@below {{expected body to terminate with `pdl.rewrite`}}
pdl.pattern : benefit(1) {
  // expected-note@below {{see terminator defined here}}
  return
}

// -----

// expected-error@below {{the pattern must contain at least one `pdl.operation`}}
pdl.pattern : benefit(1) {
  rewrite with "foo"
}

// -----
// expected-error@below {{expected only `pdl` operations within the pattern body}}
pdl.pattern : benefit(1) {
  // expected-note@below {{see non-`pdl` operation defined here}}
  "test.foo.other_op"() : () -> ()

  %root = operation "foo.op"
  rewrite %root with "foo"
}

// -----
// expected-error@below {{the operations must form a connected component}}
pdl.pattern : benefit(1) {
  %op1 = operation "foo.op"
  %op2 = operation "bar.op"
  // expected-note@below {{see a disconnected value / operation here}}
  %val = result 0 of %op2
  rewrite %op1 with "foo"(%val : !pdl.value)
}

// -----
// expected-error@below {{the operations must form a connected component}}
pdl.pattern : benefit(1) {
  %type = type
  %op1 = operation "foo.op" -> (%type : !pdl.type)
  %val = result 0 of %op1
  %op2 = operation "bar.op"(%val : !pdl.value)
  // expected-note@below {{see a disconnected value / operation here}}
  %op3 = operation "baz.op"
  rewrite {
    erase %op1
    erase %op2
    erase %op3
  }
}

// -----

pdl.pattern : benefit(1) {
  %type = type : i32
  %root = operation "foo.op" -> (%type : !pdl.type)
  rewrite %root {
    %newOp = operation "foo.op" -> (%type : !pdl.type)
    %newResult = result 0 of %newOp

    // expected-error@below {{expected no replacement values to be provided when the replacement operation is present}}
    "pdl.replace"(%root, %newOp, %newResult) {
      operand_segment_sizes = dense<1> : vector<3xi32>
    } : (!pdl.operation, !pdl.operation, !pdl.value) -> ()
  }
}

// -----

//===----------------------------------------------------------------------===//
// pdl::ResultsOp
//===----------------------------------------------------------------------===//

pdl.pattern : benefit(1) {
  %root = operation "foo.op"
  // expected-error@below {{expected `pdl.range<value>` result type when no index is specified, but got: '!pdl.value'}}
  %results = "pdl.results"(%root) : (!pdl.operation) -> !pdl.value
  rewrite %root with "rewriter"
}

// -----

//===----------------------------------------------------------------------===//
// pdl::RewriteOp
//===----------------------------------------------------------------------===//

pdl.pattern : benefit(1) {
  %op = operation "foo.op"

  // expected-error@below {{expected rewrite region to be non-empty if external name is not specified}}
  "pdl.rewrite"(%op) ({}) {
    operand_segment_sizes = dense<[1,0]> : vector<2xi32>
  } : (!pdl.operation) -> ()
}

// -----

pdl.pattern : benefit(1) {
  %op = operation "foo.op"

  // expected-error@below {{expected no external arguments when the rewrite is specified inline}}
  "pdl.rewrite"(%op, %op) ({
    ^bb1:
  }) {
    operand_segment_sizes = dense<1> : vector<2xi32>
  }: (!pdl.operation, !pdl.operation) -> ()
}

// -----

pdl.pattern : benefit(1) {
  %op = operation "foo.op"

  // expected-error@below {{expected no external constant parameters when the rewrite is specified inline}}
  "pdl.rewrite"(%op) ({
    ^bb1:
  }) {
    operand_segment_sizes = dense<[1,0]> : vector<2xi32>,
    externalConstParams = []} : (!pdl.operation) -> ()
}

// -----

pdl.pattern : benefit(1) {
  %op = operation "foo.op"

  // expected-error@below {{expected rewrite region to be empty when rewrite is external}}
  "pdl.rewrite"(%op) ({
    ^bb1:
  }) {
    name = "foo",
    operand_segment_sizes = dense<[1,0]> : vector<2xi32>
  } : (!pdl.operation) -> ()
}

// -----

//===----------------------------------------------------------------------===//
// pdl::TypeOp
//===----------------------------------------------------------------------===//

pdl.pattern : benefit(1) {
  // expected-error@below {{expected a bindable user when defined in the matcher body of a `pdl.pattern`}}
  %unused = type

  %op = operation "foo.op"
  rewrite %op with "rewriter"
}

// -----

//===----------------------------------------------------------------------===//
// pdl::TypesOp
//===----------------------------------------------------------------------===//

pdl.pattern : benefit(1) {
  // expected-error@below {{expected a bindable user when defined in the matcher body of a `pdl.pattern`}}
  %unused = types

  %op = operation "foo.op"
  rewrite %op with "rewriter"
}
