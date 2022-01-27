// RUN: mlir-opt -allow-unregistered-dialect -test-legalize-patterns -verify-diagnostics -test-legalize-mode=analysis %s | FileCheck %s
// expected-remark@-2 {{op 'builtin.module' is legalizable}}

// expected-remark@+1 {{op 'builtin.func' is legalizable}}
func @test(%arg0: f32) {
  // expected-remark@+1 {{op 'test.illegal_op_a' is legalizable}}
  %result = "test.illegal_op_a"() : () -> (i32)
  "foo.region"() ({
      // expected-remark@+1 {{op 'test.invalid' is legalizable}}
      "test.invalid"() : () -> ()
  }) : () -> ()
  return
}

// Check that none of the legalizable operations were modified.
// CHECK-LABEL: func @test
// CHECK-NEXT: "test.illegal_op_a"
// CHECK: "test.invalid"
