// RUN: mlir-opt %s -test-print-callgraph -split-input-file -allow-unregistered-dialect 2>&1 | FileCheck %s

// CHECK-LABEL: Testing : "simple"
module attributes {test.name = "simple"} {

  // CHECK: Node{{.*}}func_a
  func @func_a() {
    return
  }

  func private @func_b()

  // CHECK: Node{{.*}}func_c
  // CHECK-NEXT: Call-Edge{{.*}}External-Node
  func @func_c() {
    call @func_b() : () -> ()
    return
  }

  // CHECK: Node{{.*}}func_d
  // CHECK-NEXT: Call-Edge{{.*}}func_c
  func @func_d() {
    call @func_c() : () -> ()
    return
  }

  // CHECK: Node{{.*}}func_e
  // CHECK-DAG: Call-Edge{{.*}}func_c
  // CHECK-DAG: Call-Edge{{.*}}func_d
  // CHECK-DAG: Call-Edge{{.*}}func_e
  func @func_e() {
    call @func_c() : () -> ()
    call @func_d() : () -> ()
    call @func_e() : () -> ()
    return
  }

  // CHECK: Node{{.*}}func_f
  // CHECK: Child-Edge{{.*}}test.functional_region_op
  // CHECK: Call-Edge{{.*}}test.functional_region_op
  func @func_f() {
    // CHECK: Node{{.*}}test.functional_region_op
    // CHECK: Call-Edge{{.*}}func_f
    %fn = "test.functional_region_op"() ({
      call @func_f() : () -> ()
      "test.return"() : () -> ()
    }) : () -> (() -> ())

    call_indirect %fn() : () -> ()
    return
  }
}

// -----

// CHECK-LABEL: Testing : "nested"
module attributes {test.name = "nested"} {
  module @nested_module {
    // CHECK: Node{{.*}}func_a
    func @func_a() {
      return
    }
  }

  // CHECK: Node{{.*}}func_b
  // CHECK: Call-Edge{{.*}}func_a
  func @func_b() {
    "test.conversion_call_op"() { callee = @nested_module::@func_a } : () -> ()
    return
  }
}
