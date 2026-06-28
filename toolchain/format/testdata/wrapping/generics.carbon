// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/format/testdata/wrapping/generics.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/format/testdata/wrapping/generics.carbon

// --- test.carbon

fn Combine[template CombinerType: type, ResultType: type](combiner: CombinerType, initial_value: ResultType) -> ResultType;

class AssociativeContainer(KeyElement: type, ValueElement: type, template Hasher: type) {
  fn Insert[self: Self](key: KeyElement, value: ValueElement) -> bool;
}

// --- AUTOUPDATE-SPLIT

// CHECK:STDOUT: fn Combine[template CombinerType: type,
// CHECK:STDOUT:            ResultType: type](combiner: CombinerType,
// CHECK:STDOUT:                              initial_value: ResultType) -> ResultType;
// CHECK:STDOUT:
// CHECK:STDOUT: class AssociativeContainer(KeyElement: type, ValueElement: type,
// CHECK:STDOUT:                            template Hasher: type) {
// CHECK:STDOUT:   fn Insert[self: Self](key: KeyElement, value: ValueElement) -> bool;
// CHECK:STDOUT: }
