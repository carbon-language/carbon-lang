// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/format/testdata/wrapping/struct_literal.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/format/testdata/wrapping/struct_literal.carbon

// --- test.carbon

fn MakeDefaultConfig() -> auto {
  return {.first_field = alpha_value, .second_field = beta_value, .third_field = gamma_value};
}

fn AssignFittingStruct() {
  var server_config: ServerConfiguration = {.host = "localhost", .port = 8080, .tls = true};
}

fn ReturnWideSingleFieldType() -> {.very_long_field_name_in_return_type: VeryLongFieldTypeName} {
  return {.very_long_field_name_in_return_type = default_value};
}

fn BuildMultiline() {
  var configuration_with_an_extra_long_variable_name_here: ServerConfiguration = {
    .port = 8080,
  };
}

// --- AUTOUPDATE-SPLIT

// CHECK:STDOUT: fn MakeDefaultConfig() -> auto {
// CHECK:STDOUT:   return {.first_field = alpha_value, .second_field = beta_value,
// CHECK:STDOUT:           .third_field = gamma_value};
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn AssignFittingStruct() {
// CHECK:STDOUT:   var server_config: ServerConfiguration =
// CHECK:STDOUT:       {.host = "localhost", .port = 8080, .tls = true};
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn ReturnWideSingleFieldType() -> {
// CHECK:STDOUT:     .very_long_field_name_in_return_type: VeryLongFieldTypeName} {
// CHECK:STDOUT:   return {.very_long_field_name_in_return_type = default_value};
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn BuildMultiline() {
// CHECK:STDOUT:   var configuration_with_an_extra_long_variable_name_here:
// CHECK:STDOUT:       ServerConfiguration = {
// CHECK:STDOUT:     .port = 8080,
// CHECK:STDOUT:   };
// CHECK:STDOUT: }
