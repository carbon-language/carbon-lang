// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/format/testdata/wrapping/declarations.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/format/testdata/wrapping/declarations.carbon

// --- test.carbon

// The return type moves to its own line before the parameter's binding colon
// is split from its type.
fn MakeSortedElementContainer(initial_capacity: i32) -> SortedContainerOfElement;

// The same preference with a parameter name too long to share a line with
// anything else.
fn Q(some_extremely_long_parameter_name_that_overflows_all_on_its_own: i32) -> i32;

// A modifier or introducer keyword is never split from its declaration; the
// return type breaks instead.
class C {
  private fn Configure() -> ValidationResultStatusNameGoesHereAndOverflowsXyzzyy;
}

// With no other legal break, a statement-level binding wraps after its colon
// as the last resort.
fn T() {
  var accumulated_statistics_record_value: ProcessingStatisticsRecordContainerXyz;
}

// --- AUTOUPDATE-SPLIT

// CHECK:STDOUT: // The return type moves to its own line before the parameter's binding colon
// CHECK:STDOUT: // is split from its type.
// CHECK:STDOUT: fn MakeSortedElementContainer(initial_capacity: i32)
// CHECK:STDOUT:     -> SortedContainerOfElement;
// CHECK:STDOUT:
// CHECK:STDOUT: // The same preference with a parameter name too long to share a line with
// CHECK:STDOUT: // anything else.
// CHECK:STDOUT: fn Q(some_extremely_long_parameter_name_that_overflows_all_on_its_own: i32)
// CHECK:STDOUT:     -> i32;
// CHECK:STDOUT:
// CHECK:STDOUT: // A modifier or introducer keyword is never split from its declaration; the
// CHECK:STDOUT: // return type breaks instead.
// CHECK:STDOUT: class C {
// CHECK:STDOUT:   private fn Configure()
// CHECK:STDOUT:       -> ValidationResultStatusNameGoesHereAndOverflowsXyzzyy;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: // With no other legal break, a statement-level binding wraps after its colon
// CHECK:STDOUT: // as the last resort.
// CHECK:STDOUT: fn T() {
// CHECK:STDOUT:   var accumulated_statistics_record_value:
// CHECK:STDOUT:       ProcessingStatisticsRecordContainerXyz;
// CHECK:STDOUT: }
