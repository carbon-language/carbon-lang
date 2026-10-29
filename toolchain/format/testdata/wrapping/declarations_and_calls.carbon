// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/format/testdata/wrapping/declarations_and_calls.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/format/testdata/wrapping/declarations_and_calls.carbon

// --- test.carbon

fn RegisterHandler(event_name: String, priority: i32, callback: Callback) -> bool {
  RegisterHandler(event_name, default_priority, my_callback, extra_argument_value);
  TakeTheAddress(some_rather_long_variable_name_that_makes_this_overflow_xyzzyx);
  if (aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa and bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb) {
    return false;
  }
  return true;
}

fn Combine[template CombinerType: type, ResultType: type](combiner: CombinerType, initial_value: ResultType) -> ResultType;

class AssociativeContainer(KeyElement: type, ValueElement: type, template Hasher: type) {
  fn Insert[self: Self](key: KeyElement, value: ValueElement) -> bool;
  private fn Configure() -> ValidationResultStatusNameGoesHereAndOverflowsXyzzyy;
}

// Parameters wrap after `(` or before `->` rather than splitting a parameter's
// binding colon; statement-level bindings wrap after `:` as a last resort.
fn MakeSortedElementContainer(initial_capacity: i32) -> SortedContainerOfElement;

fn Q(some_extremely_long_parameter_name_that_overflows_all_on_its_own: i32) -> i32;

fn T() {
  var accumulated_statistics_record_value: ProcessingStatisticsRecordContainerXyz;
  for (var item_record: ProcessingItemRecord in available_processing_items_list) {}
}

// --- AUTOUPDATE-SPLIT

// CHECK:STDOUT: fn RegisterHandler(event_name: String, priority: i32,
// CHECK:STDOUT:                    callback: Callback) -> bool {
// CHECK:STDOUT:   RegisterHandler(event_name, default_priority, my_callback,
// CHECK:STDOUT:                   extra_argument_value);
// CHECK:STDOUT:   TakeTheAddress(
// CHECK:STDOUT:       some_rather_long_variable_name_that_makes_this_overflow_xyzzyx);
// CHECK:STDOUT:   if (aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa and
// CHECK:STDOUT:       bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb) {
// CHECK:STDOUT:     return false;
// CHECK:STDOUT:   }
// CHECK:STDOUT:   return true;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn Combine[template CombinerType: type, ResultType: type](
// CHECK:STDOUT:     combiner: CombinerType, initial_value: ResultType) -> ResultType;
// CHECK:STDOUT:
// CHECK:STDOUT: class AssociativeContainer(KeyElement: type, ValueElement: type,
// CHECK:STDOUT:                            template Hasher: type) {
// CHECK:STDOUT:   fn Insert[self: Self](key: KeyElement, value: ValueElement) -> bool;
// CHECK:STDOUT:   private fn Configure()
// CHECK:STDOUT:       -> ValidationResultStatusNameGoesHereAndOverflowsXyzzyy;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: // Parameters wrap after `(` or before `->` rather than splitting a parameter's
// CHECK:STDOUT: // binding colon; statement-level bindings wrap after `:` as a last resort.
// CHECK:STDOUT: fn MakeSortedElementContainer(
// CHECK:STDOUT:     initial_capacity: i32) -> SortedContainerOfElement;
// CHECK:STDOUT:
// CHECK:STDOUT: fn Q(some_extremely_long_parameter_name_that_overflows_all_on_its_own: i32)
// CHECK:STDOUT:     -> i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn T() {
// CHECK:STDOUT:   var accumulated_statistics_record_value:
// CHECK:STDOUT:       ProcessingStatisticsRecordContainerXyz;
// CHECK:STDOUT:   for (var item_record: ProcessingItemRecord in
// CHECK:STDOUT:        available_processing_items_list) {}
// CHECK:STDOUT: }
