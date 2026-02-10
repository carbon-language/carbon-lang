// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/uint.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/primitives/string_literals.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/primitives/string_literals.carbon

// --- fail_no_string.carbon

library "[[@TEST_NAME]]";

fn F() {
  // CHECK:STDERR: fail_no_string.carbon:[[@LINE+4]]:3: error: name `Core.String` implicitly referenced here, but not found [CoreNameNotFound]
  // CHECK:STDERR:   "hello";
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  "hello";
}

// --- unusual_string.carbon

package Core library "[[@TEST_NAME]]";

import library "prelude/parts/uint";

class String {
  var begin: u8*;
  var end: u8*;
}

// --- fail_use_unusual_string.carbon

import Core library "unusual_string";

// CHECK:STDERR: fail_use_unusual_string.carbon:[[@LINE+4]]:14: error: unexpected representation for type `str` [StringLiteralTypeUnexpected]
// CHECK:STDERR: let x: str = "hello";
// CHECK:STDERR:              ^~~~~~~
// CHECK:STDERR:
let x: str = "hello";

// --- incomplete_string.carbon

package Core library "[[@TEST_NAME]]";

class String;

// --- fail_use_incomplete_string.carbon

library "[[@TEST_NAME]]";

import Core library "incomplete_string";

fn F() {
  // CHECK:STDERR: fail_use_incomplete_string.carbon:[[@LINE+7]]:3: error: type `Core.String` is incomplete [StringLiteralTypeIncomplete]
  // CHECK:STDERR:   "hello";
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR: incomplete_string.carbon:4:1: note: class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR: class String;
  // CHECK:STDERR: ^~~~~~~~~~~~~
  // CHECK:STDERR:
  "hello";
}

// --- small_string.carbon

package Core library "[[@TEST_NAME]]";

import library "prelude/parts/uint";

class String {
  var ptr: u8*;
  // An intentionally small `size` field to test the behavior when a string
  // literal is large enough that its size can't be represented.
  var size: u8;
}

// --- use_small_string.carbon

library "[[@TEST_NAME]]";

import Core library "small_string";

//@dump-sem-ir-begin
let x: str = "hello";
//@dump-sem-ir-end

// --- fail_overfill_small_string.carbon

library "[[@TEST_NAME]]";

import Core library "small_string";

//@dump-sem-ir-begin
// CHECK:STDERR: fail_overfill_small_string.carbon:[[@LINE+4]]:14: error: string literal is too long [StringLiteralTooLong]
// CHECK:STDERR: let x: str = '''
// CHECK:STDERR:              ^~~
// CHECK:STDERR:
let x: str = '''
A string so long its size does not fit in `u8`. This results in an error,
because we can't form a `Core.String` value that describes this string.

This string contains 256 characters. Among them are:

* 6 `g`s,
* 21 `s`s,
* 9 newlines,
* and only one `p`.
''';
//@dump-sem-ir-end

// CHECK:STDOUT: --- use_small_string.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %str.ee0: type = class_type @String [concrete]
// CHECK:STDOUT:   %int_8: Core.IntLiteral = int_value 8 [concrete]
// CHECK:STDOUT:   %u8: type = class_type @UInt, @UInt(%int_8) [concrete]
// CHECK:STDOUT:   %ptr.3e8: type = ptr_type %u8 [concrete]
// CHECK:STDOUT:   %pattern_type.461: type = pattern_type %str.ee0 [concrete]
// CHECK:STDOUT:   %str.3b1: %ptr.3e8 = string_literal "hello" [concrete]
// CHECK:STDOUT:   %int_5: %u8 = int_value 5 [concrete]
// CHECK:STDOUT:   %String.val: %str.ee0 = struct_value (%str.3b1, %int_5) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.461 = value_binding_pattern x [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: %str.ee0 = value_binding x, @__global_init.%String.val
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %str: %ptr.3e8 = string_literal "hello" [concrete = constants.%str.3b1]
// CHECK:STDOUT:   %int_5: %u8 = int_value 5 [concrete = constants.%int_5]
// CHECK:STDOUT:   %String.val: %str.ee0 = struct_value (%str, %int_5) [concrete = constants.%String.val]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_overfill_small_string.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %str.ee0: type = class_type @String [concrete]
// CHECK:STDOUT:   %int_8: Core.IntLiteral = int_value 8 [concrete]
// CHECK:STDOUT:   %u8: type = class_type @UInt, @UInt(%int_8) [concrete]
// CHECK:STDOUT:   %ptr.3e8: type = ptr_type %u8 [concrete]
// CHECK:STDOUT:   %pattern_type.461: type = pattern_type %str.ee0 [concrete]
// CHECK:STDOUT:   %str.294: %ptr.3e8 = string_literal "A string so long its size does not fit in `u8`. This results in an error,\nbecause we can't form a `Core.String` value that describes this string.\n\nThis string contains 256 characters. Among them are:\n\n* 6 `g`s,\n* 21 `s`s,\n* 9 newlines,\n* and only one `p`.\n" [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.461 = value_binding_pattern x [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: %str.ee0 = value_binding x, <error> [concrete = <error>]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %str: %ptr.3e8 = string_literal "A string so long its size does not fit in `u8`. This results in an error,\nbecause we can't form a `Core.String` value that describes this string.\n\nThis string contains 256 characters. Among them are:\n\n* 6 `g`s,\n* 21 `s`s,\n* 9 newlines,\n* and only one `p`.\n" [concrete = constants.%str.294]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
