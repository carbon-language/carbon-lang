// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/method/fail_call_without_method_syntax.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/method/fail_call_without_method_syntax.carbon

// --- fail_call_without_method_syntax.carbon
library "[[@TEST_NAME]]";

class Point {
  fn Get(self) -> i32 { return self.x; }

  var x: i32;
}

class Other {}

// When a method is called as an ordinary function, `self` is an ordinary
// argument, so the usual arity and conversion rules apply.
fn TooFew() -> i32 {
  // CHECK:STDERR:        ╭─┤ fail_call_without_method_syntax.carbon:[[@LINE+11]]:10
  // CHECK:STDERR:        │   fn Get(self) -> i32 { return self.x; }
  // CHECK:STDERR:        ·   ──────────┬──────────
  // CHECK:STDERR:        ·             ╰──┤ calling function declared here [InCallToEntity]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR: ->     │   return Point.Get();
  // CHECK:STDERR:        ·          ━━━━━┳━━━━━
  // CHECK:STDERR:        ·               ┗━━┫ error: 0 arguments passed to function expecting 1 argument
  // CHECK:STDERR:        ·                  ┃        [CallArgCountMismatch]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  return Point.Get();
}

fn TooMany(p: Point) -> i32 {
  // CHECK:STDERR:        ╭─┤ fail_call_without_method_syntax.carbon:[[@LINE+11]]:10
  // CHECK:STDERR:        │   fn Get(self) -> i32 { return self.x; }
  // CHECK:STDERR:        ·   ──────────┬──────────
  // CHECK:STDERR:        ·             ╰──┤ calling function declared here [InCallToEntity]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR: ->     │   return Point.Get(p, p);
  // CHECK:STDERR:        ·          ━━━━━━━┳━━━━━━━
  // CHECK:STDERR:        ·                 ┗━━┫ error: 2 arguments passed to function expecting 1 argument
  // CHECK:STDERR:        ·                    ┃        [CallArgCountMismatch]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  return Point.Get(p, p);
}

fn WrongSelfType(o: Other) -> i32 {
  // CHECK:STDERR:        ╭─┤ fail_call_without_method_syntax.carbon:[[@LINE+15]]:20
  // CHECK:STDERR:        │   fn Get(self) -> i32 { return self.x; }
  // CHECK:STDERR:        ·          ──┬─
  // CHECK:STDERR:        ·            ╰──┤ initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR: ->     │   return Point.Get(o);
  // CHECK:STDERR:        ·                    ┳
  // CHECK:STDERR:        ·                    ┗━━┫ error: cannot implicitly convert expression of type
  // CHECK:STDERR:        ·                    │  ┃        `Other` to `Point`
  // CHECK:STDERR:        ·                    │  ┃        [ConversionFailure]
  // CHECK:STDERR:        ·                    ╰────┤ type `Other` does not implement interface
  // CHECK:STDERR:        ·                         │ `Core.ImplicitAs(Point)`
  // CHECK:STDERR:        ·                         │ [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  return Point.Get(o);
}
