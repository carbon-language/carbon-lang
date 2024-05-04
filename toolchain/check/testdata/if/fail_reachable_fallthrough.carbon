// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn If1(b: bool) -> i32 {
  if (b) {
    return 1;
  } else {
  }
// CHECK:STDERR: fail_reachable_fallthrough.carbon:[[@LINE+4]]:1: ERROR: Missing `return` at end of function with declared return type.
// CHECK:STDERR: }
// CHECK:STDERR: ^
// CHECK:STDERR:
}

fn If2(b: bool) -> i32 {
  if (b) {
  } else {
    return 2;
  }
// CHECK:STDERR: fail_reachable_fallthrough.carbon:[[@LINE+4]]:1: ERROR: Missing `return` at end of function with declared return type.
// CHECK:STDERR: }
// CHECK:STDERR: ^
// CHECK:STDERR:
}

fn If3(b: bool) -> i32 {
  if (b) {
    return 1;
  }
// CHECK:STDERR: fail_reachable_fallthrough.carbon:[[@LINE+3]]:1: ERROR: Missing `return` at end of function with declared return type.
// CHECK:STDERR: }
// CHECK:STDERR: ^
}

// CHECK:STDOUT: --- fail_reachable_fallthrough.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %If1: type = fn_type @If1 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: If1 = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %If2: type = fn_type @If2 [template]
// CHECK:STDOUT:   %struct.2: If2 = struct_value () [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %If3: type = fn_type @If3 [template]
// CHECK:STDOUT:   %struct.3: If3 = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .If1 = %If1.decl
// CHECK:STDOUT:     .If2 = %If2.decl
// CHECK:STDOUT:     .If3 = %If3.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %If1.decl: If1 = fn_decl @If1 [template = constants.%struct.1] {
// CHECK:STDOUT:     %b.loc7_8.1: bool = param b
// CHECK:STDOUT:     @If1.%b: bool = bind_name b, %b.loc7_8.1
// CHECK:STDOUT:     @If1.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %If2.decl: If2 = fn_decl @If2 [template = constants.%struct.2] {
// CHECK:STDOUT:     %b.loc18_8.1: bool = param b
// CHECK:STDOUT:     @If2.%b: bool = bind_name b, %b.loc18_8.1
// CHECK:STDOUT:     @If2.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %If3.decl: If3 = fn_decl @If3 [template = constants.%struct.3] {
// CHECK:STDOUT:     %b.loc29_8.1: bool = param b
// CHECK:STDOUT:     @If3.%b: bool = bind_name b, %b.loc29_8.1
// CHECK:STDOUT:     @If3.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @If1(%b: bool) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %b.ref: bool = name_ref b, %b
// CHECK:STDOUT:   if %b.ref br !if.then else br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then:
// CHECK:STDOUT:   %.loc9: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   return %.loc9
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else:
// CHECK:STDOUT:   br !if.done
// CHECK:STDOUT:
// CHECK:STDOUT: !if.done:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @If2(%b: bool) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %b.ref: bool = name_ref b, %b
// CHECK:STDOUT:   if %b.ref br !if.then else br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then:
// CHECK:STDOUT:   br !if.done
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else:
// CHECK:STDOUT:   %.loc21: i32 = int_literal 2 [template = constants.%.3]
// CHECK:STDOUT:   return %.loc21
// CHECK:STDOUT:
// CHECK:STDOUT: !if.done:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @If3(%b: bool) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %b.ref: bool = name_ref b, %b
// CHECK:STDOUT:   if %b.ref br !if.then else br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then:
// CHECK:STDOUT:   %.loc31: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   return %.loc31
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else:
// CHECK:STDOUT: }
// CHECK:STDOUT:
