//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// XFAIL: *

// Make sure the test DOES NOT pass if the expected diagnostic is wrong.

struct Foo { };
typedef Foo::x x; // expected-error{{this is not found in the errors}}
