// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Inline C++ is scoped `meta.embedded.block.cpp`, so an editor that has a C++
// grammar highlights them as C++; these renderings come from the Carbon grammar
// alone, so that content shows plain.

package Sample library "interop";

import Cpp library "<cstdio>";
import Cpp library "<string_view>";

import Cpp inline '''c++
int FromImport() { return 1; }
''';

import Cpp inline "int OneLiner() { return 2; }";

inline Cpp '''
int FromBlock() { return 3; }
''';

fn Uses() {
  var value: Cpp.int = 1 as Cpp.int;
  var widened: Cpp.size_t = value as Cpp.size_t;
  var back: i32 = value unsafe as i32;
  Cpp.printf("%d\n", value);
}
