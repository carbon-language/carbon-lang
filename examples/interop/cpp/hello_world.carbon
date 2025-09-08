// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import Cpp inline "#include <cstdio>";

fn Run() {
  // TODO: Requires operator <<, declaration type Var (cout) and class with
  // virtual bases (basic_ostream).
  // Cpp.std.cout << "Hello world!\n";

  // TODO: Requires variadic function.
  // Cpp.printf("Hello world!\n");

  // TODO: Requires nullable pointer.
  // Cpp.puts("Hello world!\n");

  // TODO: Requires nullable void pointer.
  // Cpp.write(1, "Hello world!\n", 13);

  // TODO: Requires Core.String API.
  // let message: str = "Hello world!\n\n";
  // for (c: Core.Char in msg) {
  //   Cpp.putchar((c as u8) as i32);
  // }

  let message: array(Core.Char, 13) =
      ('H', 'e', 'l', 'l', 'o', ' ', 'w', 'o', 'r', 'l', 'd', '!', '\n');
  for (c: Core.Char in message) {
    // TODO: u8 should probably have an implicit cast to i32.
    Cpp.putchar((c as u8) as i32);
  }
}
