// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import Cpp library "<cstdio>";
import Cpp library "<iostream>";
import Cpp library "<string_view>";
import Cpp library "<unistd.h>";

// Demonstrate passing `char`s to a C++ function.
fn HelloPutchar() {
  let message: array(char, 13) =
      ('H', 'e', 'l', 'l', 'o', ' ', 'w', 'o', 'r', 'l', 'd', '!', '\n');
  for (c: char in message) {
    // TODO: u8 should probably have an implicit cast to i32.
    Cpp.putchar((c as u8) as i32);
  }

  // TODO: Use a loop over a string literal instead. Requires IndexWith support
  // for str.
  //
  // let message: str = "Hello world!\n";
  // for (c: char in message) {
  //   Cpp.putchar((c as u8) as i32);
  // }
}

// Demonstrate passing a null-terminated string to a C++ function.
fn HelloStdio() {
  // TODO: Requires mapping from Optional(const char*) into C++.
  // TODO: There should be a better way to interact with functions that expect a
  // null-terminated string.
  // Cpp.puts(Cpp.std.data("Hello world!\n\0"));

  // TODO: Requires variadic function support.
  // Cpp.printf("Hello world!\n\0");
}

// Demonstrate passing a string as a void pointer to a C++ function.
fn HelloWrite() {
  // TODO: Requires mapping from Optional(const char*) into a C++ const void*.
  // let s: str = "Hello world!\n";
  // Cpp.write(1, Cpp.std.data(s), Cpp.std.size(s));
}

fn HelloIostreams() {
  Cpp.std.cout << "Hello world!\n";
}

fn Run() {
  HelloPutchar();
  HelloStdio();
  HelloWrite();
  HelloIostreams();
}
