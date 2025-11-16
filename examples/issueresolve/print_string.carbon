// Example: Printing a string using character iteration.
// This avoids unsupported indexing (`msg[i]`) because `str` does not
// implement `Core.IndexWith(i32)` yet.

import Core library "io";

fn PrintStr(msg: str) {
  // Carbon supports iteration over the characters of a string.
  for (c: char in msg) {
    Core.PrintChar(c);
  }
}

fn Run() {
  PrintStr("Hello World!\n");
}
