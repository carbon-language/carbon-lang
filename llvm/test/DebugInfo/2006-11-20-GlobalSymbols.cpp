// Check to make sure debug symbols use the correct name for globals and
// functions.  Will not assemble if it fails to.
// RUN: %llvmgcc -O0 -g -c %s

// PR1013

int foo __asm__("f\001oo");

int bar() {
  return foo;
}
