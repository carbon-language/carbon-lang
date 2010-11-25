// PR1013
// Check to make sure debug symbols use the correct name for globals and
// functions.  Will not assemble if it fails to.
// RUN: %llvmgcc_only -O0 -g -c %s

int foo __asm__("f\001oo");

int bar() {
  return foo;
}
