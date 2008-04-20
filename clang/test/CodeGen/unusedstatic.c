// RUN: clang %s -emit-llvm -verify
// PR1998
// PR2236
static void a (void);
void b (void) { a (); }
static void a(void) {}
static void c(void) {}  // expected-warning {{static 'c' defined but not used}}
