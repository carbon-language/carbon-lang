// RUN: clang-cc %s -fsyntax-only -verify -fblocks

int main() {
  void (^b) (int arg, const char * format, ...) __attribute__ ((__format__ (__printf__, 1, 3))) =   // expected-error {{format argument not a string type}}
    ^ __attribute__ ((__format__ (__printf__, 1, 3))) (int arg, const char * format, ...) {}; // expected-error {{format argument not a string type}}
 
  void (^z) (int arg, const char * format, ...) __attribute__ ((__format__ (__printf__, 2, 3))) = ^ __attribute__ ((__format__ (__printf__, 2, 3))) (int arg, const char * format, ...) {};

  // FIXME: argument type poking not yet supportted.
  z(1, "%s", 1); /* { dg-warning "format \\'\%s\\' expects type \\'char \\*\\'\, but argument 3 has type \\'int\\'" } */
  z(1, "%s", "HELLO"); // OK
}
