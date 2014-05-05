// RUN: %clang_cc1 -emit-llvm-only %s -triple=i386-pc-win32 -mconstructor-aliases -fcxx-exceptions -fexceptions -fno-rtti -verify -DTRY
// RUN: %clang_cc1 -emit-llvm-only %s -triple=i386-pc-win32 -mconstructor-aliases -fcxx-exceptions -fexceptions -fno-rtti -verify -DTHROW

void external();

inline void not_emitted() {
  throw int(13); // no error
}

int main() {
  int rv = 0;
#ifdef TRY
  try { // expected-error {{cannot compile this try statement yet}}
    external();
  } catch (int) {
    rv = 1;
  }
#endif
#ifdef THROW
  throw int(42); // expected-error {{cannot compile this throw expression yet}}
#endif
  return rv;
}
