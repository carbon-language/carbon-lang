// RUN: cp %s %t
// RUN: not %clang_cc1 -fixit -Werror -x c++ -std=c++98 %t
// RUN: %clang_cc1 -fsyntax-only -Werror -x c++ -std=c++98 %t
// RUN: cp %s %t
// RUN: not %clang_cc1 -DUSE_BUILTINS -fixit -Werror -x c++ -std=c++98 %t
// RUN: %clang_cc1 -DUSE_BUILTINS -fsyntax-only -Werror -x c++ -std=c++98 %t

extern "C" void *memcpy(void *s1, const void *s2, unsigned n);

#ifdef USE_BUILTINS
# define BUILTIN(f) __builtin_ ## f
#else
# define BUILTIN(f) f
#endif

#define memcpy BUILTIN(memcpy)

int testFixits(int *to, int *from) {
  memcpy(to, from, sizeof(to)); // \
         // expected-warning {{argument to 'sizeof' in 'memcpy' call is the same expression as the destination; did you mean to dereference it?}}
  memcpy(0, &from, sizeof(&from)); // \
         // expected-warning {{argument to 'sizeof' in 'memcpy' call is the same expression as the source; did you mean to remove the addressof?}}
  return 0;
}
