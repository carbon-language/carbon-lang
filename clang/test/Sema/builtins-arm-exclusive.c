// RUN: %clang_cc1 -triple armv7 -fsyntax-only -verify %s

struct Simple {
  char a, b;
};

int test_ldrex(char *addr) {
  int sum = 0;
  sum += __builtin_arm_ldrex(addr);
  sum += __builtin_arm_ldrex((short *)addr);
  sum += __builtin_arm_ldrex((int *)addr);
  sum += __builtin_arm_ldrex((long long *)addr);
  sum += __builtin_arm_ldrex((float *)addr);
  sum += __builtin_arm_ldrex((double *)addr);
  sum += *__builtin_arm_ldrex((int **)addr);
  sum += __builtin_arm_ldrex((struct Simple **)addr)->a;
  sum += __builtin_arm_ldrex((volatile char *)addr);
  sum += __builtin_arm_ldrex((const volatile char *)addr);

  // In principle this might be valid, but stick to ints and floats for scalar
  // types at the moment.
  sum += __builtin_arm_ldrex((struct Simple *)addr).a; // expected-error {{address argument to atomic builtin must be a pointer to}}

  sum += __builtin_arm_ldrex((__int128 *)addr); // expected-error {{__int128 is not supported on this target}} expected-error {{address argument to load or store exclusive builtin must be a pointer to 1,2,4 or 8 byte type}}

  __builtin_arm_ldrex(); // expected-error {{too few arguments to function call}}
  __builtin_arm_ldrex(1, 2); // expected-error {{too many arguments to function call}}
  return sum;
}

int test_strex(char *addr) {
  int res = 0;
  struct Simple var = {0};
  res |= __builtin_arm_strex(4, addr);
  res |= __builtin_arm_strex(42, (short *)addr);
  res |= __builtin_arm_strex(42, (int *)addr);
  res |= __builtin_arm_strex(42, (long long *)addr);
  res |= __builtin_arm_strex(2.71828f, (float *)addr);
  res |= __builtin_arm_strex(3.14159, (double *)addr);
  res |= __builtin_arm_strex(&var, (struct Simple **)addr);

  res |= __builtin_arm_strex(42, (volatile char *)addr);
  res |= __builtin_arm_strex(42, (char *const)addr);
  res |= __builtin_arm_strex(42, (const char *)addr); // expected-warning {{passing 'const char *' to parameter of type 'volatile char *' discards qualifiers}}


  res |= __builtin_arm_strex(var, (struct Simple *)addr); // expected-error {{address argument to atomic builtin must be a pointer to}}
  res |= __builtin_arm_strex(var, (struct Simple **)addr); // expected-error {{passing 'struct Simple' to parameter of incompatible type 'struct Simple *'}}
  res |= __builtin_arm_strex(&var, (struct Simple **)addr).a; // expected-error {{is not a structure or union}}

  res |= __builtin_arm_strex(1, (__int128 *)addr); // expected-error {{__int128 is not supported on this target}} expected-error {{address argument to load or store exclusive builtin must be a pointer to 1,2,4 or 8 byte type}}

  __builtin_arm_strex(1); // expected-error {{too few arguments to function call}}
  __builtin_arm_strex(1, 2, 3); // expected-error {{too many arguments to function call}}
  return res;
}

void test_clrex() {
  __builtin_arm_clrex();
  __builtin_arm_clrex(1); // expected-error {{too many arguments to function call}}
}
