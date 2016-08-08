// RUN: %clang_cc1 -triple i386-pc-linux -fms-extensions -fsyntax-only -verify %s

int __attribute__((ms_hook_prologue)) foo(int a, int b) {
  return a+b;
}

// expected-note@+2{{conflicting attribute is here}}
// expected-error@+1{{'naked' and 'ms_hook_prologue' attributes are not compatible}}
__declspec(naked) int __attribute__((ms_hook_prologue)) bar(int a, int b) {
}

// expected-note@+2{{conflicting attribute is here}}
// expected-error@+1{{'__forceinline' and 'ms_hook_prologue' attributes are not compatible}}
__forceinline int __attribute__((ms_hook_prologue)) baz(int a, int b) {
  return a-b;
}

// expected-warning@+1{{'ms_hook_prologue' attribute only applies to functions}}
int x __attribute__((ms_hook_prologue));

// expected-error@+1{{'ms_hook_prologue' attribute takes no arguments}}
int f(int a, int b) __attribute__((ms_hook_prologue(2)));
