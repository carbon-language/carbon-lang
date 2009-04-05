// RUN: clang-cc -fsyntax-only -verify %s -triple x86_64-apple-darwin9

// rdar://6726818
void f1() {
  const __builtin_va_list args2;
  (void)__builtin_va_arg(args2, int); // expected-warning {{va_arg applied to va_list type 'struct __va_list_tag const *' with unexpected qualifiers}}
}

