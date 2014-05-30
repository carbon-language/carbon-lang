// Test that -Weverything does not trigger any backend remarks.
//
// This was triggering backend remarks for which there were no frontend
// flags to filter them. The handler in BackendConsumer::DiagnosticHandlerImpl
// should not emitting diagnostics for unhandled kinds.

// RUN: %clang -target x86_64-unknown-unknown -c -S -Weverything -O0 -o /dev/null %s 2> %t.err
// RUN: FileCheck < %t.err %s

typedef __char32_t char32_t;
typedef long unsigned int size_t;
template <class _CharT>
struct __attribute__((__type_visibility__("default"))) char_traits;

template <>
struct __attribute__((__type_visibility__("default"))) char_traits<char32_t> {
  typedef char32_t char_type;
  static void assign(char_type& __c1, const char_type& __c2) throw() {
    __c1 = __c2;
  }
  static char_type* move(char_type* __s1, const char_type* __s2, size_t __n);
};
char32_t* char_traits<char32_t>::move(char_type* __s1, const char_type* __s2,
                                      size_t __n) {
  { assign(*--__s1, *--__s2); }
}

// CHECK-NOT: {{^remark:}}
