// RUN: %clang_cc1 -std=c++11 -verify %s -fms-extensions -triple x86_64-apple-darwin9.0.0

// A ud-suffix cannot be used on string literals in a whole bunch of contexts:

#include "foo"_bar // expected-error {{expected "FILENAME" or <FILENAME>}}
#line 1 "foo"_bar // expected-error {{user-defined suffix cannot be used here}}
# 1 "foo"_bar 1 // expected-error {{user-defined suffix cannot be used here}}
#ident "foo"_bar // expected-error {{user-defined suffix cannot be used here}}
_Pragma("foo"_bar) // expected-error {{user-defined suffix cannot be used here}}
#pragma comment(lib, "foo"_bar) // expected-error {{user-defined suffix cannot be used here}}
_Pragma("comment(lib, \"foo\"_bar)") // expected-error {{user-defined suffix cannot be used here}}
#pragma message "hi"_there // expected-error {{user-defined suffix cannot be used here}} expected-warning {{hi}}
#pragma push_macro("foo"_bar) // expected-error {{user-defined suffix cannot be used here}}
#if __has_warning("-Wan-island-to-discover"_bar) // expected-error {{user-defined suffix cannot be used here}}
#elif __has_include("foo"_bar) // expected-error {{expected "FILENAME" or <FILENAME>}}
#endif

extern "C++"_x {} // expected-error {{user-defined suffix cannot be used here}} expected-error {{unknown linkage language}}

int f() {
  asm("mov %eax, %rdx"_foo); // expected-error {{user-defined suffix cannot be used here}}
}

static_assert(true, "foo"_bar); // expected-error {{user-defined suffix cannot be used here}}

int cake() __attribute__((availability(macosx, unavailable, message = "is a lie"_x))); // expected-error {{user-defined suffix cannot be used here}}

// A ud-suffix cannot be used on character literals in preprocessor constant
// expressions:
#if 'x'_y - u'x'_z // expected-error 2{{character literal with user-defined suffix cannot be used in preprocessor constant expression}}
#error error
#endif

// But they can appear in expressions.
constexpr char operator"" _id(char c) { return c; }
constexpr wchar_t operator"" _id(wchar_t c) { return c; }
constexpr char16_t operator"" _id(char16_t c) { return c; }
constexpr char32_t operator"" _id(char32_t c) { return c; }

using size_t = decltype(sizeof(int));
constexpr const char operator"" _id(const char *p, size_t n) { return *p; }
constexpr const wchar_t operator"" _id(const wchar_t *p, size_t n) { return *p; }
constexpr const char16_t operator"" _id(const char16_t *p, size_t n) { return *p; }
constexpr const char32_t operator"" _id(const char32_t *p, size_t n) { return *p; }

template<int n> struct S {};
S<"a"_id[0]> sa;
S<L"b"_id[0]> sb;
S<u8"c"_id[0]> sc;
S<u"d"_id[0]> sd;
S<U"e"_id[0]> se;

S<'w'_id> sw;
S<L'x'_id> sx;
S<u'y'_id> sy;
S<U'z'_id> sz;

void h() {
  (void)"test"_id "test" L"test";
}
