// RUN: %clang_cc1 -std=c++11 -verify %s -fms-extensions -triple x86_64-apple-darwin9.0.0

using size_t = decltype(sizeof(int));
enum class LitKind {
  Char, WideChar, Char16, Char32,
  CharStr, WideStr, Char16Str, Char32Str,
  Integer, Floating
};
constexpr LitKind operator"" _kind(char p) { return LitKind::Char; }
constexpr LitKind operator"" _kind(wchar_t p) { return LitKind::WideChar; }
constexpr LitKind operator"" _kind(char16_t p) { return LitKind::Char16; }
constexpr LitKind operator"" _kind(char32_t p) { return LitKind::Char32; }
constexpr LitKind operator"" _kind(const char *p, size_t n) { return LitKind::CharStr; }
constexpr LitKind operator"" _kind(const wchar_t *p, size_t n) { return LitKind::WideStr; }
constexpr LitKind operator"" _kind(const char16_t *p, size_t n) { return LitKind::Char16Str; }
constexpr LitKind operator"" _kind(const char32_t *p, size_t n) { return LitKind::Char32Str; }
constexpr LitKind operator"" _kind(unsigned long long n) { return LitKind::Integer; }
constexpr LitKind operator"" _kind(long double n) { return LitKind::Floating; }

static_assert('x'_kind == LitKind::Char, "");
static_assert(L'x'_kind == LitKind::WideChar, "");
static_assert(u'x'_kind == LitKind::Char16, "");
static_assert(U'x'_kind == LitKind::Char32, "");
static_assert("foo"_kind == LitKind::CharStr, "");
static_assert(u8"foo"_kind == LitKind::CharStr, "");
static_assert(L"foo"_kind == LitKind::WideStr, "");
static_assert(u"foo"_kind == LitKind::Char16Str, "");
static_assert(U"foo"_kind == LitKind::Char32Str, "");
static_assert(194_kind == LitKind::Integer, "");
static_assert(0377_kind == LitKind::Integer, "");
static_assert(0x5ffc_kind == LitKind::Integer, "");
static_assert(.5954_kind == LitKind::Floating, "");
static_assert(1._kind == LitKind::Floating, "");
static_assert(1.e-2_kind == LitKind::Floating, "");
static_assert(4e6_kind == LitKind::Floating, "");
