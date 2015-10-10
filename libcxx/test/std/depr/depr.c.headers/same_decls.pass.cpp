//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// [depr.c.headers]p2:
//   Every C header, each of which has a name of the form name.h, behaves as if
//   each name placed in the standard library namespace by the corresponding
//   cname header is placed within the global namespace scope.
//
// This implies that the name in the global namespace and the name in namespace
// std declare the same entity, so check that that is actually the case.

// Pull in libc++'s static_assert emulation if in C++98 mode.
#include <ciso646>

template<typename T, T *F, T *G> struct check { static_assert(F == G, ""); };

// ctype.h

#include <ctype.h>
#include <cctype>

void test_ctype() {
  check<int(int), isalnum, std::isalnum>();
  check<int(int), isalpha, std::isalpha>();
  check<int(int), isblank, std::isblank>();
  check<int(int), iscntrl, std::iscntrl>();
  check<int(int), isdigit, std::isdigit>();
  check<int(int), isgraph, std::isgraph>();
  check<int(int), islower, std::islower>();
  check<int(int), isprint, std::isprint>();
  check<int(int), ispunct, std::ispunct>();
  check<int(int), isspace, std::isspace>();
  check<int(int), isupper, std::isupper>();
  check<int(int), isxdigit, std::isxdigit>();
  check<int(int), tolower, std::tolower>();
  check<int(int), toupper, std::toupper>();
}

// fenv.h

#include <fenv.h>
#include <cfenv>

void test_fenv() {
  check<int(int), feclearexcept, std::feclearexcept>();
  check<int(fexcept_t *, int), fegetexceptflag, std::fegetexceptflag>();
  check<int(int), feraiseexcept, std::feraiseexcept>();
  check<int(const fexcept_t *, int), fesetexceptflag, std::fesetexceptflag>();
  check<int(int), fetestexcept, std::fetestexcept>();
  check<int(), fegetround, std::fegetround>();
  check<int(int), fesetround, std::fesetround>();
  check<int(fenv_t *), fegetenv, std::fegetenv>();
  check<int(fenv_t *), feholdexcept, std::feholdexcept>();
  check<int(const fenv_t *), fesetenv, std::fesetenv>();
  check<int(const fenv_t *), feupdateenv, std::feupdateenv>();
}

// inttypes.h

#include <inttypes.h>
#include <cinttypes>

// Avoid error if abs and div are not declared by <inttypes.h>.
struct nat {};
static void abs(nat);
static void div(nat);
namespace std {
  static void abs(nat);
  static void div(nat);
}

// These may or may not exist, depending on whether intmax_t
// is an extended integer type (larger than long long).
template<typename I = intmax_t, bool = (sizeof(I) > sizeof(long long))>
void test_inttypes_abs_div() {
  check<I(I), abs, std::abs>();
  check<imaxdiv_t(I, I), div, std::div>();
}
template<> void test_inttypes_abs_div<intmax_t, false>() {}

void test_inttypes() {
  test_inttypes_abs_div();

  check<intmax_t(intmax_t), imaxabs, std::imaxabs>();
  check<imaxdiv_t(intmax_t, intmax_t), imaxdiv, std::imaxdiv>();
  check<intmax_t(const char *, char **, int), strtoimax, std::strtoimax>();
  check<uintmax_t(const char *, char **, int), strtoumax, std::strtoumax>();
  check<intmax_t(const wchar_t *, wchar_t **, int), wcstoimax, std::wcstoimax>();
  check<uintmax_t(const wchar_t *, wchar_t **, int), wcstoumax, std::wcstoumax>();
}

// locale.h

#include <locale.h>
#include <clocale>

void test_locale() {
  check<char *(int, const char *), setlocale, std::setlocale>();
  check<lconv *(), localeconv, std::localeconv>();
}

// math.h

#include <math.h>
#include <cmath>

template<typename Float>
void test_math_float() {
  typedef Float ff(Float);
  typedef Float ffi(Float, int);
  typedef Float ffpi(Float, int*);
  typedef Float ffl(Float, long);
  typedef Float fff(Float, Float);
  typedef Float ffpf(Float, Float*);
  typedef Float ffld(Float, long double);
  typedef Float fffpi(Float, Float, int*);
  typedef Float ffff(Float, Float, Float);
  typedef bool bf(Float);
  typedef bool bff(Float, Float);
  typedef int i_f(Float);
  typedef long lf(Float);
  typedef long long llf(Float);

  check<ff, abs, std::abs>();
  check<ff, acos, std::acos>();
  check<ff, acosh, std::acosh>();
  check<ff, asin, std::asin>();
  check<ff, asinh, std::asinh>();
  check<ff, atan, std::atan>();
  check<fff, atan2, std::atan2>();
  check<ff, atanh, std::atanh>();
  check<ff, cbrt, std::cbrt>();
  check<ff, ceil, std::ceil>();
  check<fff, copysign, std::copysign>();
  check<ff, cos, std::cos>();
  check<ff, cosh, std::cosh>();
  check<ff, erf, std::erf>();
  check<ff, erfc, std::erfc>();
  check<ff, exp, std::exp>();
  check<ff, exp2, std::exp2>();
  check<ff, expm1, std::expm1>();
  check<ff, fabs, std::fabs>();
  check<fff, fdim, std::fdim>();
  check<ff, floor, std::floor>();
  check<ffff, fma, std::fma>();
  check<fff, fmax, std::fmax>();
  check<fff, fmin, std::fmin>();
  check<fff, fmod, std::fmod>();
  check<ffpi, frexp, std::frexp>();
  check<fff, hypot, std::hypot>();
  check<i_f, ilogb, std::ilogb>();
  check<ffi, ldexp, std::ldexp>();
  check<ff, lgamma, std::lgamma>();
  check<llf, llrint, std::llrint>();
  check<llf, llround, std::llround>();
  check<ff, log, std::log>();
  check<ff, log10, std::log10>();
  check<ff, log1p, std::log1p>();
  check<ff, log2, std::log2>();
  check<ff, logb, std::logb>();
  check<lf, lrint, std::lrint>();
  check<lf, lround, std::lround>();
  check<ffpf, modf, std::modf>();
  check<ff, nearbyint, std::nearbyint>();
  check<fff, nextafter, std::nextafter>();
  check<ffld, nexttoward, std::nexttoward>();
  check<fff, pow, std::pow>();
  check<fff, remainder, std::remainder>();
  check<fffpi, remquo, std::remquo>();
  check<ff, rint, std::rint>();
  check<ff, round, std::round>();
  check<ffl, scalbln, std::scalbln>();
  check<ffi, scalbn, std::scalbn>();
  check<ff, sin, std::sin>();
  check<ff, sinh, std::sinh>();
  check<ff, sqrt, std::sqrt>();
  check<ff, tan, std::tan>();
  check<ff, tanh, std::tanh>();
  check<ff, tgamma, std::tgamma>();
  check<ff, trunc, std::trunc>();

  check<i_f, fpclassify, std::fpclassify>();
  check<bf, isfinite, std::isfinite>();
  check<bff, isgreater, std::isgreater>();
  check<bff, isgreaterequal, std::isgreaterequal>();
  check<bf, isinf, std::isinf>();
  check<bff, isless, std::isless>();
  check<bff, islessequal, std::islessequal>();
  check<bff, islessgreater, std::islessgreater>();
  check<bf, isnan, std::isnan>();
  check<bf, isnormal, std::isnormal>();
  check<bff, isunordered, std::isunordered>();
  check<bf, signbit, std::signbit>();
}

void test_math() {
  test_math_float<float>();
  test_math_float<double>();
  test_math_float<long double>();

  check<float(const char *), nanf, std::nanf>();
  check<double(const char *), nan, std::nan>();
  check<long double(const char *), nanl, std::nanl>();
}

// setjmp.h

#include <setjmp.h>
#include <csetjmp>

void test_setjmp() {
  check<void(jmp_buf, int), longjmp, std::longjmp>();
}

// signal.h

#include <signal.h>
#include <csignal>

void test_signal() {
  check<int(int), raise, std::raise>();
  typedef void Handler(int);
  check<Handler *(int, Handler *), signal, std::signal>();
}

// stdio.h

#include <stdio.h>
#include <cstdio>

void test_stdio() {
  check<void(FILE *), clearerr, std::clearerr>();
  check<int(FILE *), fclose, std::fclose>();
  check<int(FILE *), feof, std::feof>();
  check<int(FILE *), ferror, std::ferror>();
  check<int(FILE *), fflush, std::fflush>();
  check<int(FILE *), fgetc, std::fgetc>();
  check<int(FILE *, fpos_t *), fgetpos, std::fgetpos>();
  check<char *(char *, int, FILE *), fgets, std::fgets>();
  check<FILE *(const char *, const char *), fopen, std::fopen>();
  check<int(FILE *, const char *, ...), fprintf, std::fprintf>();
  check<int(int, FILE *), fputc, std::fputc>();
  check<int(const char *, FILE *), fputs, std::fputs>();
  check<size_t(void *, size_t, size_t, FILE *), fread, std::fread>();
  check<FILE *(const char *, const char *, FILE *), freopen, std::freopen>();
  check<int(FILE *, const char *, ...), fscanf, std::fscanf>();
  check<int(FILE *, long, int), fseek, std::fseek>();
  check<int(FILE *, const fpos_t *), fsetpos, std::fsetpos>();
  check<long(FILE *), ftell, std::ftell>();
  check<size_t(const void *, size_t, size_t, FILE *), fwrite, std::fwrite>();
  check<int(FILE *), getc, std::getc>();
  check<int(), getchar, std::getchar>();
  check<void(const char *), perror, std::perror>();
  check<int(const char *, ...), printf, std::printf>();
  check<int(int, FILE *), putc, std::putc>();
  check<int(int), putchar, std::putchar>();
  check<int(const char *), puts, std::puts>();
  check<int(const char *), remove, std::remove>();
  check<int(const char *, const char *), rename, std::rename>();
  check<void(FILE *), rewind, std::rewind>();
  check<int(const char *, ...), scanf, std::scanf>();
  check<void(FILE *, char *), setbuf, std::setbuf>();
  check<int(FILE *, char *, int, size_t), setvbuf, std::setvbuf>();
  check<int(char *, size_t, const char *, ...), snprintf, std::snprintf>();
  check<int(char *, const char *, ...), sprintf, std::sprintf>();
  check<int(const char *, const char *, ...), sscanf, std::sscanf>();
  check<FILE *(), tmpfile, std::tmpfile>();
  check<char *(char *), tmpnam, std::tmpnam>();
  check<int(int, FILE *), ungetc, std::ungetc>();
  check<int(FILE *, const char *, va_list), vfprintf, std::vfprintf>();
  check<int(const char *, va_list), vprintf, std::vprintf>();
  check<int(const char *, va_list), vscanf, std::vscanf>();
  check<int(char *, size_t, const char *, va_list), vsnprintf, std::vsnprintf>();
  check<int(char *, const char *, va_list), vsprintf, std::vsprintf>();
  check<int(const char *, const char *, va_list), vsscanf, std::vsscanf>();
}

// stdlib.h

#include <stdlib.h>
#include <cstdlib>

void test_stdlib() {
  typedef int Comp(const void*, const void*);

  check<void(int), _Exit, std::_Exit>();
  check<int(void()), at_quick_exit, std::at_quick_exit>();
  check<void(), abort, std::abort>();
  check<void(int), exit, std::exit>();
  check<int(void()), atexit, std::atexit>();
  check<void(int), quick_exit, std::quick_exit>();
  check<char *(const char *), getenv, std::getenv>();
  check<int(const char *), system, std::system>();
  check<void *(size_t, size_t), calloc, std::calloc>();
  check<void(void *), free, std::free>();
  check<void *(size_t), malloc, std::malloc>();
  check<void *(void *, size_t), realloc, std::realloc>();
  check<double(const char *), atof, std::atof>();
  check<int(const char *), atoi, std::atoi>();
  check<long(const char *), atol, std::atol>();
  check<long long(const char *), atoll, std::atoll>();
  check<int(const char *, size_t), mblen, std::mblen>();
  check<int(wchar_t *, const char *, size_t), mbtowc, std::mbtowc>();
  check<size_t(wchar_t *, const char *, size_t), mbstowcs, std::mbstowcs>();
  check<double(const char *, char **), strtod, std::strtod>();
  check<float(const char *, char **), strtof, std::strtof>();
  check<long(const char *, char **, int), strtol, std::strtol>();
  check<long double(const char *, char **), strtold, std::strtold>();
  check<long long(const char *, char **, int), strtoll, std::strtoll>();
  check<unsigned long(const char *, char **, int), strtoul, std::strtoul>();
  check<unsigned long long(const char *, char **, int), strtoull, std::strtoull>();
  check<int(char *, wchar_t), wctomb, std::wctomb>();
  check<size_t(char *, const wchar_t *, size_t), wcstombs, std::wcstombs>();
  check<void *(const void *, const void *, size_t, size_t, Comp *), bsearch, std::bsearch>();
  check<void(void *, size_t, size_t, Comp *), qsort, std::qsort>();
  check<int(int), abs, std::abs>();
  check<div_t(int, int), div, std::div>();
  check<long(long), abs, std::abs>();
  check<long(long), labs, std::labs>();
  check<ldiv_t(long, long), div, std::div>();
  check<ldiv_t(long, long), ldiv, std::ldiv>();
  check<long long(long long), abs, std::abs>();
  check<long long(long long), llabs, std::llabs>();
  check<lldiv_t(long long, long long), div, std::div>();
  check<lldiv_t(long long, long long), lldiv, std::lldiv>();
  check<int(), rand, std::rand>();
  check<void(unsigned), srand, std::srand>();
}

// string.h

#include <string.h>
#include <cstring>

void test_string() {
  check<void *(void *, int, size_t), memchr, std::memchr>();
  check<const void *(const void *, int, size_t), memchr, std::memchr>();
  check<int(const void *, const void *, size_t), memcmp, std::memcmp>();
  check<void *(void *, const void *, size_t), memcpy, std::memcpy>();
  check<void *(void *, const void *, size_t), memmove, std::memmove>();
  check<void *(void *, int, size_t), memset, std::memset>();
  check<char *(char *, const char *), strcat, std::strcat>();
  check<char *(char *, int), strchr, std::strchr>();
  check<const char *(const char *, int), strchr, std::strchr>();
  check<int(const char *, const char *), strcmp, std::strcmp>();
  check<int(const char *, const char *), strcoll, std::strcoll>();
  check<char *(char *, const char *), strcpy, std::strcpy>();
  check<size_t(const char *, const char *), strcspn, std::strcspn>();
  check<char *(int), strerror, std::strerror>();
  check<size_t(const char *), strlen, std::strlen>();
  check<char *(char *, const char *, size_t), strncat, std::strncat>();
  check<int(const char *, const char *, size_t), strncmp, std::strncmp>();
  check<char *(char *, const char *, size_t), strncpy, std::strncpy>();
  check<char *(char *, const char *), strpbrk, std::strpbrk>();
  check<const char *(const char *, const char *), strpbrk, std::strpbrk>();
  check<char *(char *, int), strrchr, std::strrchr>();
  check<const char *(const char *, int), strrchr, std::strrchr>();
  check<size_t(const char *, const char *), strspn, std::strspn>();
  check<char *(char *, const char *), strstr, std::strstr>();
  check<const char *(const char *, const char *), strstr, std::strstr>();
  check<char *(char *, const char *), strtok, std::strtok>();
  check<size_t(char *, const char *, size_t), strxfrm, std::strxfrm>();
}

// time.h

#include <time.h>
#include <ctime>

void test_time() {
  check<char *(const tm *), asctime, std::asctime>();
  check<char *(const time_t *), ctime, std::ctime>();
  check<clock_t(), clock, std::clock>();
  check<tm *(const time_t *), gmtime, std::gmtime>();
  check<double(time_t, time_t), difftime, std::difftime>();
  check<time_t(tm *), mktime, std::mktime>();
  check<tm *(const time_t *), localtime, std::localtime>();
  check<time_t(time_t *), time, std::time>();
  check<size_t(char *, size_t, const char *, const tm *), strftime, std::strftime>();
}

#if 0
// FIXME: <cuchar> and <uchar.h> are missing.

// uchar.h

#include <uchar.h>
#include <cuchar>

void test_uchar() {
  check<size_t(char16_t *, const char *, size_t, mbstate_t *), mbrtoc16, std::mbrtoc16>();
  check<size_t(char32_t *, const char *, size_t, mbstate_t *), mbrtoc32, std::mbrtoc32>();
  check<size_t(char *, char16_t, mbstate_t *), c16rtomb, std::c16rtomb>();
  check<size_t(char *, char32_t, mbstate_t *), c32rtomb, std::c32rtomb>();
}
#endif

// wchar.h

#include <wchar.h>
#include <cwchar>

void test_wchar() {
  check<wint_t(int), btowc, std::btowc>();
  check<wint_t(FILE *), fgetwc, std::fgetwc>();
  check<wchar_t *(wchar_t *, int, FILE *), fgetws, std::fgetws>();
  check<wint_t(wchar_t, FILE *), fputwc, std::fputwc>();
  check<int(const wchar_t *, FILE *), fputws, std::fputws>();
  check<int(FILE *, int), fwide, std::fwide>();
  check<int(FILE *, const wchar_t *, ...), fwprintf, std::fwprintf>();
  check<int(FILE *, const wchar_t *, ...), fwscanf, std::fwscanf>();
  check<wint_t(), getwchar, std::getwchar>();
  check<wint_t(FILE *), getwc, std::getwc>();
  check<size_t(const char *, size_t, mbstate_t *), mbrlen, std::mbrlen>();
  check<size_t(wchar_t *, const char *, size_t, mbstate_t *), mbrtowc, std::mbrtowc>();
  check<int(const mbstate_t *), mbsinit, std::mbsinit>();
  check<size_t(wchar_t *, const char **, size_t, mbstate_t *), mbsrtowcs, std::mbsrtowcs>();
  check<wint_t(wchar_t), putwchar, std::putwchar>();
  check<wint_t(wchar_t, FILE *), putwc, std::putwc>();
  check<int(wchar_t *, size_t, const wchar_t *, ...), swprintf, std::swprintf>();
  check<int(const wchar_t *, const wchar_t *, ...), swscanf, std::swscanf>();
  check<wint_t(wint_t, FILE *), ungetwc, std::ungetwc>();
  check<int(FILE *, const wchar_t *, va_list), vfwprintf, std::vfwprintf>();
  check<int(FILE *, const wchar_t *, va_list), vfwscanf, std::vfwscanf>();
  check<int(wchar_t *, size_t, const wchar_t *, va_list), vswprintf, std::vswprintf>();
  check<int(const wchar_t *, const wchar_t *, va_list), vswscanf, std::vswscanf>();
  check<int(const wchar_t *, va_list), vwprintf, std::vwprintf>();
  check<int(const wchar_t *, va_list), vwscanf, std::vwscanf>();
  check<size_t(char *, wchar_t, mbstate_t *), wcrtomb, std::wcrtomb>();
  check<wchar_t *(wchar_t *, const wchar_t *), wcscat, std::wcscat>();
  check<wchar_t *(wchar_t *, wchar_t), wcschr, std::wcschr>();
  check<const wchar_t *(const wchar_t *, wchar_t), wcschr, std::wcschr>();
  check<int(const wchar_t *, const wchar_t *), wcscmp, std::wcscmp>();
  check<int(const wchar_t *, const wchar_t *), wcscoll, std::wcscoll>();
  check<wchar_t *(wchar_t *, const wchar_t *), wcscpy, std::wcscpy>();
  check<size_t(const wchar_t *, const wchar_t *), wcscspn, std::wcscspn>();
  check<size_t(wchar_t *, size_t, const wchar_t *, const tm *), wcsftime, std::wcsftime>();
  check<size_t(const wchar_t *), wcslen, std::wcslen>();
  check<wchar_t *(wchar_t *, const wchar_t *, size_t), wcsncat, std::wcsncat>();
  check<int(const wchar_t *, const wchar_t *, size_t), wcsncmp, std::wcsncmp>();
  check<wchar_t *(wchar_t *, const wchar_t *, size_t), wcsncpy, std::wcsncpy>();
  check<wchar_t *(wchar_t *, const wchar_t *), wcspbrk, std::wcspbrk>();
  check<const wchar_t *(const wchar_t *, const wchar_t *), wcspbrk, std::wcspbrk>();
  check<wchar_t *(wchar_t *, wchar_t), wcsrchr, std::wcsrchr>();
  check<const wchar_t *(const wchar_t *, wchar_t), wcsrchr, std::wcsrchr>();
  check<size_t(char *, const wchar_t **, size_t, mbstate_t *), wcsrtombs, std::wcsrtombs>();
  check<size_t(const wchar_t *, const wchar_t *), wcsspn, std::wcsspn>();
  check<wchar_t *(wchar_t *, const wchar_t *), wcsstr, std::wcsstr>();
  check<const wchar_t *(const wchar_t *, const wchar_t *), wcsstr, std::wcsstr>();
  check<double(const wchar_t *, wchar_t **), wcstod, std::wcstod>();
  check<float(const wchar_t *, wchar_t **), wcstof, std::wcstof>();
  check<wchar_t *(wchar_t *, const wchar_t *, wchar_t **), wcstok, std::wcstok>();
  check<long double(const wchar_t *, wchar_t **), wcstold, std::wcstold>();
  check<long long(const wchar_t *, wchar_t **, int), wcstoll, std::wcstoll>();
  check<long(const wchar_t *, wchar_t **, int), wcstol, std::wcstol>();
  check<unsigned long long(const wchar_t *, wchar_t **, int), wcstoull, std::wcstoull>();
  check<unsigned long(const wchar_t *, wchar_t **, int), wcstoul, std::wcstoul>();
  check<size_t(wchar_t *, const wchar_t *, size_t), wcsxfrm, std::wcsxfrm>();
  check<int(wint_t), wctob, std::wctob>();
  check<wchar_t *(wchar_t *, wchar_t, size_t), wmemchr, std::wmemchr>();
  check<const wchar_t *(const wchar_t *, wchar_t, size_t), wmemchr, std::wmemchr>();
  check<int(const wchar_t *, const wchar_t *, size_t), wmemcmp, std::wmemcmp>();
  check<wchar_t *(wchar_t *, const wchar_t *, size_t), wmemcpy, std::wmemcpy>();
  check<wchar_t *(wchar_t *, const wchar_t *, size_t), wmemmove, std::wmemmove>();
  check<wchar_t *(wchar_t *, wchar_t, size_t), wmemset, std::wmemset>();
  check<int(const wchar_t *, ...), wprintf, std::wprintf>();
  check<int(const wchar_t *, ...), wscanf, std::wscanf>();
}

// wctype.h

#include <wctype.h>
#include <cwctype>

void test_wctype() {
  check<int(wint_t), iswalnum, std::iswalnum>();
  check<int(wint_t), iswalpha, std::iswalpha>();
  check<int(wint_t), iswblank, std::iswblank>();
  check<int(wint_t), iswcntrl, std::iswcntrl>();
  check<int(wint_t, wctype_t), iswctype, std::iswctype>();
  check<int(wint_t), iswdigit, std::iswdigit>();
  check<int(wint_t), iswgraph, std::iswgraph>();
  check<int(wint_t), iswlower, std::iswlower>();
  check<int(wint_t), iswprint, std::iswprint>();
  check<int(wint_t), iswpunct, std::iswpunct>();
  check<int(wint_t), iswspace, std::iswspace>();
  check<int(wint_t), iswupper, std::iswupper>();
  check<int(wint_t), iswxdigit, std::iswxdigit>();
  check<wint_t(wint_t, wctrans_t), towctrans, std::towctrans>();
  check<wint_t(wint_t), towlower, std::towlower>();
  check<wint_t(wint_t), towupper, std::towupper>();
  check<wctrans_t(const char *), wctrans, std::wctrans>();
  check<wctype_t(const char *), wctype, std::wctype>();
}

int main() {
  test_ctype();
  test_fenv();
  test_inttypes();
  test_locale();
  test_math();
  test_setjmp();
  test_signal();
  test_stdio();
  test_stdlib();
  test_string();
  test_time();
  //test_uchar();
  test_wchar();
  test_wctype();
}
