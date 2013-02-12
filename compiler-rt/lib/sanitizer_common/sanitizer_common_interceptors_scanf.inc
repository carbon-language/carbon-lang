//===-- sanitizer_common_interceptors_scanf.inc -----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Scanf implementation for use in *Sanitizer interceptors.
// Follows http://pubs.opengroup.org/onlinepubs/9699919799/functions/fscanf.html
// with a few common GNU extensions.
//
//===----------------------------------------------------------------------===//
#include <stdarg.h>

struct ScanfDirective {
  int argIdx; // argument index, or -1 of not specified ("%n$")
  int fieldWidth;
  bool suppressed; // suppress assignment ("*")
  bool allocate;   // allocate space ("m")
  char lengthModifier[2];
  char convSpecifier;
  bool maybeGnuMalloc;
};

static const char *parse_number(const char *p, int *out) {
  *out = internal_atoll(p);
  while (*p >= '0' && *p <= '9')
    ++p;
  return p;
}

static bool char_is_one_of(char c, const char *s) {
  return !!internal_strchr(s, c);
}

// Parse scanf format string. If a valid directive in encountered, it is
// returned in dir. This function returns the pointer to the first
// unprocessed character, or 0 in case of error.
// In case of the end-of-string, a pointer to the closing \0 is returned.
static const char *scanf_parse_next(const char *p, bool allowGnuMalloc,
                                    ScanfDirective *dir) {
  internal_memset(dir, 0, sizeof(*dir));
  dir->argIdx = -1;

  while (*p) {
    if (*p != '%') {
      ++p;
      continue;
    }
    ++p;
    // %%
    if (*p == '%') {
      ++p;
      continue;
    }
    if (*p == '\0') {
      return 0;
    }
    // %n$
    if (*p >= '0' && *p <= '9') {
      int number;
      const char *q = parse_number(p, &number);
      if (*q == '$') {
        dir->argIdx = number;
        p = q + 1;
      }
      // Otherwise, do not change p. This will be re-parsed later as the field
      // width.
    }
    // *
    if (*p == '*') {
      dir->suppressed = true;
      ++p;
    }
    // Field width.
    if (*p >= '0' && *p <= '9') {
      p = parse_number(p, &dir->fieldWidth);
      if (dir->fieldWidth <= 0)
        return 0;
    }
    // m
    if (*p == 'm') {
      dir->allocate = true;
      ++p;
    }
    // Length modifier.
    if (char_is_one_of(*p, "jztLq")) {
      dir->lengthModifier[0] = *p;
      ++p;
    } else if (*p == 'h') {
      dir->lengthModifier[0] = 'h';
      ++p;
      if (*p == 'h') {
        dir->lengthModifier[1] = 'h';
        ++p;
      }
    } else if (*p == 'l') {
      dir->lengthModifier[0] = 'l';
      ++p;
      if (*p == 'l') {
        dir->lengthModifier[1] = 'l';
        ++p;
      }
    }
    // Conversion specifier.
    dir->convSpecifier = *p++;
    // Consume %[...] expression.
    if (dir->convSpecifier == '[') {
      if (*p == '^')
        ++p;
      if (*p == ']')
        ++p;
      while (*p && *p != ']')
        ++p;
      if (*p == 0)
        return 0; // unexpected end of string
                  // Consume the closing ']'.
      ++p;
    }
    // This is unfortunately ambiguous between old GNU extension
    // of %as, %aS and %a[...] and newer POSIX %a followed by
    // letters s, S or [.
    if (allowGnuMalloc && dir->convSpecifier == 'a' &&
        !dir->lengthModifier[0]) {
      if (*p == 's' || *p == 'S') {
        dir->maybeGnuMalloc = true;
        ++p;
      } else if (*p == '[') {
        // Watch for %a[h-j%d], if % appears in the
        // [...] range, then we need to give up, we don't know
        // if scanf will parse it as POSIX %a [h-j %d ] or
        // GNU allocation of string with range dh-j plus %.
        const char *q = p + 1;
        if (*q == '^')
          ++q;
        if (*q == ']')
          ++q;
        while (*q && *q != ']' && *q != '%')
          ++q;
        if (*q == 0 || *q == '%')
          return 0;
        p = q + 1; // Consume the closing ']'.
        dir->maybeGnuMalloc = true;
      }
    }
    break;
  }
  return p;
}

// Returns true if the character is an integer conversion specifier.
static bool scanf_is_integer_conv(char c) {
  return char_is_one_of(c, "diouxXn");
}

// Returns true if the character is an floating point conversion specifier.
static bool scanf_is_float_conv(char c) {
  return char_is_one_of(c, "aAeEfFgG");
}

// Returns string output character size for string-like conversions,
// or 0 if the conversion is invalid.
static int scanf_get_char_size(ScanfDirective *dir) {
  if (char_is_one_of(dir->convSpecifier, "CS")) {
    // wchar_t
    return 0;
  }

  if (char_is_one_of(dir->convSpecifier, "cs[")) {
    if (dir->lengthModifier[0] == 'l')
      // wchar_t
      return 0;
    else if (dir->lengthModifier[0] == 0)
      return sizeof(char);
    else
      return 0;
  }

  return 0;
}

enum ScanfStoreSize {
  // Store size not known in advance; can be calculated as strlen() of the
  // destination buffer.
  SSS_STRLEN = -1,
  // Invalid conversion specifier.
  SSS_INVALID = 0
};

// Returns the store size of a scanf directive (if >0), or a value of
// ScanfStoreSize.
static int scanf_get_store_size(ScanfDirective *dir) {
  if (dir->allocate) {
    if (!char_is_one_of(dir->convSpecifier, "cCsS["))
      return SSS_INVALID;
    return sizeof(char *);
  }

  if (dir->maybeGnuMalloc) {
    if (dir->convSpecifier != 'a' || dir->lengthModifier[0])
      return SSS_INVALID;
    // This is ambiguous, so check the smaller size of char * (if it is
    // a GNU extension of %as, %aS or %a[...]) and float (if it is
    // POSIX %a followed by s, S or [ letters).
    return sizeof(char *) < sizeof(float) ? sizeof(char *) : sizeof(float);
  }

  if (scanf_is_integer_conv(dir->convSpecifier)) {
    switch (dir->lengthModifier[0]) {
    case 'h':
      return dir->lengthModifier[1] == 'h' ? sizeof(char) : sizeof(short);
    case 'l':
      return dir->lengthModifier[1] == 'l' ? sizeof(long long) : sizeof(long);
    case 'L':
      return sizeof(long long);
    case 'j':
      return sizeof(INTMAX_T);
    case 'z':
      return sizeof(SIZE_T);
    case 't':
      return sizeof(PTRDIFF_T);
    case 0:
      return sizeof(int);
    default:
      return SSS_INVALID;
    }
  }

  if (scanf_is_float_conv(dir->convSpecifier)) {
    switch (dir->lengthModifier[0]) {
    case 'L':
    case 'q':
      return sizeof(long double);
    case 'l':
      return dir->lengthModifier[1] == 'l' ? sizeof(long double)
                                           : sizeof(double);
    case 0:
      return sizeof(float);
    default:
      return SSS_INVALID;
    }
  }

  if (char_is_one_of(dir->convSpecifier, "sS[")) {
    unsigned charSize = scanf_get_char_size(dir);
    if (charSize == 0)
      return SSS_INVALID;
    if (dir->fieldWidth == 0)
      return SSS_STRLEN;
    return (dir->fieldWidth + 1) * charSize;
  }

  if (char_is_one_of(dir->convSpecifier, "cC")) {
    unsigned charSize = scanf_get_char_size(dir);
    if (charSize == 0)
      return SSS_INVALID;
    if (dir->fieldWidth == 0)
      return charSize;
    return dir->fieldWidth * charSize;
  }

  if (dir->convSpecifier == 'p') {
    if (dir->lengthModifier[1] != 0)
      return SSS_INVALID;
    return sizeof(void *);
  }

  return SSS_INVALID;
}

// Common part of *scanf interceptors.
// Process format string and va_list, and report all store ranges.
// Stops when "consuming" n_inputs input items.
static void scanf_common(void *ctx, int n_inputs, bool allowGnuMalloc,
                         const char *format, va_list aq) {
  CHECK_GT(n_inputs, 0);
  const char *p = format;

  while (*p && n_inputs) {
    ScanfDirective dir;
    p = scanf_parse_next(p, allowGnuMalloc, &dir);
    if (!p)
      break;
    if (dir.convSpecifier == 0) {
      // This can only happen at the end of the format string.
      CHECK_EQ(*p, 0);
      break;
    }
    // Here the directive is valid. Do what it says.
    if (dir.argIdx != -1) {
      // Unsupported.
      break;
    }
    if (dir.suppressed)
      continue;
    int size = scanf_get_store_size(&dir);
    if (size == SSS_INVALID)
      break;
    void *argp = va_arg(aq, void *);
    if (dir.convSpecifier != 'n')
      --n_inputs;
    if (size == SSS_STRLEN) {
      size = internal_strlen((const char *)argp) + 1;
    }
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, argp, size);
  }
}
