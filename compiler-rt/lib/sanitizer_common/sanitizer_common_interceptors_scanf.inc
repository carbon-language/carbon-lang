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
//
//===----------------------------------------------------------------------===//
#include <stdarg.h>

struct ScanfSpec {
  char c;
  unsigned size;
};

// One-letter specs.
static const ScanfSpec scanf_specs[] = {
  {'p', sizeof(void *)},
  {'e', sizeof(float)},
  {'E', sizeof(float)},
  {'a', sizeof(float)},
  {'f', sizeof(float)},
  {'g', sizeof(float)},
  {'d', sizeof(int)},
  {'i', sizeof(int)},
  {'o', sizeof(int)},
  {'u', sizeof(int)},
  {'x', sizeof(int)},
  {'X', sizeof(int)},
  {'n', sizeof(int)},
  {'t', sizeof(PTRDIFF_T)},
  {'z', sizeof(SIZE_T)},
  {'j', sizeof(INTMAX_T)},
  {'h', sizeof(short)}
};

static const unsigned scanf_specs_cnt =
  sizeof(scanf_specs) / sizeof(scanf_specs[0]);

// %ll?, %L?, %q? specs
static const ScanfSpec scanf_llspecs[] = {
  {'e', sizeof(long double)},
  {'f', sizeof(long double)},
  {'g', sizeof(long double)},
  {'d', sizeof(long long)},
  {'i', sizeof(long long)},
  {'o', sizeof(long long)},
  {'u', sizeof(long long)},
  {'x', sizeof(long long)}
};

static const unsigned scanf_llspecs_cnt =
  sizeof(scanf_llspecs) / sizeof(scanf_llspecs[0]);

// %l? specs
static const ScanfSpec scanf_lspecs[] = {
  {'e', sizeof(double)},
  {'f', sizeof(double)},
  {'g', sizeof(double)},
  {'d', sizeof(long)},
  {'i', sizeof(long)},
  {'o', sizeof(long)},
  {'u', sizeof(long)},
  {'x', sizeof(long)},
  {'X', sizeof(long)},
};

static const unsigned scanf_lspecs_cnt =
  sizeof(scanf_lspecs) / sizeof(scanf_lspecs[0]);

static unsigned match_spec(const struct ScanfSpec *spec, unsigned n, char c) {
  for (unsigned i = 0; i < n; ++i)
    if (spec[i].c == c)
      return spec[i].size;
  return 0;
}

static void scanf_common(void *ctx, const char *format, va_list ap_const) {
  va_list aq;
  va_copy(aq, ap_const);

  const char *p = format;
  unsigned size;

  while (*p) {
    if (*p != '%') {
      ++p;
      continue;
    }
    ++p;
    if (*p == '*' || *p == '%' || *p == 0) {
      ++p;
      continue;
    }
    if (*p == '0' || (*p >= '1' && *p <= '9')) {
      size = internal_atoll(p);
      // +1 for the \0 at the end
      COMMON_INTERCEPTOR_WRITE_RANGE(ctx, va_arg(aq, void *), size + 1);
      ++p;
      continue;
    }

    if (*p == 'L' || *p == 'q') {
      ++p;
      size = match_spec(scanf_llspecs, scanf_llspecs_cnt, *p);
      COMMON_INTERCEPTOR_WRITE_RANGE(ctx, va_arg(aq, void *), size);
      continue;
    }

    if (*p == 'l') {
      ++p;
      if (*p == 'l') {
        ++p;
        size = match_spec(scanf_llspecs, scanf_llspecs_cnt, *p);
        COMMON_INTERCEPTOR_WRITE_RANGE(ctx, va_arg(aq, void *), size);
        continue;
      } else {
        size = match_spec(scanf_lspecs, scanf_lspecs_cnt, *p);
        COMMON_INTERCEPTOR_WRITE_RANGE(ctx, va_arg(aq, void *), size);
        continue;
      }
    }

    if (*p == 'h' && *(p + 1) == 'h') {
      COMMON_INTERCEPTOR_WRITE_RANGE(ctx, va_arg(aq, void *), sizeof(char));
      p += 2;
      continue;
    }

    size = match_spec(scanf_specs, scanf_specs_cnt, *p);
    if (size) {
      COMMON_INTERCEPTOR_WRITE_RANGE(ctx, va_arg(aq, void *), size);
      ++p;
      continue;
    }
  }
  va_end(aq);
}
