//===-- sanitizer_scanf_interceptor_test.cc -------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Tests for *scanf interceptors implementation in sanitizer_common.
//
//===----------------------------------------------------------------------===//
#include <vector>

#include "interception/interception.h"
#include "sanitizer_test_utils.h"
#include "sanitizer_common/sanitizer_libc.h"
#include "gtest/gtest.h"

using namespace __sanitizer;

#define COMMON_INTERCEPTOR_WRITE_RANGE(ctx, ptr, size)                         \
  ((std::vector<unsigned> *)ctx)->push_back(size)

#include "sanitizer_common/sanitizer_common_interceptors_scanf.inc"

static const char scanf_buf[] = "Test string.";
static size_t scanf_buf_size = sizeof(scanf_buf);
static const unsigned SCANF_ARGS_MAX = 16;

static void testScanf3(void *ctx, int result, bool allowGnuMalloc,
                       const char *format, ...) {
  va_list ap;
  va_start(ap, format);
  scanf_common(ctx, result, allowGnuMalloc, format, ap);
  va_end(ap);
}

static void testScanf2(const char *format, int scanf_result,
                       bool allowGnuMalloc, unsigned n,
                       va_list expected_sizes) {
  std::vector<unsigned> scanf_sizes;
  // 16 args should be enough.
  testScanf3((void *)&scanf_sizes, scanf_result, allowGnuMalloc, format,
             scanf_buf, scanf_buf, scanf_buf, scanf_buf, scanf_buf, scanf_buf,
             scanf_buf, scanf_buf, scanf_buf, scanf_buf, scanf_buf, scanf_buf,
             scanf_buf, scanf_buf, scanf_buf, scanf_buf);
  ASSERT_EQ(n, scanf_sizes.size()) << "Unexpected number of format arguments: '"
                                   << format << "'";
  for (unsigned i = 0; i < n; ++i)
    EXPECT_EQ(va_arg(expected_sizes, unsigned), scanf_sizes[i])
        << "Unexpect write size for argument " << i << ", format string '"
        << format << "'";
}

static void testScanf(const char *format, unsigned n, ...) {
  va_list ap;
  va_start(ap, n);
  testScanf2(format, SCANF_ARGS_MAX, /* allowGnuMalloc */ true, n, ap);
  va_end(ap);
}

static void testScanfPartial(const char *format, int scanf_result, unsigned n,
                             ...) {
  va_list ap;
  va_start(ap, n);
  testScanf2(format, scanf_result, /* allowGnuMalloc */ true,  n, ap);
  va_end(ap);
}

static void testScanfNoGnuMalloc(const char *format, unsigned n, ...) {
  va_list ap;
  va_start(ap, n);
  testScanf2(format, SCANF_ARGS_MAX, /* allowGnuMalloc */ false, n, ap);
  va_end(ap);
}

TEST(SanitizerCommonInterceptors, Scanf) {
  const unsigned I = sizeof(int);          // NOLINT
  const unsigned L = sizeof(long);         // NOLINT
  const unsigned LL = sizeof(long long);   // NOLINT
  const unsigned S = sizeof(short);        // NOLINT
  const unsigned C = sizeof(char);         // NOLINT
  const unsigned D = sizeof(double);       // NOLINT
  const unsigned LD = sizeof(long double); // NOLINT
  const unsigned F = sizeof(float);        // NOLINT
  const unsigned P = sizeof(char *);       // NOLINT

  testScanf("%d", 1, I);
  testScanf("%d%d%d", 3, I, I, I);
  testScanf("ab%u%dc", 2, I, I);
  testScanf("%ld", 1, L);
  testScanf("%llu", 1, LL);
  testScanf("a %hd%hhx", 2, S, C);
  testScanf("%c", 1, C);

  testScanf("%%", 0);
  testScanf("a%%", 0);
  testScanf("a%%b", 0);
  testScanf("a%%%%b", 0);
  testScanf("a%%b%%", 0);
  testScanf("a%%%%%%b", 0);
  testScanf("a%%%%%b", 0);
  testScanf("a%%%%%f", 1, F);
  testScanf("a%%%lxb", 1, L);
  testScanf("a%lf%%%lxb", 2, D, L);
  testScanf("%nf", 1, I);

  testScanf("%10s", 1, 11);
  testScanf("%10c", 1, 10);
  testScanf("%%10s", 0);
  testScanf("%*10s", 0);
  testScanf("%*d", 0);

  testScanf("%4d%8f%c", 3, I, F, C);
  testScanf("%s%d", 2, scanf_buf_size, I);
  testScanf("%[abc]", 1, scanf_buf_size);
  testScanf("%4[bcdef]", 1, 5);
  testScanf("%[]]", 1, scanf_buf_size);
  testScanf("%8[^]%d0-9-]%c", 2, 9, C);

  testScanf("%*[^:]%n:%d:%1[ ]%n", 4, I, I, 2, I);

  testScanf("%*d%u", 1, I);

  testScanf("%c%d", 2, C, I);
  testScanf("%A%lf", 2, F, D);

  testScanf("%ms %Lf", 2, P, LD);
  testScanf("s%Las", 1, LD);
  testScanf("%ar", 1, F);

  // In the cases with std::min below the format spec can be interpreted as
  // either floating-something, or (GNU extension) callee-allocated string.
  // Our conservative implementation reports one of the two possibilities with
  // the least store range.
  testScanf("%a[", 0);
  testScanf("%a[]", 0);
  testScanf("%a[]]", 1, std::min(F, P));
  testScanf("%a[abc]", 1, std::min(F, P));
  testScanf("%a[^abc]", 1, std::min(F, P));
  testScanf("%a[ab%c] %d", 0);
  testScanf("%a[^ab%c] %d", 0);
  testScanf("%as", 1, std::min(F, P));
  testScanf("%aS", 1, std::min(F, P));
  testScanf("%a13S", 1, std::min(F, P));
  testScanf("%alS", 1, std::min(F, P));

  testScanfNoGnuMalloc("s%Las", 1, LD);
  testScanfNoGnuMalloc("%ar", 1, F);
  testScanfNoGnuMalloc("%a[", 1, F);
  testScanfNoGnuMalloc("%a[]", 1, F);
  testScanfNoGnuMalloc("%a[]]", 1, F);
  testScanfNoGnuMalloc("%a[abc]", 1, F);
  testScanfNoGnuMalloc("%a[^abc]", 1, F);
  testScanfNoGnuMalloc("%a[ab%c] %d", 3, F, C, I);
  testScanfNoGnuMalloc("%a[^ab%c] %d", 3, F, C, I);
  testScanfNoGnuMalloc("%as", 1, F);
  testScanfNoGnuMalloc("%aS", 1, F);
  testScanfNoGnuMalloc("%a13S", 1, F);
  testScanfNoGnuMalloc("%alS", 1, F);

  testScanf("%5$d", 0);
  testScanf("%md", 0);
  testScanf("%m10s", 0);

  testScanfPartial("%d%d%d%d //1\n", 1, 1, I);
  testScanfPartial("%d%d%d%d //2\n", 2, 2, I, I);
  testScanfPartial("%d%d%d%d //3\n", 3, 3, I, I, I);
  testScanfPartial("%d%d%d%d //4\n", 4, 4, I, I, I, I);

  testScanfPartial("%d%n%n%d //1\n", 1, 1, I);
  testScanfPartial("%d%n%n%d //2\n", 2, 4, I, I, I, I);

  testScanfPartial("%d%n%n%d %s %s", 3, 5, I, I, I, I, scanf_buf_size);
  testScanfPartial("%d%n%n%d %s %s", 4, 6, I, I, I, I, scanf_buf_size,
                   scanf_buf_size);
}
