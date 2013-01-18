#include <vector>

#include "interception/interception.h"
#include "sanitizer_test_utils.h"
#include "sanitizer_common/sanitizer_libc.h"
#include "gtest/gtest.h"

using namespace __sanitizer;

#define COMMON_INTERCEPTOR_WRITE_RANGE(ctx, ptr, size) \
  ((std::vector<unsigned> *)ctx)->push_back(size)

#include "sanitizer_common/sanitizer_common_interceptors_scanf.h"

static void testScanf2(void *ctx, const char *format, ...) {
  va_list ap;
  va_start(ap, format);
  scanf_common(ctx, format, ap);
  va_end(ap);
}

static void testScanf(const char *format, unsigned n, ...) {
  std::vector<unsigned> scanf_sizes;
  // 16 args should be enough.
  testScanf2((void *)&scanf_sizes, format,
      (void*)0, (void*)0, (void*)0, (void*)0,
      (void*)0, (void*)0, (void*)0, (void*)0,
      (void*)0, (void*)0, (void*)0, (void*)0,
      (void*)0, (void*)0, (void*)0, (void*)0);
  ASSERT_EQ(n, scanf_sizes.size()) <<
    "Unexpected number of format arguments: '" << format << "'";
  va_list ap;
  va_start(ap, n);
  for (unsigned i = 0; i < n; ++i)
    EXPECT_EQ(va_arg(ap, unsigned), scanf_sizes[i]) <<
      "Unexpect write size for argument " << i << ", format string '" <<
      format << "'";
  va_end(ap);
}

TEST(SanitizerCommonInterceptors, Scanf) {
  const unsigned I = sizeof(int);
  const unsigned L = sizeof(long);
  const unsigned LL = sizeof(long long);
  const unsigned S = sizeof(short);
  const unsigned C = sizeof(char);
  const unsigned D = sizeof(double);
  const unsigned F = sizeof(float);

  testScanf("%d", 1, I);
  testScanf("%d%d%d", 3, I, I, I);
  testScanf("ab%u%dc", 2, I, I);
  testScanf("%ld", 1, L);
  testScanf("%llu", 1, LL);
  testScanf("a %hd%hhx", 2, S, C);

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
  testScanf("%%10s", 0);
  testScanf("%*10s", 0);
  testScanf("%*d", 0);
}
