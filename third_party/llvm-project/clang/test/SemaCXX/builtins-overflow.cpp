// RUN: %clang_cc1 -fsyntax-only -std=c++17 -verify %s
// expected-no-diagnostics

#include <limits.h>
#include <stdint.h>

int a() {
  const int x = 3;
  static int z;
  constexpr int *y = &z;
  return []() { return __builtin_sub_overflow((int)x, (int)x, (int *)y); }();
}
int a2() {
  const int x = 3;
  static int z;
  constexpr int *y = &z;
  return []() { return __builtin_sub_overflow(x, x, y); }();
}

template<typename T>
struct Result {
  bool B;
  T Value;
  constexpr bool operator==(const Result<T> &Other) {
    return B == Other.B && Value == Other.Value;
  }
};


template <typename RET, typename LHS, typename RHS>
constexpr Result<RET> add(LHS &&lhs, RHS &&rhs) {
  RET sum{};
  return {__builtin_add_overflow(lhs, rhs, &sum), sum};
}

static_assert(add<short>(static_cast<char>(120), static_cast<char>(10)) == Result<short>{false, 130});
static_assert(add<char>(static_cast<short>(120), static_cast<short>(10)) == Result<char>{true, -126});
static_assert(add<unsigned int>(INT_MAX, INT_MAX) == Result<unsigned int>{false, static_cast<unsigned int>(INT_MAX) * 2u});
static_assert(add<int>(static_cast<unsigned int>(INT_MAX), 1u) == Result<int>{true, INT_MIN});

static_assert(add<int>(17, 22) == Result<int>{false, 39});
static_assert(add<int>(INT_MAX - 22, 24) == Result<int>{true, INT_MIN + 1});
static_assert(add<int>(INT_MIN + 22, -23) == Result<int>{true, INT_MAX});

template <typename RET, typename LHS, typename RHS>
constexpr Result<RET> sub(LHS &&lhs, RHS &&rhs) {
  RET sum{};
  return {__builtin_sub_overflow(lhs, rhs, &sum), sum};
}

static_assert(sub<unsigned char>(static_cast<char>(0),static_cast<char>(1)) == Result<unsigned char>{true, UCHAR_MAX});
static_assert(sub<char>(static_cast<unsigned char>(0),static_cast<unsigned char>(1)) == Result<char>{false, -1});
static_assert(sub<unsigned short>(static_cast<short>(0),static_cast<short>(1)) == Result<unsigned short>{true, USHRT_MAX});
static_assert(sub<uint8_t>(static_cast<uint8_t>(255),static_cast<int>(100)) == Result<uint8_t>{false, 155});

static_assert(sub<int>(17,22) == Result<int>{false, -5});
static_assert(sub<int>(INT_MAX - 22, -23) == Result<int>{true, INT_MIN});
static_assert(sub<int>(INT_MIN + 22, 23) == Result<int>{true, INT_MAX});

template <typename RET, typename LHS, typename RHS>
constexpr Result<RET> mul(LHS &&lhs, RHS &&rhs) {
  RET sum{};
  return {__builtin_mul_overflow(lhs, rhs, &sum), sum};
}

static_assert(mul<int>(17,22) == Result<int>{false, 374});
static_assert(mul<int>(INT_MAX / 22, 23) == Result<int>{true, -2049870757});
static_assert(mul<int>(INT_MIN / 22, -23) == Result<int>{true, -2049870757});

constexpr Result<int> sadd(int lhs, int rhs) {
  int sum{};
  return {__builtin_sadd_overflow(lhs, rhs, &sum), sum};
}

static_assert(sadd(17,22) == Result<int>{false, 39});
static_assert(sadd(INT_MAX - 22, 23) == Result<int>{true, INT_MIN});
static_assert(sadd(INT_MIN + 22, -23) == Result<int>{true, INT_MAX});

constexpr Result<int> ssub(int lhs, int rhs) {
  int sum{};
  return {__builtin_ssub_overflow(lhs, rhs, &sum), sum};
}

static_assert(ssub(17,22) == Result<int>{false, -5});
static_assert(ssub(INT_MAX - 22, -23) == Result<int>{true, INT_MIN});
static_assert(ssub(INT_MIN + 22, 23) == Result<int>{true, INT_MAX});

constexpr Result<int> smul(int lhs, int rhs) {
  int sum{};
  return {__builtin_smul_overflow(lhs, rhs, &sum), sum};
}

static_assert(smul(17,22) == Result<int>{false, 374});
static_assert(smul(INT_MAX / 22, 23) == Result<int>{true, -2049870757});
static_assert(smul(INT_MIN / 22, -23) == Result<int>{true, -2049870757});

