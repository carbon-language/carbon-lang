// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/error.h"

#include <gtest/gtest.h>

#include <concepts>

#include "common/error_test_helpers.h"
#include "common/raw_string_ostream.h"

namespace Carbon {
namespace {

using ::Carbon::Testing::IsError;
using ::Carbon::Testing::IsSuccess;
using ::testing::Eq;

TEST(ErrorTest, Error) {
  Error err("test");
  EXPECT_EQ(err.message(), "test");
}

TEST(ErrorTest, ErrorEmptyString) {
  ASSERT_DEATH({ Error err(""); }, "CHECK failure at");
}

auto IndirectError() -> Error { return Error("test"); }

TEST(ErrorTest, IndirectError) { EXPECT_EQ(IndirectError().message(), "test"); }

TEST(ErrorTest, ErrorBuilderOperatorImplicitCast) {
  ErrorOr<int> result = ErrorBuilder() << "msg";
  EXPECT_THAT(result, IsError("msg"));
}

TEST(ErrorTest, StreamError) {
  Error result = ErrorBuilder("TestFunc") << "msg";
  RawStringOstream result_stream;
  result_stream << result;
  EXPECT_EQ(result_stream.TakeStr(), "TestFunc: msg");
}

class CustomError : public ErrorBase<CustomError> {
 public:
  auto Print(llvm::raw_ostream& os) const -> void {
    os << "Custom test error!";
  }
};

template <typename ErrorT>
class ErrorOrTest : public ::testing::Test {
 public:
  auto ErrorStr() -> std::string {
    if constexpr (std::same_as<ErrorT, Error>) {
      return "test error";
    } else if constexpr (std::same_as<ErrorT, CustomError>) {
      return CustomError().ToString();
    } else {
      static_assert(false, "Unsupported custom error type!");
    }
  }

  auto MakeError() -> ErrorT {
    if constexpr (std::same_as<ErrorT, Error>) {
      return Error("test error");
    } else if constexpr (std::same_as<ErrorT, CustomError>) {
      return CustomError();
    } else {
      static_assert(false, "Unsupported custom error type!");
    }
  }
};

using ErrorOrTestParams = ::testing::Types<Error, CustomError>;
TYPED_TEST_SUITE(ErrorOrTest, ErrorOrTestParams);

TYPED_TEST(ErrorOrTest, ErrorOr) {
  using TestErrorOr = ErrorOr<int, TypeParam>;
  TestErrorOr err(this->MakeError());

  EXPECT_THAT(err, IsError(this->ErrorStr()));
}

TYPED_TEST(ErrorOrTest, ErrorOrValue) {
  using TestErrorOr = ErrorOr<int, TypeParam>;
  EXPECT_TRUE(TestErrorOr(0).ok());
}

template <typename ErrorT, typename Fixture>
auto IndirectErrorOrTest(Fixture& fixture) -> ErrorOr<int, ErrorT> {
  return fixture.MakeError();
}

TYPED_TEST(ErrorOrTest, IndirectErrorOr) {
  EXPECT_FALSE(IndirectErrorOrTest<TypeParam>(*this).ok());
}

struct Val {
  int val;
};

TYPED_TEST(ErrorOrTest, ErrorOrArrowOp) {
  using TestErrorOr = ErrorOr<Val, TypeParam>;
  TestErrorOr err({1});
  EXPECT_EQ(err->val, 1);
}

TYPED_TEST(ErrorOrTest, ErrorOrReference) {
  using TestErrorOr = ErrorOr<Val&, TypeParam>;
  Val val = {1};
  TestErrorOr maybe_val(val);
  EXPECT_EQ(maybe_val->val, 1);
}

template <typename ErrorT>
auto IndirectErrorOrSuccessTest() -> ErrorOr<Success, ErrorT> {
  return Success();
}

TYPED_TEST(ErrorOrTest, IndirectErrorOrSuccess) {
  EXPECT_TRUE(IndirectErrorOrSuccessTest<TypeParam>().ok());
}

TYPED_TEST(ErrorOrTest, ReturnIfErrorNoError) {
  using TestErrorOr = ErrorOr<Success, TypeParam>;
  auto result = []() -> TestErrorOr {
    CARBON_RETURN_IF_ERROR(TestErrorOr(Success()));
    CARBON_RETURN_IF_ERROR(TestErrorOr(Success()));
    return Success();
  }();
  EXPECT_TRUE(result.ok());
}

TYPED_TEST(ErrorOrTest, ReturnIfErrorHasError) {
  using TestErrorOr = ErrorOr<Success, TypeParam>;
  auto result = [this]() -> TestErrorOr {
    CARBON_RETURN_IF_ERROR(TestErrorOr(Success()));
    CARBON_RETURN_IF_ERROR(TestErrorOr(this->MakeError()));
    return Success();
  }();
  EXPECT_THAT(result, IsError(this->ErrorStr()));
}

TYPED_TEST(ErrorOrTest, AssignOrReturnNoError) {
  using TestErrorOr = ErrorOr<int, TypeParam>;
  auto result = []() -> TestErrorOr {
    CARBON_ASSIGN_OR_RETURN(int a, TestErrorOr(1));
    CARBON_ASSIGN_OR_RETURN(const int b, TestErrorOr(2));
    int c = 0;
    CARBON_ASSIGN_OR_RETURN(c, TestErrorOr(3));
    return a + b + c;
  }();
  EXPECT_THAT(result, IsSuccess(Eq(6)));
}

TYPED_TEST(ErrorOrTest, AssignOrReturnHasDirectError) {
  using TestErrorOr = ErrorOr<int, TypeParam>;
  auto result = [this]() -> TestErrorOr {
    CARBON_RETURN_IF_ERROR(TestErrorOr(this->MakeError()));
    return 0;
  }();
  EXPECT_THAT(result, IsError(this->ErrorStr()));
}

TYPED_TEST(ErrorOrTest, AssignOrReturnHasErrorInExpected) {
  using TestErrorOr = ErrorOr<int, TypeParam>;
  auto result = [this]() -> TestErrorOr {
    CARBON_ASSIGN_OR_RETURN(int a, TestErrorOr(this->MakeError()));
    return a;
  }();
  EXPECT_THAT(result, IsError(this->ErrorStr()));
}

class AnotherCustomError : public ErrorBase<AnotherCustomError> {
 public:
  auto Print(llvm::raw_ostream& os) const -> void {
    os << "Another custom test error!";
  }

  explicit operator CustomError() { return CustomError(); }
};

TYPED_TEST(ErrorOrTest, AssignOrReturnNoErrorAcrossErrorTypes) {
  using TestErrorOr = ErrorOr<int, TypeParam>;
  auto result = []() -> ErrorOr<int> {
    CARBON_ASSIGN_OR_RETURN(int a, TestErrorOr(1));
    CARBON_ASSIGN_OR_RETURN(const int b, []() -> TestErrorOr {
      CARBON_ASSIGN_OR_RETURN(int inner, (ErrorOr<int, AnotherCustomError>(2)));
      return inner;
    }());
    int c = 0;
    CARBON_ASSIGN_OR_RETURN(c, TestErrorOr(3));
    return a + b + c;
  }();
  EXPECT_THAT(result, IsSuccess(Eq(6)));
}

TYPED_TEST(ErrorOrTest, AssignOrReturnErrorAcrossErrorTypes) {
  using TestErrorOr = ErrorOr<int, TypeParam>;
  auto result = []() -> ErrorOr<int> {
    CARBON_ASSIGN_OR_RETURN(int a, TestErrorOr(1));
    CARBON_ASSIGN_OR_RETURN(const int b, []() -> TestErrorOr {
      CARBON_ASSIGN_OR_RETURN(
          int inner, (ErrorOr<int, AnotherCustomError>(AnotherCustomError())));
      return inner;
    }());
    int c = 0;
    CARBON_ASSIGN_OR_RETURN(c, TestErrorOr(3));
    return a + b + c;
  }();

  // When directly using the `Error` type, the explicit custom type above has
  // its message preserved. When testing against `CustomError`, that one
  // overrides the message.
  if constexpr (std::same_as<TypeParam, Error>) {
    EXPECT_THAT(result, IsError("Another custom test error!"));
  } else {
    EXPECT_THAT(result, IsError("Custom test error!"));
  }
}

}  // namespace
}  // namespace Carbon
