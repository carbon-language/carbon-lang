//===-- flang/unittests/Runtime/Matmul.cpp--------- -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Runtime/matmul.h"
#include "gtest/gtest.h"
#include "tools.h"
#include "flang/Runtime/allocatable.h"
#include "flang/Runtime/cpp-type.h"
#include "flang/Runtime/descriptor.h"
#include "flang/Runtime/type-code.h"

using namespace Fortran::runtime;
using Fortran::common::TypeCategory;

TEST(Matmul, Basic) {
  // X 0 2 4   Y 6  9   V -1 -2
  //   1 3 5     7 10
  //             8 11
  auto x{MakeArray<TypeCategory::Integer, 4>(
      std::vector<int>{2, 3}, std::vector<std::int32_t>{0, 1, 2, 3, 4, 5})};
  auto y{MakeArray<TypeCategory::Integer, 2>(
      std::vector<int>{3, 2}, std::vector<std::int16_t>{6, 7, 8, 9, 10, 11})};
  auto v{MakeArray<TypeCategory::Integer, 8>(
      std::vector<int>{2}, std::vector<std::int64_t>{-1, -2})};
  StaticDescriptor<2, true> statDesc;
  Descriptor &result{statDesc.descriptor()};

  RTNAME(Matmul)(result, *x, *y, __FILE__, __LINE__);
  ASSERT_EQ(result.rank(), 2);
  EXPECT_EQ(result.GetDimension(0).LowerBound(), 1);
  EXPECT_EQ(result.GetDimension(0).Extent(), 2);
  EXPECT_EQ(result.GetDimension(1).LowerBound(), 1);
  EXPECT_EQ(result.GetDimension(1).Extent(), 2);
  ASSERT_EQ(result.type(), (TypeCode{TypeCategory::Integer, 4}));
  EXPECT_EQ(*result.ZeroBasedIndexedElement<std::int32_t>(0), 46);
  EXPECT_EQ(*result.ZeroBasedIndexedElement<std::int32_t>(1), 67);
  EXPECT_EQ(*result.ZeroBasedIndexedElement<std::int32_t>(2), 64);
  EXPECT_EQ(*result.ZeroBasedIndexedElement<std::int32_t>(3), 94);

  std::memset(
      result.raw().base_addr, 0, result.Elements() * result.ElementBytes());
  result.GetDimension(0).SetLowerBound(0);
  result.GetDimension(1).SetLowerBound(2);
  RTNAME(MatmulDirect)(result, *x, *y, __FILE__, __LINE__);
  EXPECT_EQ(*result.ZeroBasedIndexedElement<std::int32_t>(0), 46);
  EXPECT_EQ(*result.ZeroBasedIndexedElement<std::int32_t>(1), 67);
  EXPECT_EQ(*result.ZeroBasedIndexedElement<std::int32_t>(2), 64);
  EXPECT_EQ(*result.ZeroBasedIndexedElement<std::int32_t>(3), 94);
  result.Destroy();

  RTNAME(Matmul)(result, *v, *x, __FILE__, __LINE__);
  ASSERT_EQ(result.rank(), 1);
  EXPECT_EQ(result.GetDimension(0).LowerBound(), 1);
  EXPECT_EQ(result.GetDimension(0).Extent(), 3);
  ASSERT_EQ(result.type(), (TypeCode{TypeCategory::Integer, 8}));
  EXPECT_EQ(*result.ZeroBasedIndexedElement<std::int64_t>(0), -2);
  EXPECT_EQ(*result.ZeroBasedIndexedElement<std::int64_t>(1), -8);
  EXPECT_EQ(*result.ZeroBasedIndexedElement<std::int64_t>(2), -14);
  result.Destroy();

  RTNAME(Matmul)(result, *y, *v, __FILE__, __LINE__);
  ASSERT_EQ(result.rank(), 1);
  EXPECT_EQ(result.GetDimension(0).LowerBound(), 1);
  EXPECT_EQ(result.GetDimension(0).Extent(), 3);
  ASSERT_EQ(result.type(), (TypeCode{TypeCategory::Integer, 8}));
  EXPECT_EQ(*result.ZeroBasedIndexedElement<std::int64_t>(0), -24);
  EXPECT_EQ(*result.ZeroBasedIndexedElement<std::int64_t>(1), -27);
  EXPECT_EQ(*result.ZeroBasedIndexedElement<std::int64_t>(2), -30);
  result.Destroy();

  // X F F T  Y F T
  //   F T T    F T
  //            F F
  auto xLog{MakeArray<TypeCategory::Logical, 1>(std::vector<int>{2, 3},
      std::vector<std::uint8_t>{false, false, false, true, true, false})};
  auto yLog{MakeArray<TypeCategory::Logical, 2>(std::vector<int>{3, 2},
      std::vector<std::uint16_t>{false, false, false, true, true, false})};
  RTNAME(Matmul)(result, *xLog, *yLog, __FILE__, __LINE__);
  ASSERT_EQ(result.rank(), 2);
  EXPECT_EQ(result.GetDimension(0).LowerBound(), 1);
  EXPECT_EQ(result.GetDimension(0).Extent(), 2);
  EXPECT_EQ(result.GetDimension(1).LowerBound(), 1);
  EXPECT_EQ(result.GetDimension(1).Extent(), 2);
  ASSERT_EQ(result.type(), (TypeCode{TypeCategory::Logical, 2}));
  EXPECT_FALSE(
      static_cast<bool>(*result.ZeroBasedIndexedElement<std::uint16_t>(0)));
  EXPECT_FALSE(
      static_cast<bool>(*result.ZeroBasedIndexedElement<std::uint16_t>(1)));
  EXPECT_FALSE(
      static_cast<bool>(*result.ZeroBasedIndexedElement<std::uint16_t>(2)));
  EXPECT_TRUE(
      static_cast<bool>(*result.ZeroBasedIndexedElement<std::uint16_t>(3)));
}
