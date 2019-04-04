// Copyright (c) 2018-2019, NVIDIA CORPORATION.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "testing.h"
#include "../../lib/evaluate/call.h"
#include "../../lib/evaluate/expression.h"
#include "../../lib/evaluate/fold.h"
#include "../../lib/evaluate/host.h"
#include "../../lib/evaluate/tools.h"
#include <tuple>

using namespace Fortran::evaluate;

template<typename A> std::string AsFortran(const A &x) {
  std::stringstream ss;
  ss << x;
  return ss.str();
}

// helper to call functions on all types from tuple
template<typename... T> struct RunOnTypes {};
template<typename Test, typename... T>
struct RunOnTypes<Test, std::tuple<T...>> {
  static void Run() { (..., Test::template Run<T>()); }
};

// test for fold.h GetScalarConstantValue function
struct TestGetScalarConstantValue {
  template<typename T> static void Run() {
    Expr<T> exprFullyTyped{Constant<T>{Scalar<T>{}}};
    Expr<SomeKind<T::category>> exprSomeKind{exprFullyTyped};
    Expr<SomeType> exprSomeType{exprSomeKind};
    TEST(GetScalarConstantValue<T>(exprFullyTyped).has_value());
    TEST(GetScalarConstantValue<T>(exprSomeKind).has_value());
    TEST(GetScalarConstantValue<T>(exprSomeType).has_value());
  }
};

template<typename T>
static FunctionRef<T> CreateIntrinsicElementalCall(
    const std::string &name, const Expr<T> &arg) {
  Fortran::semantics::Attrs attrs;
  attrs.set(Fortran::semantics::Attr::ELEMENTAL);
  ActualArguments args{ActualArgument{AsGenericExpr(arg)}};
  ProcedureDesignator intrinsic{
      SpecificIntrinsic{name, T::GetType(), 0, attrs}};
  return FunctionRef<T>{std::move(intrinsic), std::move(args)};
}

// Test flushSubnormalsToZero when folding with host runtime.
// Subnormal value flushing on host is handle in host.cc
// HostFloatingPointEnvironment::SetUpHostFloatingPointEnvironment

void TestSubnormalFlushing() {
  using R4 = Type<TypeCategory::Real, 4>;
  if constexpr (host::HostTypeExists<R4>()) {
    Fortran::parser::CharBlock src;
    Fortran::parser::ContextualMessages messages{src, nullptr};
    FoldingContext flushingContext{messages, defaultRounding, true};
    FoldingContext noFlushingContext{messages, defaultRounding, false};

    // Biggest IEEE 32bits subnormal value
    host::HostType<R4> subnormal{5.87747175411144e-39};
    Scalar<R4> x{host::CastHostToFortran<R4>(subnormal)};
    Expr<R4> arg{Constant<R4>{x}};
    FunctionRef<R4> func{CreateIntrinsicElementalCall("log", arg)};

    auto resFlushing{Fold(flushingContext, AsGenericExpr(func))};
    auto resNoFlushing{Fold(noFlushingContext, AsGenericExpr(func))};
    TEST("(-1._4/0.)" == AsFortran(resFlushing));
    TEST("(-1._4/0.)" != AsFortran(resNoFlushing));

    // Check that the NoFlushing gave a correct result
    if (auto *typedExpr{UnwrapExpr<Expr<R4>>(resNoFlushing)}) {
      if (auto y{GetScalarConstantValue(*typedExpr)}) {
        // log around zero is not very precise allow 2% error.
        host::HostType<R4> yhost{host::CastFortranToHost<R4>(*y)};
        TEST(std::abs(yhost - (-88.)) < 2);
      } else {
        TEST(false);
      }
    } else {
      TEST(false);
    }
  } else {
    TEST(false);  // Cannot run this test on the host
  }
}

int main() {
  RunOnTypes<TestGetScalarConstantValue, AllIntrinsicTypes>::Run();
  TestSubnormalFlushing();
  return testing::Complete();
}
