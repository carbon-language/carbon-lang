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
#include "../../lib/evaluate/expression.h"
#include "../../lib/evaluate/fold.h"
#include "../../lib/evaluate/type.h"
#include <tuple>

using namespace Fortran::evaluate;

// helper to call functions on all types from tuple
template<typename... T> struct RunOnTypes {};
template<typename Test, typename... T>
struct RunOnTypes<Test, std::tuple<T...>> {
  static void Run() { (..., Test::template Run<T>()); }
};

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

int main() {
  using TestTypes = AllIntrinsicTypes;
  RunOnTypes<TestGetScalarConstantValue, TestTypes>::Run();
  return testing::Complete();
}
