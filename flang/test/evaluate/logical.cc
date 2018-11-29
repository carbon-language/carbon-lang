// Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
#include "../../lib/evaluate/type.h"
#include <cstdio>

template<int KIND> void testKind() {
  using Type =
      Fortran::evaluate::Type<Fortran::common::TypeCategory::Logical, KIND>;
  TEST(Fortran::evaluate::IsSpecificIntrinsicType<Type>);
  TEST(Type::category == Fortran::common::TypeCategory::Logical);
  TEST(Type::kind == KIND);
  using Value = Fortran::evaluate::Scalar<Type>;
  MATCH(8 * KIND, Value::bits);
  TEST(!Value{}.IsTrue());
  TEST(!Value{false}.IsTrue());
  TEST(Value{true}.IsTrue());
  TEST(Value{false}.NOT().IsTrue());
  TEST(!Value{true}.NOT().IsTrue());
  TEST(!Value{false}.AND(Value{false}).IsTrue());
  TEST(!Value{false}.AND(Value{true}).IsTrue());
  TEST(!Value{true}.AND(Value{false}).IsTrue());
  TEST(Value{true}.AND(Value{true}).IsTrue());
  TEST(!Value{false}.OR(Value{false}).IsTrue());
  TEST(Value{false}.OR(Value{true}).IsTrue());
  TEST(Value{true}.OR(Value{false}).IsTrue());
  TEST(Value{true}.OR(Value{true}).IsTrue());
  TEST(Value{false}.EQV(Value{false}).IsTrue());
  TEST(!Value{false}.EQV(Value{true}).IsTrue());
  TEST(!Value{true}.EQV(Value{false}).IsTrue());
  TEST(Value{true}.EQV(Value{true}).IsTrue());
  TEST(!Value{false}.NEQV(Value{false}).IsTrue());
  TEST(Value{false}.NEQV(Value{true}).IsTrue());
  TEST(Value{true}.NEQV(Value{false}).IsTrue());
  TEST(!Value{true}.NEQV(Value{true}).IsTrue());
}

int main() {
  testKind<1>();
  testKind<2>();
  testKind<4>();
  testKind<8>();
  return testing::Complete();
}
