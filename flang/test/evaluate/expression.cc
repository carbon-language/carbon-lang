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

#include "../../lib/evaluate/expression.h"
#include "testing.h"
#include "../../lib/evaluate/fold.h"
#include "../../lib/evaluate/tools.h"
#include "../../lib/parser/message.h"
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>

using namespace Fortran::evaluate;

template<typename A> std::string AsFortran(const A &x) {
  std::stringstream ss;
  x.AsFortran(ss);
  return ss.str();
}

int main() {
  using DefaultIntegerExpr = Expr<Type<TypeCategory::Integer, 4>>;
  TEST(DefaultIntegerExpr::Result::AsFortran() == "Integer(4)");
  MATCH("666_4", AsFortran(DefaultIntegerExpr{666}));
  MATCH("-1_4", AsFortran(-DefaultIntegerExpr{1}));
  auto ex1{
      DefaultIntegerExpr{2} + DefaultIntegerExpr{3} * -DefaultIntegerExpr{4}};
  MATCH("2_4+3_4*(-4_4)", AsFortran(ex1));
  Fortran::parser::CharBlock src;
  Fortran::parser::ContextualMessages messages{src, nullptr};
  FoldingContext context{messages};
  ex1 = Fold(context, std::move(ex1));
  MATCH("-10_4", AsFortran(ex1));
  MATCH("1_4/2_4", AsFortran(DefaultIntegerExpr{1} / DefaultIntegerExpr{2}));
  DefaultIntegerExpr a{1};
  DefaultIntegerExpr b{2};
  MATCH("1_4", AsFortran(a));
  a = b;
  MATCH("2_4", AsFortran(a));
  MATCH("2_4", AsFortran(b));
  return testing::Complete();
}
