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

#include "../../lib/evaluate/expression.h"
#include "testing.h"
#include "../../lib/parser/message.h"
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>

using namespace Fortran::evaluate;

template<typename A> std::string Dump(const A &x) {
  std::stringstream ss;
  x.Dump(ss);
  return ss.str();
}

int main() {
  using DefaultIntegerExpr = Expr<DefaultInteger>;
  TEST(DefaultIntegerExpr::Result::Dump() == "Integer(4)");
  MATCH("666", Dump(DefaultIntegerExpr{666}));
  MATCH("(-1)", Dump(-DefaultIntegerExpr{1}));
  auto ex1{
      DefaultIntegerExpr{2} + DefaultIntegerExpr{3} * -DefaultIntegerExpr{4}};
  MATCH("(2+(3*(-4)))", Dump(ex1));
  Fortran::parser::CharBlock src;
  Fortran::parser::ContextualMessages messages{src, nullptr};
  FoldingContext context{messages};
  ex1.Fold(context);
  MATCH("-10", Dump(ex1));
  MATCH("(Integer(4)::6.LE.7)",
      Dump(DefaultIntegerExpr{6} <= DefaultIntegerExpr{7}));
  DefaultIntegerExpr a{1};
  DefaultIntegerExpr b{2};
  MATCH("(1/2)", Dump(a / b));
  MATCH("1", Dump(a));
  a = b;
  MATCH("2", Dump(a));
  MATCH("2", Dump(b));
  return testing::Complete();
}
