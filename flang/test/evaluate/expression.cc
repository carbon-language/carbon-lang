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
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <iostream>  // TODO pmk rm

using namespace Fortran::evaluate;

int main() {
  TEST(DefaultIntExpr::Result::Dump() == "Integer(4)");
  DefaultIntExpr ie{666};
  std::stringstream ss;
  ie.Dump(ss);
  TEST(ss.str() == "666");
  DefaultIntExpr one{DefaultIntExpr::Constant{1}};
  DefaultIntExpr incr{std::move(ie) + DefaultIntExpr{1}};
incr.Dump(std::cout) << '\n';
  LogicalExpr cmp{std::move(incr) <= DefaultIntExpr{2}};
  return testing::Complete();
}
