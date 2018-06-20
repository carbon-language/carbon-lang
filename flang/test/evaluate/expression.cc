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
  using Int4 = Type<Category::Integer, 4>;
  TEST(Int4::Dump() == "Integer(4)");
  using IntEx4 = Expression<Int4>;
  IntEx4 ie{666};
  std::stringstream ss;
  ie.Dump(ss);
  TEST(ss.str() == "666");
  IntEx4 one{IntEx4::Constant{1}};
  IntEx4 incr{std::move(ie) + IntEx4{1}};
incr.Dump(std::cout) << '\n';
  using Log = Expression<Type<Category::Logical, 1>>;
  Log cmp{Log::EQ(std::move(incr), IntEx4{2})};
  return testing::Complete();
}
