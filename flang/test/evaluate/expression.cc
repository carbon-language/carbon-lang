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

using namespace Fortran::evaluate;

int main() {
  using Int4 = Type<Category::Integer, 4>;
  using IntEx4 = Expression<Int4>;
  IntEx4 ie{value::Integer<32>(666)};
  IntEx4 one{value::Integer<32>(1)};
  IntEx4 incr{IntEx4::Add{std::move(ie), std::move(one)}};
  using Log = Expression<Type<Category::Logical, 1>>;
  IntEx4 two{value::Integer<32>(2)};
  Log cmp{Log::EQ(std::move(incr), std::move(two))};
  return testing::Complete();
}
