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

#include "../../lib/evaluate/intrinsics.h"
#include "testing.h"
#include "../../lib/evaluate/expression.h"
#include "../../lib/evaluate/tools.h"
#include "../../lib/parser/provenance.h"
#include <iostream>
#include <string>

namespace Fortran::evaluate {

template<typename A> auto Const(A &&x) -> Constant<TypeOf<A>> {
  return Constant<TypeOf<A>>{std::move(x)};
}

template<typename A> void Push(Arguments &args, A &&x) {
  args.emplace_back(AsGenericExpr(std::move(x)));
}
template<typename A, typename... As>
void Push(Arguments &args, A &&x, As &&... xs) {
  args.emplace_back(AsGenericExpr(std::move(x)));
  Push(args, std::move(xs)...);
}
template<typename... As> Arguments Args(As &&... xs) {
  Arguments args;
  Push(args, std::move(xs)...);
  return args;
}

void TestIntrinsics() {
  IntrinsicTypeDefaultKinds defaults;
  MATCH(4, defaults.defaultIntegerKind);
  MATCH(4, defaults.defaultRealKind);
  IntrinsicProcTable table{IntrinsicProcTable::Configure(defaults)};

  parser::CookedSource cooked;
  std::string name{"abs"};
  cooked.Put(name.data(), name.size());
  cooked.PutProvenance(cooked.allSources().AddCompilerInsertion(name));
  cooked.Marshal();
  TEST(cooked.data() == name);
  parser::CharBlock nameCharBlock{cooked.data().data(), name.size()};
  CallCharacteristics call{nameCharBlock, Args(Const(value::Integer<32>{1}))};
  parser::Messages buffer;
  parser::ContextualMessages messages{cooked.data(), &buffer};
  std::optional<SpecificIntrinsic> si{table.Probe(call, &messages)};
  TEST(si.has_value());
  TEST(buffer.empty());
  buffer.Emit(std::cout, cooked);
}
}  // namespace Fortran::evaluate

int main() {
  Fortran::evaluate::TestIntrinsics();
  return testing::Complete();
}
