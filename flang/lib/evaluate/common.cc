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

#include "common.h"
#include "../common/idioms.h"

using namespace Fortran::parser::literals;

namespace Fortran::evaluate {

void RealFlagWarnings(
    FoldingContext &context, const RealFlags &flags, const char *operation) {
  if (flags.test(RealFlag::Overflow)) {
    context.messages().Say("overflow on %s"_en_US, operation);
  }
  if (flags.test(RealFlag::DivideByZero)) {
    context.messages().Say("division by zero on %s"_en_US, operation);
  }
  if (flags.test(RealFlag::InvalidArgument)) {
    context.messages().Say("invalid argument on %s"_en_US, operation);
  }
  if (flags.test(RealFlag::Underflow)) {
    context.messages().Say("underflow on %s"_en_US, operation);
  }
}

ConstantSubscript &FoldingContext::StartImpliedDo(
    parser::CharBlock name, ConstantSubscript n) {
  auto pair{impliedDos_.insert(std::make_pair(name, n))};
  CHECK(pair.second);
  return pair.first->second;
}

std::optional<ConstantSubscript> FoldingContext::GetImpliedDo(
    parser::CharBlock name) const {
  if (auto iter{impliedDos_.find(name)}; iter != impliedDos_.cend()) {
    return {iter->second};
  } else {
    return std::nullopt;
  }
}

void FoldingContext::EndImpliedDo(parser::CharBlock name) {
  auto iter{impliedDos_.find(name)};
  if (iter != impliedDos_.end()) {
    impliedDos_.erase(iter);
  }
}
}
