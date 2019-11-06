// Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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

#include "features.h"
#include "Fortran.h"
#include "idioms.h"

namespace Fortran::common {

std::vector<const char *> LanguageFeatureControl::GetNames(
    LogicalOperator opr) const {
  std::vector<const char *> result;
  result.push_back(AsFortran(opr));
  if (opr == LogicalOperator::Neqv && IsEnabled(LanguageFeature::XOROperator)) {
    result.push_back(".xor.");
  }
  if (IsEnabled(LanguageFeature::LogicalAbbreviations)) {
    switch (opr) {
      SWITCH_COVERS_ALL_CASES
    case LogicalOperator::And: result.push_back(".a."); break;
    case LogicalOperator::Or: result.push_back(".o."); break;
    case LogicalOperator::Not: result.push_back(".n."); break;
    case LogicalOperator::Neqv:
      if (IsEnabled(LanguageFeature::XOROperator)) {
        result.push_back(".x.");
      }
      break;
    case LogicalOperator::Eqv: break;
    }
  }
  return result;
}

std::vector<const char *> LanguageFeatureControl::GetNames(
    RelationalOperator opr) const {
  switch (opr) {
    SWITCH_COVERS_ALL_CASES
  case RelationalOperator::LT: return {".lt.", "<"};
  case RelationalOperator::LE: return {".le.", "<="};
  case RelationalOperator::EQ: return {".eq.", "=="};
  case RelationalOperator::GE: return {".ge.", ">="};
  case RelationalOperator::GT: return {".gt.", ">"};
  case RelationalOperator::NE:
    if (IsEnabled(LanguageFeature::AlternativeNE)) {
      return {".ne.", "/=", "<>"};
    } else {
      return {".ne.", "/="};
    }
  }
}

}
