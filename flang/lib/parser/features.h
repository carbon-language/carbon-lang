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

#ifndef FORTRAN_PARSER_FEATURES_H_
#define FORTRAN_PARSER_FEATURES_H_

#include "../common/enum-set.h"
#include "../common/idioms.h"

namespace Fortran::parser {

ENUM_CLASS(LanguageFeature, BackslashEscapes, OldDebugLines,
    FixedFormContinuationWithColumn1Ampersand, LogicalAbbreviations,
    XOROperator, PunctuationInNames, OptionalFreeFormSpace, BOZExtensions,
    EmptyStatement, OpenMP, Extension, Deprecation)

using LanguageFeatures = common::EnumSet<LanguageFeature, 32>;

class LanguageFeatureControl {
public:
  LanguageFeatureControl() {
    disable_.set(LanguageFeature::OldDebugLines);
    // These features, if enabled, conflict with valid standard usage.
    disable_.set(LanguageFeature::LogicalAbbreviations);
    disable_.set(LanguageFeature::XOROperator);
  }
  LanguageFeatureControl(const LanguageFeatureControl &) = default;
  void Enable(LanguageFeature f, bool yes = true) { disable_.set(f, !yes); }
  void EnableWarning(LanguageFeature f, bool yes = true) { warn_.set(f, yes); }
  void WarnOnAllNonstandard(bool yes = true) { warnAll_ = yes; }
  bool IsEnabled(LanguageFeature f) const { return !disable_.test(f); }
  bool ShouldWarn(LanguageFeature f) const {
    return (warnAll_ && f != LanguageFeature::OpenMP) || warn_.test(f);
  }

private:
  LanguageFeatures disable_;
  LanguageFeatures warn_;
  bool warnAll_{false};
};

}  // namespace Fortran::parser
#endif  // FORTRAN_PARSER_FEATURES_H_
