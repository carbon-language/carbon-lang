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

#ifndef FORTRAN_SEMANTICS_CHECK_COARRAY_H_
#define FORTRAN_SEMANTICS_CHECK_COARRAY_H_

#include "semantics.h"
#include <list>

namespace Fortran::parser {
class CharBlock;
class MessageFixedText;
struct ChangeTeamStmt;
struct CoarrayAssociation;
struct FormTeamStmt;
struct ImageSelectorSpec;
struct SyncTeamStmt;
struct TeamValue;
}

namespace Fortran::semantics {

class CoarrayChecker : public virtual BaseChecker {
public:
  CoarrayChecker(SemanticsContext &context) : context_{context} {}
  void Leave(const parser::ChangeTeamStmt &);
  void Leave(const parser::SyncTeamStmt &);
  void Leave(const parser::ImageSelectorSpec &);
  void Leave(const parser::FormTeamStmt &);

private:
  SemanticsContext &context_;

  void CheckNamesAreDistinct(const std::list<parser::CoarrayAssociation> &);
  void CheckTeamValue(const parser::TeamValue &);
  void Say2(const parser::CharBlock &, parser::MessageFixedText &&,
      const parser::CharBlock &, parser::MessageFixedText &&);
};

}
#endif  // FORTRAN_SEMANTICS_CHECK_COARRAY_H_
