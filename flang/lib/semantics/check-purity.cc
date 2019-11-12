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

#include "check-purity.h"
#include "tools.h"
#include "../parser/parse-tree.h"

namespace Fortran::semantics {
void PurityChecker::Enter(const parser::ExecutableConstruct &exec) {
  if (InPureSubprogram() && IsImageControlStmt(exec)) {
    context_.Say(GetImageControlStmtLocation(exec),
        "An image control statement may not appear in a PURE subprogram"_err_en_US);
  }
}
void PurityChecker::Enter(const parser::SubroutineSubprogram &subr) {
  const auto &stmt{std::get<parser::Statement<parser::SubroutineStmt>>(subr.t)};
  Entered(
      stmt.source, std::get<std::list<parser::PrefixSpec>>(stmt.statement.t));
}

void PurityChecker::Leave(const parser::SubroutineSubprogram &) { Left(); }

void PurityChecker::Enter(const parser::FunctionSubprogram &func) {
  const auto &stmt{std::get<parser::Statement<parser::FunctionStmt>>(func.t)};
  Entered(
      stmt.source, std::get<std::list<parser::PrefixSpec>>(stmt.statement.t));
}

void PurityChecker::Leave(const parser::FunctionSubprogram &) { Left(); }

bool PurityChecker::InPureSubprogram() const {
  return pureDepth_ >= 0 && depth_ >= pureDepth_;
}

bool PurityChecker::HasPurePrefix(
    const std::list<parser::PrefixSpec> &prefixes) const {
  for (const parser::PrefixSpec &prefix : prefixes) {
    if (std::holds_alternative<parser::PrefixSpec::Pure>(prefix.u)) {
      return true;
    }
  }
  return false;
}

void PurityChecker::Entered(
    parser::CharBlock source, const std::list<parser::PrefixSpec> &prefixes) {
  if (depth_ == 2) {
    context_.messages().Say(source,
        "An internal subprogram may not contain an internal subprogram"_err_en_US);
  }
  if (HasPurePrefix(prefixes)) {
    if (pureDepth_ < 0) {
      pureDepth_ = depth_;
    }
  } else if (InPureSubprogram()) {
    context_.messages().Say(source,
        "An internal subprogram of a PURE subprogram must also be PURE"_err_en_US);
  }
  ++depth_;
}

void PurityChecker::Left() {
  if (pureDepth_ == --depth_) {
    pureDepth_ = -1;
  }
}

}
