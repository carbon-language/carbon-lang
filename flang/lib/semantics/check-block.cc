// Copyright (c) 2019, ARM Ltd.  All rights reserved.
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

#include <iostream>

#include "../parser/message.h"
#include "../parser/parse-tree-visitor.h"
#include "attr.h"
#include "check-block.h"
#include "scope.h"
#include "semantics.h"
#include "symbol.h"
#include "tools.h"
#include "type.h"

namespace Fortran::semantics {

class BlockEnforcement {
public:
  BlockEnforcement(parser::Messages &messages) : messages_{messages} {}
  std::set<parser::Label> labels() { return labels_; }
  template <typename T> bool Pre(const T &) { return true; }
  template <typename T> void Post(const T &) {}
  template <typename T> bool Pre(const parser::Statement<T> &statement) {
    currentStatementSourcePosition_ = statement.source;
    if (statement.label.has_value()) {
      labels_.insert(*statement.label);
    }
    return true;
  }
  // C1107$

  void Post(const parser::EquivalenceStmt &) {
    messages_.Say(
        currentStatementSourcePosition_,
        "EQUIVALENCE statement is not allowed in a BLOCK construct"_err_en_US);
  }

  void Post(const parser::StmtFunctionStmt &x) {
    std::cout << "statementxxxxxxxx";
    messages_.Say(
        currentStatementSourcePosition_,
        "STATEMENT FUNCTION is not allowed in a BLOCK construct"_err_en_US);
  }
  // C1108
  void Post(const parser::SaveStmt &x) {
    for (const parser::SavedEntity &y : x.v) {
      auto kind{std::get<parser::SavedEntity::Kind>(y.t)};
      if (kind == parser::SavedEntity::Kind::Common) {
        messages_.Say(
            currentStatementSourcePosition_,
            "COMMON BLOCK NAME specifier not allowed in a BLOCK construct"_err_en_US);
      }
    }
  }

private:
  std::set<parser::Label> labels_;
  parser::Messages &messages_;
  parser::CharBlock currentStatementSourcePosition_;
}; // end BlockEnforcement

class BlockContext {
public:
  BlockContext(SemanticsContext &context) : messages_{context.messages()} {}

  bool operator==(const BlockContext &x) const { return this == &x; }

  void Check(const parser::BlockConstruct &blockConstruct) {
    BlockEnforcement blockEnforcement{messages_};
    parser::Walk(std::get<parser::BlockSpecificationPart>(blockConstruct.t),
                 blockEnforcement);
  }

private:
  parser::Messages &messages_;
  // parser::CharBlock currentStatementSourcePosition_;
};

BlockChecker::BlockChecker(SemanticsContext &context)
    : context_{new BlockContext{context}} {}

BlockChecker::~BlockChecker() = default;

// 11.1.4  enforce semantics constraints on a Block  loop body
void BlockChecker::Leave(const parser::BlockConstruct &x) {
  context_.value().Check(x);
}
}
template class Fortran::common::Indirection<Fortran::semantics::BlockContext>;
