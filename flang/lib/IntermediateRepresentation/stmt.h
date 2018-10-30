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

#ifndef FORTRAN_INTERMEDIATEREPRESENTATION_STMT_H_
#define FORTRAN_INTERMEDIATEREPRESENTATION_STMT_H_

#include "basicblock.h"
#include "mixin.h"
#include "statements.h"

namespace Fortran::IntermediateRepresentation {

/// Sum type over all statement classes
struct Statement : public SumTypeMixin<std::variant<
#define HANDLE_STMT(num, opcode, name) name,
#define HANDLE_LAST_STMT(num, opcode, name) name
#include "statement.def"
                       >>,
                   public ChildMixin<Statement, BasicBlock>,
                   public llvm::ilist_node<Statement> {
  template<typename A>
  Statement(BasicBlock *p, A &&t) : SumTypeMixin{t}, ChildMixin{p} {
    parent->insertBefore(this);
  }
  std::string dump() const;
};

inline std::list<BasicBlock *> succ_list(BasicBlock &block) {
  if (auto *terminator{block.getTerminator()}) {
    return reinterpret_cast<const TerminatorStmt_impl *>(&terminator->u)
        ->succ_blocks();
  }
  // CHECK(false && "block does not have terminator");
  return {};
}

}

#endif
