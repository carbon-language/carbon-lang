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

#include "canonicalize-do.h"
#include "../parser/parse-tree-visitor.h"

namespace Fortran::semantics {

class CanonicalizationOfDoLoops {
public:
  CanonicalizationOfDoLoops() = default;

  template<typename T> bool Pre(T &) { return true; }
  template<typename T> void Post(T &) {}
  bool Pre(parser::ExecutionPart &executionPart) {
    const auto &endIter{executionPart.v.end()};
    currentList_ = &executionPart.v;
    for (auto iter{executionPart.v.begin()}; iter != endIter; ++iter) {
      iter = CheckStatement(iter);
    }
    return false;
  }
  template<typename T> bool Pre(parser::Statement<T> &statement) {
    if (!labels_.empty() && statement.label.has_value() &&
        labels_.back() == *statement.label) {
      auto currentLabel{labels_.back()};
      if constexpr (std::is_same_v<T, common::Indirection<parser::EndDoStmt>>) {
        std::get<parser::ExecutableConstruct>(currentIter_->u).u =
            parser::Statement<parser::ActionStmt>{
                std::optional<parser::Label>{currentLabel},
                parser::ContinueStmt{}};
      }
      do {
        currentIter_ = MakeCanonicalForm(labelDoIters_.back(), currentIter_);
        labelDoIters_.pop_back();
        labels_.pop_back();
      } while (!labels_.empty() && labels_.back() == currentLabel);
    }
    return false;
  }

private:
  parser::Block SpliceBlock(
      parser::Block::iterator beginLoop, parser::Block::iterator endLoop) {
    parser::Block block;
    block.splice(block.begin(), *currentList_, ++beginLoop, ++endLoop);
    return block;
  }
  std::optional<parser::LoopControl> CreateLoopControl(
      std::optional<parser::LoopControl> &loopControlOpt) {
    if (loopControlOpt.has_value()) {
      return std::optional<parser::LoopControl>(
          parser::LoopControl{loopControlOpt->u});
    }
    return std::optional<parser::LoopControl>{};
  }
  std::optional<parser::LoopControl> ExtractLoopControl(
      const parser::Block::iterator &startLoop) {
    return CreateLoopControl(std::get<std::optional<parser::LoopControl>>(
        std::get<parser::Statement<common::Indirection<parser::LabelDoStmt>>>(
            std::get<parser::ExecutableConstruct>(startLoop->u).u)
            .statement->t));
  }
  parser::Block::iterator MakeCanonicalForm(
      const parser::Block::iterator &startLoop,
      const parser::Block::iterator &endLoop) {
    std::get<parser::ExecutableConstruct>(startLoop->u).u =
        common::Indirection<parser::DoConstruct>{std::make_tuple(
            parser::Statement<parser::NonLabelDoStmt>{
                std::optional<parser::Label>{},
                parser::NonLabelDoStmt{
                    std::make_tuple(std::optional<parser::Name>{},
                        ExtractLoopControl(startLoop))}},
            SpliceBlock(startLoop, endLoop),
            parser::Statement<parser::EndDoStmt>{std::optional<parser::Label>{},
                parser::EndDoStmt{std::optional<parser::Name>{}}})};
    return startLoop;
  }
  parser::Block::iterator CheckStatement(const parser::Block::iterator &iter) {
    currentIter_ = iter;
    parser::ExecutionPartConstruct &executionPartConstruct{*iter};
    if (auto *executableConstruct = std::get_if<parser::ExecutableConstruct>(
            &executionPartConstruct.u)) {
      if (auto *labelDoLoop = std::get_if<
              parser::Statement<common::Indirection<parser::LabelDoStmt>>>(
              &executableConstruct->u)) {
        labelDoIters_.push_back(iter);
        labels_.push_back(std::get<parser::Label>(labelDoLoop->statement->t));
      } else if (!labels_.empty()) {
        parser::Walk(executableConstruct->u, *this);
      }
    }
    return currentIter_;
  }

  std::vector<parser::Block::iterator> labelDoIters_;
  std::vector<parser::Label> labels_;
  parser::Block::iterator currentIter_;
  parser::Block *currentList_;
};

void CanonicalizeDo(parser::Program &program) {
  CanonicalizationOfDoLoops canonicalizationOfDoLoops;
  parser::Walk(program, canonicalizationOfDoLoops);
}

}  // namespace Fortran::semantics
