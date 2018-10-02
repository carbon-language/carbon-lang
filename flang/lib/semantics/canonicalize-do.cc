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

namespace Fortran::parser {

class CanonicalizationOfDoLoops {
public:
  CanonicalizationOfDoLoops() = default;

  template<typename T> bool Pre(T &) { return true; }
  template<typename T> void Post(T &) {}
  bool Pre(ExecutionPart &executionPart) {
    const auto &endIter{executionPart.v.end()};
    currentList_ = &executionPart.v;
    for (auto iter{executionPart.v.begin()}; iter != endIter; ++iter) {
      iter = ConvertLabelDoToStructuredDo(iter);
    }
    return false;
  }
  template<typename T> bool Pre(Statement<T> &statement) {
    if (!labels_.empty() && statement.label.has_value() &&
        labels_.back() == *statement.label) {
      auto currentLabel{labels_.back()};
      if constexpr (std::is_same_v<T, common::Indirection<EndDoStmt>>) {
        std::get<ExecutableConstruct>(currentIter_->u).u =
            Statement<ActionStmt>{
                std::optional<Label>{currentLabel}, ContinueStmt{}};
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
  Block ExtractBlock(Block::iterator beginLoop, Block::iterator endLoop) {
    Block block;
    block.splice(block.begin(), *currentList_, ++beginLoop, ++endLoop);
    return block;
  }
  std::optional<LoopControl> CreateLoopControl(
      std::optional<LoopControl> &loopControlOpt) {
    if (loopControlOpt.has_value()) {
      return std::optional<LoopControl>(LoopControl{loopControlOpt->u});
    }
    return std::optional<LoopControl>{};
  }
  std::optional<LoopControl> ExtractLoopControl(
      const Block::iterator &startLoop) {
    return CreateLoopControl(std::get<std::optional<LoopControl>>(
        std::get<Statement<common::Indirection<LabelDoStmt>>>(
            std::get<ExecutableConstruct>(startLoop->u).u)
            .statement->t));
  }
  Block::iterator MakeCanonicalForm(
      const Block::iterator &startLoop, const Block::iterator &endLoop) {
    std::get<ExecutableConstruct>(startLoop->u).u =
        common::Indirection<DoConstruct>{std::make_tuple(
            Statement<NonLabelDoStmt>{std::optional<Label>{},
                NonLabelDoStmt{std::make_tuple(
                    std::optional<Name>{}, ExtractLoopControl(startLoop))}},
            ExtractBlock(startLoop, endLoop),
            Statement<EndDoStmt>{
                std::optional<Label>{}, EndDoStmt{std::optional<Name>{}}})};
    return startLoop;
  }
  Block::iterator ConvertLabelDoToStructuredDo(const Block::iterator &iter) {
    currentIter_ = iter;
    ExecutionPartConstruct &executionPartConstruct{*iter};
    if (auto *executableConstruct{
            std::get_if<ExecutableConstruct>(&executionPartConstruct.u)}) {
      if (auto *labelDoLoop{
              std::get_if<Statement<common::Indirection<LabelDoStmt>>>(
                  &executableConstruct->u)}) {
        labelDoIters_.push_back(iter);
        labels_.push_back(std::get<Label>(labelDoLoop->statement->t));
      } else if (!labels_.empty()) {
        Walk(executableConstruct->u, *this);
      }
    }
    return currentIter_;
  }

  std::vector<Block::iterator> labelDoIters_;
  std::vector<Label> labels_;
  Block::iterator currentIter_;  ///< cursor for current ExecutionPartConstruct
  Block *currentList_;  ///< current ExectionPartConstruct list being traversed
};

void CanonicalizeDo(Program &program) {
  CanonicalizationOfDoLoops canonicalizationOfDoLoops;
  Walk(program, canonicalizationOfDoLoops);
}

}  // namespace Fortran::parser
