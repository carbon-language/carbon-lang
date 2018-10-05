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

struct LabelInfo {
  Block::iterator iter;
  Label label;
};

class CanonicalizationOfDoLoops {
public:
  CanonicalizationOfDoLoops(std::vector<LabelInfo> &labelInfos)
    : labelInfos_{labelInfos} {}
  template<typename T> bool Pre(T &) { return true; }
  template<typename T> void Post(T &) {}
  bool Pre(Block &block) { return VisitBlock(block); }
  template<typename T> bool Pre(Statement<T> &statement) {
    if (!labelInfos_.empty() && statement.label.has_value() &&
        labelInfos_.back().label == *statement.label) {
      auto currentLabel{labelInfos_.back().label};
      if constexpr (std::is_same_v<T, common::Indirection<EndDoStmt>>) {
        std::get<ExecutableConstruct>(currentPosition_.iter->u).u =
            Statement<ActionStmt>{
                std::optional<Label>{currentLabel}, ContinueStmt{}};
      }
      do {
        currentPosition_.iter = MakeCanonicalForm(*currentPosition_.block,
            labelInfos_.back().iter, currentPosition_.iter);
        labelInfos_.pop_back();
      } while (
          !labelInfos_.empty() && labelInfos_.back().label == currentLabel);
    }
    return false;
  }

private:
  bool VisitBlock(Block &block) {
    CanonicalizationOfDoLoops canonicalizationOfDoLoops{labelInfos_};
    canonicalizationOfDoLoops.TraverseBlock(block);
    return false;
  }
  void TraverseBlock(Block &block) {
    const auto &endIter{block.end()};
    currentPosition_.block = &block;
    for (auto iter{block.begin()}; iter != endIter; ++iter) {
      ExecutionPartConstruct &executionPartConstruct{*iter};
      currentPosition_.iter = iter;
      if (auto *executableConstruct{
              std::get_if<ExecutableConstruct>(&executionPartConstruct.u)}) {
        if (auto *labelDoLoop{
                std::get_if<Statement<common::Indirection<LabelDoStmt>>>(
                    &executableConstruct->u)}) {
          labelInfos_.push_back(
              LabelInfo{iter, std::get<Label>(labelDoLoop->statement->t)});
        }
      }
      Walk(executionPartConstruct.u, *this);  // may update currentPosition_
      iter = currentPosition_.iter;
    }
  }
  static Block ExtractBlock(
      Block &currentBlock, Block::iterator beginLoop, Block::iterator endLoop) {
    Block block;
    block.splice(block.begin(), currentBlock, ++beginLoop, ++endLoop);
    return block;
  }
  static Block::iterator MakeCanonicalForm(Block &currentBlock,
      const Block::iterator &startLoop, const Block::iterator &endLoop) {
    std::get<ExecutableConstruct>(startLoop->u).u =
        common::Indirection<DoConstruct>{std::make_tuple(
            Statement<NonLabelDoStmt>{std::optional<Label>{},
                NonLabelDoStmt{std::make_tuple(std::optional<Name>{},
                    std::move(std::get<std::optional<LoopControl>>(
                        std::get<Statement<common::Indirection<LabelDoStmt>>>(
                            std::get<ExecutableConstruct>(startLoop->u).u)
                            .statement->t)))}},
            ExtractBlock(currentBlock, startLoop, endLoop),
            Statement<EndDoStmt>{
                std::optional<Label>{}, EndDoStmt{std::optional<Name>{}}})};
    return startLoop;
  }

  std::vector<LabelInfo> &labelInfos_;
  struct TraversalInfo {
    Block::iterator iter;
    Block *block;
  } currentPosition_;
};

void CanonicalizeDo(Program &program) {
  std::vector<LabelInfo> labelInfos;
  CanonicalizationOfDoLoops canonicalizationOfDoLoops{labelInfos};
  Walk(program, canonicalizationOfDoLoops);
}

}  // namespace Fortran::parser
