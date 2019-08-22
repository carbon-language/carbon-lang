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

#include "canonicalize-omp.h"
#include "../parser/parse-tree-visitor.h"

// After Loop Canonicalization, rewrite OpenMP parse tree to make OpenMP
// Constructs more structured which provide explicit scopes for later
// structural checks and semantic analysis.
//   1. move structured DoConstruct and OmpEndLoopDirective into
//      OpenMPLoopConstruct. Compilation will not proceed in case of errors
//      after this pass.
//   2. TBD
namespace Fortran::semantics {

using namespace parser::literals;

class CanonicalizationOfOmp {
public:
  template<typename T> bool Pre(T &) { return true; }
  template<typename T> void Post(T &) {}
  CanonicalizationOfOmp(parser::Messages &messages) : messages_{messages} {}

  void Post(parser::Block &block) {
    for (auto it{block.begin()}; it != block.end(); ++it) {
      if (auto *ompCons{GetConstructIf<parser::OpenMPConstruct>(*it)}) {
        // OpenMPLoopConstruct
        if (auto *ompLoop{
                std::get_if<parser::OpenMPLoopConstruct>(&ompCons->u)}) {
          RewriteOpenMPLoopConstruct(*ompLoop, block, it);
        }
      } else if (auto *endDir{
                     GetConstructIf<parser::OmpEndLoopDirective>(*it)}) {
        // Unmatched OmpEndLoopDirective
        auto &dir{std::get<parser::OmpLoopDirective>(endDir->t)};
        messages_.Say(dir.source,
            "The %s directive must follow the DO loop associated with the "
            "loop construct"_err_en_US,
            parser::ToUpperCaseLetters(dir.source.ToString()));
      }
    }  // Block list
  }

private:
  template<typename T> T *GetConstructIf(parser::ExecutionPartConstruct &x) {
    if (auto *y{std::get_if<parser::ExecutableConstruct>(&x.u)}) {
      if (auto *z{std::get_if<common::Indirection<T>>(&y->u)}) {
        return &z->value();
      }
    }
    return nullptr;
  }

  void RewriteOpenMPLoopConstruct(parser::OpenMPLoopConstruct &x,
      parser::Block &block, parser::Block::iterator it) {
    // Check the sequence of DoConstruct and OmpEndLoopDirective
    // in the same iteration
    //
    // Original:
    //   ExecutableConstruct -> OpenMPConstruct -> OpenMPLoopConstruct
    //     OmpBeginLoopDirective
    //   ExecutableConstruct -> DoConstruct
    //   ExecutableConstruct -> OmpEndLoopDirective (if available)
    //
    // After rewriting:
    //   ExecutableConstruct -> OpenMPConstruct -> OpenMPLoopConstruct
    //     OmpBeginLoopDirective
    //     DoConstruct
    //     OmpEndLoopDirective (if available)
    parser::Block::iterator nextIt;
    auto &beginDir{std::get<parser::OmpBeginLoopDirective>(x.t)};
    auto &dir{std::get<parser::OmpLoopDirective>(beginDir.t)};

    nextIt = it;
    if (++nextIt != block.end()) {
      if (auto *doCons{GetConstructIf<parser::DoConstruct>(*nextIt)}) {
        if (doCons->GetLoopControl().has_value()) {
          // move DoConstruct
          std::get<std::optional<parser::DoConstruct>>(x.t) =
              std::move(*doCons);
          nextIt = block.erase(nextIt);

          // try to match OmpEndLoopDirective
          if (auto *endDir{
                  GetConstructIf<parser::OmpEndLoopDirective>(*nextIt)}) {
            std::get<std::optional<parser::OmpEndLoopDirective>>(x.t) =
                std::move(*endDir);
            nextIt = block.erase(nextIt);
          }
        } else {
          messages_.Say(dir.source,
              "DO loop after the %s directive must have loop control"_err_en_US,
              parser::ToUpperCaseLetters(dir.source.ToString()));
        }
        return;  // found do-loop
      }
    }
    messages_.Say(dir.source,
        "A DO loop must follow the %s directive"_err_en_US,
        parser::ToUpperCaseLetters(dir.source.ToString()));
  }

  parser::Messages &messages_;
};

bool CanonicalizeOmp(parser::Messages &messages, parser::Program &program) {
  CanonicalizationOfOmp omp{messages};
  Walk(program, omp);
  return !messages.AnyFatalError();
}
}
