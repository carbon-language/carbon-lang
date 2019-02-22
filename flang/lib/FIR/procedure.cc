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

#include "procedure.h"

namespace Fortran::FIR {

Procedure::Procedure(Program *program, FunctionType *ty, LinkageTypes lt,
    unsigned addrSpace, const llvm::Twine &n, Procedure *before)
  : ChildMixin{program}, procType_{ty}, linkage_{lt},
    addressSpace_{addrSpace}, name_{n.str()} {
  Region::Create(this);
  parent->insertBefore(this, before);
}

Procedure::~Procedure() { regionList_.clear(); }

Region *Procedure::insertBefore(Region *region, Region *before) {
  if (before) {
    regionList_.insert(before->getIterator(), region);
  } else {
    regionList_.push_back(region);
  }
  return region;
}

template<typename T>
static void AddCountScopes(
    unsigned count, BasicBlock *block, T callback, semantics::Scope *scope) {
  for (; count; --count) {
    block->insertBefore(
        new Statement(block, callback(scope)), block->getTerminator());
  }
}

inline static void AddEnterScopes(unsigned count, BasicBlock *block) {
  AddCountScopes(
      count, block, ScopeEnterStmt::Create, /*TODO: thread scope? */ nullptr);
}

inline static void AddExitScopes(unsigned count, BasicBlock *block) {
  AddCountScopes(
      count, block, ScopeExitStmt::Create, /*TODO: thread scope? */ nullptr);
}

static bool DistinctScopes(Region *region1, Region *region2) {
  return region1->HasScope() && region2->HasScope() &&
      region1->GetScope() != region2->GetScope();
}

void Procedure::FlattenRegions() {
  for (auto &block : GetBlocks()) {
    auto *region{block.GetRegion()};
    if (!region->IsOutermost()) {
      for (auto *succ : succ_list(block)) {
        auto *succRegion{succ->GetRegion()};
        if (succRegion != region &&
            DistinctScopes(succRegion, region)) {
          if (IsAncestor(region, succRegion)) {
            AddEnterScopes(RegionDepth(region, succRegion),
                succ->SplitEdge(&block));
          } else if (IsAncestor(succRegion, region)) {
            AddExitScopes(RegionDepth(succRegion, region),
                block.SplitEdge(succ));
          } else {
            // TODO: edge to a cousin region
            CHECK(false);
          }
        }
      }
    }
  }
}

}
