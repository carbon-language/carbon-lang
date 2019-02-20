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

#include "program.h"

namespace Fortran::IntermediateRepresentation {

Region::Region(
    Procedure *procedure, Scope *scope, Region *inRegion, Region *insertBefore)
  : ChildMixin{procedure}, basicBlockList_{procedure->GetBlocks()},
    enclosingRegion_{inRegion}, scope_{scope} {
  if (enclosingRegion_) {
    enclosingRegion_->getSublist(static_cast<Region *>(nullptr))
        .push_back(this);
  } else {
    parent->insertBefore(this, insertBefore);
  }
}

Region::~Region() { basicBlockList_.clear(); }

void Region::insertBefore(BasicBlock *block, BasicBlock *before) {
  if (before) {
    basicBlockList_.insert(before->getIterator(), block);
  } else {
    basicBlockList_.push_back(block);
  }
}

std::vector<BasicBlock *> Region::getBlocks() {
  std::vector<BasicBlock *> result;
  for (auto &block : basicBlockList_) {
    if (block.getParent() == this) result.push_back(&block);
  }
  return result;
}

}
