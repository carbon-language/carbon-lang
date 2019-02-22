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

#ifndef FORTRAN_FIR_REGION_H_
#define FORTRAN_FIR_REGION_H_

#include "procedure.h"
#include "stmt.h"
#include "../semantics/semantics.h"

namespace Fortran::FIR {

class Procedure;
class BasicBlock;

class Region final : public llvm::ilist_node<Region>,
                     public ChildMixin<Region, Procedure> {
public:
  friend Procedure;
  friend BasicBlock;
  using BasicBlockListType = llvm::iplist<BasicBlock>;
  using AllocatableListType = std::list<const semantics::Symbol *>;
  using SubregionListType = llvm::iplist<Region>;
  using iterator = SubregionListType::iterator;
  using const_iterator = SubregionListType::const_iterator;
  using reverse_iterator = SubregionListType::reverse_iterator;
  using const_reverse_iterator = SubregionListType::const_reverse_iterator;

  Region(const Region &) = delete;
  Region &operator=(const Region &) = delete;
  ~Region();
  std::vector<BasicBlock *> getBlocks();
  std::vector<BasicBlock *> getSublist(BasicBlock *) { return getBlocks(); }
  SubregionListType &getSublist(Region *) { return subregionList_; }
  iterator begin() { return subregionList_.begin(); }
  const_iterator begin() const { return subregionList_.begin(); }
  iterator end() { return subregionList_.end(); }
  const_iterator end() const { return subregionList_.end(); }
  Region *GetEnclosing() const { return enclosingRegion_; }
  bool IsOutermost() const { return GetEnclosing() == nullptr; }
  static Region *Create(Procedure *procedure, Scope *scope = nullptr,
      Region *inRegion = nullptr, Region *insertBefore = nullptr) {
    return new Region(procedure, scope, inRegion, insertBefore);
  }
  bool HasScope() const { return scope_.has_value(); }
  Scope *GetScope() const { return scope_ ? scope_.value() : nullptr; }

private:
  BasicBlockListType &basicBlockList_;
  AllocatableListType allocatableList_;
  SubregionListType subregionList_;  // direct descendants
  Region *enclosingRegion_;  // parent in nesting tree
  std::optional<Scope *> scope_;

  explicit Region(Procedure *procedure, Scope *scope, Region *inRegion,
      Region *insertBefore);
  void insertBefore(BasicBlock *block, BasicBlock *before);
};

inline bool IsAncestor(const Region *fromRegion, const Region *toRegion) {
  CHECK(fromRegion && toRegion);
  for (const auto *region{fromRegion->GetEnclosing()}; region;
       region = region->GetEnclosing()) {
    if (region == toRegion) return true;
  }
  return false;
}

inline unsigned RegionDepth(const Region *fromRegion, const Region *toRegion) {
  CHECK(IsAncestor(fromRegion, toRegion));
  unsigned result{0u};
  for (const auto *region{fromRegion}; region != toRegion;
       region = region->GetEnclosing()) {
    ++result;
  }
  return result;
}

}

#endif
