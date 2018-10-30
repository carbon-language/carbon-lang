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

#ifndef FORTRAN_INTERMEDIATEREPRESENTATION_PROCEDURE_H_
#define FORTRAN_INTERMEDIATEREPRESENTATION_PROCEDURE_H_

#include "mixin.h"
#include "program.h"
#include "region.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"

namespace Fortran::IntermediateRepresentation {

struct Program;
struct Region;
struct GraphWriter;

struct Procedure final : public llvm::ilist_node<Procedure>,
                         public ChildMixin<Procedure, Program> {
  friend GraphWriter;
  friend Program;
  friend Region;
  using BasicBlockListType = llvm::iplist<BasicBlock>;
  using RegionListType = llvm::iplist<Region>;
  using iterator = BasicBlockListType::iterator;
  using const_iterator = BasicBlockListType::const_iterator;
  using reverse_iterator = BasicBlockListType::reverse_iterator;
  using const_reverse_iterator = BasicBlockListType::const_reverse_iterator;

  Procedure(const Procedure &) = delete;
  Procedure &operator=(const Procedure &) = delete;
  ~Procedure();
  BasicBlockListType &GetBlocks() { return basicBlockList_; }
  BasicBlockListType &getSublist(BasicBlock *) { return GetBlocks(); }
  RegionListType &GetRegions() { return regionList_; }
  RegionListType &getSublist(Region *) { return GetRegions(); }
  iterator begin() { return basicBlockList_.begin(); }
  const_iterator begin() const { return basicBlockList_.begin(); }
  iterator end() { return basicBlockList_.end(); }
  const_iterator end() const { return basicBlockList_.end(); }
  Region *getLastRegion() { return &regionList_.back(); }
  BasicBlock *StartBlock() { return &basicBlockList_.front(); }
  static Procedure *Create(Program *prog, FunctionType *ty,
      LinkageTypes linkage, unsigned addrSpace = 0u,
      const llvm::Twine &name = "", Procedure *before = nullptr) {
    return new Procedure(prog, ty, linkage, addrSpace, name, before);
  }
  void setParent(Program *p) { parent = p; }
  bool hasName() const { return !getName().empty(); }
  llvm::StringRef getName() const { return name_; }
  void FlattenRegions();

private:
  RegionListType regionList_;
  BasicBlockListType basicBlockList_;
  FunctionType *procType_;
  LinkageTypes linkage_;
  unsigned addressSpace_;
  const std::string name_;

  explicit Procedure(Program *program, FunctionType *ty, LinkageTypes lt,
      unsigned addrSpace, const llvm::Twine &n, Procedure *before);
  Region *insertBefore(Region *region, Region *before = nullptr);
};

}

#endif
