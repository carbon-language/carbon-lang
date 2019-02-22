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

namespace Fortran::FIR {

Program::Program(llvm::StringRef id) : name_{id} {}

void Program::insertBefore(Procedure *subprog, Procedure *before) {
  if (before) {
    procedureList_.insert(before->getIterator(), subprog);
  } else {
    procedureList_.push_back(subprog);
  }
}

Procedure *Program::getOrInsertProcedure(
    llvm::StringRef name, FunctionType *procTy, AttributeList attrs) {
  llvm::StringMapEntry<Procedure *> *entry{nullptr};
  if (!name.empty()) {
    auto iter{procedureMap_.find(name)};
    if (iter != procedureMap_.end()) {
      return iter->getValue();
    }
    entry = &*procedureMap_.insert({name, nullptr}).first;
    name = entry->getKey();
  }
  auto *subp{Procedure::Create(this, procTy, LinkageTypes::Public, 0u, name)};
  if (entry) {
    entry->setValue(subp);
  }
  return subp;
}

}
