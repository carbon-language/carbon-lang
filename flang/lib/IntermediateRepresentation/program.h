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

#ifndef FORTRAN_INTERMEDIATEREPRESENTATION_PROGRAM_H_
#define FORTRAN_INTERMEDIATEREPRESENTATION_PROGRAM_H_

#include "common.h"
#include "procedure.h"
#include "../evaluate/type.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include <string>

namespace Fortran::IntermediateRepresentation {

struct Procedure;
struct GraphWriter;

struct Program final {
  friend GraphWriter;
  using ProcedureListType = llvm::iplist<Procedure>;
  using ProcedureMapType = llvm::StringMap<Procedure *>;

  explicit Program(llvm::StringRef id);
  ~Program();
  void insertBefore(Procedure *subprog, Procedure *before = nullptr);
  ProcedureListType &getSublist(Procedure *) { return procedureList_; }
  bool containsProcedure(llvm::StringRef name) {
    return procedureMap_.find(name) != procedureMap_.end();
  }
  std::string getName() const { return name_; }
  Procedure *getOrInsertProcedure(
      llvm::StringRef name, FunctionType *procTy, AttributeList attrs);

private:
  ProcedureListType procedureList_;
  ProcedureMapType procedureMap_;
  const std::string name_;
};

}

#endif
