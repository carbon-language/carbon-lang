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

// Static declaration checks

#ifndef FORTRAN_SEMANTICS_CHECK_DECLARATIONS_H_
#define FORTRAN_SEMANTICS_CHECK_DECLARATIONS_H_

#include <set>

namespace Fortran::semantics {
class DerivedTypeSpec;
class SemanticsContext;
class Symbol;

void CheckDeclarations(SemanticsContext &);

class TypeInspector {
public:
  TypeInspector();
  ~TypeInspector();
  const Symbol *typeBoundProcedure() const { return typeBoundProcedure_; }
  const Symbol *finalProcedure() const { return finalProcedure_; }
  const Symbol *allocatable() const { return allocatable_; }
  const Symbol *coarray() const { return coarray_; }
  const Symbol *allocatableCoarray() const { return allocatableCoarray_; }
  void Inspect(const DerivedTypeSpec &);

private:
  void Inspect(const DerivedTypeSpec &, bool inParentChain);
  void Inspect(const Symbol &, bool inParentChain);

  const Symbol *typeBoundProcedure_{nullptr};
  const Symbol *finalProcedure_{nullptr};
  const Symbol *allocatable_{nullptr};
  const Symbol *coarray_{nullptr};
  const Symbol *allocatableCoarray_{nullptr};
  std::set<const DerivedTypeSpec *> inspected_;
};
}
#endif
