// Copyright (c) 2018-2019, NVIDIA CORPORATION.  All rights reserved.
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

// The parse tree has slots in which pointers to the results of semantic
// analysis may be placed.  When using the parser without the semantics
// libraries, as here, we need to stub out the dependences on the external
// destructors, which will never actually be called.

#include "../../lib/common/indirection.h"

namespace Fortran::evaluate {
struct GenericExprWrapper {
  ~GenericExprWrapper();
};
GenericExprWrapper::~GenericExprWrapper() {}
struct GenericAssignmentWrapper {
  ~GenericAssignmentWrapper();
};
GenericAssignmentWrapper::~GenericAssignmentWrapper() {}
struct ProcedureRef {
  ~ProcedureRef();
};
ProcedureRef::~ProcedureRef() {}
}

DEFINE_DELETER(Fortran::evaluate::GenericExprWrapper)
DEFINE_DELETER(Fortran::evaluate::GenericAssignmentWrapper)
DEFINE_DELETER(Fortran::evaluate::ProcedureRef)
