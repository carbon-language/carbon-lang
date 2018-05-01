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

#ifndef FLANG_SEMA_STMT_H_
#define FLANG_SEMA_STMT_H_

#include "../parser/parse-tree.h"
#include <variant>

namespace Fortran::semantics {

enum class StmtClass {
#define DECLARE_STMT(Name, Class, Group, Text) Name,
#include "Stmt.def"
};

enum class StmtGroup {
  Single,  // A standalone statement
  Start,  // A statement that starts a construct
  Part,  // A statement that marks a part of a construct
  End  // A statement that ends a construct
};

constexpr StmtGroup StmtClassToGroup(StmtClass sc) {
  switch (sc) {
#define DECLARE_STMT(Name, Class, Group, Text) \
    case StmtClass::Name: return StmtGroup::Group; 

#include "Stmt.def"
  }
  // Hummm... should not happen but cannot add error or assert
  // in constexpr expression.
  return StmtGroup::Single;
}

constexpr const char *StmtClassName(StmtClass sc) {
  switch (sc) {
#define DECLARE_STMT(Name, Class, Group, Text) \
  case StmtClass::Name: return #Name;

#include "Stmt.def"
  }
  return "????";
}

constexpr const char *StmtClassText(StmtClass sc) {
  switch (sc) {
#define DECLARE_STMT(Name, Class, Group, Text) \
  case StmtClass::Name: return Text; 

#include "Stmt.def"
  }
  return "????";
}

}  // end of namespace Fortran::semantics

#endif // of FLANG_SEMA_STMT_H_
