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
