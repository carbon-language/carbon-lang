#ifndef FLANG_SEMA_STMT_H
#define FLANG_SEMA_STMT_H

#include "../../../lib/parser/parse-tree.h"
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

#include "flang/Sema/Stmt.def"
  }
  // Hummm... should not happen but cannot add error or assert
  // in constexpr expression.
  return StmtGroup::Single;
}

constexpr const char *StmtClassName(StmtClass sc) {
  switch (sc) {
#define DECLARE_STMT(Name, Class, Group, Text) \
  case StmtClass::Name: return #Name;

#include "flang/Sema/Stmt.def"
  }
  return "????";
}

constexpr const char *StmtClassText(StmtClass sc) {
  switch (sc) {
#define DECLARE_STMT(Name, Class, Group, Text) \
  case StmtClass::Name: return Text; 

#include "flang/Sema/Stmt.def"
  }
  return "????";
}

// AnyStmt is a std::variant<> of const pointers to all possible statement
// classes.
// typedef std::variant<
//#define DECLARE_DUMMY_STMT(Name,Group,Text)
//#define DECLARE_PARSER_STMT_ALT(Name,Class,Group,Text)
//#define DECLARE_PARSER_STMT(Name,Class,Group,Text) , const
//Fortran::parser::Class * #define DECLARE_FIRST_STMT(Name,Class,Group,Text)
//const Fortran::parser::Class * #include "flang/Sema/Stmt.def"
//  > AnyStmt ;

}  // end of namespace Fortran::semantics

#endif
