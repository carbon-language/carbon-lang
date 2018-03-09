

#include "flang/Sema/Symbol.h"
#include "flang/Sema/Identifier.h"
#include "flang/Sema/Scope.h"

#include <cassert>

namespace Fortran::semantics {

Symbol::Symbol(ClassId cid, Scope *owner, Identifier ident)
  : cid_(cid), owner_(owner), ident_(ident) {
  owner->add(this);
}

}  // namespace Fortran::semantics
