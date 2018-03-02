

#include "flang/Sema/Symbol.h"
#include "flang/Sema/Scope.h"
#include "flang/Sema/Identifier.h"

#include <cassert>

namespace Fortran::semantics {


Symbol::Symbol(ClassId cid, Scope * owner, const Identifier *name) :
  cid_(cid), 
  owner_(owner), 
  name_(name) 
{
  owner->add(this); 
} 


} // of namespace
