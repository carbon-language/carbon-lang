
#include "flang/Sema/Identifier.h"

#include <map>

using Fortran::semantics::Identifier ;

static std::map<std::string, Identifier*> all ;

const Identifier *
Identifier::get(std::string n) 
{
  Identifier * &ref = all[n] ;
  if (!ref) {
    ref = new Identifier(n) ;
  }
  return ref ;
}
