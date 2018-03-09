
#include "flang/Sema/Identifier.h"
#include <set>
#include <cassert>
#include <cstring>

using Fortran::semantics::Identifier;

static std::set<std::string> all;

Identifier::Identifier(const std::string &text) 
{
  // TODO: Produce a proper 'ICE' if empty text.
  assert( text.size() > 0 ) ;
  text_ = &(*(all.insert(text).first)) ; 
}

Identifier::Identifier(const char *text) 
{
  // TODO: Produce a proper 'ICE' if text is empty or null.
  assert(text) ;
  assert(text[0] != '\0' ); // cheaper than strlen(text)==0 
  text_ = &(*(all.insert( std::string(text) ).first)) ; 
}


