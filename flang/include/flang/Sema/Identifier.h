#ifndef FLANG_SEMA_IDENTIFIER_H
#define FLANG_SEMA_IDENTIFIER_H

#include <string>

namespace Fortran {
namespace semantics {


// A class describing an identifier.
// 
// For each name, there is one and only one identifier. 
//
// Also, identifiers are immutable and are never destroyed.
// 
// The comparison of two 'Identifier*' returns true iff their 
// name are identical.
//
class Identifier 
{
private:
  Identifier(Identifier &&) = delete ;
  ~Identifier() = delete ;
  Identifier(std::string n) : name_(n) { }
private:
  std::string name_ ;
public:
  const std::string & name() { return name_ ; }
  static const Identifier *get(std::string n) ;     
}; 

} // of namespace flang::Sema
} // of namespace flang

#endif
