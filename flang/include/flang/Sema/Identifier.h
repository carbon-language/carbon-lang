#ifndef FLANG_SEMA_IDENTIFIER_H
#define FLANG_SEMA_IDENTIFIER_H

#include <string>
#include <optional>

namespace Fortran::semantics {

  
// A class describing an identifier.
//
// All Identifiers associated to the same name are sharing the same
// storage for that name. That makes comparison of 'Identifier' 
// very fast since the strings do not have to be compared.
//

class Identifier; 

typedef std::optional<Identifier> OptIdentifier ;

class Identifier 
{
private:
  const std::string *text_ ;
public:
  Identifier() = delete ; 

  Identifier(const Identifier &in) : text_(in.text_) {}  

  Identifier(const std::string &); 

  // Construct an Identifier from a constant C-string.
  // 
  // That constructor is not entirely redundant with the std::string 
  // constructor because it serves a secundary purpose: catch the illegal 
  // use of 0 to initialize an Identifier.
  //
  // Without it, the std::string constructor would be called (since 0 
  // is somehow a legal std::string initializer). The result would be an 
  // 'std::logic_error' exception.
  // 
  Identifier(const char *); 

  
  // Helper to create 
  static Identifier make(const std::string &text) { return Identifier(text); }   

  // Helper to create OptIdentifier
  static std::optional<Identifier> make(const std::optional<std::string> &text) {
    if (text) {
      return Identifier(*text);
    } else {
      return std::nullopt;
    }
  }
    
  
  bool operator==(const Identifier &a) const {
    return text_ == a.text_ ;
  }

  bool operator!=(const Identifier &a) const {
    return text_ != a.text_ ;
  }

  bool operator<(const Identifier &a) const {
    return name() < a.name() ;
  }

  bool operator<=(const Identifier &a) const {
    return name() <= a.name() ;
  }

  bool operator>(const Identifier &a) const {
    return name() > a.name() ;
  }

  bool operator>=(const Identifier &a) const {
    return name() >= a.name() ;
  }

  const std::string & name() const {
    return *text_ ;
  }

};


}  // namespace Fortran::semantics

#endif // of FLANG_SEMA_IDENTIFIER_H
