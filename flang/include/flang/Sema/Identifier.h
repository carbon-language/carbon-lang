#ifndef FLANG_SEMA_IDENTIFIER_H
#define FLANG_SEMA_IDENTIFIER_H

#include <string>
#include <optional>

namespace Fortran {
namespace semantics {

// A class describing an identifier.
//
// For each name, there is one and only one identifier.
//
// Also, identifiers are immutable and are never destroyed.
//
// The comparison of two 'Identifier*' is expected to return
// true iff their name are identical.
//
class Identifier {
private:
  Identifier(Identifier &&) = delete;
  ~Identifier() = delete;
  Identifier(std::string n) : name_(n) {}

private:
  std::string name_;

public:
  std::string name() const { return name_; }
  static const Identifier *get(std::string n);

  // In the Parse-tree, there are a lot of optional<std::string>
  static const Identifier *get(std::optional<std::string> n) { 
    return n ? Identifier::get(n.value()) : nullptr ;
  } 
};

}  // namespace semantics
}  // namespace Fortran

#endif
