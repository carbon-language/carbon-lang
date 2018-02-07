#include "message.h"

namespace Fortran {
namespace parser {

std::ostream &operator<<(std::ostream &o, const Message &msg) {
  if (msg.context()) {
    o << *msg.context();
  }
  o << "at line " << msg.position().lineNumber();
  int column = msg.position().column();
  if (column > 0) {
    o << "(column " << column << ")";
  }
  o << ": " << msg.message() << '\n';
  return o;
}

std::ostream &operator<<(std::ostream &o, const Messages &ms) {
  for (const auto &msg : ms) {
    o << msg;
  }
  return o;
}
}  // namespace parser
}  // namespace Fortran
