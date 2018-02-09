#include "message.h"

namespace Fortran {
namespace parser {

void Message::Emit(std::ostream &o, const AllSources &sources) const {
  if (context_) {
    context_->Emit(o, sources);
  }
  sources.Identify(o, provenance_, "");
  o << "   " << message_ << '\n';
}

void Messages::Emit(std::ostream &o, const AllSources &sources) const {
  for (const auto &msg : messages_) {
    msg.Emit(o, sources);
  }
}
}  // namespace parser
}  // namespace Fortran
