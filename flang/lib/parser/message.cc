#include "message.h"

namespace Fortran {
namespace parser {

Provenance Message::Emit(
    std::ostream &o, const AllSources &sources, bool echoSourceLine) const {
  if (!context_ || context_->Emit(o, sources, false) != provenance_) {
    sources.Identify(o, provenance_, "", echoSourceLine);
  }
  o << "   " << message_ << '\n';
  return provenance_;
}

void Messages::Emit(std::ostream &o) const {
  for (const auto &msg : messages_) {
    if (msg.context()) {
      o << "In the context ";
    }
    msg.Emit(o, allSources_);
  }
}
}  // namespace parser
}  // namespace Fortran
