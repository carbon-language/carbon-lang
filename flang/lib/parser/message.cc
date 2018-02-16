#include "message.h"

namespace Fortran {
namespace parser {

std::ostream &operator<<(std::ostream &o, const MessageText &t) {
  for (size_t j{0}; j < t.size(); ++j) {
    o << t.str()[j];
  }
  return o;
}

Provenance Message::Emit(
    std::ostream &o, const AllSources &sources, bool echoSourceLine) const {
  if (!context_ || context_->Emit(o, sources, false) != provenance_) {
    sources.Identify(o, provenance_, "", echoSourceLine);
  }
  o << "   " << text_ << message_ << '\n';
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
