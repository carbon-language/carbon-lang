#include "message.h"
#include <cstdarg>
#include <cstddef>
#include <cstdio>
#include <cstring>

namespace Fortran {
namespace parser {

std::ostream &operator<<(std::ostream &o, const MessageFixedText &t) {
  for (std::size_t j{0}; j < t.size(); ++j) {
    o << t.str()[j];
  }
  return o;
}

std::string MessageFixedText::ToString() const {
  return std::string{str_, /*not in std::*/ strnlen(str_, bytes_)};
}

MessageFormattedText::MessageFormattedText(MessageFixedText text, ...) {
  const char *p{text.str()};
  std::string asString;
  if (p[text.size()] != '\0') {
    // not NUL-terminated
    asString = text.ToString();
    p = asString.data();
  }
  char buffer[256];
  va_list ap;
  va_start(ap, text);
  vsnprintf(buffer, sizeof buffer, p, ap);
  va_end(ap);
  string_ = buffer;
}

MessageFixedText MessageExpectedText::AsMessageFixedText() const {
  if (str_ != nullptr) {
    return {str_, bytes_};
  }
  static char chars[256];
  if (chars[1] == '\0') {
    // one-time initialization of array used for permanant single-byte string
    // pointers
    for (std::size_t j{0}; j < sizeof chars; ++j) {
      chars[j] = j;
    }
  }
  return {&chars[static_cast<unsigned char>(singleton_)], 1};
}

Provenance Message::Emit(
    std::ostream &o, const AllSources &sources, bool echoSourceLine) const {
  if (!context_ || context_->Emit(o, sources, false) != provenance_) {
    sources.Identify(o, provenance_, "", echoSourceLine);
  }
  o << "   ";
  if (string_.empty()) {
    if (isExpectedText_) {
      std::string goal{text_.ToString()};
      if (goal == "\n") {
        o << "expected end of line"_en_US;
      } else {
        o << MessageFormattedText("expected '%s'"_en_US, goal.data())
                 .MoveString();
      }
    } else {
      o << text_;
    }
  } else {
    o << string_;
  }
  o << '\n';
  return provenance_;
}

void Messages::Emit(std::ostream &o, const char *prefix) const {
  for (const auto &msg : messages_) {
    if (prefix) {
      o << prefix;
    }
    if (msg.context()) {
      o << "In the context ";
    }
    msg.Emit(o, allSources_);
  }
}
}  // namespace parser
}  // namespace Fortran
