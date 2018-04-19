#include "message.h"
#include "char-set.h"
#include <cstdarg>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <string>

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

MessageFormattedText::MessageFormattedText(MessageFixedText text, ...)
  : isFatal_{text.isFatal()} {
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

void Message::Incorporate(Message &that) {
  if (provenance_ == that.provenance_ &&
      cookedSourceLocation_ == that.cookedSourceLocation_ &&
      !expected_.empty()) {
    expected_ = expected_.Union(that.expected_);
  }
}

std::string Message::ToString() const {
  std::string s{string_};
  bool isExpected{isExpected_};
  if (string_.empty()) {
    if (fixedText_ != nullptr) {
      if (fixedBytes_ > 0 && fixedBytes_ < std::string::npos) {
        s = std::string(fixedText_, fixedBytes_);
      } else {
        s = std::string{fixedText_};  // NUL-terminated
      }
    } else {
      SetOfChars expect{expected_};
      if (expect.Has('\n')) {
        expect = expect.Difference('\n');
        if (expect.empty()) {
          return "expected end of line"_err_en_US.ToString();
        } else {
          s = expect.ToString();
          if (s.size() == 1) {
            return MessageFormattedText(
                "expected end of line or '%s'"_err_en_US, s.data())
                .MoveString();
          } else {
            return MessageFormattedText(
                "expected end of line or one of '%s'"_err_en_US, s.data())
                .MoveString();
          }
        }
      }
      s = expect.ToString();
      if (s.size() != 1) {
        return MessageFormattedText("expected one of '%s'"_err_en_US, s.data())
            .MoveString();
      }
      isExpected = true;
    }
  }
  if (isExpected) {
    return MessageFormattedText("expected '%s'"_err_en_US, s.data())
        .MoveString();
  }
  return s;
}

Provenance Message::Emit(
    std::ostream &o, const CookedSource &cooked, bool echoSourceLine) const {
  Provenance provenance{provenance_};
  if (cookedSourceLocation_ != nullptr) {
    provenance = cooked.GetProvenance(cookedSourceLocation_).start();
  }
  if (!context_ || context_->Emit(o, cooked, false) != provenance) {
    cooked.allSources().Identify(o, provenance, "", echoSourceLine);
  }
  o << "   ";
  if (isFatal_) {
    o << "ERROR: ";
  }
  o << ToString() << '\n';
  return provenance;
}

void Messages::Incorporate(Messages &that) {
  if (messages_.empty()) {
    *this = std::move(that);
  } else if (!that.messages_.empty()) {
    last_->Incorporate(*that.last_);
  }
}

void Messages::Emit(
    std::ostream &o, const char *prefix, bool echoSourceLines) const {
  for (const auto &msg : messages_) {
    if (prefix) {
      o << prefix;
    }
    if (msg.context()) {
      o << "In the context ";
    }
    msg.Emit(o, cooked_, echoSourceLines);
  }
}

bool Messages::AnyFatalError() const {
  for (const auto &msg : messages_) {
    if (msg.isFatal()) {
      return true;
    }
  }
  return false;
}
}  // namespace parser
}  // namespace Fortran
