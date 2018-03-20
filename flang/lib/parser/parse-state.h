#ifndef FORTRAN_PARSER_PARSE_STATE_H_
#define FORTRAN_PARSER_PARSE_STATE_H_

// Defines the ParseState type used as the argument for every parser's
// Parse member or static function.  Tracks source provenance, context,
// accumulated messages, and an arbitrary UserState instance for parsing
// attempts.  Must be efficient to duplicate and assign for backtracking
// and recovery during parsing!

#include "characters.h"
#include "idioms.h"
#include "message.h"
#include "provenance.h"
#include <cstddef>
#include <cstring>
#include <list>
#include <memory>
#include <optional>
#include <utility>

namespace Fortran {
namespace parser {

class UserState;

class ParseState {
public:
  // TODO: Add a constructor for parsing a normalized module file.
  ParseState(const CookedSource &cooked)
    : cooked_{cooked}, p_{&cooked[0]}, limit_{p_ + cooked.size()},
      messages_{*cooked.allSources()} {}
  ParseState(const ParseState &that)
    : cooked_{that.cooked_}, p_{that.p_}, limit_{that.limit_},
      column_{that.column_}, messages_{*that.cooked_.allSources()},
      userState_{that.userState_}, inFixedForm_{that.inFixedForm_},
      encoding_{that.encoding_}, strictConformance_{that.strictConformance_},
      warnOnNonstandardUsage_{that.warnOnNonstandardUsage_},
      warnOnDeprecatedUsage_{that.warnOnDeprecatedUsage_},
      anyErrorRecovery_{that.anyErrorRecovery_},
      anyConformanceViolation_{that.anyConformanceViolation_} {}
  ParseState(ParseState &&that)
    : cooked_{that.cooked_}, p_{that.p_}, limit_{that.limit_},
      column_{that.column_}, messages_{std::move(that.messages_)},
      context_{std::move(that.context_)}, userState_{that.userState_},
      inFixedForm_{that.inFixedForm_}, encoding_{that.encoding_},
      strictConformance_{that.strictConformance_},
      warnOnNonstandardUsage_{that.warnOnNonstandardUsage_},
      warnOnDeprecatedUsage_{that.warnOnDeprecatedUsage_},
      anyErrorRecovery_{that.anyErrorRecovery_},
      anyConformanceViolation_{that.anyConformanceViolation_} {}
  ParseState &operator=(ParseState &&that) {
    swap(that);
    return *this;
  }

  void swap(ParseState &that) {
    constexpr std::size_t bytes{sizeof *this};
    char buffer[bytes];
    std::memcpy(buffer, this, bytes);
    std::memcpy(this, &that, bytes);
    std::memcpy(&that, buffer, bytes);
  }

  const CookedSource &cooked() const { return cooked_; }
  int column() const { return column_; }
  Messages *messages() { return &messages_; }

  bool anyErrorRecovery() const { return anyErrorRecovery_; }
  void set_anyErrorRecovery() { anyErrorRecovery_ = true; }

  bool anyConformanceViolation() const { return anyConformanceViolation_; }
  void set_anyConformanceViolation() { anyConformanceViolation_ = true; }

  UserState *userState() const { return userState_; }
  void set_userState(UserState *u) { userState_ = u; }

  MessageContext context() const { return context_; }
  ParseState &set_context(MessageContext c) {
    context_ = c;
    return *this;
  }

  bool inFixedForm() const { return inFixedForm_; }
  ParseState &set_inFixedForm(bool yes) {
    inFixedForm_ = yes;
    return *this;
  }

  bool strictConformance() const { return strictConformance_; }
  ParseState &set_strictConformance(bool yes) {
    strictConformance_ = yes;
    return *this;
  }

  bool warnOnNonstandardUsage() const { return warnOnNonstandardUsage_; }
  ParseState &set_warnOnNonstandardUsage(bool yes) {
    warnOnNonstandardUsage_ = yes;
    return *this;
  }

  bool warnOnDeprecatedUsage() const { return warnOnDeprecatedUsage_; }
  ParseState &set_warnOnDeprecatedUsage(bool yes) {
    warnOnDeprecatedUsage_ = yes;
    return *this;
  }

  Encoding encoding() const { return encoding_; }
  ParseState &set_encoding(Encoding encoding) {
    encoding_ = encoding;
    return *this;
  }

  const char *GetLocation() const { return p_; }
  Provenance GetProvenance(const char *at) const {
    return cooked_.GetProvenance(at).start();
  }
  Provenance GetProvenance() const { return GetProvenance(p_); }

  MessageContext &PushContext(MessageFixedText text) {
    context_ = std::make_shared<Message>(GetProvenance(), text, context_);
    return context_;
  }

  void PopContext() {
    if (context_) {
      context_ = context_->context();
    }
  }

  Message &PutMessage(MessageFixedText t) { return PutMessage(p_, t); }
  Message &PutMessage(MessageFormattedText &&t) {
    return PutMessage(p_, std::move(t));
  }
  Message &PutMessage(MessageExpectedText &&t) {
    return PutMessage(p_, std::move(t));
  }
  Message &PutMessage(const char *at, MessageFixedText t) {
    return PutMessage(GetProvenance(at), t);
  }
  Message &PutMessage(const char *at, MessageFormattedText &&t) {
    return PutMessage(GetProvenance(at), std::move(t));
  }
  Message &PutMessage(const char *at, MessageExpectedText &&t) {
    return PutMessage(GetProvenance(at), std::move(t));
  }
  Message &PutMessage(Provenance at, MessageFixedText t) {
    return messages_.Put(Message{at, t, context_});
  }
  Message &PutMessage(Provenance at, MessageFormattedText &&t) {
    return messages_.Put(Message{at, std::move(t), context_});
  }
  Message &PutMessage(Provenance at, MessageExpectedText &&t) {
    return messages_.Put(Message{at, std::move(t), context_});
  }

  bool IsAtEnd() const { return p_ >= limit_; }

  char UncheckedAdvance() {
    ++column_;
    char ch{*p_++};
    if (ch == '\n') {
      column_ = 1;
    }
    return ch;
  }

  std::optional<char> GetNextChar() {
    if (p_ >= limit_) {
      return {};
    }
    return {UncheckedAdvance()};
  }

  std::optional<char> PeekAtNextChar() {
    if (p_ >= limit_) {
      return {};
    }
    return {*p_};
  }

private:
  // Text remaining to be parsed
  const CookedSource &cooked_;
  const char *p_{nullptr}, *limit_{nullptr};
  int column_{1};

  // Accumulated messages and current nested context.
  Messages messages_;
  MessageContext context_;

  UserState *userState_{nullptr};

  bool inFixedForm_{false};
  Encoding encoding_{Encoding::UTF8};
  bool strictConformance_{false};
  bool warnOnNonstandardUsage_{false};
  bool warnOnDeprecatedUsage_{false};
  bool anyErrorRecovery_{false};
  bool anyConformanceViolation_{false};
  // NOTE: Any additions or modifications to these data members must also be
  // reflected in the copy and move constructors defined at the top of this
  // class definition!
};
}  // namespace parser
}  // namespace Fortran
#endif  // FORTRAN_PARSER_PARSE_STATE_H_
