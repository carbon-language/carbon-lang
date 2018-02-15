#ifndef FORTRAN_PARSE_STATE_H_
#define FORTRAN_PARSE_STATE_H_

// Defines the ParseState type used as the argument for every parser's
// Parse member or static function.  Tracks source provenance, context,
// accumulated messages, and an arbitrary UserState instance for parsing
// attempts.  Must be efficient to duplicate and assign for backtracking
// and recovery during parsing!

#include "idioms.h"
#include "message.h"
#include "provenance.h"
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
      userState_{that.userState_}, inCharLiteral_{that.inCharLiteral_},
      inFortran_{that.inFortran_}, inFixedForm_{that.inFixedForm_},
      enableOldDebugLines_{that.enableOldDebugLines_}, columns_{that.columns_},
      enableBackslashEscapesInCharLiterals_{
          that.enableBackslashEscapesInCharLiterals_},
      strictConformance_{that.strictConformance_},
      warnOnNonstandardUsage_{that.warnOnNonstandardUsage_},
      warnOnDeprecatedUsage_{that.warnOnDeprecatedUsage_},
      skippedNewLines_{that.skippedNewLines_},
      tabInCurrentLine_{that.tabInCurrentLine_},
      anyErrorRecovery_{that.anyErrorRecovery_}, prescanned_{that.prescanned_} {
  }
  ParseState(ParseState &&that)
    : cooked_{that.cooked_}, p_{that.p_}, limit_{that.limit_},
      column_{that.column_}, messages_{std::move(that.messages_)},
      context_{std::move(that.context_)}, userState_{that.userState_},
      inCharLiteral_{that.inCharLiteral_}, inFortran_{that.inFortran_},
      inFixedForm_{that.inFixedForm_},
      enableOldDebugLines_{that.enableOldDebugLines_}, columns_{that.columns_},
      enableBackslashEscapesInCharLiterals_{
          that.enableBackslashEscapesInCharLiterals_},
      strictConformance_{that.strictConformance_},
      warnOnNonstandardUsage_{that.warnOnNonstandardUsage_},
      warnOnDeprecatedUsage_{that.warnOnDeprecatedUsage_},
      skippedNewLines_{that.skippedNewLines_},
      tabInCurrentLine_{that.tabInCurrentLine_},
      anyErrorRecovery_{that.anyErrorRecovery_}, prescanned_{that.prescanned_} {
  }
  ParseState &operator=(ParseState &&that) {
    swap(that);
    return *this;
  }

  void swap(ParseState &that) {
    constexpr size_t bytes{sizeof *this};
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

  UserState *userState() const { return userState_; }
  void set_userState(UserState *u) { userState_ = u; }

  MessageContext context() const { return context_; }
  ParseState &set_context(MessageContext c) {
    context_ = c;
    return *this;
  }

  bool inCharLiteral() const { return inCharLiteral_; }
  ParseState &set_inCharLiteral(bool yes) {
    inCharLiteral_ = yes;
    return *this;
  }

  bool inFortran() const { return inFortran_; }
  ParseState &set_inFortran(bool yes) {
    inFortran_ = yes;
    return *this;
  }

  bool inFixedForm() const { return inFixedForm_; }
  ParseState &set_inFixedForm(bool yes) {
    inFixedForm_ = yes;
    return *this;
  }

  bool enableOldDebugLines() const { return enableOldDebugLines_; }
  ParseState &set_enableOldDebugLines(bool yes) {
    enableOldDebugLines_ = yes;
    return *this;
  }

  int columns() const { return columns_; }
  ParseState &set_columns(int cols) {
    columns_ = cols;
    return *this;
  }

  bool enableBackslashEscapesInCharLiterals() const {
    return enableBackslashEscapesInCharLiterals_;
  }
  ParseState &set_enableBackslashEscapesInCharLiterals(bool yes) {
    enableBackslashEscapesInCharLiterals_ = yes;
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

  int skippedNewLines() const { return skippedNewLines_; }
  void set_skippedNewLines(int n) { skippedNewLines_ = n; }

  bool prescanned() const { return prescanned_; }  // TODO: always true, remove

  bool tabInCurrentLine() const { return tabInCurrentLine_; }

  const char *GetLocation() const { return p_; }
  Provenance GetProvenance(const char *at) const {
    return cooked_.GetProvenance(at).LocalOffsetToProvenance(0);
  }
  Provenance GetProvenance() const { return GetProvenance(p_); }

  void PushContext(const std::string &str) {
    context_ = std::make_shared<Message>(GetProvenance(), str, context_);
  }
  void PushContext(std::string &&str) {
    context_ =
        std::make_shared<Message>(GetProvenance(), std::move(str), context_);
  }
  void PushContext(const char *str) {
    context_ = std::make_shared<Message>(GetProvenance(), str, context_);
  }

  void PopContext() {
    if (context_) {
      context_ = context_->context();
    }
  }

  void PutMessage(Provenance at, const std::string &msg) {
    messages_.Put(Message{at, msg, context_});
  }
  void PutMessage(const char *at, const std::string &msg) {
    PutMessage(GetProvenance(at), msg);
  }
  void PutMessage(const std::string &msg) { PutMessage(p_, msg); }
  void PutMessage(Provenance at, std::string &&msg) {
    messages_.Put(Message{at, std::move(msg), context_});
  }
  void PutMessage(const char *at, std::string &&msg) {
    PutMessage(GetProvenance(at), std::move(msg));
  }
  void PutMessage(std::string &&msg) { PutMessage(p_, std::move(msg)); }
  void PutMessage(Provenance at, const char *msg) {
    PutMessage(at, std::string{msg});
  }
  void PutMessage(const char *at, const char *msg) {
    PutMessage(GetProvenance(at), msg);
  }
  void PutMessage(const char *msg) { PutMessage(p_, msg); }

  bool IsAtEnd() const { return p_ >= limit_; }

  std::optional<char> GetNextRawChar() const {
    if (p_ < limit_) {
      return {*p_};
    }
    return {};
  }

  void Advance() {
    CHECK(p_ < limit_);
    if (*p_ == '\n') {
      column_ = 1;
      tabInCurrentLine_ = false;
    } else if (*p_ == '\t') {
      column_ = ((column_ + 7) & -8) + 1;
      tabInCurrentLine_ = true;
    } else {
      ++column_;
    }
    ++p_;
  }

  void AdvanceColumnForPadding() { ++column_; }

private:
  // Text remaining to be parsed
  const CookedSource &cooked_;
  const char *p_{nullptr}, *limit_{nullptr};
  int column_{1};

  // Accumulated messages and current nested context.
  Messages messages_;
  MessageContext context_;

  UserState *userState_{nullptr};

  bool inCharLiteral_{false};
  bool inFortran_{true};
  bool inFixedForm_{false};
  bool enableOldDebugLines_{false};
  int columns_{72};
  bool enableBackslashEscapesInCharLiterals_{true};
  bool strictConformance_{false};
  bool warnOnNonstandardUsage_{false};
  bool warnOnDeprecatedUsage_{false};
  int skippedNewLines_{0};
  bool tabInCurrentLine_{false};
  bool anyErrorRecovery_{false};
  bool prescanned_{true};
  // NOTE: Any additions or modifications to these data members must also be
  // reflected in the copy and move constructors defined at the top of this
  // class definition!
};
}  // namespace parser
}  // namespace Fortran
#endif  // FORTRAN_PARSE_STATE_H_
