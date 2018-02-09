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
  ParseState(const CookedSource &cooked)
    : cooked_{cooked}, p_{&cooked[0]}, remaining_{cooked.size()} {}
  ParseState(const ParseState &that)
    : cooked_{that.cooked_}, p_{that.p_},
      remaining_{that.remaining_}, column_{that.column_},
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
    : cooked_{that.cooked_}, p_{that.p_},
      remaining_{that.remaining_}, column_{that.column_},
      messages_{std::move(that.messages_)}, context_{std::move(that.context_)},
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

  bool anyErrorRecovery() const { return anyErrorRecovery_; }
  void set_anyErrorRecovery() { anyErrorRecovery_ = true; }

  UserState *userState() const { return userState_; }
  void set_userState(UserState *u) { userState_ = u; }

  int column() const { return column_; }
  Messages *messages() { return &messages_; }

  MessageContext context() const { return context_; }
  MessageContext set_context(MessageContext c) {
    MessageContext was{context_};
    context_ = c;
    return was;
  }

  bool inCharLiteral() const { return inCharLiteral_; }
  bool set_inCharLiteral(bool yes) {
    bool was{inCharLiteral_};
    inCharLiteral_ = yes;
    return was;
  }

  bool inFortran() const { return inFortran_; }
  bool set_inFortran(bool yes) {
    bool was{inFortran_};
    inFortran_ = yes;
    return was;
  }

  bool inFixedForm() const { return inFixedForm_; }
  bool set_inFixedForm(bool yes) {
    bool was{inFixedForm_};
    inFixedForm_ = yes;
    return was;
  }

  bool enableOldDebugLines() const { return enableOldDebugLines_; }
  bool set_enableOldDebugLines(bool yes) {
    bool was{enableOldDebugLines_};
    enableOldDebugLines_ = yes;
    return was;
  }

  int columns() const { return columns_; }
  int set_columns(int cols) {
    int was{columns_};
    columns_ = cols;
    return was;
  }

  bool enableBackslashEscapesInCharLiterals() const {
    return enableBackslashEscapesInCharLiterals_;
  }
  bool set_enableBackslashEscapesInCharLiterals(bool yes) {
    bool was{enableBackslashEscapesInCharLiterals_};
    enableBackslashEscapesInCharLiterals_ = yes;
    return was;
  }

  bool strictConformance() const { return strictConformance_; }
  bool set_strictConformance(bool yes) {
    bool was{strictConformance_};
    strictConformance_ = yes;
    return was;
  }

  bool warnOnNonstandardUsage() const { return warnOnNonstandardUsage_; }
  bool set_warnOnNonstandardUsage(bool yes) {
    bool was{warnOnNonstandardUsage_};
    warnOnNonstandardUsage_ = yes;
    return was;
  }

  bool warnOnDeprecatedUsage() const { return warnOnDeprecatedUsage_; }
  bool set_warnOnDeprecatedUsage(bool yes) {
    bool was{warnOnDeprecatedUsage_};
    warnOnDeprecatedUsage_ = yes;
    return was;
  }

  int skippedNewLines() const { return skippedNewLines_; }
  void set_skippedNewLines(int n) { skippedNewLines_ = n; }

  bool prescanned() const { return prescanned_; }  // TODO: always true, remove

  bool tabInCurrentLine() const { return tabInCurrentLine_; }

  const AllSources &GetAllSources() const { return cooked_.sources(); }
  const char *GetLocation() const { return p_; }
  Provenance GetProvenance(const char *at) const {
    return cooked_.GetProvenance(at).start;
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

  bool IsAtEnd() const { return remaining_ == 0; }

  std::optional<char> GetNextRawChar() const {
    if (remaining_ > 0) {
      return {*p_};
    }
    return {};
  }

  void Advance() {
    CHECK(remaining_ > 0);
    --remaining_;
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
  const char *p_{nullptr};
  size_t remaining_{0};
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
