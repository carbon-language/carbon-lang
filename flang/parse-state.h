#ifndef FORTRAN_PARSE_STATE_H_
#define FORTRAN_PARSE_STATE_H_

// Defines the ParseState type used as the argument for every parser's
// Parse member or static function.  Tracks position, context, accumulated
// messages, and an arbitrary UserState instance for parsing attempts.
// Must be efficient to duplicate and assign for backtracking and recovery
// during parsing!

#include "idioms.h"
#include "message.h"
#include "position.h"
#include <cstring>
#include <list>
#include <memory>
#include <optional>
#include <utility>

namespace Fortran {

class UserState;

class ParseState {
public:
  ParseState() {}
  ParseState(const char *str) : p_{str}, remaining_{std::strlen(str)} {}
  ParseState(const char *str, size_t bytes) : p_{str}, remaining_{bytes} {}
  ParseState(const ParseState &that)
    : p_{that.p_}, remaining_{that.remaining_}, position_{that.position_},
      userState_{that.userState_},
      inCharLiteral_{that.inCharLiteral_}, inFortran_{that.inFortran_},
      inFixedForm_{that.inFixedForm_},
      enableOldDebugLines_{that.enableOldDebugLines_}, columns_{that.columns_},
      enableBackslashEscapesInCharLiterals_
        {that.enableBackslashEscapesInCharLiterals_},
      strictConformance_{that.strictConformance_},
      warnOnNonstandardUsage_{that.warnOnNonstandardUsage_},
      warnOnDeprecatedUsage_{that.warnOnDeprecatedUsage_},
      skippedNewLines_{that.skippedNewLines_},
      tabInCurrentLine_{that.tabInCurrentLine_},
      anyErrorRecovery_{that.anyErrorRecovery_},
      prescanned_{that.prescanned_} {}
  ParseState(ParseState &&that)
    : p_{that.p_}, remaining_{that.remaining_}, position_{that.position_},
      messages_{std::move(that.messages_)}, context_{std::move(that.context_)},
      userState_{that.userState_},
      inCharLiteral_{that.inCharLiteral_}, inFortran_{that.inFortran_},
      inFixedForm_{that.inFixedForm_},
      enableOldDebugLines_{that.enableOldDebugLines_}, columns_{that.columns_},
      enableBackslashEscapesInCharLiterals_
        {that.enableBackslashEscapesInCharLiterals_},
      strictConformance_{that.strictConformance_},
      warnOnNonstandardUsage_{that.warnOnNonstandardUsage_},
      warnOnDeprecatedUsage_{that.warnOnDeprecatedUsage_},
      skippedNewLines_{that.skippedNewLines_},
      tabInCurrentLine_{that.tabInCurrentLine_},
      anyErrorRecovery_{that.anyErrorRecovery_},
      prescanned_{that.prescanned_} {}
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

  Position position() const { return position_; }

  bool anyErrorRecovery() const { return anyErrorRecovery_; }
  void set_anyErrorRecovery() { anyErrorRecovery_ = true; }

  UserState *userState() const { return userState_; }
  void set_userState(UserState *u) { userState_ = u; }

  Messages *messages() { return &messages_; }

  MessageContext context() const { return context_; }
  MessageContext set_context(MessageContext c) {
    MessageContext was{context_};
    context_ = c;
    return was;
  }

  void PushContext(const std::string &str) {
    context_ = std::make_shared<Message>(position_, str, context_);
  }
  void PushContext(std::string &&str) {
    context_ = std::make_shared<Message>(position_, std::move(str), context_);
  }
  void PushContext(const char *str) {
    context_ = std::make_shared<Message>(position_, str, context_);
  }

  void PopContext() {
    if (context_) {
      context_ = context_->context();
    }
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

  bool prescanned() const { return prescanned_; }
  void set_prescanned(bool yes) { prescanned_ = yes; }

  bool tabInCurrentLine() const { return tabInCurrentLine_; }

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
      position_.AdvanceLine();
      tabInCurrentLine_ = false;
    } else if (*p_ == '\t') {
      position_.TabAdvanceColumn();
      tabInCurrentLine_ = true;
    } else {
      position_.AdvanceColumn();
    }
    ++p_;
  }

  void AdvancePositionForPadding() {
    position_.AdvanceColumn();
  }

private:
  // Text remaining to be parsed
  const char *p_{nullptr};
  size_t remaining_{0};
  Position position_;

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
  bool prescanned_{false};
  // NOTE: Any additions or modifications to these data members must also be
  // reflected in the copy and move constructors defined at the top of this
  // class definition!
};
}  // namespace Fortran
#endif  // FORTRAN_PARSE_STATE_H_
