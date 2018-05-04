// Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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

namespace Fortran::parser {

class UserState;

class ParseState {
public:
  // TODO: Add a constructor for parsing a normalized module file.
  ParseState(const CookedSource &cooked)
    : p_{&cooked[0]}, limit_{p_ + cooked.size()} {}
  ParseState(const ParseState &that)
    : p_{that.p_}, limit_{that.limit_}, context_{that.context_},
      userState_{that.userState_}, inFixedForm_{that.inFixedForm_},
      encoding_{that.encoding_}, strictConformance_{that.strictConformance_},
      warnOnNonstandardUsage_{that.warnOnNonstandardUsage_},
      warnOnDeprecatedUsage_{that.warnOnDeprecatedUsage_},
      anyErrorRecovery_{that.anyErrorRecovery_},
      anyConformanceViolation_{that.anyConformanceViolation_},
      deferMessages_{that.deferMessages_} {}
  ParseState(ParseState &&that)
    : p_{that.p_}, limit_{that.limit_}, messages_{std::move(that.messages_)},
      context_{std::move(that.context_)}, userState_{that.userState_},
      inFixedForm_{that.inFixedForm_}, encoding_{that.encoding_},
      strictConformance_{that.strictConformance_},
      warnOnNonstandardUsage_{that.warnOnNonstandardUsage_},
      warnOnDeprecatedUsage_{that.warnOnDeprecatedUsage_},
      anyErrorRecovery_{that.anyErrorRecovery_},
      anyConformanceViolation_{that.anyConformanceViolation_},
      deferMessages_{that.deferMessages_}, anyDeferredMessages_{
                                               that.anyDeferredMessages_} {}
  ParseState &operator=(const ParseState &that) {
    p_ = that.p_, limit_ = that.limit_, context_ = that.context_;
    userState_ = that.userState_, inFixedForm_ = that.inFixedForm_;
    encoding_ = that.encoding_, strictConformance_ = that.strictConformance_;
    warnOnNonstandardUsage_ = that.warnOnNonstandardUsage_;
    warnOnDeprecatedUsage_ = that.warnOnDeprecatedUsage_;
    anyErrorRecovery_ = that.anyErrorRecovery_;
    anyConformanceViolation_ = that.anyConformanceViolation_;
    deferMessages_ = that.deferMessages_;
    anyDeferredMessages_ = that.anyDeferredMessages_;
    return *this;
  }
  ParseState &operator=(ParseState &&that) {
    p_ = that.p_, limit_ = that.limit_, messages_ = std::move(that.messages_);
    context_ = std::move(that.context_);
    userState_ = that.userState_, inFixedForm_ = that.inFixedForm_;
    encoding_ = that.encoding_, strictConformance_ = that.strictConformance_;
    warnOnNonstandardUsage_ = that.warnOnNonstandardUsage_;
    warnOnDeprecatedUsage_ = that.warnOnDeprecatedUsage_;
    anyErrorRecovery_ = that.anyErrorRecovery_;
    anyConformanceViolation_ = that.anyConformanceViolation_;
    deferMessages_ = that.deferMessages_;
    anyDeferredMessages_ = that.anyDeferredMessages_;
    return *this;
  }

  const Messages &messages() const { return messages_; }
  Messages &messages() { return messages_; }

  const Message::Reference &context() const { return context_; }
  Message::Reference &context() { return context_; }

  bool anyErrorRecovery() const { return anyErrorRecovery_; }
  void set_anyErrorRecovery() { anyErrorRecovery_ = true; }

  bool anyConformanceViolation() const { return anyConformanceViolation_; }
  void set_anyConformanceViolation() { anyConformanceViolation_ = true; }

  UserState *userState() const { return userState_; }
  ParseState &set_userState(UserState *u) {
    userState_ = u;
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

  bool deferMessages() const { return deferMessages_; }
  ParseState &set_deferMessages(bool yes) {
    deferMessages_ = yes;
    return *this;
  }

  bool anyDeferredMessages() const { return anyDeferredMessages_; }
  void set_anyDeferredMessages(bool yes) { anyDeferredMessages_ = yes; }

  const char *GetLocation() const { return p_; }

  void PushContext(MessageFixedText text) {
    auto m = new Message{p_, text};  // reference-counted, it's ok
    m->set_context(context_.get());
    context_ = Message::Reference{m};
  }

  void PopContext() {
    if (context_) {
      context_ = context_->context();
    }
  }

  void Say(MessageFixedText t) { return Say(p_, t); }
  void Say(MessageFormattedText &&t) { return Say(p_, std::move(t)); }
  void Say(MessageExpectedText &&t) { return Say(p_, std::move(t)); }

  void Say(CharBlock range, MessageFixedText t) {
    if (deferMessages_) {
      anyDeferredMessages_ = true;
    } else {
      messages_.Say(range, t).set_context(context_.get());
    }
  }
  void Say(CharBlock range, MessageFormattedText &&t) {
    if (deferMessages_) {
      anyDeferredMessages_ = true;
    } else {
      messages_.Say(range, std::move(t)).set_context(context_.get());
    }
  }
  void Say(CharBlock range, MessageExpectedText &&t) {
    if (deferMessages_) {
      anyDeferredMessages_ = true;
    } else {
      messages_.Say(range, std::move(t)).set_context(context_.get());
    }
  }

  bool IsAtEnd() const { return p_ >= limit_; }

  const char *UncheckedAdvance(std::size_t n = 1) {
    const char *result{p_};
    p_ += n;
    return result;
  }

  std::optional<const char *> GetNextChar() {
    if (p_ >= limit_) {
      return {};
    }
    return {UncheckedAdvance()};
  }

  std::optional<const char *> PeekAtNextChar() const {
    if (p_ >= limit_) {
      return {};
    }
    return {p_};
  }

  std::size_t BytesRemaining() const {
    std::size_t remain = limit_ - p_;
    return remain;
  }

private:
  // Text remaining to be parsed
  const char *p_{nullptr}, *limit_{nullptr};

  // Accumulated messages and current nested context.
  Messages messages_;
  Message::Reference context_;

  UserState *userState_{nullptr};

  bool inFixedForm_{false};
  Encoding encoding_{Encoding::UTF8};
  bool strictConformance_{false};
  bool warnOnNonstandardUsage_{false};
  bool warnOnDeprecatedUsage_{false};
  bool anyErrorRecovery_{false};
  bool anyConformanceViolation_{false};
  bool deferMessages_{false};
  bool anyDeferredMessages_{false};
  // NOTE: Any additions or modifications to these data members must also be
  // reflected in the copy and move constructors defined at the top of this
  // class definition!
};

}  // namespace Fortran::parser
#endif  // FORTRAN_PARSER_PARSE_STATE_H_
