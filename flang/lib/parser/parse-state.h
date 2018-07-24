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
#include "features.h"
#include "message.h"
#include "provenance.h"
#include "user-state.h"
#include "../common/idioms.h"
#include <cstddef>
#include <cstring>
#include <list>
#include <memory>
#include <optional>
#include <utility>

namespace Fortran::parser {

class ParseState {
public:
  // TODO: Add a constructor for parsing a normalized module file.
  ParseState(const CookedSource &cooked)
    : p_{&cooked[0]}, limit_{p_ + cooked.size()} {}
  ParseState(const ParseState &that)
    : p_{that.p_}, limit_{that.limit_}, context_{that.context_},
      userState_{that.userState_}, inFixedForm_{that.inFixedForm_},
      encoding_{that.encoding_}, anyErrorRecovery_{that.anyErrorRecovery_},
      anyConformanceViolation_{that.anyConformanceViolation_},
      deferMessages_{that.deferMessages_},
      anyDeferredMessages_{that.anyDeferredMessages_},
      tokensMatched_{that.tokensMatched_} {}
  ParseState(ParseState &&that)
    : p_{that.p_}, limit_{that.limit_}, messages_{std::move(that.messages_)},
      context_{std::move(that.context_)}, userState_{that.userState_},
      inFixedForm_{that.inFixedForm_}, encoding_{that.encoding_},
      anyErrorRecovery_{that.anyErrorRecovery_},
      anyConformanceViolation_{that.anyConformanceViolation_},
      deferMessages_{that.deferMessages_},
      anyDeferredMessages_{that.anyDeferredMessages_},
      tokensMatched_{that.tokensMatched_} {}
  ParseState &operator=(const ParseState &that) {
    p_ = that.p_, limit_ = that.limit_, context_ = that.context_;
    userState_ = that.userState_, inFixedForm_ = that.inFixedForm_;
    encoding_ = that.encoding_;
    anyErrorRecovery_ = that.anyErrorRecovery_;
    anyConformanceViolation_ = that.anyConformanceViolation_;
    deferMessages_ = that.deferMessages_;
    anyDeferredMessages_ = that.anyDeferredMessages_;
    tokensMatched_ = that.tokensMatched_;
    return *this;
  }
  ParseState &operator=(ParseState &&that) {
    p_ = that.p_, limit_ = that.limit_, messages_ = std::move(that.messages_);
    context_ = std::move(that.context_);
    userState_ = that.userState_, inFixedForm_ = that.inFixedForm_;
    encoding_ = that.encoding_;
    anyErrorRecovery_ = that.anyErrorRecovery_;
    anyConformanceViolation_ = that.anyConformanceViolation_;
    deferMessages_ = that.deferMessages_;
    anyDeferredMessages_ = that.anyDeferredMessages_;
    tokensMatched_ = that.tokensMatched_;
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
  ParseState &set_inFixedForm(bool yes = true) {
    inFixedForm_ = yes;
    return *this;
  }

  Encoding encoding() const { return encoding_; }
  ParseState &set_encoding(Encoding encoding) {
    encoding_ = encoding;
    return *this;
  }

  bool deferMessages() const { return deferMessages_; }
  ParseState &set_deferMessages(bool yes = true) {
    deferMessages_ = yes;
    return *this;
  }

  bool anyDeferredMessages() const { return anyDeferredMessages_; }
  ParseState &set_anyDeferredMessages(bool yes = true) {
    anyDeferredMessages_ = yes;
    return *this;
  }

  std::size_t tokensMatched() const { return tokensMatched_; }
  ParseState &set_tokensMatched(std::size_t n) {
    tokensMatched_ = n;
    return *this;
  }
  ParseState &TokenMatched() {
    ++tokensMatched_;
    return *this;
  }

  const char *GetLocation() const { return p_; }

  void PushContext(MessageFixedText text) {
    auto m{new Message{p_, text}};  // reference-counted
    m->SetContext(context_.get());
    context_ = Message::Reference{m};
  }

  void PopContext() {
    CHECK(context_);
    context_ = context_->attachment();
  }

  void Say(const MessageFixedText &t) { Say(p_, t); }
  void Say(MessageFormattedText &&t) { Say(p_, std::move(t)); }
  void Say(const MessageExpectedText &t) { Say(p_, t); }

  void Say(CharBlock range, const MessageFixedText &t) {
    if (deferMessages_) {
      anyDeferredMessages_ = true;
    } else {
      messages_.Say(range, t).SetContext(context_.get());
    }
  }
  void Say(CharBlock range, MessageFormattedText &&t) {
    if (deferMessages_) {
      anyDeferredMessages_ = true;
    } else {
      messages_.Say(range, std::move(t)).SetContext(context_.get());
    }
  }
  void Say(CharBlock range, const MessageExpectedText &t) {
    if (deferMessages_) {
      anyDeferredMessages_ = true;
    } else {
      messages_.Say(range, t).SetContext(context_.get());
    }
  }

  void Nonstandard(LanguageFeature lf, const MessageFixedText &msg) {
    Nonstandard(p_, lf, msg);
  }
  void Nonstandard(
      CharBlock range, LanguageFeature lf, const MessageFixedText &msg) {
    anyConformanceViolation_ = true;
    if (userState_ != nullptr && userState_->features().ShouldWarn(lf)) {
      Say(range, msg);
    }
  }
  bool IsNonstandardOk(LanguageFeature lf, const MessageFixedText &msg) {
    if (userState_ != nullptr && !userState_->features().IsEnabled(lf)) {
      return false;
    }
    Nonstandard(lf, msg);
    return true;
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

  void CombineFailedParses(ParseState &prev, std::size_t origTokensMatched) {
    if (prev.tokensMatched_ > origTokensMatched) {
      if (tokensMatched_ > origTokensMatched) {
        if (prev.p_ == p_) {
          prev.messages_.Incorporate(messages_);
          prev.anyDeferredMessages_ |= anyDeferredMessages_;
        }
        if (prev.p_ >= p_) {
          *this = std::move(prev);
        }
      } else {
        *this = std::move(prev);
      }
    }
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
  bool anyErrorRecovery_{false};
  bool anyConformanceViolation_{false};
  bool deferMessages_{false};
  bool anyDeferredMessages_{false};
  std::size_t tokensMatched_{0};
  // NOTE: Any additions or modifications to these data members must also be
  // reflected in the copy and move constructors defined at the top of this
  // class definition!
};

}  // namespace Fortran::parser
#endif  // FORTRAN_PARSER_PARSE_STATE_H_
