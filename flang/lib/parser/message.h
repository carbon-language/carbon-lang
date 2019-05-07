// Copyright (c) 2018-2019, NVIDIA CORPORATION.  All rights reserved.
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

#ifndef FORTRAN_PARSER_MESSAGE_H_
#define FORTRAN_PARSER_MESSAGE_H_

// Defines a representation for sequences of compiler messages.
// Supports nested contextualization.

#include "char-block.h"
#include "char-set.h"
#include "provenance.h"
#include "../common/idioms.h"
#include "../common/reference-counted.h"
#include "../common/restorer.h"
#include <cstddef>
#include <cstring>
#include <forward_list>
#include <optional>
#include <ostream>
#include <string>
#include <utility>
#include <variant>

namespace Fortran::parser {

// Use "..."_err_en_US and "..."_en_US literals to define the static
// text and fatality of a message.
class MessageFixedText {
public:
  constexpr MessageFixedText(
      const char str[], std::size_t n, bool isFatal = false)
    : text_{str, n}, isFatal_{isFatal} {}
  constexpr MessageFixedText(const MessageFixedText &) = default;
  constexpr MessageFixedText(MessageFixedText &&) = default;
  constexpr MessageFixedText &operator=(const MessageFixedText &) = default;
  constexpr MessageFixedText &operator=(MessageFixedText &&) = default;

  const CharBlock &text() const { return text_; }
  bool isFatal() const { return isFatal_; }

private:
  CharBlock text_;
  bool isFatal_{false};
};

inline namespace literals {
constexpr MessageFixedText operator""_en_US(const char str[], std::size_t n) {
  return MessageFixedText{str, n, false /* not fatal */};
}

constexpr MessageFixedText operator""_err_en_US(
    const char str[], std::size_t n) {
  return MessageFixedText{str, n, true /* fatal */};
}
}

// The construction of a MessageFormattedText uses a MessageFixedText
// as a vsnprintf() formatting string that is applied to the
// following arguments.  CharBlock and std::string argument
// values are also supported; they are automatically converted into
// char pointers that are suitable for '%s' formatting.
class MessageFormattedText {
public:
  template<typename... A>
  MessageFormattedText(const MessageFixedText &text, A &&... x)
    : isFatal_{text.isFatal()} {
    Format(&text, Convert(std::forward<A>(x))...);
  }
  MessageFormattedText(const MessageFormattedText &) = default;
  MessageFormattedText(MessageFormattedText &&) = default;
  MessageFormattedText &operator=(const MessageFormattedText &) = default;
  MessageFormattedText &operator=(MessageFormattedText &&) = default;
  const std::string &string() const { return string_; }
  bool isFatal() const { return isFatal_; }
  std::string MoveString() { return std::move(string_); }

private:
  void Format(const MessageFixedText *text, ...);

  template<typename A> A Convert(const A &x) {
    static_assert(!std::is_class_v<std::decay_t<A>>);
    return x;
  }
  template<typename A> A Convert(A &x) {
    static_assert(!std::is_class_v<std::decay_t<A>>);
    return x;
  }
  template<typename A> common::IfNoLvalue<A, A> Convert(A &&x) {
    static_assert(!std::is_class_v<std::decay_t<A>>);
    return std::move(x);
  }
  const char *Convert(const std::string &);
  const char *Convert(std::string &);
  const char *Convert(std::string &&);
  const char *Convert(const CharBlock &);
  const char *Convert(CharBlock &);
  const char *Convert(CharBlock &&);

  bool isFatal_{false};
  std::string string_;
  std::forward_list<std::string> conversions_;  // preserves created strings
};

// Represents a formatted rendition of "expected '%s'"_err_en_US
// on a constant text or a set of characters.
class MessageExpectedText {
public:
  MessageExpectedText(const char *s, std::size_t n) {
    if (n == std::string::npos) {
      n = std::strlen(s);
    }
    if (n == 1) {
      // Treat a one-character string as a singleton set for better merging.
      u_ = SetOfChars{*s};
    } else {
      u_ = CharBlock{s, n};
    }
  }
  constexpr explicit MessageExpectedText(CharBlock cb) : u_{cb} {}
  constexpr explicit MessageExpectedText(char ch) : u_{SetOfChars{ch}} {}
  constexpr explicit MessageExpectedText(SetOfChars set) : u_{set} {}
  MessageExpectedText(const MessageExpectedText &) = default;
  MessageExpectedText(MessageExpectedText &&) = default;
  MessageExpectedText &operator=(const MessageExpectedText &) = default;
  MessageExpectedText &operator=(MessageExpectedText &&) = default;

  std::string ToString() const;
  bool Merge(const MessageExpectedText &);

private:
  std::variant<CharBlock, SetOfChars> u_;
};

class Message : public common::ReferenceCounted<Message> {
public:
  using Reference = common::CountedReference<Message>;

  Message(const Message &) = default;
  Message(Message &&) = default;
  Message &operator=(const Message &) = default;
  Message &operator=(Message &&) = default;

  Message(ProvenanceRange pr, const MessageFixedText &t)
    : location_{pr}, text_{t} {}
  Message(ProvenanceRange pr, const MessageFormattedText &s)
    : location_{pr}, text_{s} {}
  Message(ProvenanceRange pr, MessageFormattedText &&s)
    : location_{pr}, text_{std::move(s)} {}
  Message(ProvenanceRange pr, const MessageExpectedText &t)
    : location_{pr}, text_{t} {}

  Message(CharBlock csr, const MessageFixedText &t)
    : location_{csr}, text_{t} {}
  Message(CharBlock csr, const MessageFormattedText &s)
    : location_{csr}, text_{s} {}
  Message(CharBlock csr, MessageFormattedText &&s)
    : location_{csr}, text_{std::move(s)} {}
  Message(CharBlock csr, const MessageExpectedText &t)
    : location_{csr}, text_{t} {}

  template<typename RANGE, typename A, typename... As>
  Message(RANGE r, const MessageFixedText &t, A &&x, As &&... xs)
    : location_{r}, text_{MessageFormattedText{
                        t, std::forward<A>(x), std::forward<As>(xs)...}} {}

  bool attachmentIsContext() const { return attachmentIsContext_; }
  Reference attachment() const { return attachment_; }

  void SetContext(Message *c) {
    attachment_ = c;
    attachmentIsContext_ = true;
  }
  Message &Attach(Message *);
  template<typename... A> Message &Attach(A &&... args) {
    return Attach(new Message{std::forward<A>(args)...});  // reference-counted
  }

  bool SortBefore(const Message &that) const;
  bool IsFatal() const;
  std::string ToString() const;
  std::optional<ProvenanceRange> GetProvenanceRange(const CookedSource &) const;
  void Emit(
      std::ostream &, const CookedSource &, bool echoSourceLine = true) const;

  // If this Message or any of its attachments locates itself via a CharBlock
  // within a particular CookedSource, replace its location with the
  // corresponding ProvenanceRange.
  void ResolveProvenances(const CookedSource &);

  bool IsMergeable() const {
    return std::holds_alternative<MessageExpectedText>(text_);
  }
  bool Merge(const Message &);

private:
  bool AtSameLocation(const Message &) const;

  std::variant<ProvenanceRange, CharBlock> location_;
  std::variant<MessageFixedText, MessageFormattedText, MessageExpectedText>
      text_;
  bool attachmentIsContext_{false};
  Reference attachment_;
};

class Messages {
public:
  Messages() {}
  Messages(Messages &&that) : messages_{std::move(that.messages_)} {
    if (!messages_.empty()) {
      last_ = that.last_;
      that.ResetLastPointer();
    }
  }
  Messages &operator=(Messages &&that) {
    messages_ = std::move(that.messages_);
    if (messages_.empty()) {
      ResetLastPointer();
    } else {
      last_ = that.last_;
      that.ResetLastPointer();
    }
    return *this;
  }

  bool empty() const { return messages_.empty(); }

  template<typename... A> Message &Say(A &&... args) {
    last_ = messages_.emplace_after(last_, std::forward<A>(args)...);
    return *last_;
  }

  void Annex(Messages &&that) {
    if (!that.messages_.empty()) {
      messages_.splice_after(last_, that.messages_);
      last_ = that.last_;
      that.ResetLastPointer();
    }
  }

  void Restore(Messages &&that) {
    that.Annex(std::move(*this));
    *this = std::move(that);
  }

  bool Merge(const Message &);
  void Merge(Messages &&);
  void Copy(const Messages &);
  void ResolveProvenances(const CookedSource &);
  void Emit(std::ostream &, const CookedSource &cooked,
      bool echoSourceLines = true) const;
  void AttachTo(Message &);
  bool AnyFatalError() const;

private:
  void ResetLastPointer() { last_ = messages_.before_begin(); }

  std::forward_list<Message> messages_;
  std::forward_list<Message>::iterator last_{messages_.before_begin()};
};

class ContextualMessages {
public:
  ContextualMessages() = default;
  ContextualMessages(CharBlock at, Messages *m) : at_{at}, messages_{m} {}
  ContextualMessages(const ContextualMessages &that)
    : at_{that.at_}, messages_{that.messages_} {}

  CharBlock at() const { return at_; }
  Messages *messages() const { return messages_; }

  // Set CharBlock for messages; restore when the returned value is deleted
  common::Restorer<CharBlock> SetLocation(CharBlock at) {
    if (at.empty()) {
      at = at_;
    }
    return common::ScopedSet(at_, std::move(at));
  }

  template<typename... A> Message *Say(CharBlock at, A &&... args) {
    if (messages_ != nullptr) {
      return &messages_->Say(at, std::forward<A>(args)...);
    } else {
      return nullptr;
    }
  }

  template<typename... A> Message *Say(A &&... args) {
    return Say(at_, std::forward<A>(args)...);
  }

private:
  CharBlock at_;
  Messages *messages_{nullptr};
};
}
#endif  // FORTRAN_PARSER_MESSAGE_H_
