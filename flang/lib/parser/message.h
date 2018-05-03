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

#ifndef FORTRAN_PARSER_MESSAGE_H_
#define FORTRAN_PARSER_MESSAGE_H_

// Defines a representation for sequences of compiler messages.
// Supports nested contextualization.

#include "char-set.h"
#include "idioms.h"
#include "provenance.h"
#include "reference-counted.h"
#include <cstddef>
#include <cstring>
#include <forward_list>
#include <optional>
#include <ostream>
#include <string>
#include <utility>

namespace Fortran::parser {

// Use "..."_err_en_US and "..."_en_US literals to define the static
// text and fatality of a message.
class MessageFixedText {
public:
  MessageFixedText() {}
  constexpr MessageFixedText(
      const char str[], std::size_t n, bool isFatal = false)
    : str_{str}, bytes_{n}, isFatal_{isFatal} {}
  constexpr MessageFixedText(const MessageFixedText &) = default;
  MessageFixedText(MessageFixedText &&) = default;
  constexpr MessageFixedText &operator=(const MessageFixedText &) = default;
  MessageFixedText &operator=(MessageFixedText &&) = default;

  const char *str() const { return str_; }
  std::size_t size() const { return bytes_; }
  bool empty() const { return bytes_ == 0; }
  bool isFatal() const { return isFatal_; }

  std::string ToString() const;

private:
  const char *str_{nullptr};
  std::size_t bytes_{0};
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
}  // namespace literals

class MessageFormattedText {
public:
  MessageFormattedText(MessageFixedText, ...);
  std::string MoveString() { return std::move(string_); }
  bool isFatal() const { return isFatal_; }

private:
  std::string string_;
  bool isFatal_{false};
};

// Represents a formatted rendition of "expected '%s'"_err_en_US
// on a constant text.
class MessageExpectedText {
public:
  MessageExpectedText(const char *s, std::size_t n) : str_{s}, bytes_{n} {
    if (n == std::string::npos) {
      bytes_ = std::strlen(s);
    }
  }
  MessageExpectedText(MessageExpectedText &&) = default;
  explicit MessageExpectedText(char ch) : set_{ch} {}
  explicit MessageExpectedText(SetOfChars set) : set_{set} {}

  const char *str() const { return str_; }
  std::size_t size() const { return bytes_; }
  SetOfChars set() const { return set_; }

private:
  const char *str_{nullptr};
  std::size_t bytes_{0};
  SetOfChars set_;
};

class Message : public ReferenceCounted<Message> {
public:
  using Context = CountedReference<Message>;

  Message() {}
  Message(const Message &) = default;
  Message(Message &&) = default;
  Message &operator=(const Message &that) = default;
  Message &operator=(Message &&that) = default;

  Message(ProvenanceRange pr, MessageFixedText t)
    : provenanceRange_{pr}, fixedText_{t.str()},
      fixedBytes_{t.size()}, isFatal_{t.isFatal()} {}
  Message(ProvenanceRange pr, MessageFormattedText &&s)
    : provenanceRange_{pr}, string_{s.MoveString()}, isFatal_{s.isFatal()} {}
  Message(ProvenanceRange pr, MessageExpectedText t)
    : provenanceRange_{pr}, fixedText_{t.str()}, fixedBytes_{t.size()},
      isExpected_{true}, expected_{t.set()}, isFatal_{true} {}

  Message(CharBlock csr, MessageFixedText t)
    : cookedSourceRange_{csr}, fixedText_{t.str()},
      fixedBytes_{t.size()}, isFatal_{t.isFatal()} {}
  Message(CharBlock csr, MessageFormattedText &&s)
    : cookedSourceRange_{csr}, string_{s.MoveString()}, isFatal_{s.isFatal()} {}
  Message(CharBlock csr, MessageExpectedText t)
    : cookedSourceRange_{csr}, fixedText_{t.str()}, fixedBytes_{t.size()},
      isExpected_{true}, expected_{t.set()}, isFatal_{true} {}

  bool operator<(const Message &that) const {
    if (cookedSourceRange_.begin() != nullptr) {
      return cookedSourceRange_.begin() < that.cookedSourceRange_.begin();
    } else if (that.cookedSourceRange_.begin() != nullptr) {
      return false;
    } else {
      return provenanceRange_.start() < that.provenanceRange_.start();
    }
  }

  Context context() const { return context_; }
  Message &set_context(Message *c) {
    context_ = c;
    return *this;
  }
  bool isFatal() const { return isFatal_; }

  void Incorporate(Message &);
  std::string ToString() const;
  ProvenanceRange GetProvenanceRange(const CookedSource &) const;
  void Emit(
      std::ostream &, const CookedSource &, bool echoSourceLine = true) const;

private:
  ProvenanceRange provenanceRange_;
  CharBlock cookedSourceRange_;
  const char *fixedText_{nullptr};
  std::size_t fixedBytes_{0};
  bool isExpected_{false};
  std::string string_;
  SetOfChars expected_;
  Context context_;
  bool isFatal_{false};
};

class Messages {
public:
  Messages() {}
  Messages(Messages &&that) : messages_{std::move(that.messages_)} {
    if (!messages_.empty()) {
      last_ = that.last_;
      that.last_ = that.messages_.before_begin();
    }
  }
  Messages &operator=(Messages &&that) {
    messages_ = std::move(that.messages_);
    if (messages_.empty()) {
      last_ = messages_.before_begin();
    } else {
      last_ = that.last_;
      that.last_ = that.messages_.before_begin();
    }
    return *this;
  }

  bool empty() const { return messages_.empty(); }

  void Put(Message &&m) {
    last_ = messages_.emplace_after(last_, std::move(m));
  }

  template<typename... A> Message &Say(A &&... args) {
    last_ = messages_.emplace_after(last_, std::forward<A>(args)...);
    return *last_;
  }

  void Annex(Messages &that) {
    if (!that.messages_.empty()) {
      messages_.splice_after(last_, that.messages_);
      last_ = that.last_;
      that.last_ = that.messages_.before_begin();
    }
  }

  void Incorporate(Messages &);
  void Copy(const Messages &);
  void Emit(std::ostream &, const CookedSource &cooked,
      bool echoSourceLines = true) const;

  bool AnyFatalError() const;

private:
  using listType = std::forward_list<Message>;
  listType messages_;
  listType::iterator last_{messages_.before_begin()};
};

}  // namespace Fortran::parser
#endif  // FORTRAN_PARSER_MESSAGE_H_
