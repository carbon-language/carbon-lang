#ifndef FORTRAN_PARSER_MESSAGE_H_
#define FORTRAN_PARSER_MESSAGE_H_

// Defines a representation for sequences of compiler messages.
// Supports nested contextualization.

#include "idioms.h"
#include "provenance.h"
#include <forward_list>
#include <memory>
#include <optional>
#include <ostream>
#include <string>

namespace Fortran {
namespace parser {

// Use "..."_en_US literals to define the static text of a message.
class MessageFixedText {
public:
  MessageFixedText() {}
  constexpr MessageFixedText(const char str[], size_t n)
    : str_{str}, bytes_{n} {}
  constexpr MessageFixedText(const MessageFixedText &) = default;
  MessageFixedText(MessageFixedText &&) = default;
  constexpr MessageFixedText &operator=(const MessageFixedText &) = default;
  MessageFixedText &operator=(MessageFixedText &&) = default;

  const char *str() const { return str_; }
  size_t size() const { return bytes_; }
  bool empty() const { return bytes_ == 0; }

  std::string ToString() const;

private:
  const char *str_{nullptr};
  size_t bytes_{0};
};

constexpr MessageFixedText operator""_en_US(const char str[], size_t n) {
  return MessageFixedText{str, n};
}

std::ostream &operator<<(std::ostream &, const MessageFixedText &);

class MessageFormattedText {
public:
  MessageFormattedText(MessageFixedText, ...);
  std::string MoveString() { return std::move(string_); }

private:
  std::string string_;
};

// Represents a formatted rendition of "expected '%s'"_en_US on a constant text.
class MessageExpectedText {
public:
  MessageExpectedText(const char *s, size_t n) : str_{s}, bytes_{n} {}
  explicit MessageExpectedText(char ch) : singleton_{ch} {}
  MessageFixedText AsMessageFixedText() const;

private:
  const char *str_{nullptr};
  char singleton_;
  size_t bytes_{1};
};

class Message;
using MessageContext = std::shared_ptr<Message>;

class Message {
public:
  Message() {}
  Message(const Message &) = default;
  Message(Provenance p, MessageFixedText t, MessageContext c = nullptr)
    : provenance_{p}, text_{t}, context_{c} {}
  Message(Provenance p, MessageFormattedText &&s, MessageContext c = nullptr)
    : provenance_{p}, string_{s.MoveString()}, context_{c} {}
  Message(Provenance p, MessageExpectedText t, MessageContext c = nullptr)
    : provenance_{p}, text_{t.AsMessageFixedText()},
      isExpectedText_{true}, context_{c} {}
  Message(Message &&) = default;
  Message &operator=(const Message &that) = default;
  Message &operator=(Message &&that) = default;

  bool operator<(const Message &that) const {
    return provenance_ < that.provenance_;
  }

  Provenance provenance() const { return provenance_; }
  MessageContext context() const { return context_; }

  Provenance Emit(
      std::ostream &, const AllSources &, bool echoSourceLine = true) const;

private:
  Provenance provenance_;
  MessageFixedText text_;
  bool isExpectedText_{false};  // implies "expected '%s'"_en_US
  std::string string_;
  MessageContext context_;
};

class Messages {
  using list_type = std::forward_list<Message>;

public:
  using iterator = list_type::iterator;
  using const_iterator = list_type::const_iterator;

  explicit Messages(const AllSources &sources) : allSources_{sources} {}
  Messages(Messages &&that)
    : allSources_{that.allSources_}, messages_{std::move(that.messages_)},
      last_{that.last_} {}
  Messages &operator=(Messages &&that) {
    swap(that);
    return *this;
  }

  void swap(Messages &that) {
    messages_.swap(that.messages_);
    std::swap(last_, that.last_);
  }

  iterator begin() { return messages_.begin(); }
  iterator end() { return messages_.end(); }
  const_iterator begin() const { return messages_.cbegin(); }
  const_iterator end() const { return messages_.cend(); }
  const_iterator cbegin() const { return messages_.cbegin(); }
  const_iterator cend() const { return messages_.cend(); }

  const AllSources &allSources() const { return allSources_; }

  Message &Put(Message &&m) {
    CHECK(m.provenance() < allSources_.size());
    if (messages_.empty()) {
      messages_.emplace_front(std::move(m));
      last_ = messages_.begin();
    } else {
      last_ = messages_.emplace_after(last_, std::move(m));
    }
    return *last_;
  }

  void Annex(Messages *that) {
    if (!that->messages_.empty()) {
      if (messages_.empty()) {
        messages_ = std::move(that->messages_);
      } else {
        messages_.splice_after(last_, that->messages_);
      }
      last_ = that->last_;
    }
  }

  void Emit(std::ostream &, const char *prefix = nullptr) const;

private:
  const AllSources &allSources_;
  list_type messages_;
  iterator last_;  // valid iff messages_ nonempty
};
}  // namespace parser
}  // namespace Fortran
#endif  // FORTRAN_PARSER_MESSAGE_H_
