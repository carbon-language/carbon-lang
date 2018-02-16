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

class Message;
using MessageContext = std::shared_ptr<Message>;

class MessageText {
public:
  MessageText() {}
  constexpr MessageText(const char str[], size_t n) : str_{str}, bytes_{n} {}
  constexpr MessageText(const MessageText &) = default;
  MessageText(MessageText &&) = default;
  constexpr MessageText &operator=(const MessageText &) = default;
  MessageText &operator=(MessageText &&) = default;

  const char *str() const { return str_; }
  size_t size() const { return bytes_; }

  std::string ToString() const { return std::string(str_, bytes_); }

private:
  const char *str_{nullptr};
  size_t bytes_{0};
};

constexpr MessageText operator""_msg(const char str[], size_t n) {
  return MessageText{str, n};
}

std::ostream &operator<<(std::ostream &, const MessageText &);

class Message {
public:
  Message() {}
  Message(const Message &) = default;
  Message(Message &&) = default;

  Message(Provenance at, MessageText t, MessageContext ctx = nullptr);
  Message(Provenance at, const std::string &msg, MessageContext ctx = nullptr)
    : provenance_{at}, message_{msg}, context_{ctx} {}
  Message(Provenance at, std::string &&msg, MessageContext ctx = nullptr)
    : provenance_{at}, message_{std::move(msg)}, context_{ctx} {}
  Message(Provenance at, const char *msg, MessageContext ctx = nullptr)
    : provenance_{at}, message_{msg}, context_{ctx} {}

  Message &operator=(const Message &that) = default;
  Message &operator=(Message &&that) = default;

  bool operator<(const Message &that) const {
    return provenance_ < that.provenance_;
  }

  Provenance provenance() const { return provenance_; }
  MessageText text() const { return text_; }
  std::string message() const { return message_; }
  MessageContext context() const { return context_; }

  Provenance Emit(
      std::ostream &, const AllSources &, bool echoSourceLine = true) const;

private:
  Provenance provenance_;
  MessageText text_;
  std::string message_;
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

  void Put(Message &&m) {
    CHECK(m.provenance() < allSources_.size());
    if (messages_.empty()) {
      messages_.emplace_front(std::move(m));
      last_ = messages_.begin();
    } else {
      last_ = messages_.emplace_after(last_, std::move(m));
    }
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

  void Emit(std::ostream &) const;

private:
  const AllSources &allSources_;
  list_type messages_;
  iterator last_;  // valid iff messages_ nonempty
};
}  // namespace parser
}  // namespace Fortran
#endif  // FORTRAN_PARSER_MESSAGE_H_
