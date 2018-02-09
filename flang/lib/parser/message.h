#ifndef FORTRAN_MESSAGE_H_
#define FORTRAN_MESSAGE_H_

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

class Message {
public:
  Message() {}
  Message(const Message &) = default;
  Message(Message &&) = default;

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
  std::string message() const { return message_; }
  MessageContext context() const { return context_; }

  void Emit(std::ostream &, const AllSources &) const;

private:
  Provenance provenance_;
  std::string message_;
  MessageContext context_;
};

class Messages {
  using list_type = std::forward_list<Message>;

public:
  using iterator = list_type::iterator;
  using const_iterator = list_type::const_iterator;

  Messages() {}
  Messages(Messages &&that)
    : messages_{std::move(that.messages_)}, last_{that.last_} {}
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

  void Put(Message &&m) {
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

  void Emit(std::ostream &, const AllSources &) const;

private:
  list_type messages_;
  iterator last_;  // valid iff messages_ nonempty
};
}  // namespace parser
}  // namespace Fortran
#endif  // FORTRAN_MESSAGE_H_
