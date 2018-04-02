#ifndef FORTRAN_PARSER_MESSAGE_H_
#define FORTRAN_PARSER_MESSAGE_H_

// Defines a representation for sequences of compiler messages.
// Supports nested contextualization.

#include "idioms.h"
#include "provenance.h"
#include <cstddef>
#include <forward_list>
#include <memory>
#include <optional>
#include <ostream>
#include <string>

namespace Fortran {
namespace parser {

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

constexpr MessageFixedText operator""_en_US(const char str[], std::size_t n) {
  return MessageFixedText{str, n, false /* not fatal */};
}

constexpr MessageFixedText operator""_err_en_US(
    const char str[], std::size_t n) {
  return MessageFixedText{str, n, true /* fatal */};
}

std::ostream &operator<<(std::ostream &, const MessageFixedText &);

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
  MessageExpectedText(const char *s, std::size_t n) : str_{s}, bytes_{n} {}
  explicit MessageExpectedText(char ch) : singleton_{ch} {}
  MessageFixedText AsMessageFixedText() const;

private:
  const char *str_{nullptr};
  char singleton_;
  std::size_t bytes_{1};
};

class Message;
using MessageContext = std::shared_ptr<Message>;

class Message {
public:
  Message() {}
  Message(const Message &) = default;
  Message(Message &&) = default;
  Message &operator=(const Message &that) = default;
  Message &operator=(Message &&that) = default;

  Message(Provenance p, MessageFixedText t, MessageContext c = nullptr)
    : provenance_{p}, text_{t}, context_{c}, isFatal_{t.isFatal()} {}
  Message(Provenance p, MessageFormattedText &&s, MessageContext c = nullptr)
    : provenance_{p}, string_{s.MoveString()}, context_{c}, isFatal_{
                                                                s.isFatal()} {}
  Message(Provenance p, MessageExpectedText t, MessageContext c = nullptr)
    : provenance_{p}, text_{t.AsMessageFixedText()},
      isExpectedText_{true}, context_{c}, isFatal_{true} {}

  Message(const char *csl, MessageFixedText t, MessageContext c = nullptr)
    : cookedSourceLocation_{csl}, text_{t}, context_{c}, isFatal_{t.isFatal()} {
  }
  Message(const char *csl, MessageFormattedText &&s, MessageContext c = nullptr)
    : cookedSourceLocation_{csl}, string_{s.MoveString()}, context_{c},
      isFatal_{s.isFatal()} {}
  Message(const char *csl, MessageExpectedText t, MessageContext c = nullptr)
    : cookedSourceLocation_{csl}, text_{t.AsMessageFixedText()},
      isExpectedText_{true}, context_{c}, isFatal_{true} {}

  bool operator<(const Message &that) const {
    if (cookedSourceLocation_ != nullptr) {
      return cookedSourceLocation_ < that.cookedSourceLocation_;
    } else if (that.cookedSourceLocation_ != nullptr) {
      return false;
    } else {
      return provenance_ < that.provenance_;
    }
  }

  Provenance provenance() const { return provenance_; }
  const char *cookedSourceLocation() const { return cookedSourceLocation_; }
  MessageContext context() const { return context_; }
  bool isFatal() const { return isFatal_; }

  Provenance Emit(
      std::ostream &, const CookedSource &, bool echoSourceLine = true) const;

private:
  Provenance provenance_;
  const char *cookedSourceLocation_{nullptr};
  MessageFixedText text_;
  bool isExpectedText_{false};  // implies "expected '%s'"_err_en_US
  std::string string_;
  MessageContext context_;
  bool isFatal_{false};
};

class Messages {
  using list_type = std::forward_list<Message>;

public:
  using iterator = list_type::iterator;
  using const_iterator = list_type::const_iterator;

  explicit Messages(const CookedSource &cooked) : cooked_{cooked} {}
  Messages(Messages &&that)
    : cooked_{that.cooked_}, messages_{std::move(that.messages_)},
      last_{that.last_} {}
  Messages &operator=(Messages &&that) {
    swap(that);
    return *this;
  }

  void swap(Messages &that) {
    messages_.swap(that.messages_);
    std::swap(last_, that.last_);
  }

  bool empty() const { return messages_.empty(); }
  iterator begin() { return messages_.begin(); }
  iterator end() { return messages_.end(); }
  const_iterator begin() const { return messages_.cbegin(); }
  const_iterator end() const { return messages_.cend(); }
  const_iterator cbegin() const { return messages_.cbegin(); }
  const_iterator cend() const { return messages_.cend(); }

  const CookedSource &cooked() const { return cooked_; }

  bool IsValidLocation(const Message &m) {
    if (auto p{m.cookedSourceLocation()}) {
      return cooked_.IsValid(p);
    } else {
      return cooked_.IsValid(m.provenance());
    }
  }

  Message &Put(Message &&m) {
    CHECK(IsValidLocation(m));
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

  void Emit(std::ostream &, const char *prefix = nullptr,
      bool echoSourceLines = true) const;

  bool AnyFatalError() const;

private:
  const CookedSource &cooked_;
  list_type messages_;
  iterator last_;  // valid iff messages_ nonempty
};
}  // namespace parser
}  // namespace Fortran
#endif  // FORTRAN_PARSER_MESSAGE_H_
