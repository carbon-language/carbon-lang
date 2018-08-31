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

#include "message.h"
#include "char-set.h"
#include "../common/idioms.h"
#include <algorithm>
#include <cstdarg>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace Fortran::parser {

std::ostream &operator<<(std::ostream &o, const MessageFixedText &t) {
  std::size_t n{t.text().size()};
  for (std::size_t j{0}; j < n; ++j) {
    o << t.text()[j];
  }
  return o;
}

MessageFormattedText::MessageFormattedText(MessageFixedText text, ...)
  : isFatal_{text.isFatal()} {
  const char *p{text.text().begin()};
  std::string asString;
  if (*text.text().end() != '\0') {
    // not NUL-terminated
    asString = text.text().NULTerminatedToString();
    p = asString.data();
  }
  char buffer[256];
  va_list ap;
  va_start(ap, text);
  vsnprintf(buffer, sizeof buffer, p, ap);
  va_end(ap);
  string_ = buffer;
}

std::string MessageExpectedText::ToString() const {
  return std::visit(
      common::visitors{[](const CharBlock &cb) {
                         return MessageFormattedText("expected '%s'"_err_en_US,
                             cb.NULTerminatedToString().data())
                             .MoveString();
                       },
          [](const SetOfChars &set) {
            SetOfChars expect{set};
            if (expect.Has('\n')) {
              expect = expect.Difference('\n');
              if (expect.empty()) {
                return "expected end of line"_err_en_US.text().ToString();
              } else {
                std::string s{expect.ToString()};
                if (s.size() == 1) {
                  return MessageFormattedText(
                      "expected end of line or '%s'"_err_en_US, s.data())
                      .MoveString();
                } else {
                  return MessageFormattedText(
                      "expected end of line or one of '%s'"_err_en_US, s.data())
                      .MoveString();
                }
              }
            }
            std::string s{expect.ToString()};
            if (s.size() != 1) {
              return MessageFormattedText(
                  "expected one of '%s'"_err_en_US, s.data())
                  .MoveString();
            } else {
              return MessageFormattedText("expected '%s'"_err_en_US, s.data())
                  .MoveString();
            }
          }},
      u_);
}

bool MessageExpectedText::Merge(const MessageExpectedText &that) {
  return std::visit(common::visitors{[](SetOfChars &s1, const SetOfChars &s2) {
                                       s1 = s1.Union(s2);
                                       return true;
                                     },
                        [](const auto &, const auto &) { return false; }},
      u_, that.u_);
}

bool Message::SortBefore(const Message &that) const {
  // Messages from prescanning have ProvenanceRange values for their locations,
  // while messages from later phases have CharBlock values, since the
  // conversion of cooked source stream locations to provenances is not
  // free and needs to be deferred, since many messages created during parsing
  // are speculative.  Messages with ProvenanceRange locations are ordered
  // before others for sorting.
  return std::visit(
      common::visitors{[](const CharBlock &cb1, const CharBlock &cb2) {
                         return cb1.begin() < cb2.begin();
                       },
          [](const CharBlock &, const ProvenanceRange &) { return false; },
          [](const ProvenanceRange &pr1, const ProvenanceRange &pr2) {
            return pr1.start() < pr2.start();
          },
          [](const ProvenanceRange &, const CharBlock &) { return true; }},
      location_, that.location_);
}

bool Message::IsFatal() const {
  return std::visit(
      common::visitors{[](const MessageExpectedText &) { return true; },
          [](const MessageFixedText &x) { return x.isFatal(); },
          [](const MessageFormattedText &x) { return x.isFatal(); }},
      text_);
}

std::string Message::ToString() const {
  return std::visit(
      common::visitors{[](const MessageFixedText &t) {
                         return t.text().NULTerminatedToString();
                       },
          [](const MessageFormattedText &t) { return t.string(); },
          [](const MessageExpectedText &e) { return e.ToString(); }},
      text_);
}

void Message::ResolveProvenances(const CookedSource &cooked) {
  if (CharBlock * cb{std::get_if<CharBlock>(&location_)}) {
    if (std::optional<ProvenanceRange> resolved{
            cooked.GetProvenanceRange(*cb)}) {
      location_ = *resolved;
    }
  }
  if (Message * attachment{attachment_.get()}) {
    attachment->ResolveProvenances(cooked);
  }
}

std::optional<ProvenanceRange> Message::GetProvenanceRange(
    const CookedSource &cooked) const {
  return std::visit(common::visitors{[&](const CharBlock &cb) {
                                       return cooked.GetProvenanceRange(cb);
                                     },
                        [](const ProvenanceRange &pr) {
                          return std::optional<ProvenanceRange>{pr};
                        }},
      location_);
}

void Message::Emit(
    std::ostream &o, const CookedSource &cooked, bool echoSourceLine) const {
  std::optional<ProvenanceRange> provenanceRange{GetProvenanceRange(cooked)};
  std::string text;
  if (IsFatal()) {
    text += "error: ";
  }
  text += ToString();
  const AllSources &sources{cooked.allSources()};
  sources.EmitMessage(o, provenanceRange, text, echoSourceLine);
  if (attachmentIsContext_) {
    for (const Message *context{attachment_.get()}; context != nullptr;
         context = context->attachment_.get()) {
      std::optional<ProvenanceRange> contextProvenance{
          context->GetProvenanceRange(cooked)};
      text = "in the context: ";
      text += context->ToString();
      // TODO: don't echo the source lines of a context when it's the
      // same line (or maybe just never echo source for context)
      sources.EmitMessage(o, contextProvenance, text,
          echoSourceLine && contextProvenance != provenanceRange);
      provenanceRange = contextProvenance;
    }
  } else {
    for (const Message *attachment{attachment_.get()}; attachment != nullptr;
         attachment = attachment->attachment_.get()) {
      sources.EmitMessage(o, attachment->GetProvenanceRange(cooked),
          attachment->ToString(), echoSourceLine);
    }
  }
}

bool Message::Merge(const Message &that) {
  return AtSameLocation(that) &&
      (!that.attachment_.get() ||
          attachment_.get() == that.attachment_.get()) &&
      std::visit(common::visitors{[](MessageExpectedText &e1,
                                      const MessageExpectedText &e2) {
                                    return e1.Merge(e2);
                                  },
                     [](const auto &, const auto &) { return false; }},
          text_, that.text_);
}

void Message::Attach(Message *m) {
  if (!attachment_) {
    attachment_ = m;
  } else {
    attachment_->Attach(m);
  }
}

bool Message::AtSameLocation(const Message &that) const {
  return std::visit(
      common::visitors{[](const CharBlock &cb1, const CharBlock &cb2) {
                         return cb1.begin() == cb2.begin();
                       },
          [](const ProvenanceRange &pr1, const ProvenanceRange &pr2) {
            return pr1.start() == pr2.start();
          },
          [](const auto &, const auto &) { return false; }},
      location_, that.location_);
}

bool Messages::Merge(const Message &msg) {
  if (msg.IsMergeable()) {
    for (auto &m : messages_) {
      if (m.Merge(msg)) {
        return true;
      }
    }
  }
  return false;
}

void Messages::Merge(Messages &&that) {
  if (messages_.empty()) {
    *this = std::move(that);
  } else {
    while (!that.messages_.empty()) {
      if (Merge(that.messages_.front())) {
        that.messages_.pop_front();
      } else {
        messages_.splice_after(
            last_, that.messages_, that.messages_.before_begin());
        ++last_;
      }
    }
    that.ResetLastPointer();
  }
}

void Messages::Copy(const Messages &that) {
  for (const Message &m : that.messages_) {
    Message copy{m};
    Say(std::move(copy));
  }
}

void Messages::ResolveProvenances(const CookedSource &cooked) {
  for (Message &m : messages_) {
    m.ResolveProvenances(cooked);
  }
}

void Messages::Emit(
    std::ostream &o, const CookedSource &cooked, bool echoSourceLines) const {
  std::vector<const Message *> sorted;
  for (const auto &msg : messages_) {
    sorted.push_back(&msg);
  }
  std::sort(sorted.begin(), sorted.end(),
      [](const Message *x, const Message *y) { return x->SortBefore(*y); });
  for (const Message *msg : sorted) {
    msg->Emit(o, cooked, echoSourceLines);
  }
}

void Messages::AttachTo(Message &msg) {
  for (const Message &m : messages_) {
    msg.Attach(m);
  }
}

bool Messages::AnyFatalError() const {
  for (const auto &msg : messages_) {
    if (msg.IsFatal()) {
      return true;
    }
  }
  return false;
}

}  // namespace Fortran::parser
