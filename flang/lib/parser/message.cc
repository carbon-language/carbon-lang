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
#include "idioms.h"
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
      visitors{[](const CharBlock &cb) {
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

void MessageExpectedText::Incorporate(const MessageExpectedText &that) {
  std::visit(
      visitors{[&](SetOfChars &s1, const SetOfChars &s2) { s1.Union(s2); },
          [](const auto &, const auto &) {}},
      u_, that.u_);
}

bool Message::operator<(const Message &that) const {
  // Messages from prescanning have ProvenanceRange values for their locations,
  // while messages from later phases have CharBlock values, since the
  // conversion of cooked source stream locations to provenances is not
  // free and needs to be deferred, since many messages created during parsing
  // are speculative.  Messages with ProvenanceRange locations are ordered
  // before others for sorting.
  return std::visit(
      visitors{[](const CharBlock &cb1, const CharBlock &cb2) {
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
  return std::visit(visitors{[](const MessageExpectedText &) { return true; },
                        [](const auto &x) { return x.isFatal(); }},
      text_);
}

std::string Message::ToString() const {
  return std::visit(
      visitors{[](const MessageFixedText &t) {
                 return t.text().NULTerminatedToString();
               },
          [](const MessageFormattedText &t) { return t.string(); },
          [](const MessageExpectedText &e) { return e.ToString(); }},
      text_);
}

ProvenanceRange Message::GetProvenanceRange(const CookedSource &cooked) const {
  return std::visit(visitors{[&](const CharBlock &cb) {
                               return cooked.GetProvenanceRange(cb);
                             },
                        [](const ProvenanceRange &pr) { return pr; }},
      location_);
}

void Message::Emit(
    std::ostream &o, const CookedSource &cooked, bool echoSourceLine) const {
  ProvenanceRange provenanceRange{GetProvenanceRange(cooked)};
  std::string text;
  if (IsFatal()) {
    text += "error: ";
  }
  text += ToString();
  AllSources &sources{cooked.allSources()};
  sources.EmitMessage(o, provenanceRange, text, echoSourceLine);
  for (const Message *context{context_.get()}; context != nullptr;
       context = context->context_.get()) {
    ProvenanceRange contextProvenance{context->GetProvenanceRange(cooked)};
    text = "in the context: ";
    text += context->ToString();
    // TODO: don't echo the source lines of a context when it's the
    // same line (or maybe just never echo source for context)
    sources.EmitMessage(o, contextProvenance, text,
        echoSourceLine && contextProvenance != provenanceRange);
    provenanceRange = contextProvenance;
  }
  for (const Message *attachment{attachment_.get()}; attachment != nullptr;
       attachment = attachment->attachment_.get()) {
    sources.EmitMessage(o, attachment->GetProvenanceRange(cooked),
        attachment->ToString(), echoSourceLine);
  }
}

void Message::Incorporate(Message &that) {
  std::visit(
      visitors{[&](MessageExpectedText &e1, const MessageExpectedText &e2) {
                 e1.Incorporate(e2);
               },
          [](const auto &, const auto &) {}},
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
      visitors{[](const CharBlock &cb1, const CharBlock &cb2) {
                 return cb1.begin() == cb2.begin();
               },
          [](const ProvenanceRange &pr1, const ProvenanceRange &pr2) {
            return pr1.start() == pr2.start();
          },
          [](const auto &, const auto &) { return false; }},
      location_, that.location_);
}

void Messages::Incorporate(Messages &that) {
  if (messages_.empty()) {
    *this = std::move(that);
  } else if (!that.messages_.empty()) {
    last_->Incorporate(*that.last_);
  }
}

void Messages::Copy(const Messages &that) {
  for (const Message &m : that.messages_) {
    Message copy{m};
    Put(std::move(copy));
  }
}

void Messages::Emit(
    std::ostream &o, const CookedSource &cooked, bool echoSourceLines) const {
  std::vector<const Message *> sorted;
  for (const auto &msg : messages_) {
    sorted.push_back(&msg);
  }
#if 0
  // It would be great to sort the messages by location so that messages
  // from the several compiler passes would be interleaved, but we can't
  // do that until we have a means of maintaining a relationship between
  // multiple messages coming out of semantics.
  std::sort(sorted.begin(), sorted.end(),
      [](const Message *x, const Message *y) { return *x < *y; });
#endif
  for (const Message *msg : sorted) {
    msg->Emit(o, cooked, echoSourceLines);
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
