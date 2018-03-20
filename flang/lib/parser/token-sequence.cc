#include "token-sequence.h"
#include "characters.h"

namespace Fortran {
namespace parser {

void TokenSequence::clear() {
  start_.clear();
  nextStart_ = 0;
  char_.clear();
  provenances_.clear();
}

void TokenSequence::pop_back() {
  std::size_t bytes{nextStart_ - start_.back()};
  nextStart_ = start_.back();
  start_.pop_back();
  char_.resize(nextStart_);
  provenances_.RemoveLastBytes(bytes);
}

void TokenSequence::shrink_to_fit() {
  start_.shrink_to_fit();
  char_.shrink_to_fit();
  provenances_.shrink_to_fit();
}

void TokenSequence::Put(const TokenSequence &that) {
  if (nextStart_ < char_.size()) {
    start_.push_back(nextStart_);
  }
  int offset = char_.size();
  for (int st : that.start_) {
    start_.push_back(st + offset);
  }
  char_.insert(char_.end(), that.char_.begin(), that.char_.end());
  nextStart_ = char_.size();
  provenances_.Put(that.provenances_);
}

void TokenSequence::Put(const TokenSequence &that, ProvenanceRange range) {
  std::size_t offset{0};
  for (std::size_t j{0}; j < that.size(); ++j) {
    CharBlock tok{that[j]};
    Put(tok, range.OffsetMember(offset));
    offset += tok.size();
  }
  CHECK(offset == range.size());
}

void TokenSequence::Put(
    const TokenSequence &that, std::size_t at, std::size_t tokens) {
  ProvenanceRange provenance;
  std::size_t offset{0};
  for (; tokens-- > 0; ++at) {
    CharBlock tok{that[at]};
    std::size_t tokBytes{tok.size()};
    for (std::size_t j{0}; j < tokBytes; ++j) {
      if (offset == provenance.size()) {
        offset = 0;
        provenance = that.provenances_.Map(that.start_[at] + j);
      }
      PutNextTokenChar(tok[j], provenance.OffsetMember(offset++));
    }
    CloseToken();
  }
}

void TokenSequence::Put(
    const char *s, std::size_t bytes, Provenance provenance) {
  for (std::size_t j{0}; j < bytes; ++j) {
    PutNextTokenChar(s[j], provenance + j);
  }
  CloseToken();
}

void TokenSequence::Put(const CharBlock &t, Provenance provenance) {
  Put(&t[0], t.size(), provenance);
}

void TokenSequence::Put(const std::string &s, Provenance provenance) {
  Put(s.data(), s.size(), provenance);
}

void TokenSequence::Put(const std::stringstream &ss, Provenance provenance) {
  Put(ss.str(), provenance);
}

void TokenSequence::EmitLowerCase(CookedSource *cooked) const {
  std::size_t tokens{start_.size()};
  std::size_t chars{char_.size()};
  std::size_t atToken{0};
  for (std::size_t j{0}; j < chars;) {
    std::size_t nextStart{atToken + 1 < tokens ? start_[++atToken] : chars};
    const char *p{&char_[j]}, *limit{&char_[nextStart]};
    j = nextStart;
    if (IsDecimalDigit(*p)) {
      while (p < limit && IsDecimalDigit(*p)) {
        cooked->Put(*p++);
      }
      if (p < limit && (*p == 'h' || *p == 'H')) {
        // Hollerith
        cooked->Put('h');
        cooked->Put(p + 1, limit - (p + 1));
      } else {
        // exponent
        while (p < limit) {
          cooked->Put(ToLowerCaseLetter(*p++));
        }
      }
    } else if (limit[-1] == '\'' || limit[-1] == '"') {
      if (*p == limit[-1]) {
        // Character literal without prefix
        cooked->Put(p, limit - p);
      } else if (p[1] == limit[-1]) {
        // BOZX-prefixed constant
        while (p < limit) {
          cooked->Put(ToLowerCaseLetter(*p++));
        }
      } else {
        // Kanji NC'...' character literal or literal with kind-param prefix.
        while (*p != limit[-1]) {
          cooked->Put(ToLowerCaseLetter(*p++));
        }
        cooked->Put(p, limit - p);
      }
    } else {
      while (p < limit) {
        cooked->Put(ToLowerCaseLetter(*p++));
      }
    }
  }
  cooked->PutProvenanceMappings(provenances_);
}

std::string TokenSequence::ToString() const {
  return {&char_[0], char_.size()};
}

Provenance TokenSequence::GetTokenProvenance(
    std::size_t token, std::size_t offset) const {
  ProvenanceRange range{provenances_.Map(start_[token] + offset)};
  return range.start();
}

ProvenanceRange TokenSequence::GetTokenProvenanceRange(
    std::size_t token, std::size_t offset) const {
  ProvenanceRange range{provenances_.Map(start_[token] + offset)};
  return range.Prefix(TokenBytes(token) - offset);
}

ProvenanceRange TokenSequence::GetIntervalProvenanceRange(
    std::size_t token, std::size_t tokens) const {
  if (tokens == 0) {
    return {};
  }
  ProvenanceRange range{provenances_.Map(start_[token])};
  while (--tokens > 0 &&
      range.AnnexIfPredecessor(provenances_.Map(start_[++token]))) {
  }
  return range;
}

ProvenanceRange TokenSequence::GetProvenanceRange() const {
  return GetIntervalProvenanceRange(0, start_.size());
}
}  // namespace parser
}  // namespace Fortran
