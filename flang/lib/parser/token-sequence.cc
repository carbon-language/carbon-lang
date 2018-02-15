#include "token-sequence.h"

namespace Fortran {
namespace parser {

bool CharPointerWithLength::IsBlank() const {
  for (size_t j{0}; j < bytes_; ++j) {
    char ch{data_[j]};
    if (ch != ' ' && ch != '\t') {
      return false;
    }
  }
  return true;
}

void TokenSequence::clear() {
  start_.clear();
  nextStart_ = 0;
  char_.clear();
  provenances_.clear();
}

void TokenSequence::pop_back() {
  size_t bytes{nextStart_ - start_.back()};
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

void TokenSequence::Put(const TokenSequence &that, size_t at, size_t tokens) {
  ProvenanceRange provenance;
  size_t offset{0};
  for (; tokens-- > 0; ++at) {
    CharPointerWithLength tok{that[at]};
    size_t tokBytes{tok.size()};
    for (size_t j{0}; j < tokBytes; ++j) {
      if (offset == provenance.size()) {
        offset = 0;
        provenance = that.provenances_.Map(that.start_[at] + j);
      }
      PutNextTokenChar(tok[j], provenance.LocalOffsetToProvenance(offset++));
    }
    CloseToken();
  }
}

void TokenSequence::Put(const char *s, size_t bytes, Provenance provenance) {
  for (size_t j{0}; j < bytes; ++j) {
    PutNextTokenChar(s[j], provenance + j);
  }
  CloseToken();
}

void TokenSequence::Put(const CharPointerWithLength &t, Provenance provenance) {
  Put(&t[0], t.size(), provenance);
}
void TokenSequence::Put(const std::string &s, Provenance provenance) {
  Put(s.data(), s.size(), provenance);
}

void TokenSequence::Put(const std::stringstream &ss, Provenance provenance) {
  Put(ss.str(), provenance);
}

void TokenSequence::EmitWithCaseConversion(CookedSource *cooked) const {
  size_t tokens{start_.size()};
  size_t chars{char_.size()};
  size_t atToken{0};
  for (size_t j{0}; j < chars;) {
    size_t nextStart{atToken + 1 < tokens ? start_[++atToken] : chars};
    if (isalpha(char_[j])) {
      for (; j < nextStart; ++j) {
        cooked->Put(tolower(char_[j]));
      }
    } else {
      cooked->Put(&char_[j], nextStart - j);
      j = nextStart;
    }
  }
  cooked->PutProvenanceMappings(provenances_);
}

std::string TokenSequence::ToString() const {
  return {&char_[0], char_.size()};
}

Provenance TokenSequence::GetProvenance(size_t token, size_t offset) const {
  ProvenanceRange range{provenances_.Map(start_[token] + offset)};
  return range.LocalOffsetToProvenance(0);
}

ProvenanceRange TokenSequence::GetProvenanceRange(
    size_t token, size_t offset) const {
  ProvenanceRange range{provenances_.Map(start_[token] + offset)};
  return range.Prefix(TokenBytes(token) - offset);
}
}  // namespace parser
}  // namespace Fortran
