#include "provenance.h"
#include "idioms.h"
#include <utility>
namespace Fortran {
namespace parser {

void OffsetToProvenanceMappings::clear() {
  bytes_ = 0;
  provenanceMap_.clear();
}

void OffsetToProvenanceMappings::Put(ProvenanceRange range) {
  if (provenanceMap_.empty()) {
    provenanceMap_.push_back({bytes_, range});
  } else {
    ContiguousProvenanceMapping &last{provenanceMap_.back()};
    if (range.start == last.range.start + last.range.bytes) {
      last.range.bytes += range.bytes;
    } else {
      provenanceMap_.push_back({bytes_, range});
    }
  }
  bytes_ += range.bytes;
}

void OffsetToProvenanceMappings::Put(const OffsetToProvenanceMappings &that) {
  for (const auto &map : that.provenanceMap_) {
    Put(map.range);
  }
}

ProvenanceRange OffsetToProvenanceMappings::Map(size_t at) const {
  CHECK(at < bytes_);
  size_t low{0}, count{provenanceMap_.size()};
  while (count > 1) {
    size_t mid{low + (count >> 1)};
    if (provenanceMap_[mid].start > at) {
      count = mid - low;
    } else {
      count -= mid - low;
      low = mid;
    }
  }
  size_t offset{at - provenanceMap_[low].start};
  return {provenanceMap_[low].start + offset,
      provenanceMap_[low].range.bytes - offset};
}

void OffsetToProvenanceMappings::RemoveLastBytes(size_t bytes) {
  for (; bytes > 0; provenanceMap_.pop_back()) {
    if (provenanceMap_.empty()) {
      break;
    }
    ContiguousProvenanceMapping &last{provenanceMap_.back()};
    if (bytes < last.range.bytes) {
      last.range.bytes -= bytes;
      break;
    }
    bytes -= last.range.bytes;
  }
}

AllSources::AllSources(const SourceFile &initialSourceFile) {
  AddIncludedFile(initialSourceFile, ProvenanceRange{});
}

const char &AllSources::operator[](Provenance at) const {
  const Origin &origin{MapToOrigin(at)};
  return origin[at - origin.start];
}

ProvenanceRange AllSources::AddIncludedFile(
    const SourceFile &source, ProvenanceRange from) {
  size_t start{bytes_}, bytes{source.bytes()};
  bytes_ += bytes;
  origin_.emplace_back(start, source, from);
  return {start, bytes};
}

ProvenanceRange AllSources::AddMacroCall(
    ProvenanceRange def, ProvenanceRange use, const std::string &expansion) {
  size_t start{bytes_}, bytes{expansion.size()};
  bytes_ += bytes;
  origin_.emplace_back(start, def, use, expansion);
  return {start, bytes};
}

ProvenanceRange AllSources::AddCompilerInsertion(const std::string &text) {
  size_t start{bytes_}, bytes{text.size()};
  bytes_ += bytes;
  origin_.emplace_back(start, text);
  return {start, bytes};
}

void AllSources::Identify(
    std::ostream &o, Provenance at, const std::string &prefix) const {
  static const std::string indented{prefix + "  "};
  const Origin &origin{MapToOrigin(at)};
  std::visit(
      visitors{[&](const Inclusion &inc) {
                 std::pair<int, int> pos{
                     inc.source.FindOffsetLineAndColumn(at - origin.start)};
                 o << prefix << "at line " << pos.first << ", column "
                   << pos.second << " in the file " << inc.source.path()
                   << '\n';
                 if (origin.replaces.bytes > 0) {
                   o << prefix << " that was included\n";
                   Identify(o, origin.replaces.start, indented);
                 }
               },
          [&](const Macro &mac) {
            o << prefix << "in the expansion of a macro that was defined\n";
            Identify(o, mac.definition.start, indented);
            o << prefix << "... and called\n";
            Identify(o, origin.replaces.start, indented);
            o << prefix << "... and expanded to\n"
              << indented << mac.expansion << '\n';
          },
          [&](const CompilerInsertion &ins) {
            o << prefix << "in text '" << ins.text
              << "' inserted by the compiler\n";
          }},
      origin.u);
}

const SourceFile *AllSources::GetSourceFile(Provenance at) const {
  const Origin &origin{MapToOrigin(at)};
  return std::visit(visitors{[](const Inclusion &inc) { return &inc.source; },
                        [&origin, this](const Macro &mac) {
                          return GetSourceFile(origin.replaces.start);
                        },
                        [](const CompilerInsertion &) {
                          return static_cast<const SourceFile *>(nullptr);
                        }},
      origin.u);
}

std::string AllSources::GetPath(Provenance at) const {
  const SourceFile *source{GetSourceFile(at)};
  return source ? source->path() : ""s;
}

int AllSources::GetLineNumber(Provenance at) const {
  const SourceFile *source{GetSourceFile(at)};
  return source ? source->FindOffsetLineAndColumn(at).first : 0;
}

AllSources::Origin::Origin(size_t s, const SourceFile &source)
  : start{s}, u{Inclusion{source}} {}
AllSources::Origin::Origin(
    size_t s, const SourceFile &included, ProvenanceRange from)
  : start{s}, u{Inclusion{included}}, replaces{from} {}
AllSources::Origin::Origin(size_t s, ProvenanceRange def, ProvenanceRange use,
    const std::string &expansion)
  : start{s}, u{Macro{def, expansion}}, replaces{use} {}
AllSources::Origin::Origin(size_t s, const std::string &text)
  : start{s}, u{CompilerInsertion{text}} {}

size_t AllSources::Origin::size() const {
  return std::visit(
      visitors{[](const Inclusion &inc) { return inc.source.bytes(); },
          [](const Macro &mac) { return mac.expansion.size(); },
          [](const CompilerInsertion &ins) { return ins.text.size(); }},
      u);
}

const char &AllSources::Origin::operator[](size_t n) const {
  return std::visit(
      visitors{[n](const Inclusion &inc) -> const char & {
                 return inc.source.content()[n];
               },
          [n](const Macro &mac) -> const char & { return mac.expansion[n]; },
          [n](const CompilerInsertion &ins) -> const char & {
            return ins.text[n];
          }},
      u);
}

const AllSources::Origin &AllSources::MapToOrigin(Provenance at) const {
  CHECK(at < bytes_);
  size_t low{0}, count{origin_.size()};
  while (count > 1) {
    size_t mid{low + (count >> 1)};
    if (origin_[mid].start > at) {
      count = mid - low;
    } else {
      count -= mid - low;
      low = mid;
    }
  }
  CHECK(at >= origin_[low].start);
  CHECK(low + 1 == origin_.size() || at < origin_[low + 1].start);
  return origin_[low];
}

ProvenanceRange CookedSource::GetProvenance(const char *at) const {
  return provenanceMap_.Map(at - &data_[0]);
}

void CookedSource::Marshal() {
  CHECK(provenanceMap_.size() == buffer_.size());
  provenanceMap_.Put(sources_.AddCompilerInsertion("EOF"));
  data_.resize(buffer_.size());
  char *p{&data_[0]};
  for (char ch : buffer_) {
    *p++ = ch;
  }
  buffer_.clear();
}
}  // namespace parser
}  // namespace Fortran
