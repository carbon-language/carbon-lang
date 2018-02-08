#include "provenance.h"
#include "idioms.h"
#include "position.h"
#include <utility>
namespace Fortran {
namespace parser {

Origin::Origin(const SourceFile &source) : u_{Inclusion{source}} {}
Origin::Origin(const SourceFile &included, ProvenanceRange from)
  : u_{Inclusion{included}}, replaces_{from} {}
Origin::Origin(ProvenanceRange def, ProvenanceRange use,
    const std::string &expansion)
  : u_{Macro{def, expansion}}, replaces_{use} {}

size_t Origin::size() const {
  return std::visit(
      visitors{[](const Inclusion &inc) { return inc.source.bytes(); },
           [](const Macro &mac) { return mac.expansion.size(); }},
      u_);
}

const char &Origin::operator[](size_t n) const {
  return std::visit(
      visitors{[n](const Inclusion &inc) -> const char & {
          return inc.source.content()[n]; },
          [n](const Macro &mac) -> const char & { return mac.expansion[n]; }},
      u_);
}

void Origin::Identify(std::ostream &o, const AllOfTheSource &sources, size_t at,
                      const std::string &prefix) const {
  static const std::string indented{prefix + "  "};
  std::visit(
      visitors{
          [&](const Inclusion &inc) {
              Position pos{inc.source.FindOffsetPosition(at)};
              o << prefix << "at line " << pos.lineNumber() << ", column " <<
                   pos.column() << "in the file " << inc.source.path() << '\n';
              if (replaces_.bytes() > 0) {
                o << prefix << " that was included\n";
                sources.Identify(o, replaces_.start(), indented);
              }
          },
          [&](const Macro &mac) {
              o << prefix << "in the expansion of a macro that was defined\n";
              sources.Identify(o, mac.definition.start(), indented);
              o << prefix << "... and called\n";
              sources.Identify(o, replaces_.start(), indented);
              o << prefix << "... and expanded to\n" <<
                   indented << mac.expansion << '\n'; }},
      u_);
}

AllOfTheSource &AllOfTheSource::Add(Origin &&origin) {
  size_t start{bytes_};
  bytes_ += origin.size();
  chunk_.emplace_back(Chunk{std::move(origin), start});
  return *this;
}

const char &AllOfTheSource::operator[](Provenance at) const {
  const Chunk &chunk{MapToChunk(at)};
  return chunk.origin[at - chunk.start];
}

const AllOfTheSource::Chunk &AllOfTheSource::MapToChunk(Provenance at) const {
  CHECK(at < bytes_);
  size_t low{0}, count{chunk_.size()};
  while (count > 1) {
    size_t mid{low + (count >> 1)};
    if (chunk_[mid].start > at) {
      count = mid - low;
    } else {
      count -= mid - low;
      low = mid;
    }
  }
  return chunk_[low];
}

void AllOfTheSource::Identify(std::ostream &o, Provenance at,
                              const std::string &prefix) const {
  const Chunk &chunk{MapToChunk(at)};
  return chunk.origin.Identify(o, *this, at - chunk.start, prefix);
}
}  // namespace parser
}  // namespace Fortran
