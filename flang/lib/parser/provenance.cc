#include "provenance.h"
#include "idioms.h"
#include <utility>

namespace Fortran {
namespace parser {

void OffsetToProvenanceMappings::clear() { provenanceMap_.clear(); }

std::size_t OffsetToProvenanceMappings::size() const {
  if (provenanceMap_.empty()) {
    return 0;
  }
  const ContiguousProvenanceMapping &last{provenanceMap_.back()};
  return last.start + last.range.size();
}

void OffsetToProvenanceMappings::Put(ProvenanceRange range) {
  if (provenanceMap_.empty()) {
    provenanceMap_.push_back({0, range});
  } else {
    ContiguousProvenanceMapping &last{provenanceMap_.back()};
    if (!last.range.AnnexIfPredecessor(range)) {
      provenanceMap_.push_back({last.start + last.range.size(), range});
    }
  }
}

void OffsetToProvenanceMappings::Put(const OffsetToProvenanceMappings &that) {
  for (const auto &map : that.provenanceMap_) {
    Put(map.range);
  }
}

ProvenanceRange OffsetToProvenanceMappings::Map(std::size_t at) const {
  //  CHECK(!provenanceMap_.empty());
  std::size_t low{0}, count{provenanceMap_.size()};
  while (count > 1) {
    std::size_t mid{low + (count >> 1)};
    if (provenanceMap_[mid].start > at) {
      count = mid - low;
    } else {
      count -= mid - low;
      low = mid;
    }
  }
  std::size_t offset{at - provenanceMap_[low].start};
  return provenanceMap_[low].range.Suffix(offset);
}

void OffsetToProvenanceMappings::RemoveLastBytes(std::size_t bytes) {
  for (; bytes > 0; provenanceMap_.pop_back()) {
    CHECK(!provenanceMap_.empty());
    ContiguousProvenanceMapping &last{provenanceMap_.back()};
    std::size_t chunk{last.range.size()};
    if (bytes < chunk) {
      last.range = last.range.Prefix(chunk - bytes);
      break;
    }
    bytes -= chunk;
  }
}

AllSources::AllSources() : range_{1, 1} {
  // Start the origin_ array with a dummy that has a forced provenance,
  // so that provenance offset 0 remains reserved as an uninitialized
  // value.
  origin_.emplace_back(range_, std::string{'?'});
}

AllSources::~AllSources() {}

const char &AllSources::operator[](Provenance at) const {
  const Origin &origin{MapToOrigin(at)};
  return origin[origin.covers.MemberOffset(at)];
}

void AllSources::PushSearchPathDirectory(std::string directory) {
  // gfortran and ifort append to current path, PGI prepends
  searchPath_.push_back(directory);
}

std::string AllSources::PopSearchPathDirectory() {
  std::string directory{searchPath_.back()};
  searchPath_.pop_back();
  return directory;
}

const SourceFile *AllSources::Open(std::string path, std::stringstream *error) {
  std::unique_ptr<SourceFile> source{std::make_unique<SourceFile>()};
  if (source->Open(LocateSourceFile(path, searchPath_), error)) {
    return ownedSourceFiles_.emplace_back(std::move(source)).get();
  }
  return nullptr;
}

const SourceFile *AllSources::ReadStandardInput(std::stringstream *error) {
  std::unique_ptr<SourceFile> source{std::make_unique<SourceFile>()};
  if (source->ReadStandardInput(error)) {
    return ownedSourceFiles_.emplace_back(std::move(source)).get();
  }
  return nullptr;
}

ProvenanceRange AllSources::AddIncludedFile(
    const SourceFile &source, ProvenanceRange from, bool isModule) {
  ProvenanceRange covers{range_.NextAfter(), source.bytes()};
  CHECK(range_.AnnexIfPredecessor(covers));
  CHECK(origin_.back().covers.ImmediatelyPrecedes(covers));
  origin_.emplace_back(covers, source, from, isModule);
  return covers;
}

ProvenanceRange AllSources::AddMacroCall(
    ProvenanceRange def, ProvenanceRange use, const std::string &expansion) {
  ProvenanceRange covers{range_.NextAfter(), expansion.size()};
  CHECK(range_.AnnexIfPredecessor(covers));
  CHECK(origin_.back().covers.ImmediatelyPrecedes(covers));
  origin_.emplace_back(covers, def, use, expansion);
  return covers;
}

ProvenanceRange AllSources::AddCompilerInsertion(std::string text) {
  ProvenanceRange covers{range_.NextAfter(), text.size()};
  CHECK(range_.AnnexIfPredecessor(covers));
  CHECK(origin_.back().covers.ImmediatelyPrecedes(covers));
  origin_.emplace_back(covers, text);
  return covers;
}

void AllSources::Identify(std::ostream &o, Provenance at,
    const std::string &prefix, bool echoSourceLine) const {
  CHECK(IsValid(at));
  static const std::string indented{prefix + "  "};
  const Origin &origin{MapToOrigin(at)};
  std::visit(
      visitors{
          [&](const Inclusion &inc) {
            std::size_t offset{origin.covers.MemberOffset(at)};
            std::pair<int, int> pos{inc.source.FindOffsetLineAndColumn(offset)};
            o << prefix << "at line " << pos.first << ", column " << pos.second;
            if (echoSourceLine) {
              o << ":\n" << indented << "  ";
              const char *text{inc.source.content() +
                  inc.source.GetLineStartOffset(pos.first)};
              for (const char *p{text}; *p != '\n'; ++p) {
                o << *p;
              }
              o << '\n' << indented << "  ";
              for (int j{1}; j < pos.second; ++j) {
                char ch{text[j - 1]};
                o << (ch == '\t' ? '\t' : ' ');
              }
              o << "^\n" << prefix;
            } else {
              o << ' ';
            }
            o << "in the " << (inc.isModule ? "module " : "file ")
              << inc.source.path();
            if (IsValid(origin.replaces)) {
              o << (inc.isModule ? " used\n" : " included\n");
              Identify(o, origin.replaces.start(), indented);
            } else {
              o << '\n';
            }
          },
          [&](const Macro &mac) {
            o << prefix << "in the expansion of a macro that was defined\n";
            Identify(o, mac.definition.start(), indented, echoSourceLine);
            o << prefix << "and called\n";
            Identify(o, origin.replaces.start(), indented, echoSourceLine);
            if (echoSourceLine) {
              o << prefix << "and expanded to\n"
                << indented << "  " << mac.expansion << '\n'
                << indented << "  ";
              for (std::size_t j{0}; origin.covers.OffsetMember(j) < at; ++j) {
                o << (mac.expansion[j] == '\t' ? '\t' : ' ');
              }
              o << "^\n";
            }
          },
          [&](const CompilerInsertion &ins) {
            o << prefix << ins.text << '\n';
          }},
      origin.u);
}

const SourceFile *AllSources::GetSourceFile(
    Provenance at, std::size_t *offset) const {
  const Origin &origin{MapToOrigin(at)};
  return std::visit(visitors{[&](const Inclusion &inc) {
                               if (offset != nullptr) {
                                 *offset = origin.covers.MemberOffset(at);
                               }
                               return &inc.source;
                             },
                        [&](const Macro &mac) {
                          return GetSourceFile(origin.replaces.start(), offset);
                        },
                        [offset](const CompilerInsertion &) {
                          if (offset != nullptr) {
                            *offset = 0;
                          }
                          return static_cast<const SourceFile *>(nullptr);
                        }},
      origin.u);
}

ProvenanceRange AllSources::GetContiguousRangeAround(
    ProvenanceRange range) const {
  CHECK(IsValid(range));
  const Origin &origin{MapToOrigin(range.start())};
  CHECK(origin.covers.Contains(range));
  return origin.covers;
}

std::string AllSources::GetPath(Provenance at) const {
  const SourceFile *source{GetSourceFile(at)};
  return source ? source->path() : ""s;
}

int AllSources::GetLineNumber(Provenance at) const {
  std::size_t offset{0};
  const SourceFile *source{GetSourceFile(at, &offset)};
  return source ? source->FindOffsetLineAndColumn(offset).first : 0;
}

Provenance AllSources::CompilerInsertionProvenance(char ch) {
  auto iter = compilerInsertionProvenance_.find(ch);
  if (iter != compilerInsertionProvenance_.end()) {
    return iter->second;
  }
  ProvenanceRange newCharRange{AddCompilerInsertion(std::string{ch})};
  Provenance newCharProvenance{newCharRange.start()};
  compilerInsertionProvenance_.insert(std::make_pair(ch, newCharProvenance));
  return newCharProvenance;
}

AllSources::Origin::Origin(ProvenanceRange r, const SourceFile &source)
  : u{Inclusion{source}}, covers{r} {}
AllSources::Origin::Origin(ProvenanceRange r, const SourceFile &included,
    ProvenanceRange from, bool isModule)
  : u{Inclusion{included, isModule}}, covers{r}, replaces{from} {}
AllSources::Origin::Origin(ProvenanceRange r, ProvenanceRange def,
    ProvenanceRange use, const std::string &expansion)
  : u{Macro{def, expansion}}, covers{r}, replaces{use} {}
AllSources::Origin::Origin(ProvenanceRange r, const std::string &text)
  : u{CompilerInsertion{text}}, covers{r} {}

const char &AllSources::Origin::operator[](std::size_t n) const {
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
  CHECK(range_.Contains(at));
  std::size_t low{0}, count{origin_.size()};
  while (count > 1) {
    std::size_t mid{low + (count >> 1)};
    if (at < origin_[mid].covers.start()) {
      count = mid - low;
    } else {
      count -= mid - low;
      low = mid;
    }
  }
  CHECK(origin_[low].covers.Contains(at));
  return origin_[low];
}

ProvenanceRange CookedSource::GetProvenance(const char *at) const {
  return provenanceMap_.Map(at - &data_[0]);
}

void CookedSource::Marshal() {
  CHECK(provenanceMap_.size() == buffer_.size());
  provenanceMap_.Put(allSources_.AddCompilerInsertion("(after end of source)"));
  data_.resize(buffer_.size());
  char *p{&data_[0]};
  for (char ch : buffer_) {
    *p++ = ch;
  }
  buffer_.clear();
}

static void DumpRange(std::ostream &o, const ProvenanceRange &r) {
  o << "[" << r.start().offset() << ".." << r.Last().offset() << "] ("
    << r.size() << " bytes)";
}

void OffsetToProvenanceMappings::Dump(std::ostream &o) const {
  for (const ContiguousProvenanceMapping &m : provenanceMap_) {
    std::size_t n{m.range.size()};
    o << "offsets [" << m.start << ".." << (m.start + n - 1)
      << "] -> provenances ";
    DumpRange(o, m.range);
    o << '\n';
  }
}

void AllSources::Dump(std::ostream &o) const {
  o << "AllSources range_ ";
  DumpRange(o, range_);
  o << '\n';
  for (const Origin &m : origin_) {
    o << "   ";
    DumpRange(o, m.covers);
    o << " -> ";
    std::visit(visitors{[&](const Inclusion &inc) {
                          if (inc.isModule) {
                            o << "module ";
                          }
                          o << "file " << inc.source.path();
                        },
                   [&](const Macro &mac) { o << "macro " << mac.expansion; },
                   [&](const CompilerInsertion &ins) {
                     o << "compiler '" << ins.text << '\'';
                     if (ins.text.length() == 1) {
                       int ch = ins.text[0];
                       o << " (0x" << std::hex << (ch & 0xff) << std::dec
                         << ")";
                     }
                   }},
        m.u);
    o << '\n';
  }
}

void CookedSource::Dump(std::ostream &o) const {
  o << "CookedSource:\n";
  allSources_.Dump(o);
  o << "CookedSource::provenanceMap_:\n";
  provenanceMap_.Dump(o);
}
}  // namespace parser
}  // namespace Fortran
