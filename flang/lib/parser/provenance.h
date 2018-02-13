#ifndef FORTRAN_PROVENANCE_H_
#define FORTRAN_PROVENANCE_H_
#include "char-buffer.h"
#include "source.h"
#include <ostream>
#include <string>
#include <utility>
#include <variant>
#include <vector>
namespace Fortran {
namespace parser {

// Each character in the contiguous source stream built by the
// prescanner corresponds to a particular character in a source file,
// include file, macro expansion, or compiler-inserted padding.
// The location of this original character to which a parsable character
// corresponds is its provenance.
//
// Provenances are offsets into an unmaterialized marshaling of all of the
// entire contents of the original source files, include files,  macro
// expansions, &c. for each visit to each source.  These origins of the
// original source characters constitute a forest whose roots are
// the original source files named on the compiler's command line.
// We can describe provenances precisely by walking up this tree.

using Provenance = size_t;

struct ProvenanceRange {
  ProvenanceRange() {}
  ProvenanceRange(Provenance s, size_t n) : start{s}, bytes{n} {}
  ProvenanceRange(const ProvenanceRange &) = default;
  ProvenanceRange(ProvenanceRange &&) = default;
  ProvenanceRange &operator=(const ProvenanceRange &) = default;
  ProvenanceRange &operator=(ProvenanceRange &&) = default;
  Provenance start{0};
  size_t bytes{0};
};

class OffsetToProvenanceMappings {
public:
  OffsetToProvenanceMappings() {}
  size_t size() const { return bytes_; }
  void clear();
  void shrink_to_fit() { provenanceMap_.shrink_to_fit(); }
  void Put(ProvenanceRange);
  void Put(const OffsetToProvenanceMappings &);
  ProvenanceRange Map(size_t at) const;
  void RemoveLastBytes(size_t);

private:
  struct ContiguousProvenanceMapping {
    size_t start;
    ProvenanceRange range;
  };

  size_t bytes_{0};
  std::vector<ContiguousProvenanceMapping> provenanceMap_;
};

class AllSources {
public:
  explicit AllSources(const SourceFile &initialSourceFile);

  size_t size() const { return bytes_; }
  const char &operator[](Provenance) const;

  ProvenanceRange AddIncludedFile(const SourceFile &, ProvenanceRange);
  ProvenanceRange AddMacroCall(
      ProvenanceRange def, ProvenanceRange use, const std::string &expansion);
  ProvenanceRange AddCompilerInsertion(const std::string &);

  void Identify(std::ostream &, Provenance, const std::string &prefix) const;
  const SourceFile *GetSourceFile(Provenance) const;
  std::string GetPath(Provenance) const;  // __FILE__
  int GetLineNumber(Provenance) const;  // __LINE__

private:
  struct Inclusion {
    const SourceFile &source;
  };
  struct Macro {
    ProvenanceRange definition;
    std::string expansion;
  };
  struct CompilerInsertion {
    std::string text;
  };

  struct Origin {
    Origin(size_t start, const SourceFile &);
    Origin(size_t start, const SourceFile &, ProvenanceRange);
    Origin(size_t start, ProvenanceRange def, ProvenanceRange use,
        const std::string &expansion);
    Origin(size_t start, const std::string &);

    size_t size() const;
    const char &operator[](size_t) const;

    size_t start;
    std::variant<Inclusion, Macro, CompilerInsertion> u;
    ProvenanceRange replaces;
  };

  const Origin &MapToOrigin(Provenance) const;

  std::vector<Origin> origin_;
  size_t bytes_{0};
};

class CookedSource {
public:
  explicit CookedSource(AllSources *sources) : allSources_{sources} {}

  size_t size() const { return data_.size(); }
  const char &operator[](size_t n) const { return data_[n]; }
  const char &at(size_t n) const { return data_.at(n); }

  AllSources *allSources() const { return allSources_; }

  ProvenanceRange GetProvenance(const char *) const;
  void Identify(std::ostream &, const char *) const;

  void Put(const char *data, size_t bytes) { buffer_.Put(data, bytes); }
  void Put(char ch) { buffer_.Put(&ch, 1); }
  void Put(char ch, Provenance p) {
    buffer_.Put(&ch, 1);
    provenanceMap_.Put(ProvenanceRange{p, 1});
  }
  void PutProvenanceMappings(const OffsetToProvenanceMappings &pm) {
    provenanceMap_.Put(pm);
  }
  void Marshal();  // marshalls all text into one contiguous block

private:
  AllSources *allSources_;
  CharBuffer buffer_;  // before Marshal()
  std::vector<char> data_;  // all of it, prescanned and preprocessed
  OffsetToProvenanceMappings provenanceMap_;
};
}  // namespace parser
}  // namespace Fortran
#endif  // FORTRAN_PROVENANCE_H_
