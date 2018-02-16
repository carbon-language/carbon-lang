#ifndef FORTRAN_PARSER_PROVENANCE_H_
#define FORTRAN_PARSER_PROVENANCE_H_

#include "char-buffer.h"
#include "idioms.h"
#include "source.h"
#include <map>
#include <memory>
#include <ostream>
#include <sstream>
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
// Provenances are offsets into an (unmaterialized) marshaling of the
// entire contents of all the original source files, include files, macro
// expansions, &c. for each visit to each source.  These origins of the
// original source characters constitute a forest whose roots are
// the original source files named on the compiler's command line.
// Given a Provenance, we can find the tree node that contains it in time
// O(log(# of origins)), and describe the position precisely by walking
// up the tree.  (It would be possible, via a time/space trade-off, to
// cap the time by the use of an intermediate table that would be indexed
// by the upper bits of an offset, but that does not appear to be
// necessary.)

class Provenance {
public:
  Provenance() {}
  Provenance(size_t offset) : offset_{offset} { CHECK(offset > 0); }
  Provenance(const Provenance &that) = default;
  Provenance(Provenance &&that) = default;
  Provenance &operator=(const Provenance &that) = default;
  Provenance &operator=(Provenance &&that) = default;
  Provenance operator+(ptrdiff_t n) const {
    CHECK(n > -static_cast<ptrdiff_t>(offset_));
    return {offset_ + static_cast<size_t>(n)};
  }
  Provenance operator+(size_t n) const { return {offset_ + n}; }
  bool operator<(Provenance that) const { return offset_ < that.offset_; }
  bool operator<=(Provenance that) const { return !(that < *this); }
  bool operator==(Provenance that) const { return offset_ == that.offset_; }
  bool operator!=(Provenance that) const { return !(*this == that); }
  size_t offset() const { return offset_; }

private:
  size_t offset_{0};
};

class ProvenanceRange {
public:
  ProvenanceRange() {}
  ProvenanceRange(Provenance s, size_t n) : start_{s}, bytes_{n} {
    CHECK(n > 0);
  }
  ProvenanceRange(const ProvenanceRange &) = default;
  ProvenanceRange(ProvenanceRange &&) = default;
  ProvenanceRange &operator=(const ProvenanceRange &) = default;
  ProvenanceRange &operator=(ProvenanceRange &&) = default;

  bool operator==(ProvenanceRange that) const {
    return start_ == that.start_ && bytes_ == that.bytes_;
  }

  size_t size() const { return bytes_; }

  bool Contains(Provenance at) const {
    return start_ <= at && at < start_ + bytes_;
  }

  bool Contains(ProvenanceRange that) const {
    return Contains(that.start_) && Contains(that.start_ + (that.bytes_ - 1));
  }

  size_t ProvenanceToLocalOffset(Provenance at) const {
    CHECK(Contains(at));
    return at.offset() - start_.offset();
  }

  Provenance LocalOffsetToProvenance(size_t at) const {
    CHECK(at < bytes_);
    return start_ + at;
  }

  Provenance NextAfter() const { return start_ + bytes_; }

  ProvenanceRange Suffix(size_t at) const {
    CHECK(at < bytes_);
    return {start_ + at, bytes_ - at};
  }

  ProvenanceRange Prefix(size_t bytes) const {
    CHECK(bytes > 0);
    return {start_, std::min(bytes_, bytes)};
  }

  bool IsPredecessor(ProvenanceRange next) {
    return start_ + bytes_ == next.start_;
  }

  bool AnnexIfPredecessor(ProvenanceRange next) {
    if (IsPredecessor(next)) {
      bytes_ += next.bytes_;
      return true;
    }
    return false;
  }

  void Dump(std::ostream &) const;

private:
  Provenance start_;
  size_t bytes_{0};
};

// Maps 0-based local offsets in some contiguous range (e.g., a token
// sequence) to their provenances.  Lookup time is on the order of
// O(log(#of intervals with contiguous provenances)).  As mentioned
// above, this time could be capped via a time/space trade-off.
class OffsetToProvenanceMappings {
public:
  OffsetToProvenanceMappings() {}
  size_t size() const;
  void clear();
  void shrink_to_fit() { provenanceMap_.shrink_to_fit(); }
  void Put(ProvenanceRange);
  void Put(const OffsetToProvenanceMappings &);
  ProvenanceRange Map(size_t at) const;
  void RemoveLastBytes(size_t);
  void Dump(std::ostream &) const;

private:
  struct ContiguousProvenanceMapping {
    size_t start;
    ProvenanceRange range;
  };

  std::vector<ContiguousProvenanceMapping> provenanceMap_;
};

class AllSources {
public:
  AllSources();
  ~AllSources();

  size_t size() const { return range_.size(); }
  const char &operator[](Provenance) const;

  void PushSearchPathDirectory(std::string);
  std::string PopSearchPathDirectory();
  const SourceFile *Open(std::string path, std::stringstream *error);

  ProvenanceRange AddIncludedFile(
      const SourceFile &, ProvenanceRange, bool isModule = false);
  ProvenanceRange AddMacroCall(
      ProvenanceRange def, ProvenanceRange use, const std::string &expansion);
  ProvenanceRange AddCompilerInsertion(std::string);

  bool IsValid(Provenance at) const { return range_.Contains(at); }
  bool IsValid(ProvenanceRange range) const {
    return range.size() > 0 && range_.Contains(range);
  }
  void Identify(std::ostream &, Provenance, const std::string &prefix,
      bool echoSourceLine = false) const;
  const SourceFile *GetSourceFile(Provenance, size_t *offset = nullptr) const;
  ProvenanceRange GetContiguousRangeAround(ProvenanceRange) const;
  std::string GetPath(Provenance) const;  // __FILE__
  int GetLineNumber(Provenance) const;  // __LINE__
  Provenance CompilerInsertionProvenance(char ch) const;
  void Dump(std::ostream &) const;

private:
  struct Inclusion {
    const SourceFile &source;
    bool isModule;
  };
  struct Module {
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
    Origin(ProvenanceRange, const SourceFile &);
    Origin(ProvenanceRange, const SourceFile &, ProvenanceRange,
        bool isModule = false);
    Origin(ProvenanceRange, ProvenanceRange def, ProvenanceRange use,
        const std::string &expansion);
    Origin(ProvenanceRange, const std::string &);

    const char &operator[](size_t) const;

    std::variant<Inclusion, Macro, CompilerInsertion> u;
    ProvenanceRange covers, replaces;
  };

  const Origin &MapToOrigin(Provenance) const;

  std::vector<Origin> origin_;
  ProvenanceRange range_;
  std::map<char, Provenance> compilerInsertionProvenance_;
  std::vector<std::unique_ptr<SourceFile>> ownedSourceFiles_;
  std::vector<std::string> searchPath_;
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
  void Marshal();  // marshals all text into one contiguous block
  void Dump(std::ostream &) const;

private:
  AllSources *allSources_;
  CharBuffer buffer_;  // before Marshal()
  std::vector<char> data_;  // all of it, prescanned and preprocessed
  OffsetToProvenanceMappings provenanceMap_;
};
}  // namespace parser
}  // namespace Fortran
#endif  // FORTRAN_PARSER_PROVENANCE_H_
