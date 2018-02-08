#ifndef FORTRAN_PROVENANCE_H_
#define FORTRAN_PROVENANCE_H_
#include "source.h"
#include <memory>
#include <ostream>
#include <string>
#include <variant>
namespace Fortran {
namespace parser {

using Provenance = size_t;

class ProvenanceRange {
public:
  ProvenanceRange() {}
  bool empty() const { return bytes_ == 0; }
  Provenance start() const { return start_; }
  size_t bytes() const { return bytes_; }
private:
  Provenance start_{0};
  size_t bytes_{0};
};

class AllOfTheSource;

class Origin {
public:
  explicit Origin(const SourceFile &);  // initial source file
  Origin(const SourceFile &, ProvenanceRange);  // included source file
  Origin(ProvenanceRange def, ProvenanceRange use,  // macro call
         const std::string &expansion);
  size_t size() const;
  const char &operator[](size_t) const;
  void Identify(std::ostream &, const AllOfTheSource &, size_t,
                const std::string &indent) const;
private:
  struct Inclusion {
    const SourceFile &source;
  };
  struct Macro {
    ProvenanceRange definition;
    std::string expansion;
  };
  std::variant<Inclusion, Macro> u_;
  ProvenanceRange replaces_;
};

class AllOfTheSource {
public:
  AllOfTheSource() {}
  AllOfTheSource(AllOfTheSource &&) = default;
  AllOfTheSource &operator=(AllOfTheSource &&) = default;
  size_t size() const { return bytes_; }
  const char &operator[](Provenance) const;
  AllOfTheSource &Add(Origin &&);
  void Identify(std::ostream &, Provenance, const std::string &prefix) const;
private:
  struct Chunk {
    Chunk(Origin &&origin, size_t at) : origin{std::move(origin)}, start{at} {}
    Origin origin;
    size_t start;
  };
  const Chunk &MapToChunk(Provenance) const;
  std::vector<Chunk> chunk_;
  size_t bytes_;
};

class ProvenancedChar {
public:
  using type = char;
  char character() const { return static_cast<char>(packed_); }
  Provenance provenance() const { return packed_ >> 8; }
private:
  size_t packed_;
};

class ProvenancedString {
private:
  class iterator {
  public:
    iterator(const AllOfTheSource &sources, Provenance at)
      : sources_{&sources}, at_{at} {}
    iterator(const iterator &that)
      : sources_{that.sources_}, at_{that.at_} {}
    iterator &operator=(const iterator &that) {
      sources_ = that.sources_;
      at_ = that.at_;
      return *this;
    }
    const char &operator*() const;
    iterator &operator++() {
      ++at_;
      return *this;
    }
    iterator operator++(int) {
      iterator result{*this};
      ++at_;
      return result;
    }
    bool operator<(const iterator &that) { return at_ < that.at_; }
    bool operator<=(const iterator &that) { return at_ <= that.at_; }
    bool operator==(const iterator &that) { return at_ == that.at_; }
    bool operator!=(const iterator &that) { return at_ != that.at_; }
  private:
    const AllOfTheSource *sources_;
    size_t at_;
  };

  iterator begin(const AllOfTheSource &sources) const {
    return iterator(sources, start_);
  }
  iterator end(const AllOfTheSource &sources) const {
    return iterator(sources, start_ + bytes_);
  }
public:
  size_t size() const { return bytes_; }
private:
  Provenance start_;
  size_t bytes_;
};
}  // namespace parser
}  // namespace Fortran
#endif  // FORTRAN_PROVENANCE_H_
