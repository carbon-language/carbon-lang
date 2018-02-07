#ifndef FORTRAN_PROVENANCE_H_
#define FORTRAN_PROVENANCE_H_
#include "source.h"
#include <string>
#include <variant>
namespace Fortran {

class SourceContexts;

using Provenance = size_t;

struct ProvenanceRange {
  Provenance begin;
  size_t bytes;
};

class Sources {
public:
  Sources() {}
  Sources(Sources &&) = default;
  Sources &operator(Sources &&) = default;
  size_t size() const { return bytes_; }
  char &operator[size_t at] const;
private:
  struct Context {
    struct Inclusion {
      const SourceFile &source;
    };
    struct MacroUse {
      ProvenanceRange definition;
    };

    int myIndex;  // *contexts[myIndex] == this;
    ProvenanceRange replaces;
    std::variant<SourceFile, Inclusion, MacroUse> v;
  };
  std::vector<std::unique_ptr<Context>> contexts_;
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
    iterator(const Sources &sources, Provenance at)
      : sources_{&sources}, at_{at} {}
    iterator(const iterator &that)
      : sources_{that.sources_}, at_{that.at_} {}
    iterator &operator(const iterator &that) {
      sources_ = that.sources_;
      at_ = that.at_;
      return *this;
    }
    const char &operator*() const;
    iterator &operator++() {
      ++at_;
      return *this;
    }
    iterator &operator++(int) {
      iterator result{*this};
      ++at_;
      return result;
    }
    bool operator<(const iterator &that) { return at_ < that.at_; }
    bool operator<=(const iterator &that) { return at_ <= that.at_; }
    bool operator==(const iterator &that) { return at_ == that.at_; }
    bool operator!=(const iterator &that) { return at_ != that.at_; }
  private:
    const Sources *sources_;
    size_t at_;
  };

  iterator begin(const Sources &sources) const {
    return iterator(sources, start_);
  }
  iterator end(const Sources &sources) const {
    return iterator(sources, start_ + bytes_);
  }
public:
  size_t size() const { return bytes_; }
private:
  Provenance start_;
  size_t bytes_;
};
}  // namespace Fortran
#endif  // FORTRAN_PROVENANCE_H_
