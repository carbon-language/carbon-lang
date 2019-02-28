//===--- Index.h -------------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_INDEX_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_INDEX_H

#include "ExpectedTypes.h"
#include "Symbol.h"
#include "SymbolID.h"
#include "SymbolLocation.h"
#include "clang/Index/IndexSymbol.h"
#include "clang/Lex/Lexer.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/JSON.h"
#include <array>
#include <limits>
#include <mutex>
#include <string>
#include <tuple>

namespace clang {
namespace clangd {

// Describes the kind of a cross-reference.
//
// This is a bitfield which can be combined from different kinds.
enum class RefKind : uint8_t {
  Unknown = 0,
  Declaration = static_cast<uint8_t>(index::SymbolRole::Declaration),
  Definition = static_cast<uint8_t>(index::SymbolRole::Definition),
  Reference = static_cast<uint8_t>(index::SymbolRole::Reference),
  All = Declaration | Definition | Reference,
};
inline RefKind operator|(RefKind L, RefKind R) {
  return static_cast<RefKind>(static_cast<uint8_t>(L) |
                              static_cast<uint8_t>(R));
}
inline RefKind &operator|=(RefKind &L, RefKind R) { return L = L | R; }
inline RefKind operator&(RefKind A, RefKind B) {
  return static_cast<RefKind>(static_cast<uint8_t>(A) &
                              static_cast<uint8_t>(B));
}
llvm::raw_ostream &operator<<(llvm::raw_ostream &, RefKind);

// Represents a symbol occurrence in the source file.
// Despite the name, it could be a declaration/definition/reference.
//
// WARNING: Location does not own the underlying data - Copies are shallow.
struct Ref {
  // The source location where the symbol is named.
  SymbolLocation Location;
  RefKind Kind = RefKind::Unknown;
};
inline bool operator<(const Ref &L, const Ref &R) {
  return std::tie(L.Location, L.Kind) < std::tie(R.Location, R.Kind);
}
inline bool operator==(const Ref &L, const Ref &R) {
  return std::tie(L.Location, L.Kind) == std::tie(R.Location, R.Kind);
}
llvm::raw_ostream &operator<<(llvm::raw_ostream &, const Ref &);

// An efficient structure of storing large set of symbol references in memory.
// Filenames are deduplicated.
class RefSlab {
public:
  using value_type = std::pair<SymbolID, llvm::ArrayRef<Ref>>;
  using const_iterator = std::vector<value_type>::const_iterator;
  using iterator = const_iterator;

  RefSlab() = default;
  RefSlab(RefSlab &&Slab) = default;
  RefSlab &operator=(RefSlab &&RHS) = default;

  const_iterator begin() const { return Refs.begin(); }
  const_iterator end() const { return Refs.end(); }
  /// Gets the number of symbols.
  size_t size() const { return Refs.size(); }
  size_t numRefs() const { return NumRefs; }
  bool empty() const { return Refs.empty(); }

  size_t bytes() const {
    return sizeof(*this) + Arena.getTotalMemory() +
           sizeof(value_type) * Refs.size();
  }

  // RefSlab::Builder is a mutable container that can 'freeze' to RefSlab.
  class Builder {
  public:
    Builder() : UniqueStrings(Arena) {}
    // Adds a ref to the slab. Deep copy: Strings will be owned by the slab.
    void insert(const SymbolID &ID, const Ref &S);
    // Consumes the builder to finalize the slab.
    RefSlab build() &&;

  private:
    llvm::BumpPtrAllocator Arena;
    llvm::UniqueStringSaver UniqueStrings; // Contents on the arena.
    llvm::DenseMap<SymbolID, std::vector<Ref>> Refs;
  };

private:
  RefSlab(std::vector<value_type> Refs, llvm::BumpPtrAllocator Arena,
          size_t NumRefs)
      : Arena(std::move(Arena)), Refs(std::move(Refs)), NumRefs(NumRefs) {}

  llvm::BumpPtrAllocator Arena;
  std::vector<value_type> Refs;
  // Number of all references.
  size_t NumRefs = 0;
};

struct FuzzyFindRequest {
  /// \brief A query string for the fuzzy find. This is matched against symbols'
  /// un-qualified identifiers and should not contain qualifiers like "::".
  std::string Query;
  /// \brief If this is non-empty, symbols must be in at least one of the scopes
  /// (e.g. namespaces) excluding nested scopes. For example, if a scope "xyz::"
  /// is provided, the matched symbols must be defined in namespace xyz but not
  /// namespace xyz::abc.
  ///
  /// The global scope is "", a top level scope is "foo::", etc.
  std::vector<std::string> Scopes;
  /// If set to true, allow symbols from any scope. Scopes explicitly listed
  /// above will be ranked higher.
  bool AnyScope = false;
  /// \brief The number of top candidates to return. The index may choose to
  /// return more than this, e.g. if it doesn't know which candidates are best.
  llvm::Optional<uint32_t> Limit;
  /// If set to true, only symbols for completion support will be considered.
  bool RestrictForCodeCompletion = false;
  /// Contextually relevant files (e.g. the file we're code-completing in).
  /// Paths should be absolute.
  std::vector<std::string> ProximityPaths;
  /// Preferred types of symbols. These are raw representation of `OpaqueType`.
  std::vector<std::string> PreferredTypes;

  bool operator==(const FuzzyFindRequest &Req) const {
    return std::tie(Query, Scopes, Limit, RestrictForCodeCompletion,
                    ProximityPaths, PreferredTypes) ==
           std::tie(Req.Query, Req.Scopes, Req.Limit,
                    Req.RestrictForCodeCompletion, Req.ProximityPaths,
                    Req.PreferredTypes);
  }
  bool operator!=(const FuzzyFindRequest &Req) const { return !(*this == Req); }
};
bool fromJSON(const llvm::json::Value &Value, FuzzyFindRequest &Request);
llvm::json::Value toJSON(const FuzzyFindRequest &Request);

struct LookupRequest {
  llvm::DenseSet<SymbolID> IDs;
};

struct RefsRequest {
  llvm::DenseSet<SymbolID> IDs;
  RefKind Filter = RefKind::All;
  /// If set, limit the number of refers returned from the index. The index may
  /// choose to return less than this, e.g. it tries to avoid returning stale
  /// results.
  llvm::Optional<uint32_t> Limit;
};

/// Interface for symbol indexes that can be used for searching or
/// matching symbols among a set of symbols based on names or unique IDs.
class SymbolIndex {
public:
  virtual ~SymbolIndex() = default;

  /// \brief Matches symbols in the index fuzzily and applies \p Callback on
  /// each matched symbol before returning.
  /// If returned Symbols are used outside Callback, they must be deep-copied!
  ///
  /// Returns true if there may be more results (limited by Req.Limit).
  virtual bool
  fuzzyFind(const FuzzyFindRequest &Req,
            llvm::function_ref<void(const Symbol &)> Callback) const = 0;

  /// Looks up symbols with any of the given symbol IDs and applies \p Callback
  /// on each matched symbol.
  /// The returned symbol must be deep-copied if it's used outside Callback.
  virtual void
  lookup(const LookupRequest &Req,
         llvm::function_ref<void(const Symbol &)> Callback) const = 0;

  /// Finds all occurrences (e.g. references, declarations, definitions) of a
  /// symbol and applies \p Callback on each result.
  ///
  /// Results should be returned in arbitrary order.
  /// The returned result must be deep-copied if it's used outside Callback.
  virtual void refs(const RefsRequest &Req,
                    llvm::function_ref<void(const Ref &)> Callback) const = 0;

  /// Returns estimated size of index (in bytes).
  // FIXME(kbobyrev): Currently, this only returns the size of index itself
  // excluding the size of actual symbol slab index refers to. We should include
  // both.
  virtual size_t estimateMemoryUsage() const = 0;
};

// Delegating implementation of SymbolIndex whose delegate can be swapped out.
class SwapIndex : public SymbolIndex {
public:
  // If an index is not provided, reset() must be called.
  SwapIndex(std::unique_ptr<SymbolIndex> Index = nullptr)
      : Index(std::move(Index)) {}
  void reset(std::unique_ptr<SymbolIndex>);

  // SymbolIndex methods delegate to the current index, which is kept alive
  // until the call returns (even if reset() is called).
  bool fuzzyFind(const FuzzyFindRequest &,
                 llvm::function_ref<void(const Symbol &)>) const override;
  void lookup(const LookupRequest &,
              llvm::function_ref<void(const Symbol &)>) const override;
  void refs(const RefsRequest &,
            llvm::function_ref<void(const Ref &)>) const override;
  size_t estimateMemoryUsage() const override;

private:
  std::shared_ptr<SymbolIndex> snapshot() const;
  mutable std::mutex Mutex;
  std::shared_ptr<SymbolIndex> Index;
};

} // namespace clangd
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_INDEX_H
