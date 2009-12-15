//===--- GlobalSelector.h - Cross-translation-unit "token" for selectors --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  GlobalSelector is a ASTContext-independent way to refer to selectors.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_INDEX_GLOBALSELECTOR_H
#define LLVM_CLANG_INDEX_GLOBALSELECTOR_H

#include "llvm/ADT/DenseMap.h"
#include <string>

namespace clang {
  class ASTContext;
  class Selector;

namespace idx {
  class Program;

/// \brief A ASTContext-independent way to refer to selectors.
class GlobalSelector {
  void *Val;

  explicit GlobalSelector(void *val) : Val(val) { }

public:
  GlobalSelector() : Val(0) { }

  /// \brief Get the ASTContext-specific selector.
  Selector getSelector(ASTContext &AST) const;

  bool isValid() const { return Val != 0; }
  bool isInvalid() const { return !isValid(); }

  /// \brief Get a printable name for debugging purpose.
  std::string getPrintableName() const;

  /// \brief Get a GlobalSelector for the ASTContext-specific selector.
  static GlobalSelector get(Selector Sel, Program &Prog);

  void *getAsOpaquePtr() const { return Val; }

  static GlobalSelector getFromOpaquePtr(void *Ptr) {
    return GlobalSelector(Ptr);
  }

  friend bool operator==(const GlobalSelector &LHS, const GlobalSelector &RHS) {
    return LHS.getAsOpaquePtr() == RHS.getAsOpaquePtr();
  }

  // For use in a std::map.
  friend bool operator< (const GlobalSelector &LHS, const GlobalSelector &RHS) {
    return LHS.getAsOpaquePtr() < RHS.getAsOpaquePtr();
  }

  // For use in DenseMap/DenseSet.
  static GlobalSelector getEmptyMarker() { return GlobalSelector((void*)-1); }
  static GlobalSelector getTombstoneMarker() {
    return GlobalSelector((void*)-2);
  }
};

} // namespace idx

} // namespace clang

namespace llvm {
/// Define DenseMapInfo so that GlobalSelectors can be used as keys in DenseMap
/// and DenseSets.
template<>
struct DenseMapInfo<clang::idx::GlobalSelector> {
  static inline clang::idx::GlobalSelector getEmptyKey() {
    return clang::idx::GlobalSelector::getEmptyMarker();
  }

  static inline clang::idx::GlobalSelector getTombstoneKey() {
    return clang::idx::GlobalSelector::getTombstoneMarker();
  }

  static unsigned getHashValue(clang::idx::GlobalSelector);

  static inline bool
  isEqual(clang::idx::GlobalSelector LHS, clang::idx::GlobalSelector RHS) {
    return LHS == RHS;
  }
};
  
template <>
struct isPodLike<clang::idx::GlobalSelector> { static const bool value = true;};

}  // end namespace llvm

#endif
