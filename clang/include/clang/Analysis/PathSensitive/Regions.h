//==- Regions.h - Abstract memory locations ------------------------*- C++ -*-//
//             
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines Region and its subclasses.  Regions represent abstract
//  memory locations.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/DataTypes.h"
#include "llvm/Support/Casting.h"
#include "llvm/ADT/FoldingSet.h"
#include "clang/Analysis/PathSensitive/SymbolManager.h"

#ifndef LLVM_CLANG_ANALYSIS_REGIONS_H
#define LLVM_CLANG_ANALYSIS_REGIONS_H

namespace llvm {
  class APSInt;
}

namespace clang {
  
class BasicValueFactory;

  
//===----------------------------------------------------------------------===//
// Region Extents.
//===----------------------------------------------------------------------===//

class RegionExtent {
public:
  enum Kind { Unknown = 0, Int = 0, Sym = 1 };

protected:
  const uintptr_t Raw;
  RegionExtent(uintptr_t raw, Kind k) : Raw(raw | k) {}
  uintptr_t getData() const { return Raw & ~0x1; }
  
public:
  // Folding-set profiling.
  void Profile(llvm::FoldingSetNodeID& ID) const {
    ID.AddPointer((void*) Raw);
  }  
  // Comparing extents.
  bool operator==(const RegionExtent& R) const {
    return Raw == R.Raw;
  }
  bool operator!=(const RegionExtent& R) const {
    return Raw != R.Raw;
  }  
  // Implement isa<T> support.
  Kind getKind() const { return Kind(Raw & 0x1); }
  uintptr_t getRaw() const { return Raw; }
  
  static inline bool classof(const RegionExtent*) {
    return true;
  }
};

class UnknownExtent : public RegionExtent {
public:
  UnknownExtent() : RegionExtent(0,Unknown) {}
  
  // Implement isa<T> support.
  static inline bool classof(const RegionExtent* E) {
    return E->getRaw() == 0;
  }  
};
  
class IntExtent : public RegionExtent {
public:
  IntExtent(const llvm::APSInt& X) : RegionExtent((uintptr_t) &X, Int) {}
  
  const llvm::APSInt& getInt() const {
    return *((llvm::APSInt*) getData());
  }
  
  // Implement isa<T> support.
  static inline bool classof(const RegionExtent* E) {
    return E->getKind() == Int && E->getRaw() != 0;
  }
};

class SymExtent : public RegionExtent {
public:
  SymExtent(SymbolID S) : RegionExtent(S.getNumber() << 1, Sym) {}
  
  SymbolID getSymbol() const { return SymbolID(getData() >> 1); }
  
  // Implement isa<T> support.
  static inline bool classof(const RegionExtent* E) {
    return E->getKind() == Sym;
  }  
};

//===----------------------------------------------------------------------===//
// Regions.
//===----------------------------------------------------------------------===//
  
class Region {
public:
  enum Kind { Var = 0x0, Anon = 0x1 };
  
private:
  uintptr_t Raw;
  
protected:
  Region(const void* data, Kind kind)
    : Raw((uintptr_t) data | (uintptr_t) kind) {
      assert ((reinterpret_cast<uintptr_t>(const_cast<void*>(data)) & 0x1) == 0
              && "Address must have at least a 2-byte alignment."); 
    }
  
  const void* getData() const { return (const void*) (Raw & ~0x1); }
  
public:  
  // Folding-set profiling.
  void Profile(llvm::FoldingSetNodeID& ID) const { ID.AddPointer((void*) Raw); }

  // Comparing regions.
  bool operator==(const Region& R) const { return Raw == R.Raw; }
  bool operator!=(const Region& R) const { return Raw != R.Raw; }

  // Implement isa<T> support.
  Kind getKind() const { return Kind (Raw & 0x1); }
  static inline bool classof(const Region*) { return true; }
};
  
//===----------------------------------------------------------------------===//
// Region Types.
//===----------------------------------------------------------------------===//
  
class VarRegion : public Region {
public:
  VarRegion(VarDecl* VD) : Region(VD, Region::Var) {}
  
  /// getDecl - Return the declaration of the variable the region represents.
  const VarDecl* getDecl() const { return (const VarDecl*) getData(); }  
  operator const VarDecl*() const { return getDecl(); }
  
  RegionExtent getExtent(BasicValueFactory& BV) const;
  
  // Implement isa<T> support.
  static inline bool classof(const Region* R) {
    return R->getKind() == Region::Var;
  }
  
  static inline bool classof(const VarRegion*) {
    return true;
  }
};

class AnonRegion : public Region {
protected:
  friend class Region;
  
  AnonRegion(uintptr_t RegionID) : Region((void*) (RegionID<<1), Region::Anon) {
    assert (((RegionID << 1) >> 1) == RegionID);
  }
  
public:
  
  uintptr_t getID() const { return ((uintptr_t) getData()) >> 1; }
  
  // Implement isa<T> support.
  static inline bool classof(const Region* R) {
    return R->getKind() == Region::Anon;
  }
  
  static inline bool classof(const AnonRegion*) {
    return true;
  }
};
  
} // end clang namespace

#endif
