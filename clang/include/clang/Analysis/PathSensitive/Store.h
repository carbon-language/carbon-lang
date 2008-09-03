//== Store.h - Interface for maps from Locations to Values ------*- C++ -*--==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defined the types Store and StoreManager.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_STORE_H
#define LLVM_CLANG_ANALYSIS_STORE_H

#include "clang/Analysis/PathSensitive/RValues.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include <vector>
#include <iosfwd>

namespace clang {
  
typedef const void* Store;
  
namespace store {
  /// Region - A region represents an abstract chunk of memory.  Subclasses
  ///  of StoreManager are responsible for defining the particular semantics
  ///  of Region for the store they represent.
  class Region {
  protected:
    const void* Data;
    Region(const void* data) : Data(data) {}
  public:
    Region() : Data(0) {}
  };
  
  /// Binding - A "binding" represents a binding of a value to an abstract
  ///  chunk of memory (which is represented by a region).  Subclasses of
  ///  StoreManager are responsible for defining the particular semantics
  ///  of a Binding.
  class Binding {
  protected:
    const void* first;
    const void* second;
    Binding(const void* f, const void* s = 0) : first(f), second(s) {}
  public:
    Binding() : first(0), second(0) {}
    operator bool() const { return first || second; }
  };
  
  /// RegionExtent - Represents the size, or extent, or an abstract memory
  ///  chunk (a region).  Sizes are in bits.  RegionExtent is essentially a
  ///  variant with three subclasses: UnknownExtent, FixedExtent,
  ///  and SymbolicExtent.  
  class RegionExtent {
  public:
    enum Kind { Unknown = 0, Fixed = 0, Sym = 1 };
    
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
  
  /// UnknownExtent - Represents a region extent with no available information
  ///  about the size of the region.
  class UnknownExtent : public RegionExtent {
  public:
    UnknownExtent() : RegionExtent(0,Unknown) {}
    
    // Implement isa<T> support.
    static inline bool classof(const RegionExtent* E) {
      return E->getRaw() == 0;
    }  
  };
  
  /// FixedExtent - Represents a region extent with a known fixed size.
  ///  Typically FixedExtents are used to represent the size of variables, but
  ///  they can also be used to represent the size of a constant-sized array.
  class FixedExtent : public RegionExtent {
  public:
    FixedExtent(const llvm::APSInt& X) : RegionExtent((uintptr_t) &X, Fixed) {}
    
    const llvm::APSInt& getInt() const {
      return *((llvm::APSInt*) getData());
    }
    
    // Implement isa<T> support.
    static inline bool classof(const RegionExtent* E) {
      return E->getKind() == Fixed && E->getRaw() != 0;
    }
  };
  
  /// SymbolicExtent - Represents the extent of a region where the extent
  ///  itself is a symbolic value.  These extents can be used to represent
  ///  the sizes of dynamically allocated chunks of memory with variable size.
  class SymbolicExtent : public RegionExtent {
  public:
    SymbolicExtent(SymbolID S) : RegionExtent(S.getNumber() << 1, Sym) {}
    
    SymbolID getSymbol() const { return SymbolID(getData() >> 1); }
    
    // Implement isa<T> support.
    static inline bool classof(const RegionExtent* E) {
      return E->getKind() == Sym;
    }  
  };
} // end store namespace
  
class GRStateManager;
class LiveVariables;
class Stmt;
  
class StoreManager {
public:
  typedef llvm::SmallSet<SymbolID, 20>      LiveSymbolsTy;
  typedef llvm::DenseSet<SymbolID>          DeadSymbolsTy;
  typedef std::vector<ValueDecl*>           DeclRootsTy;
  
  virtual ~StoreManager() {}
  virtual RVal GetRVal(Store St, LVal LV, QualType T = QualType()) = 0;
  virtual Store SetRVal(Store St, LVal LV, RVal V) = 0;
  virtual Store Remove(Store St, LVal LV) = 0;
  virtual Store getInitialStore(GRStateManager& StateMgr) = 0;
  
  virtual Store RemoveDeadBindings(Store store, Stmt* Loc,
                                   const LiveVariables& Live,
                                   DeclRootsTy& DRoots, LiveSymbolsTy& LSymbols,                                  
                                   DeadSymbolsTy& DSymbols) = 0;

  virtual Store AddDecl(Store store, GRStateManager& StMgr,
                        const VarDecl* VD, Expr* Ex, 
                        RVal InitVal = UndefinedVal(), unsigned Count = 0) = 0;

  virtual void print(Store store, std::ostream& Out,
                     const char* nl, const char *sep) = 0;
      
  class BindingsHandler {
  public:    
    virtual ~BindingsHandler();
    virtual bool HandleBinding(StoreManager& SMgr, Store store,
                               store::Binding binding) = 0;
  };
  
  /// iterBindings - Iterate over the bindings in the Store.
  virtual void iterBindings(Store store, BindingsHandler& f) = 0;
  
  /// getBindings - Returns all bindings in the specified store that bind
  ///  to the specified symbolic value.
  void getBindings(llvm::SmallVectorImpl<store::Binding>& bindings,
                   Store store, SymbolID Sym);
  
  /// BindingAsString - Returns a string representing the given binding.
  virtual std::string BindingAsString(store::Binding binding) = 0;
  
  /// getExtent - Returns the size of the region in bits.
  virtual store::RegionExtent getExtent(store::Region R) = 0;

  /// getRVal - Returns the bound RVal for a given binding.
  virtual RVal getRVal(Store store, store::Binding binding) = 0;
};
  
StoreManager* CreateBasicStoreManager(GRStateManager& StMgr);

  
} // end clang namespace

#endif
