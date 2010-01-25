//== MemRegion.h - Abstract memory regions for static analysis --*- C++ -*--==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines MemRegion and its subclasses.  MemRegion defines a
//  partially-typed abstraction of memory useful for path-sensitive dataflow
//  analyses.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_MEMREGION_H
#define LLVM_CLANG_ANALYSIS_MEMREGION_H

#include "clang/AST/Decl.h"
#include "clang/AST/DeclObjC.h"
#include "clang/Checker/PathSensitive/SymbolManager.h"
#include "clang/Checker/PathSensitive/SVals.h"
#include "clang/AST/ASTContext.h"
#include "llvm/Support/Casting.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/ImmutableList.h"
#include "llvm/ADT/ImmutableMap.h"
#include "llvm/Support/Allocator.h"
#include <string>

namespace llvm { class raw_ostream; }

namespace clang {

class MemRegionManager;
class MemSpaceRegion;
class LocationContext;
class StackFrameContext;
class VarRegion;

//===----------------------------------------------------------------------===//
// Base region classes.
//===----------------------------------------------------------------------===//

/// MemRegion - The root abstract class for all memory regions.
class MemRegion : public llvm::FoldingSetNode {
  friend class MemRegionManager;
public:
  enum Kind {
    // Memory spaces.
    BEG_MEMSPACES,
    GenericMemSpaceRegionKind = BEG_MEMSPACES,
    StackLocalsSpaceRegionKind,
    StackArgumentsSpaceRegionKind,
    HeapSpaceRegionKind,
    UnknownSpaceRegionKind,
    GlobalsSpaceRegionKind,
    END_MEMSPACES = GlobalsSpaceRegionKind,
    // Untyped regions.
    SymbolicRegionKind,
    AllocaRegionKind,
    // Typed regions.
    BEG_TYPED_REGIONS,
    FunctionTextRegionKind = BEG_TYPED_REGIONS,
    BlockTextRegionKind,
    BlockDataRegionKind,
    CompoundLiteralRegionKind,
    CXXThisRegionKind,
    StringRegionKind,
    ElementRegionKind,
    // Decl Regions.
    BEG_DECL_REGIONS,
    VarRegionKind = BEG_DECL_REGIONS,
    FieldRegionKind,
    ObjCIvarRegionKind,
    CXXObjectRegionKind,
    END_DECL_REGIONS = CXXObjectRegionKind,
    END_TYPED_REGIONS = END_DECL_REGIONS
  };
    
private:
  const Kind kind;

protected:
  MemRegion(Kind k) : kind(k) {}
  virtual ~MemRegion();

public:
  ASTContext &getContext() const;

  virtual void Profile(llvm::FoldingSetNodeID& ID) const = 0;

  virtual MemRegionManager* getMemRegionManager() const = 0;

  std::string getString() const;

  const MemSpaceRegion *getMemorySpace() const;

  const MemRegion *getBaseRegion() const;

  const MemRegion *StripCasts() const;

  bool hasGlobalsOrParametersStorage() const;

  bool hasStackStorage() const;
  
  bool hasStackNonParametersStorage() const;
  
  bool hasStackParametersStorage() const;

  virtual void dumpToStream(llvm::raw_ostream& os) const;

  void dump() const;

  Kind getKind() const { return kind; }

  template<typename RegionTy> const RegionTy* getAs() const;

  virtual bool isBoundable() const { return false; }

  static bool classof(const MemRegion*) { return true; }
};

/// MemSpaceRegion - A memory region that represents and "memory space";
///  for example, the set of global variables, the stack frame, etc.
class MemSpaceRegion : public MemRegion {
protected:
  friend class MemRegionManager;
  
  MemRegionManager *Mgr;

  MemSpaceRegion(MemRegionManager *mgr, Kind k = GenericMemSpaceRegionKind)
    : MemRegion(k), Mgr(mgr) {
    assert(classof(this));
  }

  MemRegionManager* getMemRegionManager() const { return Mgr; }

public:
  bool isBoundable() const { return false; }
  
  void Profile(llvm::FoldingSetNodeID &ID) const;

  static bool classof(const MemRegion *R) {
    Kind k = R->getKind();
    return k >= BEG_MEMSPACES && k <= END_MEMSPACES;
  }
};
  
class GlobalsSpaceRegion : public MemSpaceRegion {
  friend class MemRegionManager;

  GlobalsSpaceRegion(MemRegionManager *mgr)
    : MemSpaceRegion(mgr, GlobalsSpaceRegionKind) {}
public:
  static bool classof(const MemRegion *R) {
    return R->getKind() == GlobalsSpaceRegionKind;
  }
};
  
class HeapSpaceRegion : public MemSpaceRegion {
  friend class MemRegionManager;
  
  HeapSpaceRegion(MemRegionManager *mgr)
    : MemSpaceRegion(mgr, HeapSpaceRegionKind) {}
public:
  static bool classof(const MemRegion *R) {
    return R->getKind() == HeapSpaceRegionKind;
  }
};
  
class UnknownSpaceRegion : public MemSpaceRegion {
  friend class MemRegionManager;
  UnknownSpaceRegion(MemRegionManager *mgr)
    : MemSpaceRegion(mgr, UnknownSpaceRegionKind) {}
public:
  static bool classof(const MemRegion *R) {
    return R->getKind() == UnknownSpaceRegionKind;
  }
};
  
class StackSpaceRegion : public MemSpaceRegion {
private:
  const StackFrameContext *SFC;

protected:
  StackSpaceRegion(MemRegionManager *mgr, Kind k, const StackFrameContext *sfc)
    : MemSpaceRegion(mgr, k), SFC(sfc) {
    assert(classof(this));
  }

public:  
  const StackFrameContext *getStackFrame() const { return SFC; }
  
  void Profile(llvm::FoldingSetNodeID &ID) const;

  static bool classof(const MemRegion *R) {
    Kind k = R->getKind();
    return k >= StackLocalsSpaceRegionKind &&
           k <= StackArgumentsSpaceRegionKind;
  }  
};
  
class StackLocalsSpaceRegion : public StackSpaceRegion {
private:
  friend class MemRegionManager;
  StackLocalsSpaceRegion(MemRegionManager *mgr, const StackFrameContext *sfc)
    : StackSpaceRegion(mgr, StackLocalsSpaceRegionKind, sfc) {}
public:
  static bool classof(const MemRegion *R) {
    return R->getKind() == StackLocalsSpaceRegionKind;
  }
};

class StackArgumentsSpaceRegion : public StackSpaceRegion {
private:
  friend class MemRegionManager;
  StackArgumentsSpaceRegion(MemRegionManager *mgr, const StackFrameContext *sfc)
    : StackSpaceRegion(mgr, StackArgumentsSpaceRegionKind, sfc) {}
public:
  static bool classof(const MemRegion *R) {
    return R->getKind() == StackArgumentsSpaceRegionKind;
  }
};

/// SubRegion - A region that subsets another larger region.  Most regions
///  are subclasses of SubRegion.
class SubRegion : public MemRegion {
protected:
  const MemRegion* superRegion;
  SubRegion(const MemRegion* sReg, Kind k) : MemRegion(k), superRegion(sReg) {}
public:
  const MemRegion* getSuperRegion() const {
    return superRegion;
  }

  MemRegionManager* getMemRegionManager() const;

  bool isSubRegionOf(const MemRegion* R) const;

  static bool classof(const MemRegion* R) {
    return R->getKind() > END_MEMSPACES;
  }
};

//===----------------------------------------------------------------------===//
// Auxillary data classes for use with MemRegions.
//===----------------------------------------------------------------------===//

class ElementRegion;

class RegionRawOffset : public std::pair<const MemRegion*, int64_t> {
private:
  friend class ElementRegion;

  RegionRawOffset(const MemRegion* reg, int64_t offset = 0)
    : std::pair<const MemRegion*, int64_t>(reg, offset) {}

public:
  // FIXME: Eventually support symbolic offsets.
  int64_t getByteOffset() const { return second; }
  const MemRegion *getRegion() const { return first; }

  void dumpToStream(llvm::raw_ostream& os) const;
  void dump() const;
};

//===----------------------------------------------------------------------===//
// MemRegion subclasses.
//===----------------------------------------------------------------------===//

/// AllocaRegion - A region that represents an untyped blob of bytes created
///  by a call to 'alloca'.
class AllocaRegion : public SubRegion {
  friend class MemRegionManager;
protected:
  unsigned Cnt; // Block counter.  Used to distinguish different pieces of
                // memory allocated by alloca at the same call site.
  const Expr* Ex;

  AllocaRegion(const Expr* ex, unsigned cnt, const MemRegion *superRegion)
    : SubRegion(superRegion, AllocaRegionKind), Cnt(cnt), Ex(ex) {}

public:

  const Expr* getExpr() const { return Ex; }

  bool isBoundable() const { return true; }

  void Profile(llvm::FoldingSetNodeID& ID) const;

  static void ProfileRegion(llvm::FoldingSetNodeID& ID, const Expr* Ex,
                            unsigned Cnt, const MemRegion *superRegion);

  void dumpToStream(llvm::raw_ostream& os) const;

  static bool classof(const MemRegion* R) {
    return R->getKind() == AllocaRegionKind;
  }
};

/// TypedRegion - An abstract class representing regions that are typed.
class TypedRegion : public SubRegion {
protected:
  TypedRegion(const MemRegion* sReg, Kind k) : SubRegion(sReg, k) {}

public:
  virtual QualType getValueType(ASTContext &C) const = 0;

  virtual QualType getLocationType(ASTContext& C) const {
    // FIXME: We can possibly optimize this later to cache this value.
    return C.getPointerType(getValueType(C));
  }

  QualType getDesugaredValueType(ASTContext& C) const {
    QualType T = getValueType(C);
    return T.getTypePtr() ? T.getDesugaredType() : T;
  }

  QualType getDesugaredLocationType(ASTContext& C) const {
    return getLocationType(C).getDesugaredType();
  }

  bool isBoundable() const {
    return !getValueType(getContext()).isNull();
  }

  static bool classof(const MemRegion* R) {
    unsigned k = R->getKind();
    return k >= BEG_TYPED_REGIONS && k <= END_TYPED_REGIONS;
  }
};


class CodeTextRegion : public TypedRegion {
protected:
  CodeTextRegion(const MemRegion *sreg, Kind k) : TypedRegion(sreg, k) {}
public:
  QualType getValueType(ASTContext &C) const {
    // Do not get the object type of a CodeTextRegion.
    assert(0);
    return QualType();
  }
  
  bool isBoundable() const { return false; }
    
  static bool classof(const MemRegion* R) {
    Kind k = R->getKind();
    return k >= FunctionTextRegionKind && k <= BlockTextRegionKind;
  }
};

/// FunctionTextRegion - A region that represents code texts of function.
class FunctionTextRegion : public CodeTextRegion {
  const FunctionDecl *FD;
public:
  FunctionTextRegion(const FunctionDecl* fd, const MemRegion* sreg)
    : CodeTextRegion(sreg, FunctionTextRegionKind), FD(fd) {}
  
  QualType getLocationType(ASTContext &C) const {
    return C.getPointerType(FD->getType());
  }
  
  const FunctionDecl *getDecl() const {
    return FD;
  }
    
  virtual void dumpToStream(llvm::raw_ostream& os) const;
  
  void Profile(llvm::FoldingSetNodeID& ID) const;
  
  static void ProfileRegion(llvm::FoldingSetNodeID& ID, const FunctionDecl *FD,
                            const MemRegion*);
  
  static bool classof(const MemRegion* R) {
    return R->getKind() == FunctionTextRegionKind;
  }
};
  
  
/// BlockTextRegion - A region that represents code texts of blocks (closures).
///  Blocks are represented with two kinds of regions.  BlockTextRegions
///  represent the "code", while BlockDataRegions represent instances of blocks,
///  which correspond to "code+data".  The distinction is important, because
///  like a closure a block captures the values of externally referenced
///  variables.
class BlockTextRegion : public CodeTextRegion {
  friend class MemRegionManager;

  const BlockDecl *BD;
  AnalysisContext *AC;
  CanQualType locTy;

  BlockTextRegion(const BlockDecl *bd, CanQualType lTy,
                  AnalysisContext *ac, const MemRegion* sreg)
    : CodeTextRegion(sreg, BlockTextRegionKind), BD(bd), AC(ac), locTy(lTy) {}

public:
  QualType getLocationType(ASTContext &C) const {
    return locTy;
  }
  
  const BlockDecl *getDecl() const {
    return BD;
  }

  AnalysisContext *getAnalysisContext() const { return AC; }
    
  virtual void dumpToStream(llvm::raw_ostream& os) const;
  
  void Profile(llvm::FoldingSetNodeID& ID) const;
  
  static void ProfileRegion(llvm::FoldingSetNodeID& ID, const BlockDecl *BD,
                            CanQualType, const AnalysisContext*,
                            const MemRegion*);
  
  static bool classof(const MemRegion* R) {
    return R->getKind() == BlockTextRegionKind;
  }
};
  
/// BlockDataRegion - A region that represents a block instance.
///  Blocks are represented with two kinds of regions.  BlockTextRegions
///  represent the "code", while BlockDataRegions represent instances of blocks,
///  which correspond to "code+data".  The distinction is important, because
///  like a closure a block captures the values of externally referenced
///  variables.
/// BlockDataRegion - A region that represents code texts of blocks (closures).
class BlockDataRegion : public SubRegion {
  friend class MemRegionManager;
  const BlockTextRegion *BC;
  const LocationContext *LC; // Can be null */
  void *ReferencedVars;

  BlockDataRegion(const BlockTextRegion *bc, const LocationContext *lc,
                  const MemRegion *sreg)
  : SubRegion(sreg, BlockDataRegionKind), BC(bc), LC(lc), ReferencedVars(0) {}

public:  
  const BlockTextRegion *getCodeRegion() const { return BC; }
  
  const BlockDecl *getDecl() const { return BC->getDecl(); }
  
  class referenced_vars_iterator {
    const MemRegion * const *R;
  public:
    explicit referenced_vars_iterator(const MemRegion * const *r) : R(r) {}
    
    operator const MemRegion * const *() const {
      return R;
    }
    
    const VarRegion* operator*() const {
      return cast<VarRegion>(*R);
    }
    
    bool operator==(const referenced_vars_iterator &I) const {
      return I.R == R;
    }
    bool operator!=(const referenced_vars_iterator &I) const {
      return I.R != R;
    }
    referenced_vars_iterator& operator++() {
      ++R;
      return *this;
    }
  };
      
  referenced_vars_iterator referenced_vars_begin() const;
  referenced_vars_iterator referenced_vars_end() const;  
    
  virtual void dumpToStream(llvm::raw_ostream& os) const;
    
  void Profile(llvm::FoldingSetNodeID& ID) const;
    
  static void ProfileRegion(llvm::FoldingSetNodeID&, const BlockTextRegion *,
                            const LocationContext *, const MemRegion *);
    
  static bool classof(const MemRegion* R) {
    return R->getKind() == BlockDataRegionKind;
  }
private:
  void LazyInitializeReferencedVars();
};

/// SymbolicRegion - A special, "non-concrete" region. Unlike other region
///  clases, SymbolicRegion represents a region that serves as an alias for
///  either a real region, a NULL pointer, etc.  It essentially is used to
///  map the concept of symbolic values into the domain of regions.  Symbolic
///  regions do not need to be typed.
class SymbolicRegion : public SubRegion {
protected:
  const SymbolRef sym;

public:
  SymbolicRegion(const SymbolRef s, const MemRegion* sreg)
    : SubRegion(sreg, SymbolicRegionKind), sym(s) {}

  SymbolRef getSymbol() const {
    return sym;
  }

  bool isBoundable() const { return true; }

  void Profile(llvm::FoldingSetNodeID& ID) const;

  static void ProfileRegion(llvm::FoldingSetNodeID& ID,
                            SymbolRef sym,
                            const MemRegion* superRegion);

  void dumpToStream(llvm::raw_ostream& os) const;

  static bool classof(const MemRegion* R) {
    return R->getKind() == SymbolicRegionKind;
  }
};

/// StringRegion - Region associated with a StringLiteral.
class StringRegion : public TypedRegion {
  friend class MemRegionManager;
  const StringLiteral* Str;
protected:

  StringRegion(const StringLiteral* str, const MemRegion* sreg)
    : TypedRegion(sreg, StringRegionKind), Str(str) {}

  static void ProfileRegion(llvm::FoldingSetNodeID& ID,
                            const StringLiteral* Str,
                            const MemRegion* superRegion);

public:

  const StringLiteral* getStringLiteral() const { return Str; }

  QualType getValueType(ASTContext& C) const {
    return Str->getType();
  }

  bool isBoundable() const { return false; }

  void Profile(llvm::FoldingSetNodeID& ID) const {
    ProfileRegion(ID, Str, superRegion);
  }

  void dumpToStream(llvm::raw_ostream& os) const;

  static bool classof(const MemRegion* R) {
    return R->getKind() == StringRegionKind;
  }
};

/// CompoundLiteralRegion - A memory region representing a compound literal.
///   Compound literals are essentially temporaries that are stack allocated
///   or in the global constant pool.
class CompoundLiteralRegion : public TypedRegion {
private:
  friend class MemRegionManager;
  const CompoundLiteralExpr* CL;

  CompoundLiteralRegion(const CompoundLiteralExpr* cl, const MemRegion* sReg)
    : TypedRegion(sReg, CompoundLiteralRegionKind), CL(cl) {}

  static void ProfileRegion(llvm::FoldingSetNodeID& ID,
                            const CompoundLiteralExpr* CL,
                            const MemRegion* superRegion);
public:
  QualType getValueType(ASTContext& C) const {
    return C.getCanonicalType(CL->getType());
  }

  bool isBoundable() const { return !CL->isFileScope(); }

  void Profile(llvm::FoldingSetNodeID& ID) const;

  void dumpToStream(llvm::raw_ostream& os) const;

  const CompoundLiteralExpr* getLiteralExpr() const { return CL; }

  static bool classof(const MemRegion* R) {
    return R->getKind() == CompoundLiteralRegionKind;
  }
};

class DeclRegion : public TypedRegion {
protected:
  const Decl* D;

  DeclRegion(const Decl* d, const MemRegion* sReg, Kind k)
    : TypedRegion(sReg, k), D(d) {}

  static void ProfileRegion(llvm::FoldingSetNodeID& ID, const Decl* D,
                      const MemRegion* superRegion, Kind k);

public:
  const Decl* getDecl() const { return D; }
  void Profile(llvm::FoldingSetNodeID& ID) const;

  static bool classof(const MemRegion* R) {
    unsigned k = R->getKind();
    return k >= BEG_DECL_REGIONS && k <= END_DECL_REGIONS;
  }
};

class VarRegion : public DeclRegion {
  friend class MemRegionManager;

  // Constructors and private methods.
  VarRegion(const VarDecl* vd, const MemRegion* sReg)
    : DeclRegion(vd, sReg, VarRegionKind) {}

  static void ProfileRegion(llvm::FoldingSetNodeID& ID, const VarDecl* VD,
                            const MemRegion *superRegion) {
    DeclRegion::ProfileRegion(ID, VD, superRegion, VarRegionKind);
  }

  void Profile(llvm::FoldingSetNodeID& ID) const;

public:
  const VarDecl *getDecl() const { return cast<VarDecl>(D); }

  const StackFrameContext *getStackFrame() const;
  
  QualType getValueType(ASTContext& C) const {
    // FIXME: We can cache this if needed.
    return C.getCanonicalType(getDecl()->getType());
  }

  void dumpToStream(llvm::raw_ostream& os) const;

  static bool classof(const MemRegion* R) {
    return R->getKind() == VarRegionKind;
  }
};
  
/// CXXThisRegion - Represents the region for the implicit 'this' parameter
///  in a call to a C++ method.  This region doesn't represent the object
///  referred to by 'this', but rather 'this' itself.
class CXXThisRegion : public TypedRegion {
  friend class MemRegionManager;
  CXXThisRegion(const PointerType *thisPointerTy,
                const MemRegion *sReg)
    : TypedRegion(sReg, CXXThisRegionKind), ThisPointerTy(thisPointerTy) {}

  static void ProfileRegion(llvm::FoldingSetNodeID &ID,
                            const PointerType *PT,
                            const MemRegion *sReg);

  void Profile(llvm::FoldingSetNodeID &ID) const;

public:  
  QualType getValueType(ASTContext &C) const {
    return QualType(ThisPointerTy, 0);
  }
  
  void dumpToStream(llvm::raw_ostream& os) const;
  
  static bool classof(const MemRegion* R) {
    return R->getKind() == CXXThisRegionKind;
  }

private:
  const PointerType *ThisPointerTy;
};

class FieldRegion : public DeclRegion {
  friend class MemRegionManager;

  FieldRegion(const FieldDecl* fd, const MemRegion* sReg)
    : DeclRegion(fd, sReg, FieldRegionKind) {}

public:

  void dumpToStream(llvm::raw_ostream& os) const;

  const FieldDecl* getDecl() const { return cast<FieldDecl>(D); }

  QualType getValueType(ASTContext& C) const {
    // FIXME: We can cache this if needed.
    return C.getCanonicalType(getDecl()->getType());
  }

  static void ProfileRegion(llvm::FoldingSetNodeID& ID, const FieldDecl* FD,
                            const MemRegion* superRegion) {
    DeclRegion::ProfileRegion(ID, FD, superRegion, FieldRegionKind);
  }

  static bool classof(const MemRegion* R) {
    return R->getKind() == FieldRegionKind;
  }
};

class ObjCIvarRegion : public DeclRegion {

  friend class MemRegionManager;

  ObjCIvarRegion(const ObjCIvarDecl* ivd, const MemRegion* sReg)
    : DeclRegion(ivd, sReg, ObjCIvarRegionKind) {}

  static void ProfileRegion(llvm::FoldingSetNodeID& ID, const ObjCIvarDecl* ivd,
                            const MemRegion* superRegion) {
    DeclRegion::ProfileRegion(ID, ivd, superRegion, ObjCIvarRegionKind);
  }

public:
  const ObjCIvarDecl* getDecl() const { return cast<ObjCIvarDecl>(D); }
  QualType getValueType(ASTContext&) const { return getDecl()->getType(); }

  void dumpToStream(llvm::raw_ostream& os) const;

  static bool classof(const MemRegion* R) {
    return R->getKind() == ObjCIvarRegionKind;
  }
};

class ElementRegion : public TypedRegion {
  friend class MemRegionManager;

  QualType ElementType;
  SVal Index;

  ElementRegion(QualType elementType, SVal Idx, const MemRegion* sReg)
    : TypedRegion(sReg, ElementRegionKind),
      ElementType(elementType), Index(Idx) {
    assert((!isa<nonloc::ConcreteInt>(&Idx) ||
           cast<nonloc::ConcreteInt>(&Idx)->getValue().isSigned()) &&
           "The index must be signed");
  }

  static void ProfileRegion(llvm::FoldingSetNodeID& ID, QualType elementType,
                            SVal Idx, const MemRegion* superRegion);

public:

  SVal getIndex() const { return Index; }

  QualType getValueType(ASTContext&) const {
    return ElementType;
  }

  QualType getElementType() const {
    return ElementType;
  }

  RegionRawOffset getAsRawOffset() const;

  void dumpToStream(llvm::raw_ostream& os) const;

  void Profile(llvm::FoldingSetNodeID& ID) const;

  static bool classof(const MemRegion* R) {
    return R->getKind() == ElementRegionKind;
  }
};

// C++ temporary object associated with an expression.
class CXXObjectRegion : public TypedRegion {
  friend class MemRegionManager;

  Expr const *Ex;

  CXXObjectRegion(Expr const *E, MemRegion const *sReg) 
    : TypedRegion(sReg, CXXObjectRegionKind), Ex(E) {}

  static void ProfileRegion(llvm::FoldingSetNodeID &ID,
                            Expr const *E, const MemRegion *sReg);
  
public:
  QualType getValueType(ASTContext& C) const {
    return Ex->getType();
  }

  void Profile(llvm::FoldingSetNodeID &ID) const;

  static bool classof(const MemRegion* R) {
    return R->getKind() == CXXObjectRegionKind;
  }
};

template<typename RegionTy>
const RegionTy* MemRegion::getAs() const {
  if (const RegionTy* RT = dyn_cast<RegionTy>(this))
    return RT;

  return NULL;
}

//===----------------------------------------------------------------------===//
// MemRegionManager - Factory object for creating regions.
//===----------------------------------------------------------------------===//

class MemRegionManager {
  ASTContext &C;
  llvm::BumpPtrAllocator& A;
  llvm::FoldingSet<MemRegion> Regions;

  GlobalsSpaceRegion *globals;
  
  const StackFrameContext *cachedStackLocalsFrame;
  StackLocalsSpaceRegion *cachedStackLocalsRegion;
  
  const StackFrameContext *cachedStackArgumentsFrame;
  StackArgumentsSpaceRegion *cachedStackArgumentsRegion;

  HeapSpaceRegion *heap;
  UnknownSpaceRegion *unknown;
  MemSpaceRegion *code;

public:
  MemRegionManager(ASTContext &c, llvm::BumpPtrAllocator& a)
    : C(c), A(a), globals(0),
      cachedStackLocalsFrame(0), cachedStackLocalsRegion(0),
      cachedStackArgumentsFrame(0), cachedStackArgumentsRegion(0),
      heap(0), unknown(0), code(0) {}

  ~MemRegionManager();

  ASTContext &getContext() { return C; }
  
  llvm::BumpPtrAllocator &getAllocator() { return A; }

  /// getStackLocalsRegion - Retrieve the memory region associated with the
  ///  specified stack frame.
  const StackLocalsSpaceRegion *
  getStackLocalsRegion(const StackFrameContext *STC);

  /// getStackArgumentsRegion - Retrieve the memory region associated with
  ///  function/method arguments of the specified stack frame.
  const StackArgumentsSpaceRegion *
  getStackArgumentsRegion(const StackFrameContext *STC);

  /// getGlobalsRegion - Retrieve the memory region associated with
  ///  all global variables.
  const GlobalsSpaceRegion *getGlobalsRegion();

  /// getHeapRegion - Retrieve the memory region associated with the
  ///  generic "heap".
  const HeapSpaceRegion *getHeapRegion();

  /// getUnknownRegion - Retrieve the memory region associated with unknown
  /// memory space.
  const MemSpaceRegion *getUnknownRegion();

  const MemSpaceRegion *getCodeRegion();

  /// getAllocaRegion - Retrieve a region associated with a call to alloca().
  const AllocaRegion *getAllocaRegion(const Expr* Ex, unsigned Cnt,
                                      const LocationContext *LC);

  /// getCompoundLiteralRegion - Retrieve the region associated with a
  ///  given CompoundLiteral.
  const CompoundLiteralRegion*
  getCompoundLiteralRegion(const CompoundLiteralExpr* CL,
                           const LocationContext *LC);
  
  /// getCXXThisRegion - Retrieve the [artifical] region associated with the
  ///  parameter 'this'.
  const CXXThisRegion *getCXXThisRegion(QualType thisPointerTy,
                                        const LocationContext *LC);

  /// getSymbolicRegion - Retrieve or create a "symbolic" memory region.
  const SymbolicRegion* getSymbolicRegion(SymbolRef sym);

  const StringRegion* getStringRegion(const StringLiteral* Str);

  /// getVarRegion - Retrieve or create the memory region associated with
  ///  a specified VarDecl and LocationContext.
  const VarRegion* getVarRegion(const VarDecl *D, const LocationContext *LC);

  /// getVarRegion - Retrieve or create the memory region associated with
  ///  a specified VarDecl and super region.
  const VarRegion* getVarRegion(const VarDecl *D, const MemRegion *superR);
  
  /// getElementRegion - Retrieve the memory region associated with the
  ///  associated element type, index, and super region.
  const ElementRegion *getElementRegion(QualType elementType, SVal Idx,
                                  const MemRegion *superRegion,
                                  ASTContext &Ctx);

  const ElementRegion *getElementRegionWithSuper(const ElementRegion *ER,
                                           const MemRegion *superRegion) {
    return getElementRegion(ER->getElementType(), ER->getIndex(),
                            superRegion, ER->getContext());
  }

  /// getFieldRegion - Retrieve or create the memory region associated with
  ///  a specified FieldDecl.  'superRegion' corresponds to the containing
  ///  memory region (which typically represents the memory representing
  ///  a structure or class).
  const FieldRegion *getFieldRegion(const FieldDecl* fd,
                                    const MemRegion* superRegion);

  const FieldRegion *getFieldRegionWithSuper(const FieldRegion *FR,
                                             const MemRegion *superRegion) {
    return getFieldRegion(FR->getDecl(), superRegion);
  }

  /// getObjCIvarRegion - Retrieve or create the memory region associated with
  ///   a specified Objective-c instance variable.  'superRegion' corresponds
  ///   to the containing region (which typically represents the Objective-C
  ///   object).
  const ObjCIvarRegion *getObjCIvarRegion(const ObjCIvarDecl* ivd,
                                          const MemRegion* superRegion);

  const CXXObjectRegion *getCXXObjectRegion(Expr const *Ex,
                                            LocationContext const *LC);

  const FunctionTextRegion *getFunctionTextRegion(const FunctionDecl *FD);
  const BlockTextRegion *getBlockTextRegion(const BlockDecl *BD,
                                            CanQualType locTy,
                                            AnalysisContext *AC);
  
  /// getBlockDataRegion - Get the memory region associated with an instance
  ///  of a block.  Unlike many other MemRegions, the LocationContext*
  ///  argument is allowed to be NULL for cases where we have no known
  ///  context.
  const BlockDataRegion *getBlockDataRegion(const BlockTextRegion *bc,
                                            const LocationContext *lc = NULL);

  bool isGlobalsRegion(const MemRegion* R) {
    assert(R);
    return R == globals;
  }
  
private:
  template <typename RegionTy, typename A1>
  RegionTy* getRegion(const A1 a1);

  template <typename RegionTy, typename A1>
  RegionTy* getSubRegion(const A1 a1, const MemRegion* superRegion);

  template <typename RegionTy, typename A1, typename A2>
  RegionTy* getRegion(const A1 a1, const A2 a2);

  template <typename RegionTy, typename A1, typename A2>
  RegionTy* getSubRegion(const A1 a1, const A2 a2,
                         const MemRegion* superRegion);

  template <typename RegionTy, typename A1, typename A2, typename A3>
  RegionTy* getSubRegion(const A1 a1, const A2 a2, const A3 a3,
                         const MemRegion* superRegion);
  
  template <typename REG>
  const REG* LazyAllocate(REG*& region);
  
  template <typename REG, typename ARG>
  const REG* LazyAllocate(REG*& region, ARG a);
};

//===----------------------------------------------------------------------===//
// Out-of-line member definitions.
//===----------------------------------------------------------------------===//

inline ASTContext& MemRegion::getContext() const {
  return getMemRegionManager()->getContext();
}
  
} // end clang namespace

//===----------------------------------------------------------------------===//
// Pretty-printing regions.
//===----------------------------------------------------------------------===//

namespace llvm {
static inline raw_ostream& operator<<(raw_ostream& os,
                                      const clang::MemRegion* R) {
  R->dumpToStream(os);
  return os;
}
} // end llvm namespace

#endif
