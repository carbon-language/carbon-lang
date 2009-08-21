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
#include "clang/Analysis/PathSensitive/SymbolManager.h"
#include "clang/Analysis/PathSensitive/SVals.h"
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
      
//===----------------------------------------------------------------------===//
// Base region classes.
//===----------------------------------------------------------------------===//
  
/// MemRegion - The root abstract class for all memory regions.
class MemRegion : public llvm::FoldingSetNode {
public:
  enum Kind { MemSpaceRegionKind,
              SymbolicRegionKind,
              AllocaRegionKind,
              // Typed regions.
              BEG_TYPED_REGIONS,
               CodeTextRegionKind,
               CompoundLiteralRegionKind,
               StringRegionKind, ElementRegionKind,
               // Decl Regions.
                 BEG_DECL_REGIONS,
                  VarRegionKind, FieldRegionKind,
                  ObjCIvarRegionKind, ObjCObjectRegionKind,
                 END_DECL_REGIONS,
              END_TYPED_REGIONS };  
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
    
  bool hasStackStorage() const;
  
  bool hasParametersStorage() const;
  
  bool hasGlobalsStorage() const;
  
  bool hasGlobalsOrParametersStorage() const;
  
  bool hasHeapStorage() const;
  
  bool hasHeapOrStackStorage() const;

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
  friend class MemRegionManager;

protected:
  MemRegionManager *Mgr;

  MemSpaceRegion(MemRegionManager *mgr) : MemRegion(MemSpaceRegionKind),
                                          Mgr(mgr) {}
  
  MemRegionManager* getMemRegionManager() const {
    return Mgr;
  }

public:
  void Profile(llvm::FoldingSetNodeID& ID) const;

  bool isBoundable() const { return false; }

  static bool classof(const MemRegion* R) {
    return R->getKind() == MemSpaceRegionKind;
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
    return R->getKind() > MemSpaceRegionKind;
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
    return T.getTypePtr() ? T->getDesugaredType() : T;
  }
  
  QualType getDesugaredLocationType(ASTContext& C) const {
    return getLocationType(C)->getDesugaredType();
  }

  bool isBoundable() const {
    return !getValueType(getContext()).isNull();
  }

  static bool classof(const MemRegion* R) {
    unsigned k = R->getKind();
    return k > BEG_TYPED_REGIONS && k < END_TYPED_REGIONS;
  }
};

/// CodeTextRegion - A region that represents code texts of a function. It wraps
/// two kinds of code texts: real function and symbolic function. Real function
/// is a function declared in the program. Symbolic function is a function
/// pointer that we don't know which function it points to.
class CodeTextRegion : public TypedRegion {
public:
  enum CodeKind { Declared, Symbolic };

private:
  // The function pointer kind that this CodeTextRegion represents.
  CodeKind codekind;

  // Data may be a SymbolRef or FunctionDecl*.
  const void* Data;

  // Cached function pointer type.
  QualType LocationType;

public:

  CodeTextRegion(const FunctionDecl* fd, QualType t, const MemRegion* sreg)
    : TypedRegion(sreg, CodeTextRegionKind), 
      codekind(Declared),
      Data(fd),
      LocationType(t) {}

  CodeTextRegion(SymbolRef sym, QualType t, const MemRegion* sreg)
    : TypedRegion(sreg, CodeTextRegionKind), 
      codekind(Symbolic),
      Data(sym),
      LocationType(t) {}

  QualType getValueType(ASTContext &C) const {
    // Do not get the object type of a CodeTextRegion.
    assert(0);
    return QualType();
  }

  QualType getLocationType(ASTContext &C) const {
    return LocationType;
  }

  bool isDeclared() const { return codekind == Declared; }
  bool isSymbolic() const { return codekind == Symbolic; }

  const FunctionDecl* getDecl() const {
    assert(codekind == Declared);
    return static_cast<const FunctionDecl*>(Data);
  }
  
  SymbolRef getSymbol() const {
    assert(codekind == Symbolic);
    return const_cast<SymbolRef>(static_cast<const SymbolRef>(Data));
  }
  
  bool isBoundable() const { return false; }
  
  virtual void dumpToStream(llvm::raw_ostream& os) const;

  void Profile(llvm::FoldingSetNodeID& ID) const;

  static void ProfileRegion(llvm::FoldingSetNodeID& ID, 
                            const void* data, QualType t, const MemRegion*);

  static bool classof(const MemRegion* R) {
    return R->getKind() == CodeTextRegionKind;
  }
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
    return k > BEG_DECL_REGIONS && k < END_DECL_REGIONS;
  }
};
  
class VarRegion : public DeclRegion {
  friend class MemRegionManager;

  // Data.
  const LocationContext *LC;
  
  // Constructors and private methods.
  VarRegion(const VarDecl* vd, const LocationContext *lC, const MemRegion* sReg)
    : DeclRegion(vd, sReg, VarRegionKind), LC(lC) {}

  static void ProfileRegion(llvm::FoldingSetNodeID& ID, const VarDecl* VD,
                            const LocationContext *LC,
                            const MemRegion *superRegion) {
    DeclRegion::ProfileRegion(ID, VD, superRegion, VarRegionKind);
    ID.AddPointer(LC);
  }
  
  void Profile(llvm::FoldingSetNodeID& ID) const;
  
public:  
  const VarDecl *getDecl() const { return cast<VarDecl>(D); }
  
  const LocationContext *getLocationContext() const { return LC; }
  
  QualType getValueType(ASTContext& C) const { 
    // FIXME: We can cache this if needed.
    return C.getCanonicalType(getDecl()->getType());
  }    
    
  void dumpToStream(llvm::raw_ostream& os) const;
  
  static bool classof(const MemRegion* R) {
    return R->getKind() == VarRegionKind;
  }  
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
  
class ObjCObjectRegion : public DeclRegion {
  
  friend class MemRegionManager;
  
  ObjCObjectRegion(const ObjCInterfaceDecl* ivd, const MemRegion* sReg)
  : DeclRegion(ivd, sReg, ObjCObjectRegionKind) {}
  
  static void ProfileRegion(llvm::FoldingSetNodeID& ID,
                            const ObjCInterfaceDecl* ivd,
                            const MemRegion* superRegion) {
    DeclRegion::ProfileRegion(ID, ivd, superRegion, ObjCObjectRegionKind);
  }
  
public:
  const ObjCInterfaceDecl* getInterface() const {
    return cast<ObjCInterfaceDecl>(D);
  }
  
  QualType getValueType(ASTContext& C) const {
    return C.getObjCInterfaceType(getInterface());
  }
  
  static bool classof(const MemRegion* R) {
    return R->getKind() == ObjCObjectRegionKind;
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
  
  MemSpaceRegion *globals;
  MemSpaceRegion *stack;
  MemSpaceRegion *stackArguments;
  MemSpaceRegion *heap;
  MemSpaceRegion *unknown;
  MemSpaceRegion *code;

public:
  MemRegionManager(ASTContext &c, llvm::BumpPtrAllocator& a)
    : C(c), A(a), globals(0), stack(0), stackArguments(0), heap(0),
      unknown(0), code(0) {}
  
  ~MemRegionManager() {}
  
  ASTContext &getContext() { return C; }
  
  /// getStackRegion - Retrieve the memory region associated with the
  ///  current stack frame.
  MemSpaceRegion *getStackRegion();

  /// getStackArgumentsRegion - Retrieve the memory region associated with
  ///  function/method arguments of the current stack frame.
  MemSpaceRegion *getStackArgumentsRegion();
  
  /// getGlobalsRegion - Retrieve the memory region associated with
  ///  all global variables.
  MemSpaceRegion *getGlobalsRegion();
  
  /// getHeapRegion - Retrieve the memory region associated with the
  ///  generic "heap".
  MemSpaceRegion *getHeapRegion();

  /// getUnknownRegion - Retrieve the memory region associated with unknown
  /// memory space.
  MemSpaceRegion *getUnknownRegion();

  MemSpaceRegion *getCodeRegion();

  /// getAllocaRegion - Retrieve a region associated with a call to alloca().
  AllocaRegion *getAllocaRegion(const Expr* Ex, unsigned Cnt);
  
  /// getCompoundLiteralRegion - Retrieve the region associated with a
  ///  given CompoundLiteral.
  CompoundLiteralRegion*
  getCompoundLiteralRegion(const CompoundLiteralExpr* CL);  
  
  /// getSymbolicRegion - Retrieve or create a "symbolic" memory region.
  SymbolicRegion* getSymbolicRegion(SymbolRef sym);

  StringRegion* getStringRegion(const StringLiteral* Str);

  /// getVarRegion - Retrieve or create the memory region associated with
  ///  a specified VarDecl and LocationContext.
  VarRegion* getVarRegion(const VarDecl *D, const LocationContext *LC);
  
  /// getElementRegion - Retrieve the memory region associated with the
  ///  associated element type, index, and super region.
  ElementRegion *getElementRegion(QualType elementType, SVal Idx,
                                  const MemRegion *superRegion,
                                  ASTContext &Ctx);
  
  ElementRegion *getElementRegionWithSuper(const ElementRegion *ER,
                                           const MemRegion *superRegion) {
    return getElementRegion(ER->getElementType(), ER->getIndex(),
                            superRegion, ER->getContext());
  }

  /// getFieldRegion - Retrieve or create the memory region associated with
  ///  a specified FieldDecl.  'superRegion' corresponds to the containing
  ///  memory region (which typically represents the memory representing
  ///  a structure or class).
  FieldRegion *getFieldRegion(const FieldDecl* fd,
                              const MemRegion* superRegion);
  
  FieldRegion *getFieldRegionWithSuper(const FieldRegion *FR,
                                       const MemRegion *superRegion) {
    return getFieldRegion(FR->getDecl(), superRegion);
  }
  
  /// getObjCObjectRegion - Retrieve or create the memory region associated with
  ///  the instance of a specified Objective-C class.
  ObjCObjectRegion* getObjCObjectRegion(const ObjCInterfaceDecl* ID,
                                  const MemRegion* superRegion);
  
  /// getObjCIvarRegion - Retrieve or create the memory region associated with
  ///   a specified Objective-c instance variable.  'superRegion' corresponds
  ///   to the containing region (which typically represents the Objective-C
  ///   object).
  ObjCIvarRegion* getObjCIvarRegion(const ObjCIvarDecl* ivd,
                                    const MemRegion* superRegion);
  
  CodeTextRegion* getCodeTextRegion(SymbolRef sym, QualType t);
  CodeTextRegion* getCodeTextRegion(const FunctionDecl* fd, QualType t);
  
  template <typename RegionTy, typename A1>
  RegionTy* getRegion(const A1 a1);
  
  template <typename RegionTy, typename A1>
  RegionTy* getSubRegion(const A1 a1, const MemRegion* superRegion);
  
  template <typename RegionTy, typename A1, typename A2>
  RegionTy* getRegion(const A1 a1, const A2 a2);

  bool isGlobalsRegion(const MemRegion* R) { 
    assert(R);
    return R == globals; 
  }
  
private:
  MemSpaceRegion* LazyAllocate(MemSpaceRegion*& region);
};
  
//===----------------------------------------------------------------------===//
// Out-of-line member definitions.
//===----------------------------------------------------------------------===//

inline ASTContext& MemRegion::getContext() const {
  return getMemRegionManager()->getContext();
}
  
template<typename RegionTy> struct MemRegionManagerTrait;
  
template <typename RegionTy, typename A1>
RegionTy* MemRegionManager::getRegion(const A1 a1) {

  const typename MemRegionManagerTrait<RegionTy>::SuperRegionTy *superRegion =
    MemRegionManagerTrait<RegionTy>::getSuperRegion(*this, a1);
  
  llvm::FoldingSetNodeID ID;  
  RegionTy::ProfileRegion(ID, a1, superRegion);  
  void* InsertPos;
  RegionTy* R = cast_or_null<RegionTy>(Regions.FindNodeOrInsertPos(ID,
                                                                   InsertPos));
  
  if (!R) {
    R = (RegionTy*) A.Allocate<RegionTy>();
    new (R) RegionTy(a1, superRegion);
    Regions.InsertNode(R, InsertPos);
  }
  
  return R;
}

template <typename RegionTy, typename A1>
RegionTy* MemRegionManager::getSubRegion(const A1 a1,
                                         const MemRegion *superRegion) {
  llvm::FoldingSetNodeID ID;  
  RegionTy::ProfileRegion(ID, a1, superRegion);  
  void* InsertPos;
  RegionTy* R = cast_or_null<RegionTy>(Regions.FindNodeOrInsertPos(ID,
                                                                   InsertPos));
  
  if (!R) {
    R = (RegionTy*) A.Allocate<RegionTy>();
    new (R) RegionTy(a1, superRegion);
    Regions.InsertNode(R, InsertPos);
  }
  
  return R;
}
  
template <typename RegionTy, typename A1, typename A2>
RegionTy* MemRegionManager::getRegion(const A1 a1, const A2 a2) {
  
  const typename MemRegionManagerTrait<RegionTy>::SuperRegionTy *superRegion =
    MemRegionManagerTrait<RegionTy>::getSuperRegion(*this, a1, a2);
  
  llvm::FoldingSetNodeID ID;  
  RegionTy::ProfileRegion(ID, a1, a2, superRegion);  
  void* InsertPos;
  RegionTy* R = cast_or_null<RegionTy>(Regions.FindNodeOrInsertPos(ID,
                                                                   InsertPos));
  
  if (!R) {
    R = (RegionTy*) A.Allocate<RegionTy>();
    new (R) RegionTy(a1, a2, superRegion);
    Regions.InsertNode(R, InsertPos);
  }
  
  return R;
}
  
//===----------------------------------------------------------------------===//
// Traits for constructing regions.
//===----------------------------------------------------------------------===//

template <> struct MemRegionManagerTrait<AllocaRegion> {
  typedef MemRegion SuperRegionTy;
  static const SuperRegionTy* getSuperRegion(MemRegionManager& MRMgr,
                                             const Expr *, unsigned) {
    return MRMgr.getStackRegion();
  }
};  
  
template <> struct MemRegionManagerTrait<CompoundLiteralRegion> {
  typedef MemRegion SuperRegionTy;
  static const SuperRegionTy* getSuperRegion(MemRegionManager& MRMgr,
                                             const CompoundLiteralExpr *CL) {
    
    return CL->isFileScope() ? MRMgr.getGlobalsRegion() 
                             : MRMgr.getStackRegion();
  }
};
  
template <> struct MemRegionManagerTrait<StringRegion> {
  typedef MemSpaceRegion SuperRegionTy;
  static const SuperRegionTy* getSuperRegion(MemRegionManager& MRMgr,
                                             const StringLiteral*) {
    return MRMgr.getGlobalsRegion();
  }
};
  
template <> struct MemRegionManagerTrait<VarRegion> {
  typedef MemRegion SuperRegionTy;
  static const SuperRegionTy* getSuperRegion(MemRegionManager &MRMgr,
                                             const VarDecl *D,
                                             const LocationContext *LC) {
    
    // FIXME: Make stack regions have a location context?
    
    if (D->hasLocalStorage()) {
      return isa<ParmVarDecl>(D) || isa<ImplicitParamDecl>(D)
             ? MRMgr.getStackArgumentsRegion() : MRMgr.getStackRegion();
    }
    
    return MRMgr.getGlobalsRegion();
  }
};
  
template <> struct MemRegionManagerTrait<SymbolicRegion> {
  typedef MemRegion SuperRegionTy;
  static const SuperRegionTy* getSuperRegion(MemRegionManager& MRMgr,
                                             SymbolRef) {
    return MRMgr.getUnknownRegion();
  }
};

template<> struct MemRegionManagerTrait<CodeTextRegion> {
  typedef MemSpaceRegion SuperRegionTy;
  static const SuperRegionTy* getSuperRegion(MemRegionManager& MRMgr,
                                             const FunctionDecl*, QualType) {
    return MRMgr.getCodeRegion();
  }
  static const SuperRegionTy* getSuperRegion(MemRegionManager& MRMgr,
                                             SymbolRef, QualType) {
    return MRMgr.getCodeRegion();
  }
};
  
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
