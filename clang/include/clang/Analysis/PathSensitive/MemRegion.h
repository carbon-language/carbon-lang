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
               TypedViewRegionKind,
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
  // virtual MemExtent getExtent(MemRegionManager& mrm) const = 0;
  virtual void Profile(llvm::FoldingSetNodeID& ID) const = 0;
  
  std::string getString() const;

  virtual void print(llvm::raw_ostream& os) const;  
  
  Kind getKind() const { return kind; }  
  
  template<typename RegionTy> const RegionTy* getAs() const;
  
  virtual bool isBoundable(ASTContext&) const { return true; }

  static bool classof(const MemRegion*) { return true; }
};
  
/// MemSpaceRegion - A memory region that represents and "memory space";
///  for example, the set of global variables, the stack frame, etc.
class MemSpaceRegion : public MemRegion {
  friend class MemRegionManager;
  MemSpaceRegion() : MemRegion(MemSpaceRegionKind) {}
  
public:
  //RegionExtent getExtent() const { return UndefinedExtent(); }

  void Profile(llvm::FoldingSetNodeID& ID) const;

  bool isBoundable(ASTContext &) const { return false; }

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

  bool isSubRegionOf(const MemRegion* R) const;

  static bool classof(const MemRegion* R) {
    return R->getKind() > MemSpaceRegionKind;
  }
};

/// AllocaRegion - A region that represents an untyped blob of bytes created
///  by a call to 'alloca'.
class AllocaRegion : public SubRegion {
  friend class MemRegionManager;
protected:
  unsigned Cnt; // Block counter.  Used to distinguish different pieces of
                // memory allocated by alloca at the same call site.
  const Expr* Ex;

  AllocaRegion(const Expr* ex, unsigned cnt, const MemRegion* superRegion)
    : SubRegion(superRegion, AllocaRegionKind), Cnt(cnt), Ex(ex) {}
  
public:
  
  const Expr* getExpr() const { return Ex; }
  
  void Profile(llvm::FoldingSetNodeID& ID) const;

  static void ProfileRegion(llvm::FoldingSetNodeID& ID, const Expr* Ex,
                            unsigned Cnt);
  
  void print(llvm::raw_ostream& os) const;
  
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

  bool isBoundable(ASTContext &C) const {
    return !getValueType(C).isNull();
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
  
  bool isBoundable(ASTContext&) const { return false; }
  
  virtual void print(llvm::raw_ostream& os) const;

  void Profile(llvm::FoldingSetNodeID& ID) const;

  static void ProfileRegion(llvm::FoldingSetNodeID& ID, 
                            const void* data, QualType t);

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

  void Profile(llvm::FoldingSetNodeID& ID) const;

  static void ProfileRegion(llvm::FoldingSetNodeID& ID, SymbolRef sym);
  
  void print(llvm::raw_ostream& os) const;
  
  static bool classof(const MemRegion* R) {
    return R->getKind() == SymbolicRegionKind;
  }
};  

/// StringRegion - Region associated with a StringLiteral.
class StringRegion : public TypedRegion {
  friend class MemRegionManager;
  const StringLiteral* Str;
protected:

  StringRegion(const StringLiteral* str, MemRegion* sreg)
    : TypedRegion(sreg, StringRegionKind), Str(str) {}

  static void ProfileRegion(llvm::FoldingSetNodeID& ID,
                            const StringLiteral* Str,
                            const MemRegion* superRegion);

public:

  const StringLiteral* getStringLiteral() const { return Str; }
    
  QualType getValueType(ASTContext& C) const {
    return Str->getType();
  }

  void Profile(llvm::FoldingSetNodeID& ID) const {
    ProfileRegion(ID, Str, superRegion);
  }

  void print(llvm::raw_ostream& os) const;

  static bool classof(const MemRegion* R) {
    return R->getKind() == StringRegionKind;
  }
};

class TypedViewRegion : public TypedRegion {
  friend class MemRegionManager;
  QualType LValueType;

  TypedViewRegion(QualType lvalueType, const MemRegion* sreg)
    : TypedRegion(sreg, TypedViewRegionKind), LValueType(lvalueType) {}

  static void ProfileRegion(llvm::FoldingSetNodeID& ID, QualType T, 
                            const MemRegion* superRegion);

public:

  void print(llvm::raw_ostream& os) const;
  
  QualType getLocationType(ASTContext&) const {
    return LValueType;
  }

  QualType getValueType(ASTContext&) const {
    const PointerType* PTy = LValueType->getAsPointerType();
    assert(PTy);
    return PTy->getPointeeType();
  }
  
  bool isBoundable(ASTContext &C) const {
    return isa<PointerType>(LValueType);
  }  

  void Profile(llvm::FoldingSetNodeID& ID) const {
    ProfileRegion(ID, LValueType, superRegion);
  }

  static bool classof(const MemRegion* R) {
    return R->getKind() == TypedViewRegionKind;
  }
  
  const MemRegion *removeViews() const;
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
  
  void Profile(llvm::FoldingSetNodeID& ID) const;
  
  void print(llvm::raw_ostream& os) const;

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
  
  VarRegion(const VarDecl* vd, const MemRegion* sReg)
    : DeclRegion(vd, sReg, VarRegionKind) {}

  static void ProfileRegion(llvm::FoldingSetNodeID& ID, VarDecl* VD,
                      const MemRegion* superRegion) {
    DeclRegion::ProfileRegion(ID, VD, superRegion, VarRegionKind);
  }
  
public:  
  const VarDecl* getDecl() const { return cast<VarDecl>(D); }  
  
  QualType getValueType(ASTContext& C) const { 
    // FIXME: We can cache this if needed.
    return C.getCanonicalType(getDecl()->getType());
  }    
    
  void print(llvm::raw_ostream& os) const;
  
  static bool classof(const MemRegion* R) {
    return R->getKind() == VarRegionKind;
  }  
};

class FieldRegion : public DeclRegion {
  friend class MemRegionManager;

  FieldRegion(const FieldDecl* fd, const MemRegion* sReg)
    : DeclRegion(fd, sReg, FieldRegionKind) {}

public:
  
  void print(llvm::raw_ostream& os) const;
  
  const FieldDecl* getDecl() const { return cast<FieldDecl>(D); }
    
  QualType getValueType(ASTContext& C) const { 
    // FIXME: We can cache this if needed.
    return C.getCanonicalType(getDecl()->getType());
  }    

  static void ProfileRegion(llvm::FoldingSetNodeID& ID, FieldDecl* FD,
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
  
  static void ProfileRegion(llvm::FoldingSetNodeID& ID, ObjCInterfaceDecl* ivd,
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

  static void ProfileRegion(llvm::FoldingSetNodeID& ID, ObjCIvarDecl* ivd,
                      const MemRegion* superRegion) {
    DeclRegion::ProfileRegion(ID, ivd, superRegion, ObjCIvarRegionKind);
  }
  
public:
  const ObjCIvarDecl* getDecl() const { return cast<ObjCIvarDecl>(D); }
  QualType getValueType(ASTContext&) const { return getDecl()->getType(); }
  
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
  
  void print(llvm::raw_ostream& os) const;

  void Profile(llvm::FoldingSetNodeID& ID) const;

  static bool classof(const MemRegion* R) {
    return R->getKind() == ElementRegionKind;
  }
};
  
template<typename RegionTy>
const RegionTy* MemRegion::getAs() const {
  const MemRegion *R = this;
  
  do {
    if (const RegionTy* RT = dyn_cast<RegionTy>(R))
      return RT;
    
    if (const TypedViewRegion *TR = dyn_cast<TypedViewRegion>(R)) {
      R = TR->getSuperRegion();
      continue;
    }
    
    break;
  }
  while (R);
  
  return 0;
}

//===----------------------------------------------------------------------===//
// MemRegionManager - Factory object for creating regions.
//===----------------------------------------------------------------------===//

class MemRegionManager {
  llvm::BumpPtrAllocator& A;
  llvm::FoldingSet<MemRegion> Regions;
  
  MemSpaceRegion* globals;
  MemSpaceRegion* stack;
  MemSpaceRegion* heap;
  MemSpaceRegion* unknown;
  MemSpaceRegion* code;

public:
  MemRegionManager(llvm::BumpPtrAllocator& a)
    : A(a), globals(0), stack(0), heap(0), unknown(0), code(0) {}
  
  ~MemRegionManager() {}
  
  /// getStackRegion - Retrieve the memory region associated with the
  ///  current stack frame.
  MemSpaceRegion* getStackRegion();
  
  /// getGlobalsRegion - Retrieve the memory region associated with
  ///  all global variables.
  MemSpaceRegion* getGlobalsRegion();
  
  /// getHeapRegion - Retrieve the memory region associated with the
  ///  generic "heap".
  MemSpaceRegion* getHeapRegion();

  /// getUnknownRegion - Retrieve the memory region associated with unknown
  /// memory space.
  MemSpaceRegion* getUnknownRegion();

  MemSpaceRegion* getCodeRegion();

  bool isGlobalsRegion(const MemRegion* R) { 
    assert(R);
    return R == globals; 
  }

  /// onStack - check if the region is allocated on the stack.
  bool onStack(const MemRegion* R);

  /// onHeap - check if the region is allocated on the heap, usually by malloc.
  bool onHeap(const MemRegion* R);
  
  /// getAllocaRegion - Retrieve a region associated with a call to alloca().
  AllocaRegion* getAllocaRegion(const Expr* Ex, unsigned Cnt);
  
  /// getCompoundLiteralRegion - Retrieve the region associated with a
  ///  given CompoundLiteral.
  CompoundLiteralRegion*
  getCompoundLiteralRegion(const CompoundLiteralExpr* CL);  
  
  /// getSymbolicRegion - Retrieve or create a "symbolic" memory region.
  SymbolicRegion* getSymbolicRegion(SymbolRef sym);

  StringRegion* getStringRegion(const StringLiteral* Str);

  /// getVarRegion - Retrieve or create the memory region associated with
  ///  a specified VarDecl.
  VarRegion* getVarRegion(const VarDecl* vd);
  
  /// getElementRegion - Retrieve the memory region associated with the
  ///  associated element type, index, and super region.
  ElementRegion* getElementRegion(QualType elementType, SVal Idx,
                                  const MemRegion* superRegion);

  /// getFieldRegion - Retrieve or create the memory region associated with
  ///  a specified FieldDecl.  'superRegion' corresponds to the containing
  ///  memory region (which typically represents the memory representing
  ///  a structure or class).
  FieldRegion* getFieldRegion(const FieldDecl* fd,
                              const MemRegion* superRegion);
  
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

  TypedViewRegion* getTypedViewRegion(QualType LValueType,
                                      const MemRegion* superRegion);

  CodeTextRegion* getCodeTextRegion(SymbolRef sym, QualType t);
  CodeTextRegion* getCodeTextRegion(const FunctionDecl* fd, QualType t);

  bool hasStackStorage(const MemRegion* R);

private:
  MemSpaceRegion* LazyAllocate(MemSpaceRegion*& region);
};
} // end clang namespace

namespace llvm {
static inline raw_ostream& operator<<(raw_ostream& O,
                                      const clang::MemRegion* R) { 
  R->print(O);
  return O;
}
} // end llvm namespace

#endif
