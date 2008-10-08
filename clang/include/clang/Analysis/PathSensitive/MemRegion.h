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

#include "llvm/Support/Casting.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/Support/Allocator.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclObjC.h"
#include <string>

namespace llvm { class raw_ostream; }

namespace clang {
 
class MemRegionManager;
  
      
/// MemRegion - The root abstract class for all memory regions.
class MemRegion : public llvm::FoldingSetNode {
public:
  enum Kind { MemSpaceRegionKind,
              // Typed regions.
              BEG_TYPED_REGIONS,
              VarRegionKind, FieldRegionKind, ObjCIvarRegionKind,
              AnonTypedRegionKind, AnonPointeeRegionKind,
              END_TYPED_REGIONS };  
private:
  const Kind kind;
  
protected:
  MemRegion(Kind k) : kind(k) {}
  virtual ~MemRegion();

public:
  // virtual MemExtent getExtent(MemRegionManager& mrm) const = 0;
  virtual const MemRegion* getSuperRegion() const = 0;
  virtual void Profile(llvm::FoldingSetNodeID& ID) const = 0;
  
  std::string getString() const;
  virtual void print(llvm::raw_ostream& os) const;  
  
  Kind getKind() const { return kind; }  
  static bool classof(const MemRegion*) { return true; }
};
  
/// MemSpaceRegion - A memory region that represents and "memory space";
///  for example, the set of global variables, the stack frame, etc.
class MemSpaceRegion : public MemRegion {
  friend class MemRegionManager;
  MemSpaceRegion() : MemRegion(MemSpaceRegionKind) {}
  
public:
  //RegionExtent getExtent() const { return UndefinedExtent(); }

  const MemRegion* getSuperRegion() const {
    return 0;
  }
    
  //static void ProfileRegion(llvm::FoldingSetNodeID& ID);
  void Profile(llvm::FoldingSetNodeID& ID) const;

  static bool classof(const MemRegion* R) {
    return R->getKind() == MemSpaceRegionKind;
  }
};

/// TypedRegion - An abstract class representing regions that are typed.
class TypedRegion : public MemRegion {
protected:
  const MemRegion* superRegion;

  TypedRegion(const MemRegion* sReg, Kind k)
    : MemRegion(k), superRegion(sReg) {};
  
public:
  virtual QualType getType() const = 0;
  
  // MemExtent getExtent(MemRegionManager& mrm) const;
  const MemRegion* getSuperRegion() const {
    return superRegion;
  }
  
  static bool classof(const MemRegion* R) {
    unsigned k = R->getKind();
    return k > BEG_TYPED_REGIONS && k < END_TYPED_REGIONS;
  }
};

/// AnonTypedRegion - An "anonymous" region that simply types a chunk
///  of memory.
class AnonTypedRegion : public TypedRegion {
protected:
  QualType T;

  friend class MemRegionManager;
  
  AnonTypedRegion(QualType t, MemRegion* sreg)
    : TypedRegion(sreg, AnonTypedRegionKind), T(t) {}

  static void ProfileRegion(llvm::FoldingSetNodeID& ID, QualType T,
                            const MemRegion* superRegion);

public:
  QualType getType() const { return T; }
  

  void Profile(llvm::FoldingSetNodeID& ID) const;
  
  static bool classof(const MemRegion* R) {
    return R->getKind() == AnonTypedRegionKind;
  }
};

/// AnonPointeeRegion - anonymous regions pointed-at by pointer function
///  parameters or pointer globals. In RegionStoreManager, we assume pointer
///  parameters or globals point at some anonymous region initially. Such
///  regions are not the regions associated with the pointers themselves, but
///  are identified with the VarDecl of the parameters or globals.
class AnonPointeeRegion : public AnonTypedRegion {
  friend class MemRegionManager;
  // VD - the pointer variable that points at this region.
  const VarDecl* VD;

  AnonPointeeRegion(const VarDecl* d, QualType t, MemRegion* sreg)
    : AnonTypedRegion(t, sreg), VD(d) {}

public:
  static void ProfileRegion(llvm::FoldingSetNodeID& ID, const VarDecl* PVD,
                            QualType T, const MemRegion* superRegion);
};

/// AnonHeapRegion - anonymous region created by malloc().
class AnonHeapRegion : public AnonTypedRegion {
};

class DeclRegion : public TypedRegion {
protected:
  const Decl* D;

  DeclRegion(const Decl* d, MemRegion* sReg, Kind k)
    : TypedRegion(sReg, k), D(d) {}

  static void ProfileRegion(llvm::FoldingSetNodeID& ID, const Decl* D,
                      const MemRegion* superRegion, Kind k);
  
public:  
  void Profile(llvm::FoldingSetNodeID& ID) const;
};
  
class VarRegion : public DeclRegion {
  friend class MemRegionManager;
  
  VarRegion(const VarDecl* vd, MemRegion* sReg)
    : DeclRegion(vd, sReg, VarRegionKind) {}

  static void ProfileRegion(llvm::FoldingSetNodeID& ID, VarDecl* VD,
                      const MemRegion* superRegion) {
    DeclRegion::ProfileRegion(ID, VD, superRegion, VarRegionKind);
  }
  
public:  
  const VarDecl* getDecl() const { return cast<VarDecl>(D); }
  QualType getType() const { return getDecl()->getType(); }
  
  void print(llvm::raw_ostream& os) const;
  
  static bool classof(const MemRegion* R) {
    return R->getKind() == VarRegionKind;
  }  
};

class FieldRegion : public DeclRegion {
  friend class MemRegionManager;

  FieldRegion(const FieldDecl* fd, MemRegion* sReg)
    : DeclRegion(fd, sReg, FieldRegionKind) {}

  static void ProfileRegion(llvm::FoldingSetNodeID& ID, FieldDecl* FD,
                      const MemRegion* superRegion) {
    DeclRegion::ProfileRegion(ID, FD, superRegion, FieldRegionKind);
  }
  
public:
  const FieldDecl* getDecl() const { return cast<FieldDecl>(D); }
  QualType getType() const { return getDecl()->getType(); }
    
  static bool classof(const MemRegion* R) {
    return R->getKind() == FieldRegionKind;
  }
};
  
class ObjCIvarRegion : public DeclRegion {
  
  friend class MemRegionManager;
  
  ObjCIvarRegion(const ObjCIvarDecl* ivd, MemRegion* sReg)
    : DeclRegion(ivd, sReg, ObjCIvarRegionKind) {}

  static void ProfileRegion(llvm::FoldingSetNodeID& ID, ObjCIvarDecl* ivd,
                      const MemRegion* superRegion) {
    DeclRegion::ProfileRegion(ID, ivd, superRegion, ObjCIvarRegionKind);
  }
  
public:
  const ObjCIvarDecl* getDecl() const { return cast<ObjCIvarDecl>(D); }
  QualType getType() const { return getDecl()->getType(); }
  
  static bool classof(const MemRegion* R) {
    return R->getKind() == ObjCIvarRegionKind;
  }
};
  
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
  
public:
  MemRegionManager(llvm::BumpPtrAllocator& a)
  : A(a), globals(0), stack(0), heap(0) {}
  
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
  
  /// getVarRegion - Retrieve or create the memory region associated with
  ///  a specified VarDecl.  'superRegion' corresponds to the containing
  ///  memory region, and 'off' is the offset within the containing region.
  VarRegion* getVarRegion(const VarDecl* vd, MemRegion* superRegion);
  
  VarRegion* getVarRegion(const VarDecl* vd) {
    return getVarRegion(vd, vd->hasLocalStorage() ? getStackRegion() 
                        : getGlobalsRegion());
  }
  
  /// getFieldRegion - Retrieve or create the memory region associated with
  ///  a specified FieldDecl.  'superRegion' corresponds to the containing
  ///  memory region (which typically represents the memory representing
  ///  a structure or class).
  FieldRegion* getFieldRegion(const FieldDecl* fd, MemRegion* superRegion);
  
  /// getObjCIvarRegion - Retrieve or create the memory region associated with
  ///   a specified Objective-c instance variable.  'superRegion' corresponds
  ///   to the containing region (which typically represents the Objective-C
  ///   object).
  ObjCIvarRegion* getObjCIvarRegion(const ObjCIvarDecl* ivd,
                                    MemRegion* superRegion);

  AnonPointeeRegion* getAnonPointeeRegion(const VarDecl* d);

  bool hasStackStorage(const MemRegion* R);
  
private:
  MemSpaceRegion* LazyAllocate(MemSpaceRegion*& region);
};


  
} // end clang namespace
#endif
