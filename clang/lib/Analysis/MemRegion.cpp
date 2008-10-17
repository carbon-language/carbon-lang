//== MemRegion.cpp - Abstract memory regions for static analysis --*- C++ -*--//
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

#include "llvm/Support/raw_ostream.h"
#include "clang/Analysis/PathSensitive/MemRegion.h"

using namespace clang;


MemRegion::~MemRegion() {}

void MemSpaceRegion::Profile(llvm::FoldingSetNodeID& ID) const {
  ID.AddInteger((unsigned)getKind());
}

void AnonTypedRegion::ProfileRegion(llvm::FoldingSetNodeID& ID, QualType T,
                                    const MemRegion* superRegion) {
  ID.AddInteger((unsigned) AnonTypedRegionKind);
  ID.Add(T);
  ID.AddPointer(superRegion);
}

void AnonPointeeRegion::ProfileRegion(llvm::FoldingSetNodeID& ID, 
                                      const VarDecl* VD, QualType T,
                                      const MemRegion* superRegion) {
  ID.AddInteger((unsigned) AnonPointeeRegionKind);
  ID.Add(T);
  ID.AddPointer(VD);
  ID.AddPointer(superRegion);
}

void AnonTypedRegion::Profile(llvm::FoldingSetNodeID& ID) const {
  AnonTypedRegion::ProfileRegion(ID, T, superRegion);
}

void DeclRegion::ProfileRegion(llvm::FoldingSetNodeID& ID, const Decl* D,
                               const MemRegion* superRegion, Kind k) {
  ID.AddInteger((unsigned) k);
  ID.AddPointer(D);
  ID.AddPointer(superRegion);
}

void DeclRegion::Profile(llvm::FoldingSetNodeID& ID) const {
  DeclRegion::ProfileRegion(ID, D, superRegion, getKind());
}

void SymbolicRegion::ProfileRegion(llvm::FoldingSetNodeID& ID, SymbolID sym) {
  ID.AddInteger((unsigned) MemRegion::SymbolicRegionKind);
  ID.AddInteger(sym.getNumber());
}

void SymbolicRegion::Profile(llvm::FoldingSetNodeID& ID) const {
  SymbolicRegion::ProfileRegion(ID, sym);
}

//===----------------------------------------------------------------------===//
// Region pretty-printing.
//===----------------------------------------------------------------------===//

std::string MemRegion::getString() const {
  std::string s;
  llvm::raw_string_ostream os(s);
  print(os);
  return os.str();
}

void MemRegion::print(llvm::raw_ostream& os) const {
  os << "<Unknown Region>";
}

void VarRegion::print(llvm::raw_ostream& os) const {
  os << cast<VarDecl>(D)->getName();
}

void SymbolicRegion::print(llvm::raw_ostream& os) const {
  os << "$" << sym.getNumber();
}

void FieldRegion::print(llvm::raw_ostream& os) const {
  superRegion->print(os);
  os << "->" << getDecl()->getName();
}

//===----------------------------------------------------------------------===//
// MemRegionManager methods.
//===----------------------------------------------------------------------===//
  
MemSpaceRegion* MemRegionManager::LazyAllocate(MemSpaceRegion*& region) {
  
  if (!region) {  
    region = (MemSpaceRegion*) A.Allocate<MemSpaceRegion>();
    new (region) MemSpaceRegion();
  }
  
  return region;
}

MemSpaceRegion* MemRegionManager::getStackRegion() {
  return LazyAllocate(stack);
}

MemSpaceRegion* MemRegionManager::getGlobalsRegion() {
  return LazyAllocate(globals);
}

MemSpaceRegion* MemRegionManager::getHeapRegion() {
  return LazyAllocate(heap);
}

MemSpaceRegion* MemRegionManager::getUnknownRegion() {
  return LazyAllocate(unknown);
}

VarRegion* MemRegionManager::getVarRegion(const VarDecl* d,
                                          const MemRegion* superRegion) {
  llvm::FoldingSetNodeID ID;
  DeclRegion::ProfileRegion(ID, d, superRegion, MemRegion::VarRegionKind);
  
  void* InsertPos;
  MemRegion* data = Regions.FindNodeOrInsertPos(ID, InsertPos);
  VarRegion* R = cast_or_null<VarRegion>(data);
  
  if (!R) {
    R = (VarRegion*) A.Allocate<VarRegion>();
    new (R) VarRegion(d, superRegion);
    Regions.InsertNode(R, InsertPos);
  }
  
  return R;
}

/// getSymbolicRegion - Retrieve or create a "symbolic" memory region.
SymbolicRegion* MemRegionManager::getSymbolicRegion(const SymbolID sym) {
  
  llvm::FoldingSetNodeID ID;
  SymbolicRegion::ProfileRegion(ID, sym);
  
  void* InsertPos;
  MemRegion* data = Regions.FindNodeOrInsertPos(ID, InsertPos);
  SymbolicRegion* R = cast_or_null<SymbolicRegion>(data);
  
  if (!R) {
    R = (SymbolicRegion*) A.Allocate<SymbolicRegion>();
    new (R) SymbolicRegion(sym);
    Regions.InsertNode(R, InsertPos);
  }
  
  return R;  
}

FieldRegion* MemRegionManager::getFieldRegion(const FieldDecl* d,
                                              const MemRegion* superRegion) {
  llvm::FoldingSetNodeID ID;
  DeclRegion::ProfileRegion(ID, d, superRegion, MemRegion::FieldRegionKind);
  
  void* InsertPos;
  MemRegion* data = Regions.FindNodeOrInsertPos(ID, InsertPos);
  FieldRegion* R = cast_or_null<FieldRegion>(data);
  
  if (!R) {
    R = (FieldRegion*) A.Allocate<FieldRegion>();
    new (R) FieldRegion(d, superRegion);
    Regions.InsertNode(R, InsertPos);
  }
  
  return R;
}

ObjCIvarRegion*
MemRegionManager::getObjCIvarRegion(const ObjCIvarDecl* d,
                                    const MemRegion* superRegion) {
  llvm::FoldingSetNodeID ID;
  DeclRegion::ProfileRegion(ID, d, superRegion, MemRegion::ObjCIvarRegionKind);
  
  void* InsertPos;
  MemRegion* data = Regions.FindNodeOrInsertPos(ID, InsertPos);
  ObjCIvarRegion* R = cast_or_null<ObjCIvarRegion>(data);
  
  if (!R) {
    R = (ObjCIvarRegion*) A.Allocate<ObjCIvarRegion>();
    new (R) ObjCIvarRegion(d, superRegion);
    Regions.InsertNode(R, InsertPos);
  }
  
  return R;
}

AnonPointeeRegion* MemRegionManager::getAnonPointeeRegion(const VarDecl* d) {
  llvm::FoldingSetNodeID ID;
  QualType T = d->getType();
  QualType PointeeType = cast<PointerType>(T.getTypePtr())->getPointeeType();
  MemRegion* superRegion = getUnknownRegion();

  AnonPointeeRegion::ProfileRegion(ID, d, PointeeType, superRegion);

  void* InsertPos;
  MemRegion* data = Regions.FindNodeOrInsertPos(ID, InsertPos);
  AnonPointeeRegion* R = cast_or_null<AnonPointeeRegion>(data);

  if (!R) {
    R = (AnonPointeeRegion*) A.Allocate<AnonPointeeRegion>();
    new (R) AnonPointeeRegion(d, PointeeType, superRegion);
    Regions.InsertNode(R, InsertPos);
  }

  return R;
}

bool MemRegionManager::hasStackStorage(const MemRegion* R) {
  const SubRegion* SR = dyn_cast<SubRegion>(R);

  // Only subregions can have stack storage.
  if (!SR)
    return false;
  
  MemSpaceRegion* S = getStackRegion();
  
  while (SR) {
    R = SR->getSuperRegion();
    if (R == S)
      return true;
    
    SR = dyn_cast<SubRegion>(R);    
  }
  
  return false;
}
