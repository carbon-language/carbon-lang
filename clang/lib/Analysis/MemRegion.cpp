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

void StringRegion::ProfileRegion(llvm::FoldingSetNodeID& ID, 
                                 const StringLiteral* Str, 
                                 const MemRegion* superRegion) {
  ID.AddInteger((unsigned) StringRegionKind);
  ID.AddPointer(Str);
  ID.AddPointer(superRegion);
}

void AllocaRegion::ProfileRegion(llvm::FoldingSetNodeID& ID,
                                 const Expr* Ex, unsigned cnt) {
  ID.AddInteger((unsigned) AllocaRegionKind);
  ID.AddPointer(Ex);
  ID.AddInteger(cnt);
}

void AllocaRegion::Profile(llvm::FoldingSetNodeID& ID) const {
  ProfileRegion(ID, Ex, Cnt);
}

QualType AnonPointeeRegion::getType(ASTContext& C) const {
  QualType T = C.getCanonicalType(Pointer->getType());
  PointerType* PTy = cast<PointerType>(T.getTypePtr());

  QualType PointeeTy = C.getCanonicalType(PTy->getPointeeType());
  return PointeeTy;
}

void AnonPointeeRegion::ProfileRegion(llvm::FoldingSetNodeID& ID, 
                                      const VarDecl* VD,
                                      const MemRegion* superRegion) {
  ID.AddInteger((unsigned) AnonPointeeRegionKind);
  ID.AddPointer(VD);
  ID.AddPointer(superRegion);
}

void CompoundLiteralRegion::Profile(llvm::FoldingSetNodeID& ID) const {
  CompoundLiteralRegion::ProfileRegion(ID, CL, superRegion);
}

void CompoundLiteralRegion::ProfileRegion(llvm::FoldingSetNodeID& ID,
                                          const CompoundLiteralExpr* CL,
                                          const MemRegion* superRegion) {
  ID.AddInteger((unsigned) CompoundLiteralRegionKind);
  ID.AddPointer(CL);
  ID.AddPointer(superRegion);
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

void ElementRegion::ProfileRegion(llvm::FoldingSetNodeID& ID, SVal Idx, 
                                  const MemRegion* superRegion) {
  ID.AddInteger(MemRegion::ElementRegionKind);
  ID.AddPointer(superRegion);
  Idx.Profile(ID);
}

void ElementRegion::Profile(llvm::FoldingSetNodeID& ID) const {
  ElementRegion::ProfileRegion(ID, Index, superRegion);
}

QualType ElementRegion::getType(ASTContext& C) const {
  QualType T = cast<TypedRegion>(superRegion)->getType(C);
  ArrayType* AT = cast<ArrayType>(T.getTypePtr());
  return AT->getElementType();
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

void AllocaRegion::print(llvm::raw_ostream& os) const {
  os << "alloca{" << (void*) Ex << ',' << Cnt << '}';
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

void ElementRegion::print(llvm::raw_ostream& os) const {
  superRegion->print(os);
  os << '['; Index.print(os); os << ']';
}

void CompoundLiteralRegion::print(llvm::raw_ostream& os) const {
  // FIXME: More elaborate pretty-printing.
  os << "{ " << (void*) CL <<  " }";
}

void StringRegion::print(llvm::raw_ostream& os) const {
  if (Str->isWide()) os << 'L';
  os << '"';

  // FIXME: this doesn't print wstrings right.
  for (unsigned i = 0, e = Str->getByteLength(); i != e; ++i) {
    switch (Str->getStrData()[i]) {
    default: os << Str->getStrData()[i]; break;
    // Handle some common ones to make dumps prettier.
    case '\\': os << "\\\\"; break;
    case '"': os << "\\\""; break;
    case '\n': os << "\\n"; break;
    case '\t': os << "\\t"; break;
    case '\a': os << "\\a"; break;
    case '\b': os << "\\b"; break;
    }
  }
  os << '"';
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

StringRegion* MemRegionManager::getStringRegion(const StringLiteral* Str) {
  llvm::FoldingSetNodeID ID;
  MemSpaceRegion* GlobalsR = getGlobalsRegion();

  StringRegion::ProfileRegion(ID, Str, GlobalsR);

  void* InsertPos;
  MemRegion* data = Regions.FindNodeOrInsertPos(ID, InsertPos);
  StringRegion* R = cast_or_null<StringRegion>(data);

  if (!R) {
    R = (StringRegion*) A.Allocate<StringRegion>();
    new (R) StringRegion(Str, GlobalsR);
    Regions.InsertNode(R, InsertPos);
  }

  return R;
}

VarRegion* MemRegionManager::getVarRegion(const VarDecl* d) {
  
  const MemRegion* superRegion = d->hasLocalStorage() ? getStackRegion() 
                                 : getGlobalsRegion();
  
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

CompoundLiteralRegion*
MemRegionManager::getCompoundLiteralRegion(const CompoundLiteralExpr* CL) {
  // Is this compound literal allocated on the stack or is part of the
  //  global constant pool?
  const MemRegion* superRegion = CL->isFileScope() ?
                                 getGlobalsRegion() : getStackRegion();

  // Profile the compound literal.
  llvm::FoldingSetNodeID ID;  
  CompoundLiteralRegion::ProfileRegion(ID, CL, superRegion);  
  
  void* InsertPos;
  MemRegion* data = Regions.FindNodeOrInsertPos(ID, InsertPos);
  CompoundLiteralRegion* R = cast_or_null<CompoundLiteralRegion>(data);
  
  if (!R) {
    R = (CompoundLiteralRegion*) A.Allocate<CompoundLiteralRegion>();
    new (R) CompoundLiteralRegion(CL, superRegion);
    Regions.InsertNode(R, InsertPos);
  }
  
  return R;
}

ElementRegion* MemRegionManager::getElementRegion(SVal Idx,
                                                  const MemRegion* superRegion){
  llvm::FoldingSetNodeID ID;
  ElementRegion::ProfileRegion(ID, Idx, superRegion);

  void* InsertPos;
  MemRegion* data = Regions.FindNodeOrInsertPos(ID, InsertPos);
  ElementRegion* R = cast_or_null<ElementRegion>(data);

  if (!R) {
    R = (ElementRegion*) A.Allocate<ElementRegion>();
    new (R) ElementRegion(Idx, superRegion);
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

ObjCObjectRegion*
MemRegionManager::getObjCObjectRegion(const ObjCInterfaceDecl* d,
                                    const MemRegion* superRegion) {
  llvm::FoldingSetNodeID ID;
  DeclRegion::ProfileRegion(ID, d, superRegion,
                            MemRegion::ObjCObjectRegionKind);
  
  void* InsertPos;
  MemRegion* data = Regions.FindNodeOrInsertPos(ID, InsertPos);
  ObjCObjectRegion* R = cast_or_null<ObjCObjectRegion>(data);
  
  if (!R) {
    R = (ObjCObjectRegion*) A.Allocate<ObjCObjectRegion>();
    new (R) ObjCObjectRegion(d, superRegion);
    Regions.InsertNode(R, InsertPos);
  }
  
  return R;
}


AnonPointeeRegion* MemRegionManager::getAnonPointeeRegion(const VarDecl* d) {
  llvm::FoldingSetNodeID ID;
  MemRegion* superRegion = getUnknownRegion();

  AnonPointeeRegion::ProfileRegion(ID, d, superRegion);

  void* InsertPos;
  MemRegion* data = Regions.FindNodeOrInsertPos(ID, InsertPos);
  AnonPointeeRegion* R = cast_or_null<AnonPointeeRegion>(data);

  if (!R) {
    R = (AnonPointeeRegion*) A.Allocate<AnonPointeeRegion>();
    new (R) AnonPointeeRegion(d, superRegion);
    Regions.InsertNode(R, InsertPos);
  }

  return R;
}

AllocaRegion* MemRegionManager::getAllocaRegion(const Expr* E, unsigned cnt) {
  llvm::FoldingSetNodeID ID;
  AllocaRegion::ProfileRegion(ID, E, cnt);
  
  void* InsertPos;
  MemRegion* data = Regions.FindNodeOrInsertPos(ID, InsertPos);
  AllocaRegion* R = cast_or_null<AllocaRegion>(data);
  
  if (!R) {
    R = (AllocaRegion*) A.Allocate<AllocaRegion>();
    new (R) AllocaRegion(E, cnt, getStackRegion());
    Regions.InsertNode(R, InsertPos);
  }
  
  return R;
}

bool MemRegionManager::hasStackStorage(const MemRegion* R) {

  // Only subregions can have stack storage.
  const SubRegion* SR = dyn_cast<SubRegion>(R);

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
