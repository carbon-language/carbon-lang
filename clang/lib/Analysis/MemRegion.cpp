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
#include "clang/Analysis/PathSensitive/ValueManager.h"

using namespace clang;

//===----------------------------------------------------------------------===//
// Basic methods.
//===----------------------------------------------------------------------===//

MemRegion::~MemRegion() {}

bool SubRegion::isSubRegionOf(const MemRegion* R) const {
  const MemRegion* r = getSuperRegion();
  while (r != 0) {
    if (r == R)
      return true;
    if (const SubRegion* sr = dyn_cast<SubRegion>(r))
      r = sr->getSuperRegion();
    else
      break;
  }
  return false;
}


MemRegionManager* SubRegion::getMemRegionManager() const {
  const SubRegion* r = this;
  do {
    const MemRegion *superRegion = r->getSuperRegion();
    if (const SubRegion *sr = dyn_cast<SubRegion>(superRegion)) {
      r = sr;
      continue;
    }
    return superRegion->getMemRegionManager();
  } while (1);
}

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
                                 const Expr* Ex, unsigned cnt,
                                 const MemRegion *) {
  ID.AddInteger((unsigned) AllocaRegionKind);
  ID.AddPointer(Ex);
  ID.AddInteger(cnt);
}

void AllocaRegion::Profile(llvm::FoldingSetNodeID& ID) const {
  ProfileRegion(ID, Ex, Cnt, superRegion);
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

void VarRegion::Profile(llvm::FoldingSetNodeID &ID) const {
  VarRegion::ProfileRegion(ID, getDecl(), LC, superRegion);
}

void SymbolicRegion::ProfileRegion(llvm::FoldingSetNodeID& ID, SymbolRef sym,
                                   const MemRegion *sreg) {
  ID.AddInteger((unsigned) MemRegion::SymbolicRegionKind);
  ID.Add(sym);
  ID.AddPointer(sreg);
}

void SymbolicRegion::Profile(llvm::FoldingSetNodeID& ID) const {
  SymbolicRegion::ProfileRegion(ID, sym, getSuperRegion());
}

void ElementRegion::ProfileRegion(llvm::FoldingSetNodeID& ID,
                                  QualType ElementType, SVal Idx, 
                                  const MemRegion* superRegion) {
  ID.AddInteger(MemRegion::ElementRegionKind);
  ID.Add(ElementType);
  ID.AddPointer(superRegion);
  Idx.Profile(ID);
}

void ElementRegion::Profile(llvm::FoldingSetNodeID& ID) const {
  ElementRegion::ProfileRegion(ID, ElementType, Index, superRegion);
}

void CodeTextRegion::ProfileRegion(llvm::FoldingSetNodeID& ID, const void* data,
                                   QualType t, const MemRegion*) {
  ID.AddInteger(MemRegion::CodeTextRegionKind);
  ID.AddPointer(data);
  ID.Add(t);
}

void CodeTextRegion::Profile(llvm::FoldingSetNodeID& ID) const {
  CodeTextRegion::ProfileRegion(ID, Data, LocationType, superRegion);
}

//===----------------------------------------------------------------------===//
// Region pretty-printing.
//===----------------------------------------------------------------------===//

void MemRegion::dump() const {
  dumpToStream(llvm::errs());
}

std::string MemRegion::getString() const {
  std::string s;
  llvm::raw_string_ostream os(s);
  dumpToStream(os);
  return os.str();
}

void MemRegion::dumpToStream(llvm::raw_ostream& os) const {
  os << "<Unknown Region>";
}

void AllocaRegion::dumpToStream(llvm::raw_ostream& os) const {
  os << "alloca{" << (void*) Ex << ',' << Cnt << '}';
}

void CodeTextRegion::dumpToStream(llvm::raw_ostream& os) const {
  os << "code{";
  if (isDeclared())
    os << getDecl()->getDeclName().getAsString();
  else
    os << '$' << getSymbol();

  os << '}';
}

void CompoundLiteralRegion::dumpToStream(llvm::raw_ostream& os) const {
  // FIXME: More elaborate pretty-printing.
  os << "{ " << (void*) CL <<  " }";
}

void ElementRegion::dumpToStream(llvm::raw_ostream& os) const {
  os << "element{" << superRegion << ','
     << Index << ',' << getElementType().getAsString() << '}';
}

void FieldRegion::dumpToStream(llvm::raw_ostream& os) const {
  os << superRegion << "->" << getDecl()->getNameAsString();
}

void ObjCIvarRegion::dumpToStream(llvm::raw_ostream& os) const {
  os << "ivar{" << superRegion << ',' << getDecl()->getNameAsString() << '}';
}

void StringRegion::dumpToStream(llvm::raw_ostream& os) const {  
  Str->printPretty(os, 0, PrintingPolicy(getContext().getLangOptions()));
}

void SymbolicRegion::dumpToStream(llvm::raw_ostream& os) const {
  os << "SymRegion{" << sym << '}';
}

void VarRegion::dumpToStream(llvm::raw_ostream& os) const {
  os << cast<VarDecl>(D)->getNameAsString();
}

void RegionRawOffset::dump() const {
  dumpToStream(llvm::errs());
}

void RegionRawOffset::dumpToStream(llvm::raw_ostream& os) const {
  os << "raw_offset{" << getRegion() << ',' << getByteOffset() << '}';
}

//===----------------------------------------------------------------------===//
// MemRegionManager methods.
//===----------------------------------------------------------------------===//

MemSpaceRegion* MemRegionManager::LazyAllocate(MemSpaceRegion*& region) {  
  if (!region) {  
    region = (MemSpaceRegion*) A.Allocate<MemSpaceRegion>();
    new (region) MemSpaceRegion(this);
  }

  return region;
}

MemSpaceRegion* MemRegionManager::getStackRegion() {
  return LazyAllocate(stack);
}

MemSpaceRegion* MemRegionManager::getStackArgumentsRegion() {
  return LazyAllocate(stackArguments);
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

MemSpaceRegion* MemRegionManager::getCodeRegion() {
  return LazyAllocate(code);
}

//===----------------------------------------------------------------------===//
// Constructing regions.
//===----------------------------------------------------------------------===//

StringRegion* MemRegionManager::getStringRegion(const StringLiteral* Str) {
  return getRegion<StringRegion>(Str);
}

VarRegion* MemRegionManager::getVarRegion(const VarDecl* D,
                                          const LocationContext *LC) {
  return getRegion<VarRegion>(D, LC);
}

CompoundLiteralRegion*
MemRegionManager::getCompoundLiteralRegion(const CompoundLiteralExpr* CL) {
  return getRegion<CompoundLiteralRegion>(CL);
}

ElementRegion*
MemRegionManager::getElementRegion(QualType elementType, SVal Idx,
                                   const MemRegion* superRegion,
                                   ASTContext& Ctx){

  QualType T = Ctx.getCanonicalType(elementType);

  llvm::FoldingSetNodeID ID;
  ElementRegion::ProfileRegion(ID, T, Idx, superRegion);

  void* InsertPos;
  MemRegion* data = Regions.FindNodeOrInsertPos(ID, InsertPos);
  ElementRegion* R = cast_or_null<ElementRegion>(data);

  if (!R) {
    R = (ElementRegion*) A.Allocate<ElementRegion>();
    new (R) ElementRegion(T, Idx, superRegion);
    Regions.InsertNode(R, InsertPos);
  }

  return R;
}

CodeTextRegion* MemRegionManager::getCodeTextRegion(const FunctionDecl* fd,
                                                    QualType t) {
  return getRegion<CodeTextRegion>(fd, t);
}

CodeTextRegion* MemRegionManager::getCodeTextRegion(SymbolRef sym, QualType t) {
  return getRegion<CodeTextRegion>(sym, t);
}

/// getSymbolicRegion - Retrieve or create a "symbolic" memory region.
SymbolicRegion* MemRegionManager::getSymbolicRegion(SymbolRef sym) {
  return getRegion<SymbolicRegion>(sym);
}

FieldRegion* MemRegionManager::getFieldRegion(const FieldDecl* d,
                                              const MemRegion* superRegion) {
  return getSubRegion<FieldRegion>(d, superRegion);
}

ObjCIvarRegion*
MemRegionManager::getObjCIvarRegion(const ObjCIvarDecl* d,
                                    const MemRegion* superRegion) {
  return getSubRegion<ObjCIvarRegion>(d, superRegion);
}

ObjCObjectRegion*
MemRegionManager::getObjCObjectRegion(const ObjCInterfaceDecl* d,
                                      const MemRegion* superRegion) {
  return getSubRegion<ObjCObjectRegion>(d, superRegion);
}

AllocaRegion* MemRegionManager::getAllocaRegion(const Expr* E, unsigned cnt) {
  return getRegion<AllocaRegion>(E, cnt);
}

const MemSpaceRegion *MemRegion::getMemorySpace() const {
  const MemRegion *R = this;
  const SubRegion* SR = dyn_cast<SubRegion>(this);
  
  while (SR) {
    R = SR->getSuperRegion();
    SR = dyn_cast<SubRegion>(R);
  }
  
  return dyn_cast<MemSpaceRegion>(R);
}

bool MemRegion::hasStackStorage() const {
  if (const MemSpaceRegion *MS = getMemorySpace()) {
    MemRegionManager *Mgr = getMemRegionManager();
    return MS == Mgr->getStackRegion() || MS == Mgr->getStackArgumentsRegion();
  }

  return false;
}

bool MemRegion::hasHeapStorage() const {
  if (const MemSpaceRegion *MS = getMemorySpace())
    return MS == getMemRegionManager()->getHeapRegion();

  return false;
}

bool MemRegion::hasHeapOrStackStorage() const {
  if (const MemSpaceRegion *MS = getMemorySpace()) {
    MemRegionManager *Mgr = getMemRegionManager();
    return MS == Mgr->getHeapRegion()
      || MS == Mgr->getStackRegion()
      || MS == Mgr->getStackArgumentsRegion();
  }
  return false;
}

bool MemRegion::hasGlobalsStorage() const {
  if (const MemSpaceRegion *MS = getMemorySpace())
    return MS == getMemRegionManager()->getGlobalsRegion();

  return false;
}

bool MemRegion::hasParametersStorage() const {
  if (const MemSpaceRegion *MS = getMemorySpace())
    return MS == getMemRegionManager()->getStackArgumentsRegion();
  
  return false;
}

bool MemRegion::hasGlobalsOrParametersStorage() const {
  if (const MemSpaceRegion *MS = getMemorySpace()) {
    MemRegionManager *Mgr = getMemRegionManager();
    return MS == Mgr->getGlobalsRegion()
    || MS == Mgr->getStackArgumentsRegion();
  }
  return false;
}

//===----------------------------------------------------------------------===//
// View handling.
//===----------------------------------------------------------------------===//

const MemRegion *MemRegion::getBaseRegion() const {
  const MemRegion *R = this;
  while (true) {
    if (const ElementRegion *ER = dyn_cast<ElementRegion>(R)) {      
      // FIXME: generalize.  Essentially we want to strip away ElementRegions
      // that were layered on a symbolic region because of casts.  We only
      // want to strip away ElementRegions, however, where the index is 0.
      SVal index = ER->getIndex();
      if (nonloc::ConcreteInt *CI = dyn_cast<nonloc::ConcreteInt>(&index)) {
        if (CI->getValue().getSExtValue() == 0) {
          R = ER->getSuperRegion();
          continue;
        }
      }
    }
    break;
  }
  return R;
}

// FIXME: Merge with the implementation of the same method in Store.cpp
static bool IsCompleteType(ASTContext &Ctx, QualType Ty) {
  if (const RecordType *RT = Ty->getAs<RecordType>()) {
    const RecordDecl *D = RT->getDecl();
    if (!D->getDefinition(Ctx))
      return false;
  }

  return true;
}

RegionRawOffset ElementRegion::getAsRawOffset() const {
  int64_t offset = 0;
  const ElementRegion *ER = this;
  const MemRegion *superR = NULL;
  ASTContext &C = getContext();
  
  // FIXME: Handle multi-dimensional arrays.

  while (ER) {
    superR = ER->getSuperRegion();
    
    // FIXME: generalize to symbolic offsets.
    SVal index = ER->getIndex();
    if (nonloc::ConcreteInt *CI = dyn_cast<nonloc::ConcreteInt>(&index)) {
      // Update the offset.
      int64_t i = CI->getValue().getSExtValue();
      
      if (i != 0) {
        QualType elemType = ER->getElementType();
        
        // If we are pointing to an incomplete type, go no further.
        if (!IsCompleteType(C, elemType)) {
          superR = ER;
          break;
        }
        
        int64_t size = (int64_t) (C.getTypeSize(elemType) / 8);
        offset += (i * size);
      }

      // Go to the next ElementRegion (if any).
      ER = dyn_cast<ElementRegion>(superR);
      continue;
    }
    
    return NULL;
  }
  
  assert(superR && "super region cannot be NULL");
  return RegionRawOffset(superR, offset);
}

