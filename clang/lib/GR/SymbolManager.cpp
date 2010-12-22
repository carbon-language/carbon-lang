//== SymbolManager.h - Management of Symbolic Values ------------*- C++ -*--==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines SymbolManager, a class that manages symbolic values
//  created for use by ExprEngine and related classes.
//
//===----------------------------------------------------------------------===//

#include "clang/GR/PathSensitive/SymbolManager.h"
#include "clang/Analysis/Analyses/LiveVariables.h"
#include "clang/GR/PathSensitive/MemRegion.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;
using namespace GR;

void SymExpr::dump() const {
  dumpToStream(llvm::errs());
}

static void print(llvm::raw_ostream& os, BinaryOperator::Opcode Op) {
  switch (Op) {
    default:
      assert(false && "operator printing not implemented");
      break;
    case BO_Mul: os << '*'  ; break;
    case BO_Div: os << '/'  ; break;
    case BO_Rem: os << '%'  ; break;
    case BO_Add: os << '+'  ; break;
    case BO_Sub: os << '-'  ; break;
    case BO_Shl: os << "<<" ; break;
    case BO_Shr: os << ">>" ; break;
    case BO_LT:  os << "<"  ; break;
    case BO_GT:  os << '>'  ; break;
    case BO_LE:  os << "<=" ; break;
    case BO_GE:  os << ">=" ; break;
    case BO_EQ:  os << "==" ; break;
    case BO_NE:  os << "!=" ; break;
    case BO_And: os << '&'  ; break;
    case BO_Xor: os << '^'  ; break;
    case BO_Or:  os << '|'  ; break;
  }
}

void SymIntExpr::dumpToStream(llvm::raw_ostream& os) const {
  os << '(';
  getLHS()->dumpToStream(os);
  os << ") ";
  print(os, getOpcode());
  os << ' ' << getRHS().getZExtValue();
  if (getRHS().isUnsigned()) os << 'U';
}

void SymSymExpr::dumpToStream(llvm::raw_ostream& os) const {
  os << '(';
  getLHS()->dumpToStream(os);
  os << ") ";
  os << '(';
  getRHS()->dumpToStream(os);
  os << ')';
}

void SymbolConjured::dumpToStream(llvm::raw_ostream& os) const {
  os << "conj_$" << getSymbolID() << '{' << T.getAsString() << '}';
}

void SymbolDerived::dumpToStream(llvm::raw_ostream& os) const {
  os << "derived_$" << getSymbolID() << '{'
     << getParentSymbol() << ',' << getRegion() << '}';
}

void SymbolExtent::dumpToStream(llvm::raw_ostream& os) const {
  os << "extent_$" << getSymbolID() << '{' << getRegion() << '}';
}

void SymbolMetadata::dumpToStream(llvm::raw_ostream& os) const {
  os << "meta_$" << getSymbolID() << '{'
     << getRegion() << ',' << T.getAsString() << '}';
}

void SymbolRegionValue::dumpToStream(llvm::raw_ostream& os) const {
  os << "reg_$" << getSymbolID() << "<" << R << ">";
}

const SymbolRegionValue*
SymbolManager::getRegionValueSymbol(const TypedRegion* R) {
  llvm::FoldingSetNodeID profile;
  SymbolRegionValue::Profile(profile, R);
  void* InsertPos;
  SymExpr *SD = DataSet.FindNodeOrInsertPos(profile, InsertPos);
  if (!SD) {
    SD = (SymExpr*) BPAlloc.Allocate<SymbolRegionValue>();
    new (SD) SymbolRegionValue(SymbolCounter, R);
    DataSet.InsertNode(SD, InsertPos);
    ++SymbolCounter;
  }

  return cast<SymbolRegionValue>(SD);
}

const SymbolConjured*
SymbolManager::getConjuredSymbol(const Stmt* E, QualType T, unsigned Count,
                                 const void* SymbolTag) {

  llvm::FoldingSetNodeID profile;
  SymbolConjured::Profile(profile, E, T, Count, SymbolTag);
  void* InsertPos;
  SymExpr *SD = DataSet.FindNodeOrInsertPos(profile, InsertPos);
  if (!SD) {
    SD = (SymExpr*) BPAlloc.Allocate<SymbolConjured>();
    new (SD) SymbolConjured(SymbolCounter, E, T, Count, SymbolTag);
    DataSet.InsertNode(SD, InsertPos);
    ++SymbolCounter;
  }

  return cast<SymbolConjured>(SD);
}

const SymbolDerived*
SymbolManager::getDerivedSymbol(SymbolRef parentSymbol,
                                const TypedRegion *R) {

  llvm::FoldingSetNodeID profile;
  SymbolDerived::Profile(profile, parentSymbol, R);
  void* InsertPos;
  SymExpr *SD = DataSet.FindNodeOrInsertPos(profile, InsertPos);
  if (!SD) {
    SD = (SymExpr*) BPAlloc.Allocate<SymbolDerived>();
    new (SD) SymbolDerived(SymbolCounter, parentSymbol, R);
    DataSet.InsertNode(SD, InsertPos);
    ++SymbolCounter;
  }

  return cast<SymbolDerived>(SD);
}

const SymbolExtent*
SymbolManager::getExtentSymbol(const SubRegion *R) {
  llvm::FoldingSetNodeID profile;
  SymbolExtent::Profile(profile, R);
  void* InsertPos;
  SymExpr *SD = DataSet.FindNodeOrInsertPos(profile, InsertPos);
  if (!SD) {
    SD = (SymExpr*) BPAlloc.Allocate<SymbolExtent>();
    new (SD) SymbolExtent(SymbolCounter, R);
    DataSet.InsertNode(SD, InsertPos);
    ++SymbolCounter;
  }

  return cast<SymbolExtent>(SD);
}

const SymbolMetadata*
SymbolManager::getMetadataSymbol(const MemRegion* R, const Stmt* S, QualType T,
                                 unsigned Count, const void* SymbolTag) {

  llvm::FoldingSetNodeID profile;
  SymbolMetadata::Profile(profile, R, S, T, Count, SymbolTag);
  void* InsertPos;
  SymExpr *SD = DataSet.FindNodeOrInsertPos(profile, InsertPos);
  if (!SD) {
    SD = (SymExpr*) BPAlloc.Allocate<SymbolMetadata>();
    new (SD) SymbolMetadata(SymbolCounter, R, S, T, Count, SymbolTag);
    DataSet.InsertNode(SD, InsertPos);
    ++SymbolCounter;
  }

  return cast<SymbolMetadata>(SD);
}

const SymIntExpr *SymbolManager::getSymIntExpr(const SymExpr *lhs,
                                               BinaryOperator::Opcode op,
                                               const llvm::APSInt& v,
                                               QualType t) {
  llvm::FoldingSetNodeID ID;
  SymIntExpr::Profile(ID, lhs, op, v, t);
  void *InsertPos;
  SymExpr *data = DataSet.FindNodeOrInsertPos(ID, InsertPos);

  if (!data) {
    data = (SymIntExpr*) BPAlloc.Allocate<SymIntExpr>();
    new (data) SymIntExpr(lhs, op, v, t);
    DataSet.InsertNode(data, InsertPos);
  }

  return cast<SymIntExpr>(data);
}

const SymSymExpr *SymbolManager::getSymSymExpr(const SymExpr *lhs,
                                               BinaryOperator::Opcode op,
                                               const SymExpr *rhs,
                                               QualType t) {
  llvm::FoldingSetNodeID ID;
  SymSymExpr::Profile(ID, lhs, op, rhs, t);
  void *InsertPos;
  SymExpr *data = DataSet.FindNodeOrInsertPos(ID, InsertPos);

  if (!data) {
    data = (SymSymExpr*) BPAlloc.Allocate<SymSymExpr>();
    new (data) SymSymExpr(lhs, op, rhs, t);
    DataSet.InsertNode(data, InsertPos);
  }

  return cast<SymSymExpr>(data);
}

QualType SymbolConjured::getType(ASTContext&) const {
  return T;
}

QualType SymbolDerived::getType(ASTContext& Ctx) const {
  return R->getValueType();
}

QualType SymbolExtent::getType(ASTContext& Ctx) const {
  return Ctx.getSizeType();
}

QualType SymbolMetadata::getType(ASTContext&) const {
  return T;
}

QualType SymbolRegionValue::getType(ASTContext& C) const {
  return R->getValueType();
}

SymbolManager::~SymbolManager() {}

bool SymbolManager::canSymbolicate(QualType T) {
  if (Loc::IsLocType(T))
    return true;

  if (T->isIntegerType())
    return T->isScalarType();

  if (T->isRecordType())
    return true;

  return false;
}

void SymbolReaper::markLive(SymbolRef sym) {
  TheLiving.insert(sym);
  TheDead.erase(sym);
}

void SymbolReaper::markInUse(SymbolRef sym) {
  if (isa<SymbolMetadata>(sym))
    MetadataInUse.insert(sym);
}

bool SymbolReaper::maybeDead(SymbolRef sym) {
  if (isLive(sym))
    return false;

  TheDead.insert(sym);
  return true;
}

static bool IsLiveRegion(SymbolReaper &Reaper, const MemRegion *MR) {
  MR = MR->getBaseRegion();

  if (const SymbolicRegion *SR = dyn_cast<SymbolicRegion>(MR))
    return Reaper.isLive(SR->getSymbol());

  if (const VarRegion *VR = dyn_cast<VarRegion>(MR))
    return Reaper.isLive(VR);

  // FIXME: This is a gross over-approximation. What we really need is a way to
  // tell if anything still refers to this region. Unlike SymbolicRegions,
  // AllocaRegions don't have associated symbols, though, so we don't actually
  // have a way to track their liveness.
  if (isa<AllocaRegion>(MR))
    return true;

  if (isa<CXXThisRegion>(MR))
    return true;

  if (isa<MemSpaceRegion>(MR))
    return true;

  return false;
}

bool SymbolReaper::isLive(SymbolRef sym) {
  if (TheLiving.count(sym))
    return true;

  if (const SymbolDerived *derived = dyn_cast<SymbolDerived>(sym)) {
    if (isLive(derived->getParentSymbol())) {
      markLive(sym);
      return true;
    }
    return false;
  }

  if (const SymbolExtent *extent = dyn_cast<SymbolExtent>(sym)) {
    if (IsLiveRegion(*this, extent->getRegion())) {
      markLive(sym);
      return true;
    }
    return false;
  }

  if (const SymbolMetadata *metadata = dyn_cast<SymbolMetadata>(sym)) {
    if (MetadataInUse.count(sym)) {
      if (IsLiveRegion(*this, metadata->getRegion())) {
        markLive(sym);
        MetadataInUse.erase(sym);
        return true;
      }
    }
    return false;
  }

  // Interogate the symbol.  It may derive from an input value to
  // the analyzed function/method.
  return isa<SymbolRegionValue>(sym);
}

bool SymbolReaper::isLive(const Stmt* ExprVal) const {
  return LCtx->getAnalysisContext()->getRelaxedLiveVariables()->
      isLive(Loc, ExprVal);
}

bool SymbolReaper::isLive(const VarRegion *VR) const {
  const StackFrameContext *VarContext = VR->getStackFrame();
  const StackFrameContext *CurrentContext = LCtx->getCurrentStackFrame();

  if (VarContext == CurrentContext)
    return LCtx->getAnalysisContext()->getRelaxedLiveVariables()->
        isLive(Loc, VR->getDecl());

  return VarContext->isParentOf(CurrentContext);
}

SymbolVisitor::~SymbolVisitor() {}
