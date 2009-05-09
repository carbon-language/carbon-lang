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
//  created for use by GRExprEngine and related classes.
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/PathSensitive/SymbolManager.h"
#include "clang/Analysis/PathSensitive/MemRegion.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;

static void print(llvm::raw_ostream& os, const SymExpr *SE);

static void print(llvm::raw_ostream& os, BinaryOperator::Opcode Op) {  
  switch (Op) {
    default:
      assert(false && "operator printing not implemented");
      break;
    case BinaryOperator::Mul: os << '*'  ; break;
    case BinaryOperator::Div: os << '/'  ; break;
    case BinaryOperator::Rem: os << '%'  ; break;
    case BinaryOperator::Add: os << '+'  ; break;
    case BinaryOperator::Sub: os << '-'  ; break;
    case BinaryOperator::Shl: os << "<<" ; break;
    case BinaryOperator::Shr: os << ">>" ; break;
    case BinaryOperator::LT:  os << "<"  ; break;
    case BinaryOperator::GT:  os << '>'  ; break;
    case BinaryOperator::LE:  os << "<=" ; break;
    case BinaryOperator::GE:  os << ">=" ; break;    
    case BinaryOperator::EQ:  os << "==" ; break;
    case BinaryOperator::NE:  os << "!=" ; break;
    case BinaryOperator::And: os << '&'  ; break;
    case BinaryOperator::Xor: os << '^'  ; break;
    case BinaryOperator::Or:  os << '|'  ; break;
  }        
}

static void print(llvm::raw_ostream& os, const SymIntExpr *SE) {
  os << '(';
  print(os, SE->getLHS());
  os << ") ";
  print(os, SE->getOpcode());
  os << ' ' << SE->getRHS().getZExtValue();
  if (SE->getRHS().isUnsigned()) os << 'U';
}
  
static void print(llvm::raw_ostream& os, const SymSymExpr *SE) {
  os << '(';
  print(os, SE->getLHS());
  os << ") ";
  os << '(';
  print(os, SE->getRHS());
  os << ')';  
}

static void print(llvm::raw_ostream& os, const SymExpr *SE) {
  switch (SE->getKind()) {
    case SymExpr::BEGIN_SYMBOLS:
    case SymExpr::RegionValueKind:
    case SymExpr::ConjuredKind:
    case SymExpr::END_SYMBOLS:
      os << '$' << cast<SymbolData>(SE)->getSymbolID();
      return;
    case SymExpr::SymIntKind:
      print(os, cast<SymIntExpr>(SE));
      return;
    case SymExpr::SymSymKind:
      print(os, cast<SymSymExpr>(SE));
      return;
  }
}


llvm::raw_ostream& llvm::operator<<(llvm::raw_ostream& os, const SymExpr *SE) {
  print(os, SE);
  return os;
}

std::ostream& std::operator<<(std::ostream& os, const SymExpr *SE) {
  llvm::raw_os_ostream O(os);
  print(O, SE);
  return os;
}

const SymbolRegionValue* 
SymbolManager::getRegionValueSymbol(const MemRegion* R) {
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

QualType SymbolRegionValue::getType(ASTContext& C) const {
  if (const TypedRegion* TR = dyn_cast<TypedRegion>(R))
    return TR->getValueType(C);
  
  return QualType();
}

SymbolManager::~SymbolManager() {}

bool SymbolManager::canSymbolicate(QualType T) {
  return Loc::IsLocType(T) || T->isIntegerType();  
}

void SymbolReaper::markLive(SymbolRef sym) {
  TheLiving = F.Add(TheLiving, sym);
  TheDead = F.Remove(TheDead, sym);
}

bool SymbolReaper::maybeDead(SymbolRef sym) {
  if (isLive(sym))
    return false;
  
  TheDead = F.Add(TheDead, sym);
  return true;
}

bool SymbolReaper::isLive(SymbolRef sym) {
  if (TheLiving.contains(sym))
    return true;
  
  // Interogate the symbol.  It may derive from an input value to
  // the analyzed function/method.
  return isa<SymbolRegionValue>(sym);
}

SymbolVisitor::~SymbolVisitor() {}
