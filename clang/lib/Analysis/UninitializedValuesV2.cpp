//==- UninitializedValuesV2.cpp - Find Uninitialized Values -----*- C++ --*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements uninitialized values analysis for source-level CFGs.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "clang/AST/Decl.h"
#include "clang/Analysis/CFG.h"
#include "clang/Analysis/Visitors/CFGRecStmtDeclVisitor.h"
#include "clang/Analysis/Analyses/UninitializedValuesV2.h"

using namespace clang;

static bool isTrackedVar(const VarDecl *vd) {
  return vd->isLocalVarDecl() && !vd->hasGlobalStorage() && 
         vd->getType()->isScalarType();
}

//------------------------------------------------------------------------====//
// DeclToBit: a mapping from Decls we track to bitvector indices.
//====------------------------------------------------------------------------//

namespace {
class DeclToBit {
  llvm::DenseMap<const VarDecl *, unsigned> map;
public:
  DeclToBit() {}
  
  /// Compute the actual mapping from declarations to bits.
  void computeMap(const DeclContext &dc);
  
  /// Return the number of declarations in the map.
  unsigned size() const { return map.size(); }
  
  /// Returns the bit vector index for a given declaration.
  llvm::Optional<unsigned> getBitVectorIndex(const VarDecl *d);
};
}

void DeclToBit::computeMap(const DeclContext &dc) {
  unsigned count = 0;
  DeclContext::specific_decl_iterator<VarDecl> I(dc.decls_begin()),
                                               E(dc.decls_end());
  for ( ; I != E; ++I) {
    const VarDecl *vd = *I;
    if (isTrackedVar(vd))
      map[vd] = count++;
  }
}

llvm::Optional<unsigned> DeclToBit::getBitVectorIndex(const VarDecl *d) {
  llvm::DenseMap<const VarDecl *, unsigned>::iterator I = map.find(d);
  if (I == map.end())
    return llvm::Optional<unsigned>();
  return I->second;
}

//------------------------------------------------------------------------====//
// CFGBlockValues: dataflow values for CFG blocks.
//====------------------------------------------------------------------------//

namespace {
class CFGBlockValues {
  const CFG &cfg;
  llvm::BitVector **vals;
  llvm::BitVector scratch;
  DeclToBit declToBit;
public:
  CFGBlockValues(const CFG &cfg);
  ~CFGBlockValues();
  
  void computeSetOfDeclarations(const DeclContext &dc);  
  llvm::BitVector &getBitVector(const CFGBlock *block);
  void mergeIntoScratch(llvm::BitVector const &source, bool isFirst);
  bool updateBitVectorWithScratch(const CFGBlock *block);
  
  bool hasNoDeclarations() const {
    return declToBit.size() == 0;
  }
  
  void resetScratch();
  llvm::BitVector::reference operator[](const VarDecl *vd);
};  
}

CFGBlockValues::CFGBlockValues(const CFG &c) : cfg(c), vals(0) {
  unsigned n = cfg.getNumBlockIDs();
  if (!n)
    return;
  vals = new llvm::BitVector*[n];
  memset(vals, 0, sizeof(*vals) * n);
}

CFGBlockValues::~CFGBlockValues() {
  unsigned n = cfg.getNumBlockIDs();
  if (n == 0)
    return;
  for (unsigned i = 0; i < n; ++i)
    delete vals[i];
  delete [] vals;
}

void CFGBlockValues::computeSetOfDeclarations(const DeclContext &dc) {
  declToBit.computeMap(dc);
  scratch.resize(declToBit.size());
}

llvm::BitVector &CFGBlockValues::getBitVector(const CFGBlock *block) {
  unsigned idx = block->getBlockID();
  llvm::BitVector *bv = vals[idx];
  if (!bv) {
    bv = new llvm::BitVector(declToBit.size());
    vals[idx] = bv;
  }
  return *bv;
}

void CFGBlockValues::mergeIntoScratch(llvm::BitVector const &source,
                                      bool isFirst) {
  if (isFirst)
    scratch = source;
  else
    scratch |= source;  
}

bool CFGBlockValues::updateBitVectorWithScratch(const CFGBlock *block) {
  llvm::BitVector &dst = getBitVector(block);
  bool changed = (dst != scratch);
  if (changed)
    dst = scratch;
  return changed;
}

void CFGBlockValues::resetScratch() {
  scratch.reset();
}

llvm::BitVector::reference CFGBlockValues::operator[](const VarDecl *vd) {
  const llvm::Optional<unsigned> &idx = declToBit.getBitVectorIndex(vd);
  assert(idx.hasValue());
  return scratch[idx.getValue()];
}

//------------------------------------------------------------------------====//
// Worklist: worklist for dataflow analysis.
//====------------------------------------------------------------------------//

namespace {
class DataflowWorklist {
  llvm::SmallVector<const CFGBlock *, 20> worklist;
  llvm::BitVector enqueuedBlocks;
public:
  DataflowWorklist(const CFG &cfg) : enqueuedBlocks(cfg.getNumBlockIDs()) {}
  
  void enqueue(const CFGBlock *block);
  void enqueueSuccessors(const CFGBlock *block);
  const CFGBlock *dequeue();
  
};
}

void DataflowWorklist::enqueue(const CFGBlock *block) {
  if (!block)
    return;
  unsigned idx = block->getBlockID();
  if (enqueuedBlocks[idx])
    return;
  worklist.push_back(block);
  enqueuedBlocks[idx] = true;
}

void DataflowWorklist::enqueueSuccessors(const clang::CFGBlock *block) {
  for (CFGBlock::const_succ_iterator I = block->succ_begin(),
       E = block->succ_end(); I != E; ++I) {
    enqueue(*I);
  }
}

const CFGBlock *DataflowWorklist::dequeue() {
  if (worklist.empty())
    return 0;
  const CFGBlock *b = worklist.back();
  worklist.pop_back();
  enqueuedBlocks[b->getBlockID()] = false;
  return b;
}

//------------------------------------------------------------------------====//
// Transfer function for uninitialized values analysis.
//====------------------------------------------------------------------------//

static const bool Initialized = false;
static const bool Uninitialized = true;

namespace {
class FindVarResult {
  const VarDecl *vd;
  const DeclRefExpr *dr;
public:
  FindVarResult(VarDecl *vd, DeclRefExpr *dr) : vd(vd), dr(dr) {}
  
  const DeclRefExpr *getDeclRefExpr() const { return dr; }
  const VarDecl *getDecl() const { return vd; }
};
  
class TransferFunctions : public CFGRecStmtVisitor<TransferFunctions> {
  CFGBlockValues &vals;
  const CFG &cfg;
  UninitVariablesHandler *handler;
public:
  TransferFunctions(CFGBlockValues &vals, const CFG &cfg,
                    UninitVariablesHandler *handler)
    : vals(vals), cfg(cfg), handler(handler) {}
  
  const CFG &getCFG() { return cfg; }
  void reportUninit(const DeclRefExpr *ex, const VarDecl *vd);
  
  void VisitDeclStmt(DeclStmt *ds);
  void VisitUnaryOperator(UnaryOperator *uo);
  void VisitBinaryOperator(BinaryOperator *bo);
  void VisitCastExpr(CastExpr *ce);
};
}

void TransferFunctions::reportUninit(const DeclRefExpr *ex,
                                     const VarDecl *vd) {
  if (handler) handler->handleUseOfUninitVariable(ex, vd);
}

void TransferFunctions::VisitDeclStmt(DeclStmt *ds) {
  for (DeclStmt::decl_iterator DI = ds->decl_begin(), DE = ds->decl_end();
       DI != DE; ++DI) {
    if (VarDecl *vd = dyn_cast<VarDecl>(*DI)) {
      if (isTrackedVar(vd)) {
        vals[vd] = Uninitialized;
        if (Stmt *init = vd->getInit()) {
          Visit(init);
          vals[vd] = Initialized;
        }
      }
    }
  }
}

static FindVarResult findBlockVarDecl(Expr* ex) {
  if (DeclRefExpr* dr = dyn_cast<DeclRefExpr>(ex->IgnoreParenCasts()))
    if (VarDecl *vd = dyn_cast<VarDecl>(dr->getDecl()))
      if (isTrackedVar(vd))
        return FindVarResult(vd, dr);

  return FindVarResult(0, 0);
}

void TransferFunctions::VisitBinaryOperator(clang::BinaryOperator *bo) {
  Visit(bo->getRHS());
  Visit(bo->getLHS());
  if (bo->isAssignmentOp()) {
    const FindVarResult &res = findBlockVarDecl(bo->getLHS());
    if (const VarDecl* vd = res.getDecl()) {
      llvm::BitVector::reference bit = vals[vd];
      if (bit == Uninitialized) {
        if (bo->getOpcode() != BO_Assign)
          reportUninit(res.getDeclRefExpr(), vd);
        bit = Initialized;
      }
    }
  }
}

void TransferFunctions::VisitUnaryOperator(clang::UnaryOperator *uo) {
  Visit(uo->getSubExpr());
  switch (uo->getOpcode()) {
    case clang::UO_AddrOf:
      if (const VarDecl *vd = findBlockVarDecl(uo->getSubExpr()).getDecl())
        vals[vd] = Initialized;
      break;
    case clang::UO_PostDec:
    case clang::UO_PostInc:
    case clang::UO_PreDec:
    case clang::UO_PreInc: {
      const FindVarResult &res = findBlockVarDecl(uo->getSubExpr());
      if (const VarDecl *vd = res.getDecl()) {
        llvm::BitVector::reference bit = vals[vd];
        if (bit == Uninitialized) {
          reportUninit(res.getDeclRefExpr(), vd);
          bit = Initialized;
        }
      }
      break;
    }
    default:
      break;
  }
}

void TransferFunctions::VisitCastExpr(clang::CastExpr *ce) {
  Visit(ce->getSubExpr());
  if (ce->getCastKind() == CK_LValueToRValue) {
    const FindVarResult &res = findBlockVarDecl(ce->getSubExpr());
    if (const VarDecl *vd = res.getDecl())
      if (vals[vd] == Uninitialized)
        reportUninit(res.getDeclRefExpr(), vd);
  }
}

//------------------------------------------------------------------------====//
// High-level "driver" logic for uninitialized values analysis.
//====------------------------------------------------------------------------//

static void runOnBlock(const CFGBlock *block, const CFG &cfg,
                       CFGBlockValues &vals,
                       UninitVariablesHandler *handler = 0) {
  // Merge in values of predecessor blocks.    
  vals.resetScratch();
  bool isFirst = true;
  for (CFGBlock::const_pred_iterator I = block->pred_begin(),
       E = block->pred_end(); I != E; ++I) {
    vals.mergeIntoScratch(vals.getBitVector(*I), isFirst);
    isFirst = false;
  }
  // Apply the transfer function.
  TransferFunctions tf(vals, cfg, handler);
  for (CFGBlock::const_iterator I = block->begin(), E = block->end(); 
       I != E; ++I) {
    if (const CFGStmt *cs = dyn_cast<CFGStmt>(&*I)) {
      tf.BlockStmt_Visit(cs->getStmt());
    }
  }
}

void clang::runUninitializedVariablesAnalysis(const DeclContext &dc,
                                              const CFG &cfg,
                                              UninitVariablesHandler &handler) {
  CFGBlockValues vals(cfg);
  vals.computeSetOfDeclarations(dc);
  if (vals.hasNoDeclarations())
    return;
  DataflowWorklist worklist(cfg);
  llvm::BitVector previouslyVisited(cfg.getNumBlockIDs());
  
  worklist.enqueueSuccessors(&cfg.getEntry());

  while (const CFGBlock *block = worklist.dequeue()) {
    runOnBlock(block, cfg, vals);    
    // Did the block change?
    bool changed = vals.updateBitVectorWithScratch(block);    
    if (changed || !previouslyVisited[block->getBlockID()])
      worklist.enqueueSuccessors(block);    
    previouslyVisited[block->getBlockID()] = true;
  }
  
  // Run through the blocks one more time, and report uninitialized variabes.
  for (CFG::const_iterator BI = cfg.begin(), BE = cfg.end(); BI != BE; ++BI) {
    runOnBlock(*BI, cfg, vals, &handler);
  }
}

UninitVariablesHandler::~UninitVariablesHandler() {}

