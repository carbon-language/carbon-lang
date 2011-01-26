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

#include <utility>
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "clang/AST/Decl.h"
#include "clang/Analysis/CFG.h"
#include "clang/Analysis/AnalysisContext.h"
#include "clang/Analysis/Visitors/CFGRecStmtDeclVisitor.h"
#include "clang/Analysis/Analyses/UninitializedValuesV2.h"
#include "clang/Analysis/Support/SaveAndRestore.h"

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

typedef std::pair<llvm::BitVector *, llvm::BitVector *> BVPair;

namespace {
class CFGBlockValues {
  const CFG &cfg;
  BVPair *vals;
  llvm::BitVector scratch;
  DeclToBit declToBit;
  
  llvm::BitVector &lazyCreate(llvm::BitVector *&bv);
public:
  CFGBlockValues(const CFG &cfg);
  ~CFGBlockValues();
  
  void computeSetOfDeclarations(const DeclContext &dc);  
  llvm::BitVector &getBitVector(const CFGBlock *block,
                                const CFGBlock *dstBlock);

  BVPair &getBitVectors(const CFGBlock *block);

  BVPair getPredBitVectors(const CFGBlock *block);
  
  void mergeIntoScratch(llvm::BitVector const &source, bool isFirst);
  bool updateBitVectorWithScratch(const CFGBlock *block);
  bool updateBitVectors(const CFGBlock *block, const BVPair &newVals);
  
  bool hasNoDeclarations() const {
    return declToBit.size() == 0;
  }
  
  void resetScratch();
  llvm::BitVector &getScratch() { return scratch; }
  
  llvm::BitVector::reference operator[](const VarDecl *vd);
};  
}

CFGBlockValues::CFGBlockValues(const CFG &c) : cfg(c), vals(0) {
  unsigned n = cfg.getNumBlockIDs();
  if (!n)
    return;
  vals = new std::pair<llvm::BitVector*, llvm::BitVector*>[n];
  memset(vals, 0, sizeof(*vals) * n);
}

CFGBlockValues::~CFGBlockValues() {
  unsigned n = cfg.getNumBlockIDs();
  if (n == 0)
    return;
  for (unsigned i = 0; i < n; ++i) {
    delete vals[i].first;
    delete vals[i].second;
  }
  delete [] vals;
}

void CFGBlockValues::computeSetOfDeclarations(const DeclContext &dc) {
  declToBit.computeMap(dc);
  scratch.resize(declToBit.size());
}

llvm::BitVector &CFGBlockValues::lazyCreate(llvm::BitVector *&bv) {
  if (!bv)
    bv = new llvm::BitVector(declToBit.size());
  return *bv;
}

/// This function pattern matches for a '&&' or '||' that appears at
/// the beginning of a CFGBlock that also (1) has a terminator and 
/// (2) has no other elements.  If such an expression is found, it is returned.
static BinaryOperator *getLogicalOperatorInChain(const CFGBlock *block) {
  if (block->empty())
    return 0;
  
  CFGStmt cstmt = block->front().getAs<CFGStmt>();
  BinaryOperator *b = llvm::dyn_cast_or_null<BinaryOperator>(cstmt.getStmt());
  if (!b || !b->isLogicalOp() || block->getTerminatorCondition() != b)
    return 0;
  return b;
}

llvm::BitVector &CFGBlockValues::getBitVector(const CFGBlock *block,
                                              const CFGBlock *dstBlock) {
  unsigned idx = block->getBlockID();
  if (dstBlock && block->succ_size() == 2 && block->pred_size() == 2) {
    assert(block->getTerminator());
    if (getLogicalOperatorInChain(block)) {
      if (*block->succ_begin() == dstBlock)
        return lazyCreate(vals[idx].first);
      assert(*(block->succ_begin()+1) == dstBlock);
      return lazyCreate(vals[idx].second);
    }
  }

  assert(vals[idx].second == 0);
  return lazyCreate(vals[idx].first);
}

BVPair &CFGBlockValues::getBitVectors(const clang::CFGBlock *block) {
  unsigned idx = block->getBlockID();
  lazyCreate(vals[idx].first);
  lazyCreate(vals[idx].second);
  return vals[idx];
}

BVPair CFGBlockValues::getPredBitVectors(const clang::CFGBlock *block) {
  assert(block->pred_size() == 2);
  CFGBlock::const_pred_iterator itr = block->pred_begin();
  llvm::BitVector &bvA = getBitVector(*itr, block);
  ++itr;
  return BVPair(&bvA, &getBitVector(*itr, block));
}

void CFGBlockValues::mergeIntoScratch(llvm::BitVector const &source,
                                      bool isFirst) {
  if (isFirst)
    scratch = source;
  else
    scratch |= source;  
}

bool CFGBlockValues::updateBitVectorWithScratch(const CFGBlock *block) {
  llvm::BitVector &dst = getBitVector(block, 0);
  bool changed = (dst != scratch);
  if (changed)
    dst = scratch;
  
  return changed;
}

bool CFGBlockValues::updateBitVectors(const CFGBlock *block,
                                      const BVPair &newVals) {
  BVPair &vals = getBitVectors(block);
  bool changed = *newVals.first != *vals.first ||
                 *newVals.second != *vals.second;
  *vals.first = *newVals.first;
  *vals.second = *newVals.second;
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
  AnalysisContext &ac;
  UninitVariablesHandler *handler;
  const DeclRefExpr *currentDR;
  const Expr *currentVoidCast;
  const bool flagBlockUses;
public:
  TransferFunctions(CFGBlockValues &vals, const CFG &cfg,
                    AnalysisContext &ac,
                    UninitVariablesHandler *handler,
                    bool flagBlockUses)
    : vals(vals), cfg(cfg), ac(ac), handler(handler), currentDR(0),
      currentVoidCast(0), flagBlockUses(flagBlockUses) {}
  
  const CFG &getCFG() { return cfg; }
  void reportUninit(const DeclRefExpr *ex, const VarDecl *vd);

  void VisitBlockExpr(BlockExpr *be);
  void VisitDeclStmt(DeclStmt *ds);
  void VisitDeclRefExpr(DeclRefExpr *dr);
  void VisitUnaryOperator(UnaryOperator *uo);
  void VisitBinaryOperator(BinaryOperator *bo);
  void VisitCastExpr(CastExpr *ce);
  void VisitSizeOfAlignOfExpr(SizeOfAlignOfExpr *se);
};
}

void TransferFunctions::reportUninit(const DeclRefExpr *ex,
                                     const VarDecl *vd) {
  if (handler) handler->handleUseOfUninitVariable(ex, vd);
}

void TransferFunctions::VisitBlockExpr(BlockExpr *be) {
  if (!flagBlockUses || !handler)
    return;
  AnalysisContext::referenced_decls_iterator i, e;
  llvm::tie(i, e) = ac.getReferencedBlockVars(be->getBlockDecl());
  for ( ; i != e; ++i) {
    const VarDecl *vd = *i;
    if (vd->getAttr<BlocksAttr>() || !vd->hasLocalStorage())
      continue;
    if (vals[vd] == Uninitialized)
      handler->handleUseOfUninitVariable(be, vd);      
  }
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
      else if (Stmt *init = vd->getInit()) {
        Visit(init);
      }
    }
  }
}

void TransferFunctions::VisitDeclRefExpr(DeclRefExpr *dr) {
  // We assume that DeclRefExprs wrapped in an lvalue-to-rvalue cast
  // cannot be block-level expressions.  Therefore, we determine if
  // a DeclRefExpr is involved in a "load" by comparing it to the current
  // DeclRefExpr found when analyzing the last lvalue-to-rvalue CastExpr.
  // If a DeclRefExpr is not involved in a load, we are essentially computing
  // its address, either for assignment to a reference or via the '&' operator.
  // In such cases, treat the variable as being initialized, since this
  // analysis isn't powerful enough to do alias tracking.
  if (dr != currentDR)
    if (const VarDecl *vd = dyn_cast<VarDecl>(dr->getDecl()))
      if (isTrackedVar(vd))
        vals[vd] = Initialized;
}

static FindVarResult findBlockVarDecl(Expr* ex) {
  if (DeclRefExpr* dr = dyn_cast<DeclRefExpr>(ex->IgnoreParenCasts()))
    if (VarDecl *vd = dyn_cast<VarDecl>(dr->getDecl()))
      if (isTrackedVar(vd))
        return FindVarResult(vd, dr);

  return FindVarResult(0, 0);
}

void TransferFunctions::VisitBinaryOperator(clang::BinaryOperator *bo) {
  if (bo->isAssignmentOp()) {
    const FindVarResult &res = findBlockVarDecl(bo->getLHS());
    if (const VarDecl* vd = res.getDecl()) {
      // We assume that DeclRefExprs wrapped in a BinaryOperator "assignment"
      // cannot be block-level expressions.  Therefore, we determine if
      // a DeclRefExpr is involved in a "load" by comparing it to the current
      // DeclRefExpr found when analyzing the last lvalue-to-rvalue CastExpr.
      SaveAndRestore<const DeclRefExpr*> lastDR(currentDR, 
                                                res.getDeclRefExpr());
      Visit(bo->getRHS());
      Visit(bo->getLHS());

      llvm::BitVector::reference bit = vals[vd];
      if (bit == Uninitialized) {
        if (bo->getOpcode() != BO_Assign)
          reportUninit(res.getDeclRefExpr(), vd);
        bit = Initialized;
      }
      return;
    }
  }
  Visit(bo->getRHS());
  Visit(bo->getLHS());
}

void TransferFunctions::VisitUnaryOperator(clang::UnaryOperator *uo) {
  switch (uo->getOpcode()) {
    case clang::UO_PostDec:
    case clang::UO_PostInc:
    case clang::UO_PreDec:
    case clang::UO_PreInc: {
      const FindVarResult &res = findBlockVarDecl(uo->getSubExpr());
      if (const VarDecl *vd = res.getDecl()) {
        // We assume that DeclRefExprs wrapped in a unary operator ++/--
        // cannot be block-level expressions.  Therefore, we determine if
        // a DeclRefExpr is involved in a "load" by comparing it to the current
        // DeclRefExpr found when analyzing the last lvalue-to-rvalue CastExpr.
        SaveAndRestore<const DeclRefExpr*> lastDR(currentDR, 
                                                  res.getDeclRefExpr());
        Visit(uo->getSubExpr());

        llvm::BitVector::reference bit = vals[vd];
        if (bit == Uninitialized) {
          reportUninit(res.getDeclRefExpr(), vd);
          bit = Initialized;
        }
        return;
      }
      break;
    }
    default:
      break;
  }
  Visit(uo->getSubExpr());
}

void TransferFunctions::VisitCastExpr(clang::CastExpr *ce) {
  if (ce->getCastKind() == CK_LValueToRValue) {
    const FindVarResult &res = findBlockVarDecl(ce->getSubExpr());
    if (const VarDecl *vd = res.getDecl()) {
      // We assume that DeclRefExprs wrapped in an lvalue-to-rvalue cast
      // cannot be block-level expressions.  Therefore, we determine if
      // a DeclRefExpr is involved in a "load" by comparing it to the current
      // DeclRefExpr found when analyzing the last lvalue-to-rvalue CastExpr.
      // Here we update 'currentDR' to be the one associated with this
      // lvalue-to-rvalue cast.  Then, when we analyze the DeclRefExpr, we
      // will know that we are not computing its lvalue for other purposes
      // than to perform a load.
      SaveAndRestore<const DeclRefExpr*> lastDR(currentDR, 
                                                res.getDeclRefExpr());
      Visit(ce->getSubExpr());
      if (currentVoidCast != ce && vals[vd] == Uninitialized) {
        reportUninit(res.getDeclRefExpr(), vd);
        // Don't cascade warnings.
        vals[vd] = Initialized;
      }
      return;
    }
  }
  else if (CStyleCastExpr *cse = dyn_cast<CStyleCastExpr>(ce)) {
    if (cse->getType()->isVoidType()) {
      // e.g. (void) x;
      SaveAndRestore<const Expr *>
        lastVoidCast(currentVoidCast, cse->getSubExpr()->IgnoreParens());
      Visit(cse->getSubExpr());
      return;
    }
  }
  Visit(ce->getSubExpr());
}

void TransferFunctions::VisitSizeOfAlignOfExpr(SizeOfAlignOfExpr *se) {
  if (se->isSizeOf()) {
    if (se->getType()->isConstantSizeType())
      return;
    // Handle VLAs.
    Visit(se->getArgumentExpr());
  }
}

//------------------------------------------------------------------------====//
// High-level "driver" logic for uninitialized values analysis.
//====------------------------------------------------------------------------//

static bool runOnBlock(const CFGBlock *block, const CFG &cfg,
                       AnalysisContext &ac, CFGBlockValues &vals,
                       UninitVariablesHandler *handler = 0,
                       bool flagBlockUses = false) {
  
  if (const BinaryOperator *b = getLogicalOperatorInChain(block)) {
    if (block->pred_size() == 2 && block->succ_size() == 2) {
      assert(block->getTerminatorCondition() == b);    
      BVPair valsAB = vals.getPredBitVectors(block);
      vals.mergeIntoScratch(*valsAB.first, true);
      vals.mergeIntoScratch(*valsAB.second, false);
      valsAB.second = &vals.getScratch();
      if (b->getOpcode() == BO_LOr) {
        // Ensure the invariant that 'first' corresponds to the true
        // branch and 'second' to the false.
        std::swap(valsAB.first, valsAB.second);
      }
      return vals.updateBitVectors(block, valsAB);
    }
  }

  // Default behavior: merge in values of predecessor blocks.    
  vals.resetScratch();
  bool isFirst = true;
  for (CFGBlock::const_pred_iterator I = block->pred_begin(),
       E = block->pred_end(); I != E; ++I) {
    vals.mergeIntoScratch(vals.getBitVector(*I, block), isFirst);
    isFirst = false;
  }
  // Apply the transfer function.
  TransferFunctions tf(vals, cfg, ac, handler, flagBlockUses);
  for (CFGBlock::const_iterator I = block->begin(), E = block->end(); 
       I != E; ++I) {
    if (const CFGStmt *cs = dyn_cast<CFGStmt>(&*I)) {
      tf.BlockStmt_Visit(cs->getStmt());
    }
  }
  return vals.updateBitVectorWithScratch(block);
}

void clang::runUninitializedVariablesAnalysis(const DeclContext &dc,
                                              const CFG &cfg,
                                              AnalysisContext &ac,
                                              UninitVariablesHandler &handler) {
  CFGBlockValues vals(cfg);
  vals.computeSetOfDeclarations(dc);
  if (vals.hasNoDeclarations())
    return;
  DataflowWorklist worklist(cfg);
  llvm::BitVector previouslyVisited(cfg.getNumBlockIDs());
  
  worklist.enqueueSuccessors(&cfg.getEntry());

  while (const CFGBlock *block = worklist.dequeue()) {
    // Did the block change?
    bool changed = runOnBlock(block, cfg, ac, vals);    
    if (changed || !previouslyVisited[block->getBlockID()])
      worklist.enqueueSuccessors(block);    
    previouslyVisited[block->getBlockID()] = true;
  }
  
  // Run through the blocks one more time, and report uninitialized variabes.
  for (CFG::const_iterator BI = cfg.begin(), BE = cfg.end(); BI != BE; ++BI) {
    runOnBlock(*BI, cfg, ac, vals, &handler, /* flagBlockUses */ true);
  }
}

UninitVariablesHandler::~UninitVariablesHandler() {}

