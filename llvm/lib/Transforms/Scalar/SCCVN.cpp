//===- SCCVN.cpp - Eliminate redundant values -----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass performs global value numbering to eliminate fully redundant
// instructions.  This is based on the paper "SCC-based Value Numbering"
// by Cooper, et al.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "sccvn"
#include "llvm/Transforms/Scalar.h"
#include "llvm/BasicBlock.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/Operator.h"
#include "llvm/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SparseBitVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Transforms/Utils/SSAUpdater.h"
using namespace llvm;

STATISTIC(NumSCCVNInstr,  "Number of instructions deleted by SCCVN");
STATISTIC(NumSCCVNPhi,  "Number of phis deleted by SCCVN");

//===----------------------------------------------------------------------===//
//                         ValueTable Class
//===----------------------------------------------------------------------===//

/// This class holds the mapping between values and value numbers.  It is used
/// as an efficient mechanism to determine the expression-wise equivalence of
/// two values.
namespace {
  struct Expression {
    enum ExpressionOpcode { ADD, FADD, SUB, FSUB, MUL, FMUL,
                            UDIV, SDIV, FDIV, UREM, SREM,
                            FREM, SHL, LSHR, ASHR, AND, OR, XOR, ICMPEQ,
                            ICMPNE, ICMPUGT, ICMPUGE, ICMPULT, ICMPULE,
                            ICMPSGT, ICMPSGE, ICMPSLT, ICMPSLE, FCMPOEQ,
                            FCMPOGT, FCMPOGE, FCMPOLT, FCMPOLE, FCMPONE,
                            FCMPORD, FCMPUNO, FCMPUEQ, FCMPUGT, FCMPUGE,
                            FCMPULT, FCMPULE, FCMPUNE, EXTRACT, INSERT,
                            SHUFFLE, SELECT, TRUNC, ZEXT, SEXT, FPTOUI,
                            FPTOSI, UITOFP, SITOFP, FPTRUNC, FPEXT,
                            PTRTOINT, INTTOPTR, BITCAST, GEP, CALL, CONSTANT,
                            INSERTVALUE, EXTRACTVALUE, EMPTY, TOMBSTONE };

    ExpressionOpcode opcode;
    const Type* type;
    SmallVector<uint32_t, 4> varargs;

    Expression() { }
    Expression(ExpressionOpcode o) : opcode(o) { }

    bool operator==(const Expression &other) const {
      if (opcode != other.opcode)
        return false;
      else if (opcode == EMPTY || opcode == TOMBSTONE)
        return true;
      else if (type != other.type)
        return false;
      else {
        if (varargs.size() != other.varargs.size())
          return false;

        for (size_t i = 0; i < varargs.size(); ++i)
          if (varargs[i] != other.varargs[i])
            return false;

        return true;
      }
    }

    bool operator!=(const Expression &other) const {
      return !(*this == other);
    }
  };

  class ValueTable {
    private:
      DenseMap<Value*, uint32_t> valueNumbering;
      DenseMap<Expression, uint32_t> expressionNumbering;
      DenseMap<Value*, uint32_t> constantsNumbering;

      uint32_t nextValueNumber;

      Expression::ExpressionOpcode getOpcode(BinaryOperator* BO);
      Expression::ExpressionOpcode getOpcode(CmpInst* C);
      Expression::ExpressionOpcode getOpcode(CastInst* C);
      Expression create_expression(BinaryOperator* BO);
      Expression create_expression(CmpInst* C);
      Expression create_expression(ShuffleVectorInst* V);
      Expression create_expression(ExtractElementInst* C);
      Expression create_expression(InsertElementInst* V);
      Expression create_expression(SelectInst* V);
      Expression create_expression(CastInst* C);
      Expression create_expression(GetElementPtrInst* G);
      Expression create_expression(CallInst* C);
      Expression create_expression(Constant* C);
      Expression create_expression(ExtractValueInst* C);
      Expression create_expression(InsertValueInst* C);
    public:
      ValueTable() : nextValueNumber(1) { }
      uint32_t computeNumber(Value *V);
      uint32_t lookup(Value *V);
      void add(Value *V, uint32_t num);
      void clear();
      void clearExpressions();
      void erase(Value *v);
      unsigned size();
      void verifyRemoved(const Value *) const;
  };
}

namespace llvm {
template <> struct DenseMapInfo<Expression> {
  static inline Expression getEmptyKey() {
    return Expression(Expression::EMPTY);
  }

  static inline Expression getTombstoneKey() {
    return Expression(Expression::TOMBSTONE);
  }

  static unsigned getHashValue(const Expression e) {
    unsigned hash = e.opcode;

    hash = ((unsigned)((uintptr_t)e.type >> 4) ^
            (unsigned)((uintptr_t)e.type >> 9));

    for (SmallVector<uint32_t, 4>::const_iterator I = e.varargs.begin(),
         E = e.varargs.end(); I != E; ++I)
      hash = *I + hash * 37;

    return hash;
  }
  static bool isEqual(const Expression &LHS, const Expression &RHS) {
    return LHS == RHS;
  }
};
template <>
struct isPodLike<Expression> { static const bool value = true; };

}

//===----------------------------------------------------------------------===//
//                     ValueTable Internal Functions
//===----------------------------------------------------------------------===//
Expression::ExpressionOpcode ValueTable::getOpcode(BinaryOperator* BO) {
  switch(BO->getOpcode()) {
  default: // THIS SHOULD NEVER HAPPEN
    llvm_unreachable("Binary operator with unknown opcode?");
  case Instruction::Add:  return Expression::ADD;
  case Instruction::FAdd: return Expression::FADD;
  case Instruction::Sub:  return Expression::SUB;
  case Instruction::FSub: return Expression::FSUB;
  case Instruction::Mul:  return Expression::MUL;
  case Instruction::FMul: return Expression::FMUL;
  case Instruction::UDiv: return Expression::UDIV;
  case Instruction::SDiv: return Expression::SDIV;
  case Instruction::FDiv: return Expression::FDIV;
  case Instruction::URem: return Expression::UREM;
  case Instruction::SRem: return Expression::SREM;
  case Instruction::FRem: return Expression::FREM;
  case Instruction::Shl:  return Expression::SHL;
  case Instruction::LShr: return Expression::LSHR;
  case Instruction::AShr: return Expression::ASHR;
  case Instruction::And:  return Expression::AND;
  case Instruction::Or:   return Expression::OR;
  case Instruction::Xor:  return Expression::XOR;
  }
}

Expression::ExpressionOpcode ValueTable::getOpcode(CmpInst* C) {
  if (isa<ICmpInst>(C)) {
    switch (C->getPredicate()) {
    default:  // THIS SHOULD NEVER HAPPEN
      llvm_unreachable("Comparison with unknown predicate?");
    case ICmpInst::ICMP_EQ:  return Expression::ICMPEQ;
    case ICmpInst::ICMP_NE:  return Expression::ICMPNE;
    case ICmpInst::ICMP_UGT: return Expression::ICMPUGT;
    case ICmpInst::ICMP_UGE: return Expression::ICMPUGE;
    case ICmpInst::ICMP_ULT: return Expression::ICMPULT;
    case ICmpInst::ICMP_ULE: return Expression::ICMPULE;
    case ICmpInst::ICMP_SGT: return Expression::ICMPSGT;
    case ICmpInst::ICMP_SGE: return Expression::ICMPSGE;
    case ICmpInst::ICMP_SLT: return Expression::ICMPSLT;
    case ICmpInst::ICMP_SLE: return Expression::ICMPSLE;
    }
  } else {
    switch (C->getPredicate()) {
    default: // THIS SHOULD NEVER HAPPEN
      llvm_unreachable("Comparison with unknown predicate?");
    case FCmpInst::FCMP_OEQ: return Expression::FCMPOEQ;
    case FCmpInst::FCMP_OGT: return Expression::FCMPOGT;
    case FCmpInst::FCMP_OGE: return Expression::FCMPOGE;
    case FCmpInst::FCMP_OLT: return Expression::FCMPOLT;
    case FCmpInst::FCMP_OLE: return Expression::FCMPOLE;
    case FCmpInst::FCMP_ONE: return Expression::FCMPONE;
    case FCmpInst::FCMP_ORD: return Expression::FCMPORD;
    case FCmpInst::FCMP_UNO: return Expression::FCMPUNO;
    case FCmpInst::FCMP_UEQ: return Expression::FCMPUEQ;
    case FCmpInst::FCMP_UGT: return Expression::FCMPUGT;
    case FCmpInst::FCMP_UGE: return Expression::FCMPUGE;
    case FCmpInst::FCMP_ULT: return Expression::FCMPULT;
    case FCmpInst::FCMP_ULE: return Expression::FCMPULE;
    case FCmpInst::FCMP_UNE: return Expression::FCMPUNE;
    }
  }
}

Expression::ExpressionOpcode ValueTable::getOpcode(CastInst* C) {
  switch(C->getOpcode()) {
  default: // THIS SHOULD NEVER HAPPEN
    llvm_unreachable("Cast operator with unknown opcode?");
  case Instruction::Trunc:    return Expression::TRUNC;
  case Instruction::ZExt:     return Expression::ZEXT;
  case Instruction::SExt:     return Expression::SEXT;
  case Instruction::FPToUI:   return Expression::FPTOUI;
  case Instruction::FPToSI:   return Expression::FPTOSI;
  case Instruction::UIToFP:   return Expression::UITOFP;
  case Instruction::SIToFP:   return Expression::SITOFP;
  case Instruction::FPTrunc:  return Expression::FPTRUNC;
  case Instruction::FPExt:    return Expression::FPEXT;
  case Instruction::PtrToInt: return Expression::PTRTOINT;
  case Instruction::IntToPtr: return Expression::INTTOPTR;
  case Instruction::BitCast:  return Expression::BITCAST;
  }
}

Expression ValueTable::create_expression(CallInst* C) {
  Expression e;

  e.type = C->getType();
  e.opcode = Expression::CALL;

  e.varargs.push_back(lookup(C->getCalledFunction()));
  for (CallInst::op_iterator I = C->op_begin()+1, E = C->op_end();
       I != E; ++I)
    e.varargs.push_back(lookup(*I));

  return e;
}

Expression ValueTable::create_expression(BinaryOperator* BO) {
  Expression e;
  e.varargs.push_back(lookup(BO->getOperand(0)));
  e.varargs.push_back(lookup(BO->getOperand(1)));
  e.type = BO->getType();
  e.opcode = getOpcode(BO);

  return e;
}

Expression ValueTable::create_expression(CmpInst* C) {
  Expression e;

  e.varargs.push_back(lookup(C->getOperand(0)));
  e.varargs.push_back(lookup(C->getOperand(1)));
  e.type = C->getType();
  e.opcode = getOpcode(C);

  return e;
}

Expression ValueTable::create_expression(CastInst* C) {
  Expression e;

  e.varargs.push_back(lookup(C->getOperand(0)));
  e.type = C->getType();
  e.opcode = getOpcode(C);

  return e;
}

Expression ValueTable::create_expression(ShuffleVectorInst* S) {
  Expression e;

  e.varargs.push_back(lookup(S->getOperand(0)));
  e.varargs.push_back(lookup(S->getOperand(1)));
  e.varargs.push_back(lookup(S->getOperand(2)));
  e.type = S->getType();
  e.opcode = Expression::SHUFFLE;

  return e;
}

Expression ValueTable::create_expression(ExtractElementInst* E) {
  Expression e;

  e.varargs.push_back(lookup(E->getOperand(0)));
  e.varargs.push_back(lookup(E->getOperand(1)));
  e.type = E->getType();
  e.opcode = Expression::EXTRACT;

  return e;
}

Expression ValueTable::create_expression(InsertElementInst* I) {
  Expression e;

  e.varargs.push_back(lookup(I->getOperand(0)));
  e.varargs.push_back(lookup(I->getOperand(1)));
  e.varargs.push_back(lookup(I->getOperand(2)));
  e.type = I->getType();
  e.opcode = Expression::INSERT;

  return e;
}

Expression ValueTable::create_expression(SelectInst* I) {
  Expression e;

  e.varargs.push_back(lookup(I->getCondition()));
  e.varargs.push_back(lookup(I->getTrueValue()));
  e.varargs.push_back(lookup(I->getFalseValue()));
  e.type = I->getType();
  e.opcode = Expression::SELECT;

  return e;
}

Expression ValueTable::create_expression(GetElementPtrInst* G) {
  Expression e;

  e.varargs.push_back(lookup(G->getPointerOperand()));
  e.type = G->getType();
  e.opcode = Expression::GEP;

  for (GetElementPtrInst::op_iterator I = G->idx_begin(), E = G->idx_end();
       I != E; ++I)
    e.varargs.push_back(lookup(*I));

  return e;
}

Expression ValueTable::create_expression(ExtractValueInst* E) {
  Expression e;

  e.varargs.push_back(lookup(E->getAggregateOperand()));
  for (ExtractValueInst::idx_iterator II = E->idx_begin(), IE = E->idx_end();
       II != IE; ++II)
    e.varargs.push_back(*II);
  e.type = E->getType();
  e.opcode = Expression::EXTRACTVALUE;

  return e;
}

Expression ValueTable::create_expression(InsertValueInst* E) {
  Expression e;

  e.varargs.push_back(lookup(E->getAggregateOperand()));
  e.varargs.push_back(lookup(E->getInsertedValueOperand()));
  for (InsertValueInst::idx_iterator II = E->idx_begin(), IE = E->idx_end();
       II != IE; ++II)
    e.varargs.push_back(*II);
  e.type = E->getType();
  e.opcode = Expression::INSERTVALUE;

  return e;
}

//===----------------------------------------------------------------------===//
//                     ValueTable External Functions
//===----------------------------------------------------------------------===//

/// add - Insert a value into the table with a specified value number.
void ValueTable::add(Value *V, uint32_t num) {
  valueNumbering[V] = num;
}

/// computeNumber - Returns the value number for the specified value, assigning
/// it a new number if it did not have one before.
uint32_t ValueTable::computeNumber(Value *V) {
  if (uint32_t v = valueNumbering[V])
    return v;
  else if (uint32_t v= constantsNumbering[V])
    return v;

  if (!isa<Instruction>(V)) {
    constantsNumbering[V] = nextValueNumber;
    return nextValueNumber++;
  }
  
  Instruction* I = cast<Instruction>(V);
  Expression exp;
  switch (I->getOpcode()) {
    case Instruction::Add:
    case Instruction::FAdd:
    case Instruction::Sub:
    case Instruction::FSub:
    case Instruction::Mul:
    case Instruction::FMul:
    case Instruction::UDiv:
    case Instruction::SDiv:
    case Instruction::FDiv:
    case Instruction::URem:
    case Instruction::SRem:
    case Instruction::FRem:
    case Instruction::Shl:
    case Instruction::LShr:
    case Instruction::AShr:
    case Instruction::And:
    case Instruction::Or :
    case Instruction::Xor:
      exp = create_expression(cast<BinaryOperator>(I));
      break;
    case Instruction::ICmp:
    case Instruction::FCmp:
      exp = create_expression(cast<CmpInst>(I));
      break;
    case Instruction::Trunc:
    case Instruction::ZExt:
    case Instruction::SExt:
    case Instruction::FPToUI:
    case Instruction::FPToSI:
    case Instruction::UIToFP:
    case Instruction::SIToFP:
    case Instruction::FPTrunc:
    case Instruction::FPExt:
    case Instruction::PtrToInt:
    case Instruction::IntToPtr:
    case Instruction::BitCast:
      exp = create_expression(cast<CastInst>(I));
      break;
    case Instruction::Select:
      exp = create_expression(cast<SelectInst>(I));
      break;
    case Instruction::ExtractElement:
      exp = create_expression(cast<ExtractElementInst>(I));
      break;
    case Instruction::InsertElement:
      exp = create_expression(cast<InsertElementInst>(I));
      break;
    case Instruction::ShuffleVector:
      exp = create_expression(cast<ShuffleVectorInst>(I));
      break;
    case Instruction::ExtractValue:
      exp = create_expression(cast<ExtractValueInst>(I));
      break;
    case Instruction::InsertValue:
      exp = create_expression(cast<InsertValueInst>(I));
      break;      
    case Instruction::GetElementPtr:
      exp = create_expression(cast<GetElementPtrInst>(I));
      break;
    default:
      valueNumbering[V] = nextValueNumber;
      return nextValueNumber++;
  }

  uint32_t& e = expressionNumbering[exp];
  if (!e) e = nextValueNumber++;
  valueNumbering[V] = e;
  
  return e;
}

/// lookup - Returns the value number of the specified value. Returns 0 if
/// the value has not yet been numbered.
uint32_t ValueTable::lookup(Value *V) {
  if (!isa<Instruction>(V)) {
    if (!constantsNumbering.count(V))
      constantsNumbering[V] = nextValueNumber++;
    return constantsNumbering[V];
  }
  
  return valueNumbering[V];
}

/// clear - Remove all entries from the ValueTable
void ValueTable::clear() {
  valueNumbering.clear();
  expressionNumbering.clear();
  constantsNumbering.clear();
  nextValueNumber = 1;
}

void ValueTable::clearExpressions() {
  expressionNumbering.clear();
  constantsNumbering.clear();
  nextValueNumber = 1;
}

/// erase - Remove a value from the value numbering
void ValueTable::erase(Value *V) {
  valueNumbering.erase(V);
}

/// verifyRemoved - Verify that the value is removed from all internal data
/// structures.
void ValueTable::verifyRemoved(const Value *V) const {
  for (DenseMap<Value*, uint32_t>::const_iterator
         I = valueNumbering.begin(), E = valueNumbering.end(); I != E; ++I) {
    assert(I->first != V && "Inst still occurs in value numbering map!");
  }
}

//===----------------------------------------------------------------------===//
//                              SCCVN Pass
//===----------------------------------------------------------------------===//

namespace {

  struct ValueNumberScope {
    ValueNumberScope* parent;
    DenseMap<uint32_t, Value*> table;
    SparseBitVector<128> availIn;
    SparseBitVector<128> availOut;
    
    ValueNumberScope(ValueNumberScope* p) : parent(p) { }
  };

  class SCCVN : public FunctionPass {
    bool runOnFunction(Function &F);
  public:
    static char ID; // Pass identification, replacement for typeid
    SCCVN() : FunctionPass(&ID) { }

  private:
    ValueTable VT;
    DenseMap<BasicBlock*, ValueNumberScope*> BBMap;
    
    // This transformation requires dominator postdominator info
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired<DominatorTree>();

      AU.addPreserved<DominatorTree>();
      AU.setPreservesCFG();
    }
  };

  char SCCVN::ID = 0;
}

// createSCCVNPass - The public interface to this file...
FunctionPass *llvm::createSCCVNPass() { return new SCCVN(); }

static RegisterPass<SCCVN> X("sccvn",
                              "SCC Value Numbering");

static Value *lookupNumber(ValueNumberScope *Locals, uint32_t num) {
  while (Locals) {
    DenseMap<uint32_t, Value*>::iterator I = Locals->table.find(num);
    if (I != Locals->table.end())
      return I->second;
    Locals = Locals->parent;
  }

  return 0;
}

bool SCCVN::runOnFunction(Function& F) {
  // Implement the RPO version of the SCCVN algorithm.  Conceptually, 
  // we optimisitically assume that all instructions with the same opcode have
  // the same VN.  Then we deepen our comparison by one level, to all 
  // instructions whose operands have the same opcodes get the same VN.  We
  // iterate this process until the partitioning stops changing, at which
  // point we have computed a full numbering.
  ReversePostOrderTraversal<Function*> RPOT(&F);
  bool done = false;
  while (!done) {
    done = true;
    VT.clearExpressions();
    for (ReversePostOrderTraversal<Function*>::rpo_iterator I = RPOT.begin(),
         E = RPOT.end(); I != E; ++I) {
      BasicBlock* BB = *I;
      for (BasicBlock::iterator BI = BB->begin(), BE = BB->end();
           BI != BE; ++BI) {
         uint32_t origVN = VT.lookup(BI);
         uint32_t newVN = VT.computeNumber(BI);
         if (origVN != newVN)
           done = false;
      }
    }
  }
  
  // Now, do a dominator walk, eliminating simple, dominated redundancies as we
  // go.  Also, build the ValueNumberScope structure that will be used for
  // computing full availability.
  DominatorTree& DT = getAnalysis<DominatorTree>();
  bool changed = false;
  for (df_iterator<DomTreeNode*> DI = df_begin(DT.getRootNode()),
       DE = df_end(DT.getRootNode()); DI != DE; ++DI) {
    BasicBlock* BB = DI->getBlock();
    if (DI->getIDom())
      BBMap[BB] = new ValueNumberScope(BBMap[DI->getIDom()->getBlock()]);
    else
      BBMap[BB] = new ValueNumberScope(0);
    
    for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; ) {
      uint32_t num = VT.lookup(I);
      Value* repl = lookupNumber(BBMap[BB], num);
      
      if (repl) {
        if (isa<PHINode>(I))
          ++NumSCCVNPhi;
        else
          ++NumSCCVNInstr;
        I->replaceAllUsesWith(repl);
        Instruction* OldInst = I;
        ++I;
        BBMap[BB]->table[num] = repl;
        OldInst->eraseFromParent();
        changed = true;
      } else {
        BBMap[BB]->table[num] = I;
        BBMap[BB]->availOut.set(num);
  
        ++I;
      }
    }
  }

  // Perform a forward data-flow to compute availability at all points on
  // the CFG.
  do {
    changed = false;
    for (ReversePostOrderTraversal<Function*>::rpo_iterator I = RPOT.begin(),
         E = RPOT.end(); I != E; ++I) {
      BasicBlock* BB = *I;
      ValueNumberScope *VNS = BBMap[BB];
      
      SparseBitVector<128> preds;
      bool first = true;
      for (pred_iterator PI = pred_begin(BB), PE = pred_end(BB);
           PI != PE; ++PI) {
        if (first) {
          preds = BBMap[*PI]->availOut;
          first = false;
        } else {
          preds &= BBMap[*PI]->availOut;
        }
      }
      
      changed |= (VNS->availIn |= preds);
      changed |= (VNS->availOut |= preds);
    }
  } while (changed);
  
  // Use full availability information to perform non-dominated replacements.
  SSAUpdater SSU; 
  for (Function::iterator FI = F.begin(), FE = F.end(); FI != FE; ++FI) {
    if (!BBMap.count(FI)) continue;
    for (BasicBlock::iterator BI = FI->begin(), BE = FI->end();
         BI != BE; ) {
      uint32_t num = VT.lookup(BI);
      if (!BBMap[FI]->availIn.test(num)) {
        ++BI;
        continue;
      }
      
      SSU.Initialize(BI);
      
      SmallPtrSet<BasicBlock*, 8> visited;
      SmallVector<BasicBlock*, 8> stack;
      visited.insert(FI);
      for (pred_iterator PI = pred_begin(FI), PE = pred_end(FI);
           PI != PE; ++PI)
        if (!visited.count(*PI))
          stack.push_back(*PI);
      
      while (!stack.empty()) {
        BasicBlock* CurrBB = stack.pop_back_val();
        visited.insert(CurrBB);
        
        ValueNumberScope* S = BBMap[CurrBB];
        if (S->table.count(num)) {
          SSU.AddAvailableValue(CurrBB, S->table[num]);
        } else {
          for (pred_iterator PI = pred_begin(CurrBB), PE = pred_end(CurrBB);
               PI != PE; ++PI)
            if (!visited.count(*PI))
              stack.push_back(*PI);
        }
      }
      
      Value* repl = SSU.GetValueInMiddleOfBlock(FI);
      BI->replaceAllUsesWith(repl);
      Instruction* CurInst = BI;
      ++BI;
      BBMap[FI]->table[num] = repl;
      if (isa<PHINode>(CurInst))
        ++NumSCCVNPhi;
      else
        ++NumSCCVNInstr;
        
      CurInst->eraseFromParent();
    }
  }

  VT.clear();
  for (DenseMap<BasicBlock*, ValueNumberScope*>::iterator
       I = BBMap.begin(), E = BBMap.end(); I != E; ++I)
    delete I->second;
  BBMap.clear();
  
  return changed;
}
