//===- GVN.cpp - Eliminate redundant values and loads ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass performs global value numbering to eliminate fully redundant
// instructions.  It also performs simple dead load elimination.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "gvn"
#include "llvm/Transforms/Scalar.h"
#include "llvm/BasicBlock.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/Instructions.h"
#include "llvm/ParameterAttributes.h"
#include "llvm/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SparseBitVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/MemoryDependenceAnalysis.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/GetElementPtrTypeIterator.h"
#include "llvm/Target/TargetData.h"
#include <list>
using namespace llvm;

STATISTIC(NumGVNInstr, "Number of instructions deleted");
STATISTIC(NumGVNLoad, "Number of loads deleted");
STATISTIC(NumMemSetInfer, "Number of memsets inferred");

namespace {
  cl::opt<bool>
  FormMemSet("form-memset-from-stores",
             cl::desc("Transform straight-line stores to memsets"),
             cl::init(true), cl::Hidden);
}

//===----------------------------------------------------------------------===//
//                         ValueTable Class
//===----------------------------------------------------------------------===//

/// This class holds the mapping between values and value numbers.  It is used
/// as an efficient mechanism to determine the expression-wise equivalence of
/// two values.
namespace {
  struct VISIBILITY_HIDDEN Expression {
    enum ExpressionOpcode { ADD, SUB, MUL, UDIV, SDIV, FDIV, UREM, SREM, 
                            FREM, SHL, LSHR, ASHR, AND, OR, XOR, ICMPEQ, 
                            ICMPNE, ICMPUGT, ICMPUGE, ICMPULT, ICMPULE, 
                            ICMPSGT, ICMPSGE, ICMPSLT, ICMPSLE, FCMPOEQ, 
                            FCMPOGT, FCMPOGE, FCMPOLT, FCMPOLE, FCMPONE, 
                            FCMPORD, FCMPUNO, FCMPUEQ, FCMPUGT, FCMPUGE, 
                            FCMPULT, FCMPULE, FCMPUNE, EXTRACT, INSERT,
                            SHUFFLE, SELECT, TRUNC, ZEXT, SEXT, FPTOUI,
                            FPTOSI, UITOFP, SITOFP, FPTRUNC, FPEXT, 
                            PTRTOINT, INTTOPTR, BITCAST, GEP, CALL, EMPTY,
                            TOMBSTONE };

    ExpressionOpcode opcode;
    const Type* type;
    uint32_t firstVN;
    uint32_t secondVN;
    uint32_t thirdVN;
    SmallVector<uint32_t, 4> varargs;
    Value* function;
  
    Expression() { }
    Expression(ExpressionOpcode o) : opcode(o) { }
  
    bool operator==(const Expression &other) const {
      if (opcode != other.opcode)
        return false;
      else if (opcode == EMPTY || opcode == TOMBSTONE)
        return true;
      else if (type != other.type)
        return false;
      else if (function != other.function)
        return false;
      else if (firstVN != other.firstVN)
        return false;
      else if (secondVN != other.secondVN)
        return false;
      else if (thirdVN != other.thirdVN)
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
      if (opcode != other.opcode)
        return true;
      else if (opcode == EMPTY || opcode == TOMBSTONE)
        return false;
      else if (type != other.type)
        return true;
      else if (function != other.function)
        return true;
      else if (firstVN != other.firstVN)
        return true;
      else if (secondVN != other.secondVN)
        return true;
      else if (thirdVN != other.thirdVN)
        return true;
      else {
        if (varargs.size() != other.varargs.size())
          return true;
      
        for (size_t i = 0; i < varargs.size(); ++i)
          if (varargs[i] != other.varargs[i])
            return true;
    
          return false;
      }
    }
  };
  
  class VISIBILITY_HIDDEN ValueTable {
    private:
      DenseMap<Value*, uint32_t> valueNumbering;
      DenseMap<Expression, uint32_t> expressionNumbering;
      AliasAnalysis* AA;
  
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
    public:
      ValueTable() : nextValueNumber(1) { }
      uint32_t lookup_or_add(Value* V);
      uint32_t lookup(Value* V) const;
      void add(Value* V, uint32_t num);
      void clear();
      void erase(Value* v);
      unsigned size();
      void setAliasAnalysis(AliasAnalysis* A) { AA = A; }
      uint32_t hash_operand(Value* v);
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
    
    hash = e.firstVN + hash * 37;
    hash = e.secondVN + hash * 37;
    hash = e.thirdVN + hash * 37;
    
    hash = ((unsigned)((uintptr_t)e.type >> 4) ^
            (unsigned)((uintptr_t)e.type >> 9)) +
           hash * 37;
    
    for (SmallVector<uint32_t, 4>::const_iterator I = e.varargs.begin(),
         E = e.varargs.end(); I != E; ++I)
      hash = *I + hash * 37;
    
    hash = ((unsigned)((uintptr_t)e.function >> 4) ^
            (unsigned)((uintptr_t)e.function >> 9)) +
           hash * 37;
    
    return hash;
  }
  static bool isEqual(const Expression &LHS, const Expression &RHS) {
    return LHS == RHS;
  }
  static bool isPod() { return true; }
};
}

//===----------------------------------------------------------------------===//
//                     ValueTable Internal Functions
//===----------------------------------------------------------------------===//
Expression::ExpressionOpcode ValueTable::getOpcode(BinaryOperator* BO) {
  switch(BO->getOpcode()) {
  default: // THIS SHOULD NEVER HAPPEN
    assert(0 && "Binary operator with unknown opcode?");
  case Instruction::Add:  return Expression::ADD;
  case Instruction::Sub:  return Expression::SUB;
  case Instruction::Mul:  return Expression::MUL;
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
      assert(0 && "Comparison with unknown predicate?");
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
  }
  assert(isa<FCmpInst>(C) && "Unknown compare");
  switch (C->getPredicate()) {
  default: // THIS SHOULD NEVER HAPPEN
    assert(0 && "Comparison with unknown predicate?");
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

Expression::ExpressionOpcode ValueTable::getOpcode(CastInst* C) {
  switch(C->getOpcode()) {
  default: // THIS SHOULD NEVER HAPPEN
    assert(0 && "Cast operator with unknown opcode?");
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

uint32_t ValueTable::hash_operand(Value* v) {
  if (CallInst* CI = dyn_cast<CallInst>(v))
    if (!AA->doesNotAccessMemory(CI))
      return nextValueNumber++;
  
  return lookup_or_add(v);
}

Expression ValueTable::create_expression(CallInst* C) {
  Expression e;
  
  e.type = C->getType();
  e.firstVN = 0;
  e.secondVN = 0;
  e.thirdVN = 0;
  e.function = C->getCalledFunction();
  e.opcode = Expression::CALL;
  
  for (CallInst::op_iterator I = C->op_begin()+1, E = C->op_end();
       I != E; ++I)
    e.varargs.push_back(hash_operand(*I));
  
  return e;
}

Expression ValueTable::create_expression(BinaryOperator* BO) {
  Expression e;
    
  e.firstVN = hash_operand(BO->getOperand(0));
  e.secondVN = hash_operand(BO->getOperand(1));
  e.thirdVN = 0;
  e.function = 0;
  e.type = BO->getType();
  e.opcode = getOpcode(BO);
  
  return e;
}

Expression ValueTable::create_expression(CmpInst* C) {
  Expression e;
    
  e.firstVN = hash_operand(C->getOperand(0));
  e.secondVN = hash_operand(C->getOperand(1));
  e.thirdVN = 0;
  e.function = 0;
  e.type = C->getType();
  e.opcode = getOpcode(C);
  
  return e;
}

Expression ValueTable::create_expression(CastInst* C) {
  Expression e;
    
  e.firstVN = hash_operand(C->getOperand(0));
  e.secondVN = 0;
  e.thirdVN = 0;
  e.function = 0;
  e.type = C->getType();
  e.opcode = getOpcode(C);
  
  return e;
}

Expression ValueTable::create_expression(ShuffleVectorInst* S) {
  Expression e;
    
  e.firstVN = hash_operand(S->getOperand(0));
  e.secondVN = hash_operand(S->getOperand(1));
  e.thirdVN = hash_operand(S->getOperand(2));
  e.function = 0;
  e.type = S->getType();
  e.opcode = Expression::SHUFFLE;
  
  return e;
}

Expression ValueTable::create_expression(ExtractElementInst* E) {
  Expression e;
    
  e.firstVN = hash_operand(E->getOperand(0));
  e.secondVN = hash_operand(E->getOperand(1));
  e.thirdVN = 0;
  e.function = 0;
  e.type = E->getType();
  e.opcode = Expression::EXTRACT;
  
  return e;
}

Expression ValueTable::create_expression(InsertElementInst* I) {
  Expression e;
    
  e.firstVN = hash_operand(I->getOperand(0));
  e.secondVN = hash_operand(I->getOperand(1));
  e.thirdVN = hash_operand(I->getOperand(2));
  e.function = 0;
  e.type = I->getType();
  e.opcode = Expression::INSERT;
  
  return e;
}

Expression ValueTable::create_expression(SelectInst* I) {
  Expression e;
    
  e.firstVN = hash_operand(I->getCondition());
  e.secondVN = hash_operand(I->getTrueValue());
  e.thirdVN = hash_operand(I->getFalseValue());
  e.function = 0;
  e.type = I->getType();
  e.opcode = Expression::SELECT;
  
  return e;
}

Expression ValueTable::create_expression(GetElementPtrInst* G) {
  Expression e;
    
  e.firstVN = hash_operand(G->getPointerOperand());
  e.secondVN = 0;
  e.thirdVN = 0;
  e.function = 0;
  e.type = G->getType();
  e.opcode = Expression::GEP;
  
  for (GetElementPtrInst::op_iterator I = G->idx_begin(), E = G->idx_end();
       I != E; ++I)
    e.varargs.push_back(hash_operand(*I));
  
  return e;
}

//===----------------------------------------------------------------------===//
//                     ValueTable External Functions
//===----------------------------------------------------------------------===//

/// lookup_or_add - Returns the value number for the specified value, assigning
/// it a new number if it did not have one before.
uint32_t ValueTable::lookup_or_add(Value* V) {
  DenseMap<Value*, uint32_t>::iterator VI = valueNumbering.find(V);
  if (VI != valueNumbering.end())
    return VI->second;
  
  if (CallInst* C = dyn_cast<CallInst>(V)) {
    if (AA->onlyReadsMemory(C)) { // includes doesNotAccessMemory
      Expression e = create_expression(C);
    
      DenseMap<Expression, uint32_t>::iterator EI = expressionNumbering.find(e);
      if (EI != expressionNumbering.end()) {
        valueNumbering.insert(std::make_pair(V, EI->second));
        return EI->second;
      } else {
        expressionNumbering.insert(std::make_pair(e, nextValueNumber));
        valueNumbering.insert(std::make_pair(V, nextValueNumber));
      
        return nextValueNumber++;
      }
    } else {
      valueNumbering.insert(std::make_pair(V, nextValueNumber));
      return nextValueNumber++;
    }
  } else if (BinaryOperator* BO = dyn_cast<BinaryOperator>(V)) {
    Expression e = create_expression(BO);
    
    DenseMap<Expression, uint32_t>::iterator EI = expressionNumbering.find(e);
    if (EI != expressionNumbering.end()) {
      valueNumbering.insert(std::make_pair(V, EI->second));
      return EI->second;
    } else {
      expressionNumbering.insert(std::make_pair(e, nextValueNumber));
      valueNumbering.insert(std::make_pair(V, nextValueNumber));
      
      return nextValueNumber++;
    }
  } else if (CmpInst* C = dyn_cast<CmpInst>(V)) {
    Expression e = create_expression(C);
    
    DenseMap<Expression, uint32_t>::iterator EI = expressionNumbering.find(e);
    if (EI != expressionNumbering.end()) {
      valueNumbering.insert(std::make_pair(V, EI->second));
      return EI->second;
    } else {
      expressionNumbering.insert(std::make_pair(e, nextValueNumber));
      valueNumbering.insert(std::make_pair(V, nextValueNumber));
      
      return nextValueNumber++;
    }
  } else if (ShuffleVectorInst* U = dyn_cast<ShuffleVectorInst>(V)) {
    Expression e = create_expression(U);
    
    DenseMap<Expression, uint32_t>::iterator EI = expressionNumbering.find(e);
    if (EI != expressionNumbering.end()) {
      valueNumbering.insert(std::make_pair(V, EI->second));
      return EI->second;
    } else {
      expressionNumbering.insert(std::make_pair(e, nextValueNumber));
      valueNumbering.insert(std::make_pair(V, nextValueNumber));
      
      return nextValueNumber++;
    }
  } else if (ExtractElementInst* U = dyn_cast<ExtractElementInst>(V)) {
    Expression e = create_expression(U);
    
    DenseMap<Expression, uint32_t>::iterator EI = expressionNumbering.find(e);
    if (EI != expressionNumbering.end()) {
      valueNumbering.insert(std::make_pair(V, EI->second));
      return EI->second;
    } else {
      expressionNumbering.insert(std::make_pair(e, nextValueNumber));
      valueNumbering.insert(std::make_pair(V, nextValueNumber));
      
      return nextValueNumber++;
    }
  } else if (InsertElementInst* U = dyn_cast<InsertElementInst>(V)) {
    Expression e = create_expression(U);
    
    DenseMap<Expression, uint32_t>::iterator EI = expressionNumbering.find(e);
    if (EI != expressionNumbering.end()) {
      valueNumbering.insert(std::make_pair(V, EI->second));
      return EI->second;
    } else {
      expressionNumbering.insert(std::make_pair(e, nextValueNumber));
      valueNumbering.insert(std::make_pair(V, nextValueNumber));
      
      return nextValueNumber++;
    }
  } else if (SelectInst* U = dyn_cast<SelectInst>(V)) {
    Expression e = create_expression(U);
    
    DenseMap<Expression, uint32_t>::iterator EI = expressionNumbering.find(e);
    if (EI != expressionNumbering.end()) {
      valueNumbering.insert(std::make_pair(V, EI->second));
      return EI->second;
    } else {
      expressionNumbering.insert(std::make_pair(e, nextValueNumber));
      valueNumbering.insert(std::make_pair(V, nextValueNumber));
      
      return nextValueNumber++;
    }
  } else if (CastInst* U = dyn_cast<CastInst>(V)) {
    Expression e = create_expression(U);
    
    DenseMap<Expression, uint32_t>::iterator EI = expressionNumbering.find(e);
    if (EI != expressionNumbering.end()) {
      valueNumbering.insert(std::make_pair(V, EI->second));
      return EI->second;
    } else {
      expressionNumbering.insert(std::make_pair(e, nextValueNumber));
      valueNumbering.insert(std::make_pair(V, nextValueNumber));
      
      return nextValueNumber++;
    }
  } else if (GetElementPtrInst* U = dyn_cast<GetElementPtrInst>(V)) {
    Expression e = create_expression(U);
    
    DenseMap<Expression, uint32_t>::iterator EI = expressionNumbering.find(e);
    if (EI != expressionNumbering.end()) {
      valueNumbering.insert(std::make_pair(V, EI->second));
      return EI->second;
    } else {
      expressionNumbering.insert(std::make_pair(e, nextValueNumber));
      valueNumbering.insert(std::make_pair(V, nextValueNumber));
      
      return nextValueNumber++;
    }
  } else {
    valueNumbering.insert(std::make_pair(V, nextValueNumber));
    return nextValueNumber++;
  }
}

/// lookup - Returns the value number of the specified value. Fails if
/// the value has not yet been numbered.
uint32_t ValueTable::lookup(Value* V) const {
  DenseMap<Value*, uint32_t>::iterator VI = valueNumbering.find(V);
  assert(VI != valueNumbering.end() && "Value not numbered?");
  return VI->second;
}

/// clear - Remove all entries from the ValueTable
void ValueTable::clear() {
  valueNumbering.clear();
  expressionNumbering.clear();
  nextValueNumber = 1;
}

/// erase - Remove a value from the value numbering
void ValueTable::erase(Value* V) {
  valueNumbering.erase(V);
}

//===----------------------------------------------------------------------===//
//                       ValueNumberedSet Class
//===----------------------------------------------------------------------===//
namespace {
class VISIBILITY_HIDDEN ValueNumberedSet {
  private:
    SmallPtrSet<Value*, 8> contents;
    SparseBitVector<64> numbers;
  public:
    ValueNumberedSet() { }
    ValueNumberedSet(const ValueNumberedSet& other) {
      numbers = other.numbers;
      contents = other.contents;
    }
    
    typedef SmallPtrSet<Value*, 8>::iterator iterator;
    
    iterator begin() { return contents.begin(); }
    iterator end() { return contents.end(); }
    
    bool insert(Value* v) { return contents.insert(v); }
    void insert(iterator I, iterator E) { contents.insert(I, E); }
    void erase(Value* v) { contents.erase(v); }
    unsigned count(Value* v) { return contents.count(v); }
    size_t size() { return contents.size(); }
    
    void set(unsigned i)  {
      numbers.set(i);
    }
    
    void operator=(const ValueNumberedSet& other) {
      contents = other.contents;
      numbers = other.numbers;
    }
    
    void reset(unsigned i)  {
      numbers.reset(i);
    }
    
    bool test(unsigned i)  {
      return numbers.test(i);
    }
};
}

//===----------------------------------------------------------------------===//
//                         GVN Pass
//===----------------------------------------------------------------------===//

namespace {

  class VISIBILITY_HIDDEN GVN : public FunctionPass {
    bool runOnFunction(Function &F);
  public:
    static char ID; // Pass identification, replacement for typeid
    GVN() : FunctionPass((intptr_t)&ID) { }

  private:
    ValueTable VN;
    
    DenseMap<BasicBlock*, ValueNumberedSet> availableOut;
    
    typedef DenseMap<Value*, SmallPtrSet<Instruction*, 4> > PhiMapType;
    PhiMapType phiMap;
    
    
    // This transformation requires dominator postdominator info
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesCFG();
      AU.addRequired<DominatorTree>();
      AU.addRequired<MemoryDependenceAnalysis>();
      AU.addRequired<AliasAnalysis>();
      AU.addRequired<TargetData>();
      AU.addPreserved<AliasAnalysis>();
      AU.addPreserved<MemoryDependenceAnalysis>();
      AU.addPreserved<TargetData>();
    }
  
    // Helper fuctions
    // FIXME: eliminate or document these better
    Value* find_leader(ValueNumberedSet& vals, uint32_t v) ;
    void val_insert(ValueNumberedSet& s, Value* v);
    bool processLoad(LoadInst* L,
                     DenseMap<Value*, LoadInst*> &lastLoad,
                     SmallVectorImpl<Instruction*> &toErase);
    bool processStore(StoreInst *SI, SmallVectorImpl<Instruction*> &toErase);
    bool processInstruction(Instruction* I,
                            ValueNumberedSet& currAvail,
                            DenseMap<Value*, LoadInst*>& lastSeenLoad,
                            SmallVectorImpl<Instruction*> &toErase);
    bool processNonLocalLoad(LoadInst* L,
                             SmallVectorImpl<Instruction*> &toErase);
    bool processMemCpy(MemCpyInst* M, MemCpyInst* MDep,
                       SmallVectorImpl<Instruction*> &toErase);
    bool performCallSlotOptzn(MemCpyInst* cpy, CallInst* C,
                              SmallVectorImpl<Instruction*> &toErase);
    Value *GetValueForBlock(BasicBlock *BB, LoadInst* orig,
                            DenseMap<BasicBlock*, Value*> &Phis,
                            bool top_level = false);
    void dump(DenseMap<BasicBlock*, Value*>& d);
    bool iterateOnFunction(Function &F);
    Value* CollapsePhi(PHINode* p);
    bool isSafeReplacement(PHINode* p, Instruction* inst);
  };
  
  char GVN::ID = 0;
}

// createGVNPass - The public interface to this file...
FunctionPass *llvm::createGVNPass() { return new GVN(); }

static RegisterPass<GVN> X("gvn",
                           "Global Value Numbering");

/// find_leader - Given a set and a value number, return the first
/// element of the set with that value number, or 0 if no such element
/// is present
Value* GVN::find_leader(ValueNumberedSet& vals, uint32_t v) {
  if (!vals.test(v))
    return 0;
  
  for (ValueNumberedSet::iterator I = vals.begin(), E = vals.end();
       I != E; ++I)
    if (v == VN.lookup(*I))
      return *I;
  
  assert(0 && "No leader found, but present bit is set?");
  return 0;
}

/// val_insert - Insert a value into a set only if there is not a value
/// with the same value number already in the set
void GVN::val_insert(ValueNumberedSet& s, Value* v) {
  uint32_t num = VN.lookup(v);
  if (!s.test(num))
    s.insert(v);
}

void GVN::dump(DenseMap<BasicBlock*, Value*>& d) {
  printf("{\n");
  for (DenseMap<BasicBlock*, Value*>::iterator I = d.begin(),
       E = d.end(); I != E; ++I) {
    if (I->second == MemoryDependenceAnalysis::None)
      printf("None\n");
    else
      I->second->dump();
  }
  printf("}\n");
}

Value* GVN::CollapsePhi(PHINode* p) {
  DominatorTree &DT = getAnalysis<DominatorTree>();
  Value* constVal = p->hasConstantValue();
  
  if (!constVal) return 0;
  
  Instruction* inst = dyn_cast<Instruction>(constVal);
  if (!inst)
    return constVal;
    
  if (DT.dominates(inst, p))
    if (isSafeReplacement(p, inst))
      return inst;
  return 0;
}

bool GVN::isSafeReplacement(PHINode* p, Instruction* inst) {
  if (!isa<PHINode>(inst))
    return true;
  
  for (Instruction::use_iterator UI = p->use_begin(), E = p->use_end();
       UI != E; ++UI)
    if (PHINode* use_phi = dyn_cast<PHINode>(UI))
      if (use_phi->getParent() == inst->getParent())
        return false;
  
  return true;
}

/// GetValueForBlock - Get the value to use within the specified basic block.
/// available values are in Phis.
Value *GVN::GetValueForBlock(BasicBlock *BB, LoadInst* orig,
                             DenseMap<BasicBlock*, Value*> &Phis,
                             bool top_level) { 
                                 
  // If we have already computed this value, return the previously computed val.
  DenseMap<BasicBlock*, Value*>::iterator V = Phis.find(BB);
  if (V != Phis.end() && !top_level) return V->second;
  
  BasicBlock* singlePred = BB->getSinglePredecessor();
  if (singlePred) {
    Value *ret = GetValueForBlock(singlePred, orig, Phis);
    Phis[BB] = ret;
    return ret;
  }
  
  // Otherwise, the idom is the loop, so we need to insert a PHI node.  Do so
  // now, then get values to fill in the incoming values for the PHI.
  PHINode *PN = PHINode::Create(orig->getType(), orig->getName()+".rle",
                                BB->begin());
  PN->reserveOperandSpace(std::distance(pred_begin(BB), pred_end(BB)));
  
  if (Phis.count(BB) == 0)
    Phis.insert(std::make_pair(BB, PN));
  
  // Fill in the incoming values for the block.
  for (pred_iterator PI = pred_begin(BB), E = pred_end(BB); PI != E; ++PI) {
    Value* val = GetValueForBlock(*PI, orig, Phis);
    PN->addIncoming(val, *PI);
  }
  
  AliasAnalysis& AA = getAnalysis<AliasAnalysis>();
  AA.copyValue(orig, PN);
  
  // Attempt to collapse PHI nodes that are trivially redundant
  Value* v = CollapsePhi(PN);
  if (!v) {
    // Cache our phi construction results
    phiMap[orig->getPointerOperand()].insert(PN);
    return PN;
  }
    
  MemoryDependenceAnalysis& MD = getAnalysis<MemoryDependenceAnalysis>();

  MD.removeInstruction(PN);
  PN->replaceAllUsesWith(v);

  for (DenseMap<BasicBlock*, Value*>::iterator I = Phis.begin(),
       E = Phis.end(); I != E; ++I)
    if (I->second == PN)
      I->second = v;

  PN->eraseFromParent();

  Phis[BB] = v;
  return v;
}

/// processNonLocalLoad - Attempt to eliminate a load whose dependencies are
/// non-local by performing PHI construction.
bool GVN::processNonLocalLoad(LoadInst* L,
                              SmallVectorImpl<Instruction*> &toErase) {
  MemoryDependenceAnalysis& MD = getAnalysis<MemoryDependenceAnalysis>();
  
  // Find the non-local dependencies of the load
  DenseMap<BasicBlock*, Value*> deps;
  MD.getNonLocalDependency(L, deps);
  
  DenseMap<BasicBlock*, Value*> repl;
  
  // Filter out useless results (non-locals, etc)
  for (DenseMap<BasicBlock*, Value*>::iterator I = deps.begin(), E = deps.end();
       I != E; ++I) {
    if (I->second == MemoryDependenceAnalysis::None)
      return false;
  
    if (I->second == MemoryDependenceAnalysis::NonLocal)
      continue;
  
    if (StoreInst* S = dyn_cast<StoreInst>(I->second)) {
      if (S->getPointerOperand() != L->getPointerOperand())
        return false;
      repl[I->first] = S->getOperand(0);
    } else if (LoadInst* LD = dyn_cast<LoadInst>(I->second)) {
      if (LD->getPointerOperand() != L->getPointerOperand())
        return false;
      repl[I->first] = LD;
    } else {
      return false;
    }
  }
  
  // Use cached PHI construction information from previous runs
  SmallPtrSet<Instruction*, 4>& p = phiMap[L->getPointerOperand()];
  for (SmallPtrSet<Instruction*, 4>::iterator I = p.begin(), E = p.end();
       I != E; ++I) {
    if ((*I)->getParent() == L->getParent()) {
      MD.removeInstruction(L);
      L->replaceAllUsesWith(*I);
      toErase.push_back(L);
      NumGVNLoad++;
      return true;
    }
    
    repl.insert(std::make_pair((*I)->getParent(), *I));
  }
  
  // Perform PHI construction
  SmallPtrSet<BasicBlock*, 4> visited;
  Value* v = GetValueForBlock(L->getParent(), L, repl, true);
  
  MD.removeInstruction(L);
  L->replaceAllUsesWith(v);
  toErase.push_back(L);
  NumGVNLoad++;

  return true;
}

/// processLoad - Attempt to eliminate a load, first by eliminating it
/// locally, and then attempting non-local elimination if that fails.
bool GVN::processLoad(LoadInst *L, DenseMap<Value*, LoadInst*> &lastLoad,
                      SmallVectorImpl<Instruction*> &toErase) {
  if (L->isVolatile()) {
    lastLoad[L->getPointerOperand()] = L;
    return false;
  }
  
  Value* pointer = L->getPointerOperand();
  LoadInst*& last = lastLoad[pointer];
  
  // ... to a pointer that has been loaded from before...
  MemoryDependenceAnalysis& MD = getAnalysis<MemoryDependenceAnalysis>();
  bool removedNonLocal = false;
  Instruction* dep = MD.getDependency(L);
  if (dep == MemoryDependenceAnalysis::NonLocal &&
      L->getParent() != &L->getParent()->getParent()->getEntryBlock()) {
    removedNonLocal = processNonLocalLoad(L, toErase);
    
    if (!removedNonLocal)
      last = L;
    
    return removedNonLocal;
  }
  
  
  bool deletedLoad = false;
  
  // Walk up the dependency chain until we either find
  // a dependency we can use, or we can't walk any further
  while (dep != MemoryDependenceAnalysis::None &&
         dep != MemoryDependenceAnalysis::NonLocal &&
         (isa<LoadInst>(dep) || isa<StoreInst>(dep))) {
    // ... that depends on a store ...
    if (StoreInst* S = dyn_cast<StoreInst>(dep)) {
      if (S->getPointerOperand() == pointer) {
        // Remove it!
        MD.removeInstruction(L);
        
        L->replaceAllUsesWith(S->getOperand(0));
        toErase.push_back(L);
        deletedLoad = true;
        NumGVNLoad++;
      }
      
      // Whether we removed it or not, we can't
      // go any further
      break;
    } else if (!last) {
      // If we don't depend on a store, and we haven't
      // been loaded before, bail.
      break;
    } else if (dep == last) {
      // Remove it!
      MD.removeInstruction(L);
      
      L->replaceAllUsesWith(last);
      toErase.push_back(L);
      deletedLoad = true;
      NumGVNLoad++;
        
      break;
    } else {
      dep = MD.getDependency(L, dep);
    }
  }

  if (dep != MemoryDependenceAnalysis::None &&
      dep != MemoryDependenceAnalysis::NonLocal &&
      isa<AllocationInst>(dep)) {
    // Check that this load is actually from the
    // allocation we found
    Value* v = L->getOperand(0);
    while (true) {
      if (BitCastInst *BC = dyn_cast<BitCastInst>(v))
        v = BC->getOperand(0);
      else if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(v))
        v = GEP->getOperand(0);
      else
        break;
    }
    if (v == dep) {
      // If this load depends directly on an allocation, there isn't
      // anything stored there; therefore, we can optimize this load
      // to undef.
      MD.removeInstruction(L);

      L->replaceAllUsesWith(UndefValue::get(L->getType()));
      toErase.push_back(L);
      deletedLoad = true;
      NumGVNLoad++;
    }
  }

  if (!deletedLoad)
    last = L;
  
  return deletedLoad;
}

/// isBytewiseValue - If the specified value can be set by repeating the same
/// byte in memory, return the i8 value that it is represented with.  This is
/// true for all i8 values obviously, but is also true for i32 0, i32 -1,
/// i16 0xF0F0, double 0.0 etc.  If the value can't be handled with a repeated
/// byte store (e.g. i16 0x1234), return null.
static Value *isBytewiseValue(Value *V) {
  // All byte-wide stores are splatable, even of arbitrary variables.
  if (V->getType() == Type::Int8Ty) return V;
  
  // Constant float and double values can be handled as integer values if the
  // corresponding integer value is "byteable".  An important case is 0.0. 
  if (ConstantFP *CFP = dyn_cast<ConstantFP>(V)) {
    if (CFP->getType() == Type::FloatTy)
      V = ConstantExpr::getBitCast(CFP, Type::Int32Ty);
    if (CFP->getType() == Type::DoubleTy)
      V = ConstantExpr::getBitCast(CFP, Type::Int64Ty);
    // Don't handle long double formats, which have strange constraints.
  }
  
  // We can handle constant integers that are power of two in size and a 
  // multiple of 8 bits.
  if (ConstantInt *CI = dyn_cast<ConstantInt>(V)) {
    unsigned Width = CI->getBitWidth();
    if (isPowerOf2_32(Width) && Width > 8) {
      // We can handle this value if the recursive binary decomposition is the
      // same at all levels.
      APInt Val = CI->getValue();
      APInt Val2;
      while (Val.getBitWidth() != 8) {
        unsigned NextWidth = Val.getBitWidth()/2;
        Val2  = Val.lshr(NextWidth);
        Val2.trunc(Val.getBitWidth()/2);
        Val.trunc(Val.getBitWidth()/2);

        // If the top/bottom halves aren't the same, reject it.
        if (Val != Val2)
          return 0;
      }
      return ConstantInt::get(Val);
    }
  }
  
  // Conceptually, we could handle things like:
  //   %a = zext i8 %X to i16
  //   %b = shl i16 %a, 8
  //   %c = or i16 %a, %b
  // but until there is an example that actually needs this, it doesn't seem
  // worth worrying about.
  return 0;
}

static int64_t GetOffsetFromIndex(const GetElementPtrInst *GEP, unsigned Idx,
                                  bool &VariableIdxFound, TargetData &TD) {
  // Skip over the first indices.
  gep_type_iterator GTI = gep_type_begin(GEP);
  for (unsigned i = 1; i != Idx; ++i, ++GTI)
    /*skip along*/;
  
  // Compute the offset implied by the rest of the indices.
  int64_t Offset = 0;
  for (unsigned i = Idx, e = GEP->getNumOperands(); i != e; ++i, ++GTI) {
    ConstantInt *OpC = dyn_cast<ConstantInt>(GEP->getOperand(i));
    if (OpC == 0)
      return VariableIdxFound = true;
    if (OpC->isZero()) continue;  // No offset.

    // Handle struct indices, which add their field offset to the pointer.
    if (const StructType *STy = dyn_cast<StructType>(*GTI)) {
      Offset += TD.getStructLayout(STy)->getElementOffset(OpC->getZExtValue());
      continue;
    }
    
    // Otherwise, we have a sequential type like an array or vector.  Multiply
    // the index by the ElementSize.
    uint64_t Size = TD.getABITypeSize(GTI.getIndexedType());
    Offset += Size*OpC->getSExtValue();
  }

  return Offset;
}

/// IsPointerOffset - Return true if Ptr1 is provably equal to Ptr2 plus a
/// constant offset, and return that constant offset.  For example, Ptr1 might
/// be &A[42], and Ptr2 might be &A[40].  In this case offset would be -8.
static bool IsPointerOffset(Value *Ptr1, Value *Ptr2, int64_t &Offset,
                            TargetData &TD) {
  // Right now we handle the case when Ptr1/Ptr2 are both GEPs with an identical
  // base.  After that base, they may have some number of common (and
  // potentially variable) indices.  After that they handle some constant
  // offset, which determines their offset from each other.  At this point, we
  // handle no other case.
  GetElementPtrInst *GEP1 = dyn_cast<GetElementPtrInst>(Ptr1);
  GetElementPtrInst *GEP2 = dyn_cast<GetElementPtrInst>(Ptr2);
  if (!GEP1 || !GEP2 || GEP1->getOperand(0) != GEP2->getOperand(0))
    return false;
  
  // Skip any common indices and track the GEP types.
  unsigned Idx = 1;
  for (; Idx != GEP1->getNumOperands() && Idx != GEP2->getNumOperands(); ++Idx)
    if (GEP1->getOperand(Idx) != GEP2->getOperand(Idx))
      break;

  bool VariableIdxFound = false;
  int64_t Offset1 = GetOffsetFromIndex(GEP1, Idx, VariableIdxFound, TD);
  int64_t Offset2 = GetOffsetFromIndex(GEP2, Idx, VariableIdxFound, TD);
  if (VariableIdxFound) return false;
  
  Offset = Offset2-Offset1;
  return true;
}


/// MemsetRange - Represents a range of memset'd bytes with the ByteVal value.
/// This allows us to analyze stores like:
///   store 0 -> P+1
///   store 0 -> P+0
///   store 0 -> P+3
///   store 0 -> P+2
/// which sometimes happens with stores to arrays of structs etc.  When we see
/// the first store, we make a range [1, 2).  The second store extends the range
/// to [0, 2).  The third makes a new range [2, 3).  The fourth store joins the
/// two ranges into [0, 3) which is memset'able.
namespace {
struct MemsetRange {
  // Start/End - A semi range that describes the span that this range covers.
  // The range is closed at the start and open at the end: [Start, End).  
  int64_t Start, End;

  /// StartPtr - The getelementptr instruction that points to the start of the
  /// range.
  Value *StartPtr;
  
  /// Alignment - The known alignment of the first store.
  unsigned Alignment;
  
  /// TheStores - The actual stores that make up this range.
  SmallVector<StoreInst*, 16> TheStores;
  
  bool isProfitableToUseMemset(const TargetData &TD) const;

};
} // end anon namespace

bool MemsetRange::isProfitableToUseMemset(const TargetData &TD) const {
  // If we found more than 8 stores to merge or 64 bytes, use memset.
  if (TheStores.size() >= 8 || End-Start >= 64) return true;
  
  // Assume that the code generator is capable of merging pairs of stores
  // together if it wants to.
  if (TheStores.size() <= 2) return false;
  
  // If we have fewer than 8 stores, it can still be worthwhile to do this.
  // For example, merging 4 i8 stores into an i32 store is useful almost always.
  // However, merging 2 32-bit stores isn't useful on a 32-bit architecture (the
  // memset will be split into 2 32-bit stores anyway) and doing so can
  // pessimize the llvm optimizer.
  //
  // Since we don't have perfect knowledge here, make some assumptions: assume
  // the maximum GPR width is the same size as the pointer size and assume that
  // this width can be stored.  If so, check to see whether we will end up
  // actually reducing the number of stores used.
  unsigned Bytes = unsigned(End-Start);
  unsigned NumPointerStores = Bytes/TD.getPointerSize();
  
  // Assume the remaining bytes if any are done a byte at a time.
  unsigned NumByteStores = Bytes - NumPointerStores*TD.getPointerSize();
  
  // If we will reduce the # stores (according to this heuristic), do the
  // transformation.  This encourages merging 4 x i8 -> i32 and 2 x i16 -> i32
  // etc.
  return TheStores.size() > NumPointerStores+NumByteStores;
}    


namespace {
class MemsetRanges {
  /// Ranges - A sorted list of the memset ranges.  We use std::list here
  /// because each element is relatively large and expensive to copy.
  std::list<MemsetRange> Ranges;
  typedef std::list<MemsetRange>::iterator range_iterator;
  TargetData &TD;
public:
  MemsetRanges(TargetData &td) : TD(td) {}
  
  typedef std::list<MemsetRange>::const_iterator const_iterator;
  const_iterator begin() const { return Ranges.begin(); }
  const_iterator end() const { return Ranges.end(); }
  bool empty() const { return Ranges.empty(); }
  
  void addStore(int64_t OffsetFromFirst, StoreInst *SI);
};
  
} // end anon namespace


/// addStore - Add a new store to the MemsetRanges data structure.  This adds a
/// new range for the specified store at the specified offset, merging into
/// existing ranges as appropriate.
void MemsetRanges::addStore(int64_t Start, StoreInst *SI) {
  int64_t End = Start+TD.getTypeStoreSize(SI->getOperand(0)->getType());
  
  // Do a linear search of the ranges to see if this can be joined and/or to
  // find the insertion point in the list.  We keep the ranges sorted for
  // simplicity here.  This is a linear search of a linked list, which is ugly,
  // however the number of ranges is limited, so this won't get crazy slow.
  range_iterator I = Ranges.begin(), E = Ranges.end();
  
  while (I != E && Start > I->End)
    ++I;
  
  // We now know that I == E, in which case we didn't find anything to merge
  // with, or that Start <= I->End.  If End < I->Start or I == E, then we need
  // to insert a new range.  Handle this now.
  if (I == E || End < I->Start) {
    MemsetRange &R = *Ranges.insert(I, MemsetRange());
    R.Start        = Start;
    R.End          = End;
    R.StartPtr     = SI->getPointerOperand();
    R.Alignment    = SI->getAlignment();
    R.TheStores.push_back(SI);
    return;
  }

  // This store overlaps with I, add it.
  I->TheStores.push_back(SI);
  
  // At this point, we may have an interval that completely contains our store.
  // If so, just add it to the interval and return.
  if (I->Start <= Start && I->End >= End)
    return;
  
  // Now we know that Start <= I->End and End >= I->Start so the range overlaps
  // but is not entirely contained within the range.
  
  // See if the range extends the start of the range.  In this case, it couldn't
  // possibly cause it to join the prior range, because otherwise we would have
  // stopped on *it*.
  if (Start < I->Start) {
    I->Start = Start;
    I->StartPtr = SI->getPointerOperand();
  }
    
  // Now we know that Start <= I->End and Start >= I->Start (so the startpoint
  // is in or right at the end of I), and that End >= I->Start.  Extend I out to
  // End.
  if (End > I->End) {
    I->End = End;
    range_iterator NextI = I;;
    while (++NextI != E && End >= NextI->Start) {
      // Merge the range in.
      I->TheStores.append(NextI->TheStores.begin(), NextI->TheStores.end());
      if (NextI->End > I->End)
        I->End = NextI->End;
      Ranges.erase(NextI);
      NextI = I;
    }
  }
}



/// processStore - When GVN is scanning forward over instructions, we look for
/// some other patterns to fold away.  In particular, this looks for stores to
/// neighboring locations of memory.  If it sees enough consequtive ones
/// (currently 4) it attempts to merge them together into a memcpy/memset.
bool GVN::processStore(StoreInst *SI, SmallVectorImpl<Instruction*> &toErase) {
  if (!FormMemSet) return false;
  if (SI->isVolatile()) return false;
  
  // There are two cases that are interesting for this code to handle: memcpy
  // and memset.  Right now we only handle memset.
  
  // Ensure that the value being stored is something that can be memset'able a
  // byte at a time like "0" or "-1" or any width, as well as things like
  // 0xA0A0A0A0 and 0.0.
  Value *ByteVal = isBytewiseValue(SI->getOperand(0));
  if (!ByteVal)
    return false;

  TargetData &TD = getAnalysis<TargetData>();
  AliasAnalysis &AA = getAnalysis<AliasAnalysis>();

  // Okay, so we now have a single store that can be splatable.  Scan to find
  // all subsequent stores of the same value to offset from the same pointer.
  // Join these together into ranges, so we can decide whether contiguous blocks
  // are stored.
  MemsetRanges Ranges(TD);
  
  Value *StartPtr = SI->getPointerOperand();
  
  BasicBlock::iterator BI = SI;
  for (++BI; !isa<TerminatorInst>(BI); ++BI) {
    if (isa<CallInst>(BI) || isa<InvokeInst>(BI)) { 
      // If the call is readnone, ignore it, otherwise bail out.  We don't even
      // allow readonly here because we don't want something like:
      // A[1] = 2; strlen(A); A[2] = 2; -> memcpy(A, ...); strlen(A).
      if (AA.getModRefBehavior(CallSite::get(BI)) ==
            AliasAnalysis::DoesNotAccessMemory)
        continue;
      
      // TODO: If this is a memset, try to join it in.
      
      break;
    } else if (isa<VAArgInst>(BI) || isa<LoadInst>(BI))
      break;

    // If this is a non-store instruction it is fine, ignore it.
    StoreInst *NextStore = dyn_cast<StoreInst>(BI);
    if (NextStore == 0) continue;
    
    // If this is a store, see if we can merge it in.
    if (NextStore->isVolatile()) break;
    
    // Check to see if this stored value is of the same byte-splattable value.
    if (ByteVal != isBytewiseValue(NextStore->getOperand(0)))
      break;

    // Check to see if this store is to a constant offset from the start ptr.
    int64_t Offset;
    if (!IsPointerOffset(StartPtr, NextStore->getPointerOperand(), Offset, TD))
      break;

    Ranges.addStore(Offset, NextStore);
  }

  // If we have no ranges, then we just had a single store with nothing that
  // could be merged in.  This is a very common case of course.
  if (Ranges.empty())
    return false;
  
  // If we had at least one store that could be merged in, add the starting
  // store as well.  We try to avoid this unless there is at least something
  // interesting as a small compile-time optimization.
  Ranges.addStore(0, SI);

  
  Function *MemSetF = 0;
  
  // Now that we have full information about ranges, loop over the ranges and
  // emit memset's for anything big enough to be worthwhile.
  bool MadeChange = false;
  for (MemsetRanges::const_iterator I = Ranges.begin(), E = Ranges.end();
       I != E; ++I) {
    const MemsetRange &Range = *I;

    if (Range.TheStores.size() == 1) continue;
    
    // If it is profitable to lower this range to memset, do so now.
    if (!Range.isProfitableToUseMemset(TD))
      continue;
    
    // Otherwise, we do want to transform this!  Create a new memset.  We put
    // the memset right before the first instruction that isn't part of this
    // memset block.  This ensure that the memset is dominated by any addressing
    // instruction needed by the start of the block.
    BasicBlock::iterator InsertPt = BI;
  
    if (MemSetF == 0)
      MemSetF = Intrinsic::getDeclaration(SI->getParent()->getParent()
                                          ->getParent(), Intrinsic::memset_i64);
    
    // Get the starting pointer of the block.
    StartPtr = Range.StartPtr;
  
    // Cast the start ptr to be i8* as memset requires.
    const Type *i8Ptr = PointerType::getUnqual(Type::Int8Ty);
    if (StartPtr->getType() != i8Ptr)
      StartPtr = new BitCastInst(StartPtr, i8Ptr, StartPtr->getNameStart(),
                                 InsertPt);
  
    Value *Ops[] = {
      StartPtr, ByteVal,   // Start, value
      ConstantInt::get(Type::Int64Ty, Range.End-Range.Start),  // size
      ConstantInt::get(Type::Int32Ty, Range.Alignment)   // align
    };
    Value *C = CallInst::Create(MemSetF, Ops, Ops+4, "", InsertPt);
    DEBUG(cerr << "Replace stores:\n";
          for (unsigned i = 0, e = Range.TheStores.size(); i != e; ++i)
            cerr << *Range.TheStores[i];
          cerr << "With: " << *C); C=C;
  
    // Zap all the stores.
    toErase.append(Range.TheStores.begin(), Range.TheStores.end());
    ++NumMemSetInfer;
    MadeChange = true;
  }
  
  return MadeChange;
}


/// performCallSlotOptzn - takes a memcpy and a call that it depends on,
/// and checks for the possibility of a call slot optimization by having
/// the call write its result directly into the destination of the memcpy.
bool GVN::performCallSlotOptzn(MemCpyInst *cpy, CallInst *C,
                               SmallVectorImpl<Instruction*> &toErase) {
  // The general transformation to keep in mind is
  //
  //   call @func(..., src, ...)
  //   memcpy(dest, src, ...)
  //
  // ->
  //
  //   memcpy(dest, src, ...)
  //   call @func(..., dest, ...)
  //
  // Since moving the memcpy is technically awkward, we additionally check that
  // src only holds uninitialized values at the moment of the call, meaning that
  // the memcpy can be discarded rather than moved.

  // Deliberately get the source and destination with bitcasts stripped away,
  // because we'll need to do type comparisons based on the underlying type.
  Value* cpyDest = cpy->getDest();
  Value* cpySrc = cpy->getSource();
  CallSite CS = CallSite::get(C);

  // We need to be able to reason about the size of the memcpy, so we require
  // that it be a constant.
  ConstantInt* cpyLength = dyn_cast<ConstantInt>(cpy->getLength());
  if (!cpyLength)
    return false;

  // Require that src be an alloca.  This simplifies the reasoning considerably.
  AllocaInst* srcAlloca = dyn_cast<AllocaInst>(cpySrc);
  if (!srcAlloca)
    return false;

  // Check that all of src is copied to dest.
  TargetData& TD = getAnalysis<TargetData>();

  ConstantInt* srcArraySize = dyn_cast<ConstantInt>(srcAlloca->getArraySize());
  if (!srcArraySize)
    return false;

  uint64_t srcSize = TD.getABITypeSize(srcAlloca->getAllocatedType()) *
    srcArraySize->getZExtValue();

  if (cpyLength->getZExtValue() < srcSize)
    return false;

  // Check that accessing the first srcSize bytes of dest will not cause a
  // trap.  Otherwise the transform is invalid since it might cause a trap
  // to occur earlier than it otherwise would.
  if (AllocaInst* A = dyn_cast<AllocaInst>(cpyDest)) {
    // The destination is an alloca.  Check it is larger than srcSize.
    ConstantInt* destArraySize = dyn_cast<ConstantInt>(A->getArraySize());
    if (!destArraySize)
      return false;

    uint64_t destSize = TD.getABITypeSize(A->getAllocatedType()) *
      destArraySize->getZExtValue();

    if (destSize < srcSize)
      return false;
  } else if (Argument* A = dyn_cast<Argument>(cpyDest)) {
    // If the destination is an sret parameter then only accesses that are
    // outside of the returned struct type can trap.
    if (!A->hasStructRetAttr())
      return false;

    const Type* StructTy = cast<PointerType>(A->getType())->getElementType();
    uint64_t destSize = TD.getABITypeSize(StructTy);

    if (destSize < srcSize)
      return false;
  } else {
    return false;
  }

  // Check that src is not accessed except via the call and the memcpy.  This
  // guarantees that it holds only undefined values when passed in (so the final
  // memcpy can be dropped), that it is not read or written between the call and
  // the memcpy, and that writing beyond the end of it is undefined.
  SmallVector<User*, 8> srcUseList(srcAlloca->use_begin(),
                                   srcAlloca->use_end());
  while (!srcUseList.empty()) {
    User* UI = srcUseList.back();
    srcUseList.pop_back();

    if (isa<GetElementPtrInst>(UI) || isa<BitCastInst>(UI)) {
      for (User::use_iterator I = UI->use_begin(), E = UI->use_end();
           I != E; ++I)
        srcUseList.push_back(*I);
    } else if (UI != C && UI != cpy) {
      return false;
    }
  }

  // Since we're changing the parameter to the callsite, we need to make sure
  // that what would be the new parameter dominates the callsite.
  DominatorTree& DT = getAnalysis<DominatorTree>();
  if (Instruction* cpyDestInst = dyn_cast<Instruction>(cpyDest))
    if (!DT.dominates(cpyDestInst, C))
      return false;

  // In addition to knowing that the call does not access src in some
  // unexpected manner, for example via a global, which we deduce from
  // the use analysis, we also need to know that it does not sneakily
  // access dest.  We rely on AA to figure this out for us.
  AliasAnalysis& AA = getAnalysis<AliasAnalysis>();
  if (AA.getModRefInfo(C, cpy->getRawDest(), srcSize) !=
      AliasAnalysis::NoModRef)
    return false;

  // All the checks have passed, so do the transformation.
  for (unsigned i = 0; i < CS.arg_size(); ++i)
    if (CS.getArgument(i) == cpySrc) {
      if (cpySrc->getType() != cpyDest->getType())
        cpyDest = CastInst::createPointerCast(cpyDest, cpySrc->getType(),
                                              cpyDest->getName(), C);
      CS.setArgument(i, cpyDest);
    }

  // Drop any cached information about the call, because we may have changed
  // its dependence information by changing its parameter.
  MemoryDependenceAnalysis& MD = getAnalysis<MemoryDependenceAnalysis>();
  MD.dropInstruction(C);

  // Remove the memcpy
  MD.removeInstruction(cpy);
  toErase.push_back(cpy);

  return true;
}

/// processMemCpy - perform simplication of memcpy's.  If we have memcpy A which
/// copies X to Y, and memcpy B which copies Y to Z, then we can rewrite B to be
/// a memcpy from X to Z (or potentially a memmove, depending on circumstances).
///  This allows later passes to remove the first memcpy altogether.
bool GVN::processMemCpy(MemCpyInst* M, MemCpyInst* MDep,
                        SmallVectorImpl<Instruction*> &toErase) {
  // We can only transforms memcpy's where the dest of one is the source of the
  // other
  if (M->getSource() != MDep->getDest())
    return false;
  
  // Second, the length of the memcpy's must be the same, or the preceeding one
  // must be larger than the following one.
  ConstantInt* C1 = dyn_cast<ConstantInt>(MDep->getLength());
  ConstantInt* C2 = dyn_cast<ConstantInt>(M->getLength());
  if (!C1 || !C2)
    return false;
  
  uint64_t DepSize = C1->getValue().getZExtValue();
  uint64_t CpySize = C2->getValue().getZExtValue();
  
  if (DepSize < CpySize)
    return false;
  
  // Finally, we have to make sure that the dest of the second does not
  // alias the source of the first
  AliasAnalysis& AA = getAnalysis<AliasAnalysis>();
  if (AA.alias(M->getRawDest(), CpySize, MDep->getRawSource(), DepSize) !=
      AliasAnalysis::NoAlias)
    return false;
  else if (AA.alias(M->getRawDest(), CpySize, M->getRawSource(), CpySize) !=
           AliasAnalysis::NoAlias)
    return false;
  else if (AA.alias(MDep->getRawDest(), DepSize, MDep->getRawSource(), DepSize)
           != AliasAnalysis::NoAlias)
    return false;
  
  // If all checks passed, then we can transform these memcpy's
  Function* MemCpyFun = Intrinsic::getDeclaration(
                                 M->getParent()->getParent()->getParent(),
                                 M->getIntrinsicID());
    
  std::vector<Value*> args;
  args.push_back(M->getRawDest());
  args.push_back(MDep->getRawSource());
  args.push_back(M->getLength());
  args.push_back(M->getAlignment());
  
  CallInst* C = CallInst::Create(MemCpyFun, args.begin(), args.end(), "", M);
  
  MemoryDependenceAnalysis& MD = getAnalysis<MemoryDependenceAnalysis>();
  if (MD.getDependency(C) == MDep) {
    MD.dropInstruction(M);
    toErase.push_back(M);
    return true;
  }
  
  MD.removeInstruction(C);
  toErase.push_back(C);
  return false;
}

/// processInstruction - When calculating availability, handle an instruction
/// by inserting it into the appropriate sets
bool GVN::processInstruction(Instruction *I, ValueNumberedSet &currAvail,
                             DenseMap<Value*, LoadInst*> &lastSeenLoad,
                             SmallVectorImpl<Instruction*> &toErase) {
  if (LoadInst* L = dyn_cast<LoadInst>(I))
    return processLoad(L, lastSeenLoad, toErase);
  
  if (StoreInst *SI = dyn_cast<StoreInst>(I))
    return processStore(SI, toErase);
  
  // Allocations are always uniquely numbered, so we can save time and memory
  // by fast failing them.
  if (isa<AllocationInst>(I))
    return false;
  
  if (MemCpyInst* M = dyn_cast<MemCpyInst>(I)) {
    MemoryDependenceAnalysis& MD = getAnalysis<MemoryDependenceAnalysis>();

    // The are two possible optimizations we can do for memcpy:
    //   a) memcpy-memcpy xform which exposes redundance for DSE
    //   b) call-memcpy xform for return slot optimization
    Instruction* dep = MD.getDependency(M);
    if (dep == MemoryDependenceAnalysis::None ||
        dep == MemoryDependenceAnalysis::NonLocal)
      return false;
    if (MemCpyInst *MemCpy = dyn_cast<MemCpyInst>(dep))
      return processMemCpy(M, MemCpy, toErase);
    if (CallInst* C = dyn_cast<CallInst>(dep))
      return performCallSlotOptzn(M, C, toErase);
    return false;
  }
  
  unsigned num = VN.lookup_or_add(I);
  
  // Collapse PHI nodes
  if (PHINode* p = dyn_cast<PHINode>(I)) {
    Value* constVal = CollapsePhi(p);
    
    if (constVal) {
      for (PhiMapType::iterator PI = phiMap.begin(), PE = phiMap.end();
           PI != PE; ++PI)
        if (PI->second.count(p))
          PI->second.erase(p);
        
      p->replaceAllUsesWith(constVal);
      toErase.push_back(p);
    }
  // Perform value-number based elimination
  } else if (currAvail.test(num)) {
    Value* repl = find_leader(currAvail, num);
    
    if (CallInst* CI = dyn_cast<CallInst>(I)) {
      AliasAnalysis& AA = getAnalysis<AliasAnalysis>();
      if (!AA.doesNotAccessMemory(CI)) {
        MemoryDependenceAnalysis& MD = getAnalysis<MemoryDependenceAnalysis>();
        if (cast<Instruction>(repl)->getParent() != CI->getParent() ||
            MD.getDependency(CI) != MD.getDependency(cast<CallInst>(repl))) {
          // There must be an intervening may-alias store, so nothing from
          // this point on will be able to be replaced with the preceding call
          currAvail.erase(repl);
          currAvail.insert(I);
          
          return false;
        }
      }
    }
    
    // Remove it!
    MemoryDependenceAnalysis& MD = getAnalysis<MemoryDependenceAnalysis>();
    MD.removeInstruction(I);
    
    VN.erase(I);
    I->replaceAllUsesWith(repl);
    toErase.push_back(I);
    return true;
  } else if (!I->isTerminator()) {
    currAvail.set(num);
    currAvail.insert(I);
  }
  
  return false;
}

// GVN::runOnFunction - This is the main transformation entry point for a
// function.
//
bool GVN::runOnFunction(Function& F) {
  VN.setAliasAnalysis(&getAnalysis<AliasAnalysis>());
  
  bool changed = false;
  bool shouldContinue = true;
  
  while (shouldContinue) {
    shouldContinue = iterateOnFunction(F);
    changed |= shouldContinue;
  }
  
  return changed;
}


// GVN::iterateOnFunction - Executes one iteration of GVN
bool GVN::iterateOnFunction(Function &F) {
  // Clean out global sets from any previous functions
  VN.clear();
  availableOut.clear();
  phiMap.clear();
 
  bool changed_function = false;
  
  DominatorTree &DT = getAnalysis<DominatorTree>();   
  
  SmallVector<Instruction*, 8> toErase;
  DenseMap<Value*, LoadInst*> lastSeenLoad;
  DenseMap<DomTreeNode*, size_t> numChildrenVisited;

  // Top-down walk of the dominator tree
  for (df_iterator<DomTreeNode*> DI = df_begin(DT.getRootNode()),
         E = df_end(DT.getRootNode()); DI != E; ++DI) {
    
    // Get the set to update for this block
    ValueNumberedSet& currAvail = availableOut[DI->getBlock()];     
    lastSeenLoad.clear();

    BasicBlock* BB = DI->getBlock();
  
    // A block inherits AVAIL_OUT from its dominator
    if (DI->getIDom() != 0) {
      currAvail = availableOut[DI->getIDom()->getBlock()];
      
      numChildrenVisited[DI->getIDom()]++;
      
      if (numChildrenVisited[DI->getIDom()] == DI->getIDom()->getNumChildren()) {
        availableOut.erase(DI->getIDom()->getBlock());
        numChildrenVisited.erase(DI->getIDom());
      }
    }

    for (BasicBlock::iterator BI = BB->begin(), BE = BB->end();
         BI != BE;) {
      changed_function |= processInstruction(BI, currAvail,
                                             lastSeenLoad, toErase);
      if (toErase.empty()) {
        ++BI;
        continue;
      }
      
      // If we need some instructions deleted, do it now.
      NumGVNInstr += toErase.size();
      
      // Avoid iterator invalidation.
      bool AtStart = BI == BB->begin();
      if (!AtStart)
        --BI;

      for (SmallVector<Instruction*, 4>::iterator I = toErase.begin(),
           E = toErase.end(); I != E; ++I)
        (*I)->eraseFromParent();

      if (AtStart)
        BI = BB->begin();
      else
        ++BI;
      
      toErase.clear();
    }
  }
  
  return changed_function;
}
