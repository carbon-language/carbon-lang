//===-- Verifier.cpp - Implement the Module Verifier -------------*- C++ -*-==//
//
// This file defines the function verifier interface, that can be used for some
// sanity checking of input to the system.
//
// Note that this does not provide full 'java style' security and verifications,
// instead it just tries to ensure that code is well formed.
//
//  . There are no duplicated names in a symbol table... ie there !exist a val
//    with the same name as something in the symbol table, but with a different
//    address as what is in the symbol table...
//  * Both of a binary operator's parameters are the same type
//  * Verify that the indices of mem access instructions match other operands
//  . Verify that arithmetic and other things are only performed on first class
//    types.  No adding structures or arrays.
//  . All of the constants in a switch statement are of the correct type
//  . The code is in valid SSA form
//  . It should be illegal to put a label into any other type (like a structure)
//    or to return one. [except constant arrays!]
//  * Only phi nodes can be self referential: 'add int %0, %0 ; <int>:0' is bad
//  * PHI nodes must have an entry for each predecessor, with no extras.
//  * PHI nodes must be the first thing in a basic block, all grouped together
//  * All basic blocks should only end with terminator insts, not contain them
//  * The entry node to a function must not have predecessors
//  * All Instructions must be embeded into a basic block
//  . Verify that none of the Value getType()'s are null.
//  . Function's cannot take a void typed parameter
//  * Verify that a function's argument list agrees with it's declared type.
//  . Verify that arrays and structures have fixed elements: No unsized arrays.
//  * It is illegal to specify a name for a void value.
//  * It is illegal to have a internal function that is just a declaration
//  * It is illegal to have a ret instruction that returns a value that does not
//    agree with the function return value type.
//  * Function call argument types match the function prototype
//  * All other things that are tested by asserts spread about the code...
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/Verifier.h"
#include "llvm/Pass.h"
#include "llvm/Module.h"
#include "llvm/DerivedTypes.h"
#include "llvm/iPHINode.h"
#include "llvm/iTerminators.h"
#include "llvm/iOther.h"
#include "llvm/iMemory.h"
#include "llvm/SymbolTable.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/InstVisitor.h"
#include "Support/STLExtras.h"
#include <algorithm>
#include <iostream>

namespace {  // Anonymous namespace for class

  struct Verifier : public FunctionPass, InstVisitor<Verifier> {
    bool Broken;

    Verifier() : Broken(false) {}

    virtual const char *getPassName() const { return "Module Verifier"; }

    bool doInitialization(Module &M) {
      verifySymbolTable(M.getSymbolTable());
      return false;
    }

    bool runOnFunction(Function &F) {
      visit(F);
      return false;
    }

    bool doFinalization(Module &M) {
      // Scan through, checking all of the external function's linkage now...
      for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I)
        if (I->isExternal() && I->hasInternalLinkage())
          CheckFailed("Function Declaration has Internal Linkage!", I);

      if (Broken) {
        std::cerr << "Broken module found, compilation aborted!\n";
        abort();
      }
      return false;
    }

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesAll();
    }

    // Verification methods...
    void verifySymbolTable(SymbolTable *ST);
    void visitFunction(Function &F);
    void visitBasicBlock(BasicBlock &BB);
    void visitPHINode(PHINode &PN);
    void visitBinaryOperator(BinaryOperator &B);
    void visitCallInst(CallInst &CI);
    void visitGetElementPtrInst(GetElementPtrInst &GEP);
    void visitLoadInst(LoadInst &LI);
    void visitStoreInst(StoreInst &SI);
    void visitInstruction(Instruction &I);
    void visitTerminatorInst(TerminatorInst &I);
    void visitReturnInst(ReturnInst &RI);

    // CheckFailed - A check failed, so print out the condition and the message
    // that failed.  This provides a nice place to put a breakpoint if you want
    // to see why something is not correct.
    //
    inline void CheckFailed(const std::string &Message,
                            const Value *V1 = 0, const Value *V2 = 0,
                            const Value *V3 = 0, const Value *V4 = 0) {
      std::cerr << Message << "\n";
      if (V1) std::cerr << *V1 << "\n";
      if (V2) std::cerr << *V2 << "\n";
      if (V3) std::cerr << *V3 << "\n";
      if (V4) std::cerr << *V4 << "\n";
      Broken = true;
    }
  };
}

// Assert - We know that cond should be true, if not print an error message.
#define Assert(C, M) \
  do { if (!(C)) { CheckFailed(M); return; } } while (0)
#define Assert1(C, M, V1) \
  do { if (!(C)) { CheckFailed(M, V1); return; } } while (0)
#define Assert2(C, M, V1, V2) \
  do { if (!(C)) { CheckFailed(M, V1, V2); return; } } while (0)
#define Assert3(C, M, V1, V2, V3) \
  do { if (!(C)) { CheckFailed(M, V1, V2, V3); return; } } while (0)
#define Assert4(C, M, V1, V2, V3, V4) \
  do { if (!(C)) { CheckFailed(M, V1, V2, V3, V4); return; } } while (0)


// verifySymbolTable - Verify that a function or module symbol table is ok
//
void Verifier::verifySymbolTable(SymbolTable *ST) {
  if (ST == 0) return;   // No symbol table to process

  // Loop over all of the types in the symbol table...
  for (SymbolTable::iterator TI = ST->begin(), TE = ST->end(); TI != TE; ++TI)
    for (SymbolTable::type_iterator I = TI->second.begin(),
           E = TI->second.end(); I != E; ++I) {
      Value *V = I->second;

      // Check that there are no void typed values in the symbol table.  Values
      // with a void type cannot be put into symbol tables because they cannot
      // have names!
      Assert1(V->getType() != Type::VoidTy,
              "Values with void type are not allowed to have names!", V);
    }
}


// visitFunction - Verify that a function is ok.
//
void Verifier::visitFunction(Function &F) {
  if (F.isExternal()) return;

  verifySymbolTable(F.getSymbolTable());

  // Check function arguments...
  const FunctionType *FT = F.getFunctionType();
  unsigned NumArgs = F.getArgumentList().size();

  Assert2(!FT->isVarArg(), "Cannot define varargs functions in LLVM!", &F, FT);
  Assert2(FT->getParamTypes().size() == NumArgs,
          "# formal arguments must match # of arguments for function type!",
          &F, FT);

  // Check that the argument values match the function type for this function...
  if (FT->getParamTypes().size() == NumArgs) {
    unsigned i = 0;
    for (Function::aiterator I = F.abegin(), E = F.aend(); I != E; ++I, ++i)
      Assert2(I->getType() == FT->getParamType(i),
              "Argument value does not match function argument type!",
              I, FT->getParamType(i));
  }

  // Check the entry node
  BasicBlock *Entry = &F.getEntryNode();
  Assert1(pred_begin(Entry) == pred_end(Entry),
          "Entry block to function must not have predecessors!", Entry);
}


// verifyBasicBlock - Verify that a basic block is well formed...
//
void Verifier::visitBasicBlock(BasicBlock &BB) {
  // Ensure that basic blocks have terminators!
  Assert1(BB.getTerminator(), "Basic Block does not have terminator!", &BB);
}

void Verifier::visitTerminatorInst(TerminatorInst &I) {
  // Ensure that terminators only exist at the end of the basic block.
  Assert1(&I == I.getParent()->getTerminator(),
          "Terminator found in the middle of a basic block!", I.getParent());
}

void Verifier::visitReturnInst(ReturnInst &RI) {
  Function *F = RI.getParent()->getParent();
  if (RI.getNumOperands() == 0)
    Assert1(F->getReturnType() == Type::VoidTy,
            "Function returns no value, but ret instruction found that does!",
            &RI);
  else
    Assert2(F->getReturnType() == RI.getOperand(0)->getType(),
            "Function return type does not match operand "
            "type of return inst!", &RI, F->getReturnType());

  // Check to make sure that the return value has neccesary properties for
  // terminators...
  visitTerminatorInst(RI);
}


// visitPHINode - Ensure that a PHI node is well formed.
void Verifier::visitPHINode(PHINode &PN) {
  // Ensure that the PHI nodes are all grouped together at the top of the block.
  // This can be tested by checking whether the instruction before this is
  // either nonexistant (because this is begin()) or is a PHI node.  If not,
  // then there is some other instruction before a PHI.
  Assert2(PN.getPrev() == 0 || isa<PHINode>(PN.getPrev()),
          "PHI nodes not grouped at top of basic block!",
          &PN, PN.getParent());

  std::vector<BasicBlock*> Preds(pred_begin(PN.getParent()),
                                 pred_end(PN.getParent()));
  // Loop over all of the incoming values, make sure that there are
  // predecessors for each one...
  //
  for (unsigned i = 0, e = PN.getNumIncomingValues(); i != e; ++i) {
    // Make sure all of the incoming values are the right types...
    Assert2(PN.getType() == PN.getIncomingValue(i)->getType(),
            "PHI node argument type does not agree with PHI node type!",
            &PN, PN.getIncomingValue(i));

    BasicBlock *BB = PN.getIncomingBlock(i);
    std::vector<BasicBlock*>::iterator PI =
      find(Preds.begin(), Preds.end(), BB);
    Assert2(PI != Preds.end(), "PHI node has entry for basic block that"
            " is not a predecessor!", &PN, BB);
    Preds.erase(PI);
  }
  
  // There should be no entries left in the predecessor list...
  for (std::vector<BasicBlock*>::iterator I = Preds.begin(),
         E = Preds.end(); I != E; ++I)
    Assert2(0, "PHI node does not have entry for a predecessor basic block!",
            &PN, *I);

  // Now we go through and check to make sure that if there is more than one
  // entry for a particular basic block in this PHI node, that the incoming
  // values are all identical.
  //
  std::vector<std::pair<BasicBlock*, Value*> > Values;
  Values.reserve(PN.getNumIncomingValues());
  for (unsigned i = 0, e = PN.getNumIncomingValues(); i != e; ++i)
    Values.push_back(std::make_pair(PN.getIncomingBlock(i),
                                    PN.getIncomingValue(i)));

  // Sort the Values vector so that identical basic block entries are adjacent.
  std::sort(Values.begin(), Values.end());

  // Check for identical basic blocks with differing incoming values...
  for (unsigned i = 1, e = PN.getNumIncomingValues(); i < e; ++i)
    Assert4(Values[i].first  != Values[i-1].first ||
            Values[i].second == Values[i-1].second,
            "PHI node has multiple entries for the same basic block with "
            "different incoming values!", &PN, Values[i].first,
            Values[i].second, Values[i-1].second);

  visitInstruction(PN);
}

void Verifier::visitCallInst(CallInst &CI) {
  Assert1(isa<PointerType>(CI.getOperand(0)->getType()),
          "Called function must be a pointer!", &CI);
  const PointerType *FPTy = cast<PointerType>(CI.getOperand(0)->getType());
  Assert1(isa<FunctionType>(FPTy->getElementType()),
          "Called function is not pointer to function type!", &CI);

  const FunctionType *FTy = cast<FunctionType>(FPTy->getElementType());

  // Verify that the correct number of arguments are being passed
  if (FTy->isVarArg())
    Assert1(CI.getNumOperands()-1 >= FTy->getNumParams(),
            "Called function requires more parameters than were provided!",&CI);
  else
    Assert1(CI.getNumOperands()-1 == FTy->getNumParams(),
            "Incorrect number of arguments passed to called function!", &CI);

  // Verify that all arguments to the call match the function type...
  for (unsigned i = 0, e = FTy->getNumParams(); i != e; ++i)
    Assert2(CI.getOperand(i+1)->getType() == FTy->getParamType(i),
            "Call parameter type does not match function signature!",
            CI.getOperand(i+1), FTy->getParamType(i));
}

// visitBinaryOperator - Check that both arguments to the binary operator are
// of the same type!
//
void Verifier::visitBinaryOperator(BinaryOperator &B) {
  Assert2(B.getOperand(0)->getType() == B.getOperand(1)->getType(),
          "Both operands to a binary operator are not of the same type!",
          B.getOperand(0), B.getOperand(1));

  visitInstruction(B);
}

void Verifier::visitGetElementPtrInst(GetElementPtrInst &GEP) {
  const Type *ElTy = MemAccessInst::getIndexedType(GEP.getOperand(0)->getType(),
                                                   GEP.copyIndices(), true);
  Assert1(ElTy, "Invalid indices for GEP pointer type!", &GEP);
  Assert2(PointerType::get(ElTy) == GEP.getType(),
          "GEP is not of right type for indices!", &GEP, ElTy);
  visitInstruction(GEP);
}

void Verifier::visitLoadInst(LoadInst &LI) {
  const Type *ElTy = LoadInst::getIndexedType(LI.getOperand(0)->getType(),
                                              LI.copyIndices());
  Assert1(ElTy, "Invalid indices for load pointer type!", &LI);
  Assert2(ElTy == LI.getType(),
          "Load is not of right type for indices!", &LI, ElTy);
  visitInstruction(LI);
}

void Verifier::visitStoreInst(StoreInst &SI) {
  const Type *ElTy = StoreInst::getIndexedType(SI.getOperand(1)->getType(),
                                               SI.copyIndices());
  Assert1(ElTy, "Invalid indices for store pointer type!", &SI);
  Assert2(ElTy == SI.getOperand(0)->getType(),
          "Stored value is not of right type for indices!", &SI, ElTy);
  visitInstruction(SI);
}


// verifyInstruction - Verify that a non-terminator instruction is well formed.
//
void Verifier::visitInstruction(Instruction &I) {
  Assert1(I.getParent(), "Instruction not embedded in basic block!", &I);

  // Check that all uses of the instruction, if they are instructions
  // themselves, actually have parent basic blocks.  If the use is not an
  // instruction, it is an error!
  //
  for (User::use_iterator UI = I.use_begin(), UE = I.use_end();
       UI != UE; ++UI) {
    Assert1(isa<Instruction>(*UI), "Use of instruction is not an instruction!",
            *UI);
    Instruction *Used = cast<Instruction>(*UI);
    Assert2(Used->getParent() != 0, "Instruction referencing instruction not"
            " embeded in a basic block!", &I, Used);
  }

  if (!isa<PHINode>(I)) {   // Check that non-phi nodes are not self referential
    for (Value::use_iterator UI = I.use_begin(), UE = I.use_end();
         UI != UE; ++UI)
      Assert1(*UI != (User*)&I,
              "Only PHI nodes may reference their own value!", &I);
  }

  Assert1(I.getType() != Type::VoidTy || !I.hasName(),
          "Instruction has a name, but provides a void value!", &I);
}


//===----------------------------------------------------------------------===//
//  Implement the public interfaces to this file...
//===----------------------------------------------------------------------===//

Pass *createVerifierPass() {
  return new Verifier();
}

bool verifyFunction(const Function &F) {
  Verifier V;
  V.visit((Function&)F);
  return V.Broken;
}

// verifyModule - Check a module for errors, printing messages on stderr.
// Return true if the module is corrupt.
//
bool verifyModule(const Module &M) {
  Verifier V;
  V.run((Module&)M);
  return V.Broken;
}
