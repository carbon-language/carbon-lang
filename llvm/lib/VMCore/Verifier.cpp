//===-- Verifier.cpp - Implement the Module Verifier -------------*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the function verifier interface, that can be used for some
// sanity checking of input to the system.
//
// Note that this does not provide full `Java style' security and verifications,
// instead it just tries to ensure that code is well-formed.
//
//  * Both of a binary operator's parameters are of the same type
//  * Verify that the indices of mem access instructions match other operands
//  * Verify that arithmetic and other things are only performed on first-class
//    types.  Verify that shifts & logicals only happen on integrals f.e.
//  * All of the constants in a switch statement are of the correct type
//  * The code is in valid SSA form
//  * It should be illegal to put a label into any other type (like a structure)
//    or to return one. [except constant arrays!]
//  * Only phi nodes can be self referential: 'add int %0, %0 ; <int>:0' is bad
//  * PHI nodes must have an entry for each predecessor, with no extras.
//  * PHI nodes must be the first thing in a basic block, all grouped together
//  * PHI nodes must have at least one entry
//  * All basic blocks should only end with terminator insts, not contain them
//  * The entry node to a function must not have predecessors
//  * All Instructions must be embedded into a basic block
//  * Functions cannot take a void-typed parameter
//  * Verify that a function's argument list agrees with it's declared type.
//  * It is illegal to specify a name for a void value.
//  * It is illegal to have a internal global value with no initializer
//  * It is illegal to have a ret instruction that returns a value that does not
//    agree with the function return value type.
//  * Function call argument types match the function prototype
//  * All other things that are tested by asserts spread about the code...
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/Verifier.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/CallingConv.h"
#include "llvm/Constants.h"
#include "llvm/Pass.h"
#include "llvm/Module.h"
#include "llvm/ModuleProvider.h"
#include "llvm/DerivedTypes.h"
#include "llvm/InlineAsm.h"
#include "llvm/Instructions.h"
#include "llvm/Intrinsics.h"
#include "llvm/PassManager.h"
#include "llvm/SymbolTable.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/InstVisitor.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Visibility.h"
#include <algorithm>
#include <iostream>
#include <sstream>
#include <cstdarg>
using namespace llvm;

namespace {  // Anonymous namespace for class

  struct VISIBILITY_HIDDEN
     Verifier : public FunctionPass, InstVisitor<Verifier> {
    bool Broken;          // Is this module found to be broken?
    bool RealPass;        // Are we not being run by a PassManager?
    VerifierFailureAction action;
                          // What to do if verification fails.
    Module *Mod;          // Module we are verifying right now
    ETForest *EF;     // ET-Forest, caution can be null!
    std::stringstream msgs;  // A stringstream to collect messages

    /// InstInThisBlock - when verifying a basic block, keep track of all of the
    /// instructions we have seen so far.  This allows us to do efficient
    /// dominance checks for the case when an instruction has an operand that is
    /// an instruction in the same block.
    std::set<Instruction*> InstsInThisBlock;

    Verifier()
        : Broken(false), RealPass(true), action(AbortProcessAction),
          EF(0), msgs( std::ios::app | std::ios::out ) {}
    Verifier( VerifierFailureAction ctn )
        : Broken(false), RealPass(true), action(ctn), EF(0),
          msgs( std::ios::app | std::ios::out ) {}
    Verifier(bool AB )
        : Broken(false), RealPass(true),
          action( AB ? AbortProcessAction : PrintMessageAction), EF(0),
          msgs( std::ios::app | std::ios::out ) {}
    Verifier(ETForest &ef)
      : Broken(false), RealPass(false), action(PrintMessageAction),
        EF(&ef), msgs( std::ios::app | std::ios::out ) {}


    bool doInitialization(Module &M) {
      Mod = &M;
      verifySymbolTable(M.getSymbolTable());

      // If this is a real pass, in a pass manager, we must abort before
      // returning back to the pass manager, or else the pass manager may try to
      // run other passes on the broken module.
      if (RealPass)
        abortIfBroken();
      return false;
    }

    bool runOnFunction(Function &F) {
      // Get dominator information if we are being run by PassManager
      if (RealPass) EF = &getAnalysis<ETForest>();
      visit(F);
      InstsInThisBlock.clear();

      // If this is a real pass, in a pass manager, we must abort before
      // returning back to the pass manager, or else the pass manager may try to
      // run other passes on the broken module.
      if (RealPass)
        abortIfBroken();

      return false;
    }

    bool doFinalization(Module &M) {
      // Scan through, checking all of the external function's linkage now...
      for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I) {
        visitGlobalValue(*I);

        // Check to make sure function prototypes are okay.
        if (I->isExternal()) visitFunction(*I);
      }

      for (Module::global_iterator I = M.global_begin(), E = M.global_end(); 
           I != E; ++I)
        visitGlobalVariable(*I);

      // If the module is broken, abort at this time.
      abortIfBroken();
      return false;
    }

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesAll();
      if (RealPass)
        AU.addRequired<ETForest>();
    }

    /// abortIfBroken - If the module is broken and we are supposed to abort on
    /// this condition, do so.
    ///
    void abortIfBroken() {
      if (Broken) {
        msgs << "Broken module found, ";
        switch (action) {
          case AbortProcessAction:
            msgs << "compilation aborted!\n";
            std::cerr << msgs.str();
            abort();
          case PrintMessageAction:
            msgs << "verification continues.\n";
            std::cerr << msgs.str();
            break;
          case ReturnStatusAction:
            break;
        }
      }
    }


    // Verification methods...
    void verifySymbolTable(SymbolTable &ST);
    void visitGlobalValue(GlobalValue &GV);
    void visitGlobalVariable(GlobalVariable &GV);
    void visitFunction(Function &F);
    void visitBasicBlock(BasicBlock &BB);
    void visitPHINode(PHINode &PN);
    void visitBinaryOperator(BinaryOperator &B);
    void visitShiftInst(ShiftInst &SI);
    void visitExtractElementInst(ExtractElementInst &EI);
    void visitInsertElementInst(InsertElementInst &EI);
    void visitShuffleVectorInst(ShuffleVectorInst &EI);
    void visitVAArgInst(VAArgInst &VAA) { visitInstruction(VAA); }
    void visitCallInst(CallInst &CI);
    void visitGetElementPtrInst(GetElementPtrInst &GEP);
    void visitLoadInst(LoadInst &LI);
    void visitStoreInst(StoreInst &SI);
    void visitInstruction(Instruction &I);
    void visitTerminatorInst(TerminatorInst &I);
    void visitReturnInst(ReturnInst &RI);
    void visitSwitchInst(SwitchInst &SI);
    void visitSelectInst(SelectInst &SI);
    void visitUserOp1(Instruction &I);
    void visitUserOp2(Instruction &I) { visitUserOp1(I); }
    void visitIntrinsicFunctionCall(Intrinsic::ID ID, CallInst &CI);

    void VerifyIntrinsicPrototype(Function *F, ...);

    void WriteValue(const Value *V) {
      if (!V) return;
      if (isa<Instruction>(V)) {
        msgs << *V;
      } else {
        WriteAsOperand (msgs, V, true, true, Mod);
        msgs << "\n";
      }
    }

    void WriteType(const Type* T ) {
      if ( !T ) return;
      WriteTypeSymbolic(msgs, T, Mod );
    }


    // CheckFailed - A check failed, so print out the condition and the message
    // that failed.  This provides a nice place to put a breakpoint if you want
    // to see why something is not correct.
    void CheckFailed(const std::string &Message,
                     const Value *V1 = 0, const Value *V2 = 0,
                     const Value *V3 = 0, const Value *V4 = 0) {
      msgs << Message << "\n";
      WriteValue(V1);
      WriteValue(V2);
      WriteValue(V3);
      WriteValue(V4);
      Broken = true;
    }

    void CheckFailed( const std::string& Message, const Value* V1,
                      const Type* T2, const Value* V3 = 0 ) {
      msgs << Message << "\n";
      WriteValue(V1);
      WriteType(T2);
      WriteValue(V3);
      Broken = true;
    }
  };

  RegisterOpt<Verifier> X("verify", "Module Verifier");
} // End anonymous namespace


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


void Verifier::visitGlobalValue(GlobalValue &GV) {
  Assert1(!GV.isExternal() || GV.hasExternalLinkage(),
          "Global is external, but doesn't have external linkage!", &GV);
  Assert1(!GV.hasAppendingLinkage() || isa<GlobalVariable>(GV),
          "Only global variables can have appending linkage!", &GV);

  if (GV.hasAppendingLinkage()) {
    GlobalVariable &GVar = cast<GlobalVariable>(GV);
    Assert1(isa<ArrayType>(GVar.getType()->getElementType()),
            "Only global arrays can have appending linkage!", &GV);
  }
}

void Verifier::visitGlobalVariable(GlobalVariable &GV) {
  if (GV.hasInitializer())
    Assert1(GV.getInitializer()->getType() == GV.getType()->getElementType(),
            "Global variable initializer type does not match global "
            "variable type!", &GV);

  visitGlobalValue(GV);
}


// verifySymbolTable - Verify that a function or module symbol table is ok
//
void Verifier::verifySymbolTable(SymbolTable &ST) {

  // Loop over all of the values in all type planes in the symbol table.
  for (SymbolTable::plane_const_iterator PI = ST.plane_begin(),
       PE = ST.plane_end(); PI != PE; ++PI)
    for (SymbolTable::value_const_iterator VI = PI->second.begin(),
         VE = PI->second.end(); VI != VE; ++VI) {
      Value *V = VI->second;
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
  // Check function arguments.
  const FunctionType *FT = F.getFunctionType();
  unsigned NumArgs = F.getArgumentList().size();

  Assert2(FT->getNumParams() == NumArgs,
          "# formal arguments must match # of arguments for function type!",
          &F, FT);
  Assert1(F.getReturnType()->isFirstClassType() ||
          F.getReturnType() == Type::VoidTy,
          "Functions cannot return aggregate values!", &F);

  // Check that this function meets the restrictions on this calling convention.
  switch (F.getCallingConv()) {
  default:
    break;
  case CallingConv::C:
    break;
  case CallingConv::CSRet:
    Assert1(FT->getReturnType() == Type::VoidTy && 
            FT->getNumParams() > 0 && isa<PointerType>(FT->getParamType(0)),
            "Invalid struct-return function!", &F);
    break;
  case CallingConv::Fast:
  case CallingConv::Cold:
    Assert1(!F.isVarArg(),
            "Varargs functions must have C calling conventions!", &F);
    break;
  }
  
  // Check that the argument values match the function type for this function...
  unsigned i = 0;
  for (Function::arg_iterator I = F.arg_begin(), E = F.arg_end(); I != E; ++I, ++i) {
    Assert2(I->getType() == FT->getParamType(i),
            "Argument value does not match function argument type!",
            I, FT->getParamType(i));
    // Make sure no aggregates are passed by value.
    Assert1(I->getType()->isFirstClassType(),
            "Functions cannot take aggregates as arguments by value!", I);
   }

  if (!F.isExternal()) {
    verifySymbolTable(F.getSymbolTable());

    // Check the entry node
    BasicBlock *Entry = &F.getEntryBlock();
    Assert1(pred_begin(Entry) == pred_end(Entry),
            "Entry block to function must not have predecessors!", Entry);
  }
}


// verifyBasicBlock - Verify that a basic block is well formed...
//
void Verifier::visitBasicBlock(BasicBlock &BB) {
  InstsInThisBlock.clear();

  // Ensure that basic blocks have terminators!
  Assert1(BB.getTerminator(), "Basic Block does not have terminator!", &BB);

  // Check constraints that this basic block imposes on all of the PHI nodes in
  // it.
  if (isa<PHINode>(BB.front())) {
    std::vector<BasicBlock*> Preds(pred_begin(&BB), pred_end(&BB));
    std::sort(Preds.begin(), Preds.end());
    PHINode *PN;
    for (BasicBlock::iterator I = BB.begin(); (PN = dyn_cast<PHINode>(I));++I) {

      // Ensure that PHI nodes have at least one entry!
      Assert1(PN->getNumIncomingValues() != 0,
              "PHI nodes must have at least one entry.  If the block is dead, "
              "the PHI should be removed!", PN);
      Assert1(PN->getNumIncomingValues() == Preds.size(),
              "PHINode should have one entry for each predecessor of its "
              "parent basic block!", PN);

      // Get and sort all incoming values in the PHI node...
      std::vector<std::pair<BasicBlock*, Value*> > Values;
      Values.reserve(PN->getNumIncomingValues());
      for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i)
        Values.push_back(std::make_pair(PN->getIncomingBlock(i),
                                        PN->getIncomingValue(i)));
      std::sort(Values.begin(), Values.end());

      for (unsigned i = 0, e = Values.size(); i != e; ++i) {
        // Check to make sure that if there is more than one entry for a
        // particular basic block in this PHI node, that the incoming values are
        // all identical.
        //
        Assert4(i == 0 || Values[i].first  != Values[i-1].first ||
                Values[i].second == Values[i-1].second,
                "PHI node has multiple entries for the same basic block with "
                "different incoming values!", PN, Values[i].first,
                Values[i].second, Values[i-1].second);

        // Check to make sure that the predecessors and PHI node entries are
        // matched up.
        Assert3(Values[i].first == Preds[i],
                "PHI node entries do not match predecessors!", PN,
                Values[i].first, Preds[i]);
      }
    }
  }
}

void Verifier::visitTerminatorInst(TerminatorInst &I) {
  // Ensure that terminators only exist at the end of the basic block.
  Assert1(&I == I.getParent()->getTerminator(),
          "Terminator found in the middle of a basic block!", I.getParent());
  visitInstruction(I);
}

void Verifier::visitReturnInst(ReturnInst &RI) {
  Function *F = RI.getParent()->getParent();
  if (RI.getNumOperands() == 0)
    Assert2(F->getReturnType() == Type::VoidTy,
            "Found return instr that returns void in Function of non-void "
            "return type!", &RI, F->getReturnType());
  else
    Assert2(F->getReturnType() == RI.getOperand(0)->getType(),
            "Function return type does not match operand "
            "type of return inst!", &RI, F->getReturnType());

  // Check to make sure that the return value has necessary properties for
  // terminators...
  visitTerminatorInst(RI);
}

void Verifier::visitSwitchInst(SwitchInst &SI) {
  // Check to make sure that all of the constants in the switch instruction
  // have the same type as the switched-on value.
  const Type *SwitchTy = SI.getCondition()->getType();
  for (unsigned i = 1, e = SI.getNumCases(); i != e; ++i)
    Assert1(SI.getCaseValue(i)->getType() == SwitchTy,
            "Switch constants must all be same type as switch value!", &SI);

  visitTerminatorInst(SI);
}

void Verifier::visitSelectInst(SelectInst &SI) {
  Assert1(SI.getCondition()->getType() == Type::BoolTy,
          "Select condition type must be bool!", &SI);
  Assert1(SI.getTrueValue()->getType() == SI.getFalseValue()->getType(),
          "Select values must have identical types!", &SI);
  Assert1(SI.getTrueValue()->getType() == SI.getType(),
          "Select values must have same type as select instruction!", &SI);
  visitInstruction(SI);
}


/// visitUserOp1 - User defined operators shouldn't live beyond the lifetime of
/// a pass, if any exist, it's an error.
///
void Verifier::visitUserOp1(Instruction &I) {
  Assert1(0, "User-defined operators should not live outside of a pass!", &I);
}

/// visitPHINode - Ensure that a PHI node is well formed.
///
void Verifier::visitPHINode(PHINode &PN) {
  // Ensure that the PHI nodes are all grouped together at the top of the block.
  // This can be tested by checking whether the instruction before this is
  // either nonexistent (because this is begin()) or is a PHI node.  If not,
  // then there is some other instruction before a PHI.
  Assert2(&PN.getParent()->front() == &PN || isa<PHINode>(PN.getPrev()),
          "PHI nodes not grouped at top of basic block!",
          &PN, PN.getParent());

  // Check that all of the operands of the PHI node have the same type as the
  // result.
  for (unsigned i = 0, e = PN.getNumIncomingValues(); i != e; ++i)
    Assert1(PN.getType() == PN.getIncomingValue(i)->getType(),
            "PHI node operands are not the same type as the result!", &PN);

  // All other PHI node constraints are checked in the visitBasicBlock method.

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
    Assert3(CI.getOperand(i+1)->getType() == FTy->getParamType(i),
            "Call parameter type does not match function signature!",
            CI.getOperand(i+1), FTy->getParamType(i), &CI);

  if (Function *F = CI.getCalledFunction())
    if (Intrinsic::ID ID = (Intrinsic::ID)F->getIntrinsicID())
      visitIntrinsicFunctionCall(ID, CI);

  visitInstruction(CI);
}

/// visitBinaryOperator - Check that both arguments to the binary operator are
/// of the same type!
///
void Verifier::visitBinaryOperator(BinaryOperator &B) {
  Assert1(B.getOperand(0)->getType() == B.getOperand(1)->getType(),
          "Both operands to a binary operator are not of the same type!", &B);

  // Check that logical operators are only used with integral operands.
  if (B.getOpcode() == Instruction::And || B.getOpcode() == Instruction::Or ||
      B.getOpcode() == Instruction::Xor) {
    Assert1(B.getType()->isIntegral() ||
            (isa<PackedType>(B.getType()) && 
             cast<PackedType>(B.getType())->getElementType()->isIntegral()),
            "Logical operators only work with integral types!", &B);
    Assert1(B.getType() == B.getOperand(0)->getType(),
            "Logical operators must have same type for operands and result!",
            &B);
  } else if (isa<SetCondInst>(B)) {
    // Check that setcc instructions return bool
    Assert1(B.getType() == Type::BoolTy,
            "setcc instructions must return boolean values!", &B);
  } else {
    // Arithmetic operators only work on integer or fp values
    Assert1(B.getType() == B.getOperand(0)->getType(),
            "Arithmetic operators must have same type for operands and result!",
            &B);
    Assert1(B.getType()->isInteger() || B.getType()->isFloatingPoint() ||
            isa<PackedType>(B.getType()),
            "Arithmetic operators must have integer, fp, or packed type!", &B);
  }

  visitInstruction(B);
}

void Verifier::visitShiftInst(ShiftInst &SI) {
  Assert1(SI.getType()->isInteger(),
          "Shift must return an integer result!", &SI);
  Assert1(SI.getType() == SI.getOperand(0)->getType(),
          "Shift return type must be same as first operand!", &SI);
  Assert1(SI.getOperand(1)->getType() == Type::UByteTy,
          "Second operand to shift must be ubyte type!", &SI);
  visitInstruction(SI);
}

void Verifier::visitExtractElementInst(ExtractElementInst &EI) {
  Assert1(ExtractElementInst::isValidOperands(EI.getOperand(0),
                                              EI.getOperand(1)),
          "Invalid extractelement operands!", &EI);
  visitInstruction(EI);
}

void Verifier::visitInsertElementInst(InsertElementInst &IE) {
  Assert1(InsertElementInst::isValidOperands(IE.getOperand(0),
                                             IE.getOperand(1),
                                             IE.getOperand(2)),
          "Invalid insertelement operands!", &IE);
  visitInstruction(IE);
}

void Verifier::visitShuffleVectorInst(ShuffleVectorInst &SV) {
  Assert1(ShuffleVectorInst::isValidOperands(SV.getOperand(0), SV.getOperand(1),
                                             SV.getOperand(2)),
          "Invalid shufflevector operands!", &SV);
  Assert1(SV.getType() == SV.getOperand(0)->getType(),
          "Result of shufflevector must match first operand type!", &SV);
  
  // Check to see if Mask is valid.
  if (const ConstantPacked *MV = dyn_cast<ConstantPacked>(SV.getOperand(2))) {
    for (unsigned i = 0, e = MV->getNumOperands(); i != e; ++i) {
      Assert1(isa<ConstantUInt>(MV->getOperand(i)) ||
              isa<UndefValue>(MV->getOperand(i)),
              "Invalid shufflevector shuffle mask!", &SV);
    }
  } else {
    Assert1(isa<UndefValue>(SV.getOperand(2)) || 
            isa<ConstantAggregateZero>(SV.getOperand(2)),
            "Invalid shufflevector shuffle mask!", &SV);
  }
  
  visitInstruction(SV);
}

void Verifier::visitGetElementPtrInst(GetElementPtrInst &GEP) {
  const Type *ElTy =
    GetElementPtrInst::getIndexedType(GEP.getOperand(0)->getType(),
                   std::vector<Value*>(GEP.idx_begin(), GEP.idx_end()), true);
  Assert1(ElTy, "Invalid indices for GEP pointer type!", &GEP);
  Assert2(PointerType::get(ElTy) == GEP.getType(),
          "GEP is not of right type for indices!", &GEP, ElTy);
  visitInstruction(GEP);
}

void Verifier::visitLoadInst(LoadInst &LI) {
  const Type *ElTy =
    cast<PointerType>(LI.getOperand(0)->getType())->getElementType();
  Assert2(ElTy == LI.getType(),
          "Load result type does not match pointer operand type!", &LI, ElTy);
  visitInstruction(LI);
}

void Verifier::visitStoreInst(StoreInst &SI) {
  const Type *ElTy =
    cast<PointerType>(SI.getOperand(1)->getType())->getElementType();
  Assert2(ElTy == SI.getOperand(0)->getType(),
          "Stored value type does not match pointer operand type!", &SI, ElTy);
  visitInstruction(SI);
}


/// verifyInstruction - Verify that an instruction is well formed.
///
void Verifier::visitInstruction(Instruction &I) {
  BasicBlock *BB = I.getParent();
  Assert1(BB, "Instruction not embedded in basic block!", &I);

  if (!isa<PHINode>(I)) {   // Check that non-phi nodes are not self referential
    for (Value::use_iterator UI = I.use_begin(), UE = I.use_end();
         UI != UE; ++UI)
      Assert1(*UI != (User*)&I ||
              !EF->dominates(&BB->getParent()->getEntryBlock(), BB),
              "Only PHI nodes may reference their own value!", &I);
  }

  // Check that void typed values don't have names
  Assert1(I.getType() != Type::VoidTy || !I.hasName(),
          "Instruction has a name, but provides a void value!", &I);

  // Check that the return value of the instruction is either void or a legal
  // value type.
  Assert1(I.getType() == Type::VoidTy || I.getType()->isFirstClassType(),
          "Instruction returns a non-scalar type!", &I);

  // Check that all uses of the instruction, if they are instructions
  // themselves, actually have parent basic blocks.  If the use is not an
  // instruction, it is an error!
  for (User::use_iterator UI = I.use_begin(), UE = I.use_end();
       UI != UE; ++UI) {
    Assert1(isa<Instruction>(*UI), "Use of instruction is not an instruction!",
            *UI);
    Instruction *Used = cast<Instruction>(*UI);
    Assert2(Used->getParent() != 0, "Instruction referencing instruction not"
            " embeded in a basic block!", &I, Used);
  }

  for (unsigned i = 0, e = I.getNumOperands(); i != e; ++i) {
    Assert1(I.getOperand(i) != 0, "Instruction has null operand!", &I);

    // Check to make sure that only first-class-values are operands to
    // instructions.
    Assert1(I.getOperand(i)->getType()->isFirstClassType(),
            "Instruction operands must be first-class values!", &I);
  
    if (Function *F = dyn_cast<Function>(I.getOperand(i))) {
      // Check to make sure that the "address of" an intrinsic function is never
      // taken.
      Assert1(!F->isIntrinsic() || (i == 0 && isa<CallInst>(I)),
              "Cannot take the address of an intrinsic!", &I);
    } else if (BasicBlock *OpBB = dyn_cast<BasicBlock>(I.getOperand(i))) {
      Assert1(OpBB->getParent() == BB->getParent(),
              "Referring to a basic block in another function!", &I);
    } else if (Argument *OpArg = dyn_cast<Argument>(I.getOperand(i))) {
      Assert1(OpArg->getParent() == BB->getParent(),
              "Referring to an argument in another function!", &I);
    } else if (Instruction *Op = dyn_cast<Instruction>(I.getOperand(i))) {
      BasicBlock *OpBlock = Op->getParent();

      // Check that a definition dominates all of its uses.
      if (!isa<PHINode>(I)) {
        // Invoke results are only usable in the normal destination, not in the
        // exceptional destination.
        if (InvokeInst *II = dyn_cast<InvokeInst>(Op))
          OpBlock = II->getNormalDest();
        else if (OpBlock == BB) {
          // If they are in the same basic block, make sure that the definition
          // comes before the use.
          Assert2(InstsInThisBlock.count(Op) ||
                  !EF->dominates(&BB->getParent()->getEntryBlock(), BB),
                  "Instruction does not dominate all uses!", Op, &I);
        }

        // Definition must dominate use unless use is unreachable!
        Assert2(EF->dominates(OpBlock, BB) ||
                !EF->dominates(&BB->getParent()->getEntryBlock(), BB),
                "Instruction does not dominate all uses!", Op, &I);
      } else {
        // PHI nodes are more difficult than other nodes because they actually
        // "use" the value in the predecessor basic blocks they correspond to.
        BasicBlock *PredBB = cast<BasicBlock>(I.getOperand(i+1));
        Assert2(EF->dominates(OpBlock, PredBB) ||
                !EF->dominates(&BB->getParent()->getEntryBlock(), PredBB),
                "Instruction does not dominate all uses!", Op, &I);
      }
    } else if (isa<InlineAsm>(I.getOperand(i))) {
      Assert1(i == 0 && isa<CallInst>(I),
              "Cannot take the address of an inline asm!", &I);
    }
  }
  InstsInThisBlock.insert(&I);
}

/// visitIntrinsicFunction - Allow intrinsics to be verified in different ways.
///
void Verifier::visitIntrinsicFunctionCall(Intrinsic::ID ID, CallInst &CI) {
  Function *IF = CI.getCalledFunction();
  Assert1(IF->isExternal(), "Intrinsic functions should never be defined!", IF);
  
#define GET_INTRINSIC_VERIFIER
#include "llvm/Intrinsics.gen"
#undef GET_INTRINSIC_VERIFIER
}

/// VerifyIntrinsicPrototype - TableGen emits calls to this function into
/// Intrinsics.gen.  This implements a little state machine that verifies the
/// prototype of intrinsics.
void Verifier::VerifyIntrinsicPrototype(Function *F, ...) {
  va_list VA;
  va_start(VA, F);
  
  const FunctionType *FTy = F->getFunctionType();
  
  // Note that "arg#0" is the return type.
  for (unsigned ArgNo = 0; 1; ++ArgNo) {
    int TypeID = va_arg(VA, int);

    if (TypeID == -1) {
      if (ArgNo != FTy->getNumParams()+1)
        CheckFailed("Intrinsic prototype has too many arguments!", F);
      break;
    }

    if (ArgNo == FTy->getNumParams()+1) {
      CheckFailed("Intrinsic prototype has too few arguments!", F);
      break;
    }
    
    const Type *Ty;
    if (ArgNo == 0) 
      Ty = FTy->getReturnType();
    else
      Ty = FTy->getParamType(ArgNo-1);
    
    if (Ty->getTypeID() != TypeID) {
      if (ArgNo == 0)
        CheckFailed("Intrinsic prototype has incorrect result type!", F);
      else
        CheckFailed("Intrinsic parameter #" + utostr(ArgNo-1) + " is wrong!",F);
      break;
    }

    // If this is a packed argument, verify the number and type of elements.
    if (TypeID == Type::PackedTyID) {
      const PackedType *PTy = cast<PackedType>(Ty);
      if (va_arg(VA, int) != PTy->getElementType()->getTypeID()) {
        CheckFailed("Intrinsic prototype has incorrect vector element type!",F);
        break;
      }

      if ((unsigned)va_arg(VA, int) != PTy->getNumElements()) {
        CheckFailed("Intrinsic prototype has incorrect number of "
                    "vector elements!",F);
        break;
      }
    }
  }

  va_end(VA);
}


//===----------------------------------------------------------------------===//
//  Implement the public interfaces to this file...
//===----------------------------------------------------------------------===//

FunctionPass *llvm::createVerifierPass(VerifierFailureAction action) {
  return new Verifier(action);
}


// verifyFunction - Create
bool llvm::verifyFunction(const Function &f, VerifierFailureAction action) {
  Function &F = const_cast<Function&>(f);
  assert(!F.isExternal() && "Cannot verify external functions");

  FunctionPassManager FPM(new ExistingModuleProvider(F.getParent()));
  Verifier *V = new Verifier(action);
  FPM.add(V);
  FPM.run(F);
  return V->Broken;
}

/// verifyModule - Check a module for errors, printing messages on stderr.
/// Return true if the module is corrupt.
///
bool llvm::verifyModule(const Module &M, VerifierFailureAction action,
                        std::string *ErrorInfo) {
  PassManager PM;
  Verifier *V = new Verifier(action);
  PM.add(V);
  PM.run((Module&)M);
  
  if (ErrorInfo && V->Broken)
    *ErrorInfo = V->msgs.str();
  return V->Broken;
}

// vim: sw=2
