//===-- Verifier.cpp - Implement the Module Verifier -------------*- C++ -*-==//
//
// This file defines the method verifier interface, that can be used for some
// sanity checking of input to the system.
//
// Note that this does not provide full 'java style' security and verifications,
// instead it just tries to ensure that code is well formed.
//
//  . There are no duplicated names in a symbol table... ie there !exist a val
//    with the same name as something in the symbol table, but with a different
//    address as what is in the symbol table...
//  . Both of a binary operator's parameters are the same type
//  . Verify that arithmetic and other things are only performed on first class
//    types.  No adding structures or arrays.
//  . All of the constants in a switch statement are of the correct type
//  . The code is in valid SSA form
//  . It should be illegal to put a label into any other type (like a structure)
//    or to return one. [except constant arrays!]
//  . Right now 'add bool 0, 0' is valid.  This isn't particularly good.
//  * Only phi nodes can be self referential: 'add int %0, %0 ; <int>:0' is bad
//  * PHI nodes must have an entry for each predecessor, with no extras.
//  * All basic blocks should only end with terminator insts, not contain them
//  * The entry node to a method must not have predecessors
//  * All Instructions must be embeded into a basic block
//  . Verify that none of the Value getType()'s are null.
//  . Method's cannot take a void typed parameter
//  . Verify that a method's argument list agrees with it's declared type.
//  . Verify that arrays and structures have fixed elements: No unsized arrays.
//  . All other things that are tested by asserts spread about the code...
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/Verifier.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/Pass.h"
#include "llvm/Method.h"
#include "llvm/Module.h"
#include "llvm/BasicBlock.h"
#include "llvm/Type.h"
#include "llvm/iPHINode.h"
#include "llvm/Support/CFG.h"
#include "Support/STLExtras.h"
#include <algorithm>

#if 0
#define t(x) (1 << (unsigned)Type::x)
#define SignedIntegralTypes (t(SByteTyID) | t(ShortTyID) |  \
                             t(IntTyID)   | t(LongTyID))
static long UnsignedIntegralTypes = t(UByteTyID) | t(UShortTyID) | 
                                          t(UIntTyID)  | t(ULongTyID);
static const long FloatingPointTypes    = t(FloatTyID) | t(DoubleTyID);

static const long IntegralTypes = SignedIntegralTypes | UnsignedIntegralTypes;

static long ValidTypes[Type::FirstDerivedTyID] = {
  [(unsigned)Instruction::UnaryOps::Not] t(BoolTyID),
  //[Instruction::UnaryOps::Add] = IntegralTypes,
  //  [Instruction::Sub] = IntegralTypes,
};
#undef t
#endif

// CheckFailed - A check failed, so print out the condition and the message that
// failed.  This provides a nice place to put a breakpoint if you want to see
// why something is not correct.
//
static inline void CheckFailed(const char *Cond, const std::string &Message,
                               const Value *V1 = 0, const Value *V2 = 0) {
  std::cerr << Message << "\n";
  if (V1) std::cerr << V1 << "\n";
  if (V2) std::cerr << V2 << "\n";
}

// Assert - We know that cond should be true, if not print an error message.
#define Assert(C, M) \
  do { if (!(C)) { CheckFailed(#C, M); Broken = true; } } while (0)
#define Assert1(C, M, V1) \
  do { if (!(C)) { CheckFailed(#C, M, V1); Broken = true; } } while (0)
#define Assert2(C, M, V1, V2) \
  do { if (!(C)) { CheckFailed(#C, M, V1, V2); Broken = true; } } while (0)


// verifyInstruction - Verify that a non-terminator instruction is well formed.
//
static bool verifyInstruction(const Instruction *I) {
  bool Broken = false;
  assert(I->getParent() && "Instruction not embedded in basic block!");
  Assert1(!isa<TerminatorInst>(I),
          "Terminator instruction found embedded in basic block!\n", I);

  // Check that all uses of the instruction, if they are instructions
  // themselves, actually have parent basic blocks.
  //
  for (User::use_const_iterator UI = I->use_begin(), UE = I->use_end();
       UI != UE; ++UI) {
    if (Instruction *Used = dyn_cast<Instruction>(*UI))
      Assert2(Used->getParent() != 0, "Instruction referencing instruction not"
              " embeded in a basic block!", I, Used);
  }

  // Check that PHI nodes look ok
  if (const PHINode *PN = dyn_cast<PHINode>(I)) {
    std::vector<const BasicBlock*> Preds(pred_begin(I->getParent()),
                                         pred_end(I->getParent()));
    // Loop over all of the incoming values, make sure that there are
    // predecessors for each one...
    //
    for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i) {
      const BasicBlock *BB = PN->getIncomingBlock(i);
      std::vector<const BasicBlock*>::iterator PI =
        find(Preds.begin(), Preds.end(), BB);
      Assert2(PI != Preds.end(), "PHI node has entry for basic block that"
              " is not a predecessor!", PN, BB);
      if (PI != Preds.end()) Preds.erase(PI);
    }

    // There should be no entries left in the predecessor list...
    for (std::vector<const BasicBlock*>::iterator I = Preds.begin(),
           E = Preds.end(); I != E; ++I)
      Assert2(0, "PHI node does not have entry for a predecessor basic block!",
              PN, *I);
  } else {
    // Check that non-phi nodes are not self referential...
    for (Value::use_const_iterator UI = I->use_begin(), UE = I->use_end();
         UI != UE; ++UI)
      Assert1(*UI != (const User*)I,
              "Only PHI nodes may reference their own value!", I);
  }

  return Broken;
}


// verifyBasicBlock - Verify that a basic block is well formed...
//
static bool verifyBasicBlock(const BasicBlock *BB) {
  bool Broken = false;
  Assert1(BB->getTerminator(), "Basic Block does not have terminator!\n", BB);

  // Verify all instructions, except the terminator...
  Broken |= reduce_apply_bool(BB->begin(), BB->end()-1, verifyInstruction);
  return Broken;
}


// verifyMethod - Verify that a method is ok.
//
bool verifyMethod(const Method *M) {
  if (M->isExternal()) return false;  // Can happen if called by verifyModule

  bool Broken = false;
  const BasicBlock *Entry = M->front();
  Assert1(pred_begin(Entry) == pred_end(Entry),
          "Entry block to method must not have predecessors!", Entry);

  Broken |= reduce_apply_bool(M->begin(), M->end(), verifyBasicBlock);
  return Broken;
}


namespace {  // Anonymous namespace for class
  struct VerifierPass : public MethodPass {
    bool runOnMethod(Method *M) { verifyMethod(M); return false; }
  };
}

Pass *createVerifierPass() {
  return new VerifierPass();
}

// verifyModule - Check a module for errors, printing messages on stderr.
// Return true if the module is corrupt.
//
bool verifyModule(const Module *M) {
  return reduce_apply_bool(M->begin(), M->end(), verifyMethod);
}
