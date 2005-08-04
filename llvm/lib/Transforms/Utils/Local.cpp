//===-- Local.cpp - Functions to perform local transformations ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This family of functions perform various local transformations to the
// program.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/MathExtras.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Constants.h"
#include "llvm/Instructions.h"
#include "llvm/Intrinsics.h"
#include <cerrno>
#include <cmath>
using namespace llvm;

//===----------------------------------------------------------------------===//
//  Local constant propagation...
//

/// doConstantPropagation - If an instruction references constants, try to fold
/// them together...
///
bool llvm::doConstantPropagation(BasicBlock::iterator &II) {
  if (Constant *C = ConstantFoldInstruction(II)) {
    // Replaces all of the uses of a variable with uses of the constant.
    II->replaceAllUsesWith(C);

    // Remove the instruction from the basic block...
    II = II->getParent()->getInstList().erase(II);
    return true;
  }

  return false;
}

/// ConstantFoldInstruction - Attempt to constant fold the specified
/// instruction.  If successful, the constant result is returned, if not, null
/// is returned.  Note that this function can only fail when attempting to fold
/// instructions like loads and stores, which have no constant expression form.
///
Constant *llvm::ConstantFoldInstruction(Instruction *I) {
  if (PHINode *PN = dyn_cast<PHINode>(I)) {
    if (PN->getNumIncomingValues() == 0)
      return Constant::getNullValue(PN->getType());

    Constant *Result = dyn_cast<Constant>(PN->getIncomingValue(0));
    if (Result == 0) return 0;

    // Handle PHI nodes specially here...
    for (unsigned i = 1, e = PN->getNumIncomingValues(); i != e; ++i)
      if (PN->getIncomingValue(i) != Result && PN->getIncomingValue(i) != PN)
        return 0;   // Not all the same incoming constants...

    // If we reach here, all incoming values are the same constant.
    return Result;
  } else if (CallInst *CI = dyn_cast<CallInst>(I)) {
    if (Function *F = CI->getCalledFunction())
      if (canConstantFoldCallTo(F)) {
        std::vector<Constant*> Args;
        for (unsigned i = 1, e = CI->getNumOperands(); i != e; ++i)
          if (Constant *Op = dyn_cast<Constant>(CI->getOperand(i)))
            Args.push_back(Op);
          else
            return 0;
        return ConstantFoldCall(F, Args);
      }
    return 0;
  }

  Constant *Op0 = 0, *Op1 = 0;
  switch (I->getNumOperands()) {
  default:
  case 2:
    Op1 = dyn_cast<Constant>(I->getOperand(1));
    if (Op1 == 0) return 0;        // Not a constant?, can't fold
  case 1:
    Op0 = dyn_cast<Constant>(I->getOperand(0));
    if (Op0 == 0) return 0;        // Not a constant?, can't fold
    break;
  case 0: return 0;
  }

  if (isa<BinaryOperator>(I) || isa<ShiftInst>(I))
    return ConstantExpr::get(I->getOpcode(), Op0, Op1);

  switch (I->getOpcode()) {
  default: return 0;
  case Instruction::Cast:
    return ConstantExpr::getCast(Op0, I->getType());
  case Instruction::Select:
    if (Constant *Op2 = dyn_cast<Constant>(I->getOperand(2)))
      return ConstantExpr::getSelect(Op0, Op1, Op2);
    return 0;
  case Instruction::GetElementPtr:
    std::vector<Constant*> IdxList;
    IdxList.reserve(I->getNumOperands()-1);
    if (Op1) IdxList.push_back(Op1);
    for (unsigned i = 2, e = I->getNumOperands(); i != e; ++i)
      if (Constant *C = dyn_cast<Constant>(I->getOperand(i)))
        IdxList.push_back(C);
      else
        return 0;  // Non-constant operand
    return ConstantExpr::getGetElementPtr(Op0, IdxList);
  }
}

// ConstantFoldTerminator - If a terminator instruction is predicated on a
// constant value, convert it into an unconditional branch to the constant
// destination.
//
bool llvm::ConstantFoldTerminator(BasicBlock *BB) {
  TerminatorInst *T = BB->getTerminator();

  // Branch - See if we are conditional jumping on constant
  if (BranchInst *BI = dyn_cast<BranchInst>(T)) {
    if (BI->isUnconditional()) return false;  // Can't optimize uncond branch
    BasicBlock *Dest1 = cast<BasicBlock>(BI->getOperand(0));
    BasicBlock *Dest2 = cast<BasicBlock>(BI->getOperand(1));

    if (ConstantBool *Cond = dyn_cast<ConstantBool>(BI->getCondition())) {
      // Are we branching on constant?
      // YES.  Change to unconditional branch...
      BasicBlock *Destination = Cond->getValue() ? Dest1 : Dest2;
      BasicBlock *OldDest     = Cond->getValue() ? Dest2 : Dest1;

      //cerr << "Function: " << T->getParent()->getParent()
      //     << "\nRemoving branch from " << T->getParent()
      //     << "\n\nTo: " << OldDest << endl;

      // Let the basic block know that we are letting go of it.  Based on this,
      // it will adjust it's PHI nodes.
      assert(BI->getParent() && "Terminator not inserted in block!");
      OldDest->removePredecessor(BI->getParent());

      // Set the unconditional destination, and change the insn to be an
      // unconditional branch.
      BI->setUnconditionalDest(Destination);
      return true;
    } else if (Dest2 == Dest1) {       // Conditional branch to same location?
      // This branch matches something like this:
      //     br bool %cond, label %Dest, label %Dest
      // and changes it into:  br label %Dest

      // Let the basic block know that we are letting go of one copy of it.
      assert(BI->getParent() && "Terminator not inserted in block!");
      Dest1->removePredecessor(BI->getParent());

      // Change a conditional branch to unconditional.
      BI->setUnconditionalDest(Dest1);
      return true;
    }
  } else if (SwitchInst *SI = dyn_cast<SwitchInst>(T)) {
    // If we are switching on a constant, we can convert the switch into a
    // single branch instruction!
    ConstantInt *CI = dyn_cast<ConstantInt>(SI->getCondition());
    BasicBlock *TheOnlyDest = SI->getSuccessor(0);  // The default dest
    BasicBlock *DefaultDest = TheOnlyDest;
    assert(TheOnlyDest == SI->getDefaultDest() &&
           "Default destination is not successor #0?");

    // Figure out which case it goes to...
    for (unsigned i = 1, e = SI->getNumSuccessors(); i != e; ++i) {
      // Found case matching a constant operand?
      if (SI->getSuccessorValue(i) == CI) {
        TheOnlyDest = SI->getSuccessor(i);
        break;
      }

      // Check to see if this branch is going to the same place as the default
      // dest.  If so, eliminate it as an explicit compare.
      if (SI->getSuccessor(i) == DefaultDest) {
        // Remove this entry...
        DefaultDest->removePredecessor(SI->getParent());
        SI->removeCase(i);
        --i; --e;  // Don't skip an entry...
        continue;
      }

      // Otherwise, check to see if the switch only branches to one destination.
      // We do this by reseting "TheOnlyDest" to null when we find two non-equal
      // destinations.
      if (SI->getSuccessor(i) != TheOnlyDest) TheOnlyDest = 0;
    }

    if (CI && !TheOnlyDest) {
      // Branching on a constant, but not any of the cases, go to the default
      // successor.
      TheOnlyDest = SI->getDefaultDest();
    }

    // If we found a single destination that we can fold the switch into, do so
    // now.
    if (TheOnlyDest) {
      // Insert the new branch..
      new BranchInst(TheOnlyDest, SI);
      BasicBlock *BB = SI->getParent();

      // Remove entries from PHI nodes which we no longer branch to...
      for (unsigned i = 0, e = SI->getNumSuccessors(); i != e; ++i) {
        // Found case matching a constant operand?
        BasicBlock *Succ = SI->getSuccessor(i);
        if (Succ == TheOnlyDest)
          TheOnlyDest = 0;  // Don't modify the first branch to TheOnlyDest
        else
          Succ->removePredecessor(BB);
      }

      // Delete the old switch...
      BB->getInstList().erase(SI);
      return true;
    } else if (SI->getNumSuccessors() == 2) {
      // Otherwise, we can fold this switch into a conditional branch
      // instruction if it has only one non-default destination.
      Value *Cond = new SetCondInst(Instruction::SetEQ, SI->getCondition(),
                                    SI->getSuccessorValue(1), "cond", SI);
      // Insert the new branch...
      new BranchInst(SI->getSuccessor(1), SI->getSuccessor(0), Cond, SI);

      // Delete the old switch...
      SI->getParent()->getInstList().erase(SI);
      return true;
    }
  }
  return false;
}

/// canConstantFoldCallTo - Return true if its even possible to fold a call to
/// the specified function.
bool llvm::canConstantFoldCallTo(Function *F) {
  const std::string &Name = F->getName();

  switch (F->getIntrinsicID()) {
  case Intrinsic::isunordered: return true;
  default: break;
  }

  switch (Name[0])
  {
    case 'a':
      return Name == "acos" || Name == "asin" || Name == "atan" ||
             Name == "atan2";
    case 'c':
      return Name == "ceil" || Name == "cos" || Name == "cosf" ||
             Name == "cosh";
    case 'e':
      return Name == "exp";
    case 'f':
      return Name == "fabs" || Name == "fmod" || Name == "floor";
    case 'l':
      return Name == "log" || Name == "log10";
    case 'p':
      return Name == "pow";
    case 's':
      return Name == "sin" || Name == "sinh" || Name == "sqrt";
    case 't':
      return Name == "tan" || Name == "tanh";
    default:
      return false;
  }
}

static Constant *ConstantFoldFP(double (*NativeFP)(double), double V,
                                const Type *Ty) {
  errno = 0;
  V = NativeFP(V);
  if (errno == 0)
    return ConstantFP::get(Ty, V);
  return 0;
}

/// ConstantFoldCall - Attempt to constant fold a call to the specified function
/// with the specified arguments, returning null if unsuccessful.
Constant *llvm::ConstantFoldCall(Function *F,
                                 const std::vector<Constant*> &Operands) {
  const std::string &Name = F->getName();
  const Type *Ty = F->getReturnType();

  if (Operands.size() == 1) {
    if (ConstantFP *Op = dyn_cast<ConstantFP>(Operands[0])) {
      double V = Op->getValue();
      switch (Name[0])
      {
        case 'a':
          if (Name == "acos")
            return ConstantFoldFP(acos, V, Ty);
          else if (Name == "asin")
            return ConstantFoldFP(asin, V, Ty);
          else if (Name == "atan")
            return ConstantFP::get(Ty, atan(V));
          break;
        case 'c':
          if (Name == "ceil")
            return ConstantFoldFP(ceil, V, Ty);
          else if (Name == "cos")
            return ConstantFP::get(Ty, cos(V));
          else if (Name == "cosh")
            return ConstantFP::get(Ty, cosh(V));
          break;
        case 'e':
          if (Name == "exp")
            return ConstantFP::get(Ty, exp(V));
          break;
        case 'f':
          if (Name == "fabs")
            return ConstantFP::get(Ty, fabs(V));
          else if (Name == "floor")
            return ConstantFoldFP(floor, V, Ty);
          break;
        case 'l':
          if (Name == "log" && V > 0)
            return ConstantFP::get(Ty, log(V));
          else if (Name == "log10" && V > 0)
            return ConstantFoldFP(log10, V, Ty);
          break;
        case 's':
          if (Name == "sin")
            return ConstantFP::get(Ty, sin(V));
          else if (Name == "sinh")
            return ConstantFP::get(Ty, sinh(V));
          else if (Name == "sqrt" && V >= 0)
            return ConstantFP::get(Ty, sqrt(V));
          break;
        case 't':
          if (Name == "tan")
            return ConstantFP::get(Ty, tan(V));
          else if (Name == "tanh")
            return ConstantFP::get(Ty, tanh(V));
          break;
        default:
          break;
      }
    }
  } else if (Operands.size() == 2) {
    if (ConstantFP *Op1 = dyn_cast<ConstantFP>(Operands[0])) {
      double Op1V = Op1->getValue();
      if (ConstantFP *Op2 = dyn_cast<ConstantFP>(Operands[1])) {
        double Op2V = Op2->getValue();

        if (Name == "llvm.isunordered")
          return ConstantBool::get(IsNAN(Op1V) || IsNAN(Op2V));
        else
        if (Name == "pow") {
          errno = 0;
          double V = pow(Op1V, Op2V);
          if (errno == 0)
            return ConstantFP::get(Ty, V);
        } else if (Name == "fmod") {
          errno = 0;
          double V = fmod(Op1V, Op2V);
          if (errno == 0)
            return ConstantFP::get(Ty, V);
        } else if (Name == "atan2")
          return ConstantFP::get(Ty, atan2(Op1V,Op2V));
      }
    }
  }
  return 0;
}




//===----------------------------------------------------------------------===//
//  Local dead code elimination...
//

bool llvm::isInstructionTriviallyDead(Instruction *I) {
  if (!I->use_empty() || isa<TerminatorInst>(I)) return false;

  if (!I->mayWriteToMemory()) return true;

  if (CallInst *CI = dyn_cast<CallInst>(I))
    if (Function *F = CI->getCalledFunction())
      switch (F->getIntrinsicID()) {
      default: break;
      case Intrinsic::returnaddress:
      case Intrinsic::frameaddress:
      case Intrinsic::isunordered:
      case Intrinsic::ctpop:
      case Intrinsic::ctlz:
      case Intrinsic::cttz:
      case Intrinsic::sqrt:
        return true;             // These intrinsics have no side effects.
      }
  return false;
}

// dceInstruction - Inspect the instruction at *BBI and figure out if it's
// [trivially] dead.  If so, remove the instruction and update the iterator
// to point to the instruction that immediately succeeded the original
// instruction.
//
bool llvm::dceInstruction(BasicBlock::iterator &BBI) {
  // Look for un"used" definitions...
  if (isInstructionTriviallyDead(BBI)) {
    BBI = BBI->getParent()->getInstList().erase(BBI);   // Bye bye
    return true;
  }
  return false;
}
