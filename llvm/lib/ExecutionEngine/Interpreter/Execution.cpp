//===-- Execution.cpp - Implement code to simulate the program ------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
// 
//  This file contains the actual instruction interpreter.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "interpreter"
#include "Interpreter.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Instructions.h"
#include "llvm/IntrinsicLowering.h"
#include "llvm/Support/GetElementPtrTypeIterator.h"
#include "Support/Statistic.h"
#include "Support/Debug.h"
#include <cmath>  // For fmod
using namespace llvm;

namespace {
  Statistic<> NumDynamicInsts("lli", "Number of dynamic instructions executed");

  Interpreter *TheEE = 0;
}


//===----------------------------------------------------------------------===//
//                     Value Manipulation code
//===----------------------------------------------------------------------===//

static GenericValue executeAddInst(GenericValue Src1, GenericValue Src2, 
				   const Type *Ty);
static GenericValue executeSubInst(GenericValue Src1, GenericValue Src2, 
				   const Type *Ty);
static GenericValue executeMulInst(GenericValue Src1, GenericValue Src2, 
				   const Type *Ty);
static GenericValue executeRemInst(GenericValue Src1, GenericValue Src2, 
				   const Type *Ty);
static GenericValue executeDivInst(GenericValue Src1, GenericValue Src2, 
				   const Type *Ty);
static GenericValue executeAndInst(GenericValue Src1, GenericValue Src2, 
				   const Type *Ty);
static GenericValue executeOrInst(GenericValue Src1, GenericValue Src2, 
				   const Type *Ty);
static GenericValue executeXorInst(GenericValue Src1, GenericValue Src2, 
				   const Type *Ty);
static GenericValue executeSetEQInst(GenericValue Src1, GenericValue Src2, 
				   const Type *Ty);
static GenericValue executeSetNEInst(GenericValue Src1, GenericValue Src2, 
				   const Type *Ty);
static GenericValue executeSetLTInst(GenericValue Src1, GenericValue Src2, 
				   const Type *Ty);
static GenericValue executeSetGTInst(GenericValue Src1, GenericValue Src2, 
				   const Type *Ty);
static GenericValue executeSetLEInst(GenericValue Src1, GenericValue Src2, 
				   const Type *Ty);
static GenericValue executeSetGEInst(GenericValue Src1, GenericValue Src2, 
				   const Type *Ty);
static GenericValue executeShlInst(GenericValue Src1, GenericValue Src2, 
				   const Type *Ty);
static GenericValue executeShrInst(GenericValue Src1, GenericValue Src2, 
				   const Type *Ty);
                                   
GenericValue Interpreter::getConstantExprValue (ConstantExpr *CE,
                                                ExecutionContext &SF) {
  switch (CE->getOpcode()) {
  case Instruction::Cast:
    return executeCastOperation(CE->getOperand(0), CE->getType(), SF);
  case Instruction::GetElementPtr:
    return executeGEPOperation(CE->getOperand(0), gep_type_begin(CE),
                               gep_type_end(CE), SF);
  case Instruction::Add:
    return executeAddInst(getOperandValue(CE->getOperand(0), SF),
                          getOperandValue(CE->getOperand(1), SF),
                          CE->getOperand(0)->getType());
  case Instruction::Sub:
    return executeSubInst(getOperandValue(CE->getOperand(0), SF),
                          getOperandValue(CE->getOperand(1), SF),
                          CE->getOperand(0)->getType());
  case Instruction::Mul:
    return executeMulInst(getOperandValue(CE->getOperand(0), SF),
                          getOperandValue(CE->getOperand(1), SF),
                          CE->getOperand(0)->getType());
  case Instruction::Div:
    return executeDivInst(getOperandValue(CE->getOperand(0), SF),
                          getOperandValue(CE->getOperand(1), SF),
                          CE->getOperand(0)->getType());
  case Instruction::Rem:
    return executeRemInst(getOperandValue(CE->getOperand(0), SF),
                          getOperandValue(CE->getOperand(1), SF),
                          CE->getOperand(0)->getType());
  case Instruction::And:
    return executeAndInst(getOperandValue(CE->getOperand(0), SF),
                          getOperandValue(CE->getOperand(1), SF),
                          CE->getOperand(0)->getType());
  case Instruction::Or:
    return executeOrInst(getOperandValue(CE->getOperand(0), SF),
                         getOperandValue(CE->getOperand(1), SF),
                         CE->getOperand(0)->getType());
  case Instruction::Xor:
    return executeXorInst(getOperandValue(CE->getOperand(0), SF),
                          getOperandValue(CE->getOperand(1), SF),
                          CE->getOperand(0)->getType());
  case Instruction::SetEQ:
    return executeSetEQInst(getOperandValue(CE->getOperand(0), SF),
                            getOperandValue(CE->getOperand(1), SF),
                            CE->getOperand(0)->getType());
  case Instruction::SetNE:
    return executeSetNEInst(getOperandValue(CE->getOperand(0), SF),
                            getOperandValue(CE->getOperand(1), SF),
                            CE->getOperand(0)->getType());
  case Instruction::SetLE:
    return executeSetLEInst(getOperandValue(CE->getOperand(0), SF),
                            getOperandValue(CE->getOperand(1), SF),
                            CE->getOperand(0)->getType());
  case Instruction::SetGE:
    return executeSetGEInst(getOperandValue(CE->getOperand(0), SF),
                            getOperandValue(CE->getOperand(1), SF),
                            CE->getOperand(0)->getType());
  case Instruction::SetLT:
    return executeSetLTInst(getOperandValue(CE->getOperand(0), SF),
                            getOperandValue(CE->getOperand(1), SF),
                            CE->getOperand(0)->getType());
  case Instruction::SetGT:
    return executeSetGTInst(getOperandValue(CE->getOperand(0), SF),
                            getOperandValue(CE->getOperand(1), SF),
                            CE->getOperand(0)->getType());
  case Instruction::Shl:
    return executeShlInst(getOperandValue(CE->getOperand(0), SF),
                          getOperandValue(CE->getOperand(1), SF),
                          CE->getOperand(0)->getType());
  case Instruction::Shr:
    return executeShrInst(getOperandValue(CE->getOperand(0), SF),
                          getOperandValue(CE->getOperand(1), SF),
                          CE->getOperand(0)->getType());
  
  default:
    std::cerr << "Unhandled ConstantExpr: " << CE << "\n";
    abort();
    return GenericValue();
  }
}

GenericValue Interpreter::getOperandValue(Value *V, ExecutionContext &SF) {
  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(V)) {
    return getConstantExprValue(CE, SF);
  } else if (Constant *CPV = dyn_cast<Constant>(V)) {
    return getConstantValue(CPV);
  } else if (GlobalValue *GV = dyn_cast<GlobalValue>(V)) {
    return PTOGV(getPointerToGlobal(GV));
  } else {
    return SF.Values[V];
  }
}

static void SetValue(Value *V, GenericValue Val, ExecutionContext &SF) {
  SF.Values[V] = Val;
}

void Interpreter::initializeExecutionEngine() {
  TheEE = this;
}

//===----------------------------------------------------------------------===//
//                    Binary Instruction Implementations
//===----------------------------------------------------------------------===//

#define IMPLEMENT_BINARY_OPERATOR(OP, TY) \
   case Type::TY##TyID: Dest.TY##Val = Src1.TY##Val OP Src2.TY##Val; break

static GenericValue executeAddInst(GenericValue Src1, GenericValue Src2, 
				   const Type *Ty) {
  GenericValue Dest;
  switch (Ty->getPrimitiveID()) {
    IMPLEMENT_BINARY_OPERATOR(+, UByte);
    IMPLEMENT_BINARY_OPERATOR(+, SByte);
    IMPLEMENT_BINARY_OPERATOR(+, UShort);
    IMPLEMENT_BINARY_OPERATOR(+, Short);
    IMPLEMENT_BINARY_OPERATOR(+, UInt);
    IMPLEMENT_BINARY_OPERATOR(+, Int);
    IMPLEMENT_BINARY_OPERATOR(+, ULong);
    IMPLEMENT_BINARY_OPERATOR(+, Long);
    IMPLEMENT_BINARY_OPERATOR(+, Float);
    IMPLEMENT_BINARY_OPERATOR(+, Double);
  default:
    std::cout << "Unhandled type for Add instruction: " << *Ty << "\n";
    abort();
  }
  return Dest;
}

static GenericValue executeSubInst(GenericValue Src1, GenericValue Src2, 
				   const Type *Ty) {
  GenericValue Dest;
  switch (Ty->getPrimitiveID()) {
    IMPLEMENT_BINARY_OPERATOR(-, UByte);
    IMPLEMENT_BINARY_OPERATOR(-, SByte);
    IMPLEMENT_BINARY_OPERATOR(-, UShort);
    IMPLEMENT_BINARY_OPERATOR(-, Short);
    IMPLEMENT_BINARY_OPERATOR(-, UInt);
    IMPLEMENT_BINARY_OPERATOR(-, Int);
    IMPLEMENT_BINARY_OPERATOR(-, ULong);
    IMPLEMENT_BINARY_OPERATOR(-, Long);
    IMPLEMENT_BINARY_OPERATOR(-, Float);
    IMPLEMENT_BINARY_OPERATOR(-, Double);
  default:
    std::cout << "Unhandled type for Sub instruction: " << *Ty << "\n";
    abort();
  }
  return Dest;
}

static GenericValue executeMulInst(GenericValue Src1, GenericValue Src2, 
				   const Type *Ty) {
  GenericValue Dest;
  switch (Ty->getPrimitiveID()) {
    IMPLEMENT_BINARY_OPERATOR(*, UByte);
    IMPLEMENT_BINARY_OPERATOR(*, SByte);
    IMPLEMENT_BINARY_OPERATOR(*, UShort);
    IMPLEMENT_BINARY_OPERATOR(*, Short);
    IMPLEMENT_BINARY_OPERATOR(*, UInt);
    IMPLEMENT_BINARY_OPERATOR(*, Int);
    IMPLEMENT_BINARY_OPERATOR(*, ULong);
    IMPLEMENT_BINARY_OPERATOR(*, Long);
    IMPLEMENT_BINARY_OPERATOR(*, Float);
    IMPLEMENT_BINARY_OPERATOR(*, Double);
  default:
    std::cout << "Unhandled type for Mul instruction: " << Ty << "\n";
    abort();
  }
  return Dest;
}

static GenericValue executeDivInst(GenericValue Src1, GenericValue Src2, 
				   const Type *Ty) {
  GenericValue Dest;
  switch (Ty->getPrimitiveID()) {
    IMPLEMENT_BINARY_OPERATOR(/, UByte);
    IMPLEMENT_BINARY_OPERATOR(/, SByte);
    IMPLEMENT_BINARY_OPERATOR(/, UShort);
    IMPLEMENT_BINARY_OPERATOR(/, Short);
    IMPLEMENT_BINARY_OPERATOR(/, UInt);
    IMPLEMENT_BINARY_OPERATOR(/, Int);
    IMPLEMENT_BINARY_OPERATOR(/, ULong);
    IMPLEMENT_BINARY_OPERATOR(/, Long);
    IMPLEMENT_BINARY_OPERATOR(/, Float);
    IMPLEMENT_BINARY_OPERATOR(/, Double);
  default:
    std::cout << "Unhandled type for Div instruction: " << *Ty << "\n";
    abort();
  }
  return Dest;
}

static GenericValue executeRemInst(GenericValue Src1, GenericValue Src2, 
				   const Type *Ty) {
  GenericValue Dest;
  switch (Ty->getPrimitiveID()) {
    IMPLEMENT_BINARY_OPERATOR(%, UByte);
    IMPLEMENT_BINARY_OPERATOR(%, SByte);
    IMPLEMENT_BINARY_OPERATOR(%, UShort);
    IMPLEMENT_BINARY_OPERATOR(%, Short);
    IMPLEMENT_BINARY_OPERATOR(%, UInt);
    IMPLEMENT_BINARY_OPERATOR(%, Int);
    IMPLEMENT_BINARY_OPERATOR(%, ULong);
    IMPLEMENT_BINARY_OPERATOR(%, Long);
  case Type::FloatTyID:
    Dest.FloatVal = fmod(Src1.FloatVal, Src2.FloatVal);
    break;
  case Type::DoubleTyID:
    Dest.DoubleVal = fmod(Src1.DoubleVal, Src2.DoubleVal);
    break;
  default:
    std::cout << "Unhandled type for Rem instruction: " << *Ty << "\n";
    abort();
  }
  return Dest;
}

static GenericValue executeAndInst(GenericValue Src1, GenericValue Src2, 
				   const Type *Ty) {
  GenericValue Dest;
  switch (Ty->getPrimitiveID()) {
    IMPLEMENT_BINARY_OPERATOR(&, Bool);
    IMPLEMENT_BINARY_OPERATOR(&, UByte);
    IMPLEMENT_BINARY_OPERATOR(&, SByte);
    IMPLEMENT_BINARY_OPERATOR(&, UShort);
    IMPLEMENT_BINARY_OPERATOR(&, Short);
    IMPLEMENT_BINARY_OPERATOR(&, UInt);
    IMPLEMENT_BINARY_OPERATOR(&, Int);
    IMPLEMENT_BINARY_OPERATOR(&, ULong);
    IMPLEMENT_BINARY_OPERATOR(&, Long);
  default:
    std::cout << "Unhandled type for And instruction: " << *Ty << "\n";
    abort();
  }
  return Dest;
}

static GenericValue executeOrInst(GenericValue Src1, GenericValue Src2, 
                                  const Type *Ty) {
  GenericValue Dest;
  switch (Ty->getPrimitiveID()) {
    IMPLEMENT_BINARY_OPERATOR(|, Bool);
    IMPLEMENT_BINARY_OPERATOR(|, UByte);
    IMPLEMENT_BINARY_OPERATOR(|, SByte);
    IMPLEMENT_BINARY_OPERATOR(|, UShort);
    IMPLEMENT_BINARY_OPERATOR(|, Short);
    IMPLEMENT_BINARY_OPERATOR(|, UInt);
    IMPLEMENT_BINARY_OPERATOR(|, Int);
    IMPLEMENT_BINARY_OPERATOR(|, ULong);
    IMPLEMENT_BINARY_OPERATOR(|, Long);
  default:
    std::cout << "Unhandled type for Or instruction: " << *Ty << "\n";
    abort();
  }
  return Dest;
}

static GenericValue executeXorInst(GenericValue Src1, GenericValue Src2, 
                                   const Type *Ty) {
  GenericValue Dest;
  switch (Ty->getPrimitiveID()) {
    IMPLEMENT_BINARY_OPERATOR(^, Bool);
    IMPLEMENT_BINARY_OPERATOR(^, UByte);
    IMPLEMENT_BINARY_OPERATOR(^, SByte);
    IMPLEMENT_BINARY_OPERATOR(^, UShort);
    IMPLEMENT_BINARY_OPERATOR(^, Short);
    IMPLEMENT_BINARY_OPERATOR(^, UInt);
    IMPLEMENT_BINARY_OPERATOR(^, Int);
    IMPLEMENT_BINARY_OPERATOR(^, ULong);
    IMPLEMENT_BINARY_OPERATOR(^, Long);
  default:
    std::cout << "Unhandled type for Xor instruction: " << *Ty << "\n";
    abort();
  }
  return Dest;
}

#define IMPLEMENT_SETCC(OP, TY) \
   case Type::TY##TyID: Dest.BoolVal = Src1.TY##Val OP Src2.TY##Val; break

// Handle pointers specially because they must be compared with only as much
// width as the host has.  We _do not_ want to be comparing 64 bit values when
// running on a 32-bit target, otherwise the upper 32 bits might mess up
// comparisons if they contain garbage.
#define IMPLEMENT_POINTERSETCC(OP) \
   case Type::PointerTyID: \
        Dest.BoolVal = (void*)(intptr_t)Src1.PointerVal OP \
                       (void*)(intptr_t)Src2.PointerVal; break

static GenericValue executeSetEQInst(GenericValue Src1, GenericValue Src2, 
				     const Type *Ty) {
  GenericValue Dest;
  switch (Ty->getPrimitiveID()) {
    IMPLEMENT_SETCC(==, UByte);
    IMPLEMENT_SETCC(==, SByte);
    IMPLEMENT_SETCC(==, UShort);
    IMPLEMENT_SETCC(==, Short);
    IMPLEMENT_SETCC(==, UInt);
    IMPLEMENT_SETCC(==, Int);
    IMPLEMENT_SETCC(==, ULong);
    IMPLEMENT_SETCC(==, Long);
    IMPLEMENT_SETCC(==, Float);
    IMPLEMENT_SETCC(==, Double);
    IMPLEMENT_POINTERSETCC(==);
  default:
    std::cout << "Unhandled type for SetEQ instruction: " << *Ty << "\n";
    abort();
  }
  return Dest;
}

static GenericValue executeSetNEInst(GenericValue Src1, GenericValue Src2, 
				     const Type *Ty) {
  GenericValue Dest;
  switch (Ty->getPrimitiveID()) {
    IMPLEMENT_SETCC(!=, UByte);
    IMPLEMENT_SETCC(!=, SByte);
    IMPLEMENT_SETCC(!=, UShort);
    IMPLEMENT_SETCC(!=, Short);
    IMPLEMENT_SETCC(!=, UInt);
    IMPLEMENT_SETCC(!=, Int);
    IMPLEMENT_SETCC(!=, ULong);
    IMPLEMENT_SETCC(!=, Long);
    IMPLEMENT_SETCC(!=, Float);
    IMPLEMENT_SETCC(!=, Double);
    IMPLEMENT_POINTERSETCC(!=);

  default:
    std::cout << "Unhandled type for SetNE instruction: " << *Ty << "\n";
    abort();
  }
  return Dest;
}

static GenericValue executeSetLEInst(GenericValue Src1, GenericValue Src2, 
				     const Type *Ty) {
  GenericValue Dest;
  switch (Ty->getPrimitiveID()) {
    IMPLEMENT_SETCC(<=, UByte);
    IMPLEMENT_SETCC(<=, SByte);
    IMPLEMENT_SETCC(<=, UShort);
    IMPLEMENT_SETCC(<=, Short);
    IMPLEMENT_SETCC(<=, UInt);
    IMPLEMENT_SETCC(<=, Int);
    IMPLEMENT_SETCC(<=, ULong);
    IMPLEMENT_SETCC(<=, Long);
    IMPLEMENT_SETCC(<=, Float);
    IMPLEMENT_SETCC(<=, Double);
    IMPLEMENT_POINTERSETCC(<=);
  default:
    std::cout << "Unhandled type for SetLE instruction: " << Ty << "\n";
    abort();
  }
  return Dest;
}

static GenericValue executeSetGEInst(GenericValue Src1, GenericValue Src2, 
				     const Type *Ty) {
  GenericValue Dest;
  switch (Ty->getPrimitiveID()) {
    IMPLEMENT_SETCC(>=, UByte);
    IMPLEMENT_SETCC(>=, SByte);
    IMPLEMENT_SETCC(>=, UShort);
    IMPLEMENT_SETCC(>=, Short);
    IMPLEMENT_SETCC(>=, UInt);
    IMPLEMENT_SETCC(>=, Int);
    IMPLEMENT_SETCC(>=, ULong);
    IMPLEMENT_SETCC(>=, Long);
    IMPLEMENT_SETCC(>=, Float);
    IMPLEMENT_SETCC(>=, Double);
    IMPLEMENT_POINTERSETCC(>=);
  default:
    std::cout << "Unhandled type for SetGE instruction: " << *Ty << "\n";
    abort();
  }
  return Dest;
}

static GenericValue executeSetLTInst(GenericValue Src1, GenericValue Src2, 
				     const Type *Ty) {
  GenericValue Dest;
  switch (Ty->getPrimitiveID()) {
    IMPLEMENT_SETCC(<, UByte);
    IMPLEMENT_SETCC(<, SByte);
    IMPLEMENT_SETCC(<, UShort);
    IMPLEMENT_SETCC(<, Short);
    IMPLEMENT_SETCC(<, UInt);
    IMPLEMENT_SETCC(<, Int);
    IMPLEMENT_SETCC(<, ULong);
    IMPLEMENT_SETCC(<, Long);
    IMPLEMENT_SETCC(<, Float);
    IMPLEMENT_SETCC(<, Double);
    IMPLEMENT_POINTERSETCC(<);
  default:
    std::cout << "Unhandled type for SetLT instruction: " << *Ty << "\n";
    abort();
  }
  return Dest;
}

static GenericValue executeSetGTInst(GenericValue Src1, GenericValue Src2, 
				     const Type *Ty) {
  GenericValue Dest;
  switch (Ty->getPrimitiveID()) {
    IMPLEMENT_SETCC(>, UByte);
    IMPLEMENT_SETCC(>, SByte);
    IMPLEMENT_SETCC(>, UShort);
    IMPLEMENT_SETCC(>, Short);
    IMPLEMENT_SETCC(>, UInt);
    IMPLEMENT_SETCC(>, Int);
    IMPLEMENT_SETCC(>, ULong);
    IMPLEMENT_SETCC(>, Long);
    IMPLEMENT_SETCC(>, Float);
    IMPLEMENT_SETCC(>, Double);
    IMPLEMENT_POINTERSETCC(>);
  default:
    std::cout << "Unhandled type for SetGT instruction: " << *Ty << "\n";
    abort();
  }
  return Dest;
}

void Interpreter::visitBinaryOperator(BinaryOperator &I) {
  ExecutionContext &SF = ECStack.back();
  const Type *Ty    = I.getOperand(0)->getType();
  GenericValue Src1 = getOperandValue(I.getOperand(0), SF);
  GenericValue Src2 = getOperandValue(I.getOperand(1), SF);
  GenericValue R;   // Result

  switch (I.getOpcode()) {
  case Instruction::Add:   R = executeAddInst  (Src1, Src2, Ty); break;
  case Instruction::Sub:   R = executeSubInst  (Src1, Src2, Ty); break;
  case Instruction::Mul:   R = executeMulInst  (Src1, Src2, Ty); break;
  case Instruction::Div:   R = executeDivInst  (Src1, Src2, Ty); break;
  case Instruction::Rem:   R = executeRemInst  (Src1, Src2, Ty); break;
  case Instruction::And:   R = executeAndInst  (Src1, Src2, Ty); break;
  case Instruction::Or:    R = executeOrInst   (Src1, Src2, Ty); break;
  case Instruction::Xor:   R = executeXorInst  (Src1, Src2, Ty); break;
  case Instruction::SetEQ: R = executeSetEQInst(Src1, Src2, Ty); break;
  case Instruction::SetNE: R = executeSetNEInst(Src1, Src2, Ty); break;
  case Instruction::SetLE: R = executeSetLEInst(Src1, Src2, Ty); break;
  case Instruction::SetGE: R = executeSetGEInst(Src1, Src2, Ty); break;
  case Instruction::SetLT: R = executeSetLTInst(Src1, Src2, Ty); break;
  case Instruction::SetGT: R = executeSetGTInst(Src1, Src2, Ty); break;
  default:
    std::cout << "Don't know how to handle this binary operator!\n-->" << I;
    abort();
  }

  SetValue(&I, R, SF);
}

//===----------------------------------------------------------------------===//
//                     Terminator Instruction Implementations
//===----------------------------------------------------------------------===//

void Interpreter::exitCalled(GenericValue GV) {
  // runAtExitHandlers() assumes there are no stack frames, but
  // if exit() was called, then it had a stack frame. Blow away
  // the stack before interpreting atexit handlers.
  ECStack.clear ();
  runAtExitHandlers ();
  exit (GV.IntVal);
}

/// Pop the last stack frame off of ECStack and then copy the result
/// back into the result variable if we are not returning void. The
/// result variable may be the ExitCode, or the Value of the calling
/// CallInst if there was a previous stack frame. This method may
/// invalidate any ECStack iterators you have. This method also takes
/// care of switching to the normal destination BB, if we are returning
/// from an invoke.
///
void Interpreter::popStackAndReturnValueToCaller (const Type *RetTy,
                                                  GenericValue Result) {
  // Pop the current stack frame.
  ECStack.pop_back();

  if (ECStack.empty()) {  // Finished main.  Put result into exit code... 
    if (RetTy && RetTy->isIntegral()) {          // Nonvoid return type?       
      ExitCode = Result.IntVal;   // Capture the exit code of the program 
    } else { 
      ExitCode = 0; 
    } 
  } else { 
    // If we have a previous stack frame, and we have a previous call, 
    // fill in the return value... 
    ExecutionContext &CallingSF = ECStack.back();
    if (Instruction *I = CallingSF.Caller.getInstruction()) {
      if (CallingSF.Caller.getType() != Type::VoidTy)      // Save result...
        SetValue(I, Result, CallingSF);
      if (InvokeInst *II = dyn_cast<InvokeInst> (I))
        SwitchToNewBasicBlock (II->getNormalDest (), CallingSF);
      CallingSF.Caller = CallSite();          // We returned from the call...
    }
  }
}

void Interpreter::visitReturnInst(ReturnInst &I) {
  ExecutionContext &SF = ECStack.back();
  const Type *RetTy = Type::VoidTy;
  GenericValue Result;

  // Save away the return value... (if we are not 'ret void')
  if (I.getNumOperands()) {
    RetTy  = I.getReturnValue()->getType();
    Result = getOperandValue(I.getReturnValue(), SF);
  }

  popStackAndReturnValueToCaller(RetTy, Result);
}

void Interpreter::visitUnwindInst(UnwindInst &I) {
  // Unwind stack
  Instruction *Inst;
  do {
    ECStack.pop_back ();
    if (ECStack.empty ())
      abort ();
    Inst = ECStack.back ().Caller.getInstruction ();
  } while (!(Inst && isa<InvokeInst> (Inst)));

  // Return from invoke
  ExecutionContext &InvokingSF = ECStack.back ();
  InvokingSF.Caller = CallSite ();

  // Go to exceptional destination BB of invoke instruction
  SwitchToNewBasicBlock(cast<InvokeInst>(Inst)->getUnwindDest(), InvokingSF);
}

void Interpreter::visitBranchInst(BranchInst &I) {
  ExecutionContext &SF = ECStack.back();
  BasicBlock *Dest;

  Dest = I.getSuccessor(0);          // Uncond branches have a fixed dest...
  if (!I.isUnconditional()) {
    Value *Cond = I.getCondition();
    if (getOperandValue(Cond, SF).BoolVal == 0) // If false cond...
      Dest = I.getSuccessor(1);    
  }
  SwitchToNewBasicBlock(Dest, SF);
}

void Interpreter::visitSwitchInst(SwitchInst &I) {
  ExecutionContext &SF = ECStack.back();
  GenericValue CondVal = getOperandValue(I.getOperand(0), SF);
  const Type *ElTy = I.getOperand(0)->getType();

  // Check to see if any of the cases match...
  BasicBlock *Dest = 0;
  for (unsigned i = 2, e = I.getNumOperands(); i != e; i += 2)
    if (executeSetEQInst(CondVal,
                         getOperandValue(I.getOperand(i), SF), ElTy).BoolVal) {
      Dest = cast<BasicBlock>(I.getOperand(i+1));
      break;
    }
  
  if (!Dest) Dest = I.getDefaultDest();   // No cases matched: use default
  SwitchToNewBasicBlock(Dest, SF);
}

// SwitchToNewBasicBlock - This method is used to jump to a new basic block.
// This function handles the actual updating of block and instruction iterators
// as well as execution of all of the PHI nodes in the destination block.
//
// This method does this because all of the PHI nodes must be executed
// atomically, reading their inputs before any of the results are updated.  Not
// doing this can cause problems if the PHI nodes depend on other PHI nodes for
// their inputs.  If the input PHI node is updated before it is read, incorrect
// results can happen.  Thus we use a two phase approach.
//
void Interpreter::SwitchToNewBasicBlock(BasicBlock *Dest, ExecutionContext &SF){
  BasicBlock *PrevBB = SF.CurBB;      // Remember where we came from...
  SF.CurBB   = Dest;                  // Update CurBB to branch destination
  SF.CurInst = SF.CurBB->begin();     // Update new instruction ptr...

  if (!isa<PHINode>(SF.CurInst)) return;  // Nothing fancy to do

  // Loop over all of the PHI nodes in the current block, reading their inputs.
  std::vector<GenericValue> ResultValues;

  for (; PHINode *PN = dyn_cast<PHINode>(SF.CurInst); ++SF.CurInst) {
    // Search for the value corresponding to this previous bb...
    int i = PN->getBasicBlockIndex(PrevBB);
    assert(i != -1 && "PHINode doesn't contain entry for predecessor??");
    Value *IncomingValue = PN->getIncomingValue(i);
    
    // Save the incoming value for this PHI node...
    ResultValues.push_back(getOperandValue(IncomingValue, SF));
  }

  // Now loop over all of the PHI nodes setting their values...
  SF.CurInst = SF.CurBB->begin();
  for (unsigned i = 0; PHINode *PN = dyn_cast<PHINode>(SF.CurInst);
       ++SF.CurInst, ++i)
    SetValue(PN, ResultValues[i], SF);
}

//===----------------------------------------------------------------------===//
//                     Memory Instruction Implementations
//===----------------------------------------------------------------------===//

void Interpreter::visitAllocationInst(AllocationInst &I) {
  ExecutionContext &SF = ECStack.back();

  const Type *Ty = I.getType()->getElementType();  // Type to be allocated

  // Get the number of elements being allocated by the array...
  unsigned NumElements = getOperandValue(I.getOperand(0), SF).UIntVal;

  // Allocate enough memory to hold the type...
  void *Memory = malloc(NumElements * TD.getTypeSize(Ty));

  GenericValue Result = PTOGV(Memory);
  assert(Result.PointerVal != 0 && "Null pointer returned by malloc!");
  SetValue(&I, Result, SF);

  if (I.getOpcode() == Instruction::Alloca)
    ECStack.back().Allocas.add(Memory);
}

void Interpreter::visitFreeInst(FreeInst &I) {
  ExecutionContext &SF = ECStack.back();
  assert(isa<PointerType>(I.getOperand(0)->getType()) && "Freeing nonptr?");
  GenericValue Value = getOperandValue(I.getOperand(0), SF);
  // TODO: Check to make sure memory is allocated
  free(GVTOP(Value));   // Free memory
}

// getElementOffset - The workhorse for getelementptr.
//
GenericValue Interpreter::executeGEPOperation(Value *Ptr, gep_type_iterator I,
					      gep_type_iterator E,
					      ExecutionContext &SF) {
  assert(isa<PointerType>(Ptr->getType()) &&
         "Cannot getElementOffset of a nonpointer type!");

  PointerTy Total = 0;

  for (; I != E; ++I) {
    if (const StructType *STy = dyn_cast<StructType>(*I)) {
      const StructLayout *SLO = TD.getStructLayout(STy);
      
      const ConstantUInt *CPU = cast<ConstantUInt>(I.getOperand());
      unsigned Index = CPU->getValue();
      
      Total += SLO->MemberOffsets[Index];
    } else {
      const SequentialType *ST = cast<SequentialType>(*I);
      // Get the index number for the array... which must be long type...
      GenericValue IdxGV = getOperandValue(I.getOperand(), SF);

      uint64_t Idx;
      switch (I.getOperand()->getType()->getPrimitiveID()) {
      default: assert(0 && "Illegal getelementptr index for sequential type!");
      case Type::SByteTyID:  Idx = IdxGV.SByteVal; break;
      case Type::ShortTyID:  Idx = IdxGV.ShortVal; break;
      case Type::IntTyID:    Idx = IdxGV.IntVal; break;
      case Type::LongTyID:   Idx = IdxGV.LongVal; break;
      case Type::UByteTyID:  Idx = IdxGV.UByteVal; break;
      case Type::UShortTyID: Idx = IdxGV.UShortVal; break;
      case Type::UIntTyID:   Idx = IdxGV.UIntVal; break;
      case Type::ULongTyID:  Idx = IdxGV.ULongVal; break;
      }
      Total += TD.getTypeSize(ST->getElementType())*Idx;
    }
  }

  GenericValue Result;
  Result.PointerVal = getOperandValue(Ptr, SF).PointerVal + Total;
  return Result;
}

void Interpreter::visitGetElementPtrInst(GetElementPtrInst &I) {
  ExecutionContext &SF = ECStack.back();
  SetValue(&I, TheEE->executeGEPOperation(I.getPointerOperand(),
                                   gep_type_begin(I), gep_type_end(I), SF), SF);
}

void Interpreter::visitLoadInst(LoadInst &I) {
  ExecutionContext &SF = ECStack.back();
  GenericValue SRC = getOperandValue(I.getPointerOperand(), SF);
  GenericValue *Ptr = (GenericValue*)GVTOP(SRC);
  GenericValue Result = LoadValueFromMemory(Ptr, I.getType());
  SetValue(&I, Result, SF);
}

void Interpreter::visitStoreInst(StoreInst &I) {
  ExecutionContext &SF = ECStack.back();
  GenericValue Val = getOperandValue(I.getOperand(0), SF);
  GenericValue SRC = getOperandValue(I.getPointerOperand(), SF);
  StoreValueToMemory(Val, (GenericValue *)GVTOP(SRC),
                     I.getOperand(0)->getType());
}

//===----------------------------------------------------------------------===//
//                 Miscellaneous Instruction Implementations
//===----------------------------------------------------------------------===//

void Interpreter::visitCallSite(CallSite CS) {
  ExecutionContext &SF = ECStack.back();

  // Check to see if this is an intrinsic function call...
  if (Function *F = CS.getCalledFunction())
   if (F->isExternal ())
    switch (F->getIntrinsicID()) {
    case Intrinsic::not_intrinsic:
      break;
    case Intrinsic::va_start: { // va_start
      GenericValue ArgIndex;
      ArgIndex.UIntPairVal.first = ECStack.size() - 1;
      ArgIndex.UIntPairVal.second = 0;
      SetValue(CS.getInstruction(), ArgIndex, SF);
      return;
    }
    case Intrinsic::va_end:    // va_end is a noop for the interpreter
      return;
    case Intrinsic::va_copy:   // va_copy: dest = src
      SetValue(CS.getInstruction(), getOperandValue(*CS.arg_begin(), SF), SF);
      return;
    default:
      // If it is an unknown intrinsic function, use the intrinsic lowering
      // class to transform it into hopefully tasty LLVM code.
      //
      Instruction *Prev = CS.getInstruction()->getPrev();
      BasicBlock *Parent = CS.getInstruction()->getParent();
      IL->LowerIntrinsicCall(cast<CallInst>(CS.getInstruction()));

      // Restore the CurInst pointer to the first instruction newly inserted, if
      // any.
      if (!Prev) {
        SF.CurInst = Parent->begin();
      } else {
        SF.CurInst = Prev;
        ++SF.CurInst;
      }
    }

  SF.Caller = CS;
  std::vector<GenericValue> ArgVals;
  const unsigned NumArgs = SF.Caller.arg_size();
  ArgVals.reserve(NumArgs);
  for (CallSite::arg_iterator i = SF.Caller.arg_begin(),
         e = SF.Caller.arg_end(); i != e; ++i) {
    Value *V = *i;
    ArgVals.push_back(getOperandValue(V, SF));
    // Promote all integral types whose size is < sizeof(int) into ints.  We do
    // this by zero or sign extending the value as appropriate according to the
    // source type.
    const Type *Ty = V->getType();
    if (Ty->isIntegral() && Ty->getPrimitiveSize() < 4) {
      if (Ty == Type::ShortTy)
	ArgVals.back().IntVal = ArgVals.back().ShortVal;
      else if (Ty == Type::UShortTy)
	ArgVals.back().UIntVal = ArgVals.back().UShortVal;
      else if (Ty == Type::SByteTy)
	ArgVals.back().IntVal = ArgVals.back().SByteVal;
      else if (Ty == Type::UByteTy)
	ArgVals.back().UIntVal = ArgVals.back().UByteVal;
      else if (Ty == Type::BoolTy)
	ArgVals.back().UIntVal = ArgVals.back().BoolVal;
      else
	assert(0 && "Unknown type!");
    }
  }

  // To handle indirect calls, we must get the pointer value from the argument 
  // and treat it as a function pointer.
  GenericValue SRC = getOperandValue(SF.Caller.getCalledValue(), SF);  
  callFunction((Function*)GVTOP(SRC), ArgVals);
}

#define IMPLEMENT_SHIFT(OP, TY) \
   case Type::TY##TyID: Dest.TY##Val = Src1.TY##Val OP Src2.UByteVal; break

static GenericValue executeShlInst(GenericValue Src1, GenericValue Src2,
                                   const Type *Ty) {
  GenericValue Dest;
  switch (Ty->getPrimitiveID()) {
    IMPLEMENT_SHIFT(<<, UByte);
    IMPLEMENT_SHIFT(<<, SByte);
    IMPLEMENT_SHIFT(<<, UShort);
    IMPLEMENT_SHIFT(<<, Short);
    IMPLEMENT_SHIFT(<<, UInt);
    IMPLEMENT_SHIFT(<<, Int);
    IMPLEMENT_SHIFT(<<, ULong);
    IMPLEMENT_SHIFT(<<, Long);
  default:
    std::cout << "Unhandled type for Shl instruction: " << *Ty << "\n";
  }
  return Dest;
}

static GenericValue executeShrInst(GenericValue Src1, GenericValue Src2,
                                   const Type *Ty) {
  GenericValue Dest;
  switch (Ty->getPrimitiveID()) {
    IMPLEMENT_SHIFT(>>, UByte);
    IMPLEMENT_SHIFT(>>, SByte);
    IMPLEMENT_SHIFT(>>, UShort);
    IMPLEMENT_SHIFT(>>, Short);
    IMPLEMENT_SHIFT(>>, UInt);
    IMPLEMENT_SHIFT(>>, Int);
    IMPLEMENT_SHIFT(>>, ULong);
    IMPLEMENT_SHIFT(>>, Long);
  default:
    std::cout << "Unhandled type for Shr instruction: " << *Ty << "\n";
    abort();
  }
  return Dest;
}

void Interpreter::visitShl(ShiftInst &I) {
  ExecutionContext &SF = ECStack.back();
  const Type *Ty    = I.getOperand(0)->getType();
  GenericValue Src1 = getOperandValue(I.getOperand(0), SF);
  GenericValue Src2 = getOperandValue(I.getOperand(1), SF);
  GenericValue Dest;
  Dest = executeShlInst (Src1, Src2, Ty);
  SetValue(&I, Dest, SF);
}

void Interpreter::visitShr(ShiftInst &I) {
  ExecutionContext &SF = ECStack.back();
  const Type *Ty    = I.getOperand(0)->getType();
  GenericValue Src1 = getOperandValue(I.getOperand(0), SF);
  GenericValue Src2 = getOperandValue(I.getOperand(1), SF);
  GenericValue Dest;
  Dest = executeShrInst (Src1, Src2, Ty);
  SetValue(&I, Dest, SF);
}

#define IMPLEMENT_CAST(DTY, DCTY, STY) \
   case Type::STY##TyID: Dest.DTY##Val = DCTY Src.STY##Val; break;

#define IMPLEMENT_CAST_CASE_START(DESTTY, DESTCTY)    \
  case Type::DESTTY##TyID:                      \
    switch (SrcTy->getPrimitiveID()) {          \
      IMPLEMENT_CAST(DESTTY, DESTCTY, Bool);    \
      IMPLEMENT_CAST(DESTTY, DESTCTY, UByte);   \
      IMPLEMENT_CAST(DESTTY, DESTCTY, SByte);   \
      IMPLEMENT_CAST(DESTTY, DESTCTY, UShort);  \
      IMPLEMENT_CAST(DESTTY, DESTCTY, Short);   \
      IMPLEMENT_CAST(DESTTY, DESTCTY, UInt);    \
      IMPLEMENT_CAST(DESTTY, DESTCTY, Int);     \
      IMPLEMENT_CAST(DESTTY, DESTCTY, ULong);   \
      IMPLEMENT_CAST(DESTTY, DESTCTY, Long);    \
      IMPLEMENT_CAST(DESTTY, DESTCTY, Pointer);

#define IMPLEMENT_CAST_CASE_FP_IMP(DESTTY, DESTCTY) \
      IMPLEMENT_CAST(DESTTY, DESTCTY, Float);   \
      IMPLEMENT_CAST(DESTTY, DESTCTY, Double)

#define IMPLEMENT_CAST_CASE_END()    \
    default: std::cout << "Unhandled cast: " << SrcTy << " to " << Ty << "\n"; \
      abort();                                  \
    }                                           \
    break

#define IMPLEMENT_CAST_CASE(DESTTY, DESTCTY) \
   IMPLEMENT_CAST_CASE_START(DESTTY, DESTCTY);   \
   IMPLEMENT_CAST_CASE_FP_IMP(DESTTY, DESTCTY); \
   IMPLEMENT_CAST_CASE_END()

GenericValue Interpreter::executeCastOperation(Value *SrcVal, const Type *Ty,
					       ExecutionContext &SF) {
  const Type *SrcTy = SrcVal->getType();
  GenericValue Dest, Src = getOperandValue(SrcVal, SF);

  switch (Ty->getPrimitiveID()) {
    IMPLEMENT_CAST_CASE(UByte  , (unsigned char));
    IMPLEMENT_CAST_CASE(SByte  , (  signed char));
    IMPLEMENT_CAST_CASE(UShort , (unsigned short));
    IMPLEMENT_CAST_CASE(Short  , (  signed short));
    IMPLEMENT_CAST_CASE(UInt   , (unsigned int ));
    IMPLEMENT_CAST_CASE(Int    , (  signed int ));
    IMPLEMENT_CAST_CASE(ULong  , (uint64_t));
    IMPLEMENT_CAST_CASE(Long   , ( int64_t));
    IMPLEMENT_CAST_CASE(Pointer, (PointerTy));
    IMPLEMENT_CAST_CASE(Float  , (float));
    IMPLEMENT_CAST_CASE(Double , (double));
    IMPLEMENT_CAST_CASE(Bool   , (bool));
  default:
    std::cout << "Unhandled dest type for cast instruction: " << *Ty << "\n";
    abort();
  }

  return Dest;
}

void Interpreter::visitCastInst(CastInst &I) {
  ExecutionContext &SF = ECStack.back();
  SetValue(&I, executeCastOperation(I.getOperand(0), I.getType(), SF), SF);
}

void Interpreter::visitVANextInst(VANextInst &I) {
  ExecutionContext &SF = ECStack.back();

  // Get the incoming valist parameter.  LLI treats the valist as a
  // (ec-stack-depth var-arg-index) pair.
  GenericValue VAList = getOperandValue(I.getOperand(0), SF);
  
  // Move the pointer to the next vararg.
  ++VAList.UIntPairVal.second;
  SetValue(&I, VAList, SF);
}

#define IMPLEMENT_VAARG(TY) \
   case Type::TY##TyID: Dest.TY##Val = Src.TY##Val; break

void Interpreter::visitVAArgInst(VAArgInst &I) {
  ExecutionContext &SF = ECStack.back();

  // Get the incoming valist parameter.  LLI treats the valist as a
  // (ec-stack-depth var-arg-index) pair.
  GenericValue VAList = getOperandValue(I.getOperand(0), SF);
  GenericValue Dest;
  GenericValue Src = ECStack[VAList.UIntPairVal.first]
	.VarArgs[VAList.UIntPairVal.second];
  const Type *Ty = I.getType();
  switch (Ty->getPrimitiveID()) {
    IMPLEMENT_VAARG(UByte);
    IMPLEMENT_VAARG(SByte);
    IMPLEMENT_VAARG(UShort);
    IMPLEMENT_VAARG(Short);
    IMPLEMENT_VAARG(UInt);
    IMPLEMENT_VAARG(Int);
    IMPLEMENT_VAARG(ULong);
    IMPLEMENT_VAARG(Long);
    IMPLEMENT_VAARG(Pointer);
    IMPLEMENT_VAARG(Float);
    IMPLEMENT_VAARG(Double);
    IMPLEMENT_VAARG(Bool);
  default:
    std::cout << "Unhandled dest type for vaarg instruction: " << *Ty << "\n";
    abort();
  }
  
  // Set the Value of this Instruction.
  SetValue(&I, Dest, SF);
}

//===----------------------------------------------------------------------===//
//                        Dispatch and Execution Code
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// callFunction - Execute the specified function...
//
void Interpreter::callFunction(Function *F,
                               const std::vector<GenericValue> &ArgVals) {
  assert((ECStack.empty() || ECStack.back().Caller.getInstruction() == 0 || 
	  ECStack.back().Caller.arg_size() == ArgVals.size()) &&
	 "Incorrect number of arguments passed into function call!");
  // Make a new stack frame... and fill it in.
  ECStack.push_back(ExecutionContext());
  ExecutionContext &StackFrame = ECStack.back();
  StackFrame.CurFunction = F;

  // Special handling for external functions.
  if (F->isExternal()) {
    GenericValue Result = callExternalFunction (F, ArgVals);
    // Simulate a 'ret' instruction of the appropriate type.
    popStackAndReturnValueToCaller (F->getReturnType (), Result);
    return;
  }

  // Get pointers to first LLVM BB & Instruction in function.
  StackFrame.CurBB     = F->begin();
  StackFrame.CurInst   = StackFrame.CurBB->begin();

  // Run through the function arguments and initialize their values...
  assert((ArgVals.size() == F->asize() ||
         (ArgVals.size() > F->asize() && F->getFunctionType()->isVarArg())) &&
         "Invalid number of values passed to function invocation!");

  // Handle non-varargs arguments...
  unsigned i = 0;
  for (Function::aiterator AI = F->abegin(), E = F->aend(); AI != E; ++AI, ++i)
    SetValue(AI, ArgVals[i], StackFrame);

  // Handle varargs arguments...
  StackFrame.VarArgs.assign(ArgVals.begin()+i, ArgVals.end());
}

void Interpreter::run() {
  while (!ECStack.empty()) {
    // Interpret a single instruction & increment the "PC".
    ExecutionContext &SF = ECStack.back();  // Current stack frame
    Instruction &I = *SF.CurInst++;         // Increment before execute
    
    // Track the number of dynamic instructions executed.
    ++NumDynamicInsts;

    DEBUG(std::cerr << "About to interpret: " << I);
    visit(I);   // Dispatch to one of the visit* methods...
  }
}
