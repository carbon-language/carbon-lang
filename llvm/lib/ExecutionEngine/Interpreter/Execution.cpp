//===-- Execution.cpp - Implement code to simulate the program ------------===//
// 
//  This file contains the actual instruction interpreter.
//
//===----------------------------------------------------------------------===//

#include "Interpreter.h"
#include "ExecutionAnnotations.h"
#include "llvm/Module.h"
#include "llvm/Instructions.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Constants.h"
#include "llvm/Assembly/Writer.h"
#include "Support/CommandLine.h"
#include "Support/Statistic.h"
#include <math.h>  // For fmod
#include <signal.h>
#include <setjmp.h>

Interpreter *TheEE = 0;

namespace {
  Statistic<> NumDynamicInsts("lli", "Number of dynamic instructions executed");

  cl::opt<bool>
  QuietMode("quiet", cl::desc("Do not emit any non-program output"),
	    cl::init(true));

  cl::alias 
  QuietModeA("q", cl::desc("Alias for -quiet"), cl::aliasopt(QuietMode));

  cl::opt<bool>
  ArrayChecksEnabled("array-checks", cl::desc("Enable array bound checks"));
}

// Create a TargetData structure to handle memory addressing and size/alignment
// computations
//
CachedWriter CW;     // Object to accelerate printing of LLVM

sigjmp_buf SignalRecoverBuffer;
static bool InInstruction = false;

extern "C" {
static void SigHandler(int Signal) {
  if (InInstruction)
    siglongjmp(SignalRecoverBuffer, Signal);
}
}

static void initializeSignalHandlers() {
  struct sigaction Action;
  Action.sa_handler = SigHandler;
  Action.sa_flags   = SA_SIGINFO;
  sigemptyset(&Action.sa_mask);
  sigaction(SIGSEGV, &Action, 0);
  sigaction(SIGBUS, &Action, 0);
  sigaction(SIGINT, &Action, 0);
  sigaction(SIGFPE, &Action, 0);
}


//===----------------------------------------------------------------------===//
//                     Value Manipulation code
//===----------------------------------------------------------------------===//

static unsigned getOperandSlot(Value *V) {
  SlotNumber *SN = (SlotNumber*)V->getAnnotation(SlotNumberAID);
  assert(SN && "Operand does not have a slot number annotation!");
  return SN->SlotNum;
}

// Operations used by constant expr implementations...
static GenericValue executeCastOperation(Value *Src, const Type *DestTy,
                                         ExecutionContext &SF);
static GenericValue executeAddInst(GenericValue Src1, GenericValue Src2, 
				   const Type *Ty);


GenericValue Interpreter::getOperandValue(Value *V, ExecutionContext &SF) {
  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(V)) {
    switch (CE->getOpcode()) {
    case Instruction::Cast:
      return executeCastOperation(CE->getOperand(0), CE->getType(), SF);
    case Instruction::GetElementPtr:
      return TheEE->executeGEPOperation(CE->getOperand(0), CE->op_begin()+1,
					CE->op_end(), SF);
    case Instruction::Add:
      return executeAddInst(getOperandValue(CE->getOperand(0), SF),
                            getOperandValue(CE->getOperand(1), SF),
                            CE->getType());
    default:
      std::cerr << "Unhandled ConstantExpr: " << CE << "\n";
      abort();
      return GenericValue();
    }
  } else if (Constant *CPV = dyn_cast<Constant>(V)) {
    return TheEE->getConstantValue(CPV);
  } else if (GlobalValue *GV = dyn_cast<GlobalValue>(V)) {
    return PTOGV(TheEE->getPointerToGlobal(GV));
  } else {
    unsigned TyP = V->getType()->getUniqueID();   // TypePlane for value
    unsigned OpSlot = getOperandSlot(V);
    assert(TyP < SF.Values.size() && 
           OpSlot < SF.Values[TyP].size() && "Value out of range!");
    return SF.Values[TyP][getOperandSlot(V)];
  }
}

static void SetValue(Value *V, GenericValue Val, ExecutionContext &SF) {
  unsigned TyP = V->getType()->getUniqueID();   // TypePlane for value

  //std::cout << "Setting value: " << &SF.Values[TyP][getOperandSlot(V)]<< "\n";
  SF.Values[TyP][getOperandSlot(V)] = Val;
}

//===----------------------------------------------------------------------===//
//                    Annotation Wrangling code
//===----------------------------------------------------------------------===//

void Interpreter::initializeExecutionEngine() {
  TheEE = this;
  AnnotationManager::registerAnnotationFactory(FunctionInfoAID,
                                               &FunctionInfo::Create);
  initializeSignalHandlers();
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
  if (!QuietMode) {
    std::cout << "Program returned ";
    print(Type::IntTy, GV);
    std::cout << " via 'void exit(int)'\n";
  }

  ExitCode = GV.SByteVal;
  ECStack.clear();
}

void Interpreter::visitReturnInst(ReturnInst &I) {
  ExecutionContext &SF = ECStack.back();
  const Type *RetTy = 0;
  GenericValue Result;

  // Save away the return value... (if we are not 'ret void')
  if (I.getNumOperands()) {
    RetTy  = I.getReturnValue()->getType();
    Result = getOperandValue(I.getReturnValue(), SF);
  }

  // Save previously executing meth
  const Function *M = ECStack.back().CurFunction;

  // Pop the current stack frame... this invalidates SF
  ECStack.pop_back();

  if (ECStack.empty()) {  // Finished main.  Put result into exit code...
    if (RetTy) {          // Nonvoid return type?
      if (!QuietMode) {
        CW << "Function " << M->getType() << " \"" << M->getName()
           << "\" returned ";
        print(RetTy, Result);
        std::cout << "\n";
      }

      if (RetTy->isIntegral())
	ExitCode = Result.IntVal;   // Capture the exit code of the program
    } else {
      ExitCode = 0;
    }
    return;
  }

  // If we have a previous stack frame, and we have a previous call, fill in
  // the return value...
  //
  ExecutionContext &NewSF = ECStack.back();
  if (NewSF.Caller) {
    if (NewSF.Caller->getType() != Type::VoidTy)             // Save result...
      SetValue(NewSF.Caller, Result, NewSF);

    NewSF.Caller = 0;          // We returned from the call...
  } else if (!QuietMode) {
    // This must be a function that is executing because of a user 'call'
    // instruction.
    CW << "Function " << M->getType() << " \"" << M->getName()
       << "\" returned ";
    print(RetTy, Result);
    std::cout << "\n";
  }
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
    if (Trace) CW << "Run:" << PN;

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
  // FIXME: Don't use CALLOC, use a tainted malloc.
  void *Memory = calloc(NumElements, TD.getTypeSize(Ty));

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
GenericValue Interpreter::executeGEPOperation(Value *Ptr, User::op_iterator I,
					      User::op_iterator E,
					      ExecutionContext &SF) {
  assert(isa<PointerType>(Ptr->getType()) &&
         "Cannot getElementOffset of a nonpointer type!");

  PointerTy Total = 0;
  const Type *Ty = Ptr->getType();

  for (; I != E; ++I) {
    if (const StructType *STy = dyn_cast<StructType>(Ty)) {
      const StructLayout *SLO = TD.getStructLayout(STy);
      
      // Indicies must be ubyte constants...
      const ConstantUInt *CPU = cast<ConstantUInt>(*I);
      assert(CPU->getType() == Type::UByteTy);
      unsigned Index = CPU->getValue();
      
      Total += SLO->MemberOffsets[Index];
      Ty = STy->getElementTypes()[Index];
    } else if (const SequentialType *ST = cast<SequentialType>(Ty)) {

      // Get the index number for the array... which must be long type...
      assert((*I)->getType() == Type::LongTy);
      unsigned Idx = getOperandValue(*I, SF).LongVal;
      if (const ArrayType *AT = dyn_cast<ArrayType>(ST))
        if (Idx >= AT->getNumElements() && ArrayChecksEnabled) {
          std::cerr << "Out of range memory access to element #" << Idx
                    << " of a " << AT->getNumElements() << " element array."
                    << " Subscript #" << *I << "\n";
          // Get outta here!!!
          siglongjmp(SignalRecoverBuffer, SIGTRAP);
        }

      Ty = ST->getElementType();
      unsigned Size = TD.getTypeSize(Ty);
      Total += Size*Idx;
    }  
  }

  GenericValue Result;
  Result.PointerVal = getOperandValue(Ptr, SF).PointerVal + Total;
  return Result;
}

void Interpreter::visitGetElementPtrInst(GetElementPtrInst &I) {
  ExecutionContext &SF = ECStack.back();
  SetValue(&I, TheEE->executeGEPOperation(I.getPointerOperand(),
                                   I.idx_begin(), I.idx_end(), SF), SF);
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

void Interpreter::visitCallInst(CallInst &I) {
  ExecutionContext &SF = ECStack.back();
  SF.Caller = &I;
  std::vector<GenericValue> ArgVals;
  ArgVals.reserve(I.getNumOperands()-1);
  for (unsigned i = 1; i < I.getNumOperands(); ++i) {
    ArgVals.push_back(getOperandValue(I.getOperand(i), SF));
    // Promote all integral types whose size is < sizeof(int) into ints.  We do
    // this by zero or sign extending the value as appropriate according to the
    // source type.
    if (I.getOperand(i)->getType()->isIntegral() &&
	I.getOperand(i)->getType()->getPrimitiveSize() < 4) {
      const Type *Ty = I.getOperand(i)->getType();
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
  GenericValue SRC = getOperandValue(I.getCalledValue(), SF);  
  callFunction((Function*)GVTOP(SRC), ArgVals);
}

#define IMPLEMENT_SHIFT(OP, TY) \
   case Type::TY##TyID: Dest.TY##Val = Src1.TY##Val OP Src2.UByteVal; break

void Interpreter::visitShl(ShiftInst &I) {
  ExecutionContext &SF = ECStack.back();
  const Type *Ty    = I.getOperand(0)->getType();
  GenericValue Src1 = getOperandValue(I.getOperand(0), SF);
  GenericValue Src2 = getOperandValue(I.getOperand(1), SF);
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
  SetValue(&I, Dest, SF);
}

void Interpreter::visitShr(ShiftInst &I) {
  ExecutionContext &SF = ECStack.back();
  const Type *Ty    = I.getOperand(0)->getType();
  GenericValue Src1 = getOperandValue(I.getOperand(0), SF);
  GenericValue Src2 = getOperandValue(I.getOperand(1), SF);
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

void Interpreter::visitVarArgInst(VarArgInst &I) {
  ExecutionContext &SF = ECStack.back();

  // Get the pointer to the valist element.  LLI treats the valist in memory as
  // an integer.
  GenericValue VAListPtr = getOperandValue(I.getOperand(0), SF);

  // Load the pointer
  GenericValue VAList = 
    TheEE->LoadValueFromMemory((GenericValue *)GVTOP(VAListPtr), Type::UIntTy);

  unsigned Argument = VAList.IntVal++;

  // Update the valist to point to the next argument...
  TheEE->StoreValueToMemory(VAList, (GenericValue *)GVTOP(VAListPtr),
                            Type::UIntTy);

  // Set the value...
  assert(Argument < SF.VarArgs.size() &&
         "Accessing past the last vararg argument!");
  SetValue(&I, SF.VarArgs[Argument], SF);
}

//===----------------------------------------------------------------------===//
//                        Dispatch and Execution Code
//===----------------------------------------------------------------------===//

FunctionInfo::FunctionInfo(Function *F) : Annotation(FunctionInfoAID) {
  // Assign slot numbers to the function arguments...
  for (Function::const_aiterator AI = F->abegin(), E = F->aend(); AI != E; ++AI)
    AI->addAnnotation(new SlotNumber(getValueSlot(AI)));

  // Iterate over all of the instructions...
  unsigned InstNum = 0;
  for (Function::iterator BB = F->begin(), BBE = F->end(); BB != BBE; ++BB)
    for (BasicBlock::iterator II = BB->begin(), IE = BB->end(); II != IE; ++II)
      // For each instruction... Add Annote
      II->addAnnotation(new InstNumber(++InstNum, getValueSlot(II)));
}

unsigned FunctionInfo::getValueSlot(const Value *V) {
  unsigned Plane = V->getType()->getUniqueID();
  if (Plane >= NumPlaneElements.size())
    NumPlaneElements.resize(Plane+1, 0);
  return NumPlaneElements[Plane]++;
}


//===----------------------------------------------------------------------===//
// callFunction - Execute the specified function...
//
void Interpreter::callFunction(Function *F,
                               const std::vector<GenericValue> &ArgVals) {
  assert((ECStack.empty() || ECStack.back().Caller == 0 || 
	  ECStack.back().Caller->getNumOperands()-1 == ArgVals.size()) &&
	 "Incorrect number of arguments passed into function call!");
  if (F->isExternal()) {
    GenericValue Result = callExternalFunction(F, ArgVals);
    const Type *RetTy = F->getReturnType();

    // Copy the result back into the result variable if we are not returning
    // void.
    if (RetTy != Type::VoidTy) {
      if (!ECStack.empty() && ECStack.back().Caller) {
        ExecutionContext &SF = ECStack.back();
        SetValue(SF.Caller, Result, SF);
      
        SF.Caller = 0;          // We returned from the call...
      } else if (!QuietMode) {
        // print it.
        CW << "Function " << F->getType() << " \"" << F->getName()
           << "\" returned ";
        print(RetTy, Result); 
        std::cout << "\n";
        
        if (RetTy->isIntegral())
          ExitCode = Result.IntVal;   // Capture the exit code of the program
      }
    }

    return;
  }

  // Process the function, assigning instruction numbers to the instructions in
  // the function.  Also calculate the number of values for each type slot
  // active.
  //
  FunctionInfo *FuncInfo =
    (FunctionInfo*)F->getOrCreateAnnotation(FunctionInfoAID);
  ECStack.push_back(ExecutionContext());         // Make a new stack frame...

  ExecutionContext &StackFrame = ECStack.back(); // Fill it in...
  StackFrame.CurFunction = F;
  StackFrame.CurBB     = F->begin();
  StackFrame.CurInst   = StackFrame.CurBB->begin();
  StackFrame.FuncInfo  = FuncInfo;

  // Initialize the values to nothing...
  StackFrame.Values.resize(FuncInfo->NumPlaneElements.size());
  for (unsigned i = 0; i < FuncInfo->NumPlaneElements.size(); ++i) {
    StackFrame.Values[i].resize(FuncInfo->NumPlaneElements[i]);

    // Taint the initial values of stuff
    memset(&StackFrame.Values[i][0], 42,
           FuncInfo->NumPlaneElements[i]*sizeof(GenericValue));
  }


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

// executeInstruction - Interpret a single instruction & increment the "PC".
//
void Interpreter::executeInstruction() {
  assert(!ECStack.empty() && "No program running, cannot execute inst!");

  ExecutionContext &SF = ECStack.back();  // Current stack frame
  Instruction &I = *SF.CurInst++;         // Increment before execute

  if (Trace) CW << "Run:" << I;

  // Track the number of dynamic instructions executed.
  ++NumDynamicInsts;

  // Set a sigsetjmp buffer so that we can recover if an error happens during
  // instruction execution...
  //
  if (int SigNo = sigsetjmp(SignalRecoverBuffer, 1)) {
    std::cout << "EXCEPTION OCCURRED [" << strsignal(SigNo) << "]\n";
    exit(1);
  }

  InInstruction = true;
  visit(I);   // Dispatch to one of the visit* methods...
  InInstruction = false;
  
  // Reset the current frame location to the top of stack
  CurFrame = ECStack.size()-1;
}

void Interpreter::run() {
  while (!ECStack.empty()) {
    // Run an instruction...
    executeInstruction();
  }
}

void Interpreter::printValue(const Type *Ty, GenericValue V) {
  switch (Ty->getPrimitiveID()) {
  case Type::BoolTyID:   std::cout << (V.BoolVal?"true":"false"); break;
  case Type::SByteTyID:
    std::cout << (int)V.SByteVal << " '" << V.SByteVal << "'";  break;
  case Type::UByteTyID:
    std::cout << (unsigned)V.UByteVal << " '" << V.UByteVal << "'";  break;
  case Type::ShortTyID:  std::cout << V.ShortVal;  break;
  case Type::UShortTyID: std::cout << V.UShortVal; break;
  case Type::IntTyID:    std::cout << V.IntVal;    break;
  case Type::UIntTyID:   std::cout << V.UIntVal;   break;
  case Type::LongTyID:   std::cout << (long)V.LongVal;   break;
  case Type::ULongTyID:  std::cout << (unsigned long)V.ULongVal;  break;
  case Type::FloatTyID:  std::cout << V.FloatVal;  break;
  case Type::DoubleTyID: std::cout << V.DoubleVal; break;
  case Type::PointerTyID:std::cout << (void*)GVTOP(V); break;
  default:
    std::cout << "- Don't know how to print value of this type!";
    break;
  }
}

void Interpreter::print(const Type *Ty, GenericValue V) {
  CW << Ty << " ";
  printValue(Ty, V);
}
