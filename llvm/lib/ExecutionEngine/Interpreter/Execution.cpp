//===-- Execution.cpp - Implement code to simulate the program ------------===//
// 
//  This file contains the actual instruction interpreter.
//
//===----------------------------------------------------------------------===//

#include "Interpreter.h"
#include "ExecutionAnnotations.h"
#include "llvm/iOther.h"
#include "llvm/iTerminators.h"
#include "llvm/iMemory.h"
#include "llvm/Type.h"
#include "llvm/ConstPoolVals.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Target/TargetData.h"

static unsigned getOperandSlot(Value *V) {
  SlotNumber *SN = (SlotNumber*)V->getAnnotation(SlotNumberAID);
  assert(SN && "Operand does not have a slot number annotation!");
  return SN->SlotNum;
}

#define GET_CONST_VAL(TY, CLASS) \
  case Type::TY##TyID: Result.TY##Val = ((CLASS*)CPV)->getValue(); break

static GenericValue getOperandValue(Value *V, ExecutionContext &SF) {
  if (ConstPoolVal *CPV = V->castConstant()) {
    GenericValue Result;
    switch (CPV->getType()->getPrimitiveID()) {
      GET_CONST_VAL(Bool   , ConstPoolBool);
      GET_CONST_VAL(UByte  , ConstPoolUInt);
      GET_CONST_VAL(SByte  , ConstPoolSInt);
      GET_CONST_VAL(UShort , ConstPoolUInt);
      GET_CONST_VAL(Short  , ConstPoolSInt);
      GET_CONST_VAL(UInt   , ConstPoolUInt);
      GET_CONST_VAL(Int    , ConstPoolSInt);
      GET_CONST_VAL(Float  , ConstPoolFP);
      GET_CONST_VAL(Double , ConstPoolFP);
    default:
      cout << "ERROR: Constant unimp for type: " << CPV->getType() << endl;
    }
    return Result;
  } else {
    unsigned TyP = V->getType()->getUniqueID();   // TypePlane for value
    return SF.Values[TyP][getOperandSlot(V)];
  }
}

static void printOperandInfo(Value *V, ExecutionContext &SF) {
  if (!V->isConstant()) {
    unsigned TyP  = V->getType()->getUniqueID();   // TypePlane for value
    unsigned Slot = getOperandSlot(V);
    cout << "Value=" << (void*)V << " TypeID=" << TyP << " Slot=" << Slot
	 << " Addr=" << &SF.Values[TyP][Slot] << " SF=" << &SF << endl;
  }
}



static void SetValue(Value *V, GenericValue Val, ExecutionContext &SF) {
  unsigned TyP = V->getType()->getUniqueID();   // TypePlane for value

  //cout << "Setting value: " << &SF.Values[TyP][getOperandSlot(V)] << endl;
  SF.Values[TyP][getOperandSlot(V)] = Val;
}



//===----------------------------------------------------------------------===//
//                    Binary Instruction Implementations
//===----------------------------------------------------------------------===//

#define IMPLEMENT_BINARY_OPERATOR(OP, TY) \
   case Type::TY##TyID: Dest.TY##Val = Src1.TY##Val OP Src2.TY##Val; break
#define IMPLEMENT_BINARY_PTR_OPERATOR(OP) \
   case Type::PointerTyID: Dest.PointerVal = \
     (GenericValue*)((unsigned long)Src1.PointerVal OP (unsigned long)Src2.PointerVal); break

static GenericValue executeAddInst(GenericValue Src1, GenericValue Src2, 
				   const Type *Ty, ExecutionContext &SF) {
  GenericValue Dest;
  switch (Ty->getPrimitiveID()) {
    IMPLEMENT_BINARY_OPERATOR(+, UByte);
    IMPLEMENT_BINARY_OPERATOR(+, SByte);
    IMPLEMENT_BINARY_OPERATOR(+, UShort);
    IMPLEMENT_BINARY_OPERATOR(+, Short);
    IMPLEMENT_BINARY_OPERATOR(+, UInt);
    IMPLEMENT_BINARY_OPERATOR(+, Int);
    IMPLEMENT_BINARY_OPERATOR(+, Float);
    IMPLEMENT_BINARY_OPERATOR(+, Double);
    IMPLEMENT_BINARY_PTR_OPERATOR(+);
  case Type::ULongTyID:
  case Type::LongTyID:
  default:
    cout << "Unhandled type for Add instruction: " << Ty << endl;
  }
  return Dest;
}

static GenericValue executeSubInst(GenericValue Src1, GenericValue Src2, 
				   const Type *Ty, ExecutionContext &SF) {
  GenericValue Dest;
  switch (Ty->getPrimitiveID()) {
    IMPLEMENT_BINARY_OPERATOR(-, UByte);
    IMPLEMENT_BINARY_OPERATOR(-, SByte);
    IMPLEMENT_BINARY_OPERATOR(-, UShort);
    IMPLEMENT_BINARY_OPERATOR(-, Short);
    IMPLEMENT_BINARY_OPERATOR(-, UInt);
    IMPLEMENT_BINARY_OPERATOR(-, Int);
    IMPLEMENT_BINARY_OPERATOR(-, Float);
    IMPLEMENT_BINARY_OPERATOR(-, Double);
    IMPLEMENT_BINARY_PTR_OPERATOR(-);
  case Type::ULongTyID:
  case Type::LongTyID:
  default:
    cout << "Unhandled type for Sub instruction: " << Ty << endl;
  }
  return Dest;
}

#define IMPLEMENT_SETCC(OP, TY) \
   case Type::TY##TyID: Dest.BoolVal = Src1.TY##Val OP Src2.TY##Val; break

static GenericValue executeSetEQInst(GenericValue Src1, GenericValue Src2, 
				     const Type *Ty, ExecutionContext &SF) {
  GenericValue Dest;
  switch (Ty->getPrimitiveID()) {
    IMPLEMENT_SETCC(==, UByte);
    IMPLEMENT_SETCC(==, SByte);
    IMPLEMENT_SETCC(==, UShort);
    IMPLEMENT_SETCC(==, Short);
    IMPLEMENT_SETCC(==, UInt);
    IMPLEMENT_SETCC(==, Int);
    IMPLEMENT_SETCC(==, Float);
    IMPLEMENT_SETCC(==, Double);
    IMPLEMENT_SETCC(==, Pointer);
  case Type::ULongTyID:
  case Type::LongTyID:
  default:
    cout << "Unhandled type for SetEQ instruction: " << Ty << endl;
  }
  return Dest;
}

static GenericValue executeSetNEInst(GenericValue Src1, GenericValue Src2, 
				     const Type *Ty, ExecutionContext &SF) {
  GenericValue Dest;
  switch (Ty->getPrimitiveID()) {
    IMPLEMENT_SETCC(!=, UByte);
    IMPLEMENT_SETCC(!=, SByte);
    IMPLEMENT_SETCC(!=, UShort);
    IMPLEMENT_SETCC(!=, Short);
    IMPLEMENT_SETCC(!=, UInt);
    IMPLEMENT_SETCC(!=, Int);
    IMPLEMENT_SETCC(!=, Float);
    IMPLEMENT_SETCC(!=, Double);
    IMPLEMENT_SETCC(!=, Pointer);
  case Type::ULongTyID:
  case Type::LongTyID:
  default:
    cout << "Unhandled type for SetNE instruction: " << Ty << endl;
  }
  return Dest;
}

static GenericValue executeSetLEInst(GenericValue Src1, GenericValue Src2, 
				     const Type *Ty, ExecutionContext &SF) {
  GenericValue Dest;
  switch (Ty->getPrimitiveID()) {
    IMPLEMENT_SETCC(<=, UByte);
    IMPLEMENT_SETCC(<=, SByte);
    IMPLEMENT_SETCC(<=, UShort);
    IMPLEMENT_SETCC(<=, Short);
    IMPLEMENT_SETCC(<=, UInt);
    IMPLEMENT_SETCC(<=, Int);
    IMPLEMENT_SETCC(<=, Float);
    IMPLEMENT_SETCC(<=, Double);
    IMPLEMENT_SETCC(<=, Pointer);
  case Type::ULongTyID:
  case Type::LongTyID:
  default:
    cout << "Unhandled type for SetLE instruction: " << Ty << endl;
  }
  return Dest;
}

static GenericValue executeSetGEInst(GenericValue Src1, GenericValue Src2, 
				     const Type *Ty, ExecutionContext &SF) {
  GenericValue Dest;
  switch (Ty->getPrimitiveID()) {
    IMPLEMENT_SETCC(>=, UByte);
    IMPLEMENT_SETCC(>=, SByte);
    IMPLEMENT_SETCC(>=, UShort);
    IMPLEMENT_SETCC(>=, Short);
    IMPLEMENT_SETCC(>=, UInt);
    IMPLEMENT_SETCC(>=, Int);
    IMPLEMENT_SETCC(>=, Float);
    IMPLEMENT_SETCC(>=, Double);
    IMPLEMENT_SETCC(>=, Pointer);
  case Type::ULongTyID:
  case Type::LongTyID:
  default:
    cout << "Unhandled type for SetGE instruction: " << Ty << endl;
  }
  return Dest;
}

static GenericValue executeSetLTInst(GenericValue Src1, GenericValue Src2, 
				     const Type *Ty, ExecutionContext &SF) {
  GenericValue Dest;
  switch (Ty->getPrimitiveID()) {
    IMPLEMENT_SETCC(<, UByte);
    IMPLEMENT_SETCC(<, SByte);
    IMPLEMENT_SETCC(<, UShort);
    IMPLEMENT_SETCC(<, Short);
    IMPLEMENT_SETCC(<, UInt);
    IMPLEMENT_SETCC(<, Int);
    IMPLEMENT_SETCC(<, Float);
    IMPLEMENT_SETCC(<, Double);
    IMPLEMENT_SETCC(<, Pointer);
  case Type::ULongTyID:
  case Type::LongTyID:
  default:
    cout << "Unhandled type for SetLT instruction: " << Ty << endl;
  }
  return Dest;
}

static GenericValue executeSetGTInst(GenericValue Src1, GenericValue Src2, 
				     const Type *Ty, ExecutionContext &SF) {
  GenericValue Dest;
  switch (Ty->getPrimitiveID()) {
    IMPLEMENT_SETCC(>, UByte);
    IMPLEMENT_SETCC(>, SByte);
    IMPLEMENT_SETCC(>, UShort);
    IMPLEMENT_SETCC(>, Short);
    IMPLEMENT_SETCC(>, UInt);
    IMPLEMENT_SETCC(>, Int);
    IMPLEMENT_SETCC(>, Float);
    IMPLEMENT_SETCC(>, Double);
    IMPLEMENT_SETCC(>, Pointer);
  case Type::ULongTyID:
  case Type::LongTyID:
  default:
    cout << "Unhandled type for SetGT instruction: " << Ty << endl;
  }
  return Dest;
}

static void executeBinaryInst(BinaryOperator *I, ExecutionContext &SF) {
  const Type *Ty = I->getOperand(0)->getType();
  GenericValue Src1  = getOperandValue(I->getOperand(0), SF);
  GenericValue Src2  = getOperandValue(I->getOperand(1), SF);
  GenericValue R;   // Result

  switch (I->getOpcode()) {
  case Instruction::Add: R = executeAddInst(Src1, Src2, Ty, SF); break;
  case Instruction::Sub: R = executeSubInst(Src1, Src2, Ty, SF); break;
  case Instruction::SetEQ: R = executeSetEQInst(Src1, Src2, Ty, SF); break;
  case Instruction::SetNE: R = executeSetNEInst(Src1, Src2, Ty, SF); break;
  case Instruction::SetLE: R = executeSetLEInst(Src1, Src2, Ty, SF); break;
  case Instruction::SetGE: R = executeSetGEInst(Src1, Src2, Ty, SF); break;
  case Instruction::SetLT: R = executeSetLTInst(Src1, Src2, Ty, SF); break;
  case Instruction::SetGT: R = executeSetGTInst(Src1, Src2, Ty, SF); break;
  default:
    cout << "Don't know how to handle this binary operator!\n-->" << I;
  }

  SetValue(I, R, SF);
}

//===----------------------------------------------------------------------===//
//                     Terminator Instruction Implementations
//===----------------------------------------------------------------------===//

void Interpreter::executeRetInst(ReturnInst *I, ExecutionContext &SF) {
  const Type *RetTy = 0;
  GenericValue Result;

  // Save away the return value... (if we are not 'ret void')
  if (I->getNumOperands()) {
    RetTy  = I->getReturnValue()->getType();
    Result = getOperandValue(I->getReturnValue(), SF);
  }

  // Save previously executing meth
  const Method *M = ECStack.back().CurMethod;

  // Pop the current stack frame... this invalidates SF
  ECStack.pop_back();

  if (ECStack.empty()) {  // Finished main.  Put result into exit code...
    if (RetTy) {          // Nonvoid return type?
      cout << "Method " << M->getType() << " \"" << M->getName()
	   << "\" returned ";
      printValue(RetTy, Result);
      cout << endl;

      if (RetTy->isIntegral())
	ExitCode = Result.SByteVal;   // Capture the exit code of the program
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
  } else {
    // This must be a function that is executing because of a user 'call'
    // instruction.
    cout << "Method " << M->getType() << " \"" << M->getName()
	 << "\" returned ";
    printValue(RetTy, Result);
    cout << endl;
  }
}

void Interpreter::executeBrInst(BranchInst *I, ExecutionContext &SF) {
  SF.PrevBB = SF.CurBB;               // Update PrevBB so that PHI nodes work...
  BasicBlock *Dest;

  Dest = I->getSuccessor(0);          // Uncond branches have a fixed dest...
  if (!I->isUnconditional()) {
    if (getOperandValue(I->getCondition(), SF).BoolVal == 0) // If false cond...
      Dest = I->getSuccessor(1);    
  }
  SF.CurBB   = Dest;                  // Update CurBB to branch destination
  SF.CurInst = SF.CurBB->begin();     // Update new instruction ptr...
}

//===----------------------------------------------------------------------===//
//                     Memory Instruction Implementations
//===----------------------------------------------------------------------===//

// Create a TargetData structure to handle memory addressing and size/alignment
// computations
//
static TargetData TD("lli Interpreter");

void Interpreter::executeAllocInst(AllocationInst *I, ExecutionContext &SF) {
  const Type *Ty = I->getType()->getValueType();  // Type to be allocated
  unsigned NumElements = 1;

  if (I->getNumOperands()) {   // Allocating a unsized array type?
    assert(Ty->isArrayType() && Ty->castArrayType()->isUnsized() && 
	   "Allocation inst with size operand for !unsized array type???");
    Ty = ((const ArrayType*)Ty)->getElementType();  // Get the actual type...

    // Get the number of elements being allocated by the array...
    GenericValue NumEl = getOperandValue(I->getOperand(0), SF);
    NumElements = NumEl.UIntVal;
  }

  // Allocate enough memory to hold the type...
  GenericValue Result;
  Result.PointerVal = (GenericValue*)malloc(NumElements * TD.getTypeSize(Ty));
  assert(Result.PointerVal != 0 && "Null pointer returned by malloc!");
  SetValue(I, Result, SF);

  if (I->getOpcode() == Instruction::Alloca) {
    // Keep track to free it later...
  }
}

static void executeFreeInst(FreeInst *I, ExecutionContext &SF) {
  assert(I->getOperand(0)->getType()->isPointerType() && "Freeing nonptr?");
  GenericValue Value = getOperandValue(I->getOperand(0), SF);
  // TODO: Check to make sure memory is allocated
  free(Value.PointerVal);   // Free memory
}

static void executeLoadInst(LoadInst *I, ExecutionContext &SF) {
  assert(I->getNumOperands() == 1 && "NI!");
  GenericValue *Ptr = getOperandValue(I->getPtrOperand(), SF).PointerVal;
  GenericValue Result;

  switch (I->getType()->getPrimitiveID()) {
  case Type::BoolTyID:
  case Type::UByteTyID:
  case Type::SByteTyID:   Result.SByteVal = Ptr->SByteVal; break;
  case Type::UShortTyID:
  case Type::ShortTyID:   Result.ShortVal = Ptr->ShortVal; break;
  case Type::UIntTyID:
  case Type::IntTyID:     Result.IntVal = Ptr->IntVal; break;
    //case Type::ULongTyID:
    //case Type::LongTyID:    Result.LongVal = Ptr->LongVal; break;
  case Type::FloatTyID:   Result.FloatVal = Ptr->FloatVal; break;
  case Type::DoubleTyID:  Result.DoubleVal = Ptr->DoubleVal; break;
  case Type::PointerTyID: Result.PointerVal = Ptr->PointerVal; break;
  default:
    cout << "Cannot load value of type " << I->getType() << "!\n";
  }

  SetValue(I, Result, SF);
}

static void executeStoreInst(StoreInst *I, ExecutionContext &SF) {
  GenericValue *Ptr = getOperandValue(I->getPtrOperand(), SF).PointerVal;
  GenericValue Val = getOperandValue(I->getOperand(0), SF);
  assert(I->getNumOperands() == 2 && "NI!");

  switch (I->getOperand(0)->getType()->getPrimitiveID()) {
  case Type::BoolTyID:
  case Type::UByteTyID:
  case Type::SByteTyID:   Ptr->SByteVal = Val.SByteVal; break;
  case Type::UShortTyID:
  case Type::ShortTyID:   Ptr->ShortVal = Val.ShortVal; break;
  case Type::UIntTyID:
  case Type::IntTyID:     Ptr->IntVal = Val.IntVal; break;
    //case Type::ULongTyID:
    //case Type::LongTyID:    Ptr->LongVal = Val.LongVal; break;
  case Type::FloatTyID:   Ptr->FloatVal = Val.FloatVal; break;
  case Type::DoubleTyID:  Ptr->DoubleVal = Val.DoubleVal; break;
  case Type::PointerTyID: Ptr->PointerVal = Val.PointerVal; break;
  default:
    cout << "Cannot store value of type " << I->getType() << "!\n";
  }
}


//===----------------------------------------------------------------------===//
//                 Miscellaneous Instruction Implementations
//===----------------------------------------------------------------------===//

void Interpreter::executeCallInst(CallInst *I, ExecutionContext &SF) {
  ECStack.back().Caller = I;
  vector<GenericValue> ArgVals;
  ArgVals.reserve(I->getNumOperands()-1);
  for (unsigned i = 1; i < I->getNumOperands(); ++i)
    ArgVals.push_back(getOperandValue(I->getOperand(i), SF));

  callMethod(I->getCalledMethod(), ArgVals);
}

static void executePHINode(PHINode *I, ExecutionContext &SF) {
  BasicBlock *PrevBB = SF.PrevBB;
  Value *IncomingValue = 0;

  // Search for the value corresponding to this previous bb...
  for (unsigned i = I->getNumIncomingValues(); i > 0;) {
    if (I->getIncomingBlock(--i) == PrevBB) {
      IncomingValue = I->getIncomingValue(i);
      break;
    }
  }
  assert(IncomingValue && "No PHI node predecessor for current PrevBB!");

  // Found the value, set as the result...
  SetValue(I, getOperandValue(IncomingValue, SF), SF);
}

#define IMPLEMENT_SHIFT(OP, TY) \
   case Type::TY##TyID: Dest.TY##Val = Src1.TY##Val OP Src2.UByteVal; break

static void executeShlInst(ShiftInst *I, ExecutionContext &SF) {
  const Type *Ty = I->getOperand(0)->getType();
  GenericValue Src1  = getOperandValue(I->getOperand(0), SF);
  GenericValue Src2  = getOperandValue(I->getOperand(1), SF);
  GenericValue Dest;

  switch (Ty->getPrimitiveID()) {
    IMPLEMENT_SHIFT(<<, UByte);
    IMPLEMENT_SHIFT(<<, SByte);
    IMPLEMENT_SHIFT(<<, UShort);
    IMPLEMENT_SHIFT(<<, Short);
    IMPLEMENT_SHIFT(<<, UInt);
    IMPLEMENT_SHIFT(<<, Int);
  case Type::ULongTyID:
  case Type::LongTyID:
  default:
    cout << "Unhandled type for Shl instruction: " << Ty << endl;
  }
  SetValue(I, Dest, SF);
}

static void executeShrInst(ShiftInst *I, ExecutionContext &SF) {
  const Type *Ty = I->getOperand(0)->getType();
  GenericValue Src1  = getOperandValue(I->getOperand(0), SF);
  GenericValue Src2  = getOperandValue(I->getOperand(1), SF);
  GenericValue Dest;

  switch (Ty->getPrimitiveID()) {
    IMPLEMENT_SHIFT(>>, UByte);
    IMPLEMENT_SHIFT(>>, SByte);
    IMPLEMENT_SHIFT(>>, UShort);
    IMPLEMENT_SHIFT(>>, Short);
    IMPLEMENT_SHIFT(>>, UInt);
    IMPLEMENT_SHIFT(>>, Int);
  case Type::ULongTyID:
  case Type::LongTyID:
  default:
    cout << "Unhandled type for Shr instruction: " << Ty << endl;
  }
  SetValue(I, Dest, SF);
}

#define IMPLEMENT_CAST(DTY, DCTY, STY) \
   case Type::STY##TyID: Dest.DTY##Val = (DCTY)Src.STY##Val; break;

#define IMPLEMENT_CAST_CASE_START(DESTTY, DESTCTY)    \
  case Type::DESTTY##TyID:                      \
    switch (SrcTy->getPrimitiveID()) {          \
      IMPLEMENT_CAST(DESTTY, DESTCTY, UByte);   \
      IMPLEMENT_CAST(DESTTY, DESTCTY, SByte);   \
      IMPLEMENT_CAST(DESTTY, DESTCTY, UShort);  \
      IMPLEMENT_CAST(DESTTY, DESTCTY, Short);   \
      IMPLEMENT_CAST(DESTTY, DESTCTY, UInt);    \
      IMPLEMENT_CAST(DESTTY, DESTCTY, Int);

#define IMPLEMENT_CAST_CASE_PTR_IMP(DESTTY, DESTCTY) \
      IMPLEMENT_CAST(DESTTY, DESTCTY, Pointer)

#define IMPLEMENT_CAST_CASE_FP_IMP(DESTTY, DESTCTY) \
      IMPLEMENT_CAST(DESTTY, DESTCTY, Float);   \
      IMPLEMENT_CAST(DESTTY, DESTCTY, Double)

#define IMPLEMENT_CAST_CASE_END()    \
    default: cout << "Unhandled cast: " << SrcTy << " to " << Ty << endl;  \
      break;                                    \
    }                                           \
    break

#define IMPLEMENT_CAST_CASE(DESTTY, DESTCTY) \
   IMPLEMENT_CAST_CASE_START(DESTTY, DESTCTY);   \
   IMPLEMENT_CAST_CASE_FP_IMP(DESTTY, DESTCTY); \
   IMPLEMENT_CAST_CASE_PTR_IMP(DESTTY, DESTCTY); \
   IMPLEMENT_CAST_CASE_END()

#define IMPLEMENT_CAST_CASE_FP(DESTTY, DESTCTY) \
   IMPLEMENT_CAST_CASE_START(DESTTY, DESTCTY);   \
   IMPLEMENT_CAST_CASE_FP_IMP(DESTTY, DESTCTY); \
   IMPLEMENT_CAST_CASE_END()

#define IMPLEMENT_CAST_CASE_PTR(DESTTY, DESTCTY) \
   IMPLEMENT_CAST_CASE_START(DESTTY, DESTCTY);   \
   IMPLEMENT_CAST_CASE_PTR_IMP(DESTTY, DESTCTY); \
   IMPLEMENT_CAST_CASE_END()

static void executeCastInst(CastInst *I, ExecutionContext &SF) {
  const Type *Ty = I->getType();
  const Type *SrcTy = I->getOperand(0)->getType();
  GenericValue Src  = getOperandValue(I->getOperand(0), SF);
  GenericValue Dest;

  switch (Ty->getPrimitiveID()) {
    IMPLEMENT_CAST_CASE(UByte , unsigned char);
    IMPLEMENT_CAST_CASE(SByte ,   signed char);
    IMPLEMENT_CAST_CASE(UShort, unsigned short);
    IMPLEMENT_CAST_CASE(Short ,   signed char);
    IMPLEMENT_CAST_CASE(UInt  , unsigned int );
    IMPLEMENT_CAST_CASE(Int   ,   signed int );
    IMPLEMENT_CAST_CASE_FP(Float ,          float);
    IMPLEMENT_CAST_CASE_FP(Double,          double);
    IMPLEMENT_CAST_CASE_PTR(Pointer, GenericValue *);
  case Type::ULongTyID:
  case Type::LongTyID:
  default:
    cout << "Unhandled dest type for cast instruction: " << Ty << endl;
  }
  SetValue(I, Dest, SF);
}




//===----------------------------------------------------------------------===//
//                        Dispatch and Execution Code
//===----------------------------------------------------------------------===//

MethodInfo::MethodInfo(Method *M) : Annotation(MethodInfoAID) {
  // Assign slot numbers to the method arguments...
  const Method::ArgumentListType &ArgList = M->getArgumentList();
  for (Method::ArgumentListType::const_iterator AI = ArgList.begin(), 
	 AE = ArgList.end(); AI != AE; ++AI) {
    MethodArgument *MA = *AI;
    MA->addAnnotation(new SlotNumber(getValueSlot(MA)));
  }

  // Iterate over all of the instructions...
  unsigned InstNum = 0;
  for (Method::inst_iterator MI = M->inst_begin(), ME = M->inst_end();
       MI != ME; ++MI) {
    Instruction *I = *MI;                          // For each instruction...
    I->addAnnotation(new InstNumber(++InstNum, getValueSlot(I))); // Add Annote
  }
}

unsigned MethodInfo::getValueSlot(const Value *V) {
  unsigned Plane = V->getType()->getUniqueID();
  if (Plane >= NumPlaneElements.size())
    NumPlaneElements.resize(Plane+1, 0);
  return NumPlaneElements[Plane]++;
}


void Interpreter::initializeExecutionEngine() {
  AnnotationManager::registerAnnotationFactory(MethodInfoAID, CreateMethodInfo);
}

//===----------------------------------------------------------------------===//
// callMethod - Execute the specified method...
//
void Interpreter::callMethod(Method *M, const vector<GenericValue> &ArgVals) {
  assert((ECStack.empty() || ECStack.back().Caller == 0 || 
	  ECStack.back().Caller->getNumOperands()-1 == ArgVals.size()) &&
	 "Incorrect number of arguments passed into function call!");
  if (M->isExternal()) {
    callExternalMethod(M, ArgVals);
    return;
  }

  // Process the method, assigning instruction numbers to the instructions in
  // the method.  Also calculate the number of values for each type slot active.
  //
  MethodInfo *MethInfo = (MethodInfo*)M->getOrCreateAnnotation(MethodInfoAID);
  ECStack.push_back(ExecutionContext());         // Make a new stack frame...

  ExecutionContext &StackFrame = ECStack.back(); // Fill it in...
  StackFrame.CurMethod = M;
  StackFrame.CurBB     = M->front();
  StackFrame.CurInst   = StackFrame.CurBB->begin();
  StackFrame.MethInfo  = MethInfo;

  // Initialize the values to nothing...
  StackFrame.Values.resize(MethInfo->NumPlaneElements.size());
  for (unsigned i = 0; i < MethInfo->NumPlaneElements.size(); ++i)
    StackFrame.Values[i].resize(MethInfo->NumPlaneElements[i]);

  StackFrame.PrevBB = 0;  // No previous BB for PHI nodes...


  // Run through the method arguments and initialize their values...
  unsigned i = 0;
  for (Method::ArgumentListType::iterator MI = M->getArgumentList().begin(),
	 ME = M->getArgumentList().end(); MI != ME; ++MI, ++i) {
    SetValue(*MI, ArgVals[i], StackFrame);
  }
}

// executeInstruction - Interpret a single instruction, increment the "PC", and
// return true if the next instruction is a breakpoint...
//
bool Interpreter::executeInstruction() {
  assert(!ECStack.empty() && "No program running, cannot execute inst!");

  ExecutionContext &SF = ECStack.back();  // Current stack frame
  Instruction *I = *SF.CurInst++;         // Increment before execute

  if (I->isBinaryOp()) {
    executeBinaryInst((BinaryOperator*)I, SF);
  } else {
    switch (I->getOpcode()) {
      // Terminators
    case Instruction::Ret:     executeRetInst   ((ReturnInst*)I, SF); break;
    case Instruction::Br:      executeBrInst    ((BranchInst*)I, SF); break;
      // Memory Instructions
    case Instruction::Alloca:
    case Instruction::Malloc:  executeAllocInst ((AllocationInst*)I, SF); break;
    case Instruction::Free:    executeFreeInst  ((FreeInst*)  I, SF); break;
    case Instruction::Load:    executeLoadInst  ((LoadInst*)  I, SF); break;
    case Instruction::Store:   executeStoreInst ((StoreInst*) I, SF); break;

      // Miscellaneous Instructions
    case Instruction::Call:    executeCallInst  ((CallInst*)  I, SF); break;
    case Instruction::PHINode: executePHINode   ((PHINode*)   I, SF); break;
    case Instruction::Shl:     executeShlInst   ((ShiftInst*) I, SF); break;
    case Instruction::Shr:     executeShrInst   ((ShiftInst*) I, SF); break;
    case Instruction::Cast:    executeCastInst  ((CastInst*)  I, SF); break;
    default:
      cout << "Don't know how to execute this instruction!\n-->" << I;
    }
  }
  
  // Reset the current frame location to the top of stack
  CurFrame = ECStack.size()-1;

  if (CurFrame == -1) return false;  // No breakpoint if no code

  // Return true if there is a breakpoint annotation on the instruction...
  return (*ECStack[CurFrame].CurInst)->getAnnotation(BreakpointAID) != 0;
}

void Interpreter::stepInstruction() {  // Do the 'step' command
  if (ECStack.empty()) {
    cout << "Error: no program running, cannot step!\n";
    return;
  }

  // Run an instruction...
  executeInstruction();

  // Print the next instruction to execute...
  printCurrentInstruction();
}

// --- UI Stuff...



void Interpreter::nextInstruction() {  // Do the 'next' command
  if (ECStack.empty()) {
    cout << "Error: no program running, cannot 'next'!\n";
    return;
  }

  // If this is a call instruction, step over the call instruction...
  // TODO: ICALL, CALL WITH, ...
  if ((*ECStack.back().CurInst)->getOpcode() == Instruction::Call) {
    // Step into the function...
    if (executeInstruction()) {
      // Hit a breakpoint, print current instruction, then return to user...
      cout << "Breakpoint hit!\n";
      printCurrentInstruction();
      return;
    }

    // Finish executing the function...
    finish();
  } else {
    // Normal instruction, just step...
    stepInstruction();
  }
}

void Interpreter::run() {
  if (ECStack.empty()) {
    cout << "Error: no program running, cannot run!\n";
    return;
  }

  bool HitBreakpoint = false;
  while (!ECStack.empty() && !HitBreakpoint) {
    // Run an instruction...
    HitBreakpoint = executeInstruction();
  }

  if (HitBreakpoint) {
    cout << "Breakpoint hit!\n";
  }

  // Print the next instruction to execute...
  printCurrentInstruction();
}

void Interpreter::finish() {
  if (ECStack.empty()) {
    cout << "Error: no program running, cannot run!\n";
    return;
  }

  unsigned StackSize = ECStack.size();
  bool HitBreakpoint = false;
  while (ECStack.size() >= StackSize && !HitBreakpoint) {
    // Run an instruction...
    HitBreakpoint = executeInstruction();
  }

  if (HitBreakpoint) {
    cout << "Breakpoint hit!\n";
  }

  // Print the next instruction to execute...
  printCurrentInstruction();
}



// printCurrentInstruction - Print out the instruction that the virtual PC is
// at, or fail silently if no program is running.
//
void Interpreter::printCurrentInstruction() {
  if (!ECStack.empty()) {
    Instruction *I = *ECStack.back().CurInst;
    InstNumber *IN = (InstNumber*)I->getAnnotation(SlotNumberAID);
    assert(IN && "Instruction has no numbering annotation!");
    cout << "#" << IN->InstNum << I;
  }
}

void Interpreter::printValue(const Type *Ty, GenericValue V) {
  cout << Ty << " ";

  switch (Ty->getPrimitiveID()) {
  case Type::BoolTyID:   cout << (V.BoolVal?"true":"false"); break;
  case Type::SByteTyID:  cout << V.SByteVal;  break;
  case Type::UByteTyID:  cout << V.UByteVal;  break;
  case Type::ShortTyID:  cout << V.ShortVal;  break;
  case Type::UShortTyID: cout << V.UShortVal; break;
  case Type::IntTyID:    cout << V.IntVal;    break;
  case Type::UIntTyID:   cout << V.UIntVal;   break;
  case Type::FloatTyID:  cout << V.FloatVal;  break;
  case Type::DoubleTyID: cout << V.DoubleVal; break;
  case Type::PointerTyID:cout << V.PointerVal; break;
  default:
    cout << "- Don't know how to print value of this type!";
    break;
  }
}

void Interpreter::printValue(const string &Name) {
  Value *PickedVal = ChooseOneOption(Name, LookupMatchingNames(Name));
  if (!PickedVal) return;

  if (const Method *M = PickedVal->castMethod()) {
    cout << M;  // Print the method
  } else {      // Otherwise there should be an annotation for the slot#
    printValue(PickedVal->getType(), 
	       getOperandValue(PickedVal, ECStack[CurFrame]));
    cout << endl;
  }
    
}

void Interpreter::infoValue(const string &Name) {
  Value *PickedVal = ChooseOneOption(Name, LookupMatchingNames(Name));
  if (!PickedVal) return;

  cout << "Value: ";
  printValue(PickedVal->getType(), 
	     getOperandValue(PickedVal, ECStack[CurFrame]));
  cout << endl;
  printOperandInfo(PickedVal, ECStack[CurFrame]);
}

void Interpreter::list() {
  if (ECStack.empty())
    cout << "Error: No program executing!\n";
  else
    cout << ECStack[CurFrame].CurMethod;   // Just print the method out...
}

void Interpreter::printStackTrace() {
  if (ECStack.empty()) cout << "No program executing!\n";

  for (unsigned i = 0; i < ECStack.size(); ++i) {
    cout << (((int)i == CurFrame) ? '>' : '-');
    cout << "#" << i << ". " << ECStack[i].CurMethod->getType() << " \""
	 << ECStack[i].CurMethod->getName() << "\"(";
    // TODO: Print Args
    cout << ")" << endl;
    cout << *ECStack[i].CurInst;
  }
}
