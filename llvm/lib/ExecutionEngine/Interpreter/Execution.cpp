//===-- Execution.cpp - Implement code to simulate the program ------------===//
// 
//  This file contains the actual instruction interpreter.
//
//===----------------------------------------------------------------------===//

#include "Interpreter.h"
#include "ExecutionAnnotations.h"
#include "llvm/iOther.h"
#include "llvm/iTerminators.h"
#include "llvm/Type.h"
#include "llvm/ConstPoolVals.h"
#include "llvm/Assembly/Writer.h"

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

static void SetValue(Value *V, GenericValue Val, ExecutionContext &SF) {
  unsigned TyP = V->getType()->getUniqueID();   // TypePlane for value
  SF.Values[TyP][getOperandSlot(V)] = Val;
}



//===----------------------------------------------------------------------===//
//                    Binary Instruction Implementations
//===----------------------------------------------------------------------===//

#define IMPLEMENT_BINARY_OPERATOR(OP, TY) \
   case Type::TY##TyID: Dest.TY##Val = Src1.TY##Val OP Src2.TY##Val; break

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
//                 Miscellaneous Instruction Implementations
//===----------------------------------------------------------------------===//

void Interpreter::executeCallInst(CallInst *I, ExecutionContext &SF) {
  ECStack.back().Caller = I;
  callMethod(I->getCalledMethod(), &ECStack.back());
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
void Interpreter::callMethod(Method *M, ExecutionContext *CallingSF = 0) {
  if (M->isExternal()) {
    // Handle builtin methods
    cout << "Error: Method '" << M->getName() << "' is external!\n";
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
  if (CallingSF) {
    CallInst *Call = CallingSF->Caller;
    assert(Call && "Caller improperly initialized!");
    
    unsigned i = 0;
    for (Method::ArgumentListType::iterator MI = M->getArgumentList().begin(),
	   ME = M->getArgumentList().end(); MI != ME; ++MI, ++i) {
      Value *V = Call->getOperand(i+1);
      MethodArgument *MA = *MI;

      SetValue(MA, getOperandValue(V, *CallingSF), StackFrame);
    }
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
    case Instruction::Ret:     executeRetInst   ((ReturnInst*)I, SF); break;
    case Instruction::Br:      executeBrInst    ((BranchInst*)I, SF); break;
    case Instruction::Call:    executeCallInst  ((CallInst*)  I, SF); break;
    case Instruction::PHINode: executePHINode   ((PHINode*)   I, SF); break;
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
