//===-- Execution.cpp - Implement code to simulate the program ------------===//
// 
//  This file contains the actual instruction interpreter.
//
//===----------------------------------------------------------------------===//

#include "Interpreter.h"
#include "ExecutionAnnotations.h"
#include "llvm/iPHINode.h"
#include "llvm/iOther.h"
#include "llvm/iTerminators.h"
#include "llvm/iMemory.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Constants.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/Target/TargetData.h"
#include "llvm/GlobalVariable.h"
#include "Support/CommandLine.h"
#include <math.h>  // For fmod
#include <signal.h>
#include <setjmp.h>
#include <iostream>
using std::vector;
using std::cout;
using std::cerr;

static cl::opt<bool>
QuietMode("quiet", cl::desc("Do not emit any non-program output"));

static cl::alias 
QuietModeA("q", cl::desc("Alias for -quiet"), cl::aliasopt(QuietMode));

static cl::opt<bool>
ArrayChecksEnabled("array-checks", cl::desc("Enable array bound checks"));

static cl::opt<bool>
AbortOnExceptions("abort-on-exception",
                  cl::desc("Halt execution on a machine exception"));

// Create a TargetData structure to handle memory addressing and size/alignment
// computations
//
static TargetData TD("lli Interpreter");
CachedWriter CW;     // Object to accelerate printing of LLVM


#ifdef PROFILE_STRUCTURE_FIELDS
static cl::opt<bool>
ProfileStructureFields("profilestructfields", 
                       cl::desc("Profile Structure Field Accesses"));
#include <map>
static std::map<const StructType *, vector<unsigned> > FieldAccessCounts;
#endif

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

#define GET_CONST_VAL(TY, CLASS) \
  case Type::TY##TyID: Result.TY##Val = cast<CLASS>(CPV)->getValue(); break

static GenericValue getOperandValue(Value *V, ExecutionContext &SF) {
  if (Constant *CPV = dyn_cast<Constant>(V)) {
    GenericValue Result;
    switch (CPV->getType()->getPrimitiveID()) {
      GET_CONST_VAL(Bool   , ConstantBool);
      GET_CONST_VAL(UByte  , ConstantUInt);
      GET_CONST_VAL(SByte  , ConstantSInt);
      GET_CONST_VAL(UShort , ConstantUInt);
      GET_CONST_VAL(Short  , ConstantSInt);
      GET_CONST_VAL(UInt   , ConstantUInt);
      GET_CONST_VAL(Int    , ConstantSInt);
      GET_CONST_VAL(ULong  , ConstantUInt);
      GET_CONST_VAL(Long   , ConstantSInt);
      GET_CONST_VAL(Float  , ConstantFP);
      GET_CONST_VAL(Double , ConstantFP);
    case Type::PointerTyID:
      if (isa<ConstantPointerNull>(CPV)) {
        Result.PointerVal = 0;
      } else if (isa<ConstantPointerRef>(CPV)) {
        assert(0 && "Not implemented!");
      } else {
        assert(0 && "Unknown constant pointer type!");
      }
      break;
    default:
      cout << "ERROR: Constant unimp for type: " << CPV->getType() << "\n";
    }
    return Result;
  } else if (GlobalValue *GV = dyn_cast<GlobalValue>(V)) {
    GlobalAddress *Address = 
      (GlobalAddress*)GV->getOrCreateAnnotation(GlobalAddressAID);
    GenericValue Result;
    Result.PointerVal = (PointerTy)(GenericValue*)Address->Ptr;
    return Result;
  } else {
    unsigned TyP = V->getType()->getUniqueID();   // TypePlane for value
    unsigned OpSlot = getOperandSlot(V);
    assert(TyP < SF.Values.size() && 
           OpSlot < SF.Values[TyP].size() && "Value out of range!");
    return SF.Values[TyP][getOperandSlot(V)];
  }
}

static void printOperandInfo(Value *V, ExecutionContext &SF) {
  if (isa<Constant>(V)) {
    cout << "Constant Pool Value\n";
  } else if (isa<GlobalValue>(V)) {
    cout << "Global Value\n";
  } else {
    unsigned TyP  = V->getType()->getUniqueID();   // TypePlane for value
    unsigned Slot = getOperandSlot(V);
    cout << "Value=" << (void*)V << " TypeID=" << TyP << " Slot=" << Slot
         << " Addr=" << &SF.Values[TyP][Slot] << " SF=" << &SF
         << " Contents=0x";

    const unsigned char *Buf = (const unsigned char*)&SF.Values[TyP][Slot];
    for (unsigned i = 0; i < sizeof(GenericValue); ++i) {
      unsigned char Cur = Buf[i];
      cout << ( Cur     >= 160? char((Cur>>4)+'A'-10) : char((Cur>>4) + '0'))
           << ((Cur&15) >=  10? char((Cur&15)+'A'-10) : char((Cur&15) + '0'));
    }
    cout << "\n";
  }
}



static void SetValue(Value *V, GenericValue Val, ExecutionContext &SF) {
  unsigned TyP = V->getType()->getUniqueID();   // TypePlane for value

  //cout << "Setting value: " << &SF.Values[TyP][getOperandSlot(V)] << "\n";
  SF.Values[TyP][getOperandSlot(V)] = Val;
}


//===----------------------------------------------------------------------===//
//                    Annotation Wrangling code
//===----------------------------------------------------------------------===//

void Interpreter::initializeExecutionEngine() {
  AnnotationManager::registerAnnotationFactory(MethodInfoAID,
                                               &MethodInfo::Create);
  AnnotationManager::registerAnnotationFactory(GlobalAddressAID, 
                                               &GlobalAddress::Create);
  initializeSignalHandlers();
}

// InitializeMemory - Recursive function to apply a Constant value into the
// specified memory location...
//
static void InitializeMemory(const Constant *Init, char *Addr) {
#define INITIALIZE_MEMORY(TYID, CLASS, TY)  \
  case Type::TYID##TyID: {                  \
    TY Tmp = cast<CLASS>(Init)->getValue(); \
    memcpy(Addr, &Tmp, sizeof(TY));         \
  } return

  switch (Init->getType()->getPrimitiveID()) {
    INITIALIZE_MEMORY(Bool   , ConstantBool, bool);
    INITIALIZE_MEMORY(UByte  , ConstantUInt, unsigned char);
    INITIALIZE_MEMORY(SByte  , ConstantSInt, signed   char);
    INITIALIZE_MEMORY(UShort , ConstantUInt, unsigned short);
    INITIALIZE_MEMORY(Short  , ConstantSInt, signed   short);
    INITIALIZE_MEMORY(UInt   , ConstantUInt, unsigned int);
    INITIALIZE_MEMORY(Int    , ConstantSInt, signed   int);
    INITIALIZE_MEMORY(ULong  , ConstantUInt, uint64_t);
    INITIALIZE_MEMORY(Long   , ConstantSInt,  int64_t);
    INITIALIZE_MEMORY(Float  , ConstantFP  , float);
    INITIALIZE_MEMORY(Double , ConstantFP  , double);
#undef INITIALIZE_MEMORY

  case Type::ArrayTyID: {
    const ConstantArray *CPA = cast<ConstantArray>(Init);
    const vector<Use> &Val = CPA->getValues();
    unsigned ElementSize = 
      TD.getTypeSize(cast<ArrayType>(CPA->getType())->getElementType());
    for (unsigned i = 0; i < Val.size(); ++i)
      InitializeMemory(cast<Constant>(Val[i].get()), Addr+i*ElementSize);
    return;
  }

  case Type::StructTyID: {
    const ConstantStruct *CPS = cast<ConstantStruct>(Init);
    const StructLayout *SL=TD.getStructLayout(cast<StructType>(CPS->getType()));
    const vector<Use> &Val = CPS->getValues();
    for (unsigned i = 0; i < Val.size(); ++i)
      InitializeMemory(cast<Constant>(Val[i].get()),
                       Addr+SL->MemberOffsets[i]);
    return;
  }

  case Type::PointerTyID:
    if (isa<ConstantPointerNull>(Init)) {
      *(void**)Addr = 0;
    } else if (const ConstantPointerRef *CPR =
               dyn_cast<ConstantPointerRef>(Init)) {
      GlobalAddress *Address = 
       (GlobalAddress*)CPR->getValue()->getOrCreateAnnotation(GlobalAddressAID);
      *(void**)Addr = (GenericValue*)Address->Ptr;
    } else {
      assert(0 && "Unknown Constant pointer type!");
    }
    return;

  default:
    CW << "Bad Type: " << Init->getType() << "\n";
    assert(0 && "Unknown constant type to initialize memory with!");
  }
}

Annotation *GlobalAddress::Create(AnnotationID AID, const Annotable *O, void *){
  assert(AID == GlobalAddressAID);

  // This annotation will only be created on GlobalValue objects...
  GlobalValue *GVal = cast<GlobalValue>((Value*)O);

  if (isa<Function>(GVal)) {
    // The GlobalAddress object for a function is just a pointer to function
    // itself.  Don't delete it when the annotation is gone though!
    return new GlobalAddress(GVal, false);
  }

  // Handle the case of a global variable...
  assert(isa<GlobalVariable>(GVal) && 
         "Global value found that isn't a function or global variable!");
  GlobalVariable *GV = cast<GlobalVariable>(GVal);
  
  // First off, we must allocate space for the global variable to point at...
  const Type *Ty = GV->getType()->getElementType();  // Type to be allocated

  // Allocate enough memory to hold the type...
  void *Addr = calloc(1, TD.getTypeSize(Ty));
  assert(Addr != 0 && "Null pointer returned by malloc!");

  // Initialize the memory if there is an initializer...
  if (GV->hasInitializer())
    InitializeMemory(GV->getInitializer(), (char*)Addr);

  return new GlobalAddress(Addr, true);  // Simply invoke the ctor
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
    IMPLEMENT_BINARY_OPERATOR(+, ULong);
    IMPLEMENT_BINARY_OPERATOR(+, Long);
    IMPLEMENT_BINARY_OPERATOR(+, Float);
    IMPLEMENT_BINARY_OPERATOR(+, Double);
    IMPLEMENT_BINARY_OPERATOR(+, Pointer);
  default:
    cout << "Unhandled type for Add instruction: " << Ty << "\n";
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
    IMPLEMENT_BINARY_OPERATOR(-, ULong);
    IMPLEMENT_BINARY_OPERATOR(-, Long);
    IMPLEMENT_BINARY_OPERATOR(-, Float);
    IMPLEMENT_BINARY_OPERATOR(-, Double);
    IMPLEMENT_BINARY_OPERATOR(-, Pointer);
  default:
    cout << "Unhandled type for Sub instruction: " << Ty << "\n";
  }
  return Dest;
}

static GenericValue executeMulInst(GenericValue Src1, GenericValue Src2, 
				   const Type *Ty, ExecutionContext &SF) {
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
    IMPLEMENT_BINARY_OPERATOR(*, Pointer);
  default:
    cout << "Unhandled type for Mul instruction: " << Ty << "\n";
  }
  return Dest;
}

static GenericValue executeDivInst(GenericValue Src1, GenericValue Src2, 
				   const Type *Ty, ExecutionContext &SF) {
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
    IMPLEMENT_BINARY_OPERATOR(/, Pointer);
  default:
    cout << "Unhandled type for Div instruction: " << Ty << "\n";
  }
  return Dest;
}

static GenericValue executeRemInst(GenericValue Src1, GenericValue Src2, 
				   const Type *Ty, ExecutionContext &SF) {
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
    IMPLEMENT_BINARY_OPERATOR(%, Pointer);
  case Type::FloatTyID:
    Dest.FloatVal = fmod(Src1.FloatVal, Src2.FloatVal);
    break;
  case Type::DoubleTyID:
    Dest.DoubleVal = fmod(Src1.DoubleVal, Src2.DoubleVal);
    break;
  default:
    cout << "Unhandled type for Rem instruction: " << Ty << "\n";
  }
  return Dest;
}

static GenericValue executeAndInst(GenericValue Src1, GenericValue Src2, 
				   const Type *Ty, ExecutionContext &SF) {
  GenericValue Dest;
  switch (Ty->getPrimitiveID()) {
    IMPLEMENT_BINARY_OPERATOR(&, UByte);
    IMPLEMENT_BINARY_OPERATOR(&, SByte);
    IMPLEMENT_BINARY_OPERATOR(&, UShort);
    IMPLEMENT_BINARY_OPERATOR(&, Short);
    IMPLEMENT_BINARY_OPERATOR(&, UInt);
    IMPLEMENT_BINARY_OPERATOR(&, Int);
    IMPLEMENT_BINARY_OPERATOR(&, ULong);
    IMPLEMENT_BINARY_OPERATOR(&, Long);
    IMPLEMENT_BINARY_OPERATOR(&, Pointer);
  default:
    cout << "Unhandled type for And instruction: " << Ty << "\n";
  }
  return Dest;
}


static GenericValue executeOrInst(GenericValue Src1, GenericValue Src2, 
                                  const Type *Ty, ExecutionContext &SF) {
  GenericValue Dest;
  switch (Ty->getPrimitiveID()) {
    IMPLEMENT_BINARY_OPERATOR(|, UByte);
    IMPLEMENT_BINARY_OPERATOR(|, SByte);
    IMPLEMENT_BINARY_OPERATOR(|, UShort);
    IMPLEMENT_BINARY_OPERATOR(|, Short);
    IMPLEMENT_BINARY_OPERATOR(|, UInt);
    IMPLEMENT_BINARY_OPERATOR(|, Int);
    IMPLEMENT_BINARY_OPERATOR(|, ULong);
    IMPLEMENT_BINARY_OPERATOR(|, Long);
    IMPLEMENT_BINARY_OPERATOR(|, Pointer);
  default:
    cout << "Unhandled type for Or instruction: " << Ty << "\n";
  }
  return Dest;
}


static GenericValue executeXorInst(GenericValue Src1, GenericValue Src2, 
                                   const Type *Ty, ExecutionContext &SF) {
  GenericValue Dest;
  switch (Ty->getPrimitiveID()) {
    IMPLEMENT_BINARY_OPERATOR(^, UByte);
    IMPLEMENT_BINARY_OPERATOR(^, SByte);
    IMPLEMENT_BINARY_OPERATOR(^, UShort);
    IMPLEMENT_BINARY_OPERATOR(^, Short);
    IMPLEMENT_BINARY_OPERATOR(^, UInt);
    IMPLEMENT_BINARY_OPERATOR(^, Int);
    IMPLEMENT_BINARY_OPERATOR(^, ULong);
    IMPLEMENT_BINARY_OPERATOR(^, Long);
    IMPLEMENT_BINARY_OPERATOR(^, Pointer);
  default:
    cout << "Unhandled type for Xor instruction: " << Ty << "\n";
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
    IMPLEMENT_SETCC(==, ULong);
    IMPLEMENT_SETCC(==, Long);
    IMPLEMENT_SETCC(==, Float);
    IMPLEMENT_SETCC(==, Double);
    IMPLEMENT_SETCC(==, Pointer);
  default:
    cout << "Unhandled type for SetEQ instruction: " << Ty << "\n";
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
    IMPLEMENT_SETCC(!=, ULong);
    IMPLEMENT_SETCC(!=, Long);
    IMPLEMENT_SETCC(!=, Float);
    IMPLEMENT_SETCC(!=, Double);
    IMPLEMENT_SETCC(!=, Pointer);

  default:
    cout << "Unhandled type for SetNE instruction: " << Ty << "\n";
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
    IMPLEMENT_SETCC(<=, ULong);
    IMPLEMENT_SETCC(<=, Long);
    IMPLEMENT_SETCC(<=, Float);
    IMPLEMENT_SETCC(<=, Double);
    IMPLEMENT_SETCC(<=, Pointer);
  default:
    cout << "Unhandled type for SetLE instruction: " << Ty << "\n";
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
    IMPLEMENT_SETCC(>=, ULong);
    IMPLEMENT_SETCC(>=, Long);
    IMPLEMENT_SETCC(>=, Float);
    IMPLEMENT_SETCC(>=, Double);
    IMPLEMENT_SETCC(>=, Pointer);
  default:
    cout << "Unhandled type for SetGE instruction: " << Ty << "\n";
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
    IMPLEMENT_SETCC(<, ULong);
    IMPLEMENT_SETCC(<, Long);
    IMPLEMENT_SETCC(<, Float);
    IMPLEMENT_SETCC(<, Double);
    IMPLEMENT_SETCC(<, Pointer);
  default:
    cout << "Unhandled type for SetLT instruction: " << Ty << "\n";
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
    IMPLEMENT_SETCC(>, ULong);
    IMPLEMENT_SETCC(>, Long);
    IMPLEMENT_SETCC(>, Float);
    IMPLEMENT_SETCC(>, Double);
    IMPLEMENT_SETCC(>, Pointer);
  default:
    cout << "Unhandled type for SetGT instruction: " << Ty << "\n";
  }
  return Dest;
}

static void executeBinaryInst(BinaryOperator &I, ExecutionContext &SF) {
  const Type *Ty    = I.getOperand(0)->getType();
  GenericValue Src1 = getOperandValue(I.getOperand(0), SF);
  GenericValue Src2 = getOperandValue(I.getOperand(1), SF);
  GenericValue R;   // Result

  switch (I.getOpcode()) {
  case Instruction::Add:   R = executeAddInst  (Src1, Src2, Ty, SF); break;
  case Instruction::Sub:   R = executeSubInst  (Src1, Src2, Ty, SF); break;
  case Instruction::Mul:   R = executeMulInst  (Src1, Src2, Ty, SF); break;
  case Instruction::Div:   R = executeDivInst  (Src1, Src2, Ty, SF); break;
  case Instruction::Rem:   R = executeRemInst  (Src1, Src2, Ty, SF); break;
  case Instruction::And:   R = executeAndInst  (Src1, Src2, Ty, SF); break;
  case Instruction::Or:    R = executeOrInst   (Src1, Src2, Ty, SF); break;
  case Instruction::Xor:   R = executeXorInst  (Src1, Src2, Ty, SF); break;
  case Instruction::SetEQ: R = executeSetEQInst(Src1, Src2, Ty, SF); break;
  case Instruction::SetNE: R = executeSetNEInst(Src1, Src2, Ty, SF); break;
  case Instruction::SetLE: R = executeSetLEInst(Src1, Src2, Ty, SF); break;
  case Instruction::SetGE: R = executeSetGEInst(Src1, Src2, Ty, SF); break;
  case Instruction::SetLT: R = executeSetLTInst(Src1, Src2, Ty, SF); break;
  case Instruction::SetGT: R = executeSetGTInst(Src1, Src2, Ty, SF); break;
  default:
    cout << "Don't know how to handle this binary operator!\n-->" << I;
    R = Src1;
  }

  SetValue(&I, R, SF);
}

//===----------------------------------------------------------------------===//
//                     Terminator Instruction Implementations
//===----------------------------------------------------------------------===//

static void PerformExitStuff() {
#ifdef PROFILE_STRUCTURE_FIELDS
  // Print out structure field accounting information...
  if (!FieldAccessCounts.empty()) {
    CW << "Profile Field Access Counts:\n";
    std::map<const StructType *, vector<unsigned> >::iterator 
      I = FieldAccessCounts.begin(), E = FieldAccessCounts.end();
    for (; I != E; ++I) {
      vector<unsigned> &OfC = I->second;
      CW << "  '" << (Value*)I->first << "'\t- Sum=";
      
      unsigned Sum = 0;
      for (unsigned i = 0; i < OfC.size(); ++i)
        Sum += OfC[i];
      CW << Sum << " - ";
      
      for (unsigned i = 0; i < OfC.size(); ++i) {
        if (i) CW << ", ";
        CW << OfC[i];
      }
      CW << "\n";
    }
    CW << "\n";

    CW << "Profile Field Access Percentages:\n";
    cout.precision(3);
    for (I = FieldAccessCounts.begin(); I != E; ++I) {
      vector<unsigned> &OfC = I->second;
      unsigned Sum = 0;
      for (unsigned i = 0; i < OfC.size(); ++i)
        Sum += OfC[i];
      
      CW << "  '" << (Value*)I->first << "'\t- ";
      for (unsigned i = 0; i < OfC.size(); ++i) {
        if (i) CW << ", ";
        CW << double(OfC[i])/Sum;
      }
      CW << "\n";
    }
    CW << "\n";

    FieldAccessCounts.clear();
  }
#endif
}

void Interpreter::exitCalled(GenericValue GV) {
  if (!QuietMode) {
    cout << "Program returned ";
    print(Type::IntTy, GV);
    cout << " via 'void exit(int)'\n";
  }

  ExitCode = GV.SByteVal;
  ECStack.clear();
  PerformExitStuff();
}

void Interpreter::executeRetInst(ReturnInst &I, ExecutionContext &SF) {
  const Type *RetTy = 0;
  GenericValue Result;

  // Save away the return value... (if we are not 'ret void')
  if (I.getNumOperands()) {
    RetTy  = I.getReturnValue()->getType();
    Result = getOperandValue(I.getReturnValue(), SF);
  }

  // Save previously executing meth
  const Function *M = ECStack.back().CurMethod;

  // Pop the current stack frame... this invalidates SF
  ECStack.pop_back();

  if (ECStack.empty()) {  // Finished main.  Put result into exit code...
    if (RetTy) {          // Nonvoid return type?
      if (!QuietMode) {
        CW << "Function " << M->getType() << " \"" << M->getName()
           << "\" returned ";
        print(RetTy, Result);
        cout << "\n";
      }

      if (RetTy->isIntegral())
	ExitCode = Result.IntVal;   // Capture the exit code of the program
    } else {
      ExitCode = 0;
    }

    PerformExitStuff();
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
    cout << "\n";
  }
}

void Interpreter::executeBrInst(BranchInst &I, ExecutionContext &SF) {
  SF.PrevBB = SF.CurBB;               // Update PrevBB so that PHI nodes work...
  BasicBlock *Dest;

  Dest = I.getSuccessor(0);          // Uncond branches have a fixed dest...
  if (!I.isUnconditional()) {
    Value *Cond = I.getCondition();
    GenericValue CondVal = getOperandValue(Cond, SF);
    if (CondVal.BoolVal == 0) // If false cond...
      Dest = I.getSuccessor(1);    
  }
  SF.CurBB   = Dest;                  // Update CurBB to branch destination
  SF.CurInst = SF.CurBB->begin();     // Update new instruction ptr...
}

//===----------------------------------------------------------------------===//
//                     Memory Instruction Implementations
//===----------------------------------------------------------------------===//

void Interpreter::executeAllocInst(AllocationInst &I, ExecutionContext &SF) {
  const Type *Ty = I.getType()->getElementType();  // Type to be allocated

  // Get the number of elements being allocated by the array...
  unsigned NumElements = getOperandValue(I.getOperand(0), SF).UIntVal;

  // Allocate enough memory to hold the type...
  // FIXME: Don't use CALLOC, use a tainted malloc.
  void *Memory = calloc(NumElements, TD.getTypeSize(Ty));

  GenericValue Result;
  Result.PointerVal = (PointerTy)Memory;
  assert(Result.PointerVal != 0 && "Null pointer returned by malloc!");
  SetValue(&I, Result, SF);

  if (I.getOpcode() == Instruction::Alloca)
    ECStack.back().Allocas.add(Memory);
}

static void executeFreeInst(FreeInst &I, ExecutionContext &SF) {
  assert(isa<PointerType>(I.getOperand(0)->getType()) && "Freeing nonptr?");
  GenericValue Value = getOperandValue(I.getOperand(0), SF);
  // TODO: Check to make sure memory is allocated
  free((void*)Value.PointerVal);   // Free memory
}


// getElementOffset - The workhorse for getelementptr.  This function returns
// the offset that arguments ArgOff+1 -> NumArgs specify for the pointer type
// specified by argument Arg.
//
static PointerTy getElementOffset(GetElementPtrInst &I, ExecutionContext &SF) {
  assert(isa<PointerType>(I.getPointerOperand()->getType()) &&
         "Cannot getElementOffset of a nonpointer type!");

  PointerTy Total = 0;
  const Type *Ty = I.getPointerOperand()->getType();
  
  unsigned ArgOff = 1;
  while (ArgOff < I.getNumOperands()) {
    if (const StructType *STy = dyn_cast<StructType>(Ty)) {
      const StructLayout *SLO = TD.getStructLayout(STy);
      
      // Indicies must be ubyte constants...
      const ConstantUInt *CPU = cast<ConstantUInt>(I.getOperand(ArgOff++));
      assert(CPU->getType() == Type::UByteTy);
      unsigned Index = CPU->getValue();
      
#ifdef PROFILE_STRUCTURE_FIELDS
      if (ProfileStructureFields) {
        // Do accounting for this field...
        vector<unsigned> &OfC = FieldAccessCounts[STy];
        if (OfC.size() == 0) OfC.resize(STy->getElementTypes().size());
        OfC[Index]++;
      }
#endif
      
      Total += SLO->MemberOffsets[Index];
      Ty = STy->getElementTypes()[Index];
    } else if (const SequentialType *ST = cast<SequentialType>(Ty)) {

      // Get the index number for the array... which must be uint type...
      assert(I.getOperand(ArgOff)->getType() == Type::UIntTy);
      unsigned Idx = getOperandValue(I.getOperand(ArgOff++), SF).UIntVal;
      if (const ArrayType *AT = dyn_cast<ArrayType>(ST))
        if (Idx >= AT->getNumElements() && ArrayChecksEnabled) {
          cerr << "Out of range memory access to element #" << Idx
               << " of a " << AT->getNumElements() << " element array."
               << " Subscript #" << (ArgOff-1) << "\n";
          // Get outta here!!!
          siglongjmp(SignalRecoverBuffer, SIGTRAP);
        }

      Ty = ST->getElementType();
      unsigned Size = TD.getTypeSize(Ty);
      Total += Size*Idx;
    }  
  }

  return Total;
}

static void executeGEPInst(GetElementPtrInst &I, ExecutionContext &SF) {
  GenericValue SRC = getOperandValue(I.getPointerOperand(), SF);
  PointerTy SrcPtr = SRC.PointerVal;

  GenericValue Result;
  Result.PointerVal = SrcPtr + getElementOffset(I, SF);
  SetValue(&I, Result, SF);
}

static void executeLoadInst(LoadInst &I, ExecutionContext &SF) {
  GenericValue SRC = getOperandValue(I.getPointerOperand(), SF);
  GenericValue *Ptr = (GenericValue*)SRC.PointerVal;
  GenericValue Result;

  switch (I.getType()->getPrimitiveID()) {
  case Type::BoolTyID:
  case Type::UByteTyID:
  case Type::SByteTyID:   Result.SByteVal   = Ptr->SByteVal; break;
  case Type::UShortTyID:
  case Type::ShortTyID:   Result.ShortVal   = Ptr->ShortVal; break;
  case Type::UIntTyID:
  case Type::IntTyID:     Result.IntVal     = Ptr->IntVal; break;
  case Type::ULongTyID:
  case Type::LongTyID:    Result.ULongVal   = Ptr->ULongVal; break;
  case Type::PointerTyID: Result.PointerVal = Ptr->PointerVal; break;
  case Type::FloatTyID:   Result.FloatVal   = Ptr->FloatVal; break;
  case Type::DoubleTyID:  Result.DoubleVal  = Ptr->DoubleVal; break;
  default:
    cout << "Cannot load value of type " << I.getType() << "!\n";
  }

  SetValue(&I, Result, SF);
}

static void executeStoreInst(StoreInst &I, ExecutionContext &SF) {
  GenericValue SRC = getOperandValue(I.getPointerOperand(), SF);
  GenericValue *Ptr = (GenericValue *)SRC.PointerVal;
  GenericValue Val = getOperandValue(I.getOperand(0), SF);

  switch (I.getOperand(0)->getType()->getPrimitiveID()) {
  case Type::BoolTyID:
  case Type::UByteTyID:
  case Type::SByteTyID:   Ptr->SByteVal = Val.SByteVal; break;
  case Type::UShortTyID:
  case Type::ShortTyID:   Ptr->ShortVal = Val.ShortVal; break;
  case Type::UIntTyID:
  case Type::IntTyID:     Ptr->IntVal = Val.IntVal; break;
  case Type::ULongTyID:
  case Type::LongTyID:    Ptr->LongVal = Val.LongVal; break;
  case Type::PointerTyID: Ptr->PointerVal = Val.PointerVal; break;
  case Type::FloatTyID:   Ptr->FloatVal = Val.FloatVal; break;
  case Type::DoubleTyID:  Ptr->DoubleVal = Val.DoubleVal; break;
  default:
    cout << "Cannot store value of type " << I.getType() << "!\n";
  }
}


//===----------------------------------------------------------------------===//
//                 Miscellaneous Instruction Implementations
//===----------------------------------------------------------------------===//

void Interpreter::executeCallInst(CallInst &I, ExecutionContext &SF) {
  ECStack.back().Caller = &I;
  vector<GenericValue> ArgVals;
  ArgVals.reserve(I.getNumOperands()-1);
  for (unsigned i = 1; i < I.getNumOperands(); ++i)
    ArgVals.push_back(getOperandValue(I.getOperand(i), SF));

  // To handle indirect calls, we must get the pointer value from the argument 
  // and treat it as a function pointer.
  GenericValue SRC = getOperandValue(I.getCalledValue(), SF);
  
  callMethod((Function*)SRC.PointerVal, ArgVals);
}

static void executePHINode(PHINode &I, ExecutionContext &SF) {
  BasicBlock *PrevBB = SF.PrevBB;
  Value *IncomingValue = 0;

  // Search for the value corresponding to this previous bb...
  for (unsigned i = I.getNumIncomingValues(); i > 0;) {
    if (I.getIncomingBlock(--i) == PrevBB) {
      IncomingValue = I.getIncomingValue(i);
      break;
    }
  }
  assert(IncomingValue && "No PHI node predecessor for current PrevBB!");

  // Found the value, set as the result...
  SetValue(&I, getOperandValue(IncomingValue, SF), SF);
}

#define IMPLEMENT_SHIFT(OP, TY) \
   case Type::TY##TyID: Dest.TY##Val = Src1.TY##Val OP Src2.UByteVal; break

static void executeShlInst(ShiftInst &I, ExecutionContext &SF) {
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
    IMPLEMENT_SHIFT(<<, Pointer);
  default:
    cout << "Unhandled type for Shl instruction: " << Ty << "\n";
  }
  SetValue(&I, Dest, SF);
}

static void executeShrInst(ShiftInst &I, ExecutionContext &SF) {
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
    IMPLEMENT_SHIFT(>>, Pointer);
  default:
    cout << "Unhandled type for Shr instruction: " << Ty << "\n";
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
    default: cout << "Unhandled cast: " << SrcTy << " to " << Ty << "\n";  \
      break;                                    \
    }                                           \
    break

#define IMPLEMENT_CAST_CASE(DESTTY, DESTCTY) \
   IMPLEMENT_CAST_CASE_START(DESTTY, DESTCTY);   \
   IMPLEMENT_CAST_CASE_FP_IMP(DESTTY, DESTCTY); \
   IMPLEMENT_CAST_CASE_END()

static void executeCastInst(CastInst &I, ExecutionContext &SF) {
  const Type *Ty    = I.getType();
  const Type *SrcTy = I.getOperand(0)->getType();
  GenericValue Src  = getOperandValue(I.getOperand(0), SF);
  GenericValue Dest;

  switch (Ty->getPrimitiveID()) {
    IMPLEMENT_CAST_CASE(UByte  , (unsigned char));
    IMPLEMENT_CAST_CASE(SByte  , (  signed char));
    IMPLEMENT_CAST_CASE(UShort , (unsigned short));
    IMPLEMENT_CAST_CASE(Short  , (  signed short));
    IMPLEMENT_CAST_CASE(UInt   , (unsigned int ));
    IMPLEMENT_CAST_CASE(Int    , (  signed int ));
    IMPLEMENT_CAST_CASE(ULong  , (uint64_t));
    IMPLEMENT_CAST_CASE(Long   , ( int64_t));
    IMPLEMENT_CAST_CASE(Pointer, (PointerTy)(uint32_t));
    IMPLEMENT_CAST_CASE(Float  , (float));
    IMPLEMENT_CAST_CASE(Double , (double));
  default:
    cout << "Unhandled dest type for cast instruction: " << Ty << "\n";
  }
  SetValue(&I, Dest, SF);
}




//===----------------------------------------------------------------------===//
//                        Dispatch and Execution Code
//===----------------------------------------------------------------------===//

MethodInfo::MethodInfo(Function *F) : Annotation(MethodInfoAID) {
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

unsigned MethodInfo::getValueSlot(const Value *V) {
  unsigned Plane = V->getType()->getUniqueID();
  if (Plane >= NumPlaneElements.size())
    NumPlaneElements.resize(Plane+1, 0);
  return NumPlaneElements[Plane]++;
}


//===----------------------------------------------------------------------===//
// callMethod - Execute the specified function...
//
void Interpreter::callMethod(Function *M, const vector<GenericValue> &ArgVals) {
  assert((ECStack.empty() || ECStack.back().Caller == 0 || 
	  ECStack.back().Caller->getNumOperands()-1 == ArgVals.size()) &&
	 "Incorrect number of arguments passed into function call!");
  if (M->isExternal()) {
    GenericValue Result = callExternalMethod(M, ArgVals);
    const Type *RetTy = M->getReturnType();

    // Copy the result back into the result variable if we are not returning
    // void.
    if (RetTy != Type::VoidTy) {
      if (!ECStack.empty() && ECStack.back().Caller) {
        ExecutionContext &SF = ECStack.back();
        SetValue(SF.Caller, Result, SF);
      
        SF.Caller = 0;          // We returned from the call...
      } else if (!QuietMode) {
        // print it.
        CW << "Function " << M->getType() << " \"" << M->getName()
           << "\" returned ";
        print(RetTy, Result); 
        cout << "\n";
        
        if (RetTy->isIntegral())
          ExitCode = Result.SByteVal;   // Capture the exit code of the program
      }
    }

    return;
  }

  // Process the function, assigning instruction numbers to the instructions in
  // the function.  Also calculate the number of values for each type slot
  // active.
  //
  MethodInfo *MethInfo = (MethodInfo*)M->getOrCreateAnnotation(MethodInfoAID);
  ECStack.push_back(ExecutionContext());         // Make a new stack frame...

  ExecutionContext &StackFrame = ECStack.back(); // Fill it in...
  StackFrame.CurMethod = M;
  StackFrame.CurBB     = M->begin();
  StackFrame.CurInst   = StackFrame.CurBB->begin();
  StackFrame.MethInfo  = MethInfo;

  // Initialize the values to nothing...
  StackFrame.Values.resize(MethInfo->NumPlaneElements.size());
  for (unsigned i = 0; i < MethInfo->NumPlaneElements.size(); ++i) {
    StackFrame.Values[i].resize(MethInfo->NumPlaneElements[i]);

    // Taint the initial values of stuff
    memset(&StackFrame.Values[i][0], 42,
           MethInfo->NumPlaneElements[i]*sizeof(GenericValue));
  }

  StackFrame.PrevBB = 0;  // No previous BB for PHI nodes...


  // Run through the function arguments and initialize their values...
  assert(ArgVals.size() == M->asize() &&
         "Invalid number of values passed to function invocation!");
  unsigned i = 0;
  for (Function::aiterator AI = M->abegin(), E = M->aend(); AI != E; ++AI, ++i)
    SetValue(AI, ArgVals[i], StackFrame);
}

// executeInstruction - Interpret a single instruction, increment the "PC", and
// return true if the next instruction is a breakpoint...
//
bool Interpreter::executeInstruction() {
  assert(!ECStack.empty() && "No program running, cannot execute inst!");

  ExecutionContext &SF = ECStack.back();  // Current stack frame
  Instruction &I = *SF.CurInst++;         // Increment before execute

  if (Trace)
    CW << "Run:" << I;

  // Set a sigsetjmp buffer so that we can recover if an error happens during
  // instruction execution...
  //
  if (int SigNo = sigsetjmp(SignalRecoverBuffer, 1)) {
    --SF.CurInst;   // Back up to erroring instruction
    if (SigNo != SIGINT) {
      cout << "EXCEPTION OCCURRED [" << _sys_siglistp[SigNo] << "]:\n";
      printStackTrace();
      // If -abort-on-exception was specified, terminate LLI instead of trying
      // to debug it.
      //
      if (AbortOnExceptions) exit(1);
    } else if (SigNo == SIGINT) {
      cout << "CTRL-C Detected, execution halted.\n";
    }
    InInstruction = false;
    return true;
  }

  InInstruction = true;
  if (I.isBinaryOp()) {
    executeBinaryInst(cast<BinaryOperator>(I), SF);
  } else {
    switch (I.getOpcode()) {
      // Terminators
    case Instruction::Ret:     executeRetInst  (cast<ReturnInst>(I), SF); break;
    case Instruction::Br:      executeBrInst   (cast<BranchInst>(I), SF); break;
      // Memory Instructions
    case Instruction::Alloca:
    case Instruction::Malloc:  executeAllocInst((AllocationInst&)I, SF); break;
    case Instruction::Free:    executeFreeInst (cast<FreeInst> (I), SF); break;
    case Instruction::Load:    executeLoadInst (cast<LoadInst> (I), SF); break;
    case Instruction::Store:   executeStoreInst(cast<StoreInst>(I), SF); break;
    case Instruction::GetElementPtr:
                          executeGEPInst(cast<GetElementPtrInst>(I), SF); break;

      // Miscellaneous Instructions
    case Instruction::Call:    executeCallInst (cast<CallInst> (I), SF); break;
    case Instruction::PHINode: executePHINode  (cast<PHINode>  (I), SF); break;
    case Instruction::Shl:     executeShlInst  (cast<ShiftInst>(I), SF); break;
    case Instruction::Shr:     executeShrInst  (cast<ShiftInst>(I), SF); break;
    case Instruction::Cast:    executeCastInst (cast<CastInst> (I), SF); break;
    default:
      cout << "Don't know how to execute this instruction!\n-->" << I;
    }
  }
  InInstruction = false;
  
  // Reset the current frame location to the top of stack
  CurFrame = ECStack.size()-1;

  if (CurFrame == -1) return false;  // No breakpoint if no code

  // Return true if there is a breakpoint annotation on the instruction...
  return ECStack[CurFrame].CurInst->getAnnotation(BreakpointAID) != 0;
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
  if (ECStack.back().CurInst->getOpcode() == Instruction::Call) {
    unsigned StackSize = ECStack.size();
    // Step into the function...
    if (executeInstruction()) {
      // Hit a breakpoint, print current instruction, then return to user...
      cout << "Breakpoint hit!\n";
      printCurrentInstruction();
      return;
    }

    // If we we able to step into the function, finish it now.  We might not be
    // able the step into a function, if it's external for example.
    if (ECStack.size() != StackSize)
      finish(); // Finish executing the function...
    else
      printCurrentInstruction();

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
    if (ECStack.back().CurBB->begin() == ECStack.back().CurInst)  // print label
      WriteAsOperand(cout, ECStack.back().CurBB) << ":\n";

    Instruction &I = *ECStack.back().CurInst;
    InstNumber *IN = (InstNumber*)I.getAnnotation(SlotNumberAID);
    assert(IN && "Instruction has no numbering annotation!");
    cout << "#" << IN->InstNum << I;
  }
}

void Interpreter::printValue(const Type *Ty, GenericValue V) {
  switch (Ty->getPrimitiveID()) {
  case Type::BoolTyID:   cout << (V.BoolVal?"true":"false"); break;
  case Type::SByteTyID:
    cout << (int)V.SByteVal << " '" << V.SByteVal << "'";  break;
  case Type::UByteTyID:
    cout << (unsigned)V.UByteVal << " '" << V.UByteVal << "'";  break;
  case Type::ShortTyID:  cout << V.ShortVal;  break;
  case Type::UShortTyID: cout << V.UShortVal; break;
  case Type::IntTyID:    cout << V.IntVal;    break;
  case Type::UIntTyID:   cout << V.UIntVal;   break;
  case Type::LongTyID:   cout << (long)V.LongVal;   break;
  case Type::ULongTyID:  cout << (unsigned long)V.ULongVal;  break;
  case Type::FloatTyID:  cout << V.FloatVal;  break;
  case Type::DoubleTyID: cout << V.DoubleVal; break;
  case Type::PointerTyID:cout << (void*)V.PointerVal; break;
  default:
    cout << "- Don't know how to print value of this type!";
    break;
  }
}

void Interpreter::print(const Type *Ty, GenericValue V) {
  CW << Ty << " ";
  printValue(Ty, V);
}

void Interpreter::print(const std::string &Name) {
  Value *PickedVal = ChooseOneOption(Name, LookupMatchingNames(Name));
  if (!PickedVal) return;

  if (const Function *F = dyn_cast<const Function>(PickedVal)) {
    CW << F;  // Print the function
  } else if (const Type *Ty = dyn_cast<const Type>(PickedVal)) {
    CW << "type %" << Name << " = " << Ty->getDescription() << "\n";
  } else if (const BasicBlock *BB = dyn_cast<const BasicBlock>(PickedVal)) {
    CW << BB;   // Print the basic block
  } else {      // Otherwise there should be an annotation for the slot#
    print(PickedVal->getType(), 
          getOperandValue(PickedVal, ECStack[CurFrame]));
    cout << "\n";
  }
}

void Interpreter::infoValue(const std::string &Name) {
  Value *PickedVal = ChooseOneOption(Name, LookupMatchingNames(Name));
  if (!PickedVal) return;

  cout << "Value: ";
  print(PickedVal->getType(), 
        getOperandValue(PickedVal, ECStack[CurFrame]));
  cout << "\n";
  printOperandInfo(PickedVal, ECStack[CurFrame]);
}

// printStackFrame - Print information about the specified stack frame, or -1
// for the default one.
//
void Interpreter::printStackFrame(int FrameNo) {
  if (FrameNo == -1) FrameNo = CurFrame;
  Function *F = ECStack[FrameNo].CurMethod;
  const Type *RetTy = F->getReturnType();

  CW << ((FrameNo == CurFrame) ? '>' : '-') << "#" << FrameNo << ". "
     << (Value*)RetTy << " \"" << F->getName() << "\"(";
  
  unsigned i = 0;
  for (Function::aiterator I = F->abegin(), E = F->aend(); I != E; ++I, ++i) {
    if (i != 0) cout << ", ";
    CW << *I << "=";
    
    printValue(I->getType(), getOperandValue(I, ECStack[FrameNo]));
  }

  cout << ")\n";

  if (FrameNo != int(ECStack.size()-1)) {
    BasicBlock::iterator I = ECStack[FrameNo].CurInst;
    CW << --I;
  } else {
    CW << *ECStack[FrameNo].CurInst;
  }
}

