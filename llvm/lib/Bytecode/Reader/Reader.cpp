//===- Reader.cpp - Code to read bytecode files ---------------------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This library implements the functionality defined in llvm/Bytecode/Reader.h
//
// Note that this library should be as fast as possible, reentrant, and 
// threadsafe!!
//
// TODO: Allow passing in an option to ignore the symbol table
//
//===----------------------------------------------------------------------===//

#include "Reader.h"
#include "llvm/Bytecode/BytecodeHandler.h"
#include "llvm/BasicBlock.h"
#include "llvm/Constants.h"
#include "llvm/iMemory.h"
#include "llvm/iOther.h"
#include "llvm/iPHINode.h"
#include "llvm/iTerminators.h"
#include "llvm/Bytecode/Format.h"
#include "llvm/Support/GetElementPtrTypeIterator.h"
#include "Support/StringExtras.h"
#include <sstream>

using namespace llvm;

/// A convenience macro for calling the handler. Makes maintenance easier in
/// case the interface to handler methods changes.
#define HANDLE(method) \
  if ( Handler ) Handler->handle ## method

/// A convenience macro for handling parsing errors.
#define PARSE_ERROR(inserters) { \
    std::ostringstream errormsg; \
    errormsg << inserters; \
    throw std::string(errormsg.str()); \
  }

/// @brief A class for maintaining the slot number definition
/// as a placeholder for the actual definition.
template<class SuperType>
class PlaceholderDef : public SuperType {
  unsigned ID;
  PlaceholderDef();                       // DO NOT IMPLEMENT
  void operator=(const PlaceholderDef &); // DO NOT IMPLEMENT
public:
  PlaceholderDef(const Type *Ty, unsigned id) : SuperType(Ty), ID(id) {}
  unsigned getID() { return ID; }
};

struct ConstantPlaceHolderHelper : public ConstantExpr {
  ConstantPlaceHolderHelper(const Type *Ty)
    : ConstantExpr(Instruction::UserOp1, Constant::getNullValue(Ty), Ty) {}
};

typedef PlaceholderDef<ConstantPlaceHolderHelper>  ConstPHolder;

//===----------------------------------------------------------------------===//
// Bytecode Reading Methods
//===----------------------------------------------------------------------===//

inline bool BytecodeReader::moreInBlock() {
  return At < BlockEnd;
}

inline void BytecodeReader::checkPastBlockEnd(const char * block_name) {
  if ( At > BlockEnd )
    PARSE_ERROR("Attempt to read past the end of " << block_name << " block.");
}

inline void BytecodeReader::align32() {
  BufPtr Save = At;
  At = (const unsigned char *)((unsigned long)(At+3) & (~3UL));
  if ( At > Save ) 
    HANDLE(Alignment( At - Save ));
  if (At > BlockEnd) 
    PARSE_ERROR("Ran out of data while aligning!");
}

inline unsigned BytecodeReader::read_uint() {
  if (At+4 > BlockEnd) 
    PARSE_ERROR("Ran out of data reading uint!");
  At += 4;
  return At[-4] | (At[-3] << 8) | (At[-2] << 16) | (At[-1] << 24);
}

inline unsigned BytecodeReader::read_vbr_uint() {
  unsigned Shift = 0;
  unsigned Result = 0;
  BufPtr Save = At;
  
  do {
    if (At == BlockEnd) 
      PARSE_ERROR("Ran out of data reading vbr_uint!");
    Result |= (unsigned)((*At++) & 0x7F) << Shift;
    Shift += 7;
  } while (At[-1] & 0x80);
  HANDLE(VBR32(At-Save));
  return Result;
}

inline uint64_t BytecodeReader::read_vbr_uint64() {
  unsigned Shift = 0;
  uint64_t Result = 0;
  BufPtr Save = At;
  
  do {
    if (At == BlockEnd) 
      PARSE_ERROR("Ran out of data reading vbr_uint64!");
    Result |= (uint64_t)((*At++) & 0x7F) << Shift;
    Shift += 7;
  } while (At[-1] & 0x80);
  HANDLE(VBR64(At-Save));
  return Result;
}

inline int64_t BytecodeReader::read_vbr_int64() {
  uint64_t R = read_vbr_uint64();
  if (R & 1) {
    if (R != 1)
      return -(int64_t)(R >> 1);
    else   // There is no such thing as -0 with integers.  "-0" really means
           // 0x8000000000000000.
      return 1LL << 63;
  } else
    return  (int64_t)(R >> 1);
}

inline std::string BytecodeReader::read_str() {
  unsigned Size = read_vbr_uint();
  const unsigned char *OldAt = At;
  At += Size;
  if (At > BlockEnd)             // Size invalid?
    PARSE_ERROR("Ran out of data reading a string!");
  return std::string((char*)OldAt, Size);
}

inline void BytecodeReader::read_data(void *Ptr, void *End) {
  unsigned char *Start = (unsigned char *)Ptr;
  unsigned Amount = (unsigned char *)End - Start;
  if (At+Amount > BlockEnd) 
    PARSE_ERROR("Ran out of data!");
  std::copy(At, At+Amount, Start);
  At += Amount;
}

inline void BytecodeReader::read_block(unsigned &Type, unsigned &Size) {
  Type = read_uint();
  Size = read_uint();
  BlockStart = At;
  if ( At + Size > BlockEnd )
    PARSE_ERROR("Attempt to size a block past end of memory");
  BlockEnd = At + Size;
  HANDLE(Block( Type, BlockStart, Size ));
}

//===----------------------------------------------------------------------===//
// IR Lookup Methods
//===----------------------------------------------------------------------===//

inline bool BytecodeReader::hasImplicitNull(unsigned TyID ) {
  if (!hasExplicitPrimitiveZeros)
    return TyID != Type::LabelTyID && TyID != Type::TypeTyID &&
           TyID != Type::VoidTyID;
  return TyID >= Type::FirstDerivedTyID;
}

const Type *BytecodeReader::getType(unsigned ID) {
  if (ID < Type::FirstDerivedTyID)
    if (const Type *T = Type::getPrimitiveType((Type::TypeID)ID))
      return T;   // Asked for a primitive type...

  // Otherwise, derived types need offset...
  ID -= Type::FirstDerivedTyID;

  if (!CompactionTypes.empty()) {
    if (ID >= CompactionTypes.size())
      PARSE_ERROR("Type ID out of range for compaction table!");
    return CompactionTypes[ID];
  }

  // Is it a module-level type?
    if (ID < ModuleTypes.size())
      return ModuleTypes[ID].get();

    // Nope, is it a function-level type?
    ID -= ModuleTypes.size();
    if (ID < FunctionTypes.size())
      return FunctionTypes[ID].get();

    PARSE_ERROR("Illegal type reference!");
    return Type::VoidTy;
}

unsigned BytecodeReader::getTypeSlot(const Type *Ty) {
  if (Ty->isPrimitiveType())
    return Ty->getTypeID();

  // Scan the compaction table for the type if needed.
  if (!CompactionTypes.empty()) {
      std::vector<const Type*>::const_iterator I = 
	find(CompactionTypes.begin(), CompactionTypes.end(), Ty);

      if (I == CompactionTypes.end())
	PARSE_ERROR("Couldn't find type specified in compaction table!");
      return Type::FirstDerivedTyID + (&*I - &CompactionTypes[0]);
  }

  // Check the function level types first...
  TypeListTy::iterator I = find(FunctionTypes.begin(), FunctionTypes.end(), Ty);

  if (I != FunctionTypes.end())
    return Type::FirstDerivedTyID + ModuleTypes.size() +
	     (&*I - &FunctionTypes[0]);

  // Check the module level types now...
  I = find(ModuleTypes.begin(), ModuleTypes.end(), Ty);
  if (I == ModuleTypes.end())
    PARSE_ERROR("Didn't find type in ModuleTypes.");
  return Type::FirstDerivedTyID + (&*I - &ModuleTypes[0]);
}

const Type *BytecodeReader::getGlobalTableType(unsigned Slot) {
  if (Slot < Type::FirstDerivedTyID) {
    const Type *Ty = Type::getPrimitiveType((Type::TypeID)Slot);
    assert(Ty && "Not a primitive type ID?");
    return Ty;
  }
  Slot -= Type::FirstDerivedTyID;
  if (Slot >= ModuleTypes.size())
    PARSE_ERROR("Illegal compaction table type reference!");
  return ModuleTypes[Slot];
}

unsigned BytecodeReader::getGlobalTableTypeSlot(const Type *Ty) {
  if (Ty->isPrimitiveType())
    return Ty->getTypeID();
  TypeListTy::iterator I = find(ModuleTypes.begin(),
				      ModuleTypes.end(), Ty);
  if (I == ModuleTypes.end())
    PARSE_ERROR("Didn't find type in ModuleTypes.");
  return Type::FirstDerivedTyID + (&*I - &ModuleTypes[0]);
}

Value * BytecodeReader::getValue(unsigned type, unsigned oNum, bool Create) {
  assert(type != Type::TypeTyID && "getValue() cannot get types!");
  assert(type != Type::LabelTyID && "getValue() cannot get blocks!");
  unsigned Num = oNum;

  // If there is a compaction table active, it defines the low-level numbers.
  // If not, the module values define the low-level numbers.
  if (CompactionValues.size() > type && !CompactionValues[type].empty()) {
    if (Num < CompactionValues[type].size())
      return CompactionValues[type][Num];
    Num -= CompactionValues[type].size();
  } else {
    // By default, the global type id is the type id passed in
    unsigned GlobalTyID = type;

    // If the type plane was compactified, figure out the global type ID
    // by adding the derived type ids and the distance.
    if (CompactionValues.size() > Type::TypeTyID &&
	!CompactionTypes.empty() &&
	type >= Type::FirstDerivedTyID) {
      const Type *Ty = CompactionTypes[type-Type::FirstDerivedTyID];
      TypeListTy::iterator I = 
	find(ModuleTypes.begin(), ModuleTypes.end(), Ty);
      assert(I != ModuleTypes.end());
      GlobalTyID = Type::FirstDerivedTyID + (&*I - &ModuleTypes[0]);
    }

    if (hasImplicitNull(GlobalTyID)) {
      if (Num == 0)
	return Constant::getNullValue(getType(type));
      --Num;
    }

    if (GlobalTyID < ModuleValues.size() && ModuleValues[GlobalTyID]) {
      if (Num < ModuleValues[GlobalTyID]->size())
	return ModuleValues[GlobalTyID]->getOperand(Num);
      Num -= ModuleValues[GlobalTyID]->size();
    }
  }

  if (FunctionValues.size() > type && 
      FunctionValues[type] && 
      Num < FunctionValues[type]->size())
    return FunctionValues[type]->getOperand(Num);

  if (!Create) return 0;  // Do not create a placeholder?

  std::pair<unsigned,unsigned> KeyValue(type, oNum);
  ForwardReferenceMap::iterator I = ForwardReferences.lower_bound(KeyValue);
  if (I != ForwardReferences.end() && I->first == KeyValue)
    return I->second;   // We have already created this placeholder

  Value *Val = new Argument(getType(type));
  ForwardReferences.insert(I, std::make_pair(KeyValue, Val));
  return Val;
}

Value* BytecodeReader::getGlobalTableValue(const Type *Ty, unsigned SlotNo) {
  // FIXME: getTypeSlot is inefficient!
  unsigned TyID = getGlobalTableTypeSlot(Ty);
  
  if (TyID != Type::LabelTyID) {
    if (SlotNo == 0)
      return Constant::getNullValue(Ty);
    --SlotNo;
  }

  if (TyID >= ModuleValues.size() || ModuleValues[TyID] == 0 ||
      SlotNo >= ModuleValues[TyID]->size()) {
    PARSE_ERROR("Corrupt compaction table entry!" 
	<< TyID << ", " << SlotNo << ": " << ModuleValues.size() << ", "
	<< (void*)ModuleValues[TyID] << ", "
	<< ModuleValues[TyID]->size() << "\n");
  }
  return ModuleValues[TyID]->getOperand(SlotNo);
}

Constant* BytecodeReader::getConstantValue(unsigned TypeSlot, unsigned Slot) {
  if (Value *V = getValue(TypeSlot, Slot, false))
    if (Constant *C = dyn_cast<Constant>(V))
      return C;   // If we already have the value parsed, just return it
    else if (GlobalValue *GV = dyn_cast<GlobalValue>(V))
      // ConstantPointerRef's are an abomination, but at least they don't have
      // to infest bytecode files.
      return ConstantPointerRef::get(GV);
    else
      PARSE_ERROR("Reference of a value is expected to be a constant!");

  const Type *Ty = getType(TypeSlot);
  std::pair<const Type*, unsigned> Key(Ty, Slot);
  ConstantRefsType::iterator I = ConstantFwdRefs.lower_bound(Key);

  if (I != ConstantFwdRefs.end() && I->first == Key) {
    return I->second;
  } else {
    // Create a placeholder for the constant reference and
    // keep track of the fact that we have a forward ref to recycle it
    Constant *C = new ConstPHolder(Ty, Slot);
    
    // Keep track of the fact that we have a forward ref to recycle it
    ConstantFwdRefs.insert(I, std::make_pair(Key, C));
    return C;
  }
}

//===----------------------------------------------------------------------===//
// IR Construction Methods
//===----------------------------------------------------------------------===//

unsigned BytecodeReader::insertValue(
    Value *Val, unsigned type, ValueTable &ValueTab) {
  assert((!isa<Constant>(Val) || !cast<Constant>(Val)->isNullValue()) ||
	  !hasImplicitNull(type) &&
	 "Cannot read null values from bytecode!");
  assert(type != Type::TypeTyID && "Types should never be insertValue'd!");

  if (ValueTab.size() <= type)
    ValueTab.resize(type+1);

  if (!ValueTab[type]) ValueTab[type] = new ValueList();

  ValueTab[type]->push_back(Val);

  bool HasOffset = hasImplicitNull(type);
  return ValueTab[type]->size()-1 + HasOffset;
}

void BytecodeReader::insertArguments(Function* F ) {
  const FunctionType *FT = F->getFunctionType();
  Function::aiterator AI = F->abegin();
  for (FunctionType::param_iterator It = FT->param_begin();
       It != FT->param_end(); ++It, ++AI)
    insertValue(AI, getTypeSlot(AI->getType()), FunctionValues);
}

//===----------------------------------------------------------------------===//
// Bytecode Parsing Methods
//===----------------------------------------------------------------------===//

void BytecodeReader::ParseInstruction(std::vector<unsigned> &Oprnds,
                                              BasicBlock* BB) {
  BufPtr SaveAt = At;

  // Clear instruction data
  Oprnds.clear();
  unsigned iType = 0;
  unsigned Opcode = 0;
  unsigned Op = read_uint();

  // bits   Instruction format:        Common to all formats
  // --------------------------
  // 01-00: Opcode type, fixed to 1.
  // 07-02: Opcode
  Opcode    = (Op >> 2) & 63;
  Oprnds.resize((Op >> 0) & 03);

  // Extract the operands
  switch (Oprnds.size()) {
  case 1:
    // bits   Instruction format:
    // --------------------------
    // 19-08: Resulting type plane
    // 31-20: Operand #1 (if set to (2^12-1), then zero operands)
    //
    iType   = (Op >>  8) & 4095;
    Oprnds[0] = (Op >> 20) & 4095;
    if (Oprnds[0] == 4095)    // Handle special encoding for 0 operands...
      Oprnds.resize(0);
    break;
  case 2:
    // bits   Instruction format:
    // --------------------------
    // 15-08: Resulting type plane
    // 23-16: Operand #1
    // 31-24: Operand #2  
    //
    iType   = (Op >>  8) & 255;
    Oprnds[0] = (Op >> 16) & 255;
    Oprnds[1] = (Op >> 24) & 255;
    break;
  case 3:
    // bits   Instruction format:
    // --------------------------
    // 13-08: Resulting type plane
    // 19-14: Operand #1
    // 25-20: Operand #2
    // 31-26: Operand #3
    //
    iType   = (Op >>  8) & 63;
    Oprnds[0] = (Op >> 14) & 63;
    Oprnds[1] = (Op >> 20) & 63;
    Oprnds[2] = (Op >> 26) & 63;
    break;
  case 0:
    At -= 4;  // Hrm, try this again...
    Opcode = read_vbr_uint();
    Opcode >>= 2;
    iType = read_vbr_uint();

    unsigned NumOprnds = read_vbr_uint();
    Oprnds.resize(NumOprnds);

    if (NumOprnds == 0)
      PARSE_ERROR("Zero-argument instruction found; this is invalid.");

    for (unsigned i = 0; i != NumOprnds; ++i)
      Oprnds[i] = read_vbr_uint();
    align32();
    break;
  }

  // Get the type of the instruction
  const Type *InstTy = getType(iType);

  // Hae enough to inform the handler now
  HANDLE(Instruction(Opcode, InstTy, Oprnds, At-SaveAt));

  // Declare the resulting instruction we'll build.
  Instruction *Result = 0;

  // Handle binary operators
  if (Opcode >= Instruction::BinaryOpsBegin &&
      Opcode <  Instruction::BinaryOpsEnd  && Oprnds.size() == 2)
    Result = BinaryOperator::create((Instruction::BinaryOps)Opcode,
                                    getValue(iType, Oprnds[0]),
                                    getValue(iType, Oprnds[1]));

  switch (Opcode) {
  default: 
    if (Result == 0) PARSE_ERROR("Illegal instruction read!");
    break;
  case Instruction::VAArg:
    Result = new VAArgInst(getValue(iType, Oprnds[0]), getType(Oprnds[1]));
    break;
  case Instruction::VANext:
    Result = new VANextInst(getValue(iType, Oprnds[0]), getType(Oprnds[1]));
    break;
  case Instruction::Cast:
    Result = new CastInst(getValue(iType, Oprnds[0]), getType(Oprnds[1]));
    break;
  case Instruction::Select:
    Result = new SelectInst(getValue(Type::BoolTyID, Oprnds[0]),
                            getValue(iType, Oprnds[1]),
                            getValue(iType, Oprnds[2]));
    break;
  case Instruction::PHI: {
    if (Oprnds.size() == 0 || (Oprnds.size() & 1))
      PARSE_ERROR("Invalid phi node encountered!\n");

    PHINode *PN = new PHINode(InstTy);
    PN->op_reserve(Oprnds.size());
    for (unsigned i = 0, e = Oprnds.size(); i != e; i += 2)
      PN->addIncoming(getValue(iType, Oprnds[i]), getBasicBlock(Oprnds[i+1]));
    Result = PN;
    break;
  }

  case Instruction::Shl:
  case Instruction::Shr:
    Result = new ShiftInst((Instruction::OtherOps)Opcode,
                           getValue(iType, Oprnds[0]),
                           getValue(Type::UByteTyID, Oprnds[1]));
    break;
  case Instruction::Ret:
    if (Oprnds.size() == 0)
      Result = new ReturnInst();
    else if (Oprnds.size() == 1)
      Result = new ReturnInst(getValue(iType, Oprnds[0]));
    else
      PARSE_ERROR("Unrecognized instruction!");
    break;

  case Instruction::Br:
    if (Oprnds.size() == 1)
      Result = new BranchInst(getBasicBlock(Oprnds[0]));
    else if (Oprnds.size() == 3)
      Result = new BranchInst(getBasicBlock(Oprnds[0]), 
	  getBasicBlock(Oprnds[1]), getValue(Type::BoolTyID , Oprnds[2]));
    else
      PARSE_ERROR("Invalid number of operands for a 'br' instruction!");
    break;
  case Instruction::Switch: {
    if (Oprnds.size() & 1)
      PARSE_ERROR("Switch statement with odd number of arguments!");

    SwitchInst *I = new SwitchInst(getValue(iType, Oprnds[0]),
                                   getBasicBlock(Oprnds[1]));
    for (unsigned i = 2, e = Oprnds.size(); i != e; i += 2)
      I->addCase(cast<Constant>(getValue(iType, Oprnds[i])),
                 getBasicBlock(Oprnds[i+1]));
    Result = I;
    break;
  }

  case Instruction::Call: {
    if (Oprnds.size() == 0)
      PARSE_ERROR("Invalid call instruction encountered!");

    Value *F = getValue(iType, Oprnds[0]);

    // Check to make sure we have a pointer to function type
    const PointerType *PTy = dyn_cast<PointerType>(F->getType());
    if (PTy == 0) PARSE_ERROR("Call to non function pointer value!");
    const FunctionType *FTy = dyn_cast<FunctionType>(PTy->getElementType());
    if (FTy == 0) PARSE_ERROR("Call to non function pointer value!");

    std::vector<Value *> Params;
    if (!FTy->isVarArg()) {
      FunctionType::param_iterator It = FTy->param_begin();

      for (unsigned i = 1, e = Oprnds.size(); i != e; ++i) {
        if (It == FTy->param_end())
          PARSE_ERROR("Invalid call instruction!");
        Params.push_back(getValue(getTypeSlot(*It++), Oprnds[i]));
      }
      if (It != FTy->param_end())
        PARSE_ERROR("Invalid call instruction!");
    } else {
      Oprnds.erase(Oprnds.begin(), Oprnds.begin()+1);

      unsigned FirstVariableOperand;
      if (Oprnds.size() < FTy->getNumParams())
        PARSE_ERROR("Call instruction missing operands!");

      // Read all of the fixed arguments
      for (unsigned i = 0, e = FTy->getNumParams(); i != e; ++i)
        Params.push_back(getValue(getTypeSlot(FTy->getParamType(i)),Oprnds[i]));
      
      FirstVariableOperand = FTy->getNumParams();

      if ((Oprnds.size()-FirstVariableOperand) & 1) // Must be pairs of type/value
        PARSE_ERROR("Invalid call instruction!");
        
      for (unsigned i = FirstVariableOperand, e = Oprnds.size(); 
	   i != e; i += 2)
        Params.push_back(getValue(Oprnds[i], Oprnds[i+1]));
    }

    Result = new CallInst(F, Params);
    break;
  }
  case Instruction::Invoke: {
    if (Oprnds.size() < 3) PARSE_ERROR("Invalid invoke instruction!");
    Value *F = getValue(iType, Oprnds[0]);

    // Check to make sure we have a pointer to function type
    const PointerType *PTy = dyn_cast<PointerType>(F->getType());
    if (PTy == 0) PARSE_ERROR("Invoke to non function pointer value!");
    const FunctionType *FTy = dyn_cast<FunctionType>(PTy->getElementType());
    if (FTy == 0) PARSE_ERROR("Invoke to non function pointer value!");

    std::vector<Value *> Params;
    BasicBlock *Normal, *Except;

    if (!FTy->isVarArg()) {
      Normal = getBasicBlock(Oprnds[1]);
      Except = getBasicBlock(Oprnds[2]);

      FunctionType::param_iterator It = FTy->param_begin();
      for (unsigned i = 3, e = Oprnds.size(); i != e; ++i) {
        if (It == FTy->param_end())
          PARSE_ERROR("Invalid invoke instruction!");
        Params.push_back(getValue(getTypeSlot(*It++), Oprnds[i]));
      }
      if (It != FTy->param_end())
        PARSE_ERROR("Invalid invoke instruction!");
    } else {
      Oprnds.erase(Oprnds.begin(), Oprnds.begin()+1);

      Normal = getBasicBlock(Oprnds[0]);
      Except = getBasicBlock(Oprnds[1]);
      
      unsigned FirstVariableArgument = FTy->getNumParams()+2;
      for (unsigned i = 2; i != FirstVariableArgument; ++i)
        Params.push_back(getValue(getTypeSlot(FTy->getParamType(i-2)),
                                  Oprnds[i]));
      
      if (Oprnds.size()-FirstVariableArgument & 1) // Must be type/value pairs
        PARSE_ERROR("Invalid invoke instruction!");

      for (unsigned i = FirstVariableArgument; i < Oprnds.size(); i += 2)
        Params.push_back(getValue(Oprnds[i], Oprnds[i+1]));
    }

    Result = new InvokeInst(F, Normal, Except, Params);
    break;
  }
  case Instruction::Malloc:
    if (Oprnds.size() > 2) PARSE_ERROR("Invalid malloc instruction!");
    if (!isa<PointerType>(InstTy))
      PARSE_ERROR("Invalid malloc instruction!");

    Result = new MallocInst(cast<PointerType>(InstTy)->getElementType(),
                            Oprnds.size() ? getValue(Type::UIntTyID,
                                                   Oprnds[0]) : 0);
    break;

  case Instruction::Alloca:
    if (Oprnds.size() > 2) PARSE_ERROR("Invalid alloca instruction!");
    if (!isa<PointerType>(InstTy))
      PARSE_ERROR("Invalid alloca instruction!");

    Result = new AllocaInst(cast<PointerType>(InstTy)->getElementType(),
                            Oprnds.size() ? getValue(Type::UIntTyID, 
			    Oprnds[0]) :0);
    break;
  case Instruction::Free:
    if (!isa<PointerType>(InstTy))
      PARSE_ERROR("Invalid free instruction!");
    Result = new FreeInst(getValue(iType, Oprnds[0]));
    break;
  case Instruction::GetElementPtr: {
    if (Oprnds.size() == 0 || !isa<PointerType>(InstTy))
      PARSE_ERROR("Invalid getelementptr instruction!");

    std::vector<Value*> Idx;

    const Type *NextTy = InstTy;
    for (unsigned i = 1, e = Oprnds.size(); i != e; ++i) {
      const CompositeType *TopTy = dyn_cast_or_null<CompositeType>(NextTy);
      if (!TopTy) PARSE_ERROR("Invalid getelementptr instruction!"); 

      unsigned ValIdx = Oprnds[i];
      unsigned IdxTy = 0;
      if (!hasRestrictedGEPTypes) {
        // Struct indices are always uints, sequential type indices can be any
        // of the 32 or 64-bit integer types.  The actual choice of type is
        // encoded in the low two bits of the slot number.
        if (isa<StructType>(TopTy))
          IdxTy = Type::UIntTyID;
        else {
          switch (ValIdx & 3) {
          default:
          case 0: IdxTy = Type::UIntTyID; break;
          case 1: IdxTy = Type::IntTyID; break;
          case 2: IdxTy = Type::ULongTyID; break;
          case 3: IdxTy = Type::LongTyID; break;
          }
          ValIdx >>= 2;
        }
      } else {
        IdxTy = isa<StructType>(TopTy) ? Type::UByteTyID : Type::LongTyID;
      }

      Idx.push_back(getValue(IdxTy, ValIdx));

      // Convert ubyte struct indices into uint struct indices.
      if (isa<StructType>(TopTy) && hasRestrictedGEPTypes)
        if (ConstantUInt *C = dyn_cast<ConstantUInt>(Idx.back()))
          Idx[Idx.size()-1] = ConstantExpr::getCast(C, Type::UIntTy);

      NextTy = GetElementPtrInst::getIndexedType(InstTy, Idx, true);
    }

    Result = new GetElementPtrInst(getValue(iType, Oprnds[0]), Idx);
    break;
  }

  case 62:   // volatile load
  case Instruction::Load:
    if (Oprnds.size() != 1 || !isa<PointerType>(InstTy))
      PARSE_ERROR("Invalid load instruction!");
    Result = new LoadInst(getValue(iType, Oprnds[0]), "", Opcode == 62);
    break;

  case 63:   // volatile store 
  case Instruction::Store: {
    if (!isa<PointerType>(InstTy) || Oprnds.size() != 2)
      PARSE_ERROR("Invalid store instruction!");

    Value *Ptr = getValue(iType, Oprnds[1]);
    const Type *ValTy = cast<PointerType>(Ptr->getType())->getElementType();
    Result = new StoreInst(getValue(getTypeSlot(ValTy), Oprnds[0]), Ptr,
                           Opcode == 63);
    break;
  }
  case Instruction::Unwind:
    if (Oprnds.size() != 0) PARSE_ERROR("Invalid unwind instruction!");
    Result = new UnwindInst();
    break;
  }  // end switch(Opcode) 

  unsigned TypeSlot;
  if (Result->getType() == InstTy)
    TypeSlot = iType;
  else
    TypeSlot = getTypeSlot(Result->getType());

  insertValue(Result, TypeSlot, FunctionValues);
  BB->getInstList().push_back(Result);
}

/// getBasicBlock - Get a particular numbered basic block, which might be a
/// forward reference.  This works together with ParseBasicBlock to handle these
/// forward references in a clean manner.
BasicBlock *BytecodeReader::getBasicBlock(unsigned ID) {
  // Make sure there is room in the table...
  if (ParsedBasicBlocks.size() <= ID) ParsedBasicBlocks.resize(ID+1);

  // First check to see if this is a backwards reference, i.e., ParseBasicBlock
  // has already created this block, or if the forward reference has already
  // been created.
  if (ParsedBasicBlocks[ID])
    return ParsedBasicBlocks[ID];

  // Otherwise, the basic block has not yet been created.  Do so and add it to
  // the ParsedBasicBlocks list.
  return ParsedBasicBlocks[ID] = new BasicBlock();
}

/// ParseBasicBlock - In LLVM 1.0 bytecode files, we used to output one
/// basicblock at a time.  This method reads in one of the basicblock packets.
BasicBlock *BytecodeReader::ParseBasicBlock( unsigned BlockNo) {
  HANDLE(BasicBlockBegin( BlockNo ));

  BasicBlock *BB = 0;

  if (ParsedBasicBlocks.size() == BlockNo)
    ParsedBasicBlocks.push_back(BB = new BasicBlock());
  else if (ParsedBasicBlocks[BlockNo] == 0)
    BB = ParsedBasicBlocks[BlockNo] = new BasicBlock();
  else
    BB = ParsedBasicBlocks[BlockNo];

  std::vector<unsigned> Operands;
  while ( moreInBlock() )
    ParseInstruction(Operands, BB);

  HANDLE(BasicBlockEnd( BlockNo ));
  return BB;
}

/// ParseInstructionList - Parse all of the BasicBlock's & Instruction's in the
/// body of a function.  In post 1.0 bytecode files, we no longer emit basic
/// block individually, in order to avoid per-basic-block overhead.
unsigned BytecodeReader::ParseInstructionList(Function* F) {
  unsigned BlockNo = 0;
  std::vector<unsigned> Args;

  while ( moreInBlock() ) {
    HANDLE(BasicBlockBegin( BlockNo ));
    BasicBlock *BB;
    if (ParsedBasicBlocks.size() == BlockNo)
      ParsedBasicBlocks.push_back(BB = new BasicBlock());
    else if (ParsedBasicBlocks[BlockNo] == 0)
      BB = ParsedBasicBlocks[BlockNo] = new BasicBlock();
    else
      BB = ParsedBasicBlocks[BlockNo];
    HANDLE(BasicBlockEnd( BlockNo ));
    ++BlockNo;
    F->getBasicBlockList().push_back(BB);

    // Read instructions into this basic block until we get to a terminator
    while ( moreInBlock() && !BB->getTerminator())
      ParseInstruction(Args, BB);

    if (!BB->getTerminator())
      PARSE_ERROR("Non-terminated basic block found!");
  }

  return BlockNo;
}

void BytecodeReader::ParseSymbolTable(Function *CurrentFunction,
				      SymbolTable *ST) {
  HANDLE(SymbolTableBegin(CurrentFunction,ST));

  // Allow efficient basic block lookup by number.
  std::vector<BasicBlock*> BBMap;
  if (CurrentFunction)
    for (Function::iterator I = CurrentFunction->begin(),
           E = CurrentFunction->end(); I != E; ++I)
      BBMap.push_back(I);

  while ( moreInBlock() ) {
    // Symtab block header: [num entries][type id number]
    unsigned NumEntries = read_vbr_uint();
    unsigned Typ = read_vbr_uint();
    const Type *Ty = getType(Typ);

    for (unsigned i = 0; i != NumEntries; ++i) {
      // Symtab entry: [def slot #][name]
      unsigned slot = read_vbr_uint();
      std::string Name = read_str();

      Value *V = 0;
      if (Typ == Type::TypeTyID)
        V = (Value*)getType(slot);
      else if (Typ == Type::LabelTyID) {
        if (slot < BBMap.size())
          V = BBMap[slot];
      } else {
        V = getValue(Typ, slot, false); // Find mapping...
      }
      if (V == 0)
        PARSE_ERROR("Failed value look-up for name '" << Name << "'");

      V->setName(Name, ST);
    }
  }
  checkPastBlockEnd("Symbol Table");
  HANDLE(SymbolTableEnd());
}

void BytecodeReader::ParseCompactionTable() {

  HANDLE(CompactionTableBegin());

  while ( moreInBlock() ) {
    unsigned NumEntries = read_vbr_uint();
    unsigned Ty;

    if ((NumEntries & 3) == 3) {
      NumEntries >>= 2;
      Ty = read_vbr_uint();
    } else {
      Ty = NumEntries >> 2;
      NumEntries &= 3;
    }

    if (Ty >= CompactionValues.size())
      CompactionValues.resize(Ty+1);

    if (!CompactionValues[Ty].empty())
      PARSE_ERROR("Compaction table plane contains multiple entries!");

    HANDLE(CompactionTablePlane( Ty, NumEntries ));

    if (Ty == Type::TypeTyID) {
      for (unsigned i = 0; i != NumEntries; ++i) {
        unsigned TypeSlot = read_vbr_uint();
        const Type *Typ = getGlobalTableType(TypeSlot);
        CompactionTypes.push_back(Typ);
        HANDLE(CompactionTableType( i, TypeSlot, Typ ));
      }
      CompactionTypes.resize(NumEntries+Type::FirstDerivedTyID);
    } else {
      const Type *Typ = getType(Ty);
      // Push the implicit zero
      CompactionValues[Ty].push_back(Constant::getNullValue(Typ));
      for (unsigned i = 0; i != NumEntries; ++i) {
        unsigned ValSlot = read_vbr_uint();
        Value *V = getGlobalTableValue(Typ, ValSlot);
        CompactionValues[Ty].push_back(V);
        HANDLE(CompactionTableValue( i, Ty, ValSlot, Typ ));
      }
    }
  }
  HANDLE(CompactionTableEnd());
}
    
// Parse a single type constant.
const Type *BytecodeReader::ParseTypeConstant() {
  unsigned PrimType = read_vbr_uint();

  const Type *Result = 0;
  if ((Result = Type::getPrimitiveType((Type::TypeID)PrimType)))
    return Result;
  
  switch (PrimType) {
  case Type::FunctionTyID: {
    const Type *RetType = getType(read_vbr_uint());

    unsigned NumParams = read_vbr_uint();

    std::vector<const Type*> Params;
    while (NumParams--)
      Params.push_back(getType(read_vbr_uint()));

    bool isVarArg = Params.size() && Params.back() == Type::VoidTy;
    if (isVarArg) Params.pop_back();

    Result = FunctionType::get(RetType, Params, isVarArg);
    break;
  }
  case Type::ArrayTyID: {
    unsigned ElTyp = read_vbr_uint();
    const Type *ElementType = getType(ElTyp);

    unsigned NumElements = read_vbr_uint();

    Result =  ArrayType::get(ElementType, NumElements);
    break;
  }
  case Type::StructTyID: {
    std::vector<const Type*> Elements;
    unsigned Typ = read_vbr_uint();
    while (Typ) {         // List is terminated by void/0 typeid
      Elements.push_back(getType(Typ));
      Typ = read_vbr_uint();
    }

    Result = StructType::get(Elements);
    break;
  }
  case Type::PointerTyID: {
    unsigned ElTyp = read_vbr_uint();
    Result = PointerType::get(getType(ElTyp));
    break;
  }

  case Type::OpaqueTyID: {
    Result = OpaqueType::get();
    break;
  }

  default:
    PARSE_ERROR("Don't know how to deserialize primitive type" << PrimType << "\n");
    break;
  }
  HANDLE(Type( Result ));
  return Result;
}

// ParseTypeConstants - We have to use this weird code to handle recursive
// types.  We know that recursive types will only reference the current slab of
// values in the type plane, but they can forward reference types before they
// have been read.  For example, Type #0 might be '{ Ty#1 }' and Type #1 might
// be 'Ty#0*'.  When reading Type #0, type number one doesn't exist.  To fix
// this ugly problem, we pessimistically insert an opaque type for each type we
// are about to read.  This means that forward references will resolve to
// something and when we reread the type later, we can replace the opaque type
// with a new resolved concrete type.
//
void BytecodeReader::ParseTypeConstants(TypeListTy &Tab, unsigned NumEntries){
  assert(Tab.size() == 0 && "should not have read type constants in before!");

  // Insert a bunch of opaque types to be resolved later...
  Tab.reserve(NumEntries);
  for (unsigned i = 0; i != NumEntries; ++i)
    Tab.push_back(OpaqueType::get());

  // Loop through reading all of the types.  Forward types will make use of the
  // opaque types just inserted.
  //
  for (unsigned i = 0; i != NumEntries; ++i) {
    const Type *NewTy = ParseTypeConstant(), *OldTy = Tab[i].get();
    if (NewTy == 0) PARSE_ERROR("Couldn't parse type!");

    // Don't directly push the new type on the Tab. Instead we want to replace 
    // the opaque type we previously inserted with the new concrete value. This
    // approach helps with forward references to types. The refinement from the
    // abstract (opaque) type to the new type causes all uses of the abstract
    // type to use the concrete type (NewTy). This will also cause the opaque
    // type to be deleted.
    cast<DerivedType>(const_cast<Type*>(OldTy))->refineAbstractTypeTo(NewTy);

    // This should have replaced the old opaque type with the new type in the
    // value table... or with a preexisting type that was already in the system.
    // Let's just make sure it did.
    assert(Tab[i] != OldTy && "refineAbstractType didn't work!");
  }
}

Constant *BytecodeReader::ParseConstantValue( unsigned TypeID) {
  // We must check for a ConstantExpr before switching by type because
  // a ConstantExpr can be of any type, and has no explicit value.
  // 
  // 0 if not expr; numArgs if is expr
  unsigned isExprNumArgs = read_vbr_uint();
  
  if (isExprNumArgs) {
    // FIXME: Encoding of constant exprs could be much more compact!
    std::vector<Constant*> ArgVec;
    ArgVec.reserve(isExprNumArgs);
    unsigned Opcode = read_vbr_uint();
    
    // Read the slot number and types of each of the arguments
    for (unsigned i = 0; i != isExprNumArgs; ++i) {
      unsigned ArgValSlot = read_vbr_uint();
      unsigned ArgTypeSlot = read_vbr_uint();
      
      // Get the arg value from its slot if it exists, otherwise a placeholder
      ArgVec.push_back(getConstantValue(ArgTypeSlot, ArgValSlot));
    }
    
    // Construct a ConstantExpr of the appropriate kind
    if (isExprNumArgs == 1) {           // All one-operand expressions
      assert(Opcode == Instruction::Cast);
      Constant* Result = ConstantExpr::getCast(ArgVec[0], getType(TypeID));
      HANDLE(ConstantExpression(Opcode, ArgVec, Result));
      return Result;
    } else if (Opcode == Instruction::GetElementPtr) { // GetElementPtr
      std::vector<Constant*> IdxList(ArgVec.begin()+1, ArgVec.end());

      if (hasRestrictedGEPTypes) {
        const Type *BaseTy = ArgVec[0]->getType();
        generic_gep_type_iterator<std::vector<Constant*>::iterator>
          GTI = gep_type_begin(BaseTy, IdxList.begin(), IdxList.end()),
          E = gep_type_end(BaseTy, IdxList.begin(), IdxList.end());
        for (unsigned i = 0; GTI != E; ++GTI, ++i)
          if (isa<StructType>(*GTI)) {
            if (IdxList[i]->getType() != Type::UByteTy)
              PARSE_ERROR("Invalid index for getelementptr!");
            IdxList[i] = ConstantExpr::getCast(IdxList[i], Type::UIntTy);
          }
      }

      Constant* Result = ConstantExpr::getGetElementPtr(ArgVec[0], IdxList);
      HANDLE(ConstantExpression(Opcode, ArgVec, Result));
      return Result;
    } else if (Opcode == Instruction::Select) {
      assert(ArgVec.size() == 3);
      Constant* Result = ConstantExpr::getSelect(ArgVec[0], ArgVec[1], 
	                                         ArgVec[2]);
      HANDLE(ConstantExpression(Opcode, ArgVec, Result));
      return Result;
    } else {                            // All other 2-operand expressions
      Constant* Result = ConstantExpr::get(Opcode, ArgVec[0], ArgVec[1]);
      HANDLE(ConstantExpression(Opcode, ArgVec, Result));
      return Result;
    }
  }
  
  // Ok, not an ConstantExpr.  We now know how to read the given type...
  const Type *Ty = getType(TypeID);
  switch (Ty->getTypeID()) {
  case Type::BoolTyID: {
    unsigned Val = read_vbr_uint();
    if (Val != 0 && Val != 1) 
      PARSE_ERROR("Invalid boolean value read.");
    Constant* Result = ConstantBool::get(Val == 1);
    HANDLE(ConstantValue(Result));
    return Result;
  }

  case Type::UByteTyID:   // Unsigned integer types...
  case Type::UShortTyID:
  case Type::UIntTyID: {
    unsigned Val = read_vbr_uint();
    if (!ConstantUInt::isValueValidForType(Ty, Val)) 
      PARSE_ERROR("Invalid unsigned byte/short/int read.");
    Constant* Result =  ConstantUInt::get(Ty, Val);
    HANDLE(ConstantValue(Result));
    return Result;
  }

  case Type::ULongTyID: {
    Constant* Result = ConstantUInt::get(Ty, read_vbr_uint64());
    HANDLE(ConstantValue(Result));
    return Result;
  }

  case Type::SByteTyID:   // Signed integer types...
  case Type::ShortTyID:
  case Type::IntTyID: {
  case Type::LongTyID:
    int64_t Val = read_vbr_int64();
    if (!ConstantSInt::isValueValidForType(Ty, Val)) 
      PARSE_ERROR("Invalid signed byte/short/int/long read.");
    Constant* Result = ConstantSInt::get(Ty, Val);
    HANDLE(ConstantValue(Result));
    return Result;
  }

  case Type::FloatTyID: {
    float F;
    read_data(&F, &F+1);
    Constant* Result = ConstantFP::get(Ty, F);
    HANDLE(ConstantValue(Result));
    return Result;
  }

  case Type::DoubleTyID: {
    double Val;
    read_data(&Val, &Val+1);
    Constant* Result = ConstantFP::get(Ty, Val);
    HANDLE(ConstantValue(Result));
    return Result;
  }

  case Type::TypeTyID:
    PARSE_ERROR("Type constants shouldn't live in constant table!");
    break;

  case Type::ArrayTyID: {
    const ArrayType *AT = cast<ArrayType>(Ty);
    unsigned NumElements = AT->getNumElements();
    unsigned TypeSlot = getTypeSlot(AT->getElementType());
    std::vector<Constant*> Elements;
    Elements.reserve(NumElements);
    while (NumElements--)     // Read all of the elements of the constant.
      Elements.push_back(getConstantValue(TypeSlot,
                                          read_vbr_uint()));
    Constant* Result = ConstantArray::get(AT, Elements);
    HANDLE(ConstantArray(AT, Elements, TypeSlot, Result));
    return Result;
  }

  case Type::StructTyID: {
    const StructType *ST = cast<StructType>(Ty);

    std::vector<Constant *> Elements;
    Elements.reserve(ST->getNumElements());
    for (unsigned i = 0; i != ST->getNumElements(); ++i)
      Elements.push_back(getConstantValue(ST->getElementType(i),
                                          read_vbr_uint()));

    Constant* Result = ConstantStruct::get(ST, Elements);
    HANDLE(ConstantStruct(ST, Elements, Result));
    return Result;
  }    

  case Type::PointerTyID: {  // ConstantPointerRef value...
    const PointerType *PT = cast<PointerType>(Ty);
    unsigned Slot = read_vbr_uint();
    
    // Check to see if we have already read this global variable...
    Value *Val = getValue(TypeID, Slot, false);
    GlobalValue *GV;
    if (Val) {
      if (!(GV = dyn_cast<GlobalValue>(Val))) 
        PARSE_ERROR("Value of ConstantPointerRef not in ValueTable!");
    } else {
      PARSE_ERROR("Forward references are not allowed here.");
    }
    
    Constant* Result = ConstantPointerRef::get(GV);
    HANDLE(ConstantPointer(PT, Slot, GV, Result));
    return Result;
  }

  default:
    PARSE_ERROR("Don't know how to deserialize constant value of type '"+
                      Ty->getDescription());
    break;
  }
}

void BytecodeReader::ResolveReferencesToConstant(Constant *NewV, unsigned Slot){
  ConstantRefsType::iterator I =
    ConstantFwdRefs.find(std::make_pair(NewV->getType(), Slot));
  if (I == ConstantFwdRefs.end()) return;   // Never forward referenced?

  Value *PH = I->second;   // Get the placeholder...
  PH->replaceAllUsesWith(NewV);
  delete PH;                               // Delete the old placeholder
  ConstantFwdRefs.erase(I);                // Remove the map entry for it
}

void BytecodeReader::ParseStringConstants(unsigned NumEntries, ValueTable &Tab){
  for (; NumEntries; --NumEntries) {
    unsigned Typ = read_vbr_uint();
    const Type *Ty = getType(Typ);
    if (!isa<ArrayType>(Ty))
      PARSE_ERROR("String constant data invalid!");
    
    const ArrayType *ATy = cast<ArrayType>(Ty);
    if (ATy->getElementType() != Type::SByteTy &&
        ATy->getElementType() != Type::UByteTy)
      PARSE_ERROR("String constant data invalid!");
    
    // Read character data.  The type tells us how long the string is.
    char Data[ATy->getNumElements()]; 
    read_data(Data, Data+ATy->getNumElements());

    std::vector<Constant*> Elements(ATy->getNumElements());
    if (ATy->getElementType() == Type::SByteTy)
      for (unsigned i = 0, e = ATy->getNumElements(); i != e; ++i)
        Elements[i] = ConstantSInt::get(Type::SByteTy, (signed char)Data[i]);
    else
      for (unsigned i = 0, e = ATy->getNumElements(); i != e; ++i)
        Elements[i] = ConstantUInt::get(Type::UByteTy, (unsigned char)Data[i]);

    // Create the constant, inserting it as needed.
    Constant *C = ConstantArray::get(ATy, Elements);
    unsigned Slot = insertValue(C, Typ, Tab);
    ResolveReferencesToConstant(C, Slot);
    HANDLE(ConstantString(cast<ConstantArray>(C)));
  }
}

void BytecodeReader::ParseConstantPool(ValueTable &Tab, 
                                       TypeListTy &TypeTab) {
  HANDLE(GlobalConstantsBegin());
  while ( moreInBlock() ) {
    unsigned NumEntries = read_vbr_uint();
    unsigned Typ = read_vbr_uint();
    if (Typ == Type::TypeTyID) {
      ParseTypeConstants(TypeTab, NumEntries);
    } else if (Typ == Type::VoidTyID) {
      assert(&Tab == &ModuleValues && "Cannot read strings in functions!");
      ParseStringConstants(NumEntries, Tab);
    } else {
      for (unsigned i = 0; i < NumEntries; ++i) {
        Constant *C = ParseConstantValue(Typ);
        assert(C && "ParseConstantValue returned NULL!");
        unsigned Slot = insertValue(C, Typ, Tab);

        // If we are reading a function constant table, make sure that we adjust
        // the slot number to be the real global constant number.
        //
        if (&Tab != &ModuleValues && Typ < ModuleValues.size() &&
            ModuleValues[Typ])
          Slot += ModuleValues[Typ]->size();
        ResolveReferencesToConstant(C, Slot);
      }
    }
  }
  checkPastBlockEnd("Constant Pool");
  HANDLE(GlobalConstantsEnd());
}

void BytecodeReader::ParseFunctionBody(Function* F ) {

  unsigned FuncSize = BlockEnd - At;
  GlobalValue::LinkageTypes Linkage = GlobalValue::ExternalLinkage;

  unsigned LinkageType = read_vbr_uint();
  switch (LinkageType) {
  case 0: Linkage = GlobalValue::ExternalLinkage; break;
  case 1: Linkage = GlobalValue::WeakLinkage; break;
  case 2: Linkage = GlobalValue::AppendingLinkage; break;
  case 3: Linkage = GlobalValue::InternalLinkage; break;
  case 4: Linkage = GlobalValue::LinkOnceLinkage; break;
  default:
    PARSE_ERROR("Invalid linkage type for Function.");
    Linkage = GlobalValue::InternalLinkage;
    break;
  }

  F->setLinkage( Linkage );
  HANDLE(FunctionBegin(F,FuncSize));

  // Keep track of how many basic blocks we have read in...
  unsigned BlockNum = 0;
  bool InsertedArguments = false;

  BufPtr MyEnd = BlockEnd;
  while ( At < MyEnd ) {
    unsigned Type, Size;
    BufPtr OldAt = At;
    read_block(Type, Size);

    switch (Type) {
    case BytecodeFormat::ConstantPool:
      if (!InsertedArguments) {
        // Insert arguments into the value table before we parse the first basic
        // block in the function, but after we potentially read in the
        // compaction table.
	insertArguments(F);
        InsertedArguments = true;
      }

      ParseConstantPool(FunctionValues, FunctionTypes);
      break;

    case BytecodeFormat::CompactionTable:
      ParseCompactionTable();
      break;

    case BytecodeFormat::BasicBlock: {
      if (!InsertedArguments) {
        // Insert arguments into the value table before we parse the first basic
        // block in the function, but after we potentially read in the
        // compaction table.
	insertArguments(F);
        InsertedArguments = true;
      }

      BasicBlock *BB = ParseBasicBlock(BlockNum++);
      F->getBasicBlockList().push_back(BB);
      break;
    }

    case BytecodeFormat::InstructionList: {
      // Insert arguments into the value table before we parse the instruction
      // list for the function, but after we potentially read in the compaction
      // table.
      if (!InsertedArguments) {
	insertArguments(F);
        InsertedArguments = true;
      }

      if (BlockNum) 
	PARSE_ERROR("Already parsed basic blocks!");
      BlockNum = ParseInstructionList(F);
      break;
    }

    case BytecodeFormat::SymbolTable:
      ParseSymbolTable(F, &F->getSymbolTable());
      break;

    default:
      At += Size;
      if (OldAt > At) 
        PARSE_ERROR("Wrapped around reading bytecode.");
      break;
    }
    BlockEnd = MyEnd;

    // Malformed bc file if read past end of block.
    align32();
  }

  // Make sure there were no references to non-existant basic blocks.
  if (BlockNum != ParsedBasicBlocks.size())
    PARSE_ERROR("Illegal basic block operand reference");

  ParsedBasicBlocks.clear();

  // Resolve forward references.  Replace any uses of a forward reference value
  // with the real value.

  // replaceAllUsesWith is very inefficient for instructions which have a LARGE
  // number of operands.  PHI nodes often have forward references, and can also
  // often have a very large number of operands.
  //
  // FIXME: REEVALUATE.  replaceAllUsesWith is _much_ faster now, and this code
  // should be simplified back to using it!
  //
  std::map<Value*, Value*> ForwardRefMapping;
  for (std::map<std::pair<unsigned,unsigned>, Value*>::iterator 
         I = ForwardReferences.begin(), E = ForwardReferences.end();
       I != E; ++I)
    ForwardRefMapping[I->second] = getValue(I->first.first, I->first.second,
                                            false);

  for (Function::iterator BB = F->begin(), E = F->end(); BB != E; ++BB)
    for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; ++I)
      for (unsigned i = 0, e = I->getNumOperands(); i != e; ++i)
        if (Argument *A = dyn_cast<Argument>(I->getOperand(i))) {
          std::map<Value*, Value*>::iterator It = ForwardRefMapping.find(A);
          if (It != ForwardRefMapping.end()) I->setOperand(i, It->second);
        }

  while (!ForwardReferences.empty()) {
    std::map<std::pair<unsigned,unsigned>, Value*>::iterator I =
      ForwardReferences.begin();
    Value *PlaceHolder = I->second;
    ForwardReferences.erase(I);

    // Now that all the uses are gone, delete the placeholder...
    // If we couldn't find a def (error case), then leak a little
    // memory, because otherwise we can't remove all uses!
    delete PlaceHolder;
  }

  // Clear out function-level types...
  FunctionTypes.clear();
  CompactionTypes.clear();
  CompactionValues.clear();
  freeTable(FunctionValues);

  HANDLE(FunctionEnd(F));
}

void BytecodeReader::ParseFunctionLazily() {
  if (FunctionSignatureList.empty())
    PARSE_ERROR("FunctionSignatureList empty!");

  Function *Func = FunctionSignatureList.back();
  FunctionSignatureList.pop_back();

  // Save the information for future reading of the function
  LazyFunctionLoadMap[Func] = LazyFunctionInfo(BlockStart, BlockEnd);

  // Pretend we've `parsed' this function
  At = BlockEnd;
}

void BytecodeReader::ParseFunction(Function* Func) {
  // Find {start, end} pointers and slot in the map. If not there, we're done.
  LazyFunctionMap::iterator Fi = LazyFunctionLoadMap.find(Func);

  // Make sure we found it
  if ( Fi == LazyFunctionLoadMap.end() ) {
    PARSE_ERROR("Unrecognized function of type " << Func->getType()->getDescription());
    return;
  }

  BlockStart = At = Fi->second.Buf;
  BlockEnd = Fi->second.EndBuf;
  assert(Fi->first == Func);

  LazyFunctionLoadMap.erase(Fi);

  this->ParseFunctionBody( Func );
}

void BytecodeReader::ParseAllFunctionBodies() {
  LazyFunctionMap::iterator Fi = LazyFunctionLoadMap.begin();
  LazyFunctionMap::iterator Fe = LazyFunctionLoadMap.end();

  while ( Fi != Fe ) {
    Function* Func = Fi->first;
    BlockStart = At = Fi->second.Buf;
    BlockEnd = Fi->second.EndBuf;
    this->ParseFunctionBody(Func);
    ++Fi;
  }
}

void BytecodeReader::ParseGlobalTypes() {
  ValueTable T;
  ParseConstantPool(T, ModuleTypes);
}

void BytecodeReader::ParseModuleGlobalInfo() {

  HANDLE(ModuleGlobalsBegin());

  // Read global variables...
  unsigned VarType = read_vbr_uint();
  while (VarType != Type::VoidTyID) { // List is terminated by Void
    // VarType Fields: bit0 = isConstant, bit1 = hasInitializer, bit2,3,4 =
    // Linkage, bit4+ = slot#
    unsigned SlotNo = VarType >> 5;
    unsigned LinkageID = (VarType >> 2) & 7;
    bool isConstant = VarType & 1;
    bool hasInitializer = VarType & 2;
    GlobalValue::LinkageTypes Linkage;

    switch (LinkageID) {
    case 0: Linkage = GlobalValue::ExternalLinkage;  break;
    case 1: Linkage = GlobalValue::WeakLinkage;      break;
    case 2: Linkage = GlobalValue::AppendingLinkage; break;
    case 3: Linkage = GlobalValue::InternalLinkage;  break;
    case 4: Linkage = GlobalValue::LinkOnceLinkage;  break;
    default: 
      PARSE_ERROR("Unknown linkage type: " << LinkageID);
      Linkage = GlobalValue::InternalLinkage;
      break;
    }

    const Type *Ty = getType(SlotNo);
    if ( !Ty ) {
      PARSE_ERROR("Global has no type! SlotNo=" << SlotNo);
    }

    if ( !isa<PointerType>(Ty)) {
      PARSE_ERROR("Global not a pointer type! Ty= " << Ty->getDescription());
    }

    const Type *ElTy = cast<PointerType>(Ty)->getElementType();

    // Create the global variable...
    GlobalVariable *GV = new GlobalVariable(ElTy, isConstant, Linkage,
                                            0, "", TheModule);
    insertValue(GV, SlotNo, ModuleValues);

    unsigned initSlot = 0;
    if (hasInitializer) {   
      initSlot = read_vbr_uint();
      GlobalInits.push_back(std::make_pair(GV, initSlot));
    }

    // Notify handler about the global value.
    HANDLE(GlobalVariable( ElTy, isConstant, Linkage, SlotNo, initSlot ));

    // Get next item
    VarType = read_vbr_uint();
  }

  // Read the function objects for all of the functions that are coming
  unsigned FnSignature = read_vbr_uint();
  while (FnSignature != Type::VoidTyID) { // List is terminated by Void
    const Type *Ty = getType(FnSignature);
    if (!isa<PointerType>(Ty) ||
        !isa<FunctionType>(cast<PointerType>(Ty)->getElementType())) {
      PARSE_ERROR( "Function not a pointer to function type! Ty = " +
                        Ty->getDescription());
      // FIXME: what should Ty be if handler continues?
    }

    // We create functions by passing the underlying FunctionType to create...
    const FunctionType* FTy = 
      cast<FunctionType>(cast<PointerType>(Ty)->getElementType());

    // Insert the place hodler
    Function* Func = new Function(FTy, GlobalValue::InternalLinkage, 
	                          "", TheModule);
    insertValue(Func, FnSignature, ModuleValues);

    // Save this for later so we know type of lazily instantiated functions
    FunctionSignatureList.push_back(Func);

    HANDLE(FunctionDeclaration(Func));

    // Get Next function signature
    FnSignature = read_vbr_uint();
  }

  if (hasInconsistentModuleGlobalInfo)
    align32();

  // Now that the function signature list is set up, reverse it so that we can 
  // remove elements efficiently from the back of the vector.
  std::reverse(FunctionSignatureList.begin(), FunctionSignatureList.end());

  // This is for future proofing... in the future extra fields may be added that
  // we don't understand, so we transparently ignore them.
  //
  At = BlockEnd;

  HANDLE(ModuleGlobalsEnd());
}

void BytecodeReader::ParseVersionInfo() {
  unsigned Version = read_vbr_uint();

  // Unpack version number: low four bits are for flags, top bits = version
  Module::Endianness  Endianness;
  Module::PointerSize PointerSize;
  Endianness  = (Version & 1) ? Module::BigEndian : Module::LittleEndian;
  PointerSize = (Version & 2) ? Module::Pointer64 : Module::Pointer32;

  bool hasNoEndianness = Version & 4;
  bool hasNoPointerSize = Version & 8;
  
  RevisionNum = Version >> 4;

  // Default values for the current bytecode version
  hasInconsistentModuleGlobalInfo = false;
  hasExplicitPrimitiveZeros = false;
  hasRestrictedGEPTypes = false;

  switch (RevisionNum) {
  case 0:               //  LLVM 1.0, 1.1 release version
    // Base LLVM 1.0 bytecode format.
    hasInconsistentModuleGlobalInfo = true;
    hasExplicitPrimitiveZeros = true;
    // FALL THROUGH
  case 1:               // LLVM 1.2 release version
    // LLVM 1.2 added explicit support for emitting strings efficiently.

    // Also, it fixed the problem where the size of the ModuleGlobalInfo block
    // included the size for the alignment at the end, where the rest of the
    // blocks did not.

    // LLVM 1.2 and before required that GEP indices be ubyte constants for
    // structures and longs for sequential types.
    hasRestrictedGEPTypes = true;

    // FALL THROUGH
  case 2:               // LLVM 1.3 release version
    break;

  default:
    PARSE_ERROR("Unknown bytecode version number: " << RevisionNum);
  }

  if (hasNoEndianness) Endianness  = Module::AnyEndianness;
  if (hasNoPointerSize) PointerSize = Module::AnyPointerSize;

  HANDLE(VersionInfo(RevisionNum, Endianness, PointerSize ));
}

void BytecodeReader::ParseModule() {
  unsigned Type, Size;

  FunctionSignatureList.clear(); // Just in case...

  // Read into instance variables...
  ParseVersionInfo();
  align32(); /// FIXME: Is this redundant? VI is first and 4 bytes!

  bool SeenModuleGlobalInfo = false;
  bool SeenGlobalTypePlane = false;
  BufPtr MyEnd = BlockEnd;
  while (At < MyEnd) {
    BufPtr OldAt = At;
    read_block(Type, Size);

    switch (Type) {

    case BytecodeFormat::GlobalTypePlane:
      if ( SeenGlobalTypePlane )
        PARSE_ERROR("Two GlobalTypePlane Blocks Encountered!");

      ParseGlobalTypes();
      SeenGlobalTypePlane = true;
      break;

    case BytecodeFormat::ModuleGlobalInfo: 
      if ( SeenModuleGlobalInfo )
        PARSE_ERROR("Two ModuleGlobalInfo Blocks Encountered!");
      ParseModuleGlobalInfo();
      SeenModuleGlobalInfo = true;
      break;

    case BytecodeFormat::ConstantPool:
      ParseConstantPool(ModuleValues, ModuleTypes);
      break;

    case BytecodeFormat::Function:
      ParseFunctionLazily();
      break;

    case BytecodeFormat::SymbolTable:
      ParseSymbolTable(0, &TheModule->getSymbolTable());
      break;

    default:
      At += Size;
      if (OldAt > At) {
        PARSE_ERROR("Unexpected Block of Type" << Type << "encountered!" );
      }
      break;
    }
    BlockEnd = MyEnd;
    align32();
  }

  // After the module constant pool has been read, we can safely initialize
  // global variables...
  while (!GlobalInits.empty()) {
    GlobalVariable *GV = GlobalInits.back().first;
    unsigned Slot = GlobalInits.back().second;
    GlobalInits.pop_back();

    // Look up the initializer value...
    // FIXME: Preserve this type ID!

    const llvm::PointerType* GVType = GV->getType();
    unsigned TypeSlot = getTypeSlot(GVType->getElementType());
    if (Constant *CV = getConstantValue(TypeSlot, Slot)) {
      if (GV->hasInitializer()) 
        PARSE_ERROR("Global *already* has an initializer?!");
      HANDLE(GlobalInitializer(GV,CV));
      GV->setInitializer(CV);
    } else
      PARSE_ERROR("Cannot find initializer value.");
  }

  /// Make sure we pulled them all out. If we didn't then there's a declaration
  /// but a missing body. That's not allowed.
  if (!FunctionSignatureList.empty())
    PARSE_ERROR(
      "Function declared, but bytecode stream ended before definition");
}

void BytecodeReader::ParseBytecode(
       BufPtr Buf, unsigned Length,
       const std::string &ModuleID) {

  try {
    At = MemStart = BlockStart = Buf;
    MemEnd = BlockEnd = Buf + Length;

    // Create the module
    TheModule = new Module(ModuleID);

    HANDLE(Start(TheModule, Length));

    // Read and check signature...
    unsigned Sig = read_uint();
    if (Sig != ('l' | ('l' << 8) | ('v' << 16) | ('m' << 24))) {
      PARSE_ERROR("Invalid bytecode signature: " << Sig);
    }


    // Tell the handler we're starting a module
    HANDLE(ModuleBegin(ModuleID));

    // Get the module block and size and verify
    unsigned Type, Size;
    read_block(Type, Size);
    if ( Type != BytecodeFormat::Module ) {
      PARSE_ERROR("Expected Module Block! At: " << unsigned(intptr_t(At))
	<< ", Type:" << Type << ", Size:" << Size);
    }
    if ( At + Size != MemEnd ) {
      PARSE_ERROR("Invalid Top Level Block Length! At: " 
	<< unsigned(intptr_t(At)) << ", Type:" << Type << ", Size:" << Size);
    }

    // Parse the module contents
    this->ParseModule();

    // Tell the handler we're done
    HANDLE(ModuleEnd(ModuleID));

    // Check for missing functions
    if ( hasFunctions() )
      PARSE_ERROR("Function expected, but bytecode stream ended!");

    // Tell the handler we're
    HANDLE(Finish());

  } catch (std::string& errstr ) {
    HANDLE(Error(errstr));
    freeState();
    delete TheModule;
    TheModule = 0;
    throw;
  } catch (...) {
    std::string msg("Unknown Exception Occurred");
    HANDLE(Error(msg));
    freeState();
    delete TheModule;
    TheModule = 0;
    throw msg;
  }
}

//===----------------------------------------------------------------------===//
//=== Default Implementations of Handler Methods
//===----------------------------------------------------------------------===//

BytecodeHandler::~BytecodeHandler() {}
void BytecodeHandler::handleError(const std::string& str ) { }
void BytecodeHandler::handleStart(Module*m, unsigned Length ) { }
void BytecodeHandler::handleFinish() { }
void BytecodeHandler::handleModuleBegin(const std::string& id) { }
void BytecodeHandler::handleModuleEnd(const std::string& id) { }
void BytecodeHandler::handleVersionInfo( unsigned char RevisionNum,
  Module::Endianness Endianness, Module::PointerSize PointerSize) { }
void BytecodeHandler::handleModuleGlobalsBegin() { }
void BytecodeHandler::handleGlobalVariable( 
  const Type* ElemType, bool isConstant, GlobalValue::LinkageTypes,
  unsigned SlotNo, unsigned initSlot ) { }
void BytecodeHandler::handleType( const Type* Ty ) {}
void BytecodeHandler::handleFunctionDeclaration( 
  Function* Func ) {}
void BytecodeHandler::handleGlobalInitializer(GlobalVariable*, Constant* ) {}
void BytecodeHandler::handleModuleGlobalsEnd() { } 
void BytecodeHandler::handleCompactionTableBegin() { } 
void BytecodeHandler::handleCompactionTablePlane( 
  unsigned Ty, unsigned NumEntries) {}
void BytecodeHandler::handleCompactionTableType( 
  unsigned i, unsigned TypSlot, const Type* ) {}
void BytecodeHandler::handleCompactionTableValue( 
  unsigned i, unsigned TypSlot, unsigned ValSlot, const Type* ) {}
void BytecodeHandler::handleCompactionTableEnd() { }
void BytecodeHandler::handleSymbolTableBegin(Function*, SymbolTable*) { }
void BytecodeHandler::handleSymbolTablePlane( unsigned Ty, unsigned NumEntries, 
  const Type* Typ) { }
void BytecodeHandler::handleSymbolTableType( unsigned i, unsigned slot, 
  const std::string& name ) { }
void BytecodeHandler::handleSymbolTableValue( unsigned i, unsigned slot, 
  const std::string& name ) { }
void BytecodeHandler::handleSymbolTableEnd() { }
void BytecodeHandler::handleFunctionBegin( Function* Func, 
  unsigned Size ) {}
void BytecodeHandler::handleFunctionEnd( Function* Func) { }
void BytecodeHandler::handleBasicBlockBegin( unsigned blocknum) { } 
bool BytecodeHandler::handleInstruction( unsigned Opcode, const Type* iType,
  std::vector<unsigned>& Operands, unsigned Size) { 
    return Instruction::isTerminator(Opcode); 
  }
void BytecodeHandler::handleBasicBlockEnd(unsigned blocknum) { }
void BytecodeHandler::handleGlobalConstantsBegin() { }
void BytecodeHandler::handleConstantExpression( unsigned Opcode, 
  std::vector<Constant*> ArgVec, Constant* Val) { }
void BytecodeHandler::handleConstantValue( Constant * c ) { }
void BytecodeHandler::handleConstantArray( const ArrayType* AT, 
  std::vector<Constant*>& Elements, unsigned TypeSlot, Constant* Val ) { }
void BytecodeHandler::handleConstantStruct( const StructType* ST,
  std::vector<Constant*>& ElementSlots, Constant* Val ) { }
void BytecodeHandler::handleConstantPointer( 
  const PointerType* PT, unsigned TypeSlot, GlobalValue*, Constant* Val) { }
void BytecodeHandler::handleConstantString( const ConstantArray* CA ) {}
void BytecodeHandler::handleGlobalConstantsEnd() {}
void BytecodeHandler::handleAlignment(unsigned numBytes) {}
void BytecodeHandler::handleBlock(
  unsigned BType, const unsigned char* StartPtr, unsigned Size) {}
void BytecodeHandler::handleVBR32(unsigned Size ) {}
void BytecodeHandler::handleVBR64(unsigned Size ) {}

// vim: sw=2
