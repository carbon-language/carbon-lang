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
#include "llvm/CallingConv.h"
#include "llvm/Constants.h"
#include "llvm/InlineAsm.h"
#include "llvm/Instructions.h"
#include "llvm/ParameterAttributes.h"
#include "llvm/TypeSymbolTable.h"
#include "llvm/Bytecode/Format.h"
#include "llvm/Config/alloca.h"
#include "llvm/Support/GetElementPtrTypeIterator.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include <sstream>
#include <algorithm>
using namespace llvm;

namespace {
  /// @brief A class for maintaining the slot number definition
  /// as a placeholder for the actual definition for forward constants defs.
  class ConstantPlaceHolder : public ConstantExpr {
    ConstantPlaceHolder();                       // DO NOT IMPLEMENT
    void operator=(const ConstantPlaceHolder &); // DO NOT IMPLEMENT
  public:
    Use Op;
    ConstantPlaceHolder(const Type *Ty)
      : ConstantExpr(Ty, Instruction::UserOp1, &Op, 1),
        Op(UndefValue::get(Type::Int32Ty), this) {
    }
  };
}

// Provide some details on error
inline void BytecodeReader::error(const std::string& err) {
  ErrorMsg = err + " (Vers=" + itostr(RevisionNum) + ", Pos=" 
    + itostr(At-MemStart) + ")";
  if (Handler) Handler->handleError(ErrorMsg);
  longjmp(context,1);
}

//===----------------------------------------------------------------------===//
// Bytecode Reading Methods
//===----------------------------------------------------------------------===//

/// Determine if the current block being read contains any more data.
inline bool BytecodeReader::moreInBlock() {
  return At < BlockEnd;
}

/// Throw an error if we've read past the end of the current block
inline void BytecodeReader::checkPastBlockEnd(const char * block_name) {
  if (At > BlockEnd)
    error(std::string("Attempt to read past the end of ") + block_name +
          " block.");
}

/// Read a whole unsigned integer
inline unsigned BytecodeReader::read_uint() {
  if (At+4 > BlockEnd)
    error("Ran out of data reading uint!");
  At += 4;
  return At[-4] | (At[-3] << 8) | (At[-2] << 16) | (At[-1] << 24);
}

/// Read a variable-bit-rate encoded unsigned integer
inline unsigned BytecodeReader::read_vbr_uint() {
  unsigned Shift = 0;
  unsigned Result = 0;

  do {
    if (At == BlockEnd)
      error("Ran out of data reading vbr_uint!");
    Result |= (unsigned)((*At++) & 0x7F) << Shift;
    Shift += 7;
  } while (At[-1] & 0x80);
  return Result;
}

/// Read a variable-bit-rate encoded unsigned 64-bit integer.
inline uint64_t BytecodeReader::read_vbr_uint64() {
  unsigned Shift = 0;
  uint64_t Result = 0;

  do {
    if (At == BlockEnd)
      error("Ran out of data reading vbr_uint64!");
    Result |= (uint64_t)((*At++) & 0x7F) << Shift;
    Shift += 7;
  } while (At[-1] & 0x80);
  return Result;
}

/// Read a variable-bit-rate encoded signed 64-bit integer.
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

/// Read a pascal-style string (length followed by text)
inline std::string BytecodeReader::read_str() {
  unsigned Size = read_vbr_uint();
  const unsigned char *OldAt = At;
  At += Size;
  if (At > BlockEnd)             // Size invalid?
    error("Ran out of data reading a string!");
  return std::string((char*)OldAt, Size);
}

void BytecodeReader::read_str(SmallVectorImpl<char> &StrData) {
  StrData.clear();
  unsigned Size = read_vbr_uint();
  const unsigned char *OldAt = At;
  At += Size;
  if (At > BlockEnd)             // Size invalid?
    error("Ran out of data reading a string!");
  StrData.append(OldAt, At);
}


/// Read an arbitrary block of data
inline void BytecodeReader::read_data(void *Ptr, void *End) {
  unsigned char *Start = (unsigned char *)Ptr;
  unsigned Amount = (unsigned char *)End - Start;
  if (At+Amount > BlockEnd)
    error("Ran out of data!");
  std::copy(At, At+Amount, Start);
  At += Amount;
}

/// Read a float value in little-endian order
inline void BytecodeReader::read_float(float& FloatVal) {
  /// FIXME: This isn't optimal, it has size problems on some platforms
  /// where FP is not IEEE.
  FloatVal = BitsToFloat(At[0] | (At[1] << 8) | (At[2] << 16) | (At[3] << 24));
  At+=sizeof(uint32_t);
}

/// Read a double value in little-endian order
inline void BytecodeReader::read_double(double& DoubleVal) {
  /// FIXME: This isn't optimal, it has size problems on some platforms
  /// where FP is not IEEE.
  DoubleVal = BitsToDouble((uint64_t(At[0]) <<  0) | (uint64_t(At[1]) << 8) |
                           (uint64_t(At[2]) << 16) | (uint64_t(At[3]) << 24) |
                           (uint64_t(At[4]) << 32) | (uint64_t(At[5]) << 40) |
                           (uint64_t(At[6]) << 48) | (uint64_t(At[7]) << 56));
  At+=sizeof(uint64_t);
}

/// Read a block header and obtain its type and size
inline void BytecodeReader::read_block(unsigned &Type, unsigned &Size) {
  Size = read_uint(); // Read the header
  Type = Size & 0x1F; // mask low order five bits to get type
  Size >>= 5;         // high order 27 bits is the size
  BlockStart = At;
  if (At + Size > BlockEnd)
    error("Attempt to size a block past end of memory");
  BlockEnd = At + Size;
  if (Handler) Handler->handleBlock(Type, BlockStart, Size);
}

//===----------------------------------------------------------------------===//
// IR Lookup Methods
//===----------------------------------------------------------------------===//

/// Determine if a type id has an implicit null value
inline bool BytecodeReader::hasImplicitNull(unsigned TyID) {
  return TyID != Type::LabelTyID && TyID != Type::VoidTyID;
}

/// Obtain a type given a typeid and account for things like function level vs 
/// module level, and the offsetting for the primitive types.
const Type *BytecodeReader::getType(unsigned ID) {
  if (ID <= Type::LastPrimitiveTyID)
    if (const Type *T = Type::getPrimitiveType((Type::TypeID)ID))
      return T;   // Asked for a primitive type...

  // Otherwise, derived types need offset...
  ID -= Type::FirstDerivedTyID;

  // Is it a module-level type?
  if (ID < ModuleTypes.size())
    return ModuleTypes[ID].get();

  // Nope, is it a function-level type?
  ID -= ModuleTypes.size();
  if (ID < FunctionTypes.size())
    return FunctionTypes[ID].get();

  error("Illegal type reference!");
  return Type::VoidTy;
}

/// This method just saves some coding. It uses read_vbr_uint to read in a 
/// type id, errors that its not the type type, and then calls getType to 
/// return the type value.
inline const Type* BytecodeReader::readType() {
  return getType(read_vbr_uint());
}

/// Get the slot number associated with a type accounting for primitive
/// types and function level vs module level.
unsigned BytecodeReader::getTypeSlot(const Type *Ty) {
  if (Ty->isPrimitiveType())
    return Ty->getTypeID();

  // Check the function level types first...
  TypeListTy::iterator I = std::find(FunctionTypes.begin(),
                                     FunctionTypes.end(), Ty);

  if (I != FunctionTypes.end())
    return Type::FirstDerivedTyID + ModuleTypes.size() +
           (&*I - &FunctionTypes[0]);

  // If we don't have our cache yet, build it now.
  if (ModuleTypeIDCache.empty()) {
    unsigned N = 0;
    ModuleTypeIDCache.reserve(ModuleTypes.size());
    for (TypeListTy::iterator I = ModuleTypes.begin(), E = ModuleTypes.end();
         I != E; ++I, ++N)
      ModuleTypeIDCache.push_back(std::make_pair(*I, N));
    
    std::sort(ModuleTypeIDCache.begin(), ModuleTypeIDCache.end());
  }
  
  // Binary search the cache for the entry.
  std::vector<std::pair<const Type*, unsigned> >::iterator IT =
    std::lower_bound(ModuleTypeIDCache.begin(), ModuleTypeIDCache.end(),
                     std::make_pair(Ty, 0U));
  if (IT == ModuleTypeIDCache.end() || IT->first != Ty)
    error("Didn't find type in ModuleTypes.");
    
  return Type::FirstDerivedTyID + IT->second;
}

/// Retrieve a value of a given type and slot number, possibly creating
/// it if it doesn't already exist.
Value * BytecodeReader::getValue(unsigned type, unsigned oNum, bool Create) {
  assert(type != Type::LabelTyID && "getValue() cannot get blocks!");
  unsigned Num = oNum;

  // By default, the global type id is the type id passed in
  unsigned GlobalTyID = type;

  if (hasImplicitNull(GlobalTyID)) {
    const Type *Ty = getType(type);
    if (!isa<OpaqueType>(Ty)) {
      if (Num == 0)
        return Constant::getNullValue(Ty);
      --Num;
    }
  }

  if (GlobalTyID < ModuleValues.size()) 
    if (ValueList *Globals = ModuleValues[GlobalTyID]) {
      if (Num < Globals->size())
        return Globals->getOperand(Num);
      Num -= Globals->size();
    }

  if (type < FunctionValues.size())
    if (ValueList *Locals = FunctionValues[type])
      if (Num < Locals->size())
        return Locals->getOperand(Num);

  // We did not find the value.
  
  if (!Create) return 0;  // Do not create a placeholder?

  // Did we already create a place holder?
  std::pair<unsigned,unsigned> KeyValue(type, oNum);
  ForwardReferenceMap::iterator I = ForwardReferences.lower_bound(KeyValue);
  if (I != ForwardReferences.end() && I->first == KeyValue)
    return I->second;   // We have already created this placeholder

  // If the type exists (it should)
  if (const Type* Ty = getType(type)) {
    // Create the place holder
    Value *Val = new Argument(Ty);
    ForwardReferences.insert(I, std::make_pair(KeyValue, Val));
    return Val;
  }
  error("Can't create placeholder for value of type slot #" + utostr(type));
  return 0; // just silence warning, error calls longjmp
}


/// Just like getValue, except that it returns a null pointer
/// only on error.  It always returns a constant (meaning that if the value is
/// defined, but is not a constant, that is an error).  If the specified
/// constant hasn't been parsed yet, a placeholder is defined and used.
/// Later, after the real value is parsed, the placeholder is eliminated.
Constant* BytecodeReader::getConstantValue(unsigned TypeSlot, unsigned Slot) {
  if (Value *V = getValue(TypeSlot, Slot, false))
    if (Constant *C = dyn_cast<Constant>(V))
      return C;   // If we already have the value parsed, just return it
    else
      error("Value for slot " + utostr(Slot) +
            " is expected to be a constant!");

  std::pair<unsigned, unsigned> Key(TypeSlot, Slot);
  ConstantRefsType::iterator I = ConstantFwdRefs.lower_bound(Key);

  if (I != ConstantFwdRefs.end() && I->first == Key) {
    return I->second;
  } else {
    // Create a placeholder for the constant reference and
    // keep track of the fact that we have a forward ref to recycle it
    Constant *C = new ConstantPlaceHolder(getType(TypeSlot));

    // Keep track of the fact that we have a forward ref to recycle it
    ConstantFwdRefs.insert(I, std::make_pair(Key, C));
    return C;
  }
}

//===----------------------------------------------------------------------===//
// IR Construction Methods
//===----------------------------------------------------------------------===//

/// As values are created, they are inserted into the appropriate place
/// with this method. The ValueTable argument must be one of ModuleValues
/// or FunctionValues data members of this class.
unsigned BytecodeReader::insertValue(Value *Val, unsigned type,
                                      ValueTable &ValueTab) {
  if (ValueTab.size() <= type)
    ValueTab.resize(type+1);

  if (!ValueTab[type]) ValueTab[type] = new ValueList();

  ValueTab[type]->push_back(Val);

  bool HasOffset = hasImplicitNull(type) && !isa<OpaqueType>(Val->getType());
  return ValueTab[type]->size()-1 + HasOffset;
}

/// Insert the arguments of a function as new values in the reader.
void BytecodeReader::insertArguments(Function* F) {
  const FunctionType *FT = F->getFunctionType();
  Function::arg_iterator AI = F->arg_begin();
  for (FunctionType::param_iterator It = FT->param_begin();
       It != FT->param_end(); ++It, ++AI)
    insertValue(AI, getTypeSlot(AI->getType()), FunctionValues);
}

//===----------------------------------------------------------------------===//
// Bytecode Parsing Methods
//===----------------------------------------------------------------------===//

/// This method parses a single instruction. The instruction is
/// inserted at the end of the \p BB provided. The arguments of
/// the instruction are provided in the \p Oprnds vector.
void BytecodeReader::ParseInstruction(SmallVector<unsigned, 8> &Oprnds,
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
      error("Zero-argument instruction found; this is invalid.");

    for (unsigned i = 0; i != NumOprnds; ++i)
      Oprnds[i] = read_vbr_uint();
    break;
  }

  const Type *InstTy = getType(iType);

  // Make the necessary adjustments for dealing with backwards compatibility
  // of opcodes.
  Instruction* Result = 0;

  // First, handle the easy binary operators case
  if (Opcode >= Instruction::BinaryOpsBegin &&
      Opcode <  Instruction::BinaryOpsEnd  && Oprnds.size() == 2) {
    Result = BinaryOperator::create(Instruction::BinaryOps(Opcode),
                                    getValue(iType, Oprnds[0]),
                                    getValue(iType, Oprnds[1]));
  } else {
    // Indicate that we don't think this is a call instruction (yet).
    // Process based on the Opcode read
    switch (Opcode) {
    default: // There was an error, this shouldn't happen.
      if (Result == 0)
        error("Illegal instruction read!");
      break;
    case Instruction::VAArg:
      if (Oprnds.size() != 2)
        error("Invalid VAArg instruction!");
      Result = new VAArgInst(getValue(iType, Oprnds[0]),
                             getType(Oprnds[1]));
      break;
    case Instruction::ExtractElement: {
      if (Oprnds.size() != 2)
        error("Invalid extractelement instruction!");
      Value *V1 = getValue(iType, Oprnds[0]);
      Value *V2 = getValue(Int32TySlot, Oprnds[1]);
      
      if (!ExtractElementInst::isValidOperands(V1, V2))
        error("Invalid extractelement instruction!");

      Result = new ExtractElementInst(V1, V2);
      break;
    }
    case Instruction::InsertElement: {
      const VectorType *VectorTy = dyn_cast<VectorType>(InstTy);
      if (!VectorTy || Oprnds.size() != 3)
        error("Invalid insertelement instruction!");
      
      Value *V1 = getValue(iType, Oprnds[0]);
      Value *V2 = getValue(getTypeSlot(VectorTy->getElementType()),Oprnds[1]);
      Value *V3 = getValue(Int32TySlot, Oprnds[2]);
        
      if (!InsertElementInst::isValidOperands(V1, V2, V3))
        error("Invalid insertelement instruction!");
      Result = new InsertElementInst(V1, V2, V3);
      break;
    }
    case Instruction::ShuffleVector: {
      const VectorType *VectorTy = dyn_cast<VectorType>(InstTy);
      if (!VectorTy || Oprnds.size() != 3)
        error("Invalid shufflevector instruction!");
      Value *V1 = getValue(iType, Oprnds[0]);
      Value *V2 = getValue(iType, Oprnds[1]);
      const VectorType *EltTy = 
        VectorType::get(Type::Int32Ty, VectorTy->getNumElements());
      Value *V3 = getValue(getTypeSlot(EltTy), Oprnds[2]);
      if (!ShuffleVectorInst::isValidOperands(V1, V2, V3))
        error("Invalid shufflevector instruction!");
      Result = new ShuffleVectorInst(V1, V2, V3);
      break;
    }
    case Instruction::Trunc:
      if (Oprnds.size() != 2)
        error("Invalid cast instruction!");
      Result = new TruncInst(getValue(iType, Oprnds[0]), 
                             getType(Oprnds[1]));
      break;
    case Instruction::ZExt:
      if (Oprnds.size() != 2)
        error("Invalid cast instruction!");
      Result = new ZExtInst(getValue(iType, Oprnds[0]), 
                            getType(Oprnds[1]));
      break;
    case Instruction::SExt:
      if (Oprnds.size() != 2)
        error("Invalid Cast instruction!");
      Result = new SExtInst(getValue(iType, Oprnds[0]),
                            getType(Oprnds[1]));
      break;
    case Instruction::FPTrunc:
      if (Oprnds.size() != 2)
        error("Invalid cast instruction!");
      Result = new FPTruncInst(getValue(iType, Oprnds[0]), 
                               getType(Oprnds[1]));
      break;
    case Instruction::FPExt:
      if (Oprnds.size() != 2)
        error("Invalid cast instruction!");
      Result = new FPExtInst(getValue(iType, Oprnds[0]), 
                             getType(Oprnds[1]));
      break;
    case Instruction::UIToFP:
      if (Oprnds.size() != 2)
        error("Invalid cast instruction!");
      Result = new UIToFPInst(getValue(iType, Oprnds[0]), 
                              getType(Oprnds[1]));
      break;
    case Instruction::SIToFP:
      if (Oprnds.size() != 2)
        error("Invalid cast instruction!");
      Result = new SIToFPInst(getValue(iType, Oprnds[0]), 
                              getType(Oprnds[1]));
      break;
    case Instruction::FPToUI:
      if (Oprnds.size() != 2)
        error("Invalid cast instruction!");
      Result = new FPToUIInst(getValue(iType, Oprnds[0]), 
                              getType(Oprnds[1]));
      break;
    case Instruction::FPToSI:
      if (Oprnds.size() != 2)
        error("Invalid cast instruction!");
      Result = new FPToSIInst(getValue(iType, Oprnds[0]), 
                              getType(Oprnds[1]));
      break;
    case Instruction::IntToPtr:
      if (Oprnds.size() != 2)
        error("Invalid cast instruction!");
      Result = new IntToPtrInst(getValue(iType, Oprnds[0]), 
                                getType(Oprnds[1]));
      break;
    case Instruction::PtrToInt:
      if (Oprnds.size() != 2)
        error("Invalid cast instruction!");
      Result = new PtrToIntInst(getValue(iType, Oprnds[0]), 
                                getType(Oprnds[1]));
      break;
    case Instruction::BitCast:
      if (Oprnds.size() != 2)
        error("Invalid cast instruction!");
      Result = new BitCastInst(getValue(iType, Oprnds[0]),
                               getType(Oprnds[1]));
      break;
    case Instruction::Select:
      if (Oprnds.size() != 3)
        error("Invalid Select instruction!");
      Result = new SelectInst(getValue(BoolTySlot, Oprnds[0]),
                              getValue(iType, Oprnds[1]),
                              getValue(iType, Oprnds[2]));
      break;
    case Instruction::PHI: {
      if (Oprnds.size() == 0 || (Oprnds.size() & 1))
        error("Invalid phi node encountered!");

      PHINode *PN = new PHINode(InstTy);
      PN->reserveOperandSpace(Oprnds.size());
      for (unsigned i = 0, e = Oprnds.size(); i != e; i += 2)
        PN->addIncoming(
          getValue(iType, Oprnds[i]), getBasicBlock(Oprnds[i+1]));
      Result = PN;
      break;
    }
    case Instruction::ICmp:
    case Instruction::FCmp:
      if (Oprnds.size() != 3)
        error("Cmp instructions requires 3 operands");
      // These instructions encode the comparison predicate as the 3rd operand.
      Result = CmpInst::create(Instruction::OtherOps(Opcode),
          static_cast<unsigned short>(Oprnds[2]),
          getValue(iType, Oprnds[0]), getValue(iType, Oprnds[1]));
      break;
    case Instruction::Ret:
      if (Oprnds.size() == 0)
        Result = new ReturnInst();
      else if (Oprnds.size() == 1)
        Result = new ReturnInst(getValue(iType, Oprnds[0]));
      else
        error("Unrecognized instruction!");
      break;

    case Instruction::Br:
      if (Oprnds.size() == 1)
        Result = new BranchInst(getBasicBlock(Oprnds[0]));
      else if (Oprnds.size() == 3)
        Result = new BranchInst(getBasicBlock(Oprnds[0]),
            getBasicBlock(Oprnds[1]), getValue(BoolTySlot, Oprnds[2]));
      else
        error("Invalid number of operands for a 'br' instruction!");
      break;
    case Instruction::Switch: {
      if (Oprnds.size() & 1)
        error("Switch statement with odd number of arguments!");

      SwitchInst *I = new SwitchInst(getValue(iType, Oprnds[0]),
                                     getBasicBlock(Oprnds[1]),
                                     Oprnds.size()/2-1);
      for (unsigned i = 2, e = Oprnds.size(); i != e; i += 2)
        I->addCase(cast<ConstantInt>(getValue(iType, Oprnds[i])),
                   getBasicBlock(Oprnds[i+1]));
      Result = I;
      break;
    }
    case 58:                   // Call with extra operand for calling conv
    case 59:                   // tail call, Fast CC
    case 60:                   // normal call, Fast CC
    case 61:                   // tail call, C Calling Conv
    case Instruction::Call: {  // Normal Call, C Calling Convention
      if (Oprnds.size() == 0)
        error("Invalid call instruction encountered!");
      Value *F = getValue(iType, Oprnds[0]);

      unsigned CallingConv = CallingConv::C;
      bool isTailCall = false;

      if (Opcode == 61 || Opcode == 59)
        isTailCall = true;
      
      if (Opcode == 58) {
        isTailCall = Oprnds.back() & 1;
        CallingConv = Oprnds.back() >> 1;
        Oprnds.pop_back();
      } else if (Opcode == 59 || Opcode == 60) {
        CallingConv = CallingConv::Fast;
      }
      
      // Check to make sure we have a pointer to function type
      const PointerType *PTy = dyn_cast<PointerType>(F->getType());
      if (PTy == 0) error("Call to non function pointer value!");
      const FunctionType *FTy = dyn_cast<FunctionType>(PTy->getElementType());
      if (FTy == 0) error("Call to non function pointer value!");

      SmallVector<Value *, 8> Params;
      if (!FTy->isVarArg()) {
        FunctionType::param_iterator It = FTy->param_begin();

        for (unsigned i = 1, e = Oprnds.size(); i != e; ++i) {
          if (It == FTy->param_end())
            error("Invalid call instruction!");
          Params.push_back(getValue(getTypeSlot(*It++), Oprnds[i]));
        }
        if (It != FTy->param_end())
          error("Invalid call instruction!");
      } else {
        Oprnds.erase(Oprnds.begin(), Oprnds.begin()+1);

        unsigned FirstVariableOperand;
        if (Oprnds.size() < FTy->getNumParams())
          error("Call instruction missing operands!");

        // Read all of the fixed arguments
        for (unsigned i = 0, e = FTy->getNumParams(); i != e; ++i)
          Params.push_back(
            getValue(getTypeSlot(FTy->getParamType(i)),Oprnds[i]));

        FirstVariableOperand = FTy->getNumParams();

        if ((Oprnds.size()-FirstVariableOperand) & 1)
          error("Invalid call instruction!");   // Must be pairs of type/value

        for (unsigned i = FirstVariableOperand, e = Oprnds.size();
             i != e; i += 2)
          Params.push_back(getValue(Oprnds[i], Oprnds[i+1]));
      }

      Result = new CallInst(F, &Params[0], Params.size());
      if (isTailCall) cast<CallInst>(Result)->setTailCall();
      if (CallingConv) cast<CallInst>(Result)->setCallingConv(CallingConv);
      break;
    }
    case Instruction::Invoke: {  // Invoke C CC
      if (Oprnds.size() < 3)
        error("Invalid invoke instruction!");
      Value *F = getValue(iType, Oprnds[0]);

      // Check to make sure we have a pointer to function type
      const PointerType *PTy = dyn_cast<PointerType>(F->getType());
      if (PTy == 0)
        error("Invoke to non function pointer value!");
      const FunctionType *FTy = dyn_cast<FunctionType>(PTy->getElementType());
      if (FTy == 0)
        error("Invoke to non function pointer value!");

      SmallVector<Value *, 8> Params;
      BasicBlock *Normal, *Except;
      unsigned CallingConv = Oprnds.back();
      Oprnds.pop_back();

      if (!FTy->isVarArg()) {
        Normal = getBasicBlock(Oprnds[1]);
        Except = getBasicBlock(Oprnds[2]);

        FunctionType::param_iterator It = FTy->param_begin();
        for (unsigned i = 3, e = Oprnds.size(); i != e; ++i) {
          if (It == FTy->param_end())
            error("Invalid invoke instruction!");
          Params.push_back(getValue(getTypeSlot(*It++), Oprnds[i]));
        }
        if (It != FTy->param_end())
          error("Invalid invoke instruction!");
      } else {
        Oprnds.erase(Oprnds.begin(), Oprnds.begin()+1);

        Normal = getBasicBlock(Oprnds[0]);
        Except = getBasicBlock(Oprnds[1]);

        unsigned FirstVariableArgument = FTy->getNumParams()+2;
        for (unsigned i = 2; i != FirstVariableArgument; ++i)
          Params.push_back(getValue(getTypeSlot(FTy->getParamType(i-2)),
                                    Oprnds[i]));

        // Must be type/value pairs. If not, error out.
        if (Oprnds.size()-FirstVariableArgument & 1) 
          error("Invalid invoke instruction!");

        for (unsigned i = FirstVariableArgument; i < Oprnds.size(); i += 2)
          Params.push_back(getValue(Oprnds[i], Oprnds[i+1]));
      }

      Result = new InvokeInst(F, Normal, Except, &Params[0], Params.size());
      if (CallingConv) cast<InvokeInst>(Result)->setCallingConv(CallingConv);
      break;
    }
    case Instruction::Malloc: {
      unsigned Align = 0;
      if (Oprnds.size() == 2)
        Align = (1 << Oprnds[1]) >> 1;
      else if (Oprnds.size() > 2)
        error("Invalid malloc instruction!");
      if (!isa<PointerType>(InstTy))
        error("Invalid malloc instruction!");

      Result = new MallocInst(cast<PointerType>(InstTy)->getElementType(),
                              getValue(Int32TySlot, Oprnds[0]), Align);
      break;
    }
    case Instruction::Alloca: {
      unsigned Align = 0;
      if (Oprnds.size() == 2)
        Align = (1 << Oprnds[1]) >> 1;
      else if (Oprnds.size() > 2)
        error("Invalid alloca instruction!");
      if (!isa<PointerType>(InstTy))
        error("Invalid alloca instruction!");

      Result = new AllocaInst(cast<PointerType>(InstTy)->getElementType(),
                              getValue(Int32TySlot, Oprnds[0]), Align);
      break;
    }
    case Instruction::Free:
      if (!isa<PointerType>(InstTy))
        error("Invalid free instruction!");
      Result = new FreeInst(getValue(iType, Oprnds[0]));
      break;
    case Instruction::GetElementPtr: {
      if (Oprnds.size() == 0 || !isa<PointerType>(InstTy))
        error("Invalid getelementptr instruction!");

      SmallVector<Value*, 8> Idx;

      const Type *NextTy = InstTy;
      for (unsigned i = 1, e = Oprnds.size(); i != e; ++i) {
        const CompositeType *TopTy = dyn_cast_or_null<CompositeType>(NextTy);
        if (!TopTy)
          error("Invalid getelementptr instruction!");

        unsigned ValIdx = Oprnds[i];
        unsigned IdxTy = 0;
        // Struct indices are always uints, sequential type indices can be 
        // any of the 32 or 64-bit integer types.  The actual choice of 
        // type is encoded in the low bit of the slot number.
        if (isa<StructType>(TopTy))
          IdxTy = Int32TySlot;
        else {
          switch (ValIdx & 1) {
          default:
          case 0: IdxTy = Int32TySlot; break;
          case 1: IdxTy = Int64TySlot; break;
          }
          ValIdx >>= 1;
        }
        Idx.push_back(getValue(IdxTy, ValIdx));
        NextTy = GetElementPtrInst::getIndexedType(InstTy, &Idx[0], Idx.size(),
                                                   true);
      }

      Result = new GetElementPtrInst(getValue(iType, Oprnds[0]),
                                     &Idx[0], Idx.size());
      break;
    }
    case 62: {   // attributed load
        if (Oprnds.size() != 2 || !isa<PointerType>(InstTy))
          error("Invalid attributed load instruction!");
        signed Log2AlignVal = ((Oprnds[1]>>1)-1);
        Result = new LoadInst(getValue(iType, Oprnds[0]), "", (Oprnds[1] & 1),
                              ((Log2AlignVal < 0) ? 0 : 1<<Log2AlignVal));
        break;
      }
    case Instruction::Load:
      if (Oprnds.size() != 1 || !isa<PointerType>(InstTy))
        error("Invalid load instruction!");
      Result = new LoadInst(getValue(iType, Oprnds[0]), "");
      break;
    case 63: {   // attributed store
        if (!isa<PointerType>(InstTy) || Oprnds.size() != 3)
          error("Invalid attributed store instruction!");

        Value *Ptr = getValue(iType, Oprnds[1]);
        const Type *ValTy = cast<PointerType>(Ptr->getType())->getElementType();
        signed Log2AlignVal = ((Oprnds[2]>>1)-1);
        Result = new StoreInst(getValue(getTypeSlot(ValTy), Oprnds[0]), Ptr,
                               (Oprnds[2] & 1), 
                               ((Log2AlignVal < 0) ? 0 : 1<<Log2AlignVal));
        break;
      }
    case Instruction::Store: {
      if (!isa<PointerType>(InstTy) || Oprnds.size() != 2)
        error("Invalid store instruction!");

      Value *Ptr = getValue(iType, Oprnds[1]);
      const Type *ValTy = cast<PointerType>(Ptr->getType())->getElementType();
      Result = new StoreInst(getValue(getTypeSlot(ValTy), Oprnds[0]), Ptr,
                             Opcode == 63);
      break;
    }
    case Instruction::Unwind:
      if (Oprnds.size() != 0) error("Invalid unwind instruction!");
      Result = new UnwindInst();
      break;
    case Instruction::Unreachable:
      if (Oprnds.size() != 0) error("Invalid unreachable instruction!");
      Result = new UnreachableInst();
      break;
    }  // end switch(Opcode)
  } // end if !Result

  BB->getInstList().push_back(Result);

  unsigned TypeSlot;
  if (Result->getType() == InstTy)
    TypeSlot = iType;
  else
    TypeSlot = getTypeSlot(Result->getType());

  // We have enough info to inform the handler now.
  if (Handler) 
    Handler->handleInstruction(Opcode, InstTy, &Oprnds[0], Oprnds.size(),
                               Result, At-SaveAt);

  insertValue(Result, TypeSlot, FunctionValues);
}

/// Get a particular numbered basic block, which might be a forward reference.
/// This works together with ParseInstructionList to handle these forward 
/// references in a clean manner.  This function is used when constructing 
/// phi, br, switch, and other instructions that reference basic blocks. 
/// Blocks are numbered sequentially as they appear in the function.
BasicBlock *BytecodeReader::getBasicBlock(unsigned ID) {
  // Make sure there is room in the table...
  if (ParsedBasicBlocks.size() <= ID) ParsedBasicBlocks.resize(ID+1);

  // First check to see if this is a backwards reference, i.e. this block
  // has already been created, or if the forward reference has already
  // been created.
  if (ParsedBasicBlocks[ID])
    return ParsedBasicBlocks[ID];

  // Otherwise, the basic block has not yet been created.  Do so and add it to
  // the ParsedBasicBlocks list.
  return ParsedBasicBlocks[ID] = new BasicBlock();
}

/// Parse all of the BasicBlock's & Instruction's in the body of a function.
/// In post 1.0 bytecode files, we no longer emit basic block individually,
/// in order to avoid per-basic-block overhead.
/// @returns the number of basic blocks encountered.
unsigned BytecodeReader::ParseInstructionList(Function* F) {
  unsigned BlockNo = 0;
  SmallVector<unsigned, 8> Args;

  while (moreInBlock()) {
    if (Handler) Handler->handleBasicBlockBegin(BlockNo);
    BasicBlock *BB;
    if (ParsedBasicBlocks.size() == BlockNo)
      ParsedBasicBlocks.push_back(BB = new BasicBlock());
    else if (ParsedBasicBlocks[BlockNo] == 0)
      BB = ParsedBasicBlocks[BlockNo] = new BasicBlock();
    else
      BB = ParsedBasicBlocks[BlockNo];
    ++BlockNo;
    F->getBasicBlockList().push_back(BB);

    // Read instructions into this basic block until we get to a terminator
    while (moreInBlock() && !BB->getTerminator())
      ParseInstruction(Args, BB);

    if (!BB->getTerminator())
      error("Non-terminated basic block found!");

    if (Handler) Handler->handleBasicBlockEnd(BlockNo-1);
  }

  return BlockNo;
}

/// Parse a type symbol table.
void BytecodeReader::ParseTypeSymbolTable(TypeSymbolTable *TST) {
  // Type Symtab block header: [num entries]
  unsigned NumEntries = read_vbr_uint();
  for (unsigned i = 0; i < NumEntries; ++i) {
    // Symtab entry: [type slot #][name]
    unsigned slot = read_vbr_uint();
    std::string Name = read_str();
    const Type* T = getType(slot);
    TST->insert(Name, T);
  }
}

/// Parse a value symbol table. This works for both module level and function
/// level symbol tables.  For function level symbol tables, the CurrentFunction
/// parameter must be non-zero and the ST parameter must correspond to
/// CurrentFunction's symbol table. For Module level symbol tables, the
/// CurrentFunction argument must be zero.
void BytecodeReader::ParseValueSymbolTable(Function *CurrentFunction,
                                           ValueSymbolTable *VST) {
                                      
  if (Handler) Handler->handleValueSymbolTableBegin(CurrentFunction,VST);

  // Allow efficient basic block lookup by number.
  SmallVector<BasicBlock*, 32> BBMap;
  if (CurrentFunction)
    for (Function::iterator I = CurrentFunction->begin(),
           E = CurrentFunction->end(); I != E; ++I)
      BBMap.push_back(I);

  SmallVector<char, 32> NameStr;
  
  while (moreInBlock()) {
    // Symtab block header: [num entries][type id number]
    unsigned NumEntries = read_vbr_uint();
    unsigned Typ = read_vbr_uint();

    for (unsigned i = 0; i != NumEntries; ++i) {
      // Symtab entry: [def slot #][name]
      unsigned slot = read_vbr_uint();
      read_str(NameStr);
      Value *V = 0;
      if (Typ == LabelTySlot) {
        V = (slot < BBMap.size()) ? BBMap[slot] : 0;
      } else {
        V = getValue(Typ, slot, false); // Find mapping.
      }
      if (Handler) Handler->handleSymbolTableValue(Typ, slot,
                                                   &NameStr[0], NameStr.size());
      if (V == 0)
        error("Failed value look-up for name '" + 
              std::string(NameStr.begin(), NameStr.end()) + "', type #" + 
              utostr(Typ) + " slot #" + utostr(slot));
      V->setName(&NameStr[0], NameStr.size());
      
      NameStr.clear();
    }
  }
  checkPastBlockEnd("Symbol Table");
  if (Handler) Handler->handleValueSymbolTableEnd();
}

// Parse a single type. The typeid is read in first. If its a primitive type
// then nothing else needs to be read, we know how to instantiate it. If its
// a derived type, then additional data is read to fill out the type
// definition.
const Type *BytecodeReader::ParseType() {
  unsigned PrimType = read_vbr_uint();
  const Type *Result = 0;
  if ((Result = Type::getPrimitiveType((Type::TypeID)PrimType)))
    return Result;

  switch (PrimType) {
  case Type::IntegerTyID: {
    unsigned NumBits = read_vbr_uint();
    Result = IntegerType::get(NumBits);
    break;
  }
  case Type::FunctionTyID: {
    const Type *RetType = readType();
    unsigned NumParams = read_vbr_uint();

    std::vector<const Type*> Params;
    while (NumParams--) {
      Params.push_back(readType());
    }

    bool isVarArg = Params.size() && Params.back() == Type::VoidTy;
    if (isVarArg) 
      Params.pop_back();

    ParamAttrsList *Attrs = ParseParamAttrsList();

    Result = FunctionType::get(RetType, Params, isVarArg, Attrs);
    break;
  }
  case Type::ArrayTyID: {
    const Type *ElementType = readType();
    unsigned NumElements = read_vbr_uint();
    Result =  ArrayType::get(ElementType, NumElements);
    break;
  }
  case Type::VectorTyID: {
    const Type *ElementType = readType();
    unsigned NumElements = read_vbr_uint();
    Result =  VectorType::get(ElementType, NumElements);
    break;
  }
  case Type::StructTyID: {
    std::vector<const Type*> Elements;
    unsigned Typ = read_vbr_uint();
    while (Typ) {         // List is terminated by void/0 typeid
      Elements.push_back(getType(Typ));
      Typ = read_vbr_uint();
    }

    Result = StructType::get(Elements, false);
    break;
  }
  case Type::PackedStructTyID: {
    std::vector<const Type*> Elements;
    unsigned Typ = read_vbr_uint();
    while (Typ) {         // List is terminated by void/0 typeid
      Elements.push_back(getType(Typ));
      Typ = read_vbr_uint();
    }

    Result = StructType::get(Elements, true);
    break;
  }
  case Type::PointerTyID: {
    Result = PointerType::get(readType());
    break;
  }

  case Type::OpaqueTyID: {
    Result = OpaqueType::get();
    break;
  }

  default:
    error("Don't know how to deserialize primitive type " + utostr(PrimType));
    break;
  }
  if (Handler) Handler->handleType(Result);
  return Result;
}

ParamAttrsList *BytecodeReader::ParseParamAttrsList() {
  unsigned NumAttrs = read_vbr_uint();
  ParamAttrsList *PAL = 0;
  if (NumAttrs) {
    ParamAttrsVector Attrs;
    ParamAttrsWithIndex PAWI;
    while (NumAttrs--) {
      PAWI.index = read_vbr_uint();
      PAWI.attrs = read_vbr_uint();
      Attrs.push_back(PAWI);
    }
    PAL = ParamAttrsList::get(Attrs);
  }
  return PAL;
}


// ParseTypes - We have to use this weird code to handle recursive
// types.  We know that recursive types will only reference the current slab of
// values in the type plane, but they can forward reference types before they
// have been read.  For example, Type #0 might be '{ Ty#1 }' and Type #1 might
// be 'Ty#0*'.  When reading Type #0, type number one doesn't exist.  To fix
// this ugly problem, we pessimistically insert an opaque type for each type we
// are about to read.  This means that forward references will resolve to
// something and when we reread the type later, we can replace the opaque type
// with a new resolved concrete type.
//
void BytecodeReader::ParseTypes(TypeListTy &Tab, unsigned NumEntries){
  assert(Tab.size() == 0 && "should not have read type constants in before!");

  // Insert a bunch of opaque types to be resolved later...
  Tab.reserve(NumEntries);
  for (unsigned i = 0; i != NumEntries; ++i)
    Tab.push_back(OpaqueType::get());

  if (Handler)
    Handler->handleTypeList(NumEntries);

  // If we are about to resolve types, make sure the type cache is clear.
  if (NumEntries)
    ModuleTypeIDCache.clear();
  
  // Loop through reading all of the types.  Forward types will make use of the
  // opaque types just inserted.
  //
  for (unsigned i = 0; i != NumEntries; ++i) {
    const Type* NewTy = ParseType();
    const Type* OldTy = Tab[i].get();
    if (NewTy == 0)
      error("Couldn't parse type!");

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

/// Parse a single constant value
Value *BytecodeReader::ParseConstantPoolValue(unsigned TypeID) {
  // We must check for a ConstantExpr before switching by type because
  // a ConstantExpr can be of any type, and has no explicit value.
  //
  // 0 if not expr; numArgs if is expr
  unsigned isExprNumArgs = read_vbr_uint();

  if (isExprNumArgs) {
    // 'undef' is encoded with 'exprnumargs' == 1.
    if (isExprNumArgs == 1)
      return UndefValue::get(getType(TypeID));

    // Inline asm is encoded with exprnumargs == ~0U.
    if (isExprNumArgs == ~0U) {
      std::string AsmStr = read_str();
      std::string ConstraintStr = read_str();
      unsigned Flags = read_vbr_uint();
      
      const PointerType *PTy = dyn_cast<PointerType>(getType(TypeID));
      const FunctionType *FTy = 
        PTy ? dyn_cast<FunctionType>(PTy->getElementType()) : 0;

      if (!FTy || !InlineAsm::Verify(FTy, ConstraintStr))
        error("Invalid constraints for inline asm");
      if (Flags & ~1U)
        error("Invalid flags for inline asm");
      bool HasSideEffects = Flags & 1;
      return InlineAsm::get(FTy, AsmStr, ConstraintStr, HasSideEffects);
    }
    
    --isExprNumArgs;

    // FIXME: Encoding of constant exprs could be much more compact!
    SmallVector<Constant*, 8> ArgVec;
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
      if (!Instruction::isCast(Opcode))
        error("Only cast instruction has one argument for ConstantExpr");

      Constant *Result = ConstantExpr::getCast(Opcode, ArgVec[0], 
                                               getType(TypeID));
      if (Handler) Handler->handleConstantExpression(Opcode, &ArgVec[0],
                                                     ArgVec.size(), Result);
      return Result;
    } else if (Opcode == Instruction::GetElementPtr) { // GetElementPtr
      Constant *Result = ConstantExpr::getGetElementPtr(ArgVec[0], &ArgVec[1],
                                                        ArgVec.size()-1);
      if (Handler) Handler->handleConstantExpression(Opcode, &ArgVec[0],
                                                     ArgVec.size(), Result);
      return Result;
    } else if (Opcode == Instruction::Select) {
      if (ArgVec.size() != 3)
        error("Select instruction must have three arguments.");
      Constant* Result = ConstantExpr::getSelect(ArgVec[0], ArgVec[1],
                                                 ArgVec[2]);
      if (Handler) Handler->handleConstantExpression(Opcode, &ArgVec[0],
                                                     ArgVec.size(), Result);
      return Result;
    } else if (Opcode == Instruction::ExtractElement) {
      if (ArgVec.size() != 2 ||
          !ExtractElementInst::isValidOperands(ArgVec[0], ArgVec[1]))
        error("Invalid extractelement constand expr arguments");
      Constant* Result = ConstantExpr::getExtractElement(ArgVec[0], ArgVec[1]);
      if (Handler) Handler->handleConstantExpression(Opcode, &ArgVec[0],
                                                     ArgVec.size(), Result);
      return Result;
    } else if (Opcode == Instruction::InsertElement) {
      if (ArgVec.size() != 3 ||
          !InsertElementInst::isValidOperands(ArgVec[0], ArgVec[1], ArgVec[2]))
        error("Invalid insertelement constand expr arguments");
        
      Constant *Result = 
        ConstantExpr::getInsertElement(ArgVec[0], ArgVec[1], ArgVec[2]);
      if (Handler) Handler->handleConstantExpression(Opcode, &ArgVec[0],
                                                     ArgVec.size(), Result);
      return Result;
    } else if (Opcode == Instruction::ShuffleVector) {
      if (ArgVec.size() != 3 ||
          !ShuffleVectorInst::isValidOperands(ArgVec[0], ArgVec[1], ArgVec[2]))
        error("Invalid shufflevector constant expr arguments.");
      Constant *Result = 
        ConstantExpr::getShuffleVector(ArgVec[0], ArgVec[1], ArgVec[2]);
      if (Handler) Handler->handleConstantExpression(Opcode, &ArgVec[0],
                                                     ArgVec.size(), Result);
      return Result;
    } else if (Opcode == Instruction::ICmp) {
      if (ArgVec.size() != 2) 
        error("Invalid ICmp constant expr arguments.");
      unsigned predicate = read_vbr_uint();
      Constant *Result = ConstantExpr::getICmp(predicate, ArgVec[0], ArgVec[1]);
      if (Handler) Handler->handleConstantExpression(Opcode, &ArgVec[0],
                                                     ArgVec.size(), Result);
      return Result;
    } else if (Opcode == Instruction::FCmp) {
      if (ArgVec.size() != 2) 
        error("Invalid FCmp constant expr arguments.");
      unsigned predicate = read_vbr_uint();
      Constant *Result = ConstantExpr::getFCmp(predicate, ArgVec[0], ArgVec[1]);
      if (Handler) Handler->handleConstantExpression(Opcode, &ArgVec[0], 
                                                     ArgVec.size(), Result);
      return Result;
    } else {                            // All other 2-operand expressions
      Constant* Result = ConstantExpr::get(Opcode, ArgVec[0], ArgVec[1]);
      if (Handler) Handler->handleConstantExpression(Opcode, &ArgVec[0], 
                                                     ArgVec.size(), Result);
      return Result;
    }
  }

  // Ok, not an ConstantExpr.  We now know how to read the given type...
  const Type *Ty = getType(TypeID);
  Constant *Result = 0;
  switch (Ty->getTypeID()) {
  case Type::IntegerTyID: {
    const IntegerType *IT = cast<IntegerType>(Ty);
    if (IT->getBitWidth() <= 32) {
      uint32_t Val = read_vbr_uint();
      if (!ConstantInt::isValueValidForType(Ty, uint64_t(Val)))
        error("Integer value read is invalid for type.");
      Result = ConstantInt::get(IT, Val);
      if (Handler) Handler->handleConstantValue(Result);
    } else if (IT->getBitWidth() <= 64) {
      uint64_t Val = read_vbr_uint64();
      if (!ConstantInt::isValueValidForType(Ty, Val))
        error("Invalid constant integer read.");
      Result = ConstantInt::get(IT, Val);
      if (Handler) Handler->handleConstantValue(Result);
    } else {
      uint32_t numWords = read_vbr_uint();
      uint64_t *data = new uint64_t[numWords];
      for (uint32_t i = 0; i < numWords; ++i)
        data[i] = read_vbr_uint64();
      Result = ConstantInt::get(APInt(IT->getBitWidth(), numWords, data));
      if (Handler) Handler->handleConstantValue(Result);
    }
    break;
  }
  case Type::FloatTyID: {
    float Val;
    read_float(Val);
    Result = ConstantFP::get(Ty, Val);
    if (Handler) Handler->handleConstantValue(Result);
    break;
  }

  case Type::DoubleTyID: {
    double Val;
    read_double(Val);
    Result = ConstantFP::get(Ty, Val);
    if (Handler) Handler->handleConstantValue(Result);
    break;
  }

  case Type::ArrayTyID: {
    const ArrayType *AT = cast<ArrayType>(Ty);
    unsigned NumElements = AT->getNumElements();
    unsigned TypeSlot = getTypeSlot(AT->getElementType());
    std::vector<Constant*> Elements;
    Elements.reserve(NumElements);
    while (NumElements--)     // Read all of the elements of the constant.
      Elements.push_back(getConstantValue(TypeSlot,
                                          read_vbr_uint()));
    Result = ConstantArray::get(AT, Elements);
    if (Handler) Handler->handleConstantArray(AT, &Elements[0], Elements.size(),
                                              TypeSlot, Result);
    break;
  }

  case Type::StructTyID: {
    const StructType *ST = cast<StructType>(Ty);

    std::vector<Constant *> Elements;
    Elements.reserve(ST->getNumElements());
    for (unsigned i = 0; i != ST->getNumElements(); ++i)
      Elements.push_back(getConstantValue(ST->getElementType(i),
                                          read_vbr_uint()));

    Result = ConstantStruct::get(ST, Elements);
    if (Handler) Handler->handleConstantStruct(ST, &Elements[0],Elements.size(),
                                               Result);
    break;
  }

  case Type::VectorTyID: {
    const VectorType *PT = cast<VectorType>(Ty);
    unsigned NumElements = PT->getNumElements();
    unsigned TypeSlot = getTypeSlot(PT->getElementType());
    std::vector<Constant*> Elements;
    Elements.reserve(NumElements);
    while (NumElements--)     // Read all of the elements of the constant.
      Elements.push_back(getConstantValue(TypeSlot,
                                          read_vbr_uint()));
    Result = ConstantVector::get(PT, Elements);
    if (Handler) Handler->handleConstantVector(PT, &Elements[0],Elements.size(),
                                               TypeSlot, Result);
    break;
  }

  case Type::PointerTyID: {  // ConstantPointerRef value (backwards compat).
    const PointerType *PT = cast<PointerType>(Ty);
    unsigned Slot = read_vbr_uint();

    // Check to see if we have already read this global variable...
    Value *Val = getValue(TypeID, Slot, false);
    if (Val) {
      GlobalValue *GV = dyn_cast<GlobalValue>(Val);
      if (!GV) error("GlobalValue not in ValueTable!");
      if (Handler) Handler->handleConstantPointer(PT, Slot, GV);
      return GV;
    } else {
      error("Forward references are not allowed here.");
    }
  }

  default:
    error("Don't know how to deserialize constant value of type '" +
                      Ty->getDescription());
    break;
  }
  
  // Check that we didn't read a null constant if they are implicit for this
  // type plane.  Do not do this check for constantexprs, as they may be folded
  // to a null value in a way that isn't predicted when a .bc file is initially
  // produced.
  assert((!isa<Constant>(Result) || !cast<Constant>(Result)->isNullValue()) ||
         !hasImplicitNull(TypeID) && "Cannot read null values from bytecode!");
  return Result;
}

/// Resolve references for constants. This function resolves the forward
/// referenced constants in the ConstantFwdRefs map. It uses the
/// replaceAllUsesWith method of Value class to substitute the placeholder
/// instance with the actual instance.
void BytecodeReader::ResolveReferencesToConstant(Constant *NewV, unsigned Typ,
                                                 unsigned Slot) {
  ConstantRefsType::iterator I =
    ConstantFwdRefs.find(std::make_pair(Typ, Slot));
  if (I == ConstantFwdRefs.end()) return;   // Never forward referenced?

  Value *PH = I->second;   // Get the placeholder...
  PH->replaceAllUsesWith(NewV);
  delete PH;                               // Delete the old placeholder
  ConstantFwdRefs.erase(I);                // Remove the map entry for it
}

/// Parse the constant strings section.
void BytecodeReader::ParseStringConstants(unsigned NumEntries, ValueTable &Tab){
  for (; NumEntries; --NumEntries) {
    unsigned Typ = read_vbr_uint();
    const Type *Ty = getType(Typ);
    if (!isa<ArrayType>(Ty))
      error("String constant data invalid!");

    const ArrayType *ATy = cast<ArrayType>(Ty);
    if (ATy->getElementType() != Type::Int8Ty &&
        ATy->getElementType() != Type::Int8Ty)
      error("String constant data invalid!");

    // Read character data.  The type tells us how long the string is.
    char *Data = reinterpret_cast<char *>(alloca(ATy->getNumElements()));
    read_data(Data, Data+ATy->getNumElements());

    std::vector<Constant*> Elements(ATy->getNumElements());
    const Type* ElemType = ATy->getElementType();
    for (unsigned i = 0, e = ATy->getNumElements(); i != e; ++i)
      Elements[i] = ConstantInt::get(ElemType, (unsigned char)Data[i]);

    // Create the constant, inserting it as needed.
    Constant *C = ConstantArray::get(ATy, Elements);
    unsigned Slot = insertValue(C, Typ, Tab);
    ResolveReferencesToConstant(C, Typ, Slot);
    if (Handler) Handler->handleConstantString(cast<ConstantArray>(C));
  }
}

/// Parse the constant pool.
void BytecodeReader::ParseConstantPool(ValueTable &Tab,
                                       TypeListTy &TypeTab,
                                       bool isFunction) {
  if (Handler) Handler->handleGlobalConstantsBegin();

  /// In LLVM 1.3 Type does not derive from Value so the types
  /// do not occupy a plane. Consequently, we read the types
  /// first in the constant pool.
  if (isFunction) {
    unsigned NumEntries = read_vbr_uint();
    ParseTypes(TypeTab, NumEntries);
  }

  while (moreInBlock()) {
    unsigned NumEntries = read_vbr_uint();
    unsigned Typ = read_vbr_uint();

    if (Typ == Type::VoidTyID) {
      /// Use of Type::VoidTyID is a misnomer. It actually means
      /// that the following plane is constant strings
      assert(&Tab == &ModuleValues && "Cannot read strings in functions!");
      ParseStringConstants(NumEntries, Tab);
    } else {
      for (unsigned i = 0; i < NumEntries; ++i) {
        Value *V = ParseConstantPoolValue(Typ);
        assert(V && "ParseConstantPoolValue returned NULL!");
        unsigned Slot = insertValue(V, Typ, Tab);

        // If we are reading a function constant table, make sure that we adjust
        // the slot number to be the real global constant number.
        //
        if (&Tab != &ModuleValues && Typ < ModuleValues.size() &&
            ModuleValues[Typ])
          Slot += ModuleValues[Typ]->size();
        if (Constant *C = dyn_cast<Constant>(V))
          ResolveReferencesToConstant(C, Typ, Slot);
      }
    }
  }

  // After we have finished parsing the constant pool, we had better not have
  // any dangling references left.
  if (!ConstantFwdRefs.empty()) {
    ConstantRefsType::const_iterator I = ConstantFwdRefs.begin();
    Constant* missingConst = I->second;
    error(utostr(ConstantFwdRefs.size()) +
          " unresolved constant reference exist. First one is '" +
          missingConst->getName() + "' of type '" +
          missingConst->getType()->getDescription() + "'.");
  }

  checkPastBlockEnd("Constant Pool");
  if (Handler) Handler->handleGlobalConstantsEnd();
}

/// Parse the contents of a function. Note that this function can be
/// called lazily by materializeFunction
/// @see materializeFunction
void BytecodeReader::ParseFunctionBody(Function* F) {

  unsigned FuncSize = BlockEnd - At;
  GlobalValue::LinkageTypes Linkage = GlobalValue::ExternalLinkage;
  GlobalValue::VisibilityTypes Visibility = GlobalValue::DefaultVisibility;

  unsigned rWord = read_vbr_uint();
  unsigned LinkageID =  rWord & 65535;
  unsigned VisibilityID = rWord >> 16;
  switch (LinkageID) {
  case 0: Linkage = GlobalValue::ExternalLinkage; break;
  case 1: Linkage = GlobalValue::WeakLinkage; break;
  case 2: Linkage = GlobalValue::AppendingLinkage; break;
  case 3: Linkage = GlobalValue::InternalLinkage; break;
  case 4: Linkage = GlobalValue::LinkOnceLinkage; break;
  case 5: Linkage = GlobalValue::DLLImportLinkage; break;
  case 6: Linkage = GlobalValue::DLLExportLinkage; break;
  case 7: Linkage = GlobalValue::ExternalWeakLinkage; break;
  default:
    error("Invalid linkage type for Function.");
    Linkage = GlobalValue::InternalLinkage;
    break;
  }
  switch (VisibilityID) {
  case 0: Visibility = GlobalValue::DefaultVisibility; break;
  case 1: Visibility = GlobalValue::HiddenVisibility; break;
  default:
   error("Unknown visibility type: " + utostr(VisibilityID));
   Visibility = GlobalValue::DefaultVisibility;
   break;
  }

  F->setLinkage(Linkage);
  F->setVisibility(Visibility);
  if (Handler) Handler->handleFunctionBegin(F,FuncSize);

  // Keep track of how many basic blocks we have read in...
  unsigned BlockNum = 0;
  bool InsertedArguments = false;

  BufPtr MyEnd = BlockEnd;
  while (At < MyEnd) {
    unsigned Type, Size;
    BufPtr OldAt = At;
    read_block(Type, Size);

    switch (Type) {
    case BytecodeFormat::ConstantPoolBlockID:
      if (!InsertedArguments) {
        // Insert arguments into the value table before we parse the first basic
        // block in the function
        insertArguments(F);
        InsertedArguments = true;
      }

      ParseConstantPool(FunctionValues, FunctionTypes, true);
      break;

    case BytecodeFormat::InstructionListBlockID: {
      // Insert arguments into the value table before we parse the instruction
      // list for the function
      if (!InsertedArguments) {
        insertArguments(F);
        InsertedArguments = true;
      }

      if (BlockNum)
        error("Already parsed basic blocks!");
      BlockNum = ParseInstructionList(F);
      break;
    }

    case BytecodeFormat::ValueSymbolTableBlockID:
      ParseValueSymbolTable(F, &F->getValueSymbolTable());
      break;

    case BytecodeFormat::TypeSymbolTableBlockID:
      error("Functions don't have type symbol tables");
      break;

    default:
      At += Size;
      if (OldAt > At)
        error("Wrapped around reading bytecode.");
      break;
    }
    BlockEnd = MyEnd;
  }

  // Make sure there were no references to non-existant basic blocks.
  if (BlockNum != ParsedBasicBlocks.size())
    error("Illegal basic block operand reference");

  ParsedBasicBlocks.clear();

  // Resolve forward references.  Replace any uses of a forward reference value
  // with the real value.
  while (!ForwardReferences.empty()) {
    std::map<std::pair<unsigned,unsigned>, Value*>::iterator
      I = ForwardReferences.begin();
    Value *V = getValue(I->first.first, I->first.second, false);
    Value *PlaceHolder = I->second;
    PlaceHolder->replaceAllUsesWith(V);
    ForwardReferences.erase(I);
    delete PlaceHolder;
  }

  // Clear out function-level types...
  FunctionTypes.clear();
  freeTable(FunctionValues);

  if (Handler) Handler->handleFunctionEnd(F);
}

/// This function parses LLVM functions lazily. It obtains the type of the
/// function and records where the body of the function is in the bytecode
/// buffer. The caller can then use the ParseNextFunction and
/// ParseAllFunctionBodies to get handler events for the functions.
void BytecodeReader::ParseFunctionLazily() {
  if (FunctionSignatureList.empty())
    error("FunctionSignatureList empty!");

  Function *Func = FunctionSignatureList.back();
  FunctionSignatureList.pop_back();

  // Save the information for future reading of the function
  LazyFunctionLoadMap[Func] = LazyFunctionInfo(BlockStart, BlockEnd);

  // This function has a body but it's not loaded so it appears `External'.
  // Mark it as a `Ghost' instead to notify the users that it has a body.
  Func->setLinkage(GlobalValue::GhostLinkage);

  // Pretend we've `parsed' this function
  At = BlockEnd;
}

/// The ParserFunction method lazily parses one function. Use this method to
/// casue the parser to parse a specific function in the module. Note that
/// this will remove the function from what is to be included by
/// ParseAllFunctionBodies.
/// @see ParseAllFunctionBodies
/// @see ParseBytecode
bool BytecodeReader::ParseFunction(Function* Func, std::string* ErrMsg) {

  if (setjmp(context)) {
    // Set caller's error message, if requested
    if (ErrMsg)
      *ErrMsg = ErrorMsg;
    // Indicate an error occurred
    return true;
  }

  // Find {start, end} pointers and slot in the map. If not there, we're done.
  LazyFunctionMap::iterator Fi = LazyFunctionLoadMap.find(Func);

  // Make sure we found it
  if (Fi == LazyFunctionLoadMap.end()) {
    error("Unrecognized function of type " + Func->getType()->getDescription());
    return true;
  }

  BlockStart = At = Fi->second.Buf;
  BlockEnd = Fi->second.EndBuf;
  assert(Fi->first == Func && "Found wrong function?");

  this->ParseFunctionBody(Func);
  return false;
}

/// The ParseAllFunctionBodies method parses through all the previously
/// unparsed functions in the bytecode file. If you want to completely parse
/// a bytecode file, this method should be called after Parsebytecode because
/// Parsebytecode only records the locations in the bytecode file of where
/// the function definitions are located. This function uses that information
/// to materialize the functions.
/// @see ParseBytecode
bool BytecodeReader::ParseAllFunctionBodies(std::string* ErrMsg) {
  if (setjmp(context)) {
    // Set caller's error message, if requested
    if (ErrMsg)
      *ErrMsg = ErrorMsg;
    // Indicate an error occurred
    return true;
  }

  for (LazyFunctionMap::iterator I = LazyFunctionLoadMap.begin(),
       E = LazyFunctionLoadMap.end(); I != E; ++I) {
    Function *Func = I->first;
    if (Func->hasNotBeenReadFromBytecode()) {
      BlockStart = At = I->second.Buf;
      BlockEnd = I->second.EndBuf;
      ParseFunctionBody(Func);
    }
  }
  return false;
}

/// Parse the global type list
void BytecodeReader::ParseGlobalTypes() {
  // Read the number of types
  unsigned NumEntries = read_vbr_uint();
  ParseTypes(ModuleTypes, NumEntries);
}

/// Parse the Global info (types, global vars, constants)
void BytecodeReader::ParseModuleGlobalInfo() {

  if (Handler) Handler->handleModuleGlobalsBegin();

  // SectionID - If a global has an explicit section specified, this map
  // remembers the ID until we can translate it into a string.
  std::map<GlobalValue*, unsigned> SectionID;
  
  // Read global variables...
  unsigned VarType = read_vbr_uint();
  while (VarType != Type::VoidTyID) { // List is terminated by Void
    // VarType Fields: bit0 = isConstant, bit1 = hasInitializer, bit2,3,4 =
    // Linkage, bit5 = isThreadLocal, bit6+ = slot#
    unsigned SlotNo = VarType >> 6;
    unsigned LinkageID = (VarType >> 2) & 7;
    unsigned VisibilityID = 0;
    bool isConstant = VarType & 1;
    bool isThreadLocal = (VarType >> 5) & 1;
    bool hasInitializer = (VarType & 2) != 0;
    unsigned Alignment = 0;
    unsigned GlobalSectionID = 0;
    
    // An extension word is present when linkage = 3 (internal) and hasinit = 0.
    if (LinkageID == 3 && !hasInitializer) {
      unsigned ExtWord = read_vbr_uint();
      // The extension word has this format: bit 0 = has initializer, bit 1-3 =
      // linkage, bit 4-8 = alignment (log2), bit 9 = has section,
      // bits 10-12 = visibility, bits 13+ = future use.
      hasInitializer = ExtWord & 1;
      LinkageID = (ExtWord >> 1) & 7;
      Alignment = (1 << ((ExtWord >> 4) & 31)) >> 1;
      VisibilityID = (ExtWord >> 10) & 7;
      
      if (ExtWord & (1 << 9))  // Has a section ID.
        GlobalSectionID = read_vbr_uint();
    }

    GlobalValue::LinkageTypes Linkage;
    switch (LinkageID) {
    case 0: Linkage = GlobalValue::ExternalLinkage;  break;
    case 1: Linkage = GlobalValue::WeakLinkage;      break;
    case 2: Linkage = GlobalValue::AppendingLinkage; break;
    case 3: Linkage = GlobalValue::InternalLinkage;  break;
    case 4: Linkage = GlobalValue::LinkOnceLinkage;  break;
    case 5: Linkage = GlobalValue::DLLImportLinkage;  break;
    case 6: Linkage = GlobalValue::DLLExportLinkage;  break;
    case 7: Linkage = GlobalValue::ExternalWeakLinkage;  break;
    default:
      error("Unknown linkage type: " + utostr(LinkageID));
      Linkage = GlobalValue::InternalLinkage;
      break;
    }
    GlobalValue::VisibilityTypes Visibility;
    switch (VisibilityID) {
    case 0: Visibility = GlobalValue::DefaultVisibility; break;
    case 1: Visibility = GlobalValue::HiddenVisibility; break;
    default:
      error("Unknown visibility type: " + utostr(VisibilityID));
      Visibility = GlobalValue::DefaultVisibility;
      break;
    }
    
    const Type *Ty = getType(SlotNo);
    if (!Ty)
      error("Global has no type! SlotNo=" + utostr(SlotNo));

    if (!isa<PointerType>(Ty))
      error("Global not a pointer type! Ty= " + Ty->getDescription());

    const Type *ElTy = cast<PointerType>(Ty)->getElementType();

    // Create the global variable...
    GlobalVariable *GV = new GlobalVariable(ElTy, isConstant, Linkage,
                                            0, "", TheModule, isThreadLocal);
    GV->setAlignment(Alignment);
    GV->setVisibility(Visibility);
    insertValue(GV, SlotNo, ModuleValues);

    if (GlobalSectionID != 0)
      SectionID[GV] = GlobalSectionID;

    unsigned initSlot = 0;
    if (hasInitializer) {
      initSlot = read_vbr_uint();
      GlobalInits.push_back(std::make_pair(GV, initSlot));
    }

    // Notify handler about the global value.
    if (Handler)
      Handler->handleGlobalVariable(ElTy, isConstant, Linkage, Visibility,
                                    SlotNo, initSlot, isThreadLocal);

    // Get next item
    VarType = read_vbr_uint();
  }

  // Read the function objects for all of the functions that are coming
  unsigned FnSignature = read_vbr_uint();

  // List is terminated by VoidTy.
  while (((FnSignature & (~0U >> 1)) >> 5) != Type::VoidTyID) {
    const Type *Ty = getType((FnSignature & (~0U >> 1)) >> 5);
    if (!isa<PointerType>(Ty) ||
        !isa<FunctionType>(cast<PointerType>(Ty)->getElementType())) {
      error("Function not a pointer to function type! Ty = " +
            Ty->getDescription());
    }

    // We create functions by passing the underlying FunctionType to create...
    const FunctionType* FTy =
      cast<FunctionType>(cast<PointerType>(Ty)->getElementType());

    // Insert the place holder.
    Function *Func = new Function(FTy, GlobalValue::ExternalLinkage,
                                  "", TheModule);

    insertValue(Func, (FnSignature & (~0U >> 1)) >> 5, ModuleValues);

    // Flags are not used yet.
    unsigned Flags = FnSignature & 31;

    // Save this for later so we know type of lazily instantiated functions.
    // Note that known-external functions do not have FunctionInfo blocks, so we
    // do not add them to the FunctionSignatureList.
    if ((Flags & (1 << 4)) == 0)
      FunctionSignatureList.push_back(Func);

    // Get the calling convention from the low bits.
    unsigned CC = Flags & 15;
    unsigned Alignment = 0;
    if (FnSignature & (1 << 31)) {  // Has extension word?
      unsigned ExtWord = read_vbr_uint();
      Alignment = (1 << (ExtWord & 31)) >> 1;
      CC |= ((ExtWord >> 5) & 15) << 4;
      
      if (ExtWord & (1 << 10))  // Has a section ID.
        SectionID[Func] = read_vbr_uint();

      // Parse external declaration linkage
      switch ((ExtWord >> 11) & 3) {
       case 0: break;
       case 1: Func->setLinkage(Function::DLLImportLinkage); break;
       case 2: Func->setLinkage(Function::ExternalWeakLinkage); break;        
       default: assert(0 && "Unsupported external linkage");        
      }      
    }
    
    Func->setCallingConv(CC-1);
    Func->setAlignment(Alignment);

    if (Handler) Handler->handleFunctionDeclaration(Func);

    // Get the next function signature.
    FnSignature = read_vbr_uint();
  }

  // Now that the function signature list is set up, reverse it so that we can
  // remove elements efficiently from the back of the vector.
  std::reverse(FunctionSignatureList.begin(), FunctionSignatureList.end());

  /// SectionNames - This contains the list of section names encoded in the
  /// moduleinfoblock.  Functions and globals with an explicit section index
  /// into this to get their section name.
  std::vector<std::string> SectionNames;
  
  // Read in the dependent library information.
  unsigned num_dep_libs = read_vbr_uint();
  std::string dep_lib;
  while (num_dep_libs--) {
    dep_lib = read_str();
    TheModule->addLibrary(dep_lib);
    if (Handler)
      Handler->handleDependentLibrary(dep_lib);
  }

  // Read target triple and place into the module.
  std::string triple = read_str();
  TheModule->setTargetTriple(triple);
  if (Handler)
    Handler->handleTargetTriple(triple);
  
  // Read the data layout string and place into the module.
  std::string datalayout = read_str();
  TheModule->setDataLayout(datalayout);
  // FIXME: Implement
  // if (Handler)
    // Handler->handleDataLayout(datalayout);

  if (At != BlockEnd) {
    // If the file has section info in it, read the section names now.
    unsigned NumSections = read_vbr_uint();
    while (NumSections--)
      SectionNames.push_back(read_str());
  }
  
  // If the file has module-level inline asm, read it now.
  if (At != BlockEnd)
    TheModule->setModuleInlineAsm(read_str());

  // If any globals are in specified sections, assign them now.
  for (std::map<GlobalValue*, unsigned>::iterator I = SectionID.begin(), E =
       SectionID.end(); I != E; ++I)
    if (I->second) {
      if (I->second > SectionID.size())
        error("SectionID out of range for global!");
      I->first->setSection(SectionNames[I->second-1]);
    }

  // This is for future proofing... in the future extra fields may be added that
  // we don't understand, so we transparently ignore them.
  //
  At = BlockEnd;

  if (Handler) Handler->handleModuleGlobalsEnd();
}

/// Parse the version information and decode it by setting flags on the
/// Reader that enable backward compatibility of the reader.
void BytecodeReader::ParseVersionInfo() {
  unsigned RevisionNum = read_vbr_uint();

  // We don't provide backwards compatibility in the Reader any more. To
  // upgrade, the user should use llvm-upgrade.
  if (RevisionNum < 7)
    error("Bytecode formats < 7 are no longer supported. Use llvm-upgrade.");

  if (Handler) Handler->handleVersionInfo(RevisionNum);
}

/// Parse a whole module.
void BytecodeReader::ParseModule() {
  unsigned Type, Size;

  FunctionSignatureList.clear(); // Just in case...

  // Read into instance variables...
  ParseVersionInfo();

  bool SeenModuleGlobalInfo = false;
  bool SeenGlobalTypePlane = false;
  BufPtr MyEnd = BlockEnd;
  while (At < MyEnd) {
    BufPtr OldAt = At;
    read_block(Type, Size);

    switch (Type) {

    case BytecodeFormat::GlobalTypePlaneBlockID:
      if (SeenGlobalTypePlane)
        error("Two GlobalTypePlane Blocks Encountered!");

      if (Size > 0)
        ParseGlobalTypes();
      SeenGlobalTypePlane = true;
      break;

    case BytecodeFormat::ModuleGlobalInfoBlockID:
      if (SeenModuleGlobalInfo)
        error("Two ModuleGlobalInfo Blocks Encountered!");
      ParseModuleGlobalInfo();
      SeenModuleGlobalInfo = true;
      break;

    case BytecodeFormat::ConstantPoolBlockID:
      ParseConstantPool(ModuleValues, ModuleTypes,false);
      break;

    case BytecodeFormat::FunctionBlockID:
      ParseFunctionLazily();
      break;

    case BytecodeFormat::ValueSymbolTableBlockID:
      ParseValueSymbolTable(0, &TheModule->getValueSymbolTable());
      break;

    case BytecodeFormat::TypeSymbolTableBlockID:
      ParseTypeSymbolTable(&TheModule->getTypeSymbolTable());
      break;

    default:
      At += Size;
      if (OldAt > At) {
        error("Unexpected Block of Type #" + utostr(Type) + " encountered!");
      }
      break;
    }
    BlockEnd = MyEnd;
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
        error("Global *already* has an initializer?!");
      if (Handler) Handler->handleGlobalInitializer(GV,CV);
      GV->setInitializer(CV);
    } else
      error("Cannot find initializer value.");
  }

  if (!ConstantFwdRefs.empty())
    error("Use of undefined constants in a module");

  /// Make sure we pulled them all out. If we didn't then there's a declaration
  /// but a missing body. That's not allowed.
  if (!FunctionSignatureList.empty())
    error("Function declared, but bytecode stream ended before definition");
}

/// This function completely parses a bytecode buffer given by the \p Buf
/// and \p Length parameters.
bool BytecodeReader::ParseBytecode(volatile BufPtr Buf, unsigned Length,
                                   const std::string &ModuleID,
                                   BCDecompressor_t *Decompressor, 
                                   std::string* ErrMsg) {

  /// We handle errors by
  if (setjmp(context)) {
    // Cleanup after error
    if (Handler) Handler->handleError(ErrorMsg);
    freeState();
    delete TheModule;
    TheModule = 0;
    if (decompressedBlock != 0 ) {
      ::free(decompressedBlock);
      decompressedBlock = 0;
    }
    // Set caller's error message, if requested
    if (ErrMsg)
      *ErrMsg = ErrorMsg;
    // Indicate an error occurred
    return true;
  }

  RevisionNum = 0;
  At = MemStart = BlockStart = Buf;
  MemEnd = BlockEnd = Buf + Length;

  // Create the module
  TheModule = new Module(ModuleID);

  if (Handler) Handler->handleStart(TheModule, Length);

  // Read the four bytes of the signature.
  unsigned Sig = read_uint();

  // If this is a compressed file
  if (Sig == ('l' | ('l' << 8) | ('v' << 16) | ('c' << 24))) {
    if (!Decompressor) {
      error("Compressed bytecode found, but not decompressor available");
    }

    // Invoke the decompression of the bytecode. Note that we have to skip the
    // file's magic number which is not part of the compressed block. Hence,
    // the Buf+4 and Length-4. The result goes into decompressedBlock, a data
    // member for retention until BytecodeReader is destructed.
    unsigned decompressedLength = 
      Decompressor((char*)Buf+4,Length-4,decompressedBlock, 0);

    // We must adjust the buffer pointers used by the bytecode reader to point
    // into the new decompressed block. After decompression, the
    // decompressedBlock will point to a contiguous memory area that has
    // the decompressed data.
    At = MemStart = BlockStart = Buf = (BufPtr) decompressedBlock;
    MemEnd = BlockEnd = Buf + decompressedLength;

  // else if this isn't a regular (uncompressed) bytecode file, then its
  // and error, generate that now.
  } else if (Sig != ('l' | ('l' << 8) | ('v' << 16) | ('m' << 24))) {
    error("Invalid bytecode signature: " + utohexstr(Sig));
  }

  // Tell the handler we're starting a module
  if (Handler) Handler->handleModuleBegin(ModuleID);

  // Get the module block and size and verify. This is handled specially
  // because the module block/size is always written in long format. Other
  // blocks are written in short format so the read_block method is used.
  unsigned Type, Size;
  Type = read_uint();
  Size = read_uint();
  if (Type != BytecodeFormat::ModuleBlockID) {
    error("Expected Module Block! Type:" + utostr(Type) + ", Size:"
          + utostr(Size));
  }

  // It looks like the darwin ranlib program is broken, and adds trailing
  // garbage to the end of some bytecode files.  This hack allows the bc
  // reader to ignore trailing garbage on bytecode files.
  if (At + Size < MemEnd)
    MemEnd = BlockEnd = At+Size;

  if (At + Size != MemEnd)
    error("Invalid Top Level Block Length! Type:" + utostr(Type)
          + ", Size:" + utostr(Size));

  // Parse the module contents
  this->ParseModule();

  // Check for missing functions
  if (hasFunctions())
    error("Function expected, but bytecode stream ended!");

  // Tell the handler we're done with the module
  if (Handler)
    Handler->handleModuleEnd(ModuleID);

  // Tell the handler we're finished the parse
  if (Handler) Handler->handleFinish();

  return false;

}

//===----------------------------------------------------------------------===//
//=== Default Implementations of Handler Methods
//===----------------------------------------------------------------------===//

BytecodeHandler::~BytecodeHandler() {}
