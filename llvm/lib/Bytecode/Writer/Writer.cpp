//===-- Writer.cpp - Library for writing LLVM bytecode files --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This library implements the functionality defined in llvm/Bytecode/Writer.h
//
// Note that this file uses an unusual technique of outputting all the bytecode
// to a vector of unsigned char, then copies the vector to an ostream.  The
// reason for this is that we must do "seeking" in the stream to do back-
// patching, and some very important ostreams that we want to support (like
// pipes) do not support seeking.  :( :( :(
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "bcwriter"
#include "WriterInternals.h"
#include "llvm/Bytecode/WriteBytecodePass.h"
#include "llvm/CallingConv.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/ParameterAttributes.h"
#include "llvm/InlineAsm.h"
#include "llvm/Instructions.h"
#include "llvm/Module.h"
#include "llvm/TypeSymbolTable.h"
#include "llvm/ValueSymbolTable.h"
#include "llvm/Support/GetElementPtrTypeIterator.h"
#include "llvm/Support/Compressor.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/Streams.h"
#include "llvm/System/Program.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Statistic.h"
#include <cstring>
#include <algorithm>
using namespace llvm;

/// This value needs to be incremented every time the bytecode format changes
/// so that the reader can distinguish which format of the bytecode file has
/// been written.
/// @brief The bytecode version number
const unsigned BCVersionNum = 7;

static RegisterPass<WriteBytecodePass> X("emitbytecode", "Bytecode Writer");

STATISTIC(BytesWritten, "Number of bytecode bytes written");

//===----------------------------------------------------------------------===//
//===                           Output Primitives                          ===//
//===----------------------------------------------------------------------===//

// output - If a position is specified, it must be in the valid portion of the
// string... note that this should be inlined always so only the relevant IF
// body should be included.
inline void BytecodeWriter::output(unsigned i, int pos) {
  if (pos == -1) { // Be endian clean, little endian is our friend
    Out.push_back((unsigned char)i);
    Out.push_back((unsigned char)(i >> 8));
    Out.push_back((unsigned char)(i >> 16));
    Out.push_back((unsigned char)(i >> 24));
  } else {
    Out[pos  ] = (unsigned char)i;
    Out[pos+1] = (unsigned char)(i >> 8);
    Out[pos+2] = (unsigned char)(i >> 16);
    Out[pos+3] = (unsigned char)(i >> 24);
  }
}

inline void BytecodeWriter::output(int32_t i) {
  output((uint32_t)i);
}

/// output_vbr - Output an unsigned value, by using the least number of bytes
/// possible.  This is useful because many of our "infinite" values are really
/// very small most of the time; but can be large a few times.
/// Data format used:  If you read a byte with the high bit set, use the low
/// seven bits as data and then read another byte.
inline void BytecodeWriter::output_vbr(uint64_t i) {
  while (1) {
    if (i < 0x80) { // done?
      Out.push_back((unsigned char)i);   // We know the high bit is clear...
      return;
    }

    // Nope, we are bigger than a character, output the next 7 bits and set the
    // high bit to say that there is more coming...
    Out.push_back(0x80 | ((unsigned char)i & 0x7F));
    i >>= 7;  // Shift out 7 bits now...
  }
}

inline void BytecodeWriter::output_vbr(uint32_t i) {
  while (1) {
    if (i < 0x80) { // done?
      Out.push_back((unsigned char)i);   // We know the high bit is clear...
      return;
    }

    // Nope, we are bigger than a character, output the next 7 bits and set the
    // high bit to say that there is more coming...
    Out.push_back(0x80 | ((unsigned char)i & 0x7F));
    i >>= 7;  // Shift out 7 bits now...
  }
}

inline void BytecodeWriter::output_typeid(unsigned i) {
  if (i <= 0x00FFFFFF)
    this->output_vbr(i);
  else {
    this->output_vbr(0x00FFFFFF);
    this->output_vbr(i);
  }
}

inline void BytecodeWriter::output_vbr(int64_t i) {
  if (i < 0)
    output_vbr(((uint64_t)(-i) << 1) | 1); // Set low order sign bit...
  else
    output_vbr((uint64_t)i << 1);          // Low order bit is clear.
}


inline void BytecodeWriter::output_vbr(int i) {
  if (i < 0)
    output_vbr(((unsigned)(-i) << 1) | 1); // Set low order sign bit...
  else
    output_vbr((unsigned)i << 1);          // Low order bit is clear.
}

inline void BytecodeWriter::output_str(const char *Str, unsigned Len) {
  output_vbr(Len);             // Strings may have an arbitrary length.
  Out.insert(Out.end(), Str, Str+Len);
}

inline void BytecodeWriter::output_data(const void *Ptr, const void *End) {
  Out.insert(Out.end(), (const unsigned char*)Ptr, (const unsigned char*)End);
}

inline void BytecodeWriter::output_float(float& FloatVal) {
  /// FIXME: This isn't optimal, it has size problems on some platforms
  /// where FP is not IEEE.
  uint32_t i = FloatToBits(FloatVal);
  Out.push_back( static_cast<unsigned char>( (i      ) & 0xFF));
  Out.push_back( static_cast<unsigned char>( (i >> 8 ) & 0xFF));
  Out.push_back( static_cast<unsigned char>( (i >> 16) & 0xFF));
  Out.push_back( static_cast<unsigned char>( (i >> 24) & 0xFF));
}

inline void BytecodeWriter::output_double(double& DoubleVal) {
  /// FIXME: This isn't optimal, it has size problems on some platforms
  /// where FP is not IEEE.
  uint64_t i = DoubleToBits(DoubleVal);
  Out.push_back( static_cast<unsigned char>( (i      ) & 0xFF));
  Out.push_back( static_cast<unsigned char>( (i >> 8 ) & 0xFF));
  Out.push_back( static_cast<unsigned char>( (i >> 16) & 0xFF));
  Out.push_back( static_cast<unsigned char>( (i >> 24) & 0xFF));
  Out.push_back( static_cast<unsigned char>( (i >> 32) & 0xFF));
  Out.push_back( static_cast<unsigned char>( (i >> 40) & 0xFF));
  Out.push_back( static_cast<unsigned char>( (i >> 48) & 0xFF));
  Out.push_back( static_cast<unsigned char>( (i >> 56) & 0xFF));
}

inline BytecodeBlock::BytecodeBlock(unsigned ID, BytecodeWriter &w,
                                    bool elideIfEmpty, bool hasLongFormat)
  : Id(ID), Writer(w), ElideIfEmpty(elideIfEmpty), HasLongFormat(hasLongFormat){

  if (HasLongFormat) {
    w.output(ID);
    w.output(0U); // For length in long format
  } else {
    w.output(0U); /// Place holder for ID and length for this block
  }
  Loc = w.size();
}

inline BytecodeBlock::~BytecodeBlock() { // Do backpatch when block goes out
                                         // of scope...
  if (Loc == Writer.size() && ElideIfEmpty) {
    // If the block is empty, and we are allowed to, do not emit the block at
    // all!
    Writer.resize(Writer.size()-(HasLongFormat?8:4));
    return;
  }

  if (HasLongFormat)
    Writer.output(unsigned(Writer.size()-Loc), int(Loc-4));
  else
    Writer.output(unsigned(Writer.size()-Loc) << 5 | (Id & 0x1F), int(Loc-4));
}

//===----------------------------------------------------------------------===//
//===                           Constant Output                            ===//
//===----------------------------------------------------------------------===//

void BytecodeWriter::outputParamAttrsList(const ParamAttrsList *Attrs) {
  if (!Attrs) {
    output_vbr(unsigned(0));
    return;
  }
  unsigned numAttrs = Attrs->size();
  output_vbr(numAttrs);
  for (unsigned i = 0; i < numAttrs; ++i) {
    uint16_t index = Attrs->getParamIndex(i);
    uint16_t attrs = Attrs->getParamAttrs(index);
    output_vbr(uint32_t(index));
    output_vbr(uint32_t(attrs));
  }
}

void BytecodeWriter::outputType(const Type *T) {
  const StructType* STy = dyn_cast<StructType>(T);
  if(STy && STy->isPacked())
    output_vbr((unsigned)Type::PackedStructTyID);
  else
    output_vbr((unsigned)T->getTypeID());

  // That's all there is to handling primitive types...
  if (T->isPrimitiveType())
    return;     // We might do this if we alias a prim type: %x = type int

  switch (T->getTypeID()) {   // Handle derived types now.
  case Type::IntegerTyID:
    output_vbr(cast<IntegerType>(T)->getBitWidth());
    break;
  case Type::FunctionTyID: {
    const FunctionType *FT = cast<FunctionType>(T);
    output_typeid(Table.getTypeSlot(FT->getReturnType()));

    // Output the number of arguments to function (+1 if varargs):
    output_vbr((unsigned)FT->getNumParams()+FT->isVarArg());

    // Output all of the arguments...
    FunctionType::param_iterator I = FT->param_begin();
    for (; I != FT->param_end(); ++I)
      output_typeid(Table.getTypeSlot(*I));

    // Terminate list with VoidTy if we are a varargs function...
    if (FT->isVarArg())
      output_typeid((unsigned)Type::VoidTyID);

    // Put out all the parameter attributes
    outputParamAttrsList(FT->getParamAttrs());
    break;
  }

  case Type::ArrayTyID: {
    const ArrayType *AT = cast<ArrayType>(T);
    output_typeid(Table.getTypeSlot(AT->getElementType()));
    output_vbr(AT->getNumElements());
    break;
  }

 case Type::VectorTyID: {
    const VectorType *PT = cast<VectorType>(T);
    output_typeid(Table.getTypeSlot(PT->getElementType()));
    output_vbr(PT->getNumElements());
    break;
  }

  case Type::StructTyID: {
    const StructType *ST = cast<StructType>(T);
    // Output all of the element types...
    for (StructType::element_iterator I = ST->element_begin(),
           E = ST->element_end(); I != E; ++I) {
      output_typeid(Table.getTypeSlot(*I));
    }

    // Terminate list with VoidTy
    output_typeid((unsigned)Type::VoidTyID);
    break;
  }

  case Type::PointerTyID:
    output_typeid(Table.getTypeSlot(cast<PointerType>(T)->getElementType()));
    break;

  case Type::OpaqueTyID:
    // No need to emit anything, just the count of opaque types is enough.
    break;

  default:
    cerr << __FILE__ << ":" << __LINE__ << ": Don't know how to serialize"
         << " Type '" << T->getDescription() << "'\n";
    break;
  }
}

void BytecodeWriter::outputConstant(const Constant *CPV) {
  assert(((CPV->getType()->isPrimitiveType() || CPV->getType()->isInteger()) ||
          !CPV->isNullValue()) && "Shouldn't output null constants!");

  // We must check for a ConstantExpr before switching by type because
  // a ConstantExpr can be of any type, and has no explicit value.
  //
  if (const ConstantExpr *CE = dyn_cast<ConstantExpr>(CPV)) {
    // FIXME: Encoding of constant exprs could be much more compact!
    assert(CE->getNumOperands() > 0 && "ConstantExpr with 0 operands");
    assert(CE->getNumOperands() != 1 || CE->isCast());
    output_vbr(1+CE->getNumOperands());   // flags as an expr
    output_vbr(CE->getOpcode());          // Put out the CE op code

    for (User::const_op_iterator OI = CE->op_begin(); OI != CE->op_end(); ++OI){
      output_vbr(Table.getSlot(*OI));
      output_typeid(Table.getTypeSlot((*OI)->getType()));
    }
    if (CE->isCompare())
      output_vbr((unsigned)CE->getPredicate());
    return;
  } else if (isa<UndefValue>(CPV)) {
    output_vbr(1U);       // 1 -> UndefValue constant.
    return;
  } else {
    output_vbr(0U);       // flag as not a ConstantExpr (i.e. 0 operands)
  }

  switch (CPV->getType()->getTypeID()) {
  case Type::IntegerTyID: { // Integer types...
    const ConstantInt *CI = cast<ConstantInt>(CPV);
    unsigned NumBits = cast<IntegerType>(CPV->getType())->getBitWidth();
    if (NumBits <= 32)
      output_vbr(uint32_t(CI->getZExtValue()));
    else if (NumBits <= 64)
      output_vbr(uint64_t(CI->getZExtValue()));
    else {
      // We have an arbitrary precision integer value to write whose 
      // bit width is > 64. However, in canonical unsigned integer 
      // format it is likely that the high bits are going to be zero.
      // So, we only write the number of active words. 
      uint32_t activeWords = CI->getValue().getActiveWords();
      const uint64_t *rawData = CI->getValue().getRawData();
      output_vbr(activeWords);
      for (uint32_t i = 0; i < activeWords; ++i)
        output_vbr(rawData[i]);
    }
    break;
  }

  case Type::ArrayTyID: {
    const ConstantArray *CPA = cast<ConstantArray>(CPV);
    assert(!CPA->isString() && "Constant strings should be handled specially!");

    for (unsigned i = 0, e = CPA->getNumOperands(); i != e; ++i)
      output_vbr(Table.getSlot(CPA->getOperand(i)));
    break;
  }

  case Type::VectorTyID: {
    const ConstantVector *CP = cast<ConstantVector>(CPV);
    for (unsigned i = 0, e = CP->getNumOperands(); i != e; ++i)
      output_vbr(Table.getSlot(CP->getOperand(i)));
    break;
  }

  case Type::StructTyID: {
    const ConstantStruct *CPS = cast<ConstantStruct>(CPV);

    for (unsigned i = 0, e = CPS->getNumOperands(); i != e; ++i)
      output_vbr(Table.getSlot(CPS->getOperand(i)));
    break;
  }

  case Type::PointerTyID:
    assert(0 && "No non-null, non-constant-expr constants allowed!");
    abort();

  case Type::FloatTyID: {   // Floating point types...
    float Tmp = (float)cast<ConstantFP>(CPV)->getValue();
    output_float(Tmp);
    break;
  }
  case Type::DoubleTyID: {
    double Tmp = cast<ConstantFP>(CPV)->getValue();
    output_double(Tmp);
    break;
  }

  case Type::VoidTyID:
  case Type::LabelTyID:
  default:
    cerr << __FILE__ << ":" << __LINE__ << ": Don't know how to serialize"
         << " type '" << *CPV->getType() << "'\n";
    break;
  }
  return;
}

/// outputInlineAsm - InlineAsm's get emitted to the constant pool, so they can
/// be shared by multiple uses.
void BytecodeWriter::outputInlineAsm(const InlineAsm *IA) {
  // Output a marker, so we know when we have one one parsing the constant pool.
  // Note that this encoding is 5 bytes: not very efficient for a marker.  Since
  // unique inline asms are rare, this should hardly matter.
  output_vbr(~0U);
  
  output(IA->getAsmString());
  output(IA->getConstraintString());
  output_vbr(unsigned(IA->hasSideEffects()));
}

void BytecodeWriter::outputConstantStrings() {
  SlotCalculator::string_iterator I = Table.string_begin();
  SlotCalculator::string_iterator E = Table.string_end();
  if (I == E) return;  // No strings to emit

  // If we have != 0 strings to emit, output them now.  Strings are emitted into
  // the 'void' type plane.
  output_vbr(unsigned(E-I));
  output_typeid(Type::VoidTyID);

  // Emit all of the strings.
  for (I = Table.string_begin(); I != E; ++I) {
    const ConstantArray *Str = *I;
    output_typeid(Table.getTypeSlot(Str->getType()));

    // Now that we emitted the type (which indicates the size of the string),
    // emit all of the characters.
    std::string Val = Str->getAsString();
    output_data(Val.c_str(), Val.c_str()+Val.size());
  }
}

//===----------------------------------------------------------------------===//
//===                           Instruction Output                         ===//
//===----------------------------------------------------------------------===//

// outputInstructionFormat0 - Output those weird instructions that have a large
// number of operands or have large operands themselves.
//
// Format: [opcode] [type] [numargs] [arg0] [arg1] ... [arg<numargs-1>]
//
void BytecodeWriter::outputInstructionFormat0(const Instruction *I,
                                              unsigned Opcode,
                                              const SlotCalculator &Table,
                                              unsigned Type) {
  // Opcode must have top two bits clear...
  output_vbr(Opcode << 2);                  // Instruction Opcode ID
  output_typeid(Type);                      // Result type

  unsigned NumArgs = I->getNumOperands();
  bool HasExtraArg = false;
  if (isa<CastInst>(I)  || isa<InvokeInst>(I) || 
      isa<CmpInst>(I) || isa<VAArgInst>(I) || Opcode == 58 || 
      Opcode == 62 || Opcode == 63)
    HasExtraArg = true;
  if (const AllocationInst *AI = dyn_cast<AllocationInst>(I))
    HasExtraArg = AI->getAlignment() != 0;
  
  output_vbr(NumArgs + HasExtraArg);

  if (!isa<GetElementPtrInst>(&I)) {
    for (unsigned i = 0; i < NumArgs; ++i)
      output_vbr(Table.getSlot(I->getOperand(i)));

    if (isa<CastInst>(I) || isa<VAArgInst>(I)) {
      output_typeid(Table.getTypeSlot(I->getType()));
    } else if (isa<CmpInst>(I)) {
      output_vbr(unsigned(cast<CmpInst>(I)->getPredicate()));
    } else if (isa<InvokeInst>(I)) {  
      output_vbr(cast<InvokeInst>(I)->getCallingConv());
    } else if (Opcode == 58) {  // Call escape sequence
      output_vbr((cast<CallInst>(I)->getCallingConv() << 1) |
                 unsigned(cast<CallInst>(I)->isTailCall()));
    } else if (const AllocationInst *AI = dyn_cast<AllocationInst>(I)) {
      if (AI->getAlignment())
        output_vbr((unsigned)Log2_32(AI->getAlignment())+1);
    } else if (Opcode == 62) { // Attributed load
      output_vbr((unsigned)(((Log2_32(cast<LoadInst>(I)->getAlignment())+1)<<1)
                            + (cast<LoadInst>(I)->isVolatile() ? 1 : 0)));
    } else if (Opcode == 63) { // Attributed store
      output_vbr((unsigned)(((Log2_32(cast<StoreInst>(I)->getAlignment())+1)<<1)
                            + (cast<StoreInst>(I)->isVolatile() ? 1 : 0)));
    }
  } else {
    output_vbr(Table.getSlot(I->getOperand(0)));

    // We need to encode the type of sequential type indices into their slot #
    unsigned Idx = 1;
    for (gep_type_iterator TI = gep_type_begin(I), E = gep_type_end(I);
         Idx != NumArgs; ++TI, ++Idx) {
      unsigned Slot = Table.getSlot(I->getOperand(Idx));

      if (isa<SequentialType>(*TI)) {
        // These should be either 32-bits or 64-bits, however, with bit
        // accurate types we just distinguish between less than or equal to
        // 32-bits or greater than 32-bits.
        unsigned BitWidth = 
          cast<IntegerType>(I->getOperand(Idx)->getType())->getBitWidth();
        assert(BitWidth == 32 || BitWidth == 64 && 
               "Invalid bitwidth for GEP index");
        unsigned IdxId = BitWidth == 32 ? 0 : 1;
        Slot = (Slot << 1) | IdxId;
      }
      output_vbr(Slot);
    }
  }
}


// outputInstrVarArgsCall - Output the absurdly annoying varargs function calls.
// This are more annoying than most because the signature of the call does not
// tell us anything about the types of the arguments in the varargs portion.
// Because of this, we encode (as type 0) all of the argument types explicitly
// before the argument value.  This really sucks, but you shouldn't be using
// varargs functions in your code! *death to printf*!
//
// Format: [opcode] [type] [numargs] [arg0] [arg1] ... [arg<numargs-1>]
//
void BytecodeWriter::outputInstrVarArgsCall(const Instruction *I,
                                            unsigned Opcode,
                                            const SlotCalculator &Table,
                                            unsigned Type) {
  assert(isa<CallInst>(I) || isa<InvokeInst>(I));
  // Opcode must have top two bits clear...
  output_vbr(Opcode << 2);                  // Instruction Opcode ID
  output_typeid(Type);                      // Result type (varargs type)

  const PointerType *PTy = cast<PointerType>(I->getOperand(0)->getType());
  const FunctionType *FTy = cast<FunctionType>(PTy->getElementType());
  unsigned NumParams = FTy->getNumParams();

  unsigned NumFixedOperands;
  if (isa<CallInst>(I)) {
    // Output an operand for the callee and each fixed argument, then two for
    // each variable argument.
    NumFixedOperands = 1+NumParams;
  } else {
    assert(isa<InvokeInst>(I) && "Not call or invoke??");
    // Output an operand for the callee and destinations, then two for each
    // variable argument.
    NumFixedOperands = 3+NumParams;
  }
  output_vbr(2 * I->getNumOperands()-NumFixedOperands + 
      unsigned(Opcode == 58 || isa<InvokeInst>(I)));

  // The type for the function has already been emitted in the type field of the
  // instruction.  Just emit the slot # now.
  for (unsigned i = 0; i != NumFixedOperands; ++i)
    output_vbr(Table.getSlot(I->getOperand(i)));

  for (unsigned i = NumFixedOperands, e = I->getNumOperands(); i != e; ++i) {
    // Output Arg Type ID
    output_typeid(Table.getTypeSlot(I->getOperand(i)->getType()));

    // Output arg ID itself
    output_vbr(Table.getSlot(I->getOperand(i)));
  }
  
  if (isa<InvokeInst>(I)) {
    // Emit the tail call/calling conv for invoke instructions
    output_vbr(cast<InvokeInst>(I)->getCallingConv());
  } else if (Opcode == 58) {
    const CallInst *CI = cast<CallInst>(I);
    output_vbr((CI->getCallingConv() << 1) | unsigned(CI->isTailCall()));
  }
}


// outputInstructionFormat1 - Output one operand instructions, knowing that no
// operand index is >= 2^12.
//
inline void BytecodeWriter::outputInstructionFormat1(const Instruction *I,
                                                     unsigned Opcode,
                                                     unsigned *Slots,
                                                     unsigned Type) {
  // bits   Instruction format:
  // --------------------------
  // 01-00: Opcode type, fixed to 1.
  // 07-02: Opcode
  // 19-08: Resulting type plane
  // 31-20: Operand #1 (if set to (2^12-1), then zero operands)
  //
  output(1 | (Opcode << 2) | (Type << 8) | (Slots[0] << 20));
}


// outputInstructionFormat2 - Output two operand instructions, knowing that no
// operand index is >= 2^8.
//
inline void BytecodeWriter::outputInstructionFormat2(const Instruction *I,
                                                     unsigned Opcode,
                                                     unsigned *Slots,
                                                     unsigned Type) {
  // bits   Instruction format:
  // --------------------------
  // 01-00: Opcode type, fixed to 2.
  // 07-02: Opcode
  // 15-08: Resulting type plane
  // 23-16: Operand #1
  // 31-24: Operand #2
  //
  output(2 | (Opcode << 2) | (Type << 8) | (Slots[0] << 16) | (Slots[1] << 24));
}


// outputInstructionFormat3 - Output three operand instructions, knowing that no
// operand index is >= 2^6.
//
inline void BytecodeWriter::outputInstructionFormat3(const Instruction *I,
                                                     unsigned Opcode,
                                                     unsigned *Slots,
                                                     unsigned Type) {
  // bits   Instruction format:
  // --------------------------
  // 01-00: Opcode type, fixed to 3.
  // 07-02: Opcode
  // 13-08: Resulting type plane
  // 19-14: Operand #1
  // 25-20: Operand #2
  // 31-26: Operand #3
  //
  output(3 | (Opcode << 2) | (Type << 8) |
          (Slots[0] << 14) | (Slots[1] << 20) | (Slots[2] << 26));
}

void BytecodeWriter::outputInstruction(const Instruction &I) {
  assert(I.getOpcode() < 57 && "Opcode too big???");
  unsigned Opcode = I.getOpcode();
  unsigned NumOperands = I.getNumOperands();

  // Encode 'tail call' as 61
  // 63.
  if (const CallInst *CI = dyn_cast<CallInst>(&I)) {
    if (CI->getCallingConv() == CallingConv::C) {
      if (CI->isTailCall())
        Opcode = 61;   // CCC + Tail Call
      else
        ;     // Opcode = Instruction::Call
    } else if (CI->getCallingConv() == CallingConv::Fast) {
      if (CI->isTailCall())
        Opcode = 59;    // FastCC + TailCall
      else
        Opcode = 60;    // FastCC + Not Tail Call
    } else {
      Opcode = 58;      // Call escape sequence.
    }
  }

  // Figure out which type to encode with the instruction.  Typically we want
  // the type of the first parameter, as opposed to the type of the instruction
  // (for example, with setcc, we always know it returns bool, but the type of
  // the first param is actually interesting).  But if we have no arguments
  // we take the type of the instruction itself.
  //
  const Type *Ty;
  switch (I.getOpcode()) {
  case Instruction::Select:
  case Instruction::Malloc:
  case Instruction::Alloca:
    Ty = I.getType();  // These ALWAYS want to encode the return type
    break;
  case Instruction::Store:
    Ty = I.getOperand(1)->getType();  // Encode the pointer type...
    assert(isa<PointerType>(Ty) && "Store to nonpointer type!?!?");
    break;
  default:              // Otherwise use the default behavior...
    Ty = NumOperands ? I.getOperand(0)->getType() : I.getType();
    break;
  }

  unsigned Type = Table.getTypeSlot(Ty);

  // Varargs calls and invokes are encoded entirely different from any other
  // instructions.
  if (const CallInst *CI = dyn_cast<CallInst>(&I)){
    const PointerType *Ty =cast<PointerType>(CI->getCalledValue()->getType());
    if (cast<FunctionType>(Ty->getElementType())->isVarArg()) {
      outputInstrVarArgsCall(CI, Opcode, Table, Type);
      return;
    }
  } else if (const InvokeInst *II = dyn_cast<InvokeInst>(&I)) {
    const PointerType *Ty =cast<PointerType>(II->getCalledValue()->getType());
    if (cast<FunctionType>(Ty->getElementType())->isVarArg()) {
      outputInstrVarArgsCall(II, Opcode, Table, Type);
      return;
    }
  }

  if (NumOperands <= 3) {
    // Make sure that we take the type number into consideration.  We don't want
    // to overflow the field size for the instruction format we select.
    //
    unsigned MaxOpSlot = Type;
    unsigned Slots[3]; Slots[0] = (1 << 12)-1;   // Marker to signify 0 operands

    for (unsigned i = 0; i != NumOperands; ++i) {
      unsigned Slot = Table.getSlot(I.getOperand(i));
      if (Slot > MaxOpSlot) MaxOpSlot = Slot;
      Slots[i] = Slot;
    }

    // Handle the special cases for various instructions...
    if (isa<CastInst>(I) || isa<VAArgInst>(I)) {
      // Cast has to encode the destination type as the second argument in the
      // packet, or else we won't know what type to cast to!
      Slots[1] = Table.getTypeSlot(I.getType());
      if (Slots[1] > MaxOpSlot) MaxOpSlot = Slots[1];
      NumOperands++;
    } else if (const AllocationInst *AI = dyn_cast<AllocationInst>(&I)) {
      assert(NumOperands == 1 && "Bogus allocation!");
      if (AI->getAlignment()) {
        Slots[1] = Log2_32(AI->getAlignment())+1;
        if (Slots[1] > MaxOpSlot) MaxOpSlot = Slots[1];
        NumOperands = 2;
      }
    } else if (isa<ICmpInst>(I) || isa<FCmpInst>(I)) {
      // We need to encode the compare instruction's predicate as the third
      // operand. Its not really a slot, but we don't want to break the 
      // instruction format for these instructions.
      NumOperands++;
      assert(NumOperands == 3 && "CmpInst with wrong number of operands?");
      Slots[2] = unsigned(cast<CmpInst>(&I)->getPredicate());
      if (Slots[2] > MaxOpSlot)
        MaxOpSlot = Slots[2];
    } else if (const GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(&I)) {
      // We need to encode the type of sequential type indices into their slot #
      unsigned Idx = 1;
      for (gep_type_iterator I = gep_type_begin(GEP), E = gep_type_end(GEP);
           I != E; ++I, ++Idx)
        if (isa<SequentialType>(*I)) {
          // These should be either 32-bits or 64-bits, however, with bit
          // accurate types we just distinguish between less than or equal to
          // 32-bits or greater than 32-bits.
          unsigned BitWidth = 
            cast<IntegerType>(GEP->getOperand(Idx)->getType())->getBitWidth();
          assert(BitWidth == 32 || BitWidth == 64 && 
                 "Invalid bitwidth for GEP index");
          unsigned IdxId = BitWidth == 32 ? 0 : 1;
          Slots[Idx] = (Slots[Idx] << 1) | IdxId;
          if (Slots[Idx] > MaxOpSlot) MaxOpSlot = Slots[Idx];
        }
    } else if (Opcode == 58) {
      // If this is the escape sequence for call, emit the tailcall/cc info.
      const CallInst &CI = cast<CallInst>(I);
      ++NumOperands;
      if (NumOperands <= 3) {
        Slots[NumOperands-1] =
          (CI.getCallingConv() << 1)|unsigned(CI.isTailCall());
        if (Slots[NumOperands-1] > MaxOpSlot)
          MaxOpSlot = Slots[NumOperands-1];
      }
    } else if (isa<InvokeInst>(I)) {
      // Invoke escape seq has at least 4 operands to encode.
      ++NumOperands;
    } else if (const LoadInst *LI = dyn_cast<LoadInst>(&I)) {
      // Encode attributed load as opcode 62
      // We need to encode the attributes of the load instruction as the second
      // operand. Its not really a slot, but we don't want to break the 
      // instruction format for these instructions.
      if (LI->getAlignment() || LI->isVolatile()) {
        NumOperands = 2;
        Slots[1] = ((Log2_32(LI->getAlignment())+1)<<1) + 
                    (LI->isVolatile() ? 1 : 0);
        if (Slots[1] > MaxOpSlot) 
          MaxOpSlot = Slots[1];
        Opcode = 62;
      }
    } else if (const StoreInst *SI = dyn_cast<StoreInst>(&I)) {
      // Encode attributed store as opcode 63
      // We need to encode the attributes of the store instruction as the third
      // operand. Its not really a slot, but we don't want to break the 
      // instruction format for these instructions.
      if (SI->getAlignment() || SI->isVolatile()) {
        NumOperands = 3;
        Slots[2] = ((Log2_32(SI->getAlignment())+1)<<1) + 
                    (SI->isVolatile() ? 1 : 0);
        if (Slots[2] > MaxOpSlot) 
          MaxOpSlot = Slots[2];
        Opcode = 63;
      }
    }

    // Decide which instruction encoding to use.  This is determined primarily
    // by the number of operands, and secondarily by whether or not the max
    // operand will fit into the instruction encoding.  More operands == fewer
    // bits per operand.
    //
    switch (NumOperands) {
    case 0:
    case 1:
      if (MaxOpSlot < (1 << 12)-1) { // -1 because we use 4095 to indicate 0 ops
        outputInstructionFormat1(&I, Opcode, Slots, Type);
        return;
      }
      break;

    case 2:
      if (MaxOpSlot < (1 << 8)) {
        outputInstructionFormat2(&I, Opcode, Slots, Type);
        return;
      }
      break;

    case 3:
      if (MaxOpSlot < (1 << 6)) {
        outputInstructionFormat3(&I, Opcode, Slots, Type);
        return;
      }
      break;
    default:
      break;
    }
  }

  // If we weren't handled before here, we either have a large number of
  // operands or a large operand index that we are referring to.
  outputInstructionFormat0(&I, Opcode, Table, Type);
}

//===----------------------------------------------------------------------===//
//===                              Block Output                            ===//
//===----------------------------------------------------------------------===//

BytecodeWriter::BytecodeWriter(std::vector<unsigned char> &o, const Module *M)
  : Out(o), Table(M) {

  // Emit the signature...
  static const unsigned char *Sig = (const unsigned char*)"llvm";
  output_data(Sig, Sig+4);

  // Emit the top level CLASS block.
  BytecodeBlock ModuleBlock(BytecodeFormat::ModuleBlockID, *this, false, true);

  // Output the version identifier
  output_vbr(BCVersionNum);

  // The Global type plane comes first
  {
    BytecodeBlock CPool(BytecodeFormat::GlobalTypePlaneBlockID, *this);
    outputTypes(Type::FirstDerivedTyID);
  }

  // The ModuleInfoBlock follows directly after the type information
  outputModuleInfoBlock(M);

  // Output module level constants, used for global variable initializers
  outputConstants();

  // Do the whole module now! Process each function at a time...
  for (Module::const_iterator I = M->begin(), E = M->end(); I != E; ++I)
    outputFunction(I);

  // Output the symbole table for types
  outputTypeSymbolTable(M->getTypeSymbolTable());

  // Output the symbol table for values
  outputValueSymbolTable(M->getValueSymbolTable());
}

void BytecodeWriter::outputTypes(unsigned TypeNum) {
  // Write the type plane for types first because earlier planes (e.g. for a
  // primitive type like float) may have constants constructed using types
  // coming later (e.g., via getelementptr from a pointer type).  The type
  // plane is needed before types can be fwd or bkwd referenced.
  const std::vector<const Type*>& Types = Table.getTypes();
  assert(!Types.empty() && "No types at all?");
  assert(TypeNum <= Types.size() && "Invalid TypeNo index");

  unsigned NumEntries = Types.size() - TypeNum;

  // Output type header: [num entries]
  output_vbr(NumEntries);

  for (unsigned i = TypeNum; i < TypeNum+NumEntries; ++i)
    outputType(Types[i]);
}

// Helper function for outputConstants().
// Writes out all the constants in the plane Plane starting at entry StartNo.
//
void BytecodeWriter::outputConstantsInPlane(const Value *const *Plane,
                                            unsigned PlaneSize,
                                            unsigned StartNo) {
  unsigned ValNo = StartNo;

  // Scan through and ignore function arguments, global values, and constant
  // strings.
  for (; ValNo < PlaneSize &&
         (isa<Argument>(Plane[ValNo]) || isa<GlobalValue>(Plane[ValNo]) ||
          (isa<ConstantArray>(Plane[ValNo]) &&
           cast<ConstantArray>(Plane[ValNo])->isString())); ValNo++)
    /*empty*/;

  unsigned NC = ValNo;              // Number of constants
  for (; NC < PlaneSize && (isa<Constant>(Plane[NC]) || 
                              isa<InlineAsm>(Plane[NC])); NC++)
    /*empty*/;
  NC -= ValNo;                      // Convert from index into count
  if (NC == 0) return;              // Skip empty type planes...

  // FIXME: Most slabs only have 1 or 2 entries!  We should encode this much
  // more compactly.

  // Put out type header: [num entries][type id number]
  //
  output_vbr(NC);

  // Put out the Type ID Number.
  output_typeid(Table.getTypeSlot(Plane[0]->getType()));

  for (unsigned i = ValNo; i < ValNo+NC; ++i) {
    const Value *V = Plane[i];
    if (const Constant *C = dyn_cast<Constant>(V))
      outputConstant(C);
    else
      outputInlineAsm(cast<InlineAsm>(V));
  }
}

static inline bool hasNullValue(const Type *Ty) {
  return Ty != Type::LabelTy && Ty != Type::VoidTy && !isa<OpaqueType>(Ty);
}

void BytecodeWriter::outputConstants() {
  BytecodeBlock CPool(BytecodeFormat::ConstantPoolBlockID, *this,
                      true  /* Elide block if empty */);

  unsigned NumPlanes = Table.getNumPlanes();

  // Output module-level string constants before any other constants.
  outputConstantStrings();

  for (unsigned pno = 0; pno != NumPlanes; pno++) {
    const SlotCalculator::TypePlane &Plane = Table.getPlane(pno);
    if (!Plane.empty()) {              // Skip empty type planes...
      unsigned ValNo = 0;
      if (hasNullValue(Plane[0]->getType())) {
        // Skip zero initializer
        ValNo = 1;
      }

      // Write out constants in the plane
      outputConstantsInPlane(&Plane[0], Plane.size(), ValNo);
    }
  }
}

static unsigned getEncodedLinkage(const GlobalValue *GV) {
  switch (GV->getLinkage()) {
  default: assert(0 && "Invalid linkage!");
  case GlobalValue::ExternalLinkage:     return 0;
  case GlobalValue::WeakLinkage:         return 1;
  case GlobalValue::AppendingLinkage:    return 2;
  case GlobalValue::InternalLinkage:     return 3;
  case GlobalValue::LinkOnceLinkage:     return 4;
  case GlobalValue::DLLImportLinkage:    return 5;
  case GlobalValue::DLLExportLinkage:    return 6;
  case GlobalValue::ExternalWeakLinkage: return 7;
  }
}

static unsigned getEncodedVisibility(const GlobalValue *GV) {
  switch (GV->getVisibility()) {
  default: assert(0 && "Invalid visibility!");
  case GlobalValue::DefaultVisibility: return 0;
  case GlobalValue::HiddenVisibility:  return 1;
  }
}

void BytecodeWriter::outputModuleInfoBlock(const Module *M) {
  BytecodeBlock ModuleInfoBlock(BytecodeFormat::ModuleGlobalInfoBlockID, *this);

  // Give numbers to sections as we encounter them.
  unsigned SectionIDCounter = 0;
  std::vector<std::string> SectionNames;
  std::map<std::string, unsigned> SectionID;
  
  // Output the types for the global variables in the module...
  for (Module::const_global_iterator I = M->global_begin(),
         End = M->global_end(); I != End; ++I) {
    unsigned Slot = Table.getTypeSlot(I->getType());

    assert((I->hasInitializer() || !I->hasInternalLinkage()) &&
           "Global must have an initializer or have external linkage!");
    
    // Fields: bit0 = isConstant, bit1 = hasInitializer, bit2-4=Linkage,
    // bit5 = isThreadLocal, bit6+ = Slot # for type.
    bool HasExtensionWord = (I->getAlignment() != 0) ||
                            I->hasSection() ||
      (I->getVisibility() != GlobalValue::DefaultVisibility);
    
    // If we need to use the extension byte, set linkage=3(internal) and
    // initializer = 0 (impossible!).
    if (!HasExtensionWord) {
      unsigned oSlot = (Slot << 6)| (((unsigned)I->isThreadLocal()) << 5) |
                       (getEncodedLinkage(I) << 2) | (I->hasInitializer() << 1)
                       | (unsigned)I->isConstant();
      output_vbr(oSlot);
    } else {  
      unsigned oSlot = (Slot << 6) | (((unsigned)I->isThreadLocal()) << 5) |
                       (3 << 2) | (0 << 1) | (unsigned)I->isConstant();
      output_vbr(oSlot);
      
      // The extension word has this format: bit 0 = has initializer, bit 1-3 =
      // linkage, bit 4-8 = alignment (log2), bit 9 = has SectionID,
      // bits 10-12 = visibility, bits 13+ = future use.
      unsigned ExtWord = (unsigned)I->hasInitializer() |
                         (getEncodedLinkage(I) << 1) |
                         ((Log2_32(I->getAlignment())+1) << 4) |
                         ((unsigned)I->hasSection() << 9) |
                         (getEncodedVisibility(I) << 10);
      output_vbr(ExtWord);
      if (I->hasSection()) {
        // Give section names unique ID's.
        unsigned &Entry = SectionID[I->getSection()];
        if (Entry == 0) {
          Entry = ++SectionIDCounter;
          SectionNames.push_back(I->getSection());
        }
        output_vbr(Entry);
      }
    }

    // If we have an initializer, output it now.
    if (I->hasInitializer())
      output_vbr(Table.getSlot((Value*)I->getInitializer()));
  }
  output_typeid(Table.getTypeSlot(Type::VoidTy));

  // Output the types of the functions in this module.
  for (Module::const_iterator I = M->begin(), End = M->end(); I != End; ++I) {
    unsigned Slot = Table.getTypeSlot(I->getType());
    assert(((Slot << 6) >> 6) == Slot && "Slot # too big!");
    unsigned CC = I->getCallingConv()+1;
    unsigned ID = (Slot << 5) | (CC & 15);

    if (I->isDeclaration()) // If external, we don't have an FunctionInfo block.
      ID |= 1 << 4;
    
    if (I->getAlignment() || I->hasSection() || (CC & ~15) != 0 ||
        (I->isDeclaration() && I->hasDLLImportLinkage()) ||
        (I->isDeclaration() && I->hasExternalWeakLinkage())
       )
      ID |= 1 << 31;       // Do we need an extension word?
    
    output_vbr(ID);
    
    if (ID & (1 << 31)) {
      // Extension byte: bits 0-4 = alignment, bits 5-9 = top nibble of calling
      // convention, bit 10 = hasSectionID., bits 11-12 = external linkage type
      unsigned extLinkage = 0;

      if (I->isDeclaration()) {
        if (I->hasDLLImportLinkage()) {
          extLinkage = 1;
        } else if (I->hasExternalWeakLinkage()) {
          extLinkage = 2;
        }
      }

      ID = (Log2_32(I->getAlignment())+1) | ((CC >> 4) << 5) | 
        (I->hasSection() << 10) |
        ((extLinkage & 3) << 11);
      output_vbr(ID);
      
      // Give section names unique ID's.
      if (I->hasSection()) {
        unsigned &Entry = SectionID[I->getSection()];
        if (Entry == 0) {
          Entry = ++SectionIDCounter;
          SectionNames.push_back(I->getSection());
        }
        output_vbr(Entry);
      }
    }
  }
  output_vbr(Table.getTypeSlot(Type::VoidTy) << 5);

  // Emit the list of dependent libraries for the Module.
  Module::lib_iterator LI = M->lib_begin();
  Module::lib_iterator LE = M->lib_end();
  output_vbr(unsigned(LE - LI));   // Emit the number of dependent libraries.
  for (; LI != LE; ++LI)
    output(*LI);

  // Output the target triple from the module
  output(M->getTargetTriple());

  // Output the data layout from the module
  output(M->getDataLayout());
  
  // Emit the table of section names.
  output_vbr((unsigned)SectionNames.size());
  for (unsigned i = 0, e = SectionNames.size(); i != e; ++i)
    output(SectionNames[i]);

  // Output the inline asm string.
  output(M->getModuleInlineAsm());

  // Output aliases
  for (Module::const_alias_iterator I = M->alias_begin(), E = M->alias_end();
       I != E; ++I) {
    unsigned Slot = Table.getTypeSlot(I->getType());
    assert(((Slot << 2) >> 2) == Slot && "Slot # too big!");
    unsigned aliasLinkage = 0;
    switch (I->getLinkage()) {
     case GlobalValue::ExternalLinkage:
      aliasLinkage = 0;
      break;
     case GlobalValue::InternalLinkage:
      aliasLinkage = 1;
      break;
     case GlobalValue::WeakLinkage:
      aliasLinkage = 2;
      break;
     default:
      assert(0 && "Invalid alias linkage");
    }    
    output_vbr((Slot << 2) | aliasLinkage);
    output_vbr(Table.getTypeSlot(I->getAliasee()->getType()));
    output_vbr(Table.getSlot(I->getAliasee()));
  }
  output_typeid(Table.getTypeSlot(Type::VoidTy));
}

void BytecodeWriter::outputInstructions(const Function *F) {
  BytecodeBlock ILBlock(BytecodeFormat::InstructionListBlockID, *this);
  for (Function::const_iterator BB = F->begin(), E = F->end(); BB != E; ++BB)
    for (BasicBlock::const_iterator I = BB->begin(), E = BB->end(); I!=E; ++I)
      outputInstruction(*I);
}

void BytecodeWriter::outputFunction(const Function *F) {
  // If this is an external function, there is nothing else to emit!
  if (F->isDeclaration()) return;

  BytecodeBlock FunctionBlock(BytecodeFormat::FunctionBlockID, *this);
  unsigned rWord = (getEncodedVisibility(F) << 16) | getEncodedLinkage(F);
  output_vbr(rWord);

  // Get slot information about the function...
  Table.incorporateFunction(F);

  // Output all of the instructions in the body of the function
  outputInstructions(F);

  // If needed, output the symbol table for the function...
  outputValueSymbolTable(F->getValueSymbolTable());

  Table.purgeFunction();
}


void BytecodeWriter::outputTypeSymbolTable(const TypeSymbolTable &TST) {
  // Do not output the block for an empty symbol table, it just wastes
  // space!
  if (TST.empty()) return;

  // Create a header for the symbol table
  BytecodeBlock SymTabBlock(BytecodeFormat::TypeSymbolTableBlockID, *this,
                            true/*ElideIfEmpty*/);
  // Write the number of types
  output_vbr(TST.size());

  // Write each of the types
  for (TypeSymbolTable::const_iterator TI = TST.begin(), TE = TST.end(); 
       TI != TE; ++TI) {
    // Symtab entry:[def slot #][name]
    output_typeid(Table.getTypeSlot(TI->second));
    output(TI->first);
  }
}

void BytecodeWriter::outputValueSymbolTable(const ValueSymbolTable &VST) {
  // Do not output the Bytecode block for an empty symbol table, it just wastes
  // space!
  if (VST.empty()) return;

  BytecodeBlock SymTabBlock(BytecodeFormat::ValueSymbolTableBlockID, *this,
                            true/*ElideIfEmpty*/);

  // Organize the symbol table by type
  typedef SmallVector<const ValueName*, 8> PlaneMapVector;
  typedef DenseMap<const Type*, PlaneMapVector> PlaneMap;
  PlaneMap Planes;
  for (ValueSymbolTable::const_iterator SI = VST.begin(), SE = VST.end();
       SI != SE; ++SI) 
    Planes[SI->getValue()->getType()].push_back(&*SI);

  for (PlaneMap::iterator PI = Planes.begin(), PE = Planes.end();
       PI != PE; ++PI) {
    PlaneMapVector::const_iterator I = PI->second.begin(); 
    PlaneMapVector::const_iterator End = PI->second.end(); 

    if (I == End) continue;  // Don't mess with an absent type...

    // Write the number of values in this plane
    output_vbr((unsigned)PI->second.size());

    // Write the slot number of the type for this plane
    output_typeid(Table.getTypeSlot(PI->first));

    // Write each of the values in this plane
    for (; I != End; ++I) {
      // Symtab entry: [def slot #][name]
      output_vbr(Table.getSlot((*I)->getValue()));
      output_str((*I)->getKeyData(), (*I)->getKeyLength());
    }
  }
}

void llvm::WriteBytecodeToFile(const Module *M, OStream &Out,
                               bool compress) {
  assert(M && "You can't write a null module!!");

  // Make sure that std::cout is put into binary mode for systems
  // that care.
  if (Out == cout)
    sys::Program::ChangeStdoutToBinary();

  // Create a vector of unsigned char for the bytecode output. We
  // reserve 256KBytes of space in the vector so that we avoid doing
  // lots of little allocations. 256KBytes is sufficient for a large
  // proportion of the bytecode files we will encounter. Larger files
  // will be automatically doubled in size as needed (std::vector
  // behavior).
  std::vector<unsigned char> Buffer;
  Buffer.reserve(256 * 1024);

  // The BytecodeWriter populates Buffer for us.
  BytecodeWriter BCW(Buffer, M);

  // Keep track of how much we've written
  BytesWritten += Buffer.size();

  // Determine start and end points of the Buffer
  const unsigned char *FirstByte = &Buffer.front();

  // If we're supposed to compress this mess ...
  if (compress) {

    // We signal compression by using an alternate magic number for the
    // file. The compressed bytecode file's magic number is "llvc" instead
    // of "llvm".
    char compressed_magic[4];
    compressed_magic[0] = 'l';
    compressed_magic[1] = 'l';
    compressed_magic[2] = 'v';
    compressed_magic[3] = 'c';

    Out.stream()->write(compressed_magic,4);

    // Compress everything after the magic number (which we altered)
    Compressor::compressToStream(
      (char*)(FirstByte+4),        // Skip the magic number
      Buffer.size()-4,             // Skip the magic number
      *Out.stream()                // Where to write compressed data
    );

  } else {

    // We're not compressing, so just write the entire block.
    Out.stream()->write((char*)FirstByte, Buffer.size());
  }

  // make sure it hits disk now
  Out.stream()->flush();
}
