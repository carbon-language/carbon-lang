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

#include "AnalyzerInternals.h"
#include "ReaderPrimitives.h"
#include "llvm/Module.h"
#include "llvm/Bytecode/Format.h"
#include "Support/StringExtras.h"
#include <iostream>
#include <sstream>

using namespace llvm;

// Enable to trace to figure out what the heck is going on when parsing fails
//#define TRACE_LEVEL 10
//#define DEBUG_OUTPUT

#if TRACE_LEVEL    // ByteCodeReading_TRACEr
#define BCR_TRACE(n, X) \
    if (n < TRACE_LEVEL) std::cerr << std::string(n*2, ' ') << X
#else
#define BCR_TRACE(n, X)
#endif

#define PARSE_ERROR(inserters) \
  { \
    std::ostringstream errormsg; \
    errormsg << inserters; \
    if ( ! handler->handleError( errormsg.str() ) ) \
      throw std::string(errormsg.str()); \
  }


inline void AbstractBytecodeParser::readBlock(const unsigned char *&Buf,
			       const unsigned char *EndBuf, 
			       unsigned &Type, unsigned &Size)
{
  Type = read(Buf, EndBuf);
  Size = read(Buf, EndBuf);
}

const Type *AbstractBytecodeParser::getType(unsigned ID) {
  //cerr << "Looking up Type ID: " << ID << "\n";

  if (ID < Type::FirstDerivedTyID)
    if (const Type *T = Type::getPrimitiveType((Type::PrimitiveID)ID))
      return T;   // Asked for a primitive type...

  // Otherwise, derived types need offset...
  ID -= Type::FirstDerivedTyID;

  if (!CompactionTypeTable.empty()) {
    if (ID >= CompactionTypeTable.size())
      PARSE_ERROR("Type ID out of range for compaction table!");
    return CompactionTypeTable[ID];
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

bool AbstractBytecodeParser::ParseInstruction(BufPtr& Buf, BufPtr EndBuf,
                                      std::vector<unsigned> &Operands) {
  Operands.clear();
  unsigned iType = 0;
  unsigned Opcode = 0;
  unsigned Op = read(Buf, EndBuf);

  // bits   Instruction format:        Common to all formats
  // --------------------------
  // 01-00: Opcode type, fixed to 1.
  // 07-02: Opcode
  Opcode    = (Op >> 2) & 63;
  Operands.resize((Op >> 0) & 03);

  switch (Operands.size()) {
  case 1:
    // bits   Instruction format:
    // --------------------------
    // 19-08: Resulting type plane
    // 31-20: Operand #1 (if set to (2^12-1), then zero operands)
    //
    iType   = (Op >>  8) & 4095;
    Operands[0] = (Op >> 20) & 4095;
    if (Operands[0] == 4095)    // Handle special encoding for 0 operands...
      Operands.resize(0);
    break;
  case 2:
    // bits   Instruction format:
    // --------------------------
    // 15-08: Resulting type plane
    // 23-16: Operand #1
    // 31-24: Operand #2  
    //
    iType   = (Op >>  8) & 255;
    Operands[0] = (Op >> 16) & 255;
    Operands[1] = (Op >> 24) & 255;
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
    Operands[0] = (Op >> 14) & 63;
    Operands[1] = (Op >> 20) & 63;
    Operands[2] = (Op >> 26) & 63;
    break;
  case 0:
    Buf -= 4;  // Hrm, try this again...
    Opcode = read_vbr_uint(Buf, EndBuf);
    Opcode >>= 2;
    iType = read_vbr_uint(Buf, EndBuf);

    unsigned NumOperands = read_vbr_uint(Buf, EndBuf);
    Operands.resize(NumOperands);

    if (NumOperands == 0)
      PARSE_ERROR("Zero-argument instruction found; this is invalid.");

    for (unsigned i = 0; i != NumOperands; ++i)
      Operands[i] = read_vbr_uint(Buf, EndBuf);
    align32(Buf, EndBuf);
    break;
  }

  return handler->handleInstruction(Opcode, getType(iType), Operands);
}

/// ParseBasicBlock - In LLVM 1.0 bytecode files, we used to output one
/// basicblock at a time.  This method reads in one of the basicblock packets.
void AbstractBytecodeParser::ParseBasicBlock(BufPtr &Buf,
                                            BufPtr EndBuf,
                                            unsigned BlockNo) {
  handler->handleBasicBlockBegin( BlockNo );

  std::vector<unsigned> Args;
  bool is_terminating = false;
  while (Buf < EndBuf)
    is_terminating = ParseInstruction(Buf, EndBuf, Args);

  if ( ! is_terminating )
    PARSE_ERROR(
      "Failed to recognize instruction as terminating at end of block");

  handler->handleBasicBlockEnd( BlockNo );
}


/// ParseInstructionList - Parse all of the BasicBlock's & Instruction's in the
/// body of a function.  In post 1.0 bytecode files, we no longer emit basic
/// block individually, in order to avoid per-basic-block overhead.
unsigned AbstractBytecodeParser::ParseInstructionList( BufPtr &Buf, 
                                                       BufPtr EndBuf) {
  unsigned BlockNo = 0;
  std::vector<unsigned> Args;

  while (Buf < EndBuf) {
    handler->handleBasicBlockBegin( BlockNo );

    // Read instructions into this basic block until we get to a terminator
    bool is_terminating = false;
    while (Buf < EndBuf && !is_terminating )
	is_terminating = ParseInstruction(Buf, EndBuf, Args ) ;

    if (!is_terminating)
      PARSE_ERROR( "Non-terminated basic block found!");

    handler->handleBasicBlockEnd( BlockNo );
    ++BlockNo;
  }
  return BlockNo;
}

void AbstractBytecodeParser::ParseSymbolTable(BufPtr &Buf, BufPtr EndBuf) {
  handler->handleSymbolTableBegin();

  while (Buf < EndBuf) {
    // Symtab block header: [num entries][type id number]
    unsigned NumEntries = read_vbr_uint(Buf, EndBuf);
    unsigned Typ = read_vbr_uint(Buf, EndBuf);
    const Type *Ty = getType(Typ);

    handler->handleSymbolTablePlane( Typ, NumEntries, Ty );

    for (unsigned i = 0; i != NumEntries; ++i) {
      // Symtab entry: [def slot #][name]
      unsigned slot = read_vbr_uint(Buf, EndBuf);
      std::string Name = read_str(Buf, EndBuf);

      if (Typ == Type::TypeTyID)
        handler->handleSymbolTableType( i, slot, Name );
      else
	handler->handleSymbolTableValue( i, slot, Name );
    }
  }

  if (Buf > EndBuf) 
    PARSE_ERROR("Tried to read past end of buffer while reading symbol table.");

  handler->handleSymbolTableEnd();
}

void AbstractBytecodeParser::ParseFunctionLazily(BufPtr &Buf, BufPtr EndBuf) {
  if (FunctionSignatureList.empty())
    throw std::string("FunctionSignatureList empty!");

  const Type *FType = FunctionSignatureList.back();
  FunctionSignatureList.pop_back();

  // Save the information for future reading of the function
  LazyFunctionLoadMap[FType] = LazyFunctionInfo(Buf, EndBuf);
  // Pretend we've `parsed' this function
  Buf = EndBuf;
}

void AbstractBytecodeParser::ParseNextFunction(Type* FType) {
  // Find {start, end} pointers and slot in the map. If not there, we're done.
  LazyFunctionMap::iterator Fi = LazyFunctionLoadMap.find(FType);

  // Make sure we found it
  if ( Fi == LazyFunctionLoadMap.end() ) {
    PARSE_ERROR("Unrecognized function of type " << FType->getDescription());
    return;
  }

  BufPtr Buf = Fi->second.Buf;
  BufPtr EndBuf = Fi->second.EndBuf;
  assert(Fi->first == FType);

  LazyFunctionLoadMap.erase(Fi);

  this->ParseFunctionBody( FType, Buf, EndBuf );
}

void AbstractBytecodeParser::ParseFunctionBody(const Type* FType, 
                                               BufPtr &Buf, BufPtr EndBuf ) {

  GlobalValue::LinkageTypes Linkage = GlobalValue::ExternalLinkage;

  unsigned LinkageType = read_vbr_uint(Buf, EndBuf);
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

  handler->handleFunctionBegin(FType,Linkage);

  // Keep track of how many basic blocks we have read in...
  unsigned BlockNum = 0;
  bool InsertedArguments = false;

  while (Buf < EndBuf) {
    unsigned Type, Size;
    BufPtr OldBuf = Buf;
    readBlock(Buf, EndBuf, Type, Size);

    switch (Type) {
    case BytecodeFormat::ConstantPool:
      ParseConstantPool(Buf, Buf+Size, FunctionTypes );
      break;

    case BytecodeFormat::CompactionTable:
      ParseCompactionTable(Buf, Buf+Size);
      break;

    case BytecodeFormat::BasicBlock:
      ParseBasicBlock(Buf, Buf+Size, BlockNum++);
      break;

    case BytecodeFormat::InstructionList:
      if (BlockNum) 
	PARSE_ERROR("InstructionList must come before basic blocks!");
      BlockNum = ParseInstructionList(Buf, Buf+Size);
      break;

    case BytecodeFormat::SymbolTable:
      ParseSymbolTable(Buf, Buf+Size );
      break;

    default:
      Buf += Size;
      if (OldBuf > Buf)
	PARSE_ERROR("Wrapped around reading bytecode");
      break;
    }

    // Malformed bc file if read past end of block.
    align32(Buf, EndBuf);
  }

  handler->handleFunctionEnd(FType);

  // Clear out function-level types...
  FunctionTypes.clear();
  CompactionTypeTable.clear();
}

void AbstractBytecodeParser::ParseAllFunctionBodies() {
  LazyFunctionMap::iterator Fi = LazyFunctionLoadMap.begin();
  LazyFunctionMap::iterator Fe = LazyFunctionLoadMap.end();

  while ( Fi != Fe ) {
    const Type* FType = Fi->first;
    this->ParseFunctionBody(FType, Fi->second.Buf, Fi->second.EndBuf);
  }
}

void AbstractBytecodeParser::ParseCompactionTable(BufPtr &Buf, BufPtr End) {

  handler->handleCompactionTableBegin();

  while (Buf != End) {
    unsigned NumEntries = read_vbr_uint(Buf, End);
    unsigned Ty;

    if ((NumEntries & 3) == 3) {
      NumEntries >>= 2;
      Ty = read_vbr_uint(Buf, End);
    } else {
      Ty = NumEntries >> 2;
      NumEntries &= 3;
    }

    handler->handleCompactionTablePlane( Ty, NumEntries );

    if (Ty == Type::TypeTyID) {
      for (unsigned i = 0; i != NumEntries; ++i) {
	unsigned TypeSlot = read_vbr_uint(Buf,End);
        const Type *Typ = getGlobalTableType(TypeSlot);
	handler->handleCompactionTableType( i, TypeSlot, Typ );
      }
    } else {
      const Type *Typ = getType(Ty);
      // Push the implicit zero
      for (unsigned i = 0; i != NumEntries; ++i) {
	unsigned ValSlot = read_vbr_uint(Buf, End);
	handler->handleCompactionTableValue( i, ValSlot, Typ );
      }
    }
  }
  handler->handleCompactionTableEnd();
}

const Type *AbstractBytecodeParser::ParseTypeConstant(const unsigned char *&Buf,
					      const unsigned char *EndBuf) {
  unsigned PrimType = read_vbr_uint(Buf, EndBuf);

  const Type *Val = 0;
  if ((Val = Type::getPrimitiveType((Type::PrimitiveID)PrimType)))
    return Val;
  
  switch (PrimType) {
  case Type::FunctionTyID: {
    const Type *RetType = getType(read_vbr_uint(Buf, EndBuf));

    unsigned NumParams = read_vbr_uint(Buf, EndBuf);

    std::vector<const Type*> Params;
    while (NumParams--)
      Params.push_back(getType(read_vbr_uint(Buf, EndBuf)));

    bool isVarArg = Params.size() && Params.back() == Type::VoidTy;
    if (isVarArg) Params.pop_back();

    Type* result = FunctionType::get(RetType, Params, isVarArg);
    handler->handleType( result );
    return result;
  }
  case Type::ArrayTyID: {
    unsigned ElTyp = read_vbr_uint(Buf, EndBuf);
    const Type *ElementType = getType(ElTyp);

    unsigned NumElements = read_vbr_uint(Buf, EndBuf);

    BCR_TRACE(5, "Array Type Constant #" << ElTyp << " size=" 
              << NumElements << "\n");
    Type* result =  ArrayType::get(ElementType, NumElements);
    handler->handleType( result );
    return result;
  }
  case Type::StructTyID: {
    std::vector<const Type*> Elements;
    unsigned Typ = read_vbr_uint(Buf, EndBuf);
    while (Typ) {         // List is terminated by void/0 typeid
      Elements.push_back(getType(Typ));
      Typ = read_vbr_uint(Buf, EndBuf);
    }

    Type* result = StructType::get(Elements);
    handler->handleType( result );
    return result;
  }
  case Type::PointerTyID: {
    unsigned ElTyp = read_vbr_uint(Buf, EndBuf);
    BCR_TRACE(5, "Pointer Type Constant #" << ElTyp << "\n");
    Type* result = PointerType::get(getType(ElTyp));
    handler->handleType( result );
    return result;
  }

  case Type::OpaqueTyID: {
    Type* result = OpaqueType::get();
    handler->handleType( result );
    return result;
  }

  default:
    PARSE_ERROR("Don't know how to deserialize primitive type" << PrimType << "\n");
    return Val;
  }
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
void AbstractBytecodeParser::ParseTypeConstants(const unsigned char *&Buf,
                                        const unsigned char *EndBuf,
					TypeListTy &Tab,
					unsigned NumEntries) {
  assert(Tab.size() == 0 && "should not have read type constants in before!");

  // Insert a bunch of opaque types to be resolved later...
  Tab.reserve(NumEntries);
  for (unsigned i = 0; i != NumEntries; ++i)
    Tab.push_back(OpaqueType::get());

  // Loop through reading all of the types.  Forward types will make use of the
  // opaque types just inserted.
  //
  for (unsigned i = 0; i != NumEntries; ++i) {
    const Type *NewTy = ParseTypeConstant(Buf, EndBuf), *OldTy = Tab[i].get();
    if (NewTy == 0) throw std::string("Couldn't parse type!");
    BCR_TRACE(4, "#" << i << ": Read Type Constant: '" << NewTy <<
              "' Replacing: " << OldTy << "\n");

    // Don't insertValue the new type... instead we want to replace the opaque
    // type with the new concrete value...
    //

    // Refine the abstract type to the new type.  This causes all uses of the
    // abstract type to use NewTy.  This also will cause the opaque type to be
    // deleted...
    //
    cast<DerivedType>(const_cast<Type*>(OldTy))->refineAbstractTypeTo(NewTy);

    // This should have replace the old opaque type with the new type in the
    // value table... or with a preexisting type that was already in the system
    assert(Tab[i] != OldTy && "refineAbstractType didn't work!");
  }

  BCR_TRACE(5, "Resulting types:\n");
  for (unsigned i = 0; i < NumEntries; ++i) {
    BCR_TRACE(5, (void*)Tab[i].get() << " - " << Tab[i].get() << "\n");
  }
}


void AbstractBytecodeParser::ParseConstantValue(const unsigned char *&Buf,
                                             const unsigned char *EndBuf,
                                             unsigned TypeID) {

  // We must check for a ConstantExpr before switching by type because
  // a ConstantExpr can be of any type, and has no explicit value.
  // 
  // 0 if not expr; numArgs if is expr
  unsigned isExprNumArgs = read_vbr_uint(Buf, EndBuf);
  
  if (isExprNumArgs) {
    unsigned Opcode = read_vbr_uint(Buf, EndBuf);
    const Type* Typ = getType(TypeID);
    
    // FIXME: Encoding of constant exprs could be much more compact!
    std::vector<std::pair<const Type*,unsigned> > ArgVec;
    ArgVec.reserve(isExprNumArgs);

    // Read the slot number and types of each of the arguments
    for (unsigned i = 0; i != isExprNumArgs; ++i) {
      unsigned ArgValSlot = read_vbr_uint(Buf, EndBuf);
      unsigned ArgTypeSlot = read_vbr_uint(Buf, EndBuf);
      BCR_TRACE(4, "CE Arg " << i << ": Type: '" << *getType(ArgTypeSlot)
                << "'  slot: " << ArgValSlot << "\n");
      
      // Get the arg value from its slot if it exists, otherwise a placeholder
      ArgVec.push_back(std::make_pair(getType(ArgTypeSlot), ArgValSlot));
    }

    handler->handleConstantExpression( Opcode, Typ, ArgVec );
    return;
  }
  
  // Ok, not an ConstantExpr.  We now know how to read the given type...
  const Type *Ty = getType(TypeID);
  switch (Ty->getPrimitiveID()) {
  case Type::BoolTyID: {
    unsigned Val = read_vbr_uint(Buf, EndBuf);
    if (Val != 0 && Val != 1) 
      PARSE_ERROR("Invalid boolean value read.");

    handler->handleConstantValue( ConstantBool::get(Val == 1));
    break;
  }

  case Type::UByteTyID:   // Unsigned integer types...
  case Type::UShortTyID:
  case Type::UIntTyID: {
    unsigned Val = read_vbr_uint(Buf, EndBuf);
    if (!ConstantUInt::isValueValidForType(Ty, Val)) 
      throw std::string("Invalid unsigned byte/short/int read.");
    handler->handleConstantValue( ConstantUInt::get(Ty, Val) );
    break;
  }

  case Type::ULongTyID: {
    handler->handleConstantValue( ConstantUInt::get(Ty, read_vbr_uint64(Buf, EndBuf)) );
    break;
  }

  case Type::SByteTyID:   // Signed integer types...
  case Type::ShortTyID:
  case Type::IntTyID: {
  case Type::LongTyID:
    int64_t Val = read_vbr_int64(Buf, EndBuf);
    if (!ConstantSInt::isValueValidForType(Ty, Val)) 
      throw std::string("Invalid signed byte/short/int/long read.");
    handler->handleConstantValue(  ConstantSInt::get(Ty, Val) );
    break;
  }

  case Type::FloatTyID: {
    float F;
    input_data(Buf, EndBuf, &F, &F+1);
    handler->handleConstantValue( ConstantFP::get(Ty, F) );
    break;
  }

  case Type::DoubleTyID: {
    double Val;
    input_data(Buf, EndBuf, &Val, &Val+1);
    handler->handleConstantValue( ConstantFP::get(Ty, Val) );
    break;
  }

  case Type::TypeTyID:
    PARSE_ERROR("Type constants shouldn't live in constant table!");
    break;

  case Type::ArrayTyID: {
    const ArrayType *AT = cast<ArrayType>(Ty);
    unsigned NumElements = AT->getNumElements();
    std::vector<unsigned> Elements;
    Elements.reserve(NumElements);
    while (NumElements--)     // Read all of the elements of the constant.
      Elements.push_back(read_vbr_uint(Buf, EndBuf));

    handler->handleConstantArray( AT, Elements );
    break;
  }

  case Type::StructTyID: {
    const StructType *ST = cast<StructType>(Ty);
    std::vector<unsigned> Elements;
    Elements.reserve(ST->getNumElements());
    for (unsigned i = 0; i != ST->getNumElements(); ++i)
      Elements.push_back(read_vbr_uint(Buf, EndBuf));

    handler->handleConstantStruct( ST, Elements );
  }    

  case Type::PointerTyID: {  // ConstantPointerRef value...
    const PointerType *PT = cast<PointerType>(Ty);
    unsigned Slot = read_vbr_uint(Buf, EndBuf);
    handler->handleConstantPointer( PT, Slot );
  }

  default:
    PARSE_ERROR("Don't know how to deserialize constant value of type '"+
                      Ty->getDescription());
  }
}

void AbstractBytecodeParser::ParseGlobalTypes(const unsigned char *&Buf,
                                      const unsigned char *EndBuf) {
  ParseConstantPool(Buf, EndBuf, ModuleTypes);
}

void AbstractBytecodeParser::ParseStringConstants(const unsigned char *&Buf,
                                          const unsigned char *EndBuf,
                                          unsigned NumEntries ){
  for (; NumEntries; --NumEntries) {
    unsigned Typ = read_vbr_uint(Buf, EndBuf);
    const Type *Ty = getType(Typ);
    if (!isa<ArrayType>(Ty))
      throw std::string("String constant data invalid!");
    
    const ArrayType *ATy = cast<ArrayType>(Ty);
    if (ATy->getElementType() != Type::SByteTy &&
        ATy->getElementType() != Type::UByteTy)
      throw std::string("String constant data invalid!");
    
    // Read character data.  The type tells us how long the string is.
    char Data[ATy->getNumElements()];
    input_data(Buf, EndBuf, Data, Data+ATy->getNumElements());

    std::vector<Constant*> Elements(ATy->getNumElements());
    if (ATy->getElementType() == Type::SByteTy)
      for (unsigned i = 0, e = ATy->getNumElements(); i != e; ++i)
        Elements[i] = ConstantSInt::get(Type::SByteTy, (signed char)Data[i]);
    else
      for (unsigned i = 0, e = ATy->getNumElements(); i != e; ++i)
        Elements[i] = ConstantUInt::get(Type::UByteTy, (unsigned char)Data[i]);

    // Create the constant, inserting it as needed.
    ConstantArray *C = cast<ConstantArray>( ConstantArray::get(ATy, Elements) );
    handler->handleConstantString( C );
  }
}


void AbstractBytecodeParser::ParseConstantPool(const unsigned char *&Buf,
                                       const unsigned char *EndBuf,
                                       TypeListTy &TypeTab) {
  while (Buf < EndBuf) {
    unsigned NumEntries = read_vbr_uint(Buf, EndBuf);
    unsigned Typ = read_vbr_uint(Buf, EndBuf);
    if (Typ == Type::TypeTyID) {
      ParseTypeConstants(Buf, EndBuf, TypeTab, NumEntries);
    } else if (Typ == Type::VoidTyID) {
      ParseStringConstants(Buf, EndBuf, NumEntries);
    } else {
      BCR_TRACE(3, "Type: '" << *getType(Typ) << "'  NumEntries: "
                << NumEntries << "\n");

      for (unsigned i = 0; i < NumEntries; ++i) {
        ParseConstantValue(Buf, EndBuf, Typ);
      }
    }
  }
  
  if (Buf > EndBuf) PARSE_ERROR("Read past end of buffer.");
}

void AbstractBytecodeParser::ParseModuleGlobalInfo(BufPtr &Buf, BufPtr End) {

  handler->handleModuleGlobalsBegin();

  // Read global variables...
  unsigned VarType = read_vbr_uint(Buf, End);
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
    if (hasInitializer) {
      unsigned initSlot = read_vbr_uint(Buf,End);
      handler->handleInitializedGV( ElTy, isConstant, Linkage, initSlot );
    } else 
      handler->handleGlobalVariable( ElTy, isConstant, Linkage );

    // Get next item
    VarType = read_vbr_uint(Buf, End);
  }

  // Read the function objects for all of the functions that are coming
  unsigned FnSignature = read_vbr_uint(Buf, End);
  while (FnSignature != Type::VoidTyID) { // List is terminated by Void
    const Type *Ty = getType(FnSignature);
    if (!isa<PointerType>(Ty) ||
        !isa<FunctionType>(cast<PointerType>(Ty)->getElementType())) {
      PARSE_ERROR( "Function not a pointer to function type! Ty = " +
                        Ty->getDescription());
      // FIXME: what should Ty be if handler continues?
    }

    // We create functions by passing the underlying FunctionType to create...
    Ty = cast<PointerType>(Ty)->getElementType();

    // Save this for later so we know type of lazily instantiated functions
    FunctionSignatureList.push_back(Ty);

    handler->handleFunctionDeclaration(Ty);

    // Get Next function signature
    FnSignature = read_vbr_uint(Buf, End);
  }

  if (hasInconsistentModuleGlobalInfo)
    align32(Buf, End);

  // This is for future proofing... in the future extra fields may be added that
  // we don't understand, so we transparently ignore them.
  //
  Buf = End;

  handler->handleModuleGlobalsEnd();
}

void AbstractBytecodeParser::ParseVersionInfo(BufPtr &Buf, BufPtr EndBuf) {
  unsigned Version = read_vbr_uint(Buf, EndBuf);

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

  handler->handleVersionInfo(RevisionNum, Endianness, PointerSize );
}

void AbstractBytecodeParser::ParseModule(BufPtr &Buf, BufPtr EndBuf ) {
  unsigned Type, Size;
  readBlock(Buf, EndBuf, Type, Size);
  if (Type != BytecodeFormat::Module || Buf+Size != EndBuf)
    // Hrm, not a class?
    PARSE_ERROR("Expected Module block! B: " << unsigned(intptr_t(Buf)) <<
        ", S: " << Size << " E: " << unsigned(intptr_t(EndBuf))); 

  // Read into instance variables...
  ParseVersionInfo(Buf, EndBuf);
  align32(Buf, EndBuf);

  bool SeenModuleGlobalInfo = false;
  bool SeenGlobalTypePlane = false;
  while (Buf < EndBuf) {
    BufPtr OldBuf = Buf;
    readBlock(Buf, EndBuf, Type, Size);

    switch (Type) {

    case BytecodeFormat::GlobalTypePlane:
      if ( SeenGlobalTypePlane )
	PARSE_ERROR("Two GlobalTypePlane Blocks Encountered!");

      ParseGlobalTypes(Buf, Buf+Size);
      SeenGlobalTypePlane = true;
      break;

    case BytecodeFormat::ModuleGlobalInfo: 
      if ( SeenModuleGlobalInfo )
	PARSE_ERROR("Two ModuleGlobalInfo Blocks Encountered!");
      ParseModuleGlobalInfo(Buf, Buf+Size);
      SeenModuleGlobalInfo = true;
      break;

    case BytecodeFormat::ConstantPool:
      ParseConstantPool(Buf, Buf+Size, ModuleTypes);
      break;

    case BytecodeFormat::Function:
      ParseFunctionLazily(Buf, Buf+Size);
      break;

    case BytecodeFormat::SymbolTable:
      ParseSymbolTable(Buf, Buf+Size );
      break;

    default:
      Buf += Size;
      if (OldBuf > Buf) 
      {
	PARSE_ERROR("Unexpected Block of Type" << Type << "encountered!" );
      }
      break;
    }
    align32(Buf, EndBuf);
  }
}

void AbstractBytecodeParser::ParseBytecode(
       BufPtr Buf, unsigned Length,
       const std::string &ModuleID) {

  handler->handleStart();
  unsigned char *EndBuf = (unsigned char*)(Buf + Length);

  // Read and check signature...
  unsigned Sig = read(Buf, EndBuf);
  if (Sig != ('l' | ('l' << 8) | ('v' << 16) | ('m' << 24))) {
    PARSE_ERROR("Invalid bytecode signature: " << Sig);
  }

  handler->handleModuleBegin(ModuleID);

  this->ParseModule(Buf, EndBuf);

  handler->handleModuleEnd(ModuleID);

  handler->handleFinish();
}

// vim: sw=2
