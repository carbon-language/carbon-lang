//===- Parser.cpp - Code to parse bytecode files --------------------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This library implements the functionality defined in llvm/Bytecode/Parser.h
//
// Note that this library should be as fast as possible, reentrant, and 
// threadsafe!!
//
// TODO: Allow passing in an option to ignore the symbol table
//
//===----------------------------------------------------------------------===//

#include "AnalyzerInternals.h"
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

#define PARSE_ERROR(inserters) { \
    std::ostringstream errormsg; \
    errormsg << inserters; \
    if ( ! handler->handleError( errormsg.str() ) ) \
      throw std::string(errormsg.str()); \
  }

inline bool AbstractBytecodeParser::moreInBlock() {
  return At < BlockEnd;
}

inline void AbstractBytecodeParser::checkPastBlockEnd(const char * block_name) {
  if ( At > BlockEnd )
    PARSE_ERROR("Attempt to read past the end of " << block_name << " block.");
}

inline void AbstractBytecodeParser::align32() {
  BufPtr Save = At;
  At = (const unsigned char *)((unsigned long)(At+3) & (~3UL));
  if ( reportAlignment && At > Save ) handler->handleAlignment( At - Save );
  if (At > BlockEnd) 
    throw std::string("Ran out of data while aligning!");
}

inline unsigned AbstractBytecodeParser::read_uint() {
  if (At+4 > BlockEnd) 
    throw std::string("Ran out of data reading uint!");
  At += 4;
  return At[-4] | (At[-3] << 8) | (At[-2] << 16) | (At[-1] << 24);
}

inline unsigned AbstractBytecodeParser::read_vbr_uint() {
  unsigned Shift = 0;
  unsigned Result = 0;
  BufPtr Save = At;
  
  do {
    if (At == BlockEnd) 
      throw std::string("Ran out of data reading vbr_uint!");
    Result |= (unsigned)((*At++) & 0x7F) << Shift;
    Shift += 7;
  } while (At[-1] & 0x80);
  if (reportVBR)
    handler->handleVBR32(At-Save);
  return Result;
}

inline uint64_t AbstractBytecodeParser::read_vbr_uint64() {
  unsigned Shift = 0;
  uint64_t Result = 0;
  BufPtr Save = At;
  
  do {
    if (At == BlockEnd) 
      throw std::string("Ran out of data reading vbr_uint64!");
    Result |= (uint64_t)((*At++) & 0x7F) << Shift;
    Shift += 7;
  } while (At[-1] & 0x80);
  if (reportVBR)
    handler->handleVBR64(At-Save);
  return Result;
}

inline int64_t AbstractBytecodeParser::read_vbr_int64() {
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

inline std::string AbstractBytecodeParser::read_str() {
  unsigned Size = read_vbr_uint();
  const unsigned char *OldAt = At;
  At += Size;
  if (At > BlockEnd)             // Size invalid?
    throw std::string("Ran out of data reading a string!");
  return std::string((char*)OldAt, Size);
}

inline void AbstractBytecodeParser::read_data(void *Ptr, void *End) {
  unsigned char *Start = (unsigned char *)Ptr;
  unsigned Amount = (unsigned char *)End - Start;
  if (At+Amount > BlockEnd) 
    throw std::string("Ran out of data!");
  std::copy(At, At+Amount, Start);
  At += Amount;
}

inline void AbstractBytecodeParser::readBlock(unsigned &Type, unsigned &Size) {
  Type = read_uint();
  Size = read_uint();
  BlockStart = At;
  if ( At + Size > BlockEnd )
    throw std::string("Attempt to size a block past end of memory");
  BlockEnd = At + Size;
  if ( reportBlocks ) {
    handler->handleBlock( Type, BlockStart, Size );
  }
}

const Type *AbstractBytecodeParser::getType(unsigned ID) {
//cerr << "Looking up Type ID: " << ID << "\n";

if (ID < Type::FirstDerivedTyID)
  if (const Type *T = Type::getPrimitiveType((Type::TypeID)ID))
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

bool AbstractBytecodeParser::ParseInstruction(std::vector<unsigned> &Operands) {
  BufPtr SaveAt = At;
  Operands.clear();
  unsigned iType = 0;
  unsigned Opcode = 0;
  unsigned Op = read_uint();

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
    At -= 4;  // Hrm, try this again...
    Opcode = read_vbr_uint();
    Opcode >>= 2;
    iType = read_vbr_uint();

    unsigned NumOperands = read_vbr_uint();
    Operands.resize(NumOperands);

    if (NumOperands == 0)
      PARSE_ERROR("Zero-argument instruction found; this is invalid.");

    for (unsigned i = 0; i != NumOperands; ++i)
      Operands[i] = read_vbr_uint();
    align32();
    break;
  }

  return handler->handleInstruction(Opcode, getType(iType), Operands, At-SaveAt);
}

/// ParseBasicBlock - In LLVM 1.0 bytecode files, we used to output one
/// basicblock at a time.  This method reads in one of the basicblock packets.
void AbstractBytecodeParser::ParseBasicBlock( unsigned BlockNo) {
  handler->handleBasicBlockBegin( BlockNo );

  std::vector<unsigned> Args;
  bool is_terminating = false;
  while ( moreInBlock() )
    is_terminating = ParseInstruction(Args);

  if ( ! is_terminating )
    PARSE_ERROR("Non-terminated basic block found!");

  handler->handleBasicBlockEnd( BlockNo );
}

/// ParseInstructionList - Parse all of the BasicBlock's & Instruction's in the
/// body of a function.  In post 1.0 bytecode files, we no longer emit basic
/// block individually, in order to avoid per-basic-block overhead.
unsigned AbstractBytecodeParser::ParseInstructionList() {
  unsigned BlockNo = 0;
  std::vector<unsigned> Args;

  while ( moreInBlock() ) {
    handler->handleBasicBlockBegin( BlockNo );

    // Read instructions into this basic block until we get to a terminator
    bool is_terminating = false;
    while (moreInBlock() && !is_terminating )
        is_terminating = ParseInstruction(Args ) ;

    if (!is_terminating)
      PARSE_ERROR( "Non-terminated basic block found!");

    handler->handleBasicBlockEnd( BlockNo );
    ++BlockNo;
  }
  return BlockNo;
}

void AbstractBytecodeParser::ParseSymbolTable() {
  handler->handleSymbolTableBegin();

  while ( moreInBlock() ) {
    // Symtab block header: [num entries][type id number]
    unsigned NumEntries = read_vbr_uint();
    unsigned Typ = read_vbr_uint();
    const Type *Ty = getType(Typ);

    handler->handleSymbolTablePlane( Typ, NumEntries, Ty );

    for (unsigned i = 0; i != NumEntries; ++i) {
      // Symtab entry: [def slot #][name]
      unsigned slot = read_vbr_uint();
      std::string Name = read_str();

      if (Typ == Type::TypeTyID)
        handler->handleSymbolTableType( i, slot, Name );
      else
        handler->handleSymbolTableValue( i, slot, Name );
    }
  }
  checkPastBlockEnd("Symbol Table");

  handler->handleSymbolTableEnd();
}

void AbstractBytecodeParser::ParseFunctionLazily() {
  if (FunctionSignatureList.empty())
    throw std::string("FunctionSignatureList empty!");

  Function *Func = FunctionSignatureList.back();
  FunctionSignatureList.pop_back();

  // Save the information for future reading of the function
  LazyFunctionLoadMap[Func] = LazyFunctionInfo(BlockStart, BlockEnd);

  // Pretend we've `parsed' this function
  At = BlockEnd;
}

void AbstractBytecodeParser::ParseNextFunction(Function* Func) {
  // Find {start, end} pointers and slot in the map. If not there, we're done.
  LazyFunctionMap::iterator Fi = LazyFunctionLoadMap.find(Func);

  // Make sure we found it
  if ( Fi == LazyFunctionLoadMap.end() ) {
    PARSE_ERROR("Unrecognized function of type " << Func->getType()->getDescription());
    return;
  }

  BlockStart = At = Fi->second.Buf;
  BlockEnd = Fi->second.Buf;
  assert(Fi->first == Func);

  LazyFunctionLoadMap.erase(Fi);

  this->ParseFunctionBody( Func );
}

void AbstractBytecodeParser::ParseAllFunctionBodies() {
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

void AbstractBytecodeParser::ParseFunctionBody(Function* Func ) {

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

  Func->setLinkage( Linkage );
  handler->handleFunctionBegin(Func,FuncSize);

  // Keep track of how many basic blocks we have read in...
  unsigned BlockNum = 0;
  bool InsertedArguments = false;

  BufPtr MyEnd = BlockEnd;
  while ( At < MyEnd ) {
    unsigned Type, Size;
    BufPtr OldAt = At;
    readBlock(Type, Size);

    switch (Type) {
    case BytecodeFormat::ConstantPool:
      ParseConstantPool(FunctionTypes );
      break;

    case BytecodeFormat::CompactionTable:
      ParseCompactionTable();
      break;

    case BytecodeFormat::BasicBlock:
      ParseBasicBlock(BlockNum++);
      break;

    case BytecodeFormat::InstructionList:
      if (BlockNum) 
        PARSE_ERROR("InstructionList must come before basic blocks!");
      BlockNum = ParseInstructionList();
      break;

    case BytecodeFormat::SymbolTable:
      ParseSymbolTable();
      break;

    default:
      At += Size;
      if (OldAt > At)
        PARSE_ERROR("Wrapped around reading bytecode");
      break;
    }
    BlockEnd = MyEnd;

    // Malformed bc file if read past end of block.
    align32();
  }

  handler->handleFunctionEnd(Func);

  // Clear out function-level types...
  FunctionTypes.clear();
  CompactionTypeTable.clear();
}

void AbstractBytecodeParser::ParseCompactionTable() {

  handler->handleCompactionTableBegin();

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

    handler->handleCompactionTablePlane( Ty, NumEntries );

    if (Ty == Type::TypeTyID) {
      for (unsigned i = 0; i != NumEntries; ++i) {
        unsigned TypeSlot = read_vbr_uint();
        const Type *Typ = getGlobalTableType(TypeSlot);
        handler->handleCompactionTableType( i, TypeSlot, Typ );
      }
    } else {
      const Type *Typ = getType(Ty);
      // Push the implicit zero
      for (unsigned i = 0; i != NumEntries; ++i) {
        unsigned ValSlot = read_vbr_uint();
        handler->handleCompactionTableValue( i, ValSlot, Typ );
      }
    }
  }
  handler->handleCompactionTableEnd();
}

const Type *AbstractBytecodeParser::ParseTypeConstant() {
  unsigned PrimType = read_vbr_uint();

  const Type *Val = 0;
  if ((Val = Type::getPrimitiveType((Type::TypeID)PrimType)))
    return Val;
  
  switch (PrimType) {
  case Type::FunctionTyID: {
    const Type *RetType = getType(read_vbr_uint());

    unsigned NumParams = read_vbr_uint();

    std::vector<const Type*> Params;
    while (NumParams--)
      Params.push_back(getType(read_vbr_uint()));

    bool isVarArg = Params.size() && Params.back() == Type::VoidTy;
    if (isVarArg) Params.pop_back();

    Type* result = FunctionType::get(RetType, Params, isVarArg);
    handler->handleType( result );
    return result;
  }
  case Type::ArrayTyID: {
    unsigned ElTyp = read_vbr_uint();
    const Type *ElementType = getType(ElTyp);

    unsigned NumElements = read_vbr_uint();

    BCR_TRACE(5, "Array Type Constant #" << ElTyp << " size=" 
              << NumElements << "\n");
    Type* result =  ArrayType::get(ElementType, NumElements);
    handler->handleType( result );
    return result;
  }
  case Type::StructTyID: {
    std::vector<const Type*> Elements;
    unsigned Typ = read_vbr_uint();
    while (Typ) {         // List is terminated by void/0 typeid
      Elements.push_back(getType(Typ));
      Typ = read_vbr_uint();
    }

    Type* result = StructType::get(Elements);
    handler->handleType( result );
    return result;
  }
  case Type::PointerTyID: {
    unsigned ElTyp = read_vbr_uint();
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
void AbstractBytecodeParser::ParseTypeConstants( 
  TypeListTy &Tab, unsigned NumEntries
) {
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


void AbstractBytecodeParser::ParseConstantValue(unsigned TypeID) {

  // We must check for a ConstantExpr before switching by type because
  // a ConstantExpr can be of any type, and has no explicit value.
  // 
  // 0 if not expr; numArgs if is expr
  unsigned isExprNumArgs = read_vbr_uint();
  
  if (isExprNumArgs) {
    unsigned Opcode = read_vbr_uint();
    const Type* Typ = getType(TypeID);
    
    // FIXME: Encoding of constant exprs could be much more compact!
    std::vector<std::pair<const Type*,unsigned> > ArgVec;
    ArgVec.reserve(isExprNumArgs);

    // Read the slot number and types of each of the arguments
    for (unsigned i = 0; i != isExprNumArgs; ++i) {
      unsigned ArgValSlot = read_vbr_uint();
      unsigned ArgTypeSlot = read_vbr_uint();
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
  switch (Ty->getTypeID()) {
  case Type::BoolTyID: {
    unsigned Val = read_vbr_uint();
    if (Val != 0 && Val != 1) 
      PARSE_ERROR("Invalid boolean value read.");

    handler->handleConstantValue( ConstantBool::get(Val == 1));
    break;
  }

  case Type::UByteTyID:   // Unsigned integer types...
  case Type::UShortTyID:
  case Type::UIntTyID: {
    unsigned Val = read_vbr_uint();
    if (!ConstantUInt::isValueValidForType(Ty, Val)) 
      throw std::string("Invalid unsigned byte/short/int read.");
    handler->handleConstantValue( ConstantUInt::get(Ty, Val) );
    break;
  }

  case Type::ULongTyID: {
    handler->handleConstantValue( ConstantUInt::get(Ty, read_vbr_uint64()) );
    break;
  }

  case Type::SByteTyID:   // Signed integer types...
  case Type::ShortTyID:
  case Type::IntTyID: {
  case Type::LongTyID:
    int64_t Val = read_vbr_int64();
    if (!ConstantSInt::isValueValidForType(Ty, Val)) 
      throw std::string("Invalid signed byte/short/int/long read.");
    handler->handleConstantValue(  ConstantSInt::get(Ty, Val) );
    break;
  }

  case Type::FloatTyID: {
    float F;
    read_data(&F, &F+1);
    handler->handleConstantValue( ConstantFP::get(Ty, F) );
    break;
  }

  case Type::DoubleTyID: {
    double Val;
    read_data(&Val, &Val+1);
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
      Elements.push_back(read_vbr_uint());

    handler->handleConstantArray( AT, Elements );
    break;
  }

  case Type::StructTyID: {
    const StructType *ST = cast<StructType>(Ty);
    std::vector<unsigned> Elements;
    Elements.reserve(ST->getNumElements());
    for (unsigned i = 0; i != ST->getNumElements(); ++i)
      Elements.push_back(read_vbr_uint());
    handler->handleConstantStruct( ST, Elements );
    break;
  }    

  case Type::PointerTyID: {  // ConstantPointerRef value...
    const PointerType *PT = cast<PointerType>(Ty);
    unsigned Slot = read_vbr_uint();
    handler->handleConstantPointer( PT, Slot );
    break;
  }

  default:
    PARSE_ERROR("Don't know how to deserialize constant value of type '"+
                      Ty->getDescription());
  }
}

void AbstractBytecodeParser::ParseGlobalTypes() {
  ParseConstantPool(ModuleTypes);
}

void AbstractBytecodeParser::ParseStringConstants(unsigned NumEntries ){
  for (; NumEntries; --NumEntries) {
    unsigned Typ = read_vbr_uint();
    const Type *Ty = getType(Typ);
    if (!isa<ArrayType>(Ty))
      throw std::string("String constant data invalid!");
    
    const ArrayType *ATy = cast<ArrayType>(Ty);
    if (ATy->getElementType() != Type::SByteTy &&
        ATy->getElementType() != Type::UByteTy)
      throw std::string("String constant data invalid!");
    
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
    ConstantArray *C = cast<ConstantArray>( ConstantArray::get(ATy, Elements) );
    handler->handleConstantString( C );
  }
}


void AbstractBytecodeParser::ParseConstantPool( TypeListTy &TypeTab) {
  while ( moreInBlock() ) {
    unsigned NumEntries = read_vbr_uint();
    unsigned Typ = read_vbr_uint();
    if (Typ == Type::TypeTyID) {
      ParseTypeConstants(TypeTab, NumEntries);
    } else if (Typ == Type::VoidTyID) {
      ParseStringConstants(NumEntries);
    } else {
      BCR_TRACE(3, "Type: '" << *getType(Typ) << "'  NumEntries: "
                << NumEntries << "\n");

      for (unsigned i = 0; i < NumEntries; ++i) {
        ParseConstantValue(Typ);
      }
    }
  }
  
  checkPastBlockEnd("Constant Pool");
}

void AbstractBytecodeParser::ParseModuleGlobalInfo() {

  handler->handleModuleGlobalsBegin();

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
    if (hasInitializer) {
      unsigned initSlot = read_vbr_uint();
      handler->handleInitializedGV( ElTy, isConstant, Linkage, initSlot );
    } else 
      handler->handleGlobalVariable( ElTy, isConstant, Linkage );

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
    Function* Func = new Function(FTy, GlobalValue::ExternalLinkage);

    // Save this for later so we know type of lazily instantiated functions
    FunctionSignatureList.push_back(Func);

    handler->handleFunctionDeclaration(Func, FTy);

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

  handler->handleModuleGlobalsEnd();
}

void AbstractBytecodeParser::ParseVersionInfo() {
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

  handler->handleVersionInfo(RevisionNum, Endianness, PointerSize );
}

void AbstractBytecodeParser::ParseModule() {
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
    readBlock(Type, Size);

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
      ParseConstantPool(ModuleTypes);
      break;

    case BytecodeFormat::Function:
      ParseFunctionLazily();
      break;

    case BytecodeFormat::SymbolTable:
      ParseSymbolTable();
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

  /// Make sure we pulled them all out. If we didn't then there's a declaration
  /// but a missing body. That's not allowed.
  if (!FunctionSignatureList.empty())
    throw std::string(
      "Function declared, but bytecode stream ended before definition");
}

void AbstractBytecodeParser::ParseBytecode(
       BufPtr b, unsigned Length,
       const std::string &ModuleID) {

  At = MemStart = BlockStart = b;
  MemEnd = BlockEnd = b + Length;
  handler->handleStart();

  // Read and check signature...
  unsigned Sig = read_uint();
  if (Sig != ('l' | ('l' << 8) | ('v' << 16) | ('m' << 24))) {
    PARSE_ERROR("Invalid bytecode signature: " << Sig);
  }

  handler->handleModuleBegin(ModuleID);

  unsigned Type, Size;
  readBlock(Type, Size);
  if ( Type != BytecodeFormat::Module ) {
    PARSE_ERROR("Expected Module Block! At: " << unsigned(intptr_t(At))
      << ", Type:" << Type << ", Size:" << Size);
  }
  if ( At + Size != MemEnd ) {
    PARSE_ERROR("Invalid Top Level Block Length! At: " 
      << unsigned(intptr_t(At)) << ", Type:" << Type << ", Size:" << Size);
  }
  this->ParseModule();

  handler->handleModuleEnd(ModuleID);

  handler->handleFinish();
}

//===----------------------------------------------------------------------===//
//=== Default Implementations of Handler Methods
//===----------------------------------------------------------------------===//

bool BytecodeHandler::handleError(const std::string& str ) { return false; }
void BytecodeHandler::handleStart() { }
void BytecodeHandler::handleFinish() { }
void BytecodeHandler::handleModuleBegin(const std::string& id) { }
void BytecodeHandler::handleModuleEnd(const std::string& id) { }
void BytecodeHandler::handleVersionInfo( unsigned char RevisionNum,
  Module::Endianness Endianness, Module::PointerSize PointerSize) { }
void BytecodeHandler::handleModuleGlobalsBegin() { }
void BytecodeHandler::handleGlobalVariable( 
  const Type* ElemType, bool isConstant, GlobalValue::LinkageTypes ) { }
void BytecodeHandler::handleInitializedGV( 
  const Type* ElemType, bool isConstant, GlobalValue::LinkageTypes,
  unsigned initSlot) {}
void BytecodeHandler::handleType( const Type* Ty ) {}
void BytecodeHandler::handleFunctionDeclaration( 
  Function* Func, const FunctionType* FuncType) {}
void BytecodeHandler::handleModuleGlobalsEnd() { } 
void BytecodeHandler::handleCompactionTableBegin() { } 
void BytecodeHandler::handleCompactionTablePlane( unsigned Ty, 
  unsigned NumEntries) {}
void BytecodeHandler::handleCompactionTableType( unsigned i, unsigned TypSlot, 
  const Type* ) {}
void BytecodeHandler::handleCompactionTableValue( unsigned i, unsigned ValSlot,
  const Type* ) {}
void BytecodeHandler::handleCompactionTableEnd() { }
void BytecodeHandler::handleSymbolTableBegin() { }
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
  const Type* Typ, std::vector<std::pair<const Type*,unsigned> > ArgVec ) { }
void BytecodeHandler::handleConstantValue( Constant * c ) { }
void BytecodeHandler::handleConstantArray( const ArrayType* AT, 
  std::vector<unsigned>& Elements ) { }
void BytecodeHandler::handleConstantStruct( const StructType* ST,
  std::vector<unsigned>& ElementSlots) { }
void BytecodeHandler::handleConstantPointer( 
  const PointerType* PT, unsigned Slot) { }
void BytecodeHandler::handleConstantString( const ConstantArray* CA ) {}
void BytecodeHandler::handleGlobalConstantsEnd() {}
void BytecodeHandler::handleAlignment(unsigned numBytes) {}
void BytecodeHandler::handleBlock(
  unsigned BType, const unsigned char* StartPtr, unsigned Size) {}
void BytecodeHandler::handleVBR32(unsigned Size ) {}
void BytecodeHandler::handleVBR64(unsigned Size ) {}

// vim: sw=2
