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

#include "ReaderInternals.h"
#include "llvm/Bytecode/Reader.h"
#include "llvm/Bytecode/Format.h"
#include "llvm/Module.h"
#include "Support/StringExtras.h"
using namespace llvm;

unsigned BytecodeParser::getTypeSlot(const Type *Ty) {
  if (Ty->isPrimitiveType())
    return Ty->getPrimitiveID();

  // Scan the compaction table for the type if needed.
  if (CompactionTable.size() > Type::TypeTyID) {
    std::vector<Value*> &Plane = CompactionTable[Type::TypeTyID];
    if (!Plane.empty()) {
      std::vector<Value*>::iterator I = find(Plane.begin(), Plane.end(),
                                             const_cast<Type*>(Ty));
      if (I == Plane.end())
        throw std::string("Couldn't find type specified in compaction table!");
      return Type::FirstDerivedTyID + (&*I - &Plane[0]);
    }
  }

  // Check the function level types first...
  TypeValuesListTy::iterator I = find(FunctionTypeValues.begin(),
                                      FunctionTypeValues.end(), Ty);
  if (I != FunctionTypeValues.end())
    return Type::FirstDerivedTyID + ModuleTypeValues.size() +
             (&*I - &FunctionTypeValues[0]);

  I = find(ModuleTypeValues.begin(), ModuleTypeValues.end(), Ty);
  if (I == ModuleTypeValues.end())
    throw std::string("Didn't find type in ModuleTypeValues.");
  return Type::FirstDerivedTyID + (&*I - &ModuleTypeValues[0]);
}

const Type *BytecodeParser::getType(unsigned ID) {
  //cerr << "Looking up Type ID: " << ID << "\n";

  if (ID < Type::FirstDerivedTyID)
    if (const Type *T = Type::getPrimitiveType((Type::PrimitiveID)ID))
      return T;   // Asked for a primitive type...

  // Otherwise, derived types need offset...
  ID -= Type::FirstDerivedTyID;

  if (CompactionTable.size() > Type::TypeTyID &&
      !CompactionTable[Type::TypeTyID].empty()) {
    if (ID >= CompactionTable[Type::TypeTyID].size())
      throw std::string("Type ID out of range for compaction table!");
    return cast<Type>(CompactionTable[Type::TypeTyID][ID]);
  }

  // Is it a module-level type?
  if (ID < ModuleTypeValues.size())
    return ModuleTypeValues[ID].get();

  // Nope, is it a function-level type?
  ID -= ModuleTypeValues.size();
  if (ID < FunctionTypeValues.size())
    return FunctionTypeValues[ID].get();

  throw std::string("Illegal type reference!");
}

static inline bool hasImplicitNull(unsigned TyID, bool EncodesPrimitiveZeros) {
  if (!EncodesPrimitiveZeros)
    return TyID != Type::LabelTyID && TyID != Type::TypeTyID &&
           TyID != Type::VoidTyID;
  return TyID >= Type::FirstDerivedTyID;
}

unsigned BytecodeParser::insertValue(Value *Val, unsigned type,
                                     ValueTable &ValueTab) {
  assert((!isa<Constant>(Val) || !cast<Constant>(Val)->isNullValue()) ||
          !hasImplicitNull(type, hasExplicitPrimitiveZeros) &&
         "Cannot read null values from bytecode!");
  assert(type != Type::TypeTyID && "Types should never be insertValue'd!");

  if (ValueTab.size() <= type)
    ValueTab.resize(type+1);

  if (!ValueTab[type]) ValueTab[type] = new ValueList();

  //cerr << "insertValue Values[" << type << "][" << ValueTab[type].size() 
  //   << "] = " << Val << "\n";
  ValueTab[type]->push_back(Val);

  bool HasOffset = hasImplicitNull(type, hasExplicitPrimitiveZeros);
  return ValueTab[type]->size()-1 + HasOffset;
}

Value *BytecodeParser::getValue(unsigned type, unsigned oNum, bool Create) {
  assert(type != Type::TypeTyID && "getValue() cannot get types!");
  assert(type != Type::LabelTyID && "getValue() cannot get blocks!");
  unsigned Num = oNum;

  // If there is a compaction table active, it defines the low-level numbers.
  // If not, the module values define the low-level numbers.
  if (CompactionTable.size() > type && !CompactionTable[type].empty()) {
    if (Num < CompactionTable[type].size())
      return CompactionTable[type][Num];
    Num -= CompactionTable[type].size();
  } else {
    // If the type plane was compactified, figure out the global type ID.
    unsigned GlobalTyID = type;
    if (CompactionTable.size() > Type::TypeTyID &&
        !CompactionTable[Type::TypeTyID].empty() &&
        type >= Type::FirstDerivedTyID) {
      std::vector<Value*> &TypePlane = CompactionTable[Type::TypeTyID];
      const Type *Ty = cast<Type>(TypePlane[type-Type::FirstDerivedTyID]);
      TypeValuesListTy::iterator I =
        find(ModuleTypeValues.begin(), ModuleTypeValues.end(), Ty);
      assert(I != ModuleTypeValues.end());
      GlobalTyID = Type::FirstDerivedTyID + (&*I - &ModuleTypeValues[0]);
    }

    if (hasImplicitNull(GlobalTyID, hasExplicitPrimitiveZeros)) {
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

  if (Values.size() > type && Values[type] && Num < Values[type]->size())
    return Values[type]->getOperand(Num);

  if (!Create) return 0;  // Do not create a placeholder?

  std::pair<unsigned,unsigned> KeyValue(type, oNum);
  std::map<std::pair<unsigned,unsigned>, Value*>::iterator I = 
    ForwardReferences.lower_bound(KeyValue);
  if (I != ForwardReferences.end() && I->first == KeyValue)
    return I->second;   // We have already created this placeholder

  Value *Val = new Argument(getType(type));
  ForwardReferences.insert(I, std::make_pair(KeyValue, Val));
  return Val;
}

/// getBasicBlock - Get a particular numbered basic block, which might be a
/// forward reference.  This works together with ParseBasicBlock to handle these
/// forward references in a clean manner.
///
BasicBlock *BytecodeParser::getBasicBlock(unsigned ID) {
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

/// getConstantValue - Just like getValue, except that it returns a null pointer
/// only on error.  It always returns a constant (meaning that if the value is
/// defined, but is not a constant, that is an error).  If the specified
/// constant hasn't been parsed yet, a placeholder is defined and used.  Later,
/// after the real value is parsed, the placeholder is eliminated.
///
Constant *BytecodeParser::getConstantValue(unsigned TypeSlot, unsigned Slot) {
  if (Value *V = getValue(TypeSlot, Slot, false))
    if (Constant *C = dyn_cast<Constant>(V))
      return C;   // If we already have the value parsed, just return it
    else if (GlobalValue *GV = dyn_cast<GlobalValue>(V))
      // ConstantPointerRef's are an abomination, but at least they don't have
      // to infest bytecode files.
      return ConstantPointerRef::get(GV);
    else
      throw std::string("Reference of a value is expected to be a constant!");

  const Type *Ty = getType(TypeSlot);
  std::pair<const Type*, unsigned> Key(Ty, Slot);
  ConstantRefsType::iterator I = ConstantFwdRefs.lower_bound(Key);

  if (I != ConstantFwdRefs.end() && I->first == Key) {
    BCR_TRACE(5, "Previous forward ref found!\n");
    return I->second;
  } else {
    // Create a placeholder for the constant reference and
    // keep track of the fact that we have a forward ref to recycle it
    BCR_TRACE(5, "Creating new forward ref to a constant!\n");
    Constant *C = new ConstPHolder(Ty, Slot);
    
    // Keep track of the fact that we have a forward ref to recycle it
    ConstantFwdRefs.insert(I, std::make_pair(Key, C));
    return C;
  }
}

/// ParseBasicBlock - In LLVM 1.0 bytecode files, we used to output one
/// basicblock at a time.  This method reads in one of the basicblock packets.
BasicBlock *BytecodeParser::ParseBasicBlock(const unsigned char *&Buf,
                                            const unsigned char *EndBuf,
                                            unsigned BlockNo) {
  BasicBlock *BB;
  if (ParsedBasicBlocks.size() == BlockNo)
    ParsedBasicBlocks.push_back(BB = new BasicBlock());
  else if (ParsedBasicBlocks[BlockNo] == 0)
    BB = ParsedBasicBlocks[BlockNo] = new BasicBlock();
  else
    BB = ParsedBasicBlocks[BlockNo];

  std::vector<unsigned> Args;
  while (Buf < EndBuf)
    ParseInstruction(Buf, EndBuf, Args, BB);

  return BB;
}


/// ParseInstructionList - Parse all of the BasicBlock's & Instruction's in the
/// body of a function.  In post 1.0 bytecode files, we no longer emit basic
/// block individually, in order to avoid per-basic-block overhead.
unsigned BytecodeParser::ParseInstructionList(Function *F,
                                              const unsigned char *&Buf,
                                              const unsigned char *EndBuf) {
  unsigned BlockNo = 0;
  std::vector<unsigned> Args;

  while (Buf < EndBuf) {
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
    while (Buf < EndBuf && !BB->getTerminator())
      ParseInstruction(Buf, EndBuf, Args, BB);

    if (!BB->getTerminator())
      throw std::string("Non-terminated basic block found!");
  }

  return BlockNo;
}

void BytecodeParser::ParseSymbolTable(const unsigned char *&Buf,
                                      const unsigned char *EndBuf,
                                      SymbolTable *ST,
                                      Function *CurrentFunction) {
  // Allow efficient basic block lookup by number.
  std::vector<BasicBlock*> BBMap;
  if (CurrentFunction)
    for (Function::iterator I = CurrentFunction->begin(),
           E = CurrentFunction->end(); I != E; ++I)
      BBMap.push_back(I);

  while (Buf < EndBuf) {
    // Symtab block header: [num entries][type id number]
    unsigned NumEntries = read_vbr_uint(Buf, EndBuf);
    unsigned Typ = read_vbr_uint(Buf, EndBuf);
    const Type *Ty = getType(Typ);
    BCR_TRACE(3, "Plane Type: '" << *Ty << "' with " << NumEntries <<
                 " entries\n");

    for (unsigned i = 0; i != NumEntries; ++i) {
      // Symtab entry: [def slot #][name]
      unsigned slot = read_vbr_uint(Buf, EndBuf);
      std::string Name = read_str(Buf, EndBuf);

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
        throw "Failed value look-up for name '" + Name + "'";
      BCR_TRACE(4, "Map: '" << Name << "' to #" << slot << ":" << *V;
                if (!isa<Instruction>(V)) std::cerr << "\n");

      V->setName(Name, ST);
    }
  }

  if (Buf > EndBuf) throw std::string("Tried to read past end of buffer.");
}

void BytecodeParser::ResolveReferencesToConstant(Constant *NewV, unsigned Slot){
  ConstantRefsType::iterator I =
    ConstantFwdRefs.find(std::make_pair(NewV->getType(), Slot));
  if (I == ConstantFwdRefs.end()) return;   // Never forward referenced?

  BCR_TRACE(3, "Mutating forward refs!\n");
  Value *PH = I->second;   // Get the placeholder...
  PH->replaceAllUsesWith(NewV);
  delete PH;                               // Delete the old placeholder
  ConstantFwdRefs.erase(I);                // Remove the map entry for it
}

void BytecodeParser::ParseFunction(const unsigned char *&Buf,
                                   const unsigned char *EndBuf) {
  if (FunctionSignatureList.empty())
    throw std::string("FunctionSignatureList empty!");

  Function *F = FunctionSignatureList.back();
  FunctionSignatureList.pop_back();

  // Save the information for future reading of the function
  LazyFunctionLoadMap[F] = LazyFunctionInfo(Buf, EndBuf);
  // Pretend we've `parsed' this function
  Buf = EndBuf;
}

void BytecodeParser::materializeFunction(Function* F) {
  // Find {start, end} pointers and slot in the map. If not there, we're done.
  std::map<Function*, LazyFunctionInfo>::iterator Fi =
    LazyFunctionLoadMap.find(F);
  if (Fi == LazyFunctionLoadMap.end()) return;

  const unsigned char *Buf = Fi->second.Buf;
  const unsigned char *EndBuf = Fi->second.EndBuf;
  LazyFunctionLoadMap.erase(Fi);

  GlobalValue::LinkageTypes Linkage = GlobalValue::ExternalLinkage;

  unsigned LinkageType = read_vbr_uint(Buf, EndBuf);
  if (LinkageType > 4)
    throw std::string("Invalid linkage type for Function.");
  switch (LinkageType) {
  case 0: Linkage = GlobalValue::ExternalLinkage; break;
  case 1: Linkage = GlobalValue::WeakLinkage; break;
  case 2: Linkage = GlobalValue::AppendingLinkage; break;
  case 3: Linkage = GlobalValue::InternalLinkage; break;
  case 4: Linkage = GlobalValue::LinkOnceLinkage; break;
  }

  F->setLinkage(Linkage);

  // Keep track of how many basic blocks we have read in...
  unsigned BlockNum = 0;
  bool InsertedArguments = false;

  while (Buf < EndBuf) {
    unsigned Type, Size;
    const unsigned char *OldBuf = Buf;
    readBlock(Buf, EndBuf, Type, Size);

    switch (Type) {
    case BytecodeFormat::ConstantPool:
      if (!InsertedArguments) {
        // Insert arguments into the value table before we parse the first basic
        // block in the function, but after we potentially read in the
        // compaction table.
        const FunctionType *FT = F->getFunctionType();
        Function::aiterator AI = F->abegin();
        for (FunctionType::param_iterator It = FT->param_begin();
             It != FT->param_end(); ++It, ++AI)
          insertValue(AI, getTypeSlot(AI->getType()), Values);
        InsertedArguments = true;
      }

      BCR_TRACE(2, "BLOCK BytecodeFormat::ConstantPool: {\n");
      ParseConstantPool(Buf, Buf+Size, Values, FunctionTypeValues);
      break;

    case BytecodeFormat::CompactionTable:
      BCR_TRACE(2, "BLOCK BytecodeFormat::CompactionTable: {\n");
      ParseCompactionTable(Buf, Buf+Size);
      break;

    case BytecodeFormat::BasicBlock: {
      if (!InsertedArguments) {
        // Insert arguments into the value table before we parse the first basic
        // block in the function, but after we potentially read in the
        // compaction table.
        const FunctionType *FT = F->getFunctionType();
        Function::aiterator AI = F->abegin();
        for (FunctionType::param_iterator It = FT->param_begin();
             It != FT->param_end(); ++It, ++AI)
          insertValue(AI, getTypeSlot(AI->getType()), Values);
        InsertedArguments = true;
      }

      BCR_TRACE(2, "BLOCK BytecodeFormat::BasicBlock: {\n");
      BasicBlock *BB = ParseBasicBlock(Buf, Buf+Size, BlockNum++);
      F->getBasicBlockList().push_back(BB);
      break;
    }

    case BytecodeFormat::InstructionList: {
      // Insert arguments into the value table before we parse the instruction
      // list for the function, but after we potentially read in the compaction
      // table.
      if (!InsertedArguments) {
        const FunctionType *FT = F->getFunctionType();
        Function::aiterator AI = F->abegin();
        for (FunctionType::param_iterator It = FT->param_begin();
             It != FT->param_end(); ++It, ++AI)
          insertValue(AI, getTypeSlot(AI->getType()), Values);
        InsertedArguments = true;
      }

      BCR_TRACE(2, "BLOCK BytecodeFormat::InstructionList: {\n");
      if (BlockNum) throw std::string("Already parsed basic blocks!");
      BlockNum = ParseInstructionList(F, Buf, Buf+Size);
      break;
    }

    case BytecodeFormat::SymbolTable:
      BCR_TRACE(2, "BLOCK BytecodeFormat::SymbolTable: {\n");
      ParseSymbolTable(Buf, Buf+Size, &F->getSymbolTable(), F);
      break;

    default:
      BCR_TRACE(2, "BLOCK <unknown>:ignored! {\n");
      Buf += Size;
      if (OldBuf > Buf) 
        throw std::string("Wrapped around reading bytecode.");
      break;
    }
    BCR_TRACE(2, "} end block\n");

    // Malformed bc file if read past end of block.
    align32(Buf, EndBuf);
  }

  // Make sure there were no references to non-existant basic blocks.
  if (BlockNum != ParsedBasicBlocks.size())
    throw std::string("Illegal basic block operand reference");
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
  FunctionTypeValues.clear();
  CompactionTable.clear();
  freeTable(Values);
}

void BytecodeParser::ParseCompactionTable(const unsigned char *&Buf,
                                          const unsigned char *End) {

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

    if (Ty >= CompactionTable.size())
      CompactionTable.resize(Ty+1);

    if (!CompactionTable[Ty].empty())
      throw std::string("Compaction table plane contains multiple entries!");
    
    if (Ty == Type::TypeTyID) {
      for (unsigned i = 0; i != NumEntries; ++i) {
        const Type *Typ = getGlobalTableType(read_vbr_uint(Buf, End));
        CompactionTable[Type::TypeTyID].push_back(const_cast<Type*>(Typ));
      }

      CompactionTable.resize(NumEntries+Type::FirstDerivedTyID);
    } else {
      const Type *Typ = getType(Ty);
      // Push the implicit zero
      CompactionTable[Ty].push_back(Constant::getNullValue(Typ));
      for (unsigned i = 0; i != NumEntries; ++i) {
        Value *V = getGlobalTableValue(Typ, read_vbr_uint(Buf, End));
        CompactionTable[Ty].push_back(V);
      }
    }
  }

}



void BytecodeParser::ParseModuleGlobalInfo(const unsigned char *&Buf,
                                           const unsigned char *End) {
  if (!FunctionSignatureList.empty())
    throw std::string("Two ModuleGlobalInfo packets found!");

  // Read global variables...
  unsigned VarType = read_vbr_uint(Buf, End);
  while (VarType != Type::VoidTyID) { // List is terminated by Void
    // VarType Fields: bit0 = isConstant, bit1 = hasInitializer, bit2,3,4 =
    // Linkage, bit4+ = slot#
    unsigned SlotNo = VarType >> 5;
    unsigned LinkageID = (VarType >> 2) & 7;
    GlobalValue::LinkageTypes Linkage;

    switch (LinkageID) {
    default: assert(0 && "Unknown linkage type!");
    case 0: Linkage = GlobalValue::ExternalLinkage;  break;
    case 1: Linkage = GlobalValue::WeakLinkage;      break;
    case 2: Linkage = GlobalValue::AppendingLinkage; break;
    case 3: Linkage = GlobalValue::InternalLinkage;  break;
    case 4: Linkage = GlobalValue::LinkOnceLinkage;  break;
    }

    const Type *Ty = getType(SlotNo);
    if (!isa<PointerType>(Ty))
      throw std::string("Global not pointer type!  Ty = " + 
                        Ty->getDescription());

    const Type *ElTy = cast<PointerType>(Ty)->getElementType();

    // Create the global variable...
    GlobalVariable *GV = new GlobalVariable(ElTy, VarType & 1, Linkage,
                                            0, "", TheModule);
    BCR_TRACE(2, "Global Variable of type: " << *Ty << "\n");
    insertValue(GV, SlotNo, ModuleValues);

    if (VarType & 2)   // Does it have an initializer?
      GlobalInits.push_back(std::make_pair(GV, read_vbr_uint(Buf, End)));
    VarType = read_vbr_uint(Buf, End);
  }

  // Read the function objects for all of the functions that are coming
  unsigned FnSignature = read_vbr_uint(Buf, End);
  while (FnSignature != Type::VoidTyID) { // List is terminated by Void
    const Type *Ty = getType(FnSignature);
    if (!isa<PointerType>(Ty) ||
        !isa<FunctionType>(cast<PointerType>(Ty)->getElementType()))
      throw std::string("Function not ptr to func type!  Ty = " +
                        Ty->getDescription());

    // We create functions by passing the underlying FunctionType to create...
    Ty = cast<PointerType>(Ty)->getElementType();

    // When the ModuleGlobalInfo section is read, we load the type of each
    // function and the 'ModuleValues' slot that it lands in.  We then load a
    // placeholder into its slot to reserve it.  When the function is loaded,
    // this placeholder is replaced.

    // Insert the placeholder...
    Function *Func = new Function(cast<FunctionType>(Ty),
                                  GlobalValue::InternalLinkage, "", TheModule);
    insertValue(Func, FnSignature, ModuleValues);

    // Keep track of this information in a list that is emptied as functions are
    // loaded...
    //
    FunctionSignatureList.push_back(Func);

    FnSignature = read_vbr_uint(Buf, End);
    BCR_TRACE(2, "Function of type: " << Ty << "\n");
  }

  if (hasInconsistentModuleGlobalInfo)
    align32(Buf, End);

  // Now that the function signature list is set up, reverse it so that we can 
  // remove elements efficiently from the back of the vector.
  std::reverse(FunctionSignatureList.begin(), FunctionSignatureList.end());

  // This is for future proofing... in the future extra fields may be added that
  // we don't understand, so we transparently ignore them.
  //
  Buf = End;
}

void BytecodeParser::ParseVersionInfo(const unsigned char *&Buf,
                                      const unsigned char *EndBuf) {
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

  switch (RevisionNum) {
  case 0:               //  LLVM 1.0, 1.1 release version
    // Compared to rev #2, we added support for weak linkage, a more dense
    // encoding, and better varargs support.

    // Base LLVM 1.0 bytecode format.
    hasInconsistentModuleGlobalInfo = true;
    hasExplicitPrimitiveZeros = true;
    // FALL THROUGH
  case 1:               // LLVM 1.2 release version
    // LLVM 1.2 added explicit support for emitting strings efficiently.

    // Also, it fixed the problem where the size of the ModuleGlobalInfo block
    // included the size for the alignment at the end, where the rest of the
    // blocks did not.
    break;

  default:
    throw std::string("Unknown bytecode version number!");
  }

  if (hasNoEndianness) Endianness  = Module::AnyEndianness;
  if (hasNoPointerSize) PointerSize = Module::AnyPointerSize;

  TheModule->setEndianness(Endianness);
  TheModule->setPointerSize(PointerSize);
  BCR_TRACE(1, "Bytecode Rev = " << (unsigned)RevisionNum << "\n");
  BCR_TRACE(1, "Endianness/PointerSize = " << Endianness << ","
               << PointerSize << "\n");
}

void BytecodeParser::ParseModule(const unsigned char *Buf,
                                 const unsigned char *EndBuf) {
  unsigned Type, Size;
  readBlock(Buf, EndBuf, Type, Size);
  if (Type != BytecodeFormat::Module || Buf+Size != EndBuf)
    throw std::string("Expected Module packet! B: "+
        utostr((unsigned)(intptr_t)Buf) + ", S: "+utostr(Size)+
        " E: "+utostr((unsigned)(intptr_t)EndBuf)); // Hrm, not a class?

  BCR_TRACE(0, "BLOCK BytecodeFormat::Module: {\n");
  FunctionSignatureList.clear();                 // Just in case...

  // Read into instance variables...
  ParseVersionInfo(Buf, EndBuf);
  align32(Buf, EndBuf);

  while (Buf < EndBuf) {
    const unsigned char *OldBuf = Buf;
    readBlock(Buf, EndBuf, Type, Size);
    switch (Type) {
    case BytecodeFormat::GlobalTypePlane:
      BCR_TRACE(1, "BLOCK BytecodeFormat::GlobalTypePlane: {\n");
      ParseGlobalTypes(Buf, Buf+Size);
      break;

    case BytecodeFormat::ModuleGlobalInfo:
      BCR_TRACE(1, "BLOCK BytecodeFormat::ModuleGlobalInfo: {\n");
      ParseModuleGlobalInfo(Buf, Buf+Size);
      break;

    case BytecodeFormat::ConstantPool:
      BCR_TRACE(1, "BLOCK BytecodeFormat::ConstantPool: {\n");
      ParseConstantPool(Buf, Buf+Size, ModuleValues, ModuleTypeValues);
      break;

    case BytecodeFormat::Function: {
      BCR_TRACE(1, "BLOCK BytecodeFormat::Function: {\n");
      ParseFunction(Buf, Buf+Size);
      break;
    }

    case BytecodeFormat::SymbolTable:
      BCR_TRACE(1, "BLOCK BytecodeFormat::SymbolTable: {\n");
      ParseSymbolTable(Buf, Buf+Size, &TheModule->getSymbolTable(), 0);
      break;
    default:
      Buf += Size;
      if (OldBuf > Buf) throw std::string("Expected Module Block!");
      break;
    }
    BCR_TRACE(1, "} end block\n");
    align32(Buf, EndBuf);
  }

  // After the module constant pool has been read, we can safely initialize
  // global variables...
  while (!GlobalInits.empty()) {
    GlobalVariable *GV = GlobalInits.back().first;
    unsigned Slot = GlobalInits.back().second;
    GlobalInits.pop_back();

    // Look up the initializer value...
    // FIXME: Preserve this type ID!
    unsigned TypeSlot = getTypeSlot(GV->getType()->getElementType());
    if (Constant *CV = getConstantValue(TypeSlot, Slot)) {
      if (GV->hasInitializer()) 
        throw std::string("Global *already* has an initializer?!");
      GV->setInitializer(CV);
    } else
      throw std::string("Cannot find initializer value.");
  }

  if (!FunctionSignatureList.empty())
    throw std::string("Function expected, but bytecode stream ended!");

  BCR_TRACE(0, "} end block\n\n");
}

void BytecodeParser::ParseBytecode(const unsigned char *Buf, unsigned Length,
                                   const std::string &ModuleID) {

  unsigned char *EndBuf = (unsigned char*)(Buf + Length);

  // Read and check signature...
  unsigned Sig = read(Buf, EndBuf);
  if (Sig != ('l' | ('l' << 8) | ('v' << 16) | ('m' << 24)))
    throw std::string("Invalid bytecode signature!");

  TheModule = new Module(ModuleID);
  try { 
    ParseModule(Buf, EndBuf);
  } catch (std::string &Error) {
    freeState();       // Must destroy handles before deleting module!
    delete TheModule;
    TheModule = 0;
    throw;
  }
}
