//===- Reader.cpp - Code to read bytecode files ---------------------------===//
//
// This library implements the functionality defined in llvm/Bytecode/Reader.h
//
// Note that this library should be as fast as possible, reentrant, and 
// threadsafe!!
//
// TODO: Return error messages to caller instead of printing them out directly.
// TODO: Allow passing in an option to ignore the symbol table
//
//===----------------------------------------------------------------------===//

#include "ReaderInternals.h"
#include "llvm/Bytecode/Reader.h"
#include "llvm/Bytecode/Format.h"
#include "llvm/Constants.h"
#include "llvm/iPHINode.h"
#include "llvm/iOther.h"
#include "llvm/Module.h"
#include "Support/StringExtras.h"
#include "Config/unistd.h"
#include "Config/sys/mman.h"
#include "Config/sys/stat.h"
#include "Config/sys/types.h"
#include <algorithm>
#include <memory>

static inline void ALIGN32(const unsigned char *&begin,
                           const unsigned char *end) {
  if (align32(begin, end))
    throw std::string("Alignment error in buffer: read past end of block.");
}

unsigned BytecodeParser::getTypeSlot(const Type *Ty) {
  if (Ty->isPrimitiveType())
    return Ty->getPrimitiveID();

  // Check the function level types first...
  TypeValuesListTy::iterator I = find(FunctionTypeValues.begin(),
                                      FunctionTypeValues.end(), Ty);
  if (I != FunctionTypeValues.end())
    return FirstDerivedTyID + ModuleTypeValues.size() +
             (&*I - &FunctionTypeValues[0]);

  I = find(ModuleTypeValues.begin(), ModuleTypeValues.end(), Ty);
  if (I == ModuleTypeValues.end())
    throw std::string("Didn't find type in ModuleTypeValues.");
  return FirstDerivedTyID + (&*I - &ModuleTypeValues[0]);
}

const Type *BytecodeParser::getType(unsigned ID) {
  if (ID < Type::NumPrimitiveIDs) {
    const Type *T = Type::getPrimitiveType((Type::PrimitiveID)ID);
    if (T) return T;
  }
  
  //cerr << "Looking up Type ID: " << ID << "\n";

  if (ID < Type::NumPrimitiveIDs) {
    const Type *T = Type::getPrimitiveType((Type::PrimitiveID)ID);
    if (T) return T;   // Asked for a primitive type...
  }

  // Otherwise, derived types need offset...
  ID -= FirstDerivedTyID;

  // Is it a module-level type?
  if (ID < ModuleTypeValues.size())
    return ModuleTypeValues[ID].get();

  // Nope, is it a function-level type?
  ID -= ModuleTypeValues.size();
  if (ID < FunctionTypeValues.size())
    return FunctionTypeValues[ID].get();

  return 0;
}

int BytecodeParser::insertValue(Value *Val, ValueTable &ValueTab) {
  assert((!HasImplicitZeroInitializer || !isa<Constant>(Val) ||
          Val->getType()->isPrimitiveType() ||
          !cast<Constant>(Val)->isNullValue()) &&
         "Cannot read null values from bytecode!");
  unsigned type = getTypeSlot(Val->getType());
  assert(type != Type::TypeTyID && "Types should never be insertValue'd!");
 
  if (ValueTab.size() <= type) {
    unsigned OldSize = ValueTab.size();
    ValueTab.resize(type+1);
    while (OldSize != type+1)
      ValueTab[OldSize++] = new ValueList();
  }

  //cerr << "insertValue Values[" << type << "][" << ValueTab[type].size() 
  //   << "] = " << Val << "\n";
  ValueTab[type]->push_back(Val);

  bool HasOffset = HasImplicitZeroInitializer &&
    !Val->getType()->isPrimitiveType();

  return ValueTab[type]->size()-1 + HasOffset;
}


Value *BytecodeParser::getValue(const Type *Ty, unsigned oNum, bool Create) {
  return getValue(getTypeSlot(Ty), oNum, Create);
}

Value *BytecodeParser::getValue(unsigned type, unsigned oNum, bool Create) {
  assert(type != Type::TypeTyID && "getValue() cannot get types!");
  assert(type != Type::LabelTyID && "getValue() cannot get blocks!");
  unsigned Num = oNum;

  if (HasImplicitZeroInitializer && type >= FirstDerivedTyID) {
    if (Num == 0)
      return Constant::getNullValue(getType(type));
    --Num;
  }

  if (type < ModuleValues.size()) {
    if (Num < ModuleValues[type]->size())
      return ModuleValues[type]->getOperand(Num);
    Num -= ModuleValues[type]->size();
  }

  if (Values.size() > type && Values[type]->size() > Num)
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
Constant *BytecodeParser::getConstantValue(const Type *Ty, unsigned Slot) {
  if (Value *V = getValue(Ty, Slot, false))
    return dyn_cast<Constant>(V);      // If we already have the value parsed...

  std::pair<const Type*, unsigned> Key(Ty, Slot);
  GlobalRefsType::iterator I = GlobalRefs.lower_bound(Key);

  if (I != GlobalRefs.end() && I->first == Key) {
    BCR_TRACE(5, "Previous forward ref found!\n");
    return cast<Constant>(I->second);
  } else {
    // Create a placeholder for the constant reference and
    // keep track of the fact that we have a forward ref to recycle it
    BCR_TRACE(5, "Creating new forward ref to a constant!\n");
    Constant *C = new ConstPHolder(Ty, Slot);
    
    // Keep track of the fact that we have a forward ref to recycle it
    GlobalRefs.insert(I, std::make_pair(Key, C));
    return C;
  }
}


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

  while (Buf < EndBuf) {
    Instruction *Inst = ParseInstruction(Buf, EndBuf);
    if (insertValue(Inst, Values) == -1) { 
      throw std::string("Could not insert value.");
    }

    BB->getInstList().push_back(Inst);
    BCR_TRACE(4, Inst);
  }

  return BB;
}

void BytecodeParser::ParseSymbolTable(const unsigned char *&Buf,
                                      const unsigned char *EndBuf,
                                      SymbolTable *ST,
                                      Function *CurrentFunction) {
  while (Buf < EndBuf) {
    // Symtab block header: [num entries][type id number]
    unsigned NumEntries, Typ;
    if (read_vbr(Buf, EndBuf, NumEntries) ||
        read_vbr(Buf, EndBuf, Typ)) throw Error_readvbr;
    const Type *Ty = getType(Typ);
    if (Ty == 0) throw std::string("Invalid type read in symbol table.");

    BCR_TRACE(3, "Plane Type: '" << Ty << "' with " << NumEntries <<
                 " entries\n");

    for (unsigned i = 0; i < NumEntries; ++i) {
      // Symtab entry: [def slot #][name]
      unsigned slot;
      if (read_vbr(Buf, EndBuf, slot)) throw Error_readvbr;
      std::string Name;
      if (read(Buf, EndBuf, Name, false))  // Not aligned...
        throw std::string("Buffer not aligned.");

      Value *V = 0;
      if (Typ == Type::TypeTyID)
        V = (Value*)getType(slot);
      else if (Typ == Type::LabelTyID) {
        if (CurrentFunction) {
          // FIXME: THIS IS N^2!!!
          Function::iterator BlockIterator = CurrentFunction->begin();
          std::advance(BlockIterator, slot);
          V = BlockIterator;
        }
      } else
        V = getValue(Typ, slot, false); // Find mapping...
      if (V == 0) throw std::string("Failed value look-up.");
      BCR_TRACE(4, "Map: '" << Name << "' to #" << slot << ":" << *V;
                if (!isa<Instruction>(V)) std::cerr << "\n");

      V->setName(Name, ST);
    }
  }

  if (Buf > EndBuf) throw std::string("Tried to read past end of buffer.");
}

void BytecodeParser::ResolveReferencesToValue(Value *NewV, unsigned Slot) {
  GlobalRefsType::iterator I = GlobalRefs.find(std::make_pair(NewV->getType(),
                                                              Slot));
  if (I == GlobalRefs.end()) return;   // Never forward referenced?

  BCR_TRACE(3, "Mutating forward refs!\n");
  Value *VPH = I->second;   // Get the placeholder...

  VPH->replaceAllUsesWith(NewV);

  // If this is a global variable being resolved, remove the placeholder from
  // the module...
  if (GlobalValue* GVal = dyn_cast<GlobalValue>(NewV))
    GVal->getParent()->getGlobalList().remove(cast<GlobalVariable>(VPH));

  delete VPH;                         // Delete the old placeholder
  GlobalRefs.erase(I);                // Remove the map entry for it
}

void BytecodeParser::ParseFunction(const unsigned char *&Buf,
                                   const unsigned char *EndBuf) {
  if (FunctionSignatureList.empty())
    throw std::string("FunctionSignatureList empty!");

  Function *F = FunctionSignatureList.back().first;
  unsigned FunctionSlot = FunctionSignatureList.back().second;
  FunctionSignatureList.pop_back();

  // Save the information for future reading of the function
  LazyFunctionInfo *LFI = new LazyFunctionInfo();
  LFI->Buf = Buf; LFI->EndBuf = EndBuf; LFI->FunctionSlot = FunctionSlot;
  LazyFunctionLoadMap[F] = LFI;
  // Pretend we've `parsed' this function
  Buf = EndBuf;
}

void BytecodeParser::materializeFunction(Function* F) {
  // Find {start, end} pointers and slot in the map. If not there, we're done.
  std::map<Function*, LazyFunctionInfo*>::iterator Fi =
    LazyFunctionLoadMap.find(F);
  if (Fi == LazyFunctionLoadMap.end()) return;
  
  LazyFunctionInfo *LFI = Fi->second;
  const unsigned char *Buf = LFI->Buf;
  const unsigned char *EndBuf = LFI->EndBuf;
  unsigned FunctionSlot = LFI->FunctionSlot;
  LazyFunctionLoadMap.erase(Fi);
  delete LFI;

  GlobalValue::LinkageTypes Linkage = GlobalValue::ExternalLinkage;

  if (!hasInternalMarkerOnly) {
    unsigned LinkageType;
    if (read_vbr(Buf, EndBuf, LinkageType)) 
      throw std::string("ParseFunction: Error reading from buffer.");
    if (LinkageType & ~0x3) 
      throw std::string("Invalid linkage type for Function.");
    Linkage = (GlobalValue::LinkageTypes)LinkageType;
  } else {
    // We used to only support two linkage models: internal and external
    unsigned isInternal;
    if (read_vbr(Buf, EndBuf, isInternal)) 
      throw std::string("ParseFunction: Error reading from buffer.");
    if (isInternal) Linkage = GlobalValue::InternalLinkage;
  }

  F->setLinkage(Linkage);

  const FunctionType::ParamTypes &Params =F->getFunctionType()->getParamTypes();
  Function::aiterator AI = F->abegin();
  for (FunctionType::ParamTypes::const_iterator It = Params.begin();
       It != Params.end(); ++It, ++AI) {
    if (insertValue(AI, Values) == -1)
      throw std::string("Error reading function arguments!");
  }

  // Keep track of how many basic blocks we have read in...
  unsigned BlockNum = 0;

  while (Buf < EndBuf) {
    unsigned Type, Size;
    const unsigned char *OldBuf = Buf;
    readBlock(Buf, EndBuf, Type, Size);

    switch (Type) {
    case BytecodeFormat::ConstantPool: {
      BCR_TRACE(2, "BLOCK BytecodeFormat::ConstantPool: {\n");
      ParseConstantPool(Buf, Buf+Size, Values, FunctionTypeValues);
      break;
    }

    case BytecodeFormat::BasicBlock: {
      BCR_TRACE(2, "BLOCK BytecodeFormat::BasicBlock: {\n");
      BasicBlock *BB = ParseBasicBlock(Buf, Buf+Size, BlockNum++);
      F->getBasicBlockList().push_back(BB);
      break;
    }

    case BytecodeFormat::SymbolTable: {
      BCR_TRACE(2, "BLOCK BytecodeFormat::SymbolTable: {\n");
      ParseSymbolTable(Buf, Buf+Size, &F->getSymbolTable(), F);
      break;
    }

    default:
      BCR_TRACE(2, "BLOCK <unknown>:ignored! {\n");
      Buf += Size;
      if (OldBuf > Buf) 
        throw std::string("Wrapped around reading bytecode.");
      break;
    }
    BCR_TRACE(2, "} end block\n");

    // Malformed bc file if read past end of block.
    ALIGN32(Buf, EndBuf);
  }

  // Make sure there were no references to non-existant basic blocks.
  if (BlockNum != ParsedBasicBlocks.size())
    throw std::string("Illegal basic block operand reference");
  ParsedBasicBlocks.clear();


  // Resolve forward references
  while (!ForwardReferences.empty()) {
    std::map<std::pair<unsigned,unsigned>, Value*>::iterator I = ForwardReferences.begin();
    unsigned type = I->first.first;
    unsigned Slot = I->first.second;
    Value *PlaceHolder = I->second;
    ForwardReferences.erase(I);

    Value *NewVal = getValue(type, Slot, false);
    if (NewVal == 0)
      throw std::string("Unresolvable reference found: <" +
                        PlaceHolder->getType()->getDescription() + ">:" + 
                        utostr(Slot) + ".");

    // Fixup all of the uses of this placeholder def...
    PlaceHolder->replaceAllUsesWith(NewVal);
      
    // Now that all the uses are gone, delete the placeholder...
    // If we couldn't find a def (error case), then leak a little
    // memory, because otherwise we can't remove all uses!
    delete PlaceHolder;
  }

  // Clear out function-level types...
  FunctionTypeValues.clear();

  freeTable(Values);
}

void BytecodeParser::ParseModuleGlobalInfo(const unsigned char *&Buf,
                                           const unsigned char *End) {
  if (!FunctionSignatureList.empty())
    throw std::string("Two ModuleGlobalInfo packets found!");

  // Read global variables...
  unsigned VarType;
  if (read_vbr(Buf, End, VarType)) throw Error_readvbr;
  while (VarType != Type::VoidTyID) { // List is terminated by Void
    unsigned SlotNo;
    GlobalValue::LinkageTypes Linkage;

    if (!hasInternalMarkerOnly) {
      // VarType Fields: bit0 = isConstant, bit1 = hasInitializer,
      // bit2,3 = Linkage, bit4+ = slot#
      SlotNo = VarType >> 4;
      Linkage = (GlobalValue::LinkageTypes)((VarType >> 2) & 3);
    } else {
      // VarType Fields: bit0 = isConstant, bit1 = hasInitializer,
      // bit2 = isInternal, bit3+ = slot#
      SlotNo = VarType >> 3;
      Linkage = (VarType & 4) ? GlobalValue::InternalLinkage :
        GlobalValue::ExternalLinkage;
    }

    const Type *Ty = getType(SlotNo);
    if (!Ty || !isa<PointerType>(Ty))
      throw std::string("Global not pointer type!  Ty = " + 
                        Ty->getDescription());

    const Type *ElTy = cast<PointerType>(Ty)->getElementType();

    // Create the global variable...
    GlobalVariable *GV = new GlobalVariable(ElTy, VarType & 1, Linkage,
                                            0, "", TheModule);
    int DestSlot = insertValue(GV, ModuleValues);
    if (DestSlot == -1) throw Error_DestSlot;
    BCR_TRACE(2, "Global Variable of type: " << *Ty << "\n");
    ResolveReferencesToValue(GV, (unsigned)DestSlot);

    if (VarType & 2) { // Does it have an initializer?
      unsigned InitSlot;
      if (read_vbr(Buf, End, InitSlot)) throw Error_readvbr;
      GlobalInits.push_back(std::make_pair(GV, InitSlot));
    }
    if (read_vbr(Buf, End, VarType)) throw Error_readvbr;
  }

  // Read the function objects for all of the functions that are coming
  unsigned FnSignature;
  if (read_vbr(Buf, End, FnSignature)) throw Error_readvbr;
  while (FnSignature != Type::VoidTyID) { // List is terminated by Void
    const Type *Ty = getType(FnSignature);
    if (!Ty || !isa<PointerType>(Ty) ||
        !isa<FunctionType>(cast<PointerType>(Ty)->getElementType())) { 
      throw std::string("Function not ptr to func type!  Ty = " +
                        Ty->getDescription());
    }

    // We create functions by passing the underlying FunctionType to create...
    Ty = cast<PointerType>(Ty)->getElementType();

    // When the ModuleGlobalInfo section is read, we load the type of each
    // function and the 'ModuleValues' slot that it lands in.  We then load a
    // placeholder into its slot to reserve it.  When the function is loaded,
    // this placeholder is replaced.

    // Insert the placeholder...
    Function *Func = new Function(cast<FunctionType>(Ty),
                                  GlobalValue::InternalLinkage, "", TheModule);
    int DestSlot = insertValue(Func, ModuleValues);
    if (DestSlot == -1) throw Error_DestSlot;
    ResolveReferencesToValue(Func, (unsigned)DestSlot);

    // Keep track of this information in a list that is emptied as functions are
    // loaded...
    //
    FunctionSignatureList.push_back(std::make_pair(Func, DestSlot));

    if (read_vbr(Buf, End, FnSignature)) throw Error_readvbr;
    BCR_TRACE(2, "Function of type: " << Ty << "\n");
  }

  ALIGN32(Buf, End);

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
  unsigned Version;
  if (read_vbr(Buf, EndBuf, Version)) throw Error_readvbr;

  // Unpack version number: low four bits are for flags, top bits = version
  Module::Endianness  Endianness;
  Module::PointerSize PointerSize;
  Endianness  = (Version & 1) ? Module::BigEndian : Module::LittleEndian;
  PointerSize = (Version & 2) ? Module::Pointer64 : Module::Pointer32;

  bool hasNoEndianness = Version & 4;
  bool hasNoPointerSize = Version & 8;
  
  RevisionNum = Version >> 4;

  // Default values for the current bytecode version
  HasImplicitZeroInitializer = true;
  hasInternalMarkerOnly = false;
  FirstDerivedTyID = 14;

  switch (RevisionNum) {
  case 0:                  // Initial revision
    // Version #0 didn't have any of the flags stored correctly, and in fact as
    // only valid with a 14 in the flags values.  Also, it does not support
    // encoding zero initializers for arrays compactly.
    //
    if (Version != 14) throw std::string("Unknown revision 0 flags?");
    HasImplicitZeroInitializer = false;
    Endianness  = Module::BigEndian;
    PointerSize = Module::Pointer64;
    hasInternalMarkerOnly = true;
    hasNoEndianness = hasNoPointerSize = false;
    break;
  case 1:
    // Version #1 has four bit fields: isBigEndian, hasLongPointers,
    // hasNoEndianness, and hasNoPointerSize.
    hasInternalMarkerOnly = true;
    break;
  case 2:
    // Version #2 added information about all 4 linkage types instead of just
    // having internal and external.
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
  BCR_TRACE(1, "HasImplicitZeroInit = " << HasImplicitZeroInitializer << "\n");
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
  ALIGN32(Buf, EndBuf);

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
    ALIGN32(Buf, EndBuf);
  }

  // After the module constant pool has been read, we can safely initialize
  // global variables...
  while (!GlobalInits.empty()) {
    GlobalVariable *GV = GlobalInits.back().first;
    unsigned Slot = GlobalInits.back().second;
    GlobalInits.pop_back();

    // Look up the initializer value...
    if (Value *V = getValue(GV->getType()->getElementType(), Slot, false)) {
      if (GV->hasInitializer()) 
        throw std::string("Global *already* has an initializer?!");
      GV->setInitializer(cast<Constant>(V));
    } else
      throw std::string("Cannot find initializer value.");
  }

  if (!FunctionSignatureList.empty())
    throw std::string("Function expected, but bytecode stream ended!");

  BCR_TRACE(0, "} end block\n\n");
}

void
BytecodeParser::ParseBytecode(const unsigned char *Buf, unsigned Length,
                              const std::string &ModuleID) {

  unsigned char *EndBuf = (unsigned char*)(Buf + Length);

  // Read and check signature...
  unsigned Sig;
  if (read(Buf, EndBuf, Sig) ||
      Sig != ('l' | ('l' << 8) | ('v' << 16) | ('m' << 24)))
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
