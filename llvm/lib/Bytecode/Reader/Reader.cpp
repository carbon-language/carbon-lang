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
#include "llvm/Module.h"
#include "llvm/Constants.h"
#include "llvm/iPHINode.h"
#include "llvm/iOther.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <algorithm>

bool BytecodeParser::getTypeSlot(const Type *Ty, unsigned &Slot) {
  if (Ty->isPrimitiveType()) {
    Slot = Ty->getPrimitiveID();
  } else {
    // Check the function level types first...
    TypeValuesListTy::iterator I = find(FunctionTypeValues.begin(),
					FunctionTypeValues.end(), Ty);
    if (I != FunctionTypeValues.end()) {
      Slot = FirstDerivedTyID+ModuleTypeValues.size()+
             (&*I - &FunctionTypeValues[0]);
    } else {
      I = find(ModuleTypeValues.begin(), ModuleTypeValues.end(), Ty);
      if (I == ModuleTypeValues.end()) return true;   // Didn't find type!
      Slot = FirstDerivedTyID + (&*I - &ModuleTypeValues[0]);
    }
  }
  //cerr << "getTypeSlot '" << Ty->getName() << "' = " << Slot << "\n";
  return false;
}

const Type *BytecodeParser::getType(unsigned ID) {
  if (ID < Type::NumPrimitiveIDs) {
    const Type *T = Type::getPrimitiveType((Type::PrimitiveID)ID);
    if (T) return T;
  }
  
  //cerr << "Looking up Type ID: " << ID << "\n";
  const Value *V = getValue(Type::TypeTy, ID, false);
  return cast_or_null<Type>(V);
}

int BytecodeParser::insertValue(Value *Val, ValueTable &ValueTab) {
  assert((!HasImplicitZeroInitializer || !isa<Constant>(Val) ||
          Val->getType()->isPrimitiveType() ||
          !cast<Constant>(Val)->isNullValue()) &&
         "Cannot read null values from bytecode!");
  unsigned type;
  if (getTypeSlot(Val->getType(), type)) return -1;
  assert(type != Type::TypeTyID && "Types should never be insertValue'd!");
 
  if (ValueTab.size() <= type) {
    unsigned OldSize = ValueTab.size();
    ValueTab.resize(type+1);
    while (OldSize != type+1)
      ValueTab[OldSize++] = new ValueList();
  }

  //cerr << "insertValue Values[" << type << "][" << ValueTab[type].size() 
  //     << "] = " << Val << "\n";
  ValueTab[type]->push_back(Val);

  bool HasOffset = HasImplicitZeroInitializer &&
                       !Val->getType()->isPrimitiveType();

  return ValueTab[type]->size()-1 + HasOffset;
}


void BytecodeParser::setValueTo(ValueTable &ValueTab, unsigned Slot,
                                Value *Val) {
  assert(&ValueTab == &ModuleValues && "Can only setValueTo on Module values!");
  unsigned type;
  if (getTypeSlot(Val->getType(), type))
    assert(0 && "getTypeSlot failed!");
  
  assert((!HasImplicitZeroInitializer || Slot != 0) &&
         "Cannot change zero init");
  assert(type < ValueTab.size() && Slot <= ValueTab[type]->size());
  ValueTab[type]->setOperand(Slot-HasImplicitZeroInitializer, Val);
}

Value *BytecodeParser::getValue(const Type *Ty, unsigned oNum, bool Create) {
  unsigned Num = oNum;
  unsigned type;   // The type plane it lives in...

  if (getTypeSlot(Ty, type)) return 0;

  if (type == Type::TypeTyID) {  // The 'type' plane has implicit values
    assert(Create == false);
    if (Num < Type::NumPrimitiveIDs) {
      const Type *T = Type::getPrimitiveType((Type::PrimitiveID)Num);
      if (T) return (Value*)T;   // Asked for a primitive type...
    }

    // Otherwise, derived types need offset...
    Num -= FirstDerivedTyID;

    // Is it a module level type?
    if (Num < ModuleTypeValues.size())
      return (Value*)ModuleTypeValues[Num].get();

    // Nope, is it a function level type?
    Num -= ModuleTypeValues.size();
    if (Num < FunctionTypeValues.size())
      return (Value*)FunctionTypeValues[Num].get();

    return 0;
  }

  if (HasImplicitZeroInitializer && type >= FirstDerivedTyID) {
    if (Num == 0)
      return Constant::getNullValue(Ty);
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

  Value *d = 0;
  switch (Ty->getPrimitiveID()) {
  case Type::LabelTyID:
    d = new BBPHolder(Ty, oNum);
    break;
  default:
    d = new ValPHolder(Ty, oNum);
    break;
  }

  assert(d != 0 && "How did we not make something?");
  if (insertValue(d, LateResolveValues) == -1) return 0;
  return d;
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


bool BytecodeParser::postResolveValues(ValueTable &ValTab) {
  bool Error = false;
  while (!ValTab.empty()) {
    ValueList &DL = *ValTab.back();
    ValTab.pop_back();    

    while (!DL.empty()) {
      Value *D = DL.back();
      unsigned IDNumber = getValueIDNumberFromPlaceHolder(D);
      DL.pop_back();

      Value *NewDef = getValue(D->getType(), IDNumber, false);
      if (NewDef == 0) {
	Error = true;  // Unresolved thinger
	std::cerr << "Unresolvable reference found: <"
                  << *D->getType() << ">:" << IDNumber <<"!\n";
      } else {
	// Fixup all of the uses of this placeholder def...
        D->replaceAllUsesWith(NewDef);

        // Now that all the uses are gone, delete the placeholder...
        // If we couldn't find a def (error case), then leak a little
	delete D;  // memory, 'cause otherwise we can't remove all uses!
      }
    }
    delete &DL;
  }

  return Error;
}

bool BytecodeParser::ParseBasicBlock(const uchar *&Buf, const uchar *EndBuf, 
				     BasicBlock *&BB) {
  BB = new BasicBlock();

  while (Buf < EndBuf) {
    Instruction *Inst;
    if (ParseInstruction(Buf, EndBuf, Inst, /*HACK*/BB)) {
      delete BB;
      return true;
    }

    if (Inst == 0) { delete BB; return true; }
    if (insertValue(Inst, Values) == -1) { delete BB; return true; }

    BB->getInstList().push_back(Inst);

    BCR_TRACE(4, Inst);
  }

  return false;
}

bool BytecodeParser::ParseSymbolTable(const uchar *&Buf, const uchar *EndBuf,
				      SymbolTable *ST) {
  while (Buf < EndBuf) {
    // Symtab block header: [num entries][type id number]
    unsigned NumEntries, Typ;
    if (read_vbr(Buf, EndBuf, NumEntries) ||
        read_vbr(Buf, EndBuf, Typ)) return true;
    const Type *Ty = getType(Typ);
    if (Ty == 0) return true;

    BCR_TRACE(3, "Plane Type: '" << Ty << "' with " << NumEntries <<
	      " entries\n");

    for (unsigned i = 0; i < NumEntries; ++i) {
      // Symtab entry: [def slot #][name]
      unsigned slot;
      if (read_vbr(Buf, EndBuf, slot)) return true;
      std::string Name;
      if (read(Buf, EndBuf, Name, false))  // Not aligned...
	return true;

      Value *V = getValue(Ty, slot, false); // Find mapping...
      if (V == 0) {
	BCR_TRACE(3, "FAILED LOOKUP: Slot #" << slot << "\n");
	return true;
      }
      BCR_TRACE(4, "Map: '" << Name << "' to #" << slot << ":" << *V;
		if (!isa<Instruction>(V)) std::cerr << "\n");

      V->setName(Name, ST);
    }
  }

  if (Buf > EndBuf) return true;
  return false;
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


bool BytecodeParser::ParseFunction(const uchar *&Buf, const uchar *EndBuf) {
  // Clear out the local values table...
  if (FunctionSignatureList.empty()) {
    Error = "Function found, but FunctionSignatureList empty!";
    return true;  // Unexpected function!
  }

  unsigned isInternal;
  if (read_vbr(Buf, EndBuf, isInternal)) return true;

  Function *F = FunctionSignatureList.back().first;
  unsigned FunctionSlot = FunctionSignatureList.back().second;
  FunctionSignatureList.pop_back();
  F->setInternalLinkage(isInternal != 0);

  const FunctionType::ParamTypes &Params =F->getFunctionType()->getParamTypes();
  Function::aiterator AI = F->abegin();
  for (FunctionType::ParamTypes::const_iterator It = Params.begin();
       It != Params.end(); ++It, ++AI) {
    if (insertValue(AI, Values) == -1) {
      Error = "Error reading function arguments!\n";
      return true; 
    }
  }

  while (Buf < EndBuf) {
    unsigned Type, Size;
    const unsigned char *OldBuf = Buf;
    if (readBlock(Buf, EndBuf, Type, Size)) {
      Error = "Error reading Function level block!";
      return true; 
    }

    switch (Type) {
    case BytecodeFormat::ConstantPool:
      BCR_TRACE(2, "BLOCK BytecodeFormat::ConstantPool: {\n");
      if (ParseConstantPool(Buf, Buf+Size, Values, FunctionTypeValues))
	return true;
      break;

    case BytecodeFormat::BasicBlock: {
      BCR_TRACE(2, "BLOCK BytecodeFormat::BasicBlock: {\n");
      BasicBlock *BB;
      if (ParseBasicBlock(Buf, Buf+Size, BB) ||
	  insertValue(BB, Values) == -1)
	return true;                // Parse error... :(

      F->getBasicBlockList().push_back(BB);
      break;
    }

    case BytecodeFormat::SymbolTable:
      BCR_TRACE(2, "BLOCK BytecodeFormat::SymbolTable: {\n");
      if (ParseSymbolTable(Buf, Buf+Size, &F->getSymbolTable()))
	return true;
      break;

    default:
      BCR_TRACE(2, "BLOCK <unknown>:ignored! {\n");
      Buf += Size;
      if (OldBuf > Buf) return true; // Wrap around!
      break;
    }
    BCR_TRACE(2, "} end block\n");

    if (align32(Buf, EndBuf)) {
      Error = "Error aligning Function level block!";
      return true;   // Malformed bc file, read past end of block.
    }
  }

  if (postResolveValues(LateResolveValues)) {
    Error = "Error resolving function values!";
    return true;     // Unresolvable references!
  }

  ResolveReferencesToValue(F, FunctionSlot);

  // Clear out function level types...
  FunctionTypeValues.clear();

  freeTable(Values);
  return false;
}

bool BytecodeParser::ParseModuleGlobalInfo(const uchar *&Buf, const uchar *End){
  if (!FunctionSignatureList.empty()) {
    Error = "Two ModuleGlobalInfo packets found!";
    return true;  // Two ModuleGlobal blocks?
  }

  // Read global variables...
  unsigned VarType;
  if (read_vbr(Buf, End, VarType)) return true;
  while (VarType != Type::VoidTyID) { // List is terminated by Void
    // VarType Fields: bit0 = isConstant, bit1 = hasInitializer,
    // bit2 = isInternal, bit3+ = slot#
    const Type *Ty = getType(VarType >> 3);
    if (!Ty || !isa<PointerType>(Ty)) { 
      Error = "Global not pointer type!  Ty = " + Ty->getDescription();
      return true; 
    }

    const Type *ElTy = cast<PointerType>(Ty)->getElementType();

    // Create the global variable...
    GlobalVariable *GV = new GlobalVariable(ElTy, VarType & 1, VarType & 4,
                                            0, "", TheModule);
    int DestSlot = insertValue(GV, ModuleValues);
    if (DestSlot == -1) return true;
    BCR_TRACE(2, "Global Variable of type: " << *Ty << "\n");
    ResolveReferencesToValue(GV, (unsigned)DestSlot);

    if (VarType & 2) { // Does it have an initalizer?
      unsigned InitSlot;
      if (read_vbr(Buf, End, InitSlot)) return true;
      GlobalInits.push_back(std::make_pair(GV, InitSlot));
    }
    if (read_vbr(Buf, End, VarType)) return true;
  }

  // Read the function objects for all of the functions that are coming
  unsigned FnSignature;
  if (read_vbr(Buf, End, FnSignature)) return true;
  while (FnSignature != Type::VoidTyID) { // List is terminated by Void
    const Type *Ty = getType(FnSignature);
    if (!Ty || !isa<PointerType>(Ty) ||
        !isa<FunctionType>(cast<PointerType>(Ty)->getElementType())) { 
      Error = "Function not ptr to func type!  Ty = " + Ty->getDescription();
      return true; 
    }

    // We create functions by passing the underlying FunctionType to create...
    Ty = cast<PointerType>(Ty)->getElementType();

    // When the ModuleGlobalInfo section is read, we load the type of each
    // function and the 'ModuleValues' slot that it lands in.  We then load a
    // placeholder into its slot to reserve it.  When the function is loaded,
    // this placeholder is replaced.

    // Insert the placeholder...
    Function *Func = new Function(cast<FunctionType>(Ty), false, "", TheModule);
    int DestSlot = insertValue(Func, ModuleValues);
    if (DestSlot == -1) return true;
    ResolveReferencesToValue(Func, (unsigned)DestSlot);

    // Keep track of this information in a list that is emptied as functions are
    // loaded...
    //
    FunctionSignatureList.push_back(std::make_pair(Func, DestSlot));

    if (read_vbr(Buf, End, FnSignature)) return true;
    BCR_TRACE(2, "Function of type: " << Ty << "\n");
  }

  if (align32(Buf, End)) return true;

  // Now that the function signature list is set up, reverse it so that we can 
  // remove elements efficiently from the back of the vector.
  std::reverse(FunctionSignatureList.begin(), FunctionSignatureList.end());

  // This is for future proofing... in the future extra fields may be added that
  // we don't understand, so we transparently ignore them.
  //
  Buf = End;
  return false;
}

bool BytecodeParser::ParseVersionInfo(const uchar *&Buf, const uchar *EndBuf) {
  unsigned Version;
  if (read_vbr(Buf, EndBuf, Version)) return true;

  // Unpack version number: low four bits are for flags, top bits = version
  isBigEndian     = Version & 1;
  hasLongPointers = Version & 2;
  RevisionNum     = Version >> 4;
  HasImplicitZeroInitializer = true;

  switch (RevisionNum) {
  case 0:                  // Initial revision
    // Version #0 didn't have any of the flags stored correctly, and in fact as
    // only valid with a 14 in the flags values.  Also, it does not support
    // encoding zero initializers for arrays compactly.
    //
    if (Version != 14) return true;  // Unknown revision 0 flags?
    FirstDerivedTyID = 14;
    HasImplicitZeroInitializer = false;
    isBigEndian = hasLongPointers = true;
    break;
  case 1:
    // Version #1 has two bit fields: isBigEndian and hasLongPointers
    FirstDerivedTyID = 14;
    break;
  default:
    Error = "Unknown bytecode version number!";
    return true;
  }

  BCR_TRACE(1, "Bytecode Rev = " << (unsigned)RevisionNum << "\n");
  BCR_TRACE(1, "BigEndian/LongPointers = " << isBigEndian << ","
               << hasLongPointers << "\n");
  BCR_TRACE(1, "HasImplicitZeroInit = " << HasImplicitZeroInitializer << "\n");
  return false;
}

bool BytecodeParser::ParseModule(const uchar *Buf, const uchar *EndBuf) {
  unsigned Type, Size;
  if (readBlock(Buf, EndBuf, Type, Size)) return true;
  if (Type != BytecodeFormat::Module || Buf+Size != EndBuf) {
    Error = "Expected Module packet!";
    return true;                      // Hrm, not a class?
  }

  BCR_TRACE(0, "BLOCK BytecodeFormat::Module: {\n");
  FunctionSignatureList.clear();                 // Just in case...

  // Read into instance variables...
  if (ParseVersionInfo(Buf, EndBuf)) return true;
  if (align32(Buf, EndBuf)) return true;

  while (Buf < EndBuf) {
    const unsigned char *OldBuf = Buf;
    if (readBlock(Buf, EndBuf, Type, Size)) return true;
    switch (Type) {
    case BytecodeFormat::GlobalTypePlane:
      BCR_TRACE(1, "BLOCK BytecodeFormat::GlobalTypePlane: {\n");
      if (ParseGlobalTypes(Buf, Buf+Size)) return true;
      break;

    case BytecodeFormat::ModuleGlobalInfo:
      BCR_TRACE(1, "BLOCK BytecodeFormat::ModuleGlobalInfo: {\n");
      if (ParseModuleGlobalInfo(Buf, Buf+Size)) return true;
      break;

    case BytecodeFormat::ConstantPool:
      BCR_TRACE(1, "BLOCK BytecodeFormat::ConstantPool: {\n");
      if (ParseConstantPool(Buf, Buf+Size, ModuleValues, ModuleTypeValues))
	return true;
      break;

    case BytecodeFormat::Function: {
      BCR_TRACE(1, "BLOCK BytecodeFormat::Function: {\n");
      if (ParseFunction(Buf, Buf+Size))
        return true;  // Error parsing function
      break;
    }

    case BytecodeFormat::SymbolTable:
      BCR_TRACE(1, "BLOCK BytecodeFormat::SymbolTable: {\n");
      if (ParseSymbolTable(Buf, Buf+Size, &TheModule->getSymbolTable()))
        return true;
      break;

    default:
      Error = "Expected Module Block!";
      Buf += Size;
      if (OldBuf > Buf) return true; // Wrap around!
      break;
    }
    BCR_TRACE(1, "} end block\n");
    if (align32(Buf, EndBuf)) return true;
  }

  // After the module constant pool has been read, we can safely initialize
  // global variables...
  while (!GlobalInits.empty()) {
    GlobalVariable *GV = GlobalInits.back().first;
    unsigned Slot = GlobalInits.back().second;
    GlobalInits.pop_back();

    // Look up the initializer value...
    if (Value *V = getValue(GV->getType()->getElementType(), Slot, false)) {
      if (GV->hasInitializer()) return true;
      GV->setInitializer(cast<Constant>(V));
    } else
      return true;
  }

  if (!FunctionSignatureList.empty()) {     // Expected more functions!
    Error = "Function expected, but bytecode stream at end!";
    return true;
  }

  BCR_TRACE(0, "} end block\n\n");
  return false;
}

static inline Module *Error(std::string *ErrorStr, const char *Message) {
  if (ErrorStr) *ErrorStr = Message;
  return 0;
}

Module *BytecodeParser::ParseBytecode(const uchar *Buf, const uchar *EndBuf) {
  unsigned Sig;
  // Read and check signature...
  if (read(Buf, EndBuf, Sig) ||
      Sig != ('l' | ('l' << 8) | ('v' << 16) | 'm' << 24))
    return ::Error(&Error, "Invalid bytecode signature!");

  TheModule = new Module();
  if (ParseModule(Buf, EndBuf)) {
    delete TheModule;
    TheModule = 0;
  }
  return TheModule;
}


Module *ParseBytecodeBuffer(const unsigned char *Buffer, unsigned Length,
                            std::string *ErrorStr) {
  BytecodeParser Parser;
  Module *R = Parser.ParseBytecode(Buffer, Buffer+Length);
  if (ErrorStr) *ErrorStr = Parser.getError();
  return R;
}


/// FDHandle - Simple handle class to make sure a file descriptor gets closed
/// when the object is destroyed.
class FDHandle {
  int FD;
public:
  FDHandle(int fd) : FD(fd) {}
  operator int() const { return FD; }
  ~FDHandle() {
    if (FD != -1) close(FD);
  }
};

// Parse and return a class file...
//
Module *ParseBytecodeFile(const std::string &Filename, std::string *ErrorStr) {
  Module *Result = 0;

  if (Filename != std::string("-")) {        // Read from a file...
    FDHandle FD = open(Filename.c_str(), O_RDONLY);
    if (FD == -1)
      return Error(ErrorStr, "Error opening file!");

    // Stat the file to get its length...
    struct stat StatBuf;
    if (fstat(FD, &StatBuf) == -1 || StatBuf.st_size == 0)
      return Error(ErrorStr, "Error stat'ing file!");

    // mmap in the file all at once...
    int Length = StatBuf.st_size;
    unsigned char *Buffer = (unsigned char*)mmap(0, Length, PROT_READ, 
                                                 MAP_PRIVATE, FD, 0);
    if (Buffer == (unsigned char*)MAP_FAILED)
      return Error(ErrorStr, "Error mmapping file!");

    // Parse the bytecode we mmapped in
    Result = ParseBytecodeBuffer(Buffer, Length, ErrorStr);

    // Unmmap the bytecode...
    munmap((char*)Buffer, Length);
  } else {                              // Read from stdin
    int BlockSize;
    uchar Buffer[4096*4];
    std::vector<unsigned char> FileData;

    // Read in all of the data from stdin, we cannot mmap stdin...
    while ((BlockSize = read(0 /*stdin*/, Buffer, 4096*4))) {
      if (BlockSize == -1)
        return Error(ErrorStr, "Error reading from stdin!");

      FileData.insert(FileData.end(), Buffer, Buffer+BlockSize);
    }

    if (FileData.empty())
      return Error(ErrorStr, "Standard Input empty!");

#define ALIGN_PTRS 0
#if ALIGN_PTRS
    uchar *Buf = (uchar*)mmap(0, FileData.size(), PROT_READ|PROT_WRITE, 
			      MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    assert((Buf != (uchar*)-1) && "mmap returned error!");
    memcpy(Buf, &FileData[0], FileData.size());
#else
    unsigned char *Buf = &FileData[0];
#endif

    Result = ParseBytecodeBuffer(Buf, FileData.size(), ErrorStr);

#if ALIGN_PTRS
    munmap((char*)Buf, FileData.size());   // Free mmap'd data area
#endif
  }

  return Result;
}
