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
    // Check the method level types first...
    TypeValuesListTy::iterator I = find(MethodTypeValues.begin(),
					MethodTypeValues.end(), Ty);
    if (I != MethodTypeValues.end()) {
      Slot = FirstDerivedTyID+ModuleTypeValues.size()+
             (&*I - &MethodTypeValues[0]);
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

int BytecodeParser::insertValue(Value *Val, std::vector<ValueList> &ValueTab) {
  unsigned type;
  if (getTypeSlot(Val->getType(), type)) return -1;
  assert(type != Type::TypeTyID && "Types should never be insertValue'd!");
 
  if (ValueTab.size() <= type)
    ValueTab.resize(type+1, ValueList());

  //cerr << "insertValue Values[" << type << "][" << ValueTab[type].size() 
  //     << "] = " << Val << "\n";
  ValueTab[type].push_back(Val);

  return ValueTab[type].size()-1;
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

    // Nope, is it a method level type?
    Num -= ModuleTypeValues.size();
    if (Num < MethodTypeValues.size())
      return (Value*)MethodTypeValues[Num].get();

    return 0;
  }

  if (type < ModuleValues.size()) {
    if (Num < ModuleValues[type].size())
      return ModuleValues[type][Num];
    Num -= ModuleValues[type].size();
  }

  if (Values.size() > type && Values[type].size() > Num)
    return Values[type][Num];

  if (!Create) return 0;  // Do not create a placeholder?

  Value *d = 0;
  switch (Ty->getPrimitiveID()) {
  case Type::FunctionTyID:
    std::cerr << "Creating method pholder! : " << type << ":" << oNum << " " 
              << Ty->getName() << "\n";
    d = new FunctionPHolder(Ty, oNum);
    if (insertValue(d, LateResolveModuleValues) == -1) return 0;
    return d;
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

  GlobalRefsType::iterator I = GlobalRefs.find(std::make_pair(Ty, Slot));
  if (I != GlobalRefs.end()) {
    BCR_TRACE(5, "Previous forward ref found!\n");
    return cast<Constant>(I->second);
  } else {
    // Create a placeholder for the constant reference and
    // keep track of the fact that we have a forward ref to recycle it
    BCR_TRACE(5, "Creating new forward ref to a constant!\n");
    Constant *C = new ConstPHolder(Ty, Slot);
    
    // Keep track of the fact that we have a forward ref to recycle it
    GlobalRefs.insert(std::make_pair(std::make_pair(Ty, Slot), C));
    return C;
  }
}



bool BytecodeParser::postResolveValues(ValueTable &ValTab) {
  bool Error = false;
  for (unsigned ty = 0; ty < ValTab.size(); ++ty) {
    ValueList &DL = ValTab[ty];
    unsigned Size;
    while ((Size = DL.size())) {
      unsigned IDNumber = getValueIDNumberFromPlaceHolder(DL[Size-1]);

      Value *D = DL[Size-1];
      DL.pop_back();

      Value *NewDef = getValue(D->getType(), IDNumber, false);
      if (NewDef == 0) {
	Error = true;  // Unresolved thinger
	std::cerr << "Unresolvable reference found: <"
                  << D->getType()->getDescription() << ">:" << IDNumber <<"!\n";
      } else {
	// Fixup all of the uses of this placeholder def...
        D->replaceAllUsesWith(NewDef);

        // Now that all the uses are gone, delete the placeholder...
        // If we couldn't find a def (error case), then leak a little
	delete D;  // memory, 'cause otherwise we can't remove all uses!
      }
    }
  }

  return Error;
}

bool BytecodeParser::ParseBasicBlock(const uchar *&Buf, const uchar *EndBuf, 
				     BasicBlock *&BB) {
  BB = new BasicBlock();

  while (Buf < EndBuf) {
    Instruction *Inst;
    if (ParseInstruction(Buf, EndBuf, Inst,
                         /*HACK*/BB)) {
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

      Value *D = getValue(Ty, slot, false); // Find mapping...
      if (D == 0) {
	BCR_TRACE(3, "FAILED LOOKUP: Slot #" << slot << "\n");
	return true;
      }
      BCR_TRACE(4, "Map: '" << Name << "' to #" << slot << ":" << D;
		if (!isa<Instruction>(D)) std::cerr << "\n");

      D->setName(Name, ST);
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

  // Loop over all of the uses of the Value.  What they are depends
  // on what NewV is.  Replacing a use of the old reference takes the
  // use off the use list, so loop with !use_empty(), not the use_iterator.
  while (!VPH->use_empty()) {
    Constant *C = cast<Constant>(VPH->use_back());
    unsigned numReplaced = C->mutateReferences(VPH, NewV);
    assert(numReplaced > 0 && "Supposed user wasn't really a user?");
      
    if (GlobalValue* GVal = dyn_cast<GlobalValue>(NewV)) {
      // Remove the placeholder GlobalValue from the module...
      GVal->getParent()->getGlobalList().remove(cast<GlobalVariable>(VPH));
    }
  }

  delete VPH;                         // Delete the old placeholder
  GlobalRefs.erase(I);                // Remove the map entry for it
}

bool BytecodeParser::ParseMethod(const uchar *&Buf, const uchar *EndBuf, 
				 Module *C) {
  // Clear out the local values table...
  Values.clear();
  if (FunctionSignatureList.empty()) {
    Error = "Function found, but FunctionSignatureList empty!";
    return true;  // Unexpected method!
  }

  const PointerType *PMTy = FunctionSignatureList.back().first; // PtrMeth
  const FunctionType *MTy  = dyn_cast<FunctionType>(PMTy->getElementType());
  if (MTy == 0) return true;  // Not ptr to method!

  unsigned isInternal;
  if (read_vbr(Buf, EndBuf, isInternal)) return true;

  unsigned MethSlot = FunctionSignatureList.back().second;
  FunctionSignatureList.pop_back();
  Function *M = new Function(MTy, isInternal != 0);

  BCR_TRACE(2, "METHOD TYPE: " << MTy << "\n");

  const FunctionType::ParamTypes &Params = MTy->getParamTypes();
  Function::aiterator AI = M->abegin();
  for (FunctionType::ParamTypes::const_iterator It = Params.begin();
       It != Params.end(); ++It, ++AI) {
    if (insertValue(AI, Values) == -1) {
      Error = "Error reading method arguments!\n";
      delete M; return true; 
    }
  }

  while (Buf < EndBuf) {
    unsigned Type, Size;
    const uchar *OldBuf = Buf;
    if (readBlock(Buf, EndBuf, Type, Size)) {
      Error = "Error reading Function level block!";
      delete M; return true; 
    }

    switch (Type) {
    case BytecodeFormat::ConstantPool:
      BCR_TRACE(2, "BLOCK BytecodeFormat::ConstantPool: {\n");
      if (ParseConstantPool(Buf, Buf+Size, Values, MethodTypeValues)) {
	delete M; return true;
      }
      break;

    case BytecodeFormat::BasicBlock: {
      BCR_TRACE(2, "BLOCK BytecodeFormat::BasicBlock: {\n");
      BasicBlock *BB;
      if (ParseBasicBlock(Buf, Buf+Size, BB) ||
	  insertValue(BB, Values) == -1) {
	delete M; return true;                // Parse error... :(
      }

      M->getBasicBlockList().push_back(BB);
      break;
    }

    case BytecodeFormat::SymbolTable:
      BCR_TRACE(2, "BLOCK BytecodeFormat::SymbolTable: {\n");
      if (ParseSymbolTable(Buf, Buf+Size, &M->getSymbolTable())) {
	delete M; return true;
      }
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
      delete M;    // Malformed bc file, read past end of block.
      return true;
    }
  }

  if (postResolveValues(LateResolveValues) ||
      postResolveValues(LateResolveModuleValues)) {
    Error = "Error resolving method values!";
    delete M; return true;     // Unresolvable references!
  }

  Value *FunctionPHolder = getValue(PMTy, MethSlot, false);
  assert(FunctionPHolder && "Something is broken no placeholder found!");
  assert(isa<Function>(FunctionPHolder) && "Not a function?");

  unsigned type;  // Type slot
  assert(!getTypeSlot(MTy, type) && "How can meth type not exist?");
  getTypeSlot(PMTy, type);

  C->getFunctionList().push_back(M);

  // Replace placeholder with the real method pointer...
  ModuleValues[type][MethSlot] = M;

  // Clear out method level types...
  MethodTypeValues.clear();

  // If anyone is using the placeholder make them use the real method instead
  FunctionPHolder->replaceAllUsesWith(M);

  // We don't need the placeholder anymore!
  delete FunctionPHolder;

  ResolveReferencesToValue(M, MethSlot);

  return false;
}

bool BytecodeParser::ParseModuleGlobalInfo(const uchar *&Buf, const uchar *End,
					   Module *Mod) {
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

    const PointerType *PTy = cast<const PointerType>(Ty);
    const Type *ElTy = PTy->getElementType();

    Constant *Initializer = 0;
    if (VarType & 2) { // Does it have an initalizer?
      // Do not improvise... values must have been stored in the constant pool,
      // which should have been read before now.
      //
      unsigned InitSlot;
      if (read_vbr(Buf, End, InitSlot)) return true;
      
      Value *V = getValue(ElTy, InitSlot, false);
      if (V == 0) return true;
      Initializer = cast<Constant>(V);
    }

    // Create the global variable...
    GlobalVariable *GV = new GlobalVariable(ElTy, VarType & 1, VarType & 4,
					    Initializer);
    int DestSlot = insertValue(GV, ModuleValues);
    if (DestSlot == -1) return true;

    Mod->getGlobalList().push_back(GV);

    ResolveReferencesToValue(GV, (unsigned)DestSlot);

    BCR_TRACE(2, "Global Variable of type: " << PTy->getDescription() 
	      << " into slot #" << DestSlot << "\n");

    if (read_vbr(Buf, End, VarType)) return true;
  }

  // Read the method signatures for all of the methods that are coming, and 
  // create fillers in the Value tables.
  unsigned FnSignature;
  if (read_vbr(Buf, End, FnSignature)) return true;
  while (FnSignature != Type::VoidTyID) { // List is terminated by Void
    const Type *Ty = getType(FnSignature);
    if (!Ty || !isa<PointerType>(Ty) ||
        !isa<FunctionType>(cast<PointerType>(Ty)->getElementType())) { 
      Error = "Function not ptr to func type!  Ty = " + Ty->getDescription();
      return true; 
    }

    // We create methods by passing the underlying FunctionType to create...
    Ty = cast<PointerType>(Ty)->getElementType();

    // When the ModuleGlobalInfo section is read, we load the type of each 
    // method and the 'ModuleValues' slot that it lands in.  We then load a 
    // placeholder into its slot to reserve it.  When the method is loaded, this
    // placeholder is replaced.

    // Insert the placeholder...
    Value *Val = new FunctionPHolder(Ty, 0);
    if (insertValue(Val, ModuleValues) == -1) return true;

    // Figure out which entry of its typeslot it went into...
    unsigned TypeSlot;
    if (getTypeSlot(Val->getType(), TypeSlot)) return true;

    unsigned SlotNo = ModuleValues[TypeSlot].size()-1;
    
    // Keep track of this information in a linked list that is emptied as 
    // methods are loaded...
    //
    FunctionSignatureList.push_back(
           std::make_pair(cast<const PointerType>(Val->getType()), SlotNo));
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

bool BytecodeParser::ParseModule(const uchar *Buf, const uchar *EndBuf, 
				Module *&Mod) {

  unsigned Type, Size;
  if (readBlock(Buf, EndBuf, Type, Size)) return true;
  if (Type != BytecodeFormat::Module || Buf+Size != EndBuf) {
    Error = "Expected Module packet!";
    return true;                      // Hrm, not a class?
  }

  BCR_TRACE(0, "BLOCK BytecodeFormat::Module: {\n");
  FunctionSignatureList.clear();                 // Just in case...

  // Read into instance variables...
  if (read_vbr(Buf, EndBuf, FirstDerivedTyID)) return true;
  if (align32(Buf, EndBuf)) return true;
  BCR_TRACE(1, "FirstDerivedTyID = " << FirstDerivedTyID << "\n");

  TheModule = Mod = new Module();

  while (Buf < EndBuf) {
    const uchar *OldBuf = Buf;
    if (readBlock(Buf, EndBuf, Type, Size)) { delete Mod; return true;}
    switch (Type) {
    case BytecodeFormat::ConstantPool:
      BCR_TRACE(1, "BLOCK BytecodeFormat::ConstantPool: {\n");
      if (ParseConstantPool(Buf, Buf+Size, ModuleValues, ModuleTypeValues)) {
	delete Mod; return true;
      }
      break;

    case BytecodeFormat::ModuleGlobalInfo:
      BCR_TRACE(1, "BLOCK BytecodeFormat::ModuleGlobalInfo: {\n");

      if (ParseModuleGlobalInfo(Buf, Buf+Size, Mod)) {
	delete Mod; return true;
      }
      break;

    case BytecodeFormat::Function: {
      BCR_TRACE(1, "BLOCK BytecodeFormat::Function: {\n");
      if (ParseMethod(Buf, Buf+Size, Mod)) {
	delete Mod; return true;              // Error parsing function
      }
      break;
    }

    case BytecodeFormat::SymbolTable:
      BCR_TRACE(1, "BLOCK BytecodeFormat::SymbolTable: {\n");
      if (ParseSymbolTable(Buf, Buf+Size, &Mod->getSymbolTable())) {
	delete Mod; return true;
      }
      break;

    default:
      Error = "Expected Module Block!";
      Buf += Size;
      if (OldBuf > Buf) return true; // Wrap around!
      break;
    }
    BCR_TRACE(1, "} end block\n");
    if (align32(Buf, EndBuf)) { delete Mod; return true; }
  }

  if (!FunctionSignatureList.empty()) {     // Expected more methods!
    Error = "Function expected, but bytecode stream at end!";
    return true;
  }

  BCR_TRACE(0, "} end block\n\n");
  return false;
}

Module *BytecodeParser::ParseBytecode(const uchar *Buf, const uchar *EndBuf) {
  LateResolveValues.clear();
  unsigned Sig;
  // Read and check signature...
  if (read(Buf, EndBuf, Sig) ||
      Sig != ('l' | ('l' << 8) | ('v' << 16) | 'm' << 24)) {
    Error = "Invalid bytecode signature!";
    return 0;                          // Invalid signature!
  }

  Module *Result;
  if (ParseModule(Buf, EndBuf, Result)) return 0;
  return Result;
}


Module *ParseBytecodeBuffer(const unsigned char *Buffer, unsigned Length,
                            std::string *ErrorStr) {
  BytecodeParser Parser;
  Module *R = Parser.ParseBytecode(Buffer, Buffer+Length);
  if (ErrorStr) *ErrorStr = Parser.getError();
  return R;
}

// Parse and return a class file...
//
Module *ParseBytecodeFile(const std::string &Filename, std::string *ErrorStr) {
  Module *Result = 0;

  if (Filename != std::string("-")) {        // Read from a file...
    int FD = open(Filename.c_str(), O_RDONLY);
    if (FD == -1) {
      if (ErrorStr) *ErrorStr = "Error opening file!";
      return 0;
    }

    struct stat StatBuf;
    if (fstat(FD, &StatBuf) == -1 ||
        StatBuf.st_size == 0) { 
      if (ErrorStr) *ErrorStr = "Error stat'ing file!";
      close(FD); return 0; 
    }

    int Length = StatBuf.st_size;
    unsigned char *Buffer = (unsigned char*)mmap(0, Length, PROT_READ, 
                                                 MAP_PRIVATE, FD, 0);
    if (Buffer == (uchar*)-1) {
      if (ErrorStr) *ErrorStr = "Error mmapping file!";
      close(FD); return 0;
    }

    Result = ParseBytecodeBuffer(Buffer, Length, ErrorStr);

    munmap((char*)Buffer, Length);
    close(FD);
  } else {                              // Read from stdin
    size_t FileSize = 0;
    int BlockSize;
    uchar Buffer[4096], *FileData = 0;
    while ((BlockSize = read(0, Buffer, 4096))) {
      if (BlockSize == -1) { free(FileData); return 0; }

      FileData = (uchar*)realloc(FileData, FileSize+BlockSize);
      memcpy(FileData+FileSize, Buffer, BlockSize);
      FileSize += BlockSize;
    }

    if (FileSize == 0) {
      if (ErrorStr) *ErrorStr = "Standard Input empty!";
      free(FileData); return 0;
    }

#define ALIGN_PTRS 0
#if ALIGN_PTRS
    uchar *Buf = (uchar*)mmap(0, FileSize, PROT_READ|PROT_WRITE, 
			      MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    assert((Buf != (uchar*)-1) && "mmap returned error!");
    memcpy(Buf, FileData, FileSize);
    free(FileData);
#else
    uchar *Buf = FileData;
#endif

    Result = ParseBytecodeBuffer(Buf, FileSize, ErrorStr);

#if ALIGN_PTRS
    munmap((char*)Buf, FileSize);   // Free mmap'd data area
#else
    free(FileData);          // Free realloc'd block of memory
#endif
  }

  return Result;
}
