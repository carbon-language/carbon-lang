//===- Reader.cpp - Code to read bytecode files -----------------------------===
//
// This library implements the functionality defined in llvm/Bytecode/Reader.h
//
// Note that this library should be as fast as possible, reentrant, and 
// threadsafe!!
//
// TODO: Make error message outputs be configurable depending on an option?
// TODO: Allow passing in an option to ignore the symbol table
//
//===------------------------------------------------------------------------===

#include "llvm/Bytecode/Reader.h"
#include "llvm/Bytecode/Format.h"
#include "llvm/GlobalVariable.h"
#include "llvm/Module.h"
#include "llvm/BasicBlock.h"
#include "llvm/DerivedTypes.h"
#include "llvm/ConstPoolVals.h"
#include "llvm/iOther.h"
#include "ReaderInternals.h"
#include <sys/types.h>
#include <sys/mman.h>
#include <sys/stat.h>
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
  //cerr << "getTypeSlot '" << Ty->getName() << "' = " << Slot << endl;
  return false;
}

const Type *BytecodeParser::getType(unsigned ID) {
  const Type *T = Type::getPrimitiveType((Type::PrimitiveID)ID);
  if (T) return T;
  
  //cerr << "Looking up Type ID: " << ID << endl;

  const Value *D = getValue(Type::TypeTy, ID, false);
  if (D == 0) return failure<const Type*>(0);

  return cast<Type>(D);
}

bool BytecodeParser::insertValue(Value *Val, vector<ValueList> &ValueTab) {
  unsigned type;
  if (getTypeSlot(Val->getType(), type)) return failure(true);
  assert(type != Type::TypeTyID && "Types should never be insertValue'd!");
 
  if (ValueTab.size() <= type)
    ValueTab.resize(type+1, ValueList());

  //cerr << "insertValue Values[" << type << "][" << ValueTab[type].size() 
  //     << "] = " << Val << endl;
  ValueTab[type].push_back(Val);

  return false;
}

Value *BytecodeParser::getValue(const Type *Ty, unsigned oNum, bool Create) {
  unsigned Num = oNum;
  unsigned type;   // The type plane it lives in...

  if (getTypeSlot(Ty, type)) return failure<Value*>(0); // TODO: true

  if (type == Type::TypeTyID) {  // The 'type' plane has implicit values
    assert(Create == false);
    const Type *T = Type::getPrimitiveType((Type::PrimitiveID)Num);
    if (T) return (Value*)T;   // Asked for a primitive type...

    // Otherwise, derived types need offset...
    Num -= FirstDerivedTyID;

    // Is it a module level type?
    if (Num < ModuleTypeValues.size())
      return (Value*)(const Type*)ModuleTypeValues[Num];

    // Nope, is it a method level type?
    Num -= ModuleTypeValues.size();
    if (Num < MethodTypeValues.size())
      return (Value*)(const Type*)MethodTypeValues[Num];

    return 0;
  }

  if (type < ModuleValues.size()) {
    if (Num < ModuleValues[type].size())
      return ModuleValues[type][Num];
    Num -= ModuleValues[type].size();
  }

  if (Values.size() > type && Values[type].size() > Num)
    return Values[type][Num];

  if (!Create) return failure<Value*>(0);  // Do not create a placeholder?

  Value *d = 0;
  switch (Ty->getPrimitiveID()) {
  case Type::LabelTyID: d = new    BBPHolder(Ty, oNum); break;
  case Type::MethodTyID:
    cerr << "Creating method pholder! : " << type << ":" << oNum << " " 
	 << Ty->getName() << endl;
    d = new MethPHolder(Ty, oNum);
    insertValue(d, LateResolveModuleValues);
    return d;
  default:                   d = new   DefPHolder(Ty, oNum); break;
  }

  assert(d != 0 && "How did we not make something?");
  if (insertValue(d, LateResolveValues)) return failure<Value*>(0);
  return d;
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
	cerr << "Unresolvable reference found: <" << D->getType()->getName()
	     << ">:" << IDNumber << "!\n";
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
    if (ParseInstruction(Buf, EndBuf, Inst)) {
      delete BB;
      return failure(true);
    }

    if (Inst == 0) { delete BB; return failure(true); }
    if (insertValue(Inst, Values)) { delete BB; return failure(true); }

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
        read_vbr(Buf, EndBuf, Typ)) return failure(true);
    const Type *Ty = getType(Typ);
    if (Ty == 0) return failure(true);

    BCR_TRACE(3, "Plane Type: '" << Ty << "' with " << NumEntries <<
	      " entries\n");

    for (unsigned i = 0; i < NumEntries; ++i) {
      // Symtab entry: [def slot #][name]
      unsigned slot;
      if (read_vbr(Buf, EndBuf, slot)) return failure(true);
      string Name;
      if (read(Buf, EndBuf, Name, false))  // Not aligned...
	return failure(true);

      Value *D = getValue(Ty, slot, false); // Find mapping...
      if (D == 0) {
	BCR_TRACE(3, "FAILED LOOKUP: Slot #" << slot << endl);
	return failure(true);
      }
      BCR_TRACE(4, "Map: '" << Name << "' to #" << slot << ":" << D;
		if (!D->isInstruction()) cerr << endl);

      D->setName(Name, ST);
    }
  }

  if (Buf > EndBuf) return failure(true);
  return false;
}


bool BytecodeParser::ParseMethod(const uchar *&Buf, const uchar *EndBuf, 
				 Module *C) {
  // Clear out the local values table...
  Values.clear();
  if (MethodSignatureList.empty()) return failure(true);  // Unexpected method!

  const MethodType *MTy = MethodSignatureList.front().first;
  unsigned MethSlot = MethodSignatureList.front().second;
  MethodSignatureList.pop_front();
  Method *M = new Method(MTy);

  BCR_TRACE(2, "METHOD TYPE: " << MTy << endl);

  const MethodType::ParamTypes &Params = MTy->getParamTypes();
  for (MethodType::ParamTypes::const_iterator It = Params.begin();
       It != Params.end(); ++It) {
    MethodArgument *MA = new MethodArgument(*It);
    if (insertValue(MA, Values)) { delete M; return failure(true); }
    M->getArgumentList().push_back(MA);
  }

  while (Buf < EndBuf) {
    unsigned Type, Size;
    const uchar *OldBuf = Buf;
    if (readBlock(Buf, EndBuf, Type, Size)) { delete M; return failure(true); }

    switch (Type) {
    case BytecodeFormat::ConstantPool:
      BCR_TRACE(2, "BLOCK BytecodeFormat::ConstantPool: {\n");
      if (ParseConstantPool(Buf, Buf+Size, Values, MethodTypeValues)) {
	delete M; return failure(true);
      }
      break;

    case BytecodeFormat::BasicBlock: {
      BCR_TRACE(2, "BLOCK BytecodeFormat::BasicBlock: {\n");
      BasicBlock *BB;
      if (ParseBasicBlock(Buf, Buf+Size, BB) ||
	  insertValue(BB, Values)) {
	delete M; return failure(true);                // Parse error... :(
      }

      M->getBasicBlocks().push_back(BB);
      break;
    }

    case BytecodeFormat::SymbolTable:
      BCR_TRACE(2, "BLOCK BytecodeFormat::SymbolTable: {\n");
      if (ParseSymbolTable(Buf, Buf+Size, M->getSymbolTableSure())) {
	delete M; return failure(true);
      }
      break;

    default:
      BCR_TRACE(2, "BLOCK <unknown>:ignored! {\n");
      Buf += Size;
      if (OldBuf > Buf) return failure(true); // Wrap around!
      break;
    }
    BCR_TRACE(2, "} end block\n");

    if (align32(Buf, EndBuf)) {
      delete M;    // Malformed bc file, read past end of block.
      return failure(true);
    }
  }

  if (postResolveValues(LateResolveValues) ||
      postResolveValues(LateResolveModuleValues)) {
    delete M; return failure(true);     // Unresolvable references!
  }

  Value *MethPHolder = getValue(MTy, MethSlot, false);
  assert(MethPHolder && "Something is broken no placeholder found!");
  assert(MethPHolder->isMethod() && "Not a method?");

  unsigned type;  // Type slot
  assert(!getTypeSlot(MTy, type) && "How can meth type not exist?");
  getTypeSlot(MTy, type);

  C->getMethodList().push_back(M);

  // Replace placeholder with the real method pointer...
  ModuleValues[type][MethSlot] = M;

  // Clear out method level types...
  MethodTypeValues.clear();

  // If anyone is using the placeholder make them use the real method instead
  MethPHolder->replaceAllUsesWith(M);

  // We don't need the placeholder anymore!
  delete MethPHolder;

  return false;
}

bool BytecodeParser::ParseModuleGlobalInfo(const uchar *&Buf, const uchar *End,
					  Module *C) {
  if (!MethodSignatureList.empty()) 
    return failure(true);  // Two ModuleGlobal blocks?

  // Read global variables...
  unsigned VarType;
  if (read_vbr(Buf, End, VarType)) return failure(true);
  while (VarType != Type::VoidTyID) { // List is terminated by Void
    // VarType Fields: bit0 = isConstant, bit1 = hasInitializer, bit2+ = slot#
    const Type *Ty = getType(VarType >> 2);
    if (!Ty || !Ty->isPointerType()) { 
      cerr << "Global not pointer type!  Ty = " << Ty << endl;
      return failure(true); 
    }

    ConstPoolVal *Initializer = 0;
    if (VarType & 2) { // Does it have an initalizer?
      // Do not improvise... values must have been stored in the constant pool,
      // which should have been read before now.
      //
      unsigned InitSlot;
      if (read_vbr(Buf, End, InitSlot)) return failure(true);
      
      Value *V = getValue(Ty->castPointerType()->getValueType(),
			  InitSlot, false);
      if (V == 0) return failure(true);
      Initializer = cast<ConstPoolVal>(V);
    }

    // Create the global variable...
    GlobalVariable *GV = new GlobalVariable(Ty, VarType & 1, Initializer);
    insertValue(GV, ModuleValues);
    C->getGlobalList().push_back(GV);

    if (read_vbr(Buf, End, VarType)) return failure(true);
    BCR_TRACE(2, "Global Variable of type: " << Ty->getDescription() << endl);
  }

  // Read the method signatures for all of the methods that are coming, and 
  // create fillers in the Value tables.
  unsigned MethSignature;
  if (read_vbr(Buf, End, MethSignature)) return failure(true);
  while (MethSignature != Type::VoidTyID) { // List is terminated by Void
    const Type *Ty = getType(MethSignature);
    if (!Ty || !Ty->isMethodType()) { 
      cerr << "Method not meth type!  Ty = " << Ty << endl;
      return failure(true); 
    }

    // When the ModuleGlobalInfo section is read, we load the type of each 
    // method and the 'ModuleValues' slot that it lands in.  We then load a 
    // placeholder into its slot to reserve it.  When the method is loaded, this
    // placeholder is replaced.

    // Insert the placeholder...
    Value *Def = new MethPHolder(Ty, 0);
    insertValue(Def, ModuleValues);

    // Figure out which entry of its typeslot it went into...
    unsigned TypeSlot;
    if (getTypeSlot(Def->getType(), TypeSlot)) return failure(true);

    unsigned SlotNo = ModuleValues[TypeSlot].size()-1;
    
    // Keep track of this information in a linked list that is emptied as 
    // methods are loaded...
    //
    MethodSignatureList.push_back(make_pair((const MethodType*)Ty, SlotNo));
    if (read_vbr(Buf, End, MethSignature)) return failure(true);
    BCR_TRACE(2, "Method of type: " << Ty << endl);
  }

  if (align32(Buf, End)) return failure(true);

  // This is for future proofing... in the future extra fields may be added that
  // we don't understand, so we transparently ignore them.
  //
  Buf = End;
  return false;
}

bool BytecodeParser::ParseModule(const uchar *Buf, const uchar *EndBuf, 
				Module *&C) {

  unsigned Type, Size;
  if (readBlock(Buf, EndBuf, Type, Size)) return failure(true);
  if (Type != BytecodeFormat::Module || Buf+Size != EndBuf)
    return failure(true);                      // Hrm, not a class?

  BCR_TRACE(0, "BLOCK BytecodeFormat::Module: {\n");
  MethodSignatureList.clear();                 // Just in case...

  // Read into instance variables...
  if (read_vbr(Buf, EndBuf, FirstDerivedTyID)) return failure(true);
  if (align32(Buf, EndBuf)) return failure(true);
  BCR_TRACE(1, "FirstDerivedTyID = " << FirstDerivedTyID << "\n");

  C = new Module();
  while (Buf < EndBuf) {
    const uchar *OldBuf = Buf;
    if (readBlock(Buf, EndBuf, Type, Size)) { delete C; return failure(true); }
    switch (Type) {
    case BytecodeFormat::ConstantPool:
      BCR_TRACE(1, "BLOCK BytecodeFormat::ConstantPool: {\n");
      if (ParseConstantPool(Buf, Buf+Size, ModuleValues, ModuleTypeValues)) {
	delete C; return failure(true);
      }
      break;

    case BytecodeFormat::ModuleGlobalInfo:
      BCR_TRACE(1, "BLOCK BytecodeFormat::ModuleGlobalInfo: {\n");

      if (ParseModuleGlobalInfo(Buf, Buf+Size, C)) {
	delete C; return failure(true);
      }
      break;

    case BytecodeFormat::Method: {
      BCR_TRACE(1, "BLOCK BytecodeFormat::Method: {\n");
      if (ParseMethod(Buf, Buf+Size, C)) {
	delete C; return failure(true);               // Error parsing method
      }
      break;
    }

    case BytecodeFormat::SymbolTable:
      BCR_TRACE(1, "BLOCK BytecodeFormat::SymbolTable: {\n");
      if (ParseSymbolTable(Buf, Buf+Size, C->getSymbolTableSure())) {
	delete C; return failure(true);
      }
      break;

    default:
      cerr << "  Unknown class block: " << Type << endl;
      Buf += Size;
      if (OldBuf > Buf) return failure(true); // Wrap around!
      break;
    }
    BCR_TRACE(1, "} end block\n");
    if (align32(Buf, EndBuf)) { delete C; return failure(true); }
  }

  if (!MethodSignatureList.empty())      // Expected more methods!
    return failure(true);

  BCR_TRACE(0, "} end block\n\n");
  return false;
}

Module *BytecodeParser::ParseBytecode(const uchar *Buf, const uchar *EndBuf) {
  LateResolveValues.clear();
  unsigned Sig;
  // Read and check signature...
  if (read(Buf, EndBuf, Sig) ||
      Sig != ('l' | ('l' << 8) | ('v' << 16) | 'm' << 24))
    return failure<Module*>(0);                          // Invalid signature!

  Module *Result;
  if (ParseModule(Buf, EndBuf, Result)) return 0;
  return Result;
}


Module *ParseBytecodeBuffer(const uchar *Buffer, unsigned Length) {
  BytecodeParser Parser;
  return Parser.ParseBytecode(Buffer, Buffer+Length);
}

// Parse and return a class file...
//
Module *ParseBytecodeFile(const string &Filename) {
  struct stat StatBuf;
  Module *Result = 0;

  if (Filename != string("-")) {        // Read from a file...
    int FD = open(Filename.c_str(), O_RDONLY);
    if (FD == -1) return failure<Module*>(0);

    if (fstat(FD, &StatBuf) == -1) { close(FD); return failure<Module*>(0); }

    int Length = StatBuf.st_size;
    if (Length == 0) { close(FD); return failure<Module*>(0); }
    uchar *Buffer = (uchar*)mmap(0, Length, PROT_READ, 
				MAP_PRIVATE, FD, 0);
    if (Buffer == (uchar*)-1) { close(FD); return failure<Module*>(0); }

    BytecodeParser Parser;
    Result  = Parser.ParseBytecode(Buffer, Buffer+Length);

    munmap((char*)Buffer, Length);
    close(FD);
  } else {                              // Read from stdin
    size_t FileSize = 0;
    int BlockSize;
    uchar Buffer[4096], *FileData = 0;
    while ((BlockSize = read(0, Buffer, 4))) {
      if (BlockSize == -1) { free(FileData); return failure<Module*>(0); }

      FileData = (uchar*)realloc(FileData, FileSize+BlockSize);
      memcpy(FileData+FileSize, Buffer, BlockSize);
      FileSize += BlockSize;
    }

    if (FileSize == 0) { free(FileData); return failure<Module*>(0); }

#define ALIGN_PTRS 1
#if ALIGN_PTRS
    uchar *Buf = (uchar*)mmap(0, FileSize, PROT_READ|PROT_WRITE, 
			      MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    assert((Buf != (uchar*)-1) && "mmap returned error!");
    free(FileData);
    memcpy(Buf, FileData, FileSize);
#else
    uchar *Buf = FileData;
#endif

    BytecodeParser Parser;
    Result = Parser.ParseBytecode(Buf, Buf+FileSize);

#if ALIGN_PTRS
    munmap((char*)Buf, FileSize);   // Free mmap'd data area
#else
    free(FileData);          // Free realloc'd block of memory
#endif
  }

  return Result;
}
