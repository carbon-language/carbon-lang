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
    TypeMapType::iterator I = TypeMap.find(Ty);
    if (I == TypeMap.end()) return failure(true);   // Didn't find type!
    Slot = I->second;
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

  assert(D->getType() == Type::TypeTy);
  return ((const ConstPoolType*)D->castConstantAsserting())->getValue();
}

bool BytecodeParser::insertValue(Value *Def, vector<ValueList> &ValueTab) {
  unsigned type;
  if (getTypeSlot(Def->getType(), type)) return failure(true);
 
  if (ValueTab.size() <= type)
    ValueTab.resize(type+1, ValueList());

  //cerr << "insertValue Values[" << type << "][" << ValueTab[type].size() 
  //     << "] = " << Def << endl;

  if (type == Type::TypeTyID && Def->isConstant()) {
    const Type *Ty = ((const ConstPoolType*)Def)->getValue();
    unsigned ValueOffset = FirstDerivedTyID;

    if (&ValueTab == &Values)    // Take into consideration module level types
      ValueOffset += ModuleValues[type].size();

    if (TypeMap.find(Ty) == TypeMap.end())
      TypeMap[Ty] = ValueTab[type].size()+ValueOffset;
  }

  ValueTab[type].push_back(Def);

  return false;
}

Value *BytecodeParser::getValue(const Type *Ty, unsigned oNum, bool Create) {
  unsigned Num = oNum;
  unsigned type;   // The type plane it lives in...

  if (getTypeSlot(Ty, type)) return failure<Value*>(0); // TODO: true

  if (type == Type::TypeTyID) {  // The 'type' plane has implicit values
    const Type *T = Type::getPrimitiveType((Type::PrimitiveID)Num);
    if (T) return (Value*)T;   // Asked for a primitive type...

    // Otherwise, derived types need offset...
    Num -= FirstDerivedTyID;
  }

  if (ModuleValues.size() > type) {
    if (ModuleValues[type].size() > Num)
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
    Instruction *Def;
    if (ParseInstruction(Buf, EndBuf, Def)) {
      delete BB;
      return failure(true);
    }

    if (Def == 0) { delete BB; return failure(true); }
    if (insertValue(Def, Values)) { delete BB; return failure(true); }

    BB->getInstList().push_back(Def);
  }

  return false;
}

bool BytecodeParser::ParseSymbolTable(const uchar *&Buf, const uchar *EndBuf) {
  while (Buf < EndBuf) {
    // Symtab block header: [num entries][type id number]
    unsigned NumEntries, Typ;
    if (read_vbr(Buf, EndBuf, NumEntries) ||
        read_vbr(Buf, EndBuf, Typ)) return failure(true);
    const Type *Ty = getType(Typ);
    if (Ty == 0) return failure(true);

    for (unsigned i = 0; i < NumEntries; ++i) {
      // Symtab entry: [def slot #][name]
      unsigned slot;
      if (read_vbr(Buf, EndBuf, slot)) return failure(true);
      string Name;
      if (read(Buf, EndBuf, Name, false))  // Not aligned...
	return failure(true);

      Value *D = getValue(Ty, slot, false); // Find mapping...
      if (D == 0) return failure(true);
      D->setName(Name);
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
      if (ParseConstantPool(Buf, Buf+Size, M->getConstantPool(), Values)) {
	cerr << "Error reading constant pool!\n";
	delete M; return failure(true);
      }
      break;

    case BytecodeFormat::BasicBlock: {
      BasicBlock *BB;
      if (ParseBasicBlock(Buf, Buf+Size, BB) ||
	  insertValue(BB, Values)) {
	cerr << "Error parsing basic block!\n";
	delete M; return failure(true);                // Parse error... :(
      }

      M->getBasicBlocks().push_back(BB);
      break;
    }

    case BytecodeFormat::SymbolTable:
      if (ParseSymbolTable(Buf, Buf+Size)) {
	cerr << "Error reading method symbol table!\n";
	delete M; return failure(true);
      }
      break;

    default:
      Buf += Size;
      if (OldBuf > Buf) return failure(true); // Wrap around!
      break;
    }
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

  // Read the method signatures for all of the methods that are coming, and 
  // create fillers in the Value tables.
  unsigned MethSignature;
  if (read_vbr(Buf, End, MethSignature)) return failure(true);
  while (MethSignature != Type::VoidTyID) { // List is terminated by Void
    const Type *Ty = getType(MethSignature);
    if (!Ty || !Ty->isMethodType()) { 
      cerr << "Method not meth type! ";
      if (Ty) cerr << Ty->getName(); else cerr << MethSignature; cerr << endl; 
      return failure(true); 
    }

    // When the ModuleGlobalInfo section is read, we load the type of each method
    // and the 'ModuleValues' slot that it lands in.  We then load a placeholder
    // into its slot to reserve it.  When the method is loaded, this placeholder
    // is replaced.

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

  MethodSignatureList.clear();                 // Just in case...

  // Read into instance variables...
  if (read_vbr(Buf, EndBuf, FirstDerivedTyID)) return failure(true);
  if (align32(Buf, EndBuf)) return failure(true);

  C = new Module();

  while (Buf < EndBuf) {
    const uchar *OldBuf = Buf;
    if (readBlock(Buf, EndBuf, Type, Size)) { delete C; return failure(true); }
    switch (Type) {
    case BytecodeFormat::ModuleGlobalInfo:
      if (ParseModuleGlobalInfo(Buf, Buf+Size, C)) {
	cerr << "Error reading class global info section!\n";
	delete C; return failure(true);
      }
      break;

    case BytecodeFormat::ConstantPool:
      if (ParseConstantPool(Buf, Buf+Size, C->getConstantPool(), ModuleValues)) {
	cerr << "Error reading class constant pool!\n";
	delete C; return failure(true);
      }
      break;

    case BytecodeFormat::Method: {
      if (ParseMethod(Buf, Buf+Size, C)) {
	delete C; return failure(true);               // Error parsing method
      }
      break;
    }

    case BytecodeFormat::SymbolTable:
      if (ParseSymbolTable(Buf, Buf+Size)) {
	cerr << "Error reading class symbol table!\n";
	delete C; return failure(true);
      }
      break;

    default:
      cerr << "Unknown class block: " << Type << endl;
      Buf += Size;
      if (OldBuf > Buf) return failure(true); // Wrap around!
      break;
    }
    if (align32(Buf, EndBuf)) { delete C; return failure(true); }
  }

  if (!MethodSignatureList.empty())      // Expected more methods!
    return failure(true);
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
