//===- ReaderWrappers.cpp - Parse bytecode from file or buffer  -----------===//
//
// This file implements loading and parsing a bytecode file and parsing a
// bytecode module from a given buffer.
//
//===----------------------------------------------------------------------===//

#include "ReaderInternals.h"
#include "llvm/Module.h"
#include "llvm/Instructions.h"
#include "Support/StringExtras.h"
#include "Config/fcntl.h"
#include <sys/stat.h>
#include "Config/unistd.h"
#include "Config/sys/mman.h"

//===----------------------------------------------------------------------===//
// BytecodeFileReader - Read from an mmap'able file descriptor.
//

namespace {
  /// FDHandle - Simple handle class to make sure a file descriptor gets closed
  /// when the object is destroyed.
  ///
  class FDHandle {
    int FD;
  public:
    FDHandle(int fd) : FD(fd) {}
    operator int() const { return FD; }
    ~FDHandle() {
      if (FD != -1) close(FD);
    }
  };

  /// BytecodeFileReader - parses a bytecode file from a file
  ///
  class BytecodeFileReader : public BytecodeParser {
  private:
    unsigned char *Buffer;
    int Length;

    BytecodeFileReader(const BytecodeFileReader&); // Do not implement
    void operator=(const BytecodeFileReader &BFR); // Do not implement

  public:
    BytecodeFileReader(const std::string &Filename);
    ~BytecodeFileReader();
  };
}

BytecodeFileReader::BytecodeFileReader(const std::string &Filename) {
  FDHandle FD = open(Filename.c_str(), O_RDONLY);
  if (FD == -1)
    throw std::string("Error opening file!");

  // Stat the file to get its length...
  struct stat StatBuf;
  if (fstat(FD, &StatBuf) == -1 || StatBuf.st_size == 0)
    throw std::string("Error stat'ing file!");

  // mmap in the file all at once...
  Length = StatBuf.st_size;
  Buffer = (unsigned char*)mmap(0, Length, PROT_READ, MAP_PRIVATE, FD, 0);

  if (Buffer == (unsigned char*)MAP_FAILED)
    throw std::string("Error mmapping file!");

  try {
    // Parse the bytecode we mmapped in
    ParseBytecode(Buffer, Length, Filename);
  } catch (...) {
    munmap((char*)Buffer, Length);
    throw;
  }
}

BytecodeFileReader::~BytecodeFileReader() {
  // Unmmap the bytecode...
  munmap((char*)Buffer, Length);
}

//===----------------------------------------------------------------------===//
// BytecodeBufferReader - Read from a memory buffer
//

namespace {
  /// BytecodeBufferReader - parses a bytecode file from a buffer
  ///
  class BytecodeBufferReader : public BytecodeParser {
  private:
    const unsigned char *Buffer;
    bool MustDelete;

    BytecodeBufferReader(const BytecodeBufferReader&); // Do not implement
    void operator=(const BytecodeBufferReader &BFR);   // Do not implement

  public:
    BytecodeBufferReader(const unsigned char *Buf, unsigned Length,
                         const std::string &ModuleID);
    ~BytecodeBufferReader();

  };
}

BytecodeBufferReader::BytecodeBufferReader(const unsigned char *Buf,
                                           unsigned Length,
                                           const std::string &ModuleID)
{
  // If not aligned, allocate a new buffer to hold the bytecode...
  const unsigned char *ParseBegin = 0;
  if ((intptr_t)Buf & 3) {
    Buffer = new unsigned char[Length+4];
    unsigned Offset = 4 - ((intptr_t)Buffer & 3);   // Make sure it's aligned
    ParseBegin = Buffer + Offset;
    memcpy((unsigned char*)ParseBegin, Buf, Length);    // Copy it over
    MustDelete = true;
  } else {
    // If we don't need to copy it over, just use the caller's copy
    ParseBegin = Buffer = Buf;
    MustDelete = false;
  }
  try {
    ParseBytecode(ParseBegin, Length, ModuleID);
  } catch (...) {
    if (MustDelete) delete [] Buffer;
    throw;
  }
}

BytecodeBufferReader::~BytecodeBufferReader() {
  if (MustDelete) delete [] Buffer;
}

//===----------------------------------------------------------------------===//
//  BytecodeStdinReader - Read bytecode from Standard Input
//

namespace {
  /// BytecodeStdinReader - parses a bytecode file from stdin
  /// 
  class BytecodeStdinReader : public BytecodeParser {
  private:
    std::vector<unsigned char> FileData;
    unsigned char *FileBuf;

    BytecodeStdinReader(const BytecodeStdinReader&); // Do not implement
    void operator=(const BytecodeStdinReader &BFR);  // Do not implement

  public:
    BytecodeStdinReader();
  };
}

BytecodeStdinReader::BytecodeStdinReader() {
  int BlockSize;
  unsigned char Buffer[4096*4];

  // Read in all of the data from stdin, we cannot mmap stdin...
  while ((BlockSize = read(0 /*stdin*/, Buffer, 4096*4))) {
    if (BlockSize == -1)
      throw std::string("Error reading from stdin!");
    
    FileData.insert(FileData.end(), Buffer, Buffer+BlockSize);
  }

  if (FileData.empty())
    throw std::string("Standard Input empty!");

  FileBuf = &FileData[0];
  ParseBytecode(FileBuf, FileData.size(), "<stdin>");
}

//===----------------------------------------------------------------------===//
//  Varargs transmogrification code...
//

// CheckVarargs - This is used to automatically translate old-style varargs to
// new style varargs for backwards compatibility.
static ModuleProvider *CheckVarargs(ModuleProvider *MP) {
  Module *M = MP->getModule();
  
  // Check to see if va_start takes arguments...
  Function *F = M->getNamedFunction("llvm.va_start");
  if (F == 0) return MP;  // No varargs use, just return.

  if (F->getFunctionType()->getNumParams() == 0)
    return MP;  // Modern varargs processing, just return.

  // If we get to this point, we know that we have an old-style module.
  // Materialize the whole thing to perform the rewriting.
  MP->materializeModule();

  // If the user is making use of obsolete varargs intrinsics, adjust them for
  // the user.
  if (Function *F = M->getNamedFunction("llvm.va_start")) {
    assert(F->asize() == 1 && "Obsolete va_start takes 1 argument!");
        
    const Type *RetTy = F->getFunctionType()->getParamType(0);
    RetTy = cast<PointerType>(RetTy)->getElementType();
    Function *NF = M->getOrInsertFunction("llvm.va_start", RetTy, 0);
        
    for (Value::use_iterator I = F->use_begin(), E = F->use_end(); I != E; )
      if (CallInst *CI = dyn_cast<CallInst>(*I++)) {
        Value *V = new CallInst(NF, "", CI);
        new StoreInst(V, CI->getOperand(1), CI);
        CI->getParent()->getInstList().erase(CI);
      }
    F->setName("");
  }

  if (Function *F = M->getNamedFunction("llvm.va_end")) {
    assert(F->asize() == 1 && "Obsolete va_end takes 1 argument!");
    const Type *ArgTy = F->getFunctionType()->getParamType(0);
    ArgTy = cast<PointerType>(ArgTy)->getElementType();
    Function *NF = M->getOrInsertFunction("llvm.va_end", Type::VoidTy,
                                                  ArgTy, 0);
        
    for (Value::use_iterator I = F->use_begin(), E = F->use_end(); I != E; )
      if (CallInst *CI = dyn_cast<CallInst>(*I++)) {
        Value *V = new LoadInst(CI->getOperand(1), "", CI);
        new CallInst(NF, V, "", CI);
        CI->getParent()->getInstList().erase(CI);
      }
    F->setName("");
  }
      
  if (Function *F = M->getNamedFunction("llvm.va_copy")) {
    assert(F->asize() == 2 && "Obsolete va_copy takes 2 argument!");
    const Type *ArgTy = F->getFunctionType()->getParamType(0);
    ArgTy = cast<PointerType>(ArgTy)->getElementType();
    Function *NF = M->getOrInsertFunction("llvm.va_copy", ArgTy,
                                                  ArgTy, 0);
        
    for (Value::use_iterator I = F->use_begin(), E = F->use_end(); I != E; )
      if (CallInst *CI = dyn_cast<CallInst>(*I++)) {
        Value *V = new CallInst(NF, CI->getOperand(2), "", CI);
        new StoreInst(V, CI->getOperand(1), CI);
        CI->getParent()->getInstList().erase(CI);
      }
    F->setName("");
  }
  return MP;
}


//===----------------------------------------------------------------------===//
// Wrapper functions
//===----------------------------------------------------------------------===//

/// getBytecodeBufferModuleProvider - lazy function-at-a-time loading from a
/// buffer
ModuleProvider* 
getBytecodeBufferModuleProvider(const unsigned char *Buffer, unsigned Length,
                                const std::string &ModuleID) {
  return CheckVarargs(new BytecodeBufferReader(Buffer, Length, ModuleID));
}

/// ParseBytecodeBuffer - Parse a given bytecode buffer
///
Module *ParseBytecodeBuffer(const unsigned char *Buffer, unsigned Length,
                            const std::string &ModuleID, std::string *ErrorStr){
  try {
    std::auto_ptr<ModuleProvider>
      AMP(getBytecodeBufferModuleProvider(Buffer, Length, ModuleID));
    return AMP->releaseModule();
  } catch (std::string &err) {
    if (ErrorStr) *ErrorStr = err;
    return 0;
  }
}

/// getBytecodeModuleProvider - lazy function-at-a-time loading from a file
///
ModuleProvider *getBytecodeModuleProvider(const std::string &Filename) {
  if (Filename != std::string("-"))        // Read from a file...
    return CheckVarargs(new BytecodeFileReader(Filename));
  else                                     // Read from stdin
    return CheckVarargs(new BytecodeStdinReader());
}

/// ParseBytecodeFile - Parse the given bytecode file
///
Module *ParseBytecodeFile(const std::string &Filename, std::string *ErrorStr) {
  try {
    std::auto_ptr<ModuleProvider> AMP(getBytecodeModuleProvider(Filename));
    return AMP->releaseModule();
  } catch (std::string &err) {
    if (ErrorStr) *ErrorStr = err;
    return 0;
  }
}
