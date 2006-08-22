//===- ReaderWrappers.cpp - Parse bytecode from file or buffer  -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements loading and parsing a bytecode file and parsing a
// bytecode module from a given buffer.
//
//===----------------------------------------------------------------------===//

#include "llvm/Bytecode/Analyzer.h"
#include "llvm/Bytecode/Reader.h"
#include "Reader.h"
#include "llvm/Module.h"
#include "llvm/Instructions.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/System/MappedFile.h"
#include "llvm/System/Program.h"
#include <cerrno>
#include <iostream>
#include <memory>

using namespace llvm;

//===----------------------------------------------------------------------===//
// BytecodeFileReader - Read from an mmap'able file descriptor.
//

namespace {
  /// BytecodeFileReader - parses a bytecode file from a file
  ///
  class BytecodeFileReader : public BytecodeReader {
  private:
    sys::MappedFile mapFile;

    BytecodeFileReader(const BytecodeFileReader&); // Do not implement
    void operator=(const BytecodeFileReader &BFR); // Do not implement

  public:
    BytecodeFileReader(const std::string &Filename, llvm::BytecodeHandler* H=0);
  };
}

BytecodeFileReader::BytecodeFileReader(const std::string &Filename,
                                       llvm::BytecodeHandler* H )
  : BytecodeReader(H)
  , mapFile()
{
  std::string ErrMsg;
  if (mapFile.open(sys::Path(Filename), sys::MappedFile::READ_ACCESS, &ErrMsg))
    throw ErrMsg;
  if (!mapFile.map(&ErrMsg))
    throw ErrMsg;
  unsigned char* buffer = reinterpret_cast<unsigned char*>(mapFile.base());
  if (ParseBytecode(buffer, mapFile.size(), Filename, &ErrMsg)) {
    throw ErrMsg;
  }
}

//===----------------------------------------------------------------------===//
// BytecodeBufferReader - Read from a memory buffer
//

namespace {
  /// BytecodeBufferReader - parses a bytecode file from a buffer
  ///
  class BytecodeBufferReader : public BytecodeReader {
  private:
    const unsigned char *Buffer;
    bool MustDelete;

    BytecodeBufferReader(const BytecodeBufferReader&); // Do not implement
    void operator=(const BytecodeBufferReader &BFR);   // Do not implement

  public:
    BytecodeBufferReader(const unsigned char *Buf, unsigned Length,
                         const std::string &ModuleID,
                         llvm::BytecodeHandler* Handler = 0);
    ~BytecodeBufferReader();

  };
}

BytecodeBufferReader::BytecodeBufferReader(const unsigned char *Buf,
                                           unsigned Length,
                                           const std::string &ModuleID,
                                           llvm::BytecodeHandler* H )
  : BytecodeReader(H)
{
  // If not aligned, allocate a new buffer to hold the bytecode...
  const unsigned char *ParseBegin = 0;
  if (reinterpret_cast<uint64_t>(Buf) & 3) {
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
  std::string ErrMsg;
  if (ParseBytecode(ParseBegin, Length, ModuleID, &ErrMsg)) {
    if (MustDelete) delete [] Buffer;
    throw ErrMsg;
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
  class BytecodeStdinReader : public BytecodeReader {
  private:
    std::vector<unsigned char> FileData;
    unsigned char *FileBuf;

    BytecodeStdinReader(const BytecodeStdinReader&); // Do not implement
    void operator=(const BytecodeStdinReader &BFR);  // Do not implement

  public:
    BytecodeStdinReader( llvm::BytecodeHandler* H = 0 );
  };
}

BytecodeStdinReader::BytecodeStdinReader( BytecodeHandler* H )
  : BytecodeReader(H)
{
  sys::Program::ChangeStdinToBinary();
  char Buffer[4096*4];

  // Read in all of the data from stdin, we cannot mmap stdin...
  while (std::cin.good()) {
    std::cin.read(Buffer, 4096*4);
    int BlockSize = std::cin.gcount();
    if (0 >= BlockSize)
      break;
    FileData.insert(FileData.end(), Buffer, Buffer+BlockSize);
  }

  if (FileData.empty())
    throw std::string("Standard Input empty!");

  FileBuf = &FileData[0];
  std::string ErrMsg;
  if (ParseBytecode(FileBuf, FileData.size(), "<stdin>", &ErrMsg)) {
    throw ErrMsg;
  }
}

//===----------------------------------------------------------------------===//
// Varargs transmogrification code...
//

// CheckVarargs - This is used to automatically translate old-style varargs to
// new style varargs for backwards compatibility.
static ModuleProvider* CheckVarargs(ModuleProvider* MP) {
  Module* M = MP->getModule();

  // check to see if va_start takes arguements...
  Function* F = M->getNamedFunction("llvm.va_start");
  if(F == 0) return MP; //No varargs use, just return.

  if (F->getFunctionType()->getNumParams() == 1)
    return MP; // Modern varargs processing, just return.

  // If we get to this point, we know that we have an old-style module.
  // Materialize the whole thing to perform the rewriting.
  if (MP->materializeModule() == 0)
    return 0;

  if(Function* F = M->getNamedFunction("llvm.va_start")) {
    assert(F->arg_size() == 0 && "Obsolete va_start takes 0 argument!");

    //foo = va_start()
    // ->
    //bar = alloca typeof(foo)
    //va_start(bar)
    //foo = load bar

    const Type* RetTy = Type::getPrimitiveType(Type::VoidTyID);
    const Type* ArgTy = F->getFunctionType()->getReturnType();
    const Type* ArgTyPtr = PointerType::get(ArgTy);
    Function* NF = M->getOrInsertFunction("llvm.va_start",
                                          RetTy, ArgTyPtr, (Type *)0);

    for(Value::use_iterator I = F->use_begin(), E = F->use_end(); I != E;)
      if (CallInst* CI = dyn_cast<CallInst>(*I++)) {
        AllocaInst* bar = new AllocaInst(ArgTy, 0, "vastart.fix.1", CI);
        new CallInst(NF, bar, "", CI);
        Value* foo = new LoadInst(bar, "vastart.fix.2", CI);
        CI->replaceAllUsesWith(foo);
        CI->getParent()->getInstList().erase(CI);
      }
    F->setName("");
  }

  if(Function* F = M->getNamedFunction("llvm.va_end")) {
    assert(F->arg_size() == 1 && "Obsolete va_end takes 1 argument!");
    //vaend foo
    // ->
    //bar = alloca 1 of typeof(foo)
    //vaend bar
    const Type* RetTy = Type::getPrimitiveType(Type::VoidTyID);
    const Type* ArgTy = F->getFunctionType()->getParamType(0);
    const Type* ArgTyPtr = PointerType::get(ArgTy);
    Function* NF = M->getOrInsertFunction("llvm.va_end",
                                          RetTy, ArgTyPtr, (Type *)0);

    for(Value::use_iterator I = F->use_begin(), E = F->use_end(); I != E;)
      if (CallInst* CI = dyn_cast<CallInst>(*I++)) {
        AllocaInst* bar = new AllocaInst(ArgTy, 0, "vaend.fix.1", CI);
        new StoreInst(CI->getOperand(1), bar, CI);
        new CallInst(NF, bar, "", CI);
        CI->getParent()->getInstList().erase(CI);
      }
    F->setName("");
  }

  if(Function* F = M->getNamedFunction("llvm.va_copy")) {
    assert(F->arg_size() == 1 && "Obsolete va_copy takes 1 argument!");
    //foo = vacopy(bar)
    // ->
    //a = alloca 1 of typeof(foo)
    //b = alloca 1 of typeof(foo)
    //store bar -> b
    //vacopy(a, b)
    //foo = load a

    const Type* RetTy = Type::getPrimitiveType(Type::VoidTyID);
    const Type* ArgTy = F->getFunctionType()->getReturnType();
    const Type* ArgTyPtr = PointerType::get(ArgTy);
    Function* NF = M->getOrInsertFunction("llvm.va_copy",
                                          RetTy, ArgTyPtr, ArgTyPtr, (Type *)0);

    for(Value::use_iterator I = F->use_begin(), E = F->use_end(); I != E;)
      if (CallInst* CI = dyn_cast<CallInst>(*I++)) {
        AllocaInst* a = new AllocaInst(ArgTy, 0, "vacopy.fix.1", CI);
        AllocaInst* b = new AllocaInst(ArgTy, 0, "vacopy.fix.2", CI);
        new StoreInst(CI->getOperand(1), b, CI);
        new CallInst(NF, a, b, "", CI);
        Value* foo = new LoadInst(a, "vacopy.fix.3", CI);
        CI->replaceAllUsesWith(foo);
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
llvm::getBytecodeBufferModuleProvider(const unsigned char *Buffer,
                                      unsigned Length,
                                      const std::string &ModuleID,
                                      BytecodeHandler* H ) {
  return CheckVarargs(
     new BytecodeBufferReader(Buffer, Length, ModuleID, H));
}

/// ParseBytecodeBuffer - Parse a given bytecode buffer
///
Module *llvm::ParseBytecodeBuffer(const unsigned char *Buffer, unsigned Length,
                                  const std::string &ModuleID,
                                  std::string *ErrorStr){
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
ModuleProvider *llvm::getBytecodeModuleProvider(const std::string &Filename,
                                                BytecodeHandler* H) {
  if (Filename != std::string("-"))        // Read from a file...
    return CheckVarargs(new BytecodeFileReader(Filename,H));
  else                                     // Read from stdin
    return CheckVarargs(new BytecodeStdinReader(H));
}

/// ParseBytecodeFile - Parse the given bytecode file
///
Module *llvm::ParseBytecodeFile(const std::string &Filename,
                                std::string *ErrorStr) {
  try {
    std::auto_ptr<ModuleProvider> AMP(getBytecodeModuleProvider(Filename));
    return AMP->releaseModule();
  } catch (std::string &err) {
    if (ErrorStr) *ErrorStr = err;
    return 0;
  }
}

// AnalyzeBytecodeFile - analyze one file
Module* llvm::AnalyzeBytecodeFile(
  const std::string &Filename,  ///< File to analyze
  BytecodeAnalysis& bca,        ///< Statistical output
  std::string *ErrorStr,        ///< Error output
  std::ostream* output          ///< Dump output
)
{
  try {
    BytecodeHandler* analyzerHandler =createBytecodeAnalyzerHandler(bca,output);
    std::auto_ptr<ModuleProvider> AMP(
      getBytecodeModuleProvider(Filename,analyzerHandler));
    return AMP->releaseModule();
  } catch (std::string &err) {
    if (ErrorStr) *ErrorStr = err;
    return 0;
  }
}

// AnalyzeBytecodeBuffer - analyze a buffer
Module* llvm::AnalyzeBytecodeBuffer(
  const unsigned char* Buffer, ///< Pointer to start of bytecode buffer
  unsigned Length,             ///< Size of the bytecode buffer
  const std::string& ModuleID, ///< Identifier for the module
  BytecodeAnalysis& bca,       ///< The results of the analysis
  std::string* ErrorStr,       ///< Errors, if any.
  std::ostream* output         ///< Dump output, if any
)
{
  try {
    BytecodeHandler* hdlr = createBytecodeAnalyzerHandler(bca, output);
    std::auto_ptr<ModuleProvider>
      AMP(getBytecodeBufferModuleProvider(Buffer, Length, ModuleID, hdlr));
    return AMP->releaseModule();
  } catch (std::string &err) {
    if (ErrorStr) *ErrorStr = err;
    return 0;
  }
}

bool llvm::GetBytecodeDependentLibraries(const std::string &fname,
                                         Module::LibraryListType& deplibs) {
  try {
    std::auto_ptr<ModuleProvider> AMP( getBytecodeModuleProvider(fname));
    Module* M = AMP->releaseModule();

    deplibs = M->getLibraries();
    delete M;
    return true;
  } catch (...) {
    deplibs.clear();
    return false;
  }
}

static void getSymbols(Module*M, std::vector<std::string>& symbols) {
  // Loop over global variables
  for (Module::global_iterator GI = M->global_begin(), GE=M->global_end(); GI != GE; ++GI)
    if (!GI->isExternal() && !GI->hasInternalLinkage())
      if (!GI->getName().empty())
        symbols.push_back(GI->getName());

  // Loop over functions.
  for (Module::iterator FI = M->begin(), FE = M->end(); FI != FE; ++FI)
    if (!FI->isExternal() && !FI->hasInternalLinkage())
      if (!FI->getName().empty())
        symbols.push_back(FI->getName());
}

// Get just the externally visible defined symbols from the bytecode
bool llvm::GetBytecodeSymbols(const sys::Path& fName,
                              std::vector<std::string>& symbols) {
  std::auto_ptr<ModuleProvider> AMP(
      getBytecodeModuleProvider(fName.toString()));

  // Get the module from the provider
  Module* M = AMP->materializeModule();
  if (M == 0) return false;

  // Get the symbols
  getSymbols(M, symbols);

  // Done with the module
  return true;
}

ModuleProvider*
llvm::GetBytecodeSymbols(const unsigned char*Buffer, unsigned Length,
                         const std::string& ModuleID,
                         std::vector<std::string>& symbols) {

  ModuleProvider* MP = 0;
  try {
    // Get the module provider
    MP = getBytecodeBufferModuleProvider(Buffer, Length, ModuleID);

    // Get the module from the provider
    Module* M = MP->materializeModule();
    if (M == 0) return 0;

    // Get the symbols
    getSymbols(M, symbols);

    // Done with the module. Note that ModuleProvider will delete the
    // Module when it is deleted. Also note that its the caller's responsibility
    // to delete the ModuleProvider.
    return MP;

  } catch (...) {
    // We delete only the ModuleProvider here because its destructor will
    // also delete the Module (we used materializeModule not releaseModule).
    delete MP;
  }
  return 0;
}
