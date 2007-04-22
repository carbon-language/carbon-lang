//===- ReaderWrappers.cpp - Parse bitcode from file or buffer -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements loading and parsing a bitcode file and parsing a
// module from a memory buffer.
//
//===----------------------------------------------------------------------===//

#include "llvm/Bitcode/ReaderWriter.h"
#include "BitcodeReader.h"
#include "llvm/System/MappedFile.h"
using namespace llvm;

//===----------------------------------------------------------------------===//
// BitcodeFileReader - Read from an mmap'able file descriptor.

namespace {
  /// BitcodeFileReader - parses bitcode from a file.
  ///
  class BitcodeFileReader : public BitcodeReader {
  private:
    std::string Filename;
    sys::MappedFile File;
    
    BitcodeFileReader(const BitcodeFileReader&); // DO NOT IMPLEMENT
    void operator=(const BitcodeFileReader&); // DO NOT IMPLEMENT
  public:
    BitcodeFileReader(const std::string &FN) : Filename(FN) {}
    bool Read(std::string *ErrMsg);
    
    void FreeState() {
      BitcodeReader::FreeState();
      File.close();
    }
  };
}

bool BitcodeFileReader::Read(std::string *ErrMsg) {
  if (File.open(sys::Path(Filename), sys::MappedFile::READ_ACCESS, ErrMsg))
    return true;
  if (!File.map(ErrMsg)) {
    File.close();
    return true;
  }
  unsigned char *Buffer = reinterpret_cast<unsigned char*>(File.base());
  if (!ParseBitcode(Buffer, File.size(), Filename))
    return false;
  if (ErrMsg) *ErrMsg = getErrorString();
  return true;
}



//===----------------------------------------------------------------------===//
// External interface
//===----------------------------------------------------------------------===//

/// getBitcodeModuleProvider - lazy function-at-a-time loading from a file.
///
ModuleProvider *llvm::getBitcodeModuleProvider(const std::string &Filename,
                                               std::string *ErrMsg) {
  if (Filename != std::string("-")) {
    BitcodeFileReader *R = new BitcodeFileReader(Filename);
    if (R->Read(ErrMsg)) {
      delete R;
      return 0;
    }
    return R;
  }
  
  assert(0 && "FIXME: stdin reading unimp!");
#if 0
  // Read from stdin
  BytecodeStdinReader *R = new BytecodeStdinReader();
  if (R->Read(ErrMsg)) {
    delete R;
    return 0;
  }
  return R;
#endif
}

/// ParseBitcodeFile - Read the specified bitcode file, returning the module.
/// If an error occurs, return null and fill in *ErrMsg if non-null.
Module *llvm::ParseBitcodeFile(const std::string &Filename,std::string *ErrMsg){
  ModuleProvider *MP = getBitcodeModuleProvider(Filename, ErrMsg);
  if (!MP) return 0;
  Module *M = MP->releaseModule(ErrMsg);
  delete MP;
  return M;
}
