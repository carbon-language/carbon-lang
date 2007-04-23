//===- BitcodeReader.h - Internal BitcodeReader impl ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License.  See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This header defines the BitcodeReader class.
//
//===----------------------------------------------------------------------===//

#ifndef BITCODE_READER_H
#define BITCODE_READER_H

#include "llvm/Type.h"
#include "llvm/ModuleProvider.h"
#include "llvm/Bitcode/LLVMBitCodes.h"
#include <vector>

namespace llvm {
  class BitstreamReader;

class BitcodeReader : public ModuleProvider {
  const char *ErrorString;
  
  std::vector<PATypeHolder> TypeList;
public:
  virtual ~BitcodeReader() {}
  
  virtual void FreeState() {}
  
  virtual bool materializeFunction(Function *F, std::string *ErrInfo = 0) {
    // FIXME: TODO
    return false;
  }
  
  virtual Module *materializeModule(std::string *ErrInfo = 0) {
    // FIXME: TODO
    //if (ParseAllFunctionBodies(ErrMsg))
    //  return 0;
    return TheModule;
  }
  
  bool Error(const char *Str) {
    ErrorString = Str;
    return true;
  }
  const char *getErrorString() const { return ErrorString; }
  
  /// @brief Main interface to parsing a bitcode buffer.
  /// @returns true if an error occurred.
  bool ParseBitcode(unsigned char *Buf, unsigned Length,
                    const std::string &ModuleID);
private:
  const Type *getTypeByID(unsigned ID, bool isTypeTable = false);
  
  bool ParseModule(BitstreamReader &Stream, const std::string &ModuleID);
  bool ParseTypeTable(BitstreamReader &Stream);
  bool ParseTypeSymbolTable(BitstreamReader &Stream);
};
  
} // End llvm namespace

#endif
