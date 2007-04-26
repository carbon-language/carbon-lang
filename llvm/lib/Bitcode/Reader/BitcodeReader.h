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

#include "llvm/ModuleProvider.h"
#include "llvm/Type.h"
#include "llvm/User.h"
#include "llvm/Bitcode/LLVMBitCodes.h"
#include <vector>

namespace llvm {
  class BitstreamReader;
  
class BitcodeReaderValueList : public User {
  std::vector<Use> Uses;
public:
  BitcodeReaderValueList() : User(Type::VoidTy, Value::ArgumentVal, 0, 0) {}
  
  // vector compatibility methods
  unsigned size() const { return getNumOperands(); }
  void push_back(Value *V) {
    Uses.push_back(Use(V, this));
    OperandList = &Uses[0];
    ++NumOperands;
  }
  
  Value *operator[](unsigned i) const { return getOperand(i); }
  
  Value *back() const { return Uses.back(); }
  void pop_back() { Uses.pop_back(); --NumOperands; }
  bool empty() const { return NumOperands == 0; }
  void shrinkTo(unsigned N) {
    assert(N < NumOperands && "Invalid shrinkTo request!");
    Uses.resize(N);
    NumOperands = N;
  }
  virtual void print(std::ostream&) const {}
  
  Constant *getConstantFwdRef(unsigned Idx, const Type *Ty);
  void initVal(unsigned Idx, Value *V) {
    assert(Uses[Idx] == 0 && "Cannot init an already init'd Use!");
    Uses[Idx].init(V, this);
  }
};
  

class BitcodeReader : public ModuleProvider {
  const char *ErrorString;
  
  std::vector<PATypeHolder> TypeList;
  BitcodeReaderValueList ValueList;
  std::vector<std::pair<GlobalVariable*, unsigned> > GlobalInits;
  std::vector<std::pair<GlobalAlias*, unsigned> > AliasInits;
public:
  BitcodeReader() : ErrorString(0) {}
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
  bool ParseValueSymbolTable(BitstreamReader &Stream);
  bool ParseConstants(BitstreamReader &Stream);
  bool ResolveGlobalAndAliasInits();
};
  
} // End llvm namespace

#endif
