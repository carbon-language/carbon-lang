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
#include "llvm/Bitcode/BitstreamReader.h"
#include "llvm/Bitcode/LLVMBitCodes.h"
#include "llvm/ADT/DenseMap.h"
#include <vector>

namespace llvm {
  class MemoryBuffer;
  
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
  MemoryBuffer *Buffer;
  BitstreamReader Stream;
  
  const char *ErrorString;
  
  std::vector<PATypeHolder> TypeList;
  BitcodeReaderValueList ValueList;
  std::vector<std::pair<GlobalVariable*, unsigned> > GlobalInits;
  std::vector<std::pair<GlobalAlias*, unsigned> > AliasInits;
  
  // When reading the module header, this list is populated with functions that
  // have bodies later in the file.
  std::vector<Function*> FunctionsWithBodies;
  
  // After the module header has been read, the FunctionsWithBodies list is 
  // reversed.  This keeps track of whether we've done this yet.
  bool HasReversedFunctionsWithBodies;
  
  /// DeferredFunctionInfo - When function bodies are initially scanned, this
  /// map contains info about where to find deferred function body (in the
  /// stream) and what linkage the original function had.
  DenseMap<Function*, std::pair<uint64_t, unsigned> > DeferredFunctionInfo;
public:
  BitcodeReader(MemoryBuffer *buffer) : Buffer(buffer), ErrorString(0) {
    HasReversedFunctionsWithBodies = false;
  }
  ~BitcodeReader();
  
  
  /// releaseMemoryBuffer - This causes the reader to completely forget about
  /// the memory buffer it contains, which prevents the buffer from being
  /// destroyed when it is deleted.
  void releaseMemoryBuffer() {
    Buffer = 0;
  }
  
  virtual bool materializeFunction(Function *F, std::string *ErrInfo = 0);
  
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
  bool ParseBitcode();
private:
  const Type *getTypeByID(unsigned ID, bool isTypeTable = false);
  
  bool ParseModule(const std::string &ModuleID);
  bool ParseTypeTable();
  bool ParseTypeSymbolTable();
  bool ParseValueSymbolTable();
  bool ParseConstants();
  bool ParseFunction();
  bool ResolveGlobalAndAliasInits();
};
  
} // End llvm namespace

#endif
