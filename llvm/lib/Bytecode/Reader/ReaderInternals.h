//===-- ReaderInternals.h - Definitions internal to the reader --*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
//  This header file defines various stuff that is used by the bytecode reader.
//
//===----------------------------------------------------------------------===//

#ifndef READER_INTERNALS_H
#define READER_INTERNALS_H

#include "ReaderPrimitives.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/ModuleProvider.h"
#include <utility>
#include <map>

namespace llvm {

// Enable to trace to figure out what the heck is going on when parsing fails
//#define TRACE_LEVEL 10
//#define DEBUG_OUTPUT

#if TRACE_LEVEL    // ByteCodeReading_TRACEr
#define BCR_TRACE(n, X) \
    if (n < TRACE_LEVEL) std::cerr << std::string(n*2, ' ') << X
#else
#define BCR_TRACE(n, X)
#endif

struct LazyFunctionInfo {
  const unsigned char *Buf, *EndBuf;
  LazyFunctionInfo(const unsigned char *B = 0, const unsigned char *EB = 0)
    : Buf(B), EndBuf(EB) {}
};

class BytecodeParser : public ModuleProvider {
  BytecodeParser(const BytecodeParser &);  // DO NOT IMPLEMENT
  void operator=(const BytecodeParser &);  // DO NOT IMPLEMENT
public:
  BytecodeParser() {}
  
  ~BytecodeParser() {
    freeState();
  }
  void freeState() {
    freeTable(Values);
    freeTable(ModuleValues);
  }

  Module* materializeModule() {
    while (! LazyFunctionLoadMap.empty()) {
      std::map<Function*, LazyFunctionInfo>::iterator i = 
        LazyFunctionLoadMap.begin();
      materializeFunction((*i).first);
    }

    return TheModule;
  }

  Module* releaseModule() {
    // Since we're losing control of this Module, we must hand it back complete
    Module *M = ModuleProvider::releaseModule();
    freeState();
    return M;
  }

  void ParseBytecode(const unsigned char *Buf, unsigned Length,
                     const std::string &ModuleID);

  void dump() const {
    std::cerr << "BytecodeParser instance!\n";
  }

private:
  struct ValueList : public User {
    ValueList() : User(Type::TypeTy, Value::TypeVal) {}

    // vector compatibility methods
    unsigned size() const { return getNumOperands(); }
    void push_back(Value *V) { Operands.push_back(Use(V, this)); }
    Value *back() const { return Operands.back(); }
    void pop_back() { Operands.pop_back(); }
    bool empty() const { return Operands.empty(); }

    virtual void print(std::ostream& OS) const {
      OS << "Bytecode Reader UseHandle!";
    }
  };

  // Information about the module, extracted from the bytecode revision number.
  unsigned char RevisionNum;        // The rev # itself

  // Flags to distinguish LLVM 1.0 & 1.1 bytecode formats (revision #0)

  // Revision #0 had an explicit alignment of data only for the ModuleGlobalInfo
  // block.  This was fixed to be like all other blocks in 1.2
  bool hasInconsistentModuleGlobalInfo;

  // Revision #0 also explicitly encoded zero values for primitive types like
  // int/sbyte/etc.
  bool hasExplicitPrimitiveZeros;

  // Flags to control features specific the LLVM 1.2 and before (revision #1)

  // LLVM 1.2 and earlier required that getelementptr structure indices were
  // ubyte constants and that sequential type indices were longs.
  bool hasRestrictedGEPTypes;


  typedef std::vector<ValueList*> ValueTable;
  ValueTable Values;
  ValueTable ModuleValues;
  std::map<std::pair<unsigned,unsigned>, Value*> ForwardReferences;

  /// CompactionTable - If a compaction table is active in the current function,
  /// this is the mapping that it contains.
  std::vector<std::vector<Value*> > CompactionTable;

  std::vector<BasicBlock*> ParsedBasicBlocks;

  // ConstantFwdRefs - This maintains a mapping between <Type, Slot #>'s and
  // forward references to constants.  Such values may be referenced before they
  // are defined, and if so, the temporary object that they represent is held
  // here.
  //
  typedef std::map<std::pair<const Type*,unsigned>, Constant*> ConstantRefsType;
  ConstantRefsType ConstantFwdRefs;

  // TypesLoaded - This vector mirrors the Values[TypeTyID] plane.  It is used
  // to deal with forward references to types.
  //
  typedef std::vector<PATypeHolder> TypeValuesListTy;
  TypeValuesListTy ModuleTypeValues;
  TypeValuesListTy FunctionTypeValues;

  // When the ModuleGlobalInfo section is read, we create a function object for
  // each function in the module.  When the function is loaded, this function is
  // filled in.
  //
  std::vector<Function*> FunctionSignatureList;

  // Constant values are read in after global variables.  Because of this, we
  // must defer setting the initializers on global variables until after module
  // level constants have been read.  In the mean time, this list keeps track of
  // what we must do.
  //
  std::vector<std::pair<GlobalVariable*, unsigned> > GlobalInits;

  // For lazy reading-in of functions, we need to save away several pieces of
  // information about each function: its begin and end pointer in the buffer
  // and its FunctionSlot.
  // 
  std::map<Function*, LazyFunctionInfo> LazyFunctionLoadMap;
  
private:
  void freeTable(ValueTable &Tab) {
    while (!Tab.empty()) {
      delete Tab.back();
      Tab.pop_back();
    }
  }

  /// getGlobalTableType - This is just like getType, but when a compaction
  /// table is in use, it is ignored.  Also, no forward references or other
  /// fancy features are supported.
  const Type *getGlobalTableType(unsigned Slot) {
    if (Slot < Type::FirstDerivedTyID) {
      const Type *Ty = Type::getPrimitiveType((Type::TypeID)Slot);
      assert(Ty && "Not a primitive type ID?");
      return Ty;
    }
    Slot -= Type::FirstDerivedTyID;
    if (Slot >= ModuleTypeValues.size())
      throw std::string("Illegal compaction table type reference!");
    return ModuleTypeValues[Slot];
  }

  unsigned getGlobalTableTypeSlot(const Type *Ty) {
    if (Ty->isPrimitiveType())
      return Ty->getTypeID();
    TypeValuesListTy::iterator I = find(ModuleTypeValues.begin(),
                                        ModuleTypeValues.end(), Ty);
    if (I == ModuleTypeValues.end())
      throw std::string("Didn't find type in ModuleTypeValues.");
    return Type::FirstDerivedTyID + (&*I - &ModuleTypeValues[0]);
  }

  /// getGlobalTableValue - This is just like getValue, but when a compaction
  /// table is in use, it is ignored.  Also, no forward references or other
  /// fancy features are supported.
  Value *getGlobalTableValue(const Type *Ty, unsigned SlotNo) {
    // FIXME: getTypeSlot is inefficient!
    unsigned TyID = getGlobalTableTypeSlot(Ty);
    
    if (TyID != Type::LabelTyID) {
      if (SlotNo == 0)
        return Constant::getNullValue(Ty);
      --SlotNo;
    }

    if (TyID >= ModuleValues.size() || ModuleValues[TyID] == 0 ||
        SlotNo >= ModuleValues[TyID]->getNumOperands()) {
      std::cerr << TyID << ", " << SlotNo << ": " << ModuleValues.size() << ", "
                << (void*)ModuleValues[TyID] << ", "
                << ModuleValues[TyID]->getNumOperands() << "\n";
      throw std::string("Corrupt compaction table entry!");
    }
    return ModuleValues[TyID]->getOperand(SlotNo);
  }

public:
  void ParseModule(const unsigned char * Buf, const unsigned char *End);
  void materializeFunction(Function *F);

private:
  void ParseVersionInfo   (const unsigned char *&Buf, const unsigned char *End);
  void ParseModuleGlobalInfo(const unsigned char *&Buf, const unsigned char *E);
  void ParseSymbolTable(const unsigned char *&Buf, const unsigned char *End,
                        SymbolTable *, Function *CurrentFunction);
  void ParseFunction(const unsigned char *&Buf, const unsigned char *End);
  void ParseCompactionTable(const unsigned char *&Buf,const unsigned char *End);
  void ParseGlobalTypes(const unsigned char *&Buf, const unsigned char *EndBuf);

  BasicBlock *ParseBasicBlock(const unsigned char *&Buf,
                              const unsigned char *End,
                              unsigned BlockNo);
  unsigned ParseInstructionList(Function *F, const unsigned char *&Buf,
                                const unsigned char *EndBuf);
  
  void ParseInstruction(const unsigned char *&Buf, const unsigned char *End,
                        std::vector<unsigned> &Args, BasicBlock *BB);

  void ParseConstantPool(const unsigned char *&Buf, const unsigned char *EndBuf,
                         ValueTable &Tab, TypeValuesListTy &TypeTab);
  Constant *parseConstantValue(const unsigned char *&Buf,
                               const unsigned char *End,
                               unsigned TypeID);
  void parseTypeConstants(const unsigned char *&Buf,
                          const unsigned char *EndBuf,
                          TypeValuesListTy &Tab, unsigned NumEntries);
  const Type *parseTypeConstant(const unsigned char *&Buf,
                                const unsigned char *EndBuf);
  void parseStringConstants(const unsigned char *&Buf,
                            const unsigned char *EndBuf,
                            unsigned NumEntries, ValueTable &Tab);

  Value      *getValue(unsigned TypeID, unsigned num, bool Create = true);
  const Type *getType(unsigned ID);
  BasicBlock *getBasicBlock(unsigned ID);
  Constant   *getConstantValue(unsigned TypeID, unsigned num);
  Constant   *getConstantValue(const Type *Ty, unsigned num) {
    return getConstantValue(getTypeSlot(Ty), num);
  }

  unsigned insertValue(Value *V, unsigned Type, ValueTable &Table);

  unsigned getTypeSlot(const Type *Ty);

  // resolve all references to the placeholder (if any) for the given constant
  void ResolveReferencesToConstant(Constant *C, unsigned Slot);
};

template<class SuperType>
class PlaceholderDef : public SuperType {
  unsigned ID;
  PlaceholderDef();                       // DO NOT IMPLEMENT
  void operator=(const PlaceholderDef &); // DO NOT IMPLEMENT
public:
  PlaceholderDef(const Type *Ty, unsigned id) : SuperType(Ty), ID(id) {}
  unsigned getID() { return ID; }
};

struct ConstantPlaceHolderHelper : public ConstantExpr {
  ConstantPlaceHolderHelper(const Type *Ty)
    : ConstantExpr(Instruction::UserOp1, Constant::getNullValue(Ty), Ty) {}
};

typedef PlaceholderDef<ConstantPlaceHolderHelper>  ConstPHolder;

static inline void readBlock(const unsigned char *&Buf,
                             const unsigned char *EndBuf, 
                             unsigned &Type, unsigned &Size) {
  Type = read(Buf, EndBuf);
  Size = read(Buf, EndBuf);
}

} // End llvm namespace

#endif
