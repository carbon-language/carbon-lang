//===-- Parser.h - Definitions internal to the reader -----------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencer and is distributed under the 
// University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
//  This header file defines the interface to the Bytecode Parser
//
//===----------------------------------------------------------------------===//

#ifndef BYTECODE_PARSER_H
#define BYTECODE_PARSER_H

#include "ReaderPrimitives.h"
#include "BytecodeHandler.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include <utility>
#include <vector>
#include <map>

namespace llvm {

struct LazyFunctionInfo {
  const unsigned char *Buf, *EndBuf;
  LazyFunctionInfo(const unsigned char *B = 0, const unsigned char *EB = 0)
    : Buf(B), EndBuf(EB) {}
};

typedef std::map<const Type*, LazyFunctionInfo> LazyFunctionMap;

class AbstractBytecodeParser {
  AbstractBytecodeParser(const AbstractBytecodeParser &);  // DO NOT IMPLEMENT
  void operator=(const AbstractBytecodeParser &);  // DO NOT IMPLEMENT
public:
  AbstractBytecodeParser( BytecodeHandler* h ) { handler = h; }
  ~AbstractBytecodeParser() { }

  void ParseBytecode(const unsigned char *Buf, unsigned Length,
                     const std::string &ModuleID);

  void dump() const {
    std::cerr << "AbstractBytecodeParser instance!\n";
  }

private:
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


  /// CompactionTable - If a compaction table is active in the current function,
  /// this is the mapping that it contains.
  std::vector<Type*> CompactionTypeTable;

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
  typedef std::vector<PATypeHolder> TypeListTy;
  TypeListTy ModuleTypes;
  TypeListTy FunctionTypes;

  // When the ModuleGlobalInfo section is read, we create a FunctionType object
  // for each function in the module. When the function is loaded, this type is
  // used to instantiate the actual function object.
  std::vector<const Type*> FunctionSignatureList;

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
  LazyFunctionMap LazyFunctionLoadMap;

  /// The handler for parsing
  BytecodeHandler* handler;
  
private:
  const Type *AbstractBytecodeParser::getType(unsigned ID);
  /// getGlobalTableType - This is just like getType, but when a compaction
  /// table is in use, it is ignored.  Also, no forward references or other
  /// fancy features are supported.
  const Type *getGlobalTableType(unsigned Slot) {
    if (Slot < Type::FirstDerivedTyID) {
      const Type *Ty = Type::getPrimitiveType((Type::PrimitiveID)Slot);
      assert(Ty && "Not a primitive type ID?");
      return Ty;
    }
    Slot -= Type::FirstDerivedTyID;
    if (Slot >= ModuleTypes.size())
      throw std::string("Illegal compaction table type reference!");
    return ModuleTypes[Slot];
  }

  unsigned getGlobalTableTypeSlot(const Type *Ty) {
    if (Ty->isPrimitiveType())
      return Ty->getPrimitiveID();
    TypeListTy::iterator I = find(ModuleTypes.begin(),
                                        ModuleTypes.end(), Ty);
    if (I == ModuleTypes.end())
      throw std::string("Didn't find type in ModuleTypes.");
    return Type::FirstDerivedTyID + (&*I - &ModuleTypes[0]);
  }

public:
  typedef const unsigned char* BufPtr;
  void ParseModule             (BufPtr &Buf, BufPtr End);
  void ParseNextFunction       (Type* FType) ;
  void ParseAllFunctionBodies  ();

private:
  void ParseVersionInfo        (BufPtr &Buf, BufPtr End);
  void ParseModuleGlobalInfo   (BufPtr &Buf, BufPtr End);
  void ParseSymbolTable        (BufPtr &Buf, BufPtr End);
  void ParseFunctionLazily     (BufPtr &Buf, BufPtr End);
  void ParseFunctionBody       (const Type* FType, BufPtr &Buf, BufPtr EndBuf);
  void ParseCompactionTable    (BufPtr &Buf, BufPtr End);
  void ParseGlobalTypes        (BufPtr &Buf, BufPtr End);

  void ParseBasicBlock         (BufPtr &Buf, BufPtr End, unsigned BlockNo);
  unsigned ParseInstructionList(BufPtr &Buf, BufPtr End);
  
  bool ParseInstruction        (BufPtr &Buf, BufPtr End, 
	                        std::vector<unsigned>& Args);

  void ParseConstantPool       (BufPtr &Buf, BufPtr End, TypeListTy& List);
  void ParseConstantValue      (BufPtr &Buf, BufPtr End, unsigned TypeID);
  void ParseTypeConstants      (BufPtr &Buf, BufPtr End, TypeListTy &Tab,
					unsigned NumEntries);
  const Type *ParseTypeConstant(BufPtr &Buf, BufPtr End);
  void ParseStringConstants    (BufPtr &Buf, BufPtr End, unsigned NumEntries);

};


static inline void readBlock(const unsigned char *&Buf,
                             const unsigned char *EndBuf, 
                             unsigned &Type, unsigned &Size) {
  Type = read(Buf, EndBuf);
  Size = read(Buf, EndBuf);
}

} // End llvm namespace

#endif
// vim: sw=2
