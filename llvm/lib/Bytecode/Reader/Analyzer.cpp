//===-- BytecodeHandler.cpp - Parsing Handler -------------------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencer and is distributed under the 
// University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
//  This header file defines the BytecodeHandler class that gets called by the
//  AbstractBytecodeParser when parsing events occur.
//
//===----------------------------------------------------------------------===//

#include "AnalyzerInternals.h"

using namespace llvm;


namespace {

class AnalyzerHandler : public BytecodeHandler {
public:
  bool handleError(const std::string& str )
  {
    return false;
  }

  void handleStart()
  {
  }

  void handleFinish()
  {
  }

  void handleModuleBegin(const std::string& id)
  {
  }

  void handleModuleEnd(const std::string& id)
  {
  }

  void handleVersionInfo(
    unsigned char RevisionNum,        ///< Byte code revision number
    Module::Endianness Endianness,    ///< Endianness indicator
    Module::PointerSize PointerSize   ///< PointerSize indicator
  )
  {
  }

  void handleModuleGlobalsBegin()
  {
  }

  void handleGlobalVariable( 
    const Type* ElemType,     ///< The type of the global variable
    bool isConstant,          ///< Whether the GV is constant or not
    GlobalValue::LinkageTypes ///< The linkage type of the GV
  )
  {
  }

  void handleInitializedGV( 
    const Type* ElemType,     ///< The type of the global variable
    bool isConstant,          ///< Whether the GV is constant or not
    GlobalValue::LinkageTypes,///< The linkage type of the GV
    unsigned initSlot         ///< Slot number of GV's initializer
  )
  {
  }

  virtual void handleType( const Type* Ty ) 
  {
  }

  void handleFunctionDeclaration( 
    const Type* FuncType      ///< The type of the function
  )
  {
  }

  void handleModuleGlobalsEnd()
  {
  }

  void handleCompactionTableBegin()
  {
  }

  void handleCompactionTablePlane( 
    unsigned Ty, 
    unsigned NumEntries
  )
  {
  }

  void handleCompactionTableType( 
    unsigned i, 
    unsigned TypSlot, 
    const Type* 
  )
  {
  }

  void handleCompactionTableValue( 
    unsigned i, 
    unsigned ValSlot, 
    const Type* 
  )
  {
  }

  void handleCompactionTableEnd()
  {
  }

  void handleSymbolTableBegin()
  {
  }

  void handleSymbolTablePlane( 
    unsigned Ty, 
    unsigned NumEntries, 
    const Type* Typ
  )
  {
  }

  void handleSymbolTableType( 
    unsigned i, 
    unsigned slot, 
    const std::string& name 
  )
  {
  }

  void handleSymbolTableValue( 
    unsigned i, 
    unsigned slot, 
    const std::string& name 
  )
  {
  }

  void handleSymbolTableEnd()
  {
  }

  void handleFunctionBegin(
    const Type* FType, 
    GlobalValue::LinkageTypes linkage 
  )
  {
  }

  void handleFunctionEnd(
    const Type* FType
  )
  {
  }

  void handleBasicBlockBegin(
    unsigned blocknum
  )
  {
  }

  bool handleInstruction(
    unsigned Opcode, 
    const Type* iType, 
    std::vector<unsigned>& Operands
  )
  {
    return false;
  }

  void handleBasicBlockEnd(unsigned blocknum)
  {
  }

  void handleGlobalConstantsBegin()
  {
  }

  void handleConstantExpression( 
      unsigned Opcode, 
      const Type* Typ, 
      std::vector<std::pair<const Type*,unsigned> > ArgVec 
    )
  {
  }

  void handleConstantValue( Constant * c )
  {
  }

  void handleConstantArray( 
	  const ArrayType* AT, 
	  std::vector<unsigned>& Elements )
  {
  }

  void handleConstantStruct(
	const StructType* ST,
	std::vector<unsigned>& ElementSlots)
  {
  }

  void handleConstantPointer(
	const PointerType* PT, unsigned Slot)
  {
  }

  void handleConstantString( const ConstantArray* CA ) 
  {
  }


  void handleGlobalConstantsEnd()
  {
  }

};

}

void llvm::BytecodeAnalyzer::AnalyzeBytecode(
    const unsigned char *Buf, 
    unsigned Length,
    BytecodeAnalysis& bca,
    const std::string &ModuleID
)
{
  AnalyzerHandler TheHandler;
  AbstractBytecodeParser TheParser(&TheHandler);
  TheParser.ParseBytecode( Buf, Length, ModuleID );
  TheParser.ParseAllFunctionBodies();
}

// vim: sw=2
