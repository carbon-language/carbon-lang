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
#include <iostream>

using namespace llvm;


namespace {

class AnalyzerHandler : public BytecodeHandler {
  BytecodeAnalysis& bca;
public:
  AnalyzerHandler(BytecodeAnalysis& TheBca) 
    : bca(TheBca)
  {
  }  

  bool handleError(const std::string& str )
  {
    std::cerr << "Analysis Error: " << str;
    return false;
  }

  void handleStart()
  {
    bca.ModuleId.clear();
    bca.numTypes = 0;
    bca.numValues = 0;
    bca.numFunctions = 0;
    bca.numConstants = 0;
    bca.numGlobalVars = 0;
    bca.numInstructions = 0;
    bca.numBasicBlocks = 0;
    bca.numOperands = 0;
    bca.numCmpctnTables = 0;
    bca.numSymTab = 0;
    bca.maxTypeSlot = 0;
    bca.maxValueSlot = 0;
    bca.density = 0.0;
    bca.FunctionInfo.clear();
    bca.BytecodeDump.clear();
  }

  void handleFinish()
  {
    bca.density = bca.numTypes + bca.numFunctions + bca.numConstants +
      bca.numGlobalVars + bca.numInstructions;
    bca.density /= bca.byteSize;
  }

  void handleModuleBegin(const std::string& id)
  {
    bca.ModuleId = id;
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
    bca.numGlobalVars++;
  }

  void handleInitializedGV( 
    const Type* ElemType,     ///< The type of the global variable
    bool isConstant,          ///< Whether the GV is constant or not
    GlobalValue::LinkageTypes,///< The linkage type of the GV
    unsigned initSlot         ///< Slot number of GV's initializer
  )
  {
    bca.numGlobalVars++;
  }

  virtual void handleType( const Type* Ty ) 
  {
    bca.numTypes++;
  }

  void handleFunctionDeclaration( 
    const Type* FuncType      ///< The type of the function
  )
  {
    bca.numFunctions++;
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
    bca.numCmpctnTables++;
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
    bca.numSymTab++;
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
    bca.numBasicBlocks++;
  }

  bool handleInstruction(
    unsigned Opcode, 
    const Type* iType, 
    std::vector<unsigned>& Operands
  )
  {
    bca.numInstructions++;
    return Instruction::isTerminator(Opcode); 
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
    bca.numConstants++;
  }

  void handleConstantValue( Constant * c )
  {
    bca.numConstants++;
  }

  void handleConstantArray( 
	  const ArrayType* AT, 
	  std::vector<unsigned>& Elements )
  {
    bca.numConstants++;
  }

  void handleConstantStruct(
	const StructType* ST,
	std::vector<unsigned>& ElementSlots)
  {
    bca.numConstants++;
  }

  void handleConstantPointer(
	const PointerType* PT, unsigned Slot)
  {
    bca.numConstants++;
  }

  void handleConstantString( const ConstantArray* CA ) 
  {
    bca.numConstants++;
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
  bca.byteSize = Length;
  AnalyzerHandler TheHandler(bca);
  AbstractBytecodeParser TheParser(&TheHandler);
  TheParser.ParseBytecode( Buf, Length, ModuleID );
  if ( bca.detailedResults )
    TheParser.ParseAllFunctionBodies();
}

// vim: sw=2
