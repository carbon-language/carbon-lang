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
    return false;
  }

  void handleStart()
  {
    bca.ModuleId.clear();
    bca.numBlocks = 0;
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
    bca.numAlignment = 0;
    bca.fileDensity = 0.0;
    bca.globalsDensity = 0.0;
    bca.functionDensity = 0.0;
    bca.vbrCount32 = 0;
    bca.vbrCount64 = 0;
    bca.vbrCompBytes = 0;
    bca.vbrExpdBytes = 0;
    bca.FunctionInfo.clear();
    bca.BytecodeDump.clear();
    bca.BlockSizes[BytecodeFormat::Module] = 0;
    bca.BlockSizes[BytecodeFormat::Function] = 0;
    bca.BlockSizes[BytecodeFormat::ConstantPool] = 0;
    bca.BlockSizes[BytecodeFormat::SymbolTable] = 0;
    bca.BlockSizes[BytecodeFormat::ModuleGlobalInfo] = 0;
    bca.BlockSizes[BytecodeFormat::GlobalTypePlane] = 0;
    bca.BlockSizes[BytecodeFormat::BasicBlock] = 0;
    bca.BlockSizes[BytecodeFormat::InstructionList] = 0;
    bca.BlockSizes[BytecodeFormat::CompactionTable] = 0;
  }

  void handleFinish()
  {
    bca.fileDensity = double(bca.byteSize) / double( bca.numTypes + bca.numValues );
    double globalSize = 0.0;
    globalSize += double(bca.BlockSizes[BytecodeFormat::ConstantPool]);
    globalSize += double(bca.BlockSizes[BytecodeFormat::ModuleGlobalInfo]);
    globalSize += double(bca.BlockSizes[BytecodeFormat::GlobalTypePlane]);
    bca.globalsDensity = globalSize / double( bca.numTypes + bca.numConstants + 
      bca.numGlobalVars );
    bca.functionDensity = double(bca.BlockSizes[BytecodeFormat::Function]) / 
      double(bca.numFunctions);
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

  void handleModuleGlobalsBegin(unsigned size)
  {
    // bca.globalBytesize += size;
  }

  void handleGlobalVariable( 
    const Type* ElemType,     ///< The type of the global variable
    bool isConstant,          ///< Whether the GV is constant or not
    GlobalValue::LinkageTypes ///< The linkage type of the GV
  )
  {
    bca.numGlobalVars++;
    bca.numValues++;
  }

  void handleInitializedGV( 
    const Type* ElemType,     ///< The type of the global variable
    bool isConstant,          ///< Whether the GV is constant or not
    GlobalValue::LinkageTypes,///< The linkage type of the GV
    unsigned initSlot         ///< Slot number of GV's initializer
  )
  {
    bca.numGlobalVars++;
    bca.numValues++;
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
    bca.numValues++;
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
    bca.numValues++;
  }

  bool handleInstruction(
    unsigned Opcode, 
    const Type* iType, 
    std::vector<unsigned>& Operands,
    unsigned Size
  )
  {
    bca.numInstructions++;
    bca.numValues++;
    bca.numOperands += Operands.size();
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
    bca.numValues++;
  }

  void handleConstantValue( Constant * c )
  {
    bca.numConstants++;
    bca.numValues++;
  }

  void handleConstantArray( 
          const ArrayType* AT, 
          std::vector<unsigned>& Elements )
  {
    bca.numConstants++;
    bca.numValues++;
  }

  void handleConstantStruct(
        const StructType* ST,
        std::vector<unsigned>& ElementSlots)
  {
    bca.numConstants++;
    bca.numValues++;
  }

  void handleConstantPointer(
        const PointerType* PT, unsigned Slot)
  {
    bca.numConstants++;
    bca.numValues++;
  }

  void handleConstantString( const ConstantArray* CA ) 
  {
    bca.numConstants++;
    bca.numValues++;
  }


  void handleGlobalConstantsEnd() { }

  void handleAlignment(unsigned numBytes) {
    bca.numAlignment += numBytes;
  }

  void handleBlock(
    unsigned BType, const unsigned char* StartPtr, unsigned Size) {
    bca.numBlocks++;
    bca.BlockSizes[llvm::BytecodeFormat::FileBlockIDs(BType)] += Size;
  }

  virtual void handleVBR32(unsigned Size ) {
    bca.vbrCount32++;
    bca.vbrCompBytes += Size;
    bca.vbrExpdBytes += sizeof(uint32_t);
  }
  virtual void handleVBR64(unsigned Size ) {
    bca.vbrCount64++;
    bca.vbrCompBytes += Size;
    bca.vbrExpdBytes += sizeof(uint64_t);
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
  AbstractBytecodeParser TheParser(&TheHandler, true, true, true);
  TheParser.ParseBytecode( Buf, Length, ModuleID );
  TheParser.ParseAllFunctionBodies();
}

// vim: sw=2
