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
  BytecodeAnalysis::BytecodeFunctionInfo* currFunc;
public:
  AnalyzerHandler(BytecodeAnalysis& TheBca) 
    : bca(TheBca) 
    , currFunc(0)
    { }  

  virtual bool handleError(const std::string& str ) {
    return false;
  }

  virtual void handleStart() {
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

  virtual void handleFinish() {
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

  virtual void handleModuleBegin(const std::string& id) {
    bca.ModuleId = id;
  }

  virtual void handleModuleEnd(const std::string& id) { }

  virtual void handleVersionInfo(
    unsigned char RevisionNum,        ///< Byte code revision number
    Module::Endianness Endianness,    ///< Endianness indicator
    Module::PointerSize PointerSize   ///< PointerSize indicator
  ) { }

  virtual void handleModuleGlobalsBegin(unsigned size) { }

  virtual void handleGlobalVariable( 
    const Type* ElemType,     ///< The type of the global variable
    bool isConstant,          ///< Whether the GV is constant or not
    GlobalValue::LinkageTypes ///< The linkage type of the GV
  ) {
    bca.numGlobalVars++;
    bca.numValues++;
  }

  virtual void handleInitializedGV( 
    const Type* ElemType,     ///< The type of the global variable
    bool isConstant,          ///< Whether the GV is constant or not
    GlobalValue::LinkageTypes,///< The linkage type of the GV
    unsigned initSlot         ///< Slot number of GV's initializer
  ) {
    bca.numGlobalVars++;
    bca.numValues++;
  }

  virtual void handleType( const Type* Ty ) { bca.numTypes++; }

  virtual void handleFunctionDeclaration( 
    Function* Func,	    ///< The function
    const FunctionType* FuncType    ///< The type of the function
  ) {
    bca.numFunctions++;
    bca.numValues++;
  }

  virtual void handleModuleGlobalsEnd() { }

  virtual void handleCompactionTableBegin() { }

  virtual void handleCompactionTablePlane( unsigned Ty, unsigned NumEntries) {
    bca.numCmpctnTables++;
  }

  virtual void handleCompactionTableType( unsigned i, unsigned TypSlot, 
      const Type* ) {}

  virtual void handleCompactionTableValue( 
    unsigned i, 
    unsigned ValSlot, 
    const Type* ) { }

  virtual void handleCompactionTableEnd() { }

  virtual void handleSymbolTableBegin() { bca.numSymTab++; }

  virtual void handleSymbolTablePlane( unsigned Ty, unsigned NumEntries, 
    const Type* Typ) { }

  virtual void handleSymbolTableType( unsigned i, unsigned slot, 
    const std::string& name ) { }

  virtual void handleSymbolTableValue( unsigned i, unsigned slot, 
    const std::string& name ) { }

  virtual void handleSymbolTableEnd() { }

  virtual void handleFunctionBegin( Function* Func, unsigned Size) {
    const FunctionType* FType = 
      cast<FunctionType>(Func->getType()->getElementType());
    currFunc = &bca.FunctionInfo[Func];
    currFunc->description = FType->getDescription();
    currFunc->name = Func->getName();
    currFunc->byteSize = Size;
    currFunc->numInstructions = 0;
    currFunc->numBasicBlocks = 0;
    currFunc->numPhis = 0;
    currFunc->numOperands = 0;
    currFunc->density = 0.0;
    currFunc->vbrCount32 = 0;
    currFunc->vbrCount64 = 0;
    currFunc->vbrCompBytes = 0;
    currFunc->vbrExpdBytes = 0;
  }

  virtual void handleFunctionEnd( Function* Func) {
    currFunc->density = double(currFunc->byteSize) /
      double(currFunc->numInstructions+currFunc->numBasicBlocks);
  }

  virtual void handleBasicBlockBegin( unsigned blocknum) {
    bca.numBasicBlocks++;
    bca.numValues++;
    if ( currFunc ) currFunc->numBasicBlocks++;
  }

  virtual bool handleInstruction( unsigned Opcode, const Type* iType, 
    std::vector<unsigned>& Operands, unsigned Size) {
    bca.numInstructions++;
    bca.numValues++;
    bca.numOperands += Operands.size();
    if ( currFunc ) {
      currFunc->numInstructions++;
      if ( Opcode == Instruction::PHI ) currFunc->numPhis++;
    }
    return Instruction::isTerminator(Opcode); 
  }

  virtual void handleBasicBlockEnd(unsigned blocknum) { }

  virtual void handleGlobalConstantsBegin() { }

  virtual void handleConstantExpression( unsigned Opcode, const Type* Typ, 
      std::vector<std::pair<const Type*,unsigned> > ArgVec ) {
    bca.numConstants++;
    bca.numValues++;
  }

  virtual void handleConstantValue( Constant * c ) {
    bca.numConstants++;
    bca.numValues++;
  }

  virtual void handleConstantArray( const ArrayType* AT, 
          std::vector<unsigned>& Elements ) {
    bca.numConstants++;
    bca.numValues++;
  }

  virtual void handleConstantStruct(
        const StructType* ST,
        std::vector<unsigned>& ElementSlots)
  {
    bca.numConstants++;
    bca.numValues++;
  }

  virtual void handleConstantPointer( const PointerType* PT, unsigned Slot) {
    bca.numConstants++;
    bca.numValues++;
  }

  virtual void handleConstantString( const ConstantArray* CA ) {
    bca.numConstants++;
    bca.numValues++;
  }

  virtual void handleGlobalConstantsEnd() { }

  virtual void handleAlignment(unsigned numBytes) {
    bca.numAlignment += numBytes;
  }

  virtual void handleBlock(
    unsigned BType, const unsigned char* StartPtr, unsigned Size) {
    bca.numBlocks++;
    bca.BlockSizes[llvm::BytecodeFormat::FileBlockIDs(BType)] += Size;
  }

  virtual void handleVBR32(unsigned Size ) {
    bca.vbrCount32++;
    bca.vbrCompBytes += Size;
    bca.vbrExpdBytes += sizeof(uint32_t);
    if (currFunc) {
      currFunc->vbrCount32++;
      currFunc->vbrCompBytes += Size;
      currFunc->vbrExpdBytes += sizeof(uint32_t);
    }
  }

  virtual void handleVBR64(unsigned Size ) {
    bca.vbrCount64++;
    bca.vbrCompBytes += Size;
    bca.vbrExpdBytes += sizeof(uint64_t);
    if ( currFunc ) {
      currFunc->vbrCount64++;
      currFunc->vbrCompBytes += Size;
      currFunc->vbrExpdBytes += sizeof(uint64_t);
    }
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
