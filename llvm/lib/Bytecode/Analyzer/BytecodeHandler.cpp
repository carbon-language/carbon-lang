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

#include "Parser.h"

using namespace llvm;

bool BytecodeHandler::handleError(const std::string& str )
{
  return false;
}

void BytecodeHandler::handleStart()
{
}

void BytecodeHandler::handleFinish()
{
}

void BytecodeHandler::handleModuleBegin(const std::string& id)
{
}

void BytecodeHandler::handleModuleEnd(const std::string& id)
{
}

void BytecodeHandler::handleVersionInfo(
  unsigned char RevisionNum,        ///< Byte code revision number
  Module::Endianness Endianness,    ///< Endianness indicator
  Module::PointerSize PointerSize   ///< PointerSize indicator
)
{
}

void BytecodeHandler::handleModuleGlobalsBegin()
{
}

void BytecodeHandler::handleGlobalVariable( 
  const Type* ElemType,     ///< The type of the global variable
  bool isConstant,          ///< Whether the GV is constant or not
  GlobalValue::LinkageTypes ///< The linkage type of the GV
)
{
}

void BytecodeHandler::handleInitializedGV( 
  const Type* ElemType,     ///< The type of the global variable
  bool isConstant,          ///< Whether the GV is constant or not
  GlobalValue::LinkageTypes,///< The linkage type of the GV
  unsigned initSlot         ///< Slot number of GV's initializer
)
{
}

void BytecodeHandler::handleType( const Type* Ty ) 
{
}

void BytecodeHandler::handleFunctionDeclaration( 
  const Type* FuncType      ///< The type of the function
)
{
}

void BytecodeHandler::handleModuleGlobalsEnd()
{
}

void BytecodeHandler::handleCompactionTableBegin()
{
}

void BytecodeHandler::handleCompactionTablePlane( 
  unsigned Ty, 
  unsigned NumEntries
)
{
}

void BytecodeHandler::handleCompactionTableType( 
  unsigned i, 
  unsigned TypSlot, 
  const Type* 
)
{
}

void BytecodeHandler::handleCompactionTableValue( 
  unsigned i, 
  unsigned ValSlot, 
  const Type* 
)
{
}

void BytecodeHandler::handleCompactionTableEnd()
{
}

void BytecodeHandler::handleSymbolTableBegin()
{
}

void BytecodeHandler::handleSymbolTablePlane( 
  unsigned Ty, 
  unsigned NumEntries, 
  const Type* Typ
)
{
}

void BytecodeHandler::handleSymbolTableType( 
  unsigned i, 
  unsigned slot, 
  const std::string& name 
)
{
}

void BytecodeHandler::handleSymbolTableValue( 
  unsigned i, 
  unsigned slot, 
  const std::string& name 
)
{
}

void BytecodeHandler::handleSymbolTableEnd()
{
}

void BytecodeHandler::handleFunctionBegin(
  const Type* FType, 
  GlobalValue::LinkageTypes linkage 
)
{
}

void BytecodeHandler::handleFunctionEnd(
  const Type* FType
)
{
}

void BytecodeHandler::handleBasicBlockBegin(
  unsigned blocknum
)
{
}

bool BytecodeHandler::handleInstruction(
  unsigned Opcode, 
  const Type* iType, 
  std::vector<unsigned>& Operands
)
{
  return false;
}

void BytecodeHandler::handleBasicBlockEnd(unsigned blocknum)
{
}

void BytecodeHandler::handleGlobalConstantsBegin()
{
}

void BytecodeHandler::handleConstantExpression( 
    unsigned Opcode, 
    const Type* Typ, 
    std::vector<std::pair<const Type*,unsigned> > ArgVec 
  )
{
}

void BytecodeHandler::handleConstantValue( Constant * c )
{
}

void BytecodeHandler::handleConstantArray( 
        const ArrayType* AT, 
        std::vector<unsigned>& Elements )
{
}

void BytecodeHandler::handleConstantStruct(
      const StructType* ST,
      std::vector<unsigned>& ElementSlots)
{
}

void BytecodeHandler::handleConstantPointer(
      const PointerType* PT, unsigned Slot)
{
}

void BytecodeHandler::handleConstantString( const ConstantArray* CA ) 
{
}


void BytecodeHandler::handleGlobalConstantsEnd()
{
}

// vim: sw=2
