//===-- BytecodeDumper.cpp - Parsing Handler --------------------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencer and is distributed under the 
// University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
//  This header file defines the BytecodeDumper class that gets called by the
//  AbstractBytecodeParser when parsing events occur. It merely dumps the
//  information presented to it from the parser.
//
//===----------------------------------------------------------------------===//

#include "AnalyzerInternals.h"
#include "llvm/Constant.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Instruction.h"
#include "llvm/Type.h"

using namespace llvm;

namespace {

class BytecodeDumper : public llvm::BytecodeHandler {
public:

  virtual bool handleError(const std::string& str )
  {
    std::cout << "ERROR: " << str << "\n";
    return true;
  }

  virtual void handleStart()
  {
    std::cout << "Bytecode {\n";
  }

  virtual void handleFinish()
  {
    std::cout << "} End Bytecode\n";
  }

  virtual void handleModuleBegin(const std::string& id)
  {
    std::cout << "  Module " << id << " {\n";
  }

  virtual void handleModuleEnd(const std::string& id)
  {
    std::cout << "  } End Module " << id << "\n";
  }

  virtual void handleVersionInfo(
    unsigned char RevisionNum,        ///< Byte code revision number
    Module::Endianness Endianness,    ///< Endianness indicator
    Module::PointerSize PointerSize   ///< PointerSize indicator
  )
  {
    std::cout << "    RevisionNum: " << int(RevisionNum) 
	      << " Endianness: " << Endianness
	      << " PointerSize: " << PointerSize << "\n";
  }

  virtual void handleModuleGlobalsBegin()
  {
    std::cout << "    BLOCK: ModuleGlobalInfo {\n";
  }

  virtual void handleGlobalVariable( 
    const Type* ElemType,     ///< The type of the global variable
    bool isConstant,          ///< Whether the GV is constant or not
    GlobalValue::LinkageTypes Linkage ///< The linkage type of the GV
  )
  {
    std::cout << "      GV: Uninitialized, " 
	     << ( isConstant? "Constant, " : "Variable, ")
	     << " Linkage=" << Linkage << " Type=" 
	     << ElemType->getDescription() << "\n"; 
  }

  virtual void handleInitializedGV( 
    const Type* ElemType,     ///< The type of the global variable
    bool isConstant,          ///< Whether the GV is constant or not
    GlobalValue::LinkageTypes Linkage,///< The linkage type of the GV
    unsigned initSlot         ///< Slot number of GV's initializer
  )
  {
    std::cout << "      GV: Initialized, " 
	     << ( isConstant? "Constant, " : "Variable, ")
	     << " Linkage=" << Linkage << " Type=" 
	     << ElemType->getDescription()
	     << " InitializerSlot=" << initSlot << "\n"; 
  }

  virtual void handleType( const Type* Ty ) 
  {
    std::cout << "      Type: " << Ty->getDescription() << "\n";
  }

  virtual void handleFunctionDeclaration( const Type* FuncType )
  {
    std::cout << "      Function: " << FuncType->getDescription() << "\n";
  }

  virtual void handleModuleGlobalsEnd()
  {
    std::cout << "    } END BLOCK: ModuleGlobalInfo\n";
  }

  void handleCompactionTableBegin()
  {
    std::cout << "    BLOCK: CompactionTable {\n";
  }

  virtual void handleCompactionTablePlane( unsigned Ty, unsigned NumEntries )
  {
    std::cout << "      Plane: Ty=" << Ty << " Size=" << NumEntries << "\n";
  }

  virtual void handleCompactionTableType( 
    unsigned i, 
    unsigned TypSlot, 
    const Type* Ty
  )
  {
    std::cout << "        Type: " << i << " Slot:" << TypSlot 
	      << " is " << Ty->getDescription() << "\n"; 
  }

  virtual void handleCompactionTableValue( 
    unsigned i, 
    unsigned ValSlot, 
    const Type* Ty 
  )
  {
    std::cout << "        Value: " << i << " Slot:" << ValSlot 
	      << " is " << Ty->getDescription() << "\n"; 
  }

  virtual void handleCompactionTableEnd()
  {
    std::cout << "    } END BLOCK: CompactionTable\n";
  }

  virtual void handleSymbolTableBegin()
  {
    std::cout << "    BLOCK: SymbolTable {\n";
  }

  virtual void handleSymbolTablePlane( 
    unsigned Ty, 
    unsigned NumEntries, 
    const Type* Typ
  )
  {
    std::cout << "      Plane: Ty=" << Ty << " Size=" << NumEntries
	      << " Type: " << Typ->getDescription() << "\n"; 
  }

  virtual void handleSymbolTableType( 
    unsigned i, 
    unsigned slot, 
    const std::string& name 
  )
  {
    std::cout << "        Type " << i << " Slot=" << slot
	      << " Name: " << name << "\n"; 
  }

  virtual void handleSymbolTableValue( 
    unsigned i, 
    unsigned slot, 
    const std::string& name 
  )
  {
    std::cout << "        Value " << i << " Slot=" << slot
	      << " Name: " << name << "\n";
  }

  virtual void handleSymbolTableEnd()
  {
    std::cout << "    } END BLOCK: SymbolTable\n";
  }

  virtual void handleFunctionBegin(
    const Type* FType, 
    GlobalValue::LinkageTypes linkage 
  )
  {
    std::cout << "BLOCK: Function {\n";
    std::cout << "  Linkage: " << linkage << "\n";
    std::cout << "  Type: " << FType->getDescription() << "\n";
  }

  virtual void handleFunctionEnd(
    const Type* FType
  )
  {
    std::cout << "} END BLOCK: Function\n";
  }

  virtual void handleBasicBlockBegin(
    unsigned blocknum
  )
  {
    std::cout << "  BLOCK: BasicBlock #" << blocknum << "{\n";
  }

  virtual bool handleInstruction(
    unsigned Opcode, 
    const Type* iType, 
    std::vector<unsigned>& Operands
  )
  {
    std::cout << "    INST: OpCode=" 
	      << Instruction::getOpcodeName(Opcode) << " Type=" 
	      << iType->getDescription() << "\n";
    for ( unsigned i = 0; i < Operands.size(); ++i ) 
      std::cout << "      Op#" << i << " Slot=" << Operands[i] << "\n";
    
    return Instruction::isTerminator(Opcode); 
  }

  virtual void handleBasicBlockEnd(unsigned blocknum)
  {
    std::cout << "  } END BLOCK: BasicBlock #" << blocknum << "{\n";
  }

  virtual void handleGlobalConstantsBegin()
  {
    std::cout << "    BLOCK: GlobalConstants {\n";
  }

  virtual void handleConstantExpression( 
      unsigned Opcode, 
      const Type* Typ, 
      std::vector<std::pair<const Type*,unsigned> > ArgVec 
    )
  {
    std::cout << "      EXPR: " << Instruction::getOpcodeName(Opcode)
	      << " Type=" << Typ->getDescription() << "\n";
    for ( unsigned i = 0; i < ArgVec.size(); ++i ) 
      std::cout << "        Arg#" << i << " Type=" 
	<< ArgVec[i].first->getDescription() << " Slot=" 
	<< ArgVec[i].second << "\n";
  }

  virtual void handleConstantValue( Constant * c )
  {
    std::cout << "      VALUE: ";
    c->print(std::cout);
    std::cout << "\n";
  }

  virtual void handleConstantArray( 
	  const ArrayType* AT, 
	  std::vector<unsigned>& Elements )
  {
    std::cout << "      ARRAY: " << AT->getDescription() << "\n";
    for ( unsigned i = 0; i < Elements.size(); ++i ) 
      std::cout << "        #" << i << " Slot=" << Elements[i] << "\n";
  }

  virtual void handleConstantStruct(
	const StructType* ST,
	std::vector<unsigned>& Elements)
  {
    std::cout << "      STRUC: " << ST->getDescription() << "\n";
    for ( unsigned i = 0; i < Elements.size(); ++i ) 
      std::cout << "        #" << i << " Slot=" << Elements[i] << "\n";
  }

  virtual void handleConstantPointer(
	const PointerType* PT, unsigned Slot)
  {
    std::cout << "      POINT: " << PT->getDescription() 
	      << " Slot=" << Slot << "\n";
  }

  virtual void handleConstantString( const ConstantArray* CA ) 
  {
    std::cout << "      STRNG: ";
    CA->print(std::cout); 
    std::cout << "\n";
  }

  virtual void handleGlobalConstantsEnd()
  {
    std::cout << "    } END BLOCK: GlobalConstants\n";
  }
};

}

void BytecodeAnalyzer::DumpBytecode(
    const unsigned char *Buf, 
    unsigned Length,
    BytecodeAnalysis& bca,
    const std::string &ModuleID
  )
{
  BytecodeDumper TheHandler;
  AbstractBytecodeParser TheParser(&TheHandler);
  TheParser.ParseBytecode( Buf, Length, ModuleID );
  if ( bca.detailedResults )
    TheParser.ParseAllFunctionBodies();
}

// vim: sw=2
