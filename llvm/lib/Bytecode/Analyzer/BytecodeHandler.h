//===-- BytecodeHandler.h - Parsing Handler ---------------------*- C++ -*-===//
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

#ifndef BYTECODE_HANDLER_H
#define BYTECODE_HANDLER_H

#include "llvm/Module.h"
#include "llvm/GlobalValue.h"
#include <vector>

namespace llvm {

class ArrayType;
class StructType;
class PointerType;
class ConstantArray;

/// This class provides the interface for the handling bytecode events during
/// parsing. The methods on this interface are invoked by the 
/// AbstractBytecodeParser as it discovers the content of a bytecode stream. 
/// This class provides a a clear separation of concerns between recognizing 
/// the semantic units of a bytecode file and deciding what to do with them. 
/// The AbstractBytecodeParser recognizes the content of the bytecode file and
/// calls the BytecodeHandler methods to determine what should be done. This
/// arrangement allows Bytecode files to be read and handled for a number of
/// purposes simply by creating a subclass of BytecodeHandler. None of the
/// parsing details need to be understood, only the meaning of the calls
/// made on this interface.
/// 
/// Another paradigm that uses this design pattern is the XML SAX Parser. The
/// ContentHandler for SAX plays the same role as the BytecodeHandler here.
/// @brief Handle Bytecode Parsing Events
class BytecodeHandler {

/// @name Constructors And Operators
/// @{
public:
  /// @brief Default constructor (empty)
  BytecodeHandler() {}
  /// @brief Virtual destructor (empty)
  virtual ~BytecodeHandler() {}

private:
  BytecodeHandler(const BytecodeHandler &);  // DO NOT IMPLEMENT
  void operator=(const BytecodeHandler &);  // DO NOT IMPLEMENT

/// @}
/// @name Handler Methods
/// @{
public:

  /// This method is called whenever the parser detects an error in the
  /// bytecode formatting. Returning true will cause the parser to keep 
  /// going, however this is inadvisable in most cases. Returning false will
  /// cause the parser to throw the message as a std::string.
  /// @brief Handle parsing errors.
  virtual bool handleError(const std::string& str );

  /// This method is called at the beginning of a parse before anything is
  /// read in order to give the handler a chance to initialize.
  /// @brief Handle the start of a bytecode parse
  virtual void handleStart();

  /// This method is called at the end of a parse after everything has been
  /// read in order to give the handler a chance to terminate.
  /// @brief Handle the end of a bytecode parse
  virtual void handleFinish();

  /// This method is called at the start of a module to indicate that a
  /// module is being parsed.
  /// @brief Handle the start of a module.
  virtual void handleModuleBegin(const std::string& id);

  /// This method is called at the end of a module to indicate that the module
  /// previously being parsed has concluded.
  /// @brief Handle the end of a module.
  virtual void handleModuleEnd(const std::string& id);

  /// This method is called once the version information has been parsed. It 
  /// provides the information about the version of the bytecode file being 
  /// read.
  /// @brief Handle the bytecode prolog
  virtual void handleVersionInfo(
    unsigned char RevisionNum,        ///< Byte code revision number
    Module::Endianness Endianness,    ///< Endianness indicator
    Module::PointerSize PointerSize   ///< PointerSize indicator
  );

  /// This method is called at the start of a module globals block which
  /// contains the global variables and the function placeholders
  virtual void handleModuleGlobalsBegin();

  /// This method is called when a non-initialized global variable is 
  /// recognized. Its type, constness, and linkage type are provided.
  /// @brief Handle a non-initialized global variable
  virtual void handleGlobalVariable( 
    const Type* ElemType,     ///< The type of the global variable
    bool isConstant,          ///< Whether the GV is constant or not
    GlobalValue::LinkageTypes ///< The linkage type of the GV
  );

  /// This method is called when an initialized global variable is recognized.
  /// Its type constness, linkage type, and the slot number of the initializer
  /// are provided.
  /// @brief Handle an intialized global variable.
  virtual void handleInitializedGV( 
    const Type* ElemType,     ///< The type of the global variable
    bool isConstant,          ///< Whether the GV is constant or not
    GlobalValue::LinkageTypes,///< The linkage type of the GV
    unsigned initSlot         ///< Slot number of GV's initializer
  );

  /// This method is called when a new type is recognized. The type is 
  /// converted from the bytecode and passed to this method.
  /// @brief Handle a type
  virtual void handleType( const Type* Ty );

  /// This method is called when the function prototype for a function is
  /// encountered in the module globals block.
  virtual void handleFunctionDeclaration( 
    const Type* FuncType      ///< The type of the function
  );

  /// This method is called at the end of the module globals block.
  /// @brief Handle end of module globals block.
  virtual void handleModuleGlobalsEnd();

  /// This method is called at the beginning of a compaction table.
  /// @brief Handle start of compaction table.
  virtual void handleCompactionTableBegin();
  virtual void handleCompactionTablePlane( 
    unsigned Ty, 
    unsigned NumEntries
  );

  virtual void handleCompactionTableType( 
    unsigned i, 
    unsigned TypSlot, 
    const Type* 
  );

  virtual void handleCompactionTableValue( 
    unsigned i, 
    unsigned ValSlot, 
    const Type* 
  );

  virtual void handleCompactionTableEnd();

  virtual void handleSymbolTableBegin();

  virtual void handleSymbolTablePlane( 
    unsigned Ty, 
    unsigned NumEntries, 
    const Type* Ty 
  );

  virtual void handleSymbolTableType( 
    unsigned i, 
    unsigned slot, 
    const std::string& name 
  );

  virtual void handleSymbolTableValue( 
    unsigned i, 
    unsigned slot, 
    const std::string& name 
  );

  virtual void handleSymbolTableEnd();

  virtual void handleFunctionBegin(
    const Type* FType, 
    GlobalValue::LinkageTypes linkage 
  );

  virtual void handleFunctionEnd(
    const Type* FType
  );

  virtual void handleBasicBlockBegin(
    unsigned blocknum
  );

  /// This method is called for each instruction that is parsed. 
  /// @returns true if the instruction is a block terminating instruction
  /// @brief Handle an instruction
  virtual bool handleInstruction(
    unsigned Opcode, 
    const Type* iType, 
    std::vector<unsigned>& Operands
  );

  /// This method is called for each block that is parsed.
  virtual void handleBasicBlockEnd(unsigned blocknum);
  /// This method is called at the start of the global constants block.
  /// @brief Handle start of global constants block.
  virtual void handleGlobalConstantsBegin();

  virtual void handleConstantExpression( 
    unsigned Opcode, 
    const Type* Typ, 
    std::vector<std::pair<const Type*,unsigned> > ArgVec 
  );

  virtual void handleConstantArray( 
    const ArrayType* AT, 
    std::vector<unsigned>& ElementSlots
  );

  virtual void handleConstantStruct(
    const StructType* ST,
    std::vector<unsigned>& ElementSlots
  );

  virtual void handleConstantPointer(
    const PointerType* PT,
    unsigned Slot
  );

  virtual void handleConstantString(
    const ConstantArray* CA
  );

  virtual void handleConstantValue( Constant * c );
  virtual void handleGlobalConstantsEnd();

/// @}

};

} // End llvm namespace

#endif

// vim: sw=2
