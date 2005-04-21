//===-- BytecodeHandler.h - Handle Bytecode Parsing Events ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencer and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This header file defines the interface to the Bytecode Handler. The handler
//  is called by the Bytecode Reader to obtain out-of-band parsing events for
//  tasks other then LLVM IR construction.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_BYTECODE_BYTECODEHANDLER_H
#define LLVM_BYTECODE_BYTECODEHANDLER_H

#include "llvm/Module.h"

namespace llvm {

class ArrayType;
class StructType;
class PointerType;
class PackedType;
class ConstantArray;
class Module;

/// This class provides the interface for handling bytecode events during
/// reading of bytecode. The methods on this interface are invoked by the
/// BytecodeReader as it discovers the content of a bytecode stream.
/// This class provides a a clear separation of concerns between recognizing
/// the semantic units of a bytecode file (the Reader) and deciding what to do
/// with them (the Handler).
///
/// The BytecodeReader recognizes the content of the bytecode file and
/// calls the BytecodeHandler methods to let it perform additional tasks. This
/// arrangement allows Bytecode files to be read and handled for a number of
/// purposes simply by creating a subclass of BytecodeHandler. None of the
/// parsing details need to be understood, only the meaning of the calls
/// made on this interface.
///
/// @see BytecodeHandler
/// @brief Handle Bytecode Parsing Events
class BytecodeHandler {

/// @name Constructors And Operators
/// @{
public:
  /// @brief Default constructor (empty)
  BytecodeHandler() {}
  /// @brief Virtual destructor (empty)
  virtual ~BytecodeHandler();

private:
  BytecodeHandler(const BytecodeHandler &);  // DO NOT IMPLEMENT
  void operator=(const BytecodeHandler &);  // DO NOT IMPLEMENT

/// @}
/// @name Handler Methods
/// @{
public:

  /// This method is called whenever the parser detects an error in the
  /// bytecode formatting. It gives the handler a chance to do something
  /// with the error message before the parser throws an exception to
  /// terminate the parsing.
  /// @brief Handle parsing errors.
  virtual void handleError(const std::string& str ) {}

  /// This method is called at the beginning of a parse before anything is
  /// read in order to give the handler a chance to initialize.
  /// @brief Handle the start of a bytecode parse
  virtual void handleStart( Module* Mod, unsigned byteSize ) {}

  /// This method is called at the end of a parse after everything has been
  /// read in order to give the handler a chance to terminate.
  /// @brief Handle the end of a bytecode parse
  virtual void handleFinish() {}

  /// This method is called at the start of a module to indicate that a
  /// module is being parsed.
  /// @brief Handle the start of a module.
  virtual void handleModuleBegin(const std::string& moduleId) {}

  /// This method is called at the end of a module to indicate that the module
  /// previously being parsed has concluded.
  /// @brief Handle the end of a module.
  virtual void handleModuleEnd(
    const std::string& moduleId ///< An identifier for the module
  ) {}

  /// This method is called once the version information has been parsed. It
  /// provides the information about the version of the bytecode file being
  /// read.
  /// @brief Handle the bytecode prolog
  virtual void handleVersionInfo(
    unsigned char RevisionNum,        ///< Byte code revision number
    Module::Endianness Endianness,    ///< Endianness indicator
    Module::PointerSize PointerSize   ///< PointerSize indicator
  ) {}

  /// This method is called at the start of a module globals block which
  /// contains the global variables and the function placeholders
  virtual void handleModuleGlobalsBegin() {}

  /// This method is called when a non-initialized global variable is
  /// recognized. Its type, constness, and linkage type are provided.
  /// @brief Handle a non-initialized global variable
  virtual void handleGlobalVariable(
    const Type* ElemType,     ///< The type of the global variable
    bool isConstant,          ///< Whether the GV is constant or not
    GlobalValue::LinkageTypes,///< The linkage type of the GV
    unsigned SlotNum,         ///< Slot number of GV
    unsigned initSlot         ///< Slot number of GV's initializer (0 if none)
  ) {}

  /// This method is called when a type list is recognized. It simply
  /// provides the number of types that the list contains. The handler
  /// should expect that number of calls to handleType.
  /// @brief Handle a type
  virtual void handleTypeList(
    unsigned numEntries ///< The number of entries in the type list
  ) {}

  /// This method is called when a new type is recognized. The type is
  /// converted from the bytecode and passed to this method.
  /// @brief Handle a type
  virtual void handleType(
    const Type* Ty ///< The type that was just recognized
  ) {}

  /// This method is called when the function prototype for a function is
  /// encountered in the module globals block.
  virtual void handleFunctionDeclaration(
    Function* Func ///< The function being declared
  ) {}

  /// This method is called when a global variable is initialized with
  /// its constant value. Because of forward referencing, etc. this is
  /// done towards the end of the module globals block
  virtual void handleGlobalInitializer(GlobalVariable*, Constant* ) {}

  /// This method is called for each dependent library name found
  /// in the module globals block.
  virtual void handleDependentLibrary(const std::string& libName) {}

  /// This method is called if the module globals has a non-empty target
  /// triple
  virtual void handleTargetTriple(const std::string& triple) {}

  /// This method is called at the end of the module globals block.
  /// @brief Handle end of module globals block.
  virtual void handleModuleGlobalsEnd() {}

  /// This method is called at the beginning of a compaction table.
  /// @brief Handle start of compaction table.
  virtual void handleCompactionTableBegin() {}

  /// @brief Handle start of a compaction table plane
  virtual void handleCompactionTablePlane(
    unsigned Ty,         ///< The type of the plane (slot number)
    unsigned NumEntries  ///< The number of entries in the plane
  ) {}

  /// @brief Handle a type entry in the compaction table
  virtual void handleCompactionTableType(
    unsigned i,       ///< Index in the plane of this type
    unsigned TypSlot, ///< Slot number for this type
    const Type*       ///< The type referenced by this slot
  ) {}

  /// @brief Handle a value entry in the compaction table
  virtual void handleCompactionTableValue(
    unsigned i,       ///< Index in the compaction table's type plane
    unsigned TypSlot, ///< The slot (plane) of the type of this value
    unsigned ValSlot  ///< The global value slot of the value
  ) {}

  /// @brief Handle end of a compaction table
  virtual void handleCompactionTableEnd() {}

  /// @brief Handle start of a symbol table
  virtual void handleSymbolTableBegin(
    Function* Func,  ///< The function to which the ST belongs
    SymbolTable* ST  ///< The symbol table being filled
  ) {}

  /// @brief Handle start of a symbol table plane
  virtual void handleSymbolTablePlane(
    unsigned TySlot,      ///< The slotnum of the type plane
    unsigned NumEntries,  ///< Number of entries in the plane
    const Type* Typ       ///< The type of this type plane
  ) {}

  /// @brief Handle a named type in the symbol table
  virtual void handleSymbolTableType(
    unsigned i,              ///< The index of the type in this plane
    unsigned slot,           ///< Slot number of the named type
    const std::string& name  ///< Name of the type
  ) {}

  /// @brief Handle a named value in the symbol table
  virtual void handleSymbolTableValue(
    unsigned i,              ///< The index of the value in this plane
    unsigned slot,           ///< Slot number of the named value
    const std::string& name  ///< Name of the value.
  ) {}

  /// @brief Handle the end of a symbol table
  virtual void handleSymbolTableEnd() {}

  /// @brief Handle the beginning of a function body
  virtual void handleFunctionBegin(
    Function* Func, ///< The function being defined
    unsigned Size   ///< The size (in bytes) of the function's bytecode
  ) {}

  /// @brief Handle the end of a function body
  virtual void handleFunctionEnd(
    Function* Func  ///< The function whose definition has just finished.
  ) {}

  /// @brief Handle the beginning of a basic block
  virtual void handleBasicBlockBegin(
    unsigned blocknum ///< The block number of the block
  ) {}

  /// This method is called for each instruction that is parsed.
  /// @returns true if the instruction is a block terminating instruction
  /// @brief Handle an instruction
  virtual bool handleInstruction(
    unsigned Opcode,                 ///< Opcode of the instruction
    const Type* iType,               ///< Instruction type
    std::vector<unsigned>& Operands, ///< Vector of slot # operands
    unsigned Length                  ///< Length of instruction in bc bytes
  ) { return false; }

  /// @brief Handle the end of a basic block
  virtual void handleBasicBlockEnd(
    unsigned blocknum  ///< The block number of the block just finished
  ) {}

  /// @brief Handle start of global constants block.
  virtual void handleGlobalConstantsBegin() {}

  /// @brief Handle a constant expression
  virtual void handleConstantExpression(
    unsigned Opcode,  ///< Opcode of primary expression operator
    std::vector<Constant*> ArgVec, ///< expression args
    Constant* C ///< The constant value
  ) {}

  /// @brief Handle a constant array
  virtual void handleConstantArray(
    const ArrayType* AT,                ///< Type of the array
    std::vector<Constant*>& ElementSlots,///< Slot nums for array values
    unsigned TypeSlot,                  ///< Slot # of type
    Constant* Val                       ///< The constant value
  ) {}

  /// @brief Handle a constant structure
  virtual void handleConstantStruct(
    const StructType* ST,               ///< Type of the struct
    std::vector<Constant*>& ElementSlots,///< Slot nums for struct values
    Constant* Val                       ///< The constant value
  ) {}

  /// @brief Handle a constant packed
  virtual void handleConstantPacked(
    const PackedType* PT,                ///< Type of the array
    std::vector<Constant*>& ElementSlots,///< Slot nums for packed values
    unsigned TypeSlot,                  ///< Slot # of type
    Constant* Val                       ///< The constant value
  ) {}

  /// @brief Handle a constant pointer
  virtual void handleConstantPointer(
    const PointerType* PT, ///< Type of the pointer
    unsigned Slot,         ///< Slot num of initializer value
    GlobalValue* GV        ///< Referenced global value
  ) {}

  /// @brief Handle a constant strings (array special case)
  virtual void handleConstantString(
    const ConstantArray* CA ///< Type of the string array
  ) {}

  /// @brief Handle a primitive constant value
  virtual void handleConstantValue(
    Constant * c ///< The constant just defined
  ) {}

  /// @brief Handle the end of the global constants
  virtual void handleGlobalConstantsEnd() {}

  /// @brief Handle an alignment event
  virtual void handleAlignment(
    unsigned numBytes ///< The number of bytes added for alignment
  ) {}

  /// @brief Handle a bytecode block
  virtual void handleBlock(
    unsigned BType,                ///< The type of block
    const unsigned char* StartPtr, ///< The start of the block
    unsigned Size                  ///< The size of the block
  ) {}

  /// @brief Handle a variable bit rate 32 bit unsigned
  virtual void handleVBR32(
    unsigned Size  ///< Number of bytes the vbr_uint took up
  ) {}

  /// @brief Handle a variable bit rate 64 bit unsigned
  virtual void handleVBR64(
    unsigned Size  ///< Number of byte sthe vbr_uint64 took up
  ) {}
/// @}

};

}
// vim: sw=2 ai
#endif
