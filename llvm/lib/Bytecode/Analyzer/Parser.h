//===-- Parser.h - Abstract Interface To Bytecode Parsing -------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencer and is distributed under the 
// University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
//  This header file defines the interface to the Bytecode Parser and the
//  Bytecode Handler interface that it calls.
//
//===----------------------------------------------------------------------===//

#ifndef BYTECODE_PARSER_H
#define BYTECODE_PARSER_H

#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/GlobalValue.h"
#include "llvm/Module.h"
#include <utility>
#include <vector>
#include <map>

namespace llvm {

class BytecodeHandler; ///< Forward declare the handler interface

/// This class defines the interface for parsing a buffer of bytecode. The
/// parser itself takes no action except to call the various functions of
/// the handler interface. The parser's sole responsibility is the correct
/// interpretation of the bytecode buffer. The handler is responsible for 
/// instantiating and keeping track of all values. As a convenience, the parser 
/// is responsible for materializing types and will pass them through the
/// handler interface as necessary.
/// @see BytecodeHandler
/// @brief Abstract Bytecode Parser interface
class AbstractBytecodeParser {

/// @name Constructors
/// @{
public:
  AbstractBytecodeParser( BytecodeHandler* h ) { handler = h; }
  ~AbstractBytecodeParser() { }

/// @}
/// @name Types
/// @{
public:
  /// @brief A convenience type for the buffer pointer
  typedef const unsigned char* BufPtr;

  /// @brief The type used for vector of potentially abstract types
  typedef std::vector<PATypeHolder> TypeListTy;

  /// @brief 

/// @}
/// @name Methods
/// @{
public:

  /// @brief Main interface to parsing a bytecode buffer.
  void ParseBytecode(const unsigned char *Buf, unsigned Length,
                     const std::string &ModuleID);

  /// The ParseBytecode method lazily parses functions. Use this
  /// method to cause the parser to actually parse all the function bodies
  /// in the bytecode buffer.
  /// @see ParseBytecode
  /// @brief Parse all function bodies
  void ParseAllFunctionBodies  ();

  /// The Parsebytecode method lazily parses functions. Use this
  /// method to casue the parser to parse the next function of a given
  /// types. Note that this will remove the function from what is to be
  /// included by ParseAllFunctionBodies.
  /// @see ParseAllFunctionBodies
  /// @see ParseBytecode
  /// @brief Parse the next function of specific type
  void ParseNextFunction       (Type* FType) ;

/// @}
/// @name Parsing Units For Subclasses
/// @{
protected:
  /// @brief Parse whole module scope
  void ParseModule             (BufPtr &Buf, BufPtr End);

  /// @brief Parse the version information block
  void ParseVersionInfo        (BufPtr &Buf, BufPtr End);

  /// @brief Parse the ModuleGlobalInfo block
  void ParseModuleGlobalInfo   (BufPtr &Buf, BufPtr End);

  /// @brief Parse a symbol table
  void ParseSymbolTable        (BufPtr &Buf, BufPtr End);

  /// This function parses LLVM functions lazily. It obtains the type of the
  /// function and records where the body of the function is in the bytecode
  /// buffer. The caller can then use the ParseNextFunction and 
  /// ParseAllFunctionBodies to get handler events for the functions.
  /// @brief Parse functions lazily.
  void ParseFunctionLazily     (BufPtr &Buf, BufPtr End);

  ///  @brief Parse a function body
  void ParseFunctionBody       (const Type* FType, BufPtr &Buf, BufPtr EndBuf);

  /// @brief Parse a compaction table
  void ParseCompactionTable    (BufPtr &Buf, BufPtr End);

  /// @brief Parse global types
  void ParseGlobalTypes        (BufPtr &Buf, BufPtr End);

  /// @brief Parse a basic block (for LLVM 1.0 basic block blocks)
  void ParseBasicBlock         (BufPtr &Buf, BufPtr End, unsigned BlockNo);

  /// @brief parse an instruction list (for post LLVM 1.0 instruction lists
  /// with blocks differentiated by terminating instructions.
  unsigned ParseInstructionList(BufPtr &Buf, BufPtr End);
  
  /// @brief Parse an instruction.
  bool ParseInstruction        (BufPtr &Buf, BufPtr End, 
	                        std::vector<unsigned>& Args);

  /// @brief Parse a constant pool
  void ParseConstantPool       (BufPtr &Buf, BufPtr End, TypeListTy& List);

  /// @brief Parse a constant value
  void ParseConstantValue      (BufPtr &Buf, BufPtr End, unsigned TypeID);

  /// @brief Parse a block of types.
  void ParseTypeConstants      (BufPtr &Buf, BufPtr End, TypeListTy &Tab,
					unsigned NumEntries);

  /// @brief Parse a single type.
  const Type *ParseTypeConstant(BufPtr &Buf, BufPtr End);

  /// @brief Parse a string constants block
  void ParseStringConstants    (BufPtr &Buf, BufPtr End, unsigned NumEntries);

/// @}
/// @name Data
/// @{
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

/// @}
/// @name Implementation Details
/// @{
private:
  /// This stores the parser's handler. It makes virtual function calls through
  /// the BytecodeHandler to notify the handler of parsing events. What the
  /// handler does with the events is completely orthogonal to the business of
  /// parsing the bytecode.
  /// @brief The handler of bytecode parsing events.
  BytecodeHandler* handler;
  
  /// For lazy reading-in of functions, we need to save away several pieces of
  /// information about each function: its begin and end pointer in the buffer
  /// and its FunctionSlot.
  struct LazyFunctionInfo {
    const unsigned char *Buf, *EndBuf;
    LazyFunctionInfo(const unsigned char *B = 0, const unsigned char *EB = 0)
      : Buf(B), EndBuf(EB) {}
  };
  typedef std::map<const Type*, LazyFunctionInfo> LazyFunctionMap;
  LazyFunctionMap LazyFunctionLoadMap;

private:

  static inline void readBlock(const unsigned char *&Buf,
			       const unsigned char *EndBuf, 
			       unsigned &Type, unsigned &Size) ;

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

  AbstractBytecodeParser(const AbstractBytecodeParser &);  // DO NOT IMPLEMENT
  void operator=(const AbstractBytecodeParser &);  // DO NOT IMPLEMENT

/// @}
};

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
/// @see AbstractbytecodeParser
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

  /// @brief Handle start of a compaction table plane
  virtual void handleCompactionTablePlane( 
    unsigned Ty, 
    unsigned NumEntries
  );


  /// @brief Handle a type entry in the compaction table
  virtual void handleCompactionTableType( 
    unsigned i, 
    unsigned TypSlot, 
    const Type* 
  );

  /// @brief Handle a value entry in the compaction table
  virtual void handleCompactionTableValue( 
    unsigned i, 
    unsigned ValSlot, 
    const Type* 
  );

  /// @brief Handle end of a compaction table
  virtual void handleCompactionTableEnd();

  /// @brief Handle start of a symbol table
  virtual void handleSymbolTableBegin();

  /// @brief Handle start of a symbol table plane
  virtual void handleSymbolTablePlane( 
    unsigned Ty, 
    unsigned NumEntries, 
    const Type* Ty 
  );

  /// @brief Handle a named type in the symbol table
  virtual void handleSymbolTableType( 
    unsigned i, 
    unsigned slot, 
    const std::string& name 
  );

  /// @brief Handle a named value in the symbol table
  virtual void handleSymbolTableValue( 
    unsigned i, 
    unsigned slot, 
    const std::string& name 
  );

  /// @brief Handle the end of a symbol table
  virtual void handleSymbolTableEnd();

  /// @brief Handle the beginning of a function body
  virtual void handleFunctionBegin(
    const Type* FType, 
    GlobalValue::LinkageTypes linkage 
  );

  /// @brief Handle the end of a function body
  virtual void handleFunctionEnd(
    const Type* FType
  );

  /// @brief Handle the beginning of a basic block
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

  /// @brief Handle the end of a basic block
  virtual void handleBasicBlockEnd(unsigned blocknum);

  /// @brief Handle start of global constants block.
  virtual void handleGlobalConstantsBegin();

  /// @brief Handle a constant expression
  virtual void handleConstantExpression( 
    unsigned Opcode, 
    const Type* Typ, 
    std::vector<std::pair<const Type*,unsigned> > ArgVec 
  );

  /// @brief Handle a constant array
  virtual void handleConstantArray( 
    const ArrayType* AT, 
    std::vector<unsigned>& ElementSlots
  );

  /// @brief Handle a constant structure
  virtual void handleConstantStruct(
    const StructType* ST,
    std::vector<unsigned>& ElementSlots
  );

  /// @brief Handle a constant pointer
  virtual void handleConstantPointer(
    const PointerType* PT,
    unsigned Slot
  );

  /// @brief Handle a constant strings (array special case)
  virtual void handleConstantString(
    const ConstantArray* CA
  );

  /// @brief Handle a primitive constant value
  virtual void handleConstantValue( Constant * c );

  /// @brief Handle the end of the global constants
  virtual void handleGlobalConstantsEnd();

/// @}

};

} // End llvm namespace

// vim: sw=2
#endif
