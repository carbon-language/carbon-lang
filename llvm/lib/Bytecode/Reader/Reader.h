//===-- Reader.h - Interface To Bytecode Reading ----------------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencer and is distributed under the 
// University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
//  This header file defines the interface to the Bytecode Reader which is 
//  responsible for correctly interpreting bytecode files (backwards compatible)
//  and materializing a module from the bytecode read.
//
//===----------------------------------------------------------------------===//

#ifndef BYTECODE_PARSER_H
#define BYTECODE_PARSER_H

#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/GlobalValue.h"
#include "llvm/Function.h"
#include "llvm/ModuleProvider.h"
#include <utility>
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
/// @brief Bytecode Reader interface
class BytecodeReader : public ModuleProvider {

/// @name Constructors
/// @{
public:
  /// @brief Default constructor. By default, no handler is used.
  BytecodeReader( 
    BytecodeHandler* h = 0
  ) { 
    Handler = h; 
  }

  ~BytecodeReader() { freeState(); }

/// @}
/// @name Types
/// @{
public:
  /// @brief A convenience type for the buffer pointer
  typedef const unsigned char* BufPtr;

  /// @brief The type used for a vector of potentially abstract types
  typedef std::vector<PATypeHolder> TypeListTy;

  /// This type provides a vector of Value* via the User class for
  /// storage of Values that have been constructed when reading the
  /// bytecode. Because of forward referencing, constant replacement
  /// can occur so we ensure that our list of Value* is updated
  /// properly through those transitions. This ensures that the
  /// correct Value* is in our list when it comes time to associate
  /// constants with global variables at the end of reading the
  /// globals section.
  /// @brief A list of values as a User of those Values.
  struct ValueList : public User {
    ValueList() : User(Type::TypeTy, Value::TypeVal) {}

    // vector compatibility methods
    unsigned size() const { return getNumOperands(); }
    void push_back(Value *V) { Operands.push_back(Use(V, this)); }
    Value *back() const { return Operands.back(); }
    void pop_back() { Operands.pop_back(); }
    bool empty() const { return Operands.empty(); }
    // must override this 
    virtual void print(std::ostream& os) const {
      for ( unsigned i = 0; i < size(); i++ ) {
	os << i << " ";
	getOperand(i)->print(os);
	os << "\n";
      }
    }
  };

  /// @brief A 2 dimensional table of values
  typedef std::vector<ValueList*> ValueTable;

  /// This map is needed so that forward references to constants can be looked 
  /// up by Type and slot number when resolving those references.
  /// @brief A mapping of a Type/slot pair to a Constant*.
  typedef std::map<std::pair<const Type*,unsigned>, Constant*> ConstantRefsType;

  /// For lazy read-in of functions, we need to save the location in the
  /// data stream where the function is located. This structure provides that
  /// information. Lazy read-in is used mostly by the JIT which only wants to
  /// resolve functions as it needs them. 
  /// @brief Keeps pointers to function contents for later use.
  struct LazyFunctionInfo {
    const unsigned char *Buf, *EndBuf;
    LazyFunctionInfo(const unsigned char *B = 0, const unsigned char *EB = 0)
      : Buf(B), EndBuf(EB) {}
  };

  /// @brief A mapping of functions to their LazyFunctionInfo for lazy reading.
  typedef std::map<Function*, LazyFunctionInfo> LazyFunctionMap;

  /// @brief A list of global variables and the slot number that initializes
  /// them.
  typedef std::vector<std::pair<GlobalVariable*, unsigned> > GlobalInitsList;

  /// This type maps a typeslot/valueslot pair to the corresponding Value*.
  /// It is used for dealing with forward references as values are read in.
  /// @brief A map for dealing with forward references of values.
  typedef std::map<std::pair<unsigned,unsigned>,Value*> ForwardReferenceMap;

/// @}
/// @name Methods
/// @{
public:
  /// This function completely parses a bytecode buffer given by the \p Buf
  /// and \p Length parameters. The
  /// @brief Main interface to parsing a bytecode buffer.
  void ParseBytecode(
     const unsigned char *Buf,   ///< Beginning of the bytecode buffer
     unsigned Length,            ///< Length of the bytecode buffer
     const std::string &ModuleID ///< An identifier for the module constructed.
  );

  /// The ParseAllFunctionBodies method parses through all the previously
  /// unparsed functions in the bytecode file. If you want to completely parse
  /// a bytecode file, this method should be called after Parsebytecode because
  /// Parsebytecode only records the locations in the bytecode file of where
  /// the function definitions are located. This function uses that information
  /// to materialize the functions.
  /// @see ParseBytecode
  /// @brief Parse all function bodies
  void ParseAllFunctionBodies  ();

  /// The ParserFunction method lazily parses one function. Use this method to 
  /// casue the parser to parse a specific function in the module. Note that 
  /// this will remove the function from what is to be included by 
  /// ParseAllFunctionBodies.
  /// @see ParseAllFunctionBodies
  /// @see ParseBytecode
  /// @brief Parse the next function of specific type
  void ParseFunction (Function* Func) ;

  /// This method is abstract in the parent ModuleProvider class. Its
  /// implementation is identical to the ParseFunction method.
  /// @see ParseFunction
  /// @brief Make a specific function materialize.
  virtual void materializeFunction(Function *F) {
    LazyFunctionMap::iterator Fi = LazyFunctionLoadMap.find(F);
    if (Fi == LazyFunctionLoadMap.end()) return;
    ParseFunction(F);
  }

  /// This method is abstract in the parent ModuleProvider class. Its
  /// implementation is identical to ParseAllFunctionBodies. 
  /// @see ParseAllFunctionBodies
  /// @brief Make the whole module materialize
  virtual Module* materializeModule() {
    ParseAllFunctionBodies();
    return TheModule;
  }

  /// This method is provided by the parent ModuleProvde class and overriden
  /// here. It simply releases the module from its provided and frees up our
  /// state.
  /// @brief Release our hold on the generated module
  Module* releaseModule() {
    // Since we're losing control of this Module, we must hand it back complete
    Module *M = ModuleProvider::releaseModule();
    freeState();
    return M;
  }

/// @}
/// @name Parsing Units For Subclasses
/// @{
protected:
  /// @brief Parse whole module scope
  void ParseModule();

  /// @brief Parse the version information block
  void ParseVersionInfo();

  /// @brief Parse the ModuleGlobalInfo block
  void ParseModuleGlobalInfo();

  /// @brief Parse a symbol table
  void ParseSymbolTable( Function* Func, SymbolTable *ST);

  /// This function parses LLVM functions lazily. It obtains the type of the
  /// function and records where the body of the function is in the bytecode
  /// buffer. The caller can then use the ParseNextFunction and 
  /// ParseAllFunctionBodies to get handler events for the functions.
  /// @brief Parse functions lazily.
  void ParseFunctionLazily();

  ///  @brief Parse a function body
  void ParseFunctionBody(Function* Func);

  /// @brief Parse a compaction table
  void ParseCompactionTable();

  /// @brief Parse global types
  void ParseGlobalTypes();

  /// @returns The basic block constructed.
  /// @brief Parse a basic block (for LLVM 1.0 basic block blocks)
  BasicBlock* ParseBasicBlock(unsigned BlockNo);

  /// @returns Rhe number of basic blocks encountered.
  /// @brief parse an instruction list (for post LLVM 1.0 instruction lists
  /// with blocks differentiated by terminating instructions.
  unsigned ParseInstructionList(
    Function* F   ///< The function into which BBs will be inserted
  );
  
  /// This method parses a single instruction. The instruction is
  /// inserted at the end of the \p BB provided. The arguments of
  /// the instruction are provided in the \p Args vector.
  /// @brief Parse a single instruction.
  void ParseInstruction(
    std::vector<unsigned>& Args,   ///< The arguments to be filled in
    BasicBlock* BB             ///< The BB the instruction goes in
  );

  /// @brief Parse the whole constant pool
  void ParseConstantPool(ValueTable& Values, TypeListTy& Types);

  /// @brief Parse a single constant value
  Constant* ParseConstantValue(unsigned TypeID);

  /// @brief Parse a block of types constants
  void ParseTypeConstants(TypeListTy &Tab, unsigned NumEntries);

  /// @brief Parse a single type constant
  const Type *ParseTypeConstant();

  /// @brief Parse a string constants block
  void ParseStringConstants(unsigned NumEntries, ValueTable &Tab);

/// @}
/// @name Data
/// @{
private:
  BufPtr MemStart;     ///< Start of the memory buffer
  BufPtr MemEnd;       ///< End of the memory buffer
  BufPtr BlockStart;   ///< Start of current block being parsed
  BufPtr BlockEnd;     ///< End of current block being parsed
  BufPtr At;           ///< Where we're currently parsing at

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
  std::vector<const Type*> CompactionTypes;

  /// @brief If a compaction table is active in the current function,
  /// this is the mapping that it contains.
  std::vector<std::vector<Value*> > CompactionValues;

  /// @brief This vector is used to deal with forward references to types in
  /// a module.
  TypeListTy ModuleTypes;

  /// @brief This vector is used to deal with forward references to types in
  /// a function.
  TypeListTy FunctionTypes;

  /// When the ModuleGlobalInfo section is read, we create a Function object
  /// for each function in the module. When the function is loaded, after the
  /// module global info is read, this Function is populated. Until then, the
  /// functions in this vector just hold the function signature.
  std::vector<Function*> FunctionSignatureList;

  /// @brief This is the table of values belonging to the current function
  ValueTable FunctionValues;

  /// @brief This is the table of values belonging to the module (global)
  ValueTable ModuleValues;

  /// @brief This keeps track of function level forward references.
  ForwardReferenceMap ForwardReferences;

  /// @brief The basic blocks we've parsed, while parsing a function.
  std::vector<BasicBlock*> ParsedBasicBlocks;

  /// This maintains a mapping between <Type, Slot #>'s and
  /// forward references to constants.  Such values may be referenced before they
  /// are defined, and if so, the temporary object that they represent is held
  /// here.
  /// @brief Temporary place for forward references to constants.
  ConstantRefsType ConstantFwdRefs;

  /// Constant values are read in after global variables.  Because of this, we
  /// must defer setting the initializers on global variables until after module
  /// level constants have been read.  In the mean time, this list keeps track of
  /// what we must do.
  GlobalInitsList GlobalInits;

  // For lazy reading-in of functions, we need to save away several pieces of
  // information about each function: its begin and end pointer in the buffer
  // and its FunctionSlot.
  LazyFunctionMap LazyFunctionLoadMap;

  /// This stores the parser's handler which is used for handling tasks other 
  /// just than reading bytecode into the IR. If this is non-null, calls on 
  /// the (polymorphic) BytecodeHandler interface (see llvm/Bytecode/Handler.h) 
  /// will be made to report the logical structure of the bytecode file. What 
  /// the handler does with the events it receives is completely orthogonal to 
  /// the business of parsing the bytecode and building the IR.  This is used,
  /// for example, by the llvm-abcd tool for analysis of byte code.
  /// @brief Handler for parsing events.
  BytecodeHandler* Handler;

/// @}
/// @name Implementation Details
/// @{
private:
  /// @brief Determines if this module has a function or not.
  bool hasFunctions() { return ! FunctionSignatureList.empty(); }

  /// @brief Determines if the type id has an implicit null value.
  bool hasImplicitNull(unsigned TyID );

  /// @brief Converts a type slot number to its Type*
  const Type *getType(unsigned ID);

  /// @brief Converts a Type* to its type slot number
  unsigned getTypeSlot(const Type *Ty);

  /// @brief Converts a normal type slot number to a compacted type slot num.
  unsigned getCompactionTypeSlot(unsigned type);

  /// This is just like getType, but when a compaction table is in use, it is 
  /// ignored.  Also, no forward references or other fancy features are 
  /// supported.
  const Type *getGlobalTableType(unsigned Slot);

  /// This is just like getTypeSlot, but when a compaction table is in use,
  /// it is ignored. 
  unsigned getGlobalTableTypeSlot(const Type *Ty);
  
  /// Retrieve a value of a given type and slot number, possibly creating 
  /// it if it doesn't already exist. 
  Value* getValue(unsigned TypeID, unsigned num, bool Create = true);

  /// This is just like getValue, but when a compaction table is in use, it 
  /// is ignored.  Also, no forward references or other fancy features are 
  /// supported.
  Value *getGlobalTableValue(const Type *Ty, unsigned SlotNo);

  /// This function is used when construction phi, br, switch, and other 
  /// instructions that reference basic blocks. Blocks are numbered 
  /// sequentially as they appear in the function.
  /// @brief Get a basic block for current function
  BasicBlock *getBasicBlock(unsigned ID);

  /// Just like getValue, except that it returns a null pointer
  /// only on error.  It always returns a constant (meaning that if the value is
  /// defined, but is not a constant, that is an error).  If the specified
  /// constant hasn't been parsed yet, a placeholder is defined and used.  
  /// Later, after the real value is parsed, the placeholder is eliminated.
  Constant* getConstantValue(unsigned typeSlot, unsigned valSlot);

  /// @brief Convenience function for getting a constant value when
  /// the Type has already been resolved.
  Constant* getConstantValue(const Type *Ty, unsigned valSlot) {
    return getConstantValue(getTypeSlot(Ty), valSlot);
  }

  /// As values are created, they are inserted into the appropriate place
  /// with this method. The ValueTable argument must be one of ModuleValues
  /// or FunctionValues data members of this class.
  /// @brief Insert a newly created value
  unsigned insertValue(Value *V, unsigned Type, ValueTable &Table);

  /// @brief Insert the arguments of a function.
  void insertArguments(Function* F );

  /// @brief Resolve all references to the placeholder (if any) for the 
  /// given constant.
  void ResolveReferencesToConstant(Constant *C, unsigned Slot);

  /// @brief Release our memory.
  void freeState() {
    freeTable(FunctionValues);
    freeTable(ModuleValues);
  }

  /// @brief Free a table, making sure to free the ValueList in the table.
  void freeTable(ValueTable &Tab) {
    while (!Tab.empty()) {
      delete Tab.back();
      Tab.pop_back();
    }
  }

  BytecodeReader(const BytecodeReader &);  // DO NOT IMPLEMENT
  void operator=(const BytecodeReader &);  // DO NOT IMPLEMENT

/// @}
/// @name Reader Primitives
/// @{
private:

  /// @brief Is there more to parse in the current block?
  inline bool moreInBlock();

  /// @brief Have we read past the end of the block
  inline void checkPastBlockEnd(const char * block_name);

  /// @brief Align to 32 bits
  inline void align32();

  /// @brief Read an unsigned integer as 32-bits
  inline unsigned read_uint();

  /// @brief Read an unsigned integer with variable bit rate encoding
  inline unsigned read_vbr_uint();

  /// @brief Read an unsigned 64-bit integer with variable bit rate encoding.
  inline uint64_t read_vbr_uint64();

  /// @brief Read a signed 64-bit integer with variable bit rate encoding.
  inline int64_t read_vbr_int64();

  /// @brief Read a string
  inline std::string read_str();

  /// @brief Read an arbitrary data chunk of fixed length
  inline void read_data(void *Ptr, void *End);

  /// Read a bytecode block header
  inline void read_block(unsigned &Type, unsigned &Size);

/// @}
};

} // End llvm namespace

// vim: sw=2
#endif
