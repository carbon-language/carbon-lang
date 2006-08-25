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
#include "llvm/Bytecode/Analyzer.h"
#include <utility>
#include <map>
#include <setjmp.h>

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
  BytecodeReader(BytecodeHandler* h = 0) {
    decompressedBlock = 0;
    Handler = h;
  }

  ~BytecodeReader() {
    freeState();
    if (decompressedBlock) {
      ::free(decompressedBlock);
      decompressedBlock = 0;
    }
  }

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
  class ValueList : public User {
    std::vector<Use> Uses;
  public:
    ValueList() : User(Type::VoidTy, Value::ArgumentVal, 0, 0) {}

    // vector compatibility methods
    unsigned size() const { return getNumOperands(); }
    void push_back(Value *V) {
      Uses.push_back(Use(V, this));
      OperandList = &Uses[0];
      ++NumOperands;
    }
    Value *back() const { return Uses.back(); }
    void pop_back() { Uses.pop_back(); --NumOperands; }
    bool empty() const { return NumOperands == 0; }
    virtual void print(std::ostream& os) const {
      for (unsigned i = 0; i < size(); ++i) {
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
  typedef std::map<std::pair<unsigned,unsigned>, Constant*> ConstantRefsType;

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
  /// @returns true if an error occurred
  /// @brief Main interface to parsing a bytecode buffer.
  bool ParseBytecode(
     const unsigned char *Buf,    ///< Beginning of the bytecode buffer
     unsigned Length,             ///< Length of the bytecode buffer
     const std::string &ModuleID, ///< An identifier for the module constructed.
     std::string* ErrMsg = 0      ///< Optional place for error message 
  );

  /// @brief Parse all function bodies
  bool ParseAllFunctionBodies(std::string* ErrMsg);

  /// @brief Parse the next function of specific type
  bool ParseFunction(Function* Func, std::string* ErrMsg) ;

  /// This method is abstract in the parent ModuleProvider class. Its
  /// implementation is identical to the ParseFunction method.
  /// @see ParseFunction
  /// @brief Make a specific function materialize.
  virtual bool materializeFunction(Function *F, std::string *ErrMsg = 0) {
    LazyFunctionMap::iterator Fi = LazyFunctionLoadMap.find(F);
    if (Fi == LazyFunctionLoadMap.end()) 
      return false;
    if (ParseFunction(F,ErrMsg))
      return true;
    return false;
  }

  /// This method is abstract in the parent ModuleProvider class. Its
  /// implementation is identical to ParseAllFunctionBodies.
  /// @see ParseAllFunctionBodies
  /// @brief Make the whole module materialize
  virtual Module* materializeModule(std::string *ErrMsg = 0) {
    if (ParseAllFunctionBodies(ErrMsg))
      return 0;
    return TheModule;
  }

  /// This method is provided by the parent ModuleProvde class and overriden
  /// here. It simply releases the module from its provided and frees up our
  /// state.
  /// @brief Release our hold on the generated module
  Module* releaseModule(std::string *ErrInfo = 0) {
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

  /// @brief Parse functions lazily.
  void ParseFunctionLazily();

  ///  @brief Parse a function body
  void ParseFunctionBody(Function* Func);

  /// @brief Parse the type list portion of a compaction table
  void ParseCompactionTypes(unsigned NumEntries);

  /// @brief Parse a compaction table
  void ParseCompactionTable();

  /// @brief Parse global types
  void ParseGlobalTypes();

  /// @brief Parse a basic block (for LLVM 1.0 basic block blocks)
  BasicBlock* ParseBasicBlock(unsigned BlockNo);

  /// @brief parse an instruction list (for post LLVM 1.0 instruction lists
  /// with blocks differentiated by terminating instructions.
  unsigned ParseInstructionList(
    Function* F   ///< The function into which BBs will be inserted
  );

  /// @brief Parse a single instruction.
  void ParseInstruction(
    std::vector<unsigned>& Args,   ///< The arguments to be filled in
    BasicBlock* BB             ///< The BB the instruction goes in
  );

  /// @brief Parse the whole constant pool
  void ParseConstantPool(ValueTable& Values, TypeListTy& Types,
                         bool isFunction);

  /// @brief Parse a single constant pool value
  Value *ParseConstantPoolValue(unsigned TypeID);

  /// @brief Parse a block of types constants
  void ParseTypes(TypeListTy &Tab, unsigned NumEntries);

  /// @brief Parse a single type constant
  const Type *ParseType();

  /// @brief Parse a string constants block
  void ParseStringConstants(unsigned NumEntries, ValueTable &Tab);

/// @}
/// @name Data
/// @{
private:
  std::string ErrorMsg; ///< A place to hold an error message through longjmp
  jmp_buf context;      ///< Where to return to if an error occurs.
  char*  decompressedBlock; ///< Result of decompression
  BufPtr MemStart;     ///< Start of the memory buffer
  BufPtr MemEnd;       ///< End of the memory buffer
  BufPtr BlockStart;   ///< Start of current block being parsed
  BufPtr BlockEnd;     ///< End of current block being parsed
  BufPtr At;           ///< Where we're currently parsing at

  /// Information about the module, extracted from the bytecode revision number.
  ///
  unsigned char RevisionNum;        // The rev # itself

  /// Flags to distinguish LLVM 1.0 & 1.1 bytecode formats (revision #0)

  /// Revision #0 had an explicit alignment of data only for the
  /// ModuleGlobalInfo block.  This was fixed to be like all other blocks in 1.2
  bool hasInconsistentModuleGlobalInfo;

  /// Revision #0 also explicitly encoded zero values for primitive types like
  /// int/sbyte/etc.
  bool hasExplicitPrimitiveZeros;

  // Flags to control features specific the LLVM 1.2 and before (revision #1)

  /// LLVM 1.2 and earlier required that getelementptr structure indices were
  /// ubyte constants and that sequential type indices were longs.
  bool hasRestrictedGEPTypes;

  /// LLVM 1.2 and earlier had class Type deriving from Value and the Type
  /// objects were located in the "Type Type" plane of various lists in read
  /// by the bytecode reader. In LLVM 1.3 this is no longer the case. Types are
  /// completely distinct from Values. Consequently, Types are written in fixed
  /// locations in LLVM 1.3. This flag indicates that the older Type derived
  /// from Value style of bytecode file is being read.
  bool hasTypeDerivedFromValue;

  /// LLVM 1.2 and earlier encoded block headers as two uint (8 bytes), one for
  /// the size and one for the type. This is a bit wasteful, especially for
  /// small files where the 8 bytes per block is a large fraction of the total
  /// block size. In LLVM 1.3, the block type and length are encoded into a
  /// single uint32 by restricting the number of block types (limit 31) and the
  /// maximum size of a block (limit 2^27-1=134,217,727). Note that the module
  /// block still uses the 8-byte format so the maximum size of a file can be
  /// 2^32-1 bytes long.
  bool hasLongBlockHeaders;

  /// LLVM 1.2 and earlier wrote type slot numbers as vbr_uint32. In LLVM 1.3
  /// this has been reduced to vbr_uint24. It shouldn't make much difference
  /// since we haven't run into a module with > 24 million types, but for safety
  /// the 24-bit restriction has been enforced in 1.3 to free some bits in
  /// various places and to ensure consistency. In particular, global vars are
  /// restricted to 24-bits.
  bool has32BitTypes;

  /// LLVM 1.2 and earlier did not provide a target triple nor a list of
  /// libraries on which the bytecode is dependent. LLVM 1.3 provides these
  /// features, for use in future versions of LLVM.
  bool hasNoDependentLibraries;

  /// LLVM 1.3 and earlier caused blocks and other fields to start on 32-bit
  /// aligned boundaries. This can lead to as much as 30% bytecode size overhead
  /// in various corner cases (lots of long instructions). In LLVM 1.4,
  /// alignment of bytecode fields was done away with completely.
  bool hasAlignment;

  // In version 4 and earlier, the bytecode format did not support the 'undef'
  // constant.
  bool hasNoUndefValue;

  // In version 4 and earlier, the bytecode format did not save space for flags
  // in the global info block for functions.
  bool hasNoFlagsForFunctions;

  // In version 4 and earlier, there was no opcode space reserved for the
  // unreachable instruction.
  bool hasNoUnreachableInst;

  /// In release 1.7 we changed intrinsic functions to not be overloaded. There
  /// is no bytecode change for this, but to optimize the auto-upgrade of calls
  /// to intrinsic functions, we save a mapping of old function definitions to
  /// the new ones so call instructions can be upgraded efficiently.
  std::map<Function*,Function*> upgradedFunctions;

  /// CompactionTypes - If a compaction table is active in the current function,
  /// this is the mapping that it contains.  We keep track of what resolved type
  /// it is as well as what global type entry it is.
  std::vector<std::pair<const Type*, unsigned> > CompactionTypes;

  /// @brief If a compaction table is active in the current function,
  /// this is the mapping that it contains.
  std::vector<std::vector<Value*> > CompactionValues;

  /// @brief This vector is used to deal with forward references to types in
  /// a module.
  TypeListTy ModuleTypes;
  
  /// @brief This is an inverse mapping of ModuleTypes from the type to an
  /// index.  Because refining types causes the index of this map to be
  /// invalidated, any time we refine a type, we clear this cache and recompute
  /// it next time we need it.  These entries are ordered by the pointer value.
  std::vector<std::pair<const Type*, unsigned> > ModuleTypeIDCache;

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

  /// This maintains a mapping between <Type, Slot #>'s and forward references
  /// to constants.  Such values may be referenced before they are defined, and
  /// if so, the temporary object that they represent is held here.  @brief
  /// Temporary place for forward references to constants.
  ConstantRefsType ConstantFwdRefs;

  /// Constant values are read in after global variables.  Because of this, we
  /// must defer setting the initializers on global variables until after module
  /// level constants have been read.  In the mean time, this list keeps track
  /// of what we must do.
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

  /// @brief Converts a pre-sanitized type slot number to its Type* and
  /// sanitizes the type id.
  inline const Type* getSanitizedType(unsigned& ID );

  /// @brief Read in and get a sanitized type id
  inline const Type* readSanitizedType();

  /// @brief Converts a Type* to its type slot number
  unsigned getTypeSlot(const Type *Ty);

  /// @brief Converts a normal type slot number to a compacted type slot num.
  unsigned getCompactionTypeSlot(unsigned type);

  /// @brief Gets the global type corresponding to the TypeId
  const Type *getGlobalTableType(unsigned TypeId);

  /// This is just like getTypeSlot, but when a compaction table is in use,
  /// it is ignored.
  unsigned getGlobalTableTypeSlot(const Type *Ty);

  /// @brief Get a value from its typeid and slot number
  Value* getValue(unsigned TypeID, unsigned num, bool Create = true);

  /// @brief Get a value from its type and slot number, ignoring compaction
  /// tables.
  Value *getGlobalTableValue(unsigned TyID, unsigned SlotNo);

  /// @brief Get a basic block for current function
  BasicBlock *getBasicBlock(unsigned ID);

  /// @brief Get a constant value from its typeid and value slot.
  Constant* getConstantValue(unsigned typeSlot, unsigned valSlot);

  /// @brief Convenience function for getting a constant value when
  /// the Type has already been resolved.
  Constant* getConstantValue(const Type *Ty, unsigned valSlot) {
    return getConstantValue(getTypeSlot(Ty), valSlot);
  }

  /// @brief Insert a newly created value
  unsigned insertValue(Value *V, unsigned Type, ValueTable &Table);

  /// @brief Insert the arguments of a function.
  void insertArguments(Function* F );

  /// @brief Resolve all references to the placeholder (if any) for the
  /// given constant.
  void ResolveReferencesToConstant(Constant *C, unsigned Typ, unsigned Slot);

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

  inline void error(const std::string& errmsg);

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

  /// @brief Read an unsigned integer of no more than 24-bits with variable
  /// bit rate encoding.
  inline unsigned read_vbr_uint24();

  /// @brief Read an unsigned 64-bit integer with variable bit rate encoding.
  inline uint64_t read_vbr_uint64();

  /// @brief Read a signed 64-bit integer with variable bit rate encoding.
  inline int64_t read_vbr_int64();

  /// @brief Read a string
  inline std::string read_str();

  /// @brief Read a float value
  inline void read_float(float& FloatVal);

  /// @brief Read a double value
  inline void read_double(double& DoubleVal);

  /// @brief Read an arbitrary data chunk of fixed length
  inline void read_data(void *Ptr, void *End);

  /// @brief Read a bytecode block header
  inline void read_block(unsigned &Type, unsigned &Size);

  /// @brief Read a type identifier and sanitize it.
  inline bool read_typeid(unsigned &TypeId);

  /// @brief Recalculate type ID for pre 1.3 bytecode files.
  inline bool sanitizeTypeId(unsigned &TypeId );
/// @}
};

/// @brief A function for creating a BytecodeAnalzer as a handler
/// for the Bytecode reader.
BytecodeHandler* createBytecodeAnalyzerHandler(BytecodeAnalysis& bca,
                                               std::ostream* output );


} // End llvm namespace

// vim: sw=2
#endif
