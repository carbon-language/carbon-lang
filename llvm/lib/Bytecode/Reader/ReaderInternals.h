//===-- ReaderInternals.h - Definitions internal to the reader --*- C++ -*-===//
//
//  This header file defines various stuff that is used by the bytecode reader.
//
//===----------------------------------------------------------------------===//

#ifndef READER_INTERNALS_H
#define READER_INTERNALS_H

#include "llvm/Constant.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/ModuleProvider.h"
#include "llvm/Bytecode/Primitives.h"
#include <utility>
#include <map>

// Enable to trace to figure out what the heck is going on when parsing fails
//#define TRACE_LEVEL 10

#if TRACE_LEVEL    // ByteCodeReading_TRACEr
#define BCR_TRACE(n, X) \
    if (n < TRACE_LEVEL) std::cerr << std::string(n*2, ' ') << X
#else
#define BCR_TRACE(n, X)
#endif

struct LazyFunctionInfo {
  const unsigned char *Buf, *EndBuf;
  unsigned FunctionSlot;
};

class BytecodeParser : public ModuleProvider {
  BytecodeParser(const BytecodeParser &);  // DO NOT IMPLEMENT
  void operator=(const BytecodeParser &);  // DO NOT IMPLEMENT
public:
  BytecodeParser() {
    // Define this in case we don't see a ModuleGlobalInfo block.
    FirstDerivedTyID = Type::FirstDerivedTyID;
  }
  
  ~BytecodeParser() {
    freeState();
  }
  void freeState() {
    freeTable(Values);
    freeTable(ModuleValues);
  }

  Module* releaseModule() {
    // Since we're losing control of this Module, we must hand it back complete
    Module *M = ModuleProvider::releaseModule();
    freeState();
    return M;
  }

  void ParseBytecode(const unsigned char *Buf, unsigned Length,
                     const std::string &ModuleID);

  void dump() const {
    std::cerr << "BytecodeParser instance!\n";
  }

private:
  struct ValueList : public User {
    ValueList() : User(Type::TypeTy, Value::TypeVal) {}

    // vector compatibility methods
    unsigned size() const { return getNumOperands(); }
    void push_back(Value *V) { Operands.push_back(Use(V, this)); }
    Value *back() const { return Operands.back(); }
    void pop_back() { Operands.pop_back(); }
    bool empty() const { return Operands.empty(); }

    virtual void print(std::ostream& OS) const {
      OS << "Bytecode Reader UseHandle!";
    }
  };

  // Information about the module, extracted from the bytecode revision number.
  unsigned char RevisionNum;        // The rev # itself
  unsigned char FirstDerivedTyID;   // First variable index to use for type
  bool HasImplicitZeroInitializer;  // Is entry 0 of every slot implicity zeros?
  bool hasInternalMarkerOnly;       // Only types of linkage are intern/external

  typedef std::vector<ValueList*> ValueTable;
  ValueTable Values;
  ValueTable ModuleValues;
  std::map<std::pair<unsigned,unsigned>, Value*> ForwardReferences;

  std::vector<BasicBlock*> ParsedBasicBlocks;

  // GlobalRefs - This maintains a mapping between <Type, Slot #>'s and forward
  // references to global values or constants.  Such values may be referenced
  // before they are defined, and if so, the temporary object that they
  // represent is held here.
  //
  typedef std::map<std::pair<const Type *, unsigned>, Value*>  GlobalRefsType;
  GlobalRefsType GlobalRefs;

  // TypesLoaded - This vector mirrors the Values[TypeTyID] plane.  It is used
  // to deal with forward references to types.
  //
  typedef std::vector<PATypeHolder> TypeValuesListTy;
  TypeValuesListTy ModuleTypeValues;
  TypeValuesListTy FunctionTypeValues;

  // When the ModuleGlobalInfo section is read, we create a function object for
  // each function in the module.  When the function is loaded, this function is
  // filled in.
  //
  std::vector<std::pair<Function*, unsigned> > FunctionSignatureList;

  // Constant values are read in after global variables.  Because of this, we
  // must defer setting the initializers on global variables until after module
  // level constants have been read.  In the mean time, this list keeps track of
  // what we must do.
  //
  std::vector<std::pair<GlobalVariable*, unsigned> > GlobalInits;

  // For lazy reading-in of functions, we need to save away several pieces of
  // information about each function: its begin and end pointer in the buffer
  // and its FunctionSlot.
  // 
  std::map<Function*, LazyFunctionInfo*> LazyFunctionLoadMap;
  
private:
  void freeTable(ValueTable &Tab) {
    while (!Tab.empty()) {
      delete Tab.back();
      Tab.pop_back();
    }
  }

public:
  void ParseModule(const unsigned char * Buf, const unsigned char *End);
  void materializeFunction(Function *F);

private:
  void ParseVersionInfo   (const unsigned char *&Buf, const unsigned char *End);
  void ParseModuleGlobalInfo(const unsigned char *&Buf, const unsigned char *E);
  void ParseSymbolTable(const unsigned char *&Buf, const unsigned char *End,
                        SymbolTable *, Function *CurrentFunction);
  void ParseFunction(const unsigned char *&Buf, const unsigned char *End);
  void ParseGlobalTypes(const unsigned char *&Buf, const unsigned char *EndBuf);

  BasicBlock *ParseBasicBlock(const unsigned char *&Buf,
                              const unsigned char *End,
                              unsigned BlockNo);

  Instruction *ParseInstruction(const unsigned char *&Buf,
                                const unsigned char *End,
                                std::vector<unsigned> &Args);

  void ParseConstantPool(const unsigned char *&Buf, const unsigned char *EndBuf,
                         ValueTable &Tab, TypeValuesListTy &TypeTab);
  Constant *parseConstantValue(const unsigned char *&Buf,
                               const unsigned char *End,
                               const Type *Ty);
  void parseTypeConstants(const unsigned char *&Buf,
                          const unsigned char *EndBuf,
                          TypeValuesListTy &Tab, unsigned NumEntries);
  const Type *parseTypeConstant(const unsigned char *&Buf,
                                const unsigned char *EndBuf);

  Value      *getValue(const Type *Ty, unsigned num, bool Create = true);
  Value      *getValue(unsigned TypeID, unsigned num, bool Create = true);
  const Type *getType(unsigned ID);
  BasicBlock *getBasicBlock(unsigned ID);
  Constant   *getConstantValue(const Type *Ty, unsigned num);

  unsigned insertValue(Value *V, ValueTable &Table);
  unsigned insertValue(Value *V, unsigned Type, ValueTable &Table);

  unsigned getTypeSlot(const Type *Ty);

  // resolve all references to the placeholder (if any) for the given value
  void ResolveReferencesToValue(Value *Val, unsigned Slot);
};

template<class SuperType>
class PlaceholderDef : public SuperType {
  unsigned ID;
  PlaceholderDef();                       // DO NOT IMPLEMENT
  void operator=(const PlaceholderDef &); // DO NOT IMPLEMENT
public:
  PlaceholderDef(const Type *Ty, unsigned id) : SuperType(Ty), ID(id) {}
  unsigned getID() { return ID; }
};

struct ConstantPlaceHolderHelper : public Constant {
  ConstantPlaceHolderHelper(const Type *Ty)
    : Constant(Ty) {}
  virtual bool isNullValue() const { return false; }
};

typedef PlaceholderDef<ConstantPlaceHolderHelper>  ConstPHolder;

// Some common errors we find
static const std::string Error_readvbr   = "read_vbr(): error reading.";
static const std::string Error_read      = "read(): error reading.";
static const std::string Error_inputdata = "input_data(): error reading.";
static const std::string Error_DestSlot  = "No destination slot found.";

static inline void readBlock(const unsigned char *&Buf,
                             const unsigned char *EndBuf, 
                             unsigned &Type, unsigned &Size) {
#if DEBUG_OUTPUT
  bool Result = read(Buf, EndBuf, Type) || read(Buf, EndBuf, Size);
  std::cerr << "StartLoc = " << ((unsigned)Buf & 4095)
       << " Type = " << Type << " Size = " << Size << endl;
  if (Result) throw Error_read;
#else
  if (read(Buf, EndBuf, Type) || read(Buf, EndBuf, Size)) throw Error_read;
#endif
}

#endif
