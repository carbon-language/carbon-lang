//===-- ReaderInternals.h - Definitions internal to the reader ---*- C++ -*--=//
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
#include <memory>
class Module;

// Enable to trace to figure out what the heck is going on when parsing fails
//#define TRACE_LEVEL 10

#if TRACE_LEVEL    // ByteCodeReading_TRACEr
#define BCR_TRACE(n, X) \
    if (n < TRACE_LEVEL) std::cerr << std::string(n*2, ' ') << X
#else
#define BCR_TRACE(n, X)
#endif

struct RawInst {       // The raw fields out of the bytecode stream...
  unsigned NumOperands;
  unsigned Opcode;
  const Type *Ty;
  unsigned Arg1, Arg2;
  union {
    unsigned Arg3;
    std::vector<unsigned> *VarArgs; // Contains arg #3,4,5... if NumOperands > 3
  };
};

struct LazyFunctionInfo {
  const unsigned char *Buf, *EndBuf;
  unsigned FunctionSlot;
};

class BytecodeParser : public AbstractTypeUser, public AbstractModuleProvider {
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
    freeTable(LateResolveValues);
    freeTable(ModuleValues);
  }

  Module* releaseModule() {
    // Since we're losing control of this Module, we must hand it back complete
    materializeModule();
    freeState();
    Module *tempM = TheModule; 
    TheModule = 0; 
    return tempM; 
  }

  void ParseBytecode(const unsigned char *Buf, unsigned Length,
                     const std::string &ModuleID);

  void dump() const {
    std::cerr << "BytecodeParser instance!\n";
  }

private:          // All of this data is transient across calls to ParseBytecode
  struct ValueList : public User {
    ValueList() : User(Type::TypeTy, Value::TypeVal) {
    }
    ~ValueList() {}

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
  ValueTable Values, LateResolveValues;
  ValueTable ModuleValues;

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
  typedef std::vector<PATypeHandle> TypeValuesListTy;
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
                        SymbolTable *);
  void ParseFunction(const unsigned char *&Buf, const unsigned char *End);
  void ParseGlobalTypes(const unsigned char *&Buf, const unsigned char *EndBuf);

  std::auto_ptr<BasicBlock>
  ParseBasicBlock(const unsigned char *&Buf, const unsigned char *End);

  bool ParseInstruction   (const unsigned char *&Buf, const unsigned char *End,
                           Instruction *&);
  std::auto_ptr<RawInst> ParseRawInst(const unsigned char *&Buf,
                                      const unsigned char *End);

  void ParseConstantPool(const unsigned char *&Buf, const unsigned char *EndBuf,
                         ValueTable &Tab, TypeValuesListTy &TypeTab);
  void parseConstantValue(const unsigned char *&Buf, const unsigned char *End,
                          const Type *Ty, Constant *&V);
  void parseTypeConstants(const unsigned char *&Buf,
                          const unsigned char *EndBuf,
                          TypeValuesListTy &Tab, unsigned NumEntries);
  const Type *parseTypeConstant(const unsigned char *&Buf,
                                const unsigned char *EndBuf);

  Value      *getValue(const Type *Ty, unsigned num, bool Create = true);
  const Type *getType(unsigned ID);
  Constant   *getConstantValue(const Type *Ty, unsigned num);

  int insertValue(Value *V, ValueTable &Table);  // -1 = Failure
  void setValueTo(ValueTable &D, unsigned Slot, Value *V);
  void postResolveValues(ValueTable &ValTab);

  void getTypeSlot(const Type *Ty, unsigned &Slot);

  // resolve all references to the placeholder (if any) for the given value
  void ResolveReferencesToValue(Value *Val, unsigned Slot);

  
  // refineAbstractType - The callback method is invoked when one of the
  // elements of TypeValues becomes more concrete...
  //
  virtual void refineAbstractType(const DerivedType *OldTy, const Type *NewTy);
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

struct InstPlaceHolderHelper : public Instruction {
  InstPlaceHolderHelper(const Type *Ty) : Instruction(Ty, UserOp1, "") {}
  virtual const char *getOpcodeName() const { return "placeholder"; }

  virtual Instruction *clone() const { abort(); return 0; }
};

struct BBPlaceHolderHelper : public BasicBlock {
  BBPlaceHolderHelper(const Type *Ty) : BasicBlock() {
    assert(Ty == Type::LabelTy);
  }
};

struct ConstantPlaceHolderHelper : public Constant {
  ConstantPlaceHolderHelper(const Type *Ty)
    : Constant(Ty) {}
  virtual bool isNullValue() const { return false; }
};

typedef PlaceholderDef<InstPlaceHolderHelper>  ValPHolder;
typedef PlaceholderDef<BBPlaceHolderHelper>    BBPHolder;
typedef PlaceholderDef<ConstantPlaceHolderHelper>  ConstPHolder;

// Some common errors we find
static const std::string Error_readvbr   = "read_vbr(): error reading.";
static const std::string Error_read      = "read(): error reading.";
static const std::string Error_inputdata = "input_data(): error reading.";
static const std::string Error_DestSlot  = "No destination slot found.";

static inline unsigned getValueIDNumberFromPlaceHolder(Value *Val) {
  if (isa<Constant>(Val))
    return ((ConstPHolder*)Val)->getID();
  
  // else discriminate by type
  switch (Val->getType()->getPrimitiveID()) {
  case Type::LabelTyID:    return ((BBPHolder*)Val)->getID();
  default:                 return ((ValPHolder*)Val)->getID();
  }
}

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
