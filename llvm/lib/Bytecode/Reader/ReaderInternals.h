//===-- ReaderInternals.h - Definitions internal to the reader ---*- C++ -*--=//
//
//  This header file defines various stuff that is used by the bytecode reader.
//
//===----------------------------------------------------------------------===//

#ifndef READER_INTERNALS_H
#define READER_INTERNALS_H

#include "llvm/Bytecode/Primitives.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/Constant.h"
#include <utility>
#include <map>

// Enable to trace to figure out what the heck is going on when parsing fails
#define TRACE_LEVEL 0

#if TRACE_LEVEL    // ByteCodeReading_TRACEer
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

class BytecodeParser : public AbstractTypeUser {
  std::string Error;     // Error message string goes here...
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

  Module *ParseBytecode(const unsigned char *Buf, const unsigned char *EndBuf,
                        const std::string &ModuleID);

  std::string getError() const { return Error; }

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

  Module *TheModule;   // Current Module being read into...
  
  // Information about the module, extracted from the bytecode revision number.
  unsigned char RevisionNum;        // The rev # itself
  unsigned char FirstDerivedTyID;   // First variable index to use for type
  bool HasImplicitZeroInitializer;  // Is entry 0 of every slot implicity zeros?
  bool isBigEndian, hasLongPointers;// Information about the target compiled for
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

private:
  void freeTable(ValueTable &Tab) {
    while (!Tab.empty()) {
      delete Tab.back();
      Tab.pop_back();
    }
  }

  bool ParseModule        (const unsigned char * Buf, const unsigned char *End);
  bool ParseVersionInfo   (const unsigned char *&Buf, const unsigned char *End);
  bool ParseModuleGlobalInfo(const unsigned char *&Buf, const unsigned char *E);
  bool ParseSymbolTable   (const unsigned char *&Buf, const unsigned char *End,
                           SymbolTable *);
  bool ParseFunction      (const unsigned char *&Buf, const unsigned char *End);
  bool ParseBasicBlock    (const unsigned char *&Buf, const unsigned char *End,
                           BasicBlock *&);
  bool ParseInstruction   (const unsigned char *&Buf, const unsigned char *End,
                           Instruction *&, BasicBlock *BB /*HACK*/);
  bool ParseRawInst       (const unsigned char *&Buf, const unsigned char *End,
                           RawInst &);

  bool ParseGlobalTypes(const unsigned char *&Buf, const unsigned char *EndBuf);
  bool ParseConstantPool(const unsigned char *&Buf, const unsigned char *EndBuf,
			 ValueTable &Tab, TypeValuesListTy &TypeTab);
  bool parseConstantValue(const unsigned char *&Buf, const unsigned char *End,
                          const Type *Ty, Constant *&V);
  bool parseTypeConstants(const unsigned char *&Buf,
                          const unsigned char *EndBuf,
			  TypeValuesListTy &Tab, unsigned NumEntries);
  const Type *parseTypeConstant(const unsigned char *&Buf,
                                const unsigned char *EndBuf);

  Value      *getValue(const Type *Ty, unsigned num, bool Create = true);
  const Type *getType(unsigned ID);
  Constant   *getConstantValue(const Type *Ty, unsigned num);

  int insertValue(Value *V, ValueTable &Table);  // -1 = Failure
  void setValueTo(ValueTable &D, unsigned Slot, Value *V);
  bool postResolveValues(ValueTable &ValTab);

  bool getTypeSlot(const Type *Ty, unsigned &Slot);

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


static inline unsigned getValueIDNumberFromPlaceHolder(Value *Val) {
  if (isa<Constant>(Val))
    return ((ConstPHolder*)Val)->getID();
  
  // else discriminate by type
  switch (Val->getType()->getPrimitiveID()) {
  case Type::LabelTyID:    return ((BBPHolder*)Val)->getID();
  default:                 return ((ValPHolder*)Val)->getID();
  }
}

static inline bool readBlock(const unsigned char *&Buf,
                             const unsigned char *EndBuf, 
			     unsigned &Type, unsigned &Size) {
#if DEBUG_OUTPUT
  bool Result = read(Buf, EndBuf, Type) || read(Buf, EndBuf, Size);
  std::cerr << "StartLoc = " << ((unsigned)Buf & 4095)
       << " Type = " << Type << " Size = " << Size << endl;
  return Result;
#else
  return read(Buf, EndBuf, Type) || read(Buf, EndBuf, Size);
#endif
}

#endif
