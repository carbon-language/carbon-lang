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
#define BCR_TRACE(n, X) if (n < TRACE_LEVEL) std::cerr << std::string(n*2, ' ') << X
#else
#define BCR_TRACE(n, X)
#endif

typedef unsigned char uchar;

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

  Module *ParseBytecode(const uchar *Buf, const uchar *EndBuf);

  std::string getError() const { return Error; }

  void dump() const {
    std::cerr << "BytecodeParser instance!\n";
  }

private:          // All of this data is transient across calls to ParseBytecode
  Module *TheModule;   // Current Module being read into...
  
  typedef std::vector<Value *> ValueList;
  typedef std::vector<ValueList> ValueTable;
  ValueTable Values, LateResolveValues;
  ValueTable ModuleValues, LateResolveModuleValues;

  // GlobalRefs - This maintains a mapping between <Type, Slot #>'s and forward
  // references to global values or constants.  Such values may be referenced
  // before they are defined, and if so, the temporary object that they
  // represent is held here.
  //
  typedef std::map<std::pair<const Type *, unsigned>,
                   Value*>  GlobalRefsType;
  GlobalRefsType GlobalRefs;

  // TypesLoaded - This vector mirrors the Values[TypeTyID] plane.  It is used
  // to deal with forward references to types.
  //
  typedef std::vector<PATypeHandle<Type> > TypeValuesListTy;
  TypeValuesListTy ModuleTypeValues;
  TypeValuesListTy FunctionTypeValues;

  // Information read from the ModuleGlobalInfo section of the file...
  unsigned FirstDerivedTyID;

  // When the ModuleGlobalInfo section is read, we load the type of each
  // function and the 'ModuleValues' slot that it lands in.  We then load a
  // placeholder into its slot to reserve it.  When the function is loaded, this
  // placeholder is replaced.
  //
  std::vector<std::pair<const PointerType *, unsigned> > FunctionSignatureList;

private:
  bool ParseModule          (const uchar * Buf, const uchar *End);
  bool ParseModuleGlobalInfo(const uchar *&Buf, const uchar *End);
  bool ParseSymbolTable   (const uchar *&Buf, const uchar *End, SymbolTable *);
  bool ParseFunction      (const uchar *&Buf, const uchar *End);
  bool ParseBasicBlock    (const uchar *&Buf, const uchar *End, BasicBlock *&);
  bool ParseInstruction   (const uchar *&Buf, const uchar *End, Instruction *&,
                           BasicBlock *BB /*HACK*/);
  bool ParseRawInst       (const uchar *&Buf, const uchar *End, RawInst &);

  bool ParseConstantPool(const uchar *&Buf, const uchar *EndBuf,
			 ValueTable &Tab, TypeValuesListTy &TypeTab);
  bool parseConstantValue(const uchar *&Buf, const uchar *End,
                          const Type *Ty, Constant *&V);
  bool parseTypeConstants(const uchar *&Buf, const uchar *EndBuf,
			  TypeValuesListTy &Tab, unsigned NumEntries);
  const Type *parseTypeConstant(const uchar *&Buf, const uchar *EndBuf);

  Value      *getValue(const Type *Ty, unsigned num, bool Create = true);
  const Type *getType(unsigned ID);
  Constant   *getConstantValue(const Type *Ty, unsigned num);

  int insertValue(Value *D, std::vector<ValueList> &D);  // -1 = Failure
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

struct FunctionPlaceHolderHelper : public Function {
  FunctionPlaceHolderHelper(const Type *Ty) 
    : Function(cast<const FunctionType>(Ty), true) {
  }
};

struct ConstantPlaceHolderHelper : public Constant {
  ConstantPlaceHolderHelper(const Type *Ty)
    : Constant(Ty) {}
  virtual bool isNullValue() const { return false; }
};

typedef PlaceholderDef<InstPlaceHolderHelper>  ValPHolder;
typedef PlaceholderDef<BBPlaceHolderHelper>    BBPHolder;
typedef PlaceholderDef<FunctionPlaceHolderHelper>  FunctionPHolder;
typedef PlaceholderDef<ConstantPlaceHolderHelper>  ConstPHolder;


static inline unsigned getValueIDNumberFromPlaceHolder(Value *Val) {
  if (isa<Constant>(Val))
    return ((ConstPHolder*)Val)->getID();
  
  // else discriminate by type
  switch (Val->getType()->getPrimitiveID()) {
  case Type::LabelTyID:    return ((BBPHolder*)Val)->getID();
  case Type::FunctionTyID: return ((FunctionPHolder*)Val)->getID();
  default:                 return ((ValPHolder*)Val)->getID();
  }
}

static inline bool readBlock(const uchar *&Buf, const uchar *EndBuf, 
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
