//===-- ReaderInternals.h - Definitions internal to the reader ---*- C++ -*--=//
//
//  This header file defines various stuff that is used by the bytecode reader.
//
//===----------------------------------------------------------------------===//

#ifndef READER_INTERNALS_H
#define READER_INTERNALS_H

#include "llvm/Bytecode/Primitives.h"
#include "llvm/SymTabValue.h"
#include "llvm/Method.h"
#include "llvm/Instruction.h"
#include <map>
#include <utility>
#include <list>

// Enable to trace to figure out what the heck is going on when parsing fails
#define TRACE_LEVEL 0

#if TRACE_LEVEL    // ByteCodeReading_TRACEer
#include "llvm/Assembly/Writer.h"
#define BCR_TRACE(n, X) if (n < TRACE_LEVEL) cerr << string(n*2, ' ') << X
#else
#define BCR_TRACE(n, X)
#endif

class BasicBlock;
class Method;
class Module;
class Type;

typedef unsigned char uchar;

struct RawInst {       // The raw fields out of the bytecode stream...
  unsigned NumOperands;
  unsigned Opcode;
  const Type *Ty;
  unsigned Arg1, Arg2;
  union {
    unsigned Arg3;
    vector<unsigned> *VarArgs;   // Contains arg #3,4,5... if NumOperands > 3
  };
};

class BytecodeParser : public AbstractTypeUser {
public:
  BytecodeParser() {
    // Define this in case we don't see a ModuleGlobalInfo block.
    FirstDerivedTyID = Type::FirstDerivedTyID;
  }

  Module *ParseBytecode(const uchar *Buf, const uchar *EndBuf);
private:          // All of this data is transient across calls to ParseBytecode
  typedef vector<Value *> ValueList;
  typedef vector<ValueList> ValueTable;
  ValueTable Values, LateResolveValues;
  ValueTable ModuleValues, LateResolveModuleValues;

  // TypesLoaded - This vector mirrors the Values[TypeTyID] plane.  It is used
  // to deal with forward references to types.
  //
  typedef vector<PATypeHandle<Type> > TypeValuesListTy;
  TypeValuesListTy ModuleTypeValues;
  TypeValuesListTy MethodTypeValues;

  // Information read from the ModuleGlobalInfo section of the file...
  unsigned FirstDerivedTyID;

  // When the ModuleGlobalInfo section is read, we load the type of each method
  // and the 'ModuleValues' slot that it lands in.  We then load a placeholder
  // into its slot to reserve it.  When the method is loaded, this placeholder
  // is replaced.
  //
  list<pair<const PointerType *, unsigned> > MethodSignatureList;

private:
  bool ParseModule            (const uchar * Buf, const uchar *End, Module *&);
  bool ParseModuleGlobalInfo  (const uchar *&Buf, const uchar *End, Module *);
  bool ParseSymbolTable   (const uchar *&Buf, const uchar *End, SymbolTable *);
  bool ParseMethod        (const uchar *&Buf, const uchar *End, Module *);
  bool ParseBasicBlock    (const uchar *&Buf, const uchar *End, BasicBlock *&);
  bool ParseInstruction   (const uchar *&Buf, const uchar *End, Instruction *&);
  bool ParseRawInst       (const uchar *&Buf, const uchar *End, RawInst &);

  bool ParseConstantPool(const uchar *&Buf, const uchar *EndBuf,
			 ValueTable &Tab, TypeValuesListTy &TypeTab);
  bool parseConstPoolValue(const uchar *&Buf, const uchar *End,
			   const Type *Ty, ConstPoolVal *&V);
  bool parseTypeConstants(const uchar *&Buf, const uchar *EndBuf,
			  TypeValuesListTy &Tab, unsigned NumEntries);
  const Type *parseTypeConstant(const uchar *&Buf, const uchar *EndBuf);

  Value      *getValue(const Type *Ty, unsigned num, bool Create = true);
  const Type *getType(unsigned ID);

  bool insertValue(Value *D, vector<ValueList> &D);
  bool postResolveValues(ValueTable &ValTab);

  bool getTypeSlot(const Type *Ty, unsigned &Slot);


  // refineAbstractType - The callback method is invoked when one of the
  // elements of TypeValues becomes more concrete...
  //
  virtual void refineAbstractType(const DerivedType *OldTy, const Type *NewTy);
};

template<class SuperType>
class PlaceholderDef : public SuperType {
  unsigned ID;
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
    assert(Ty->isLabelType());
  }
};

struct MethPlaceHolderHelper : public Method {
  MethPlaceHolderHelper(const Type *Ty) 
    : Method(cast<const MethodType>(Ty)) {
  }
};

typedef PlaceholderDef<InstPlaceHolderHelper>  DefPHolder;
typedef PlaceholderDef<BBPlaceHolderHelper>    BBPHolder;
typedef PlaceholderDef<MethPlaceHolderHelper>  MethPHolder;

static inline unsigned getValueIDNumberFromPlaceHolder(Value *Def) {
  switch (Def->getType()->getPrimitiveID()) {
  case Type::LabelTyID:  return ((BBPHolder*)Def)->getID();
  case Type::MethodTyID: return ((MethPHolder*)Def)->getID();
  default:               return ((DefPHolder*)Def)->getID();
  }
}

static inline bool readBlock(const uchar *&Buf, const uchar *EndBuf, 
			     unsigned &Type, unsigned &Size) {
#if DEBUG_OUTPUT
  bool Result = read(Buf, EndBuf, Type) || read(Buf, EndBuf, Size);
  cerr << "StartLoc = " << ((unsigned)Buf & 4095)
       << " Type = " << Type << " Size = " << Size << endl;
  return Result;
#else
  return read(Buf, EndBuf, Type) || read(Buf, EndBuf, Size);
#endif
}


// failure Template - This template function is used as a place to put
// breakpoints in to debug failures of the bytecode parser.
//
template <typename X>
static X failure(X Value) {
  return Value;
}

#endif
