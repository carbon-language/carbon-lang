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

class BytecodeParser {
public:
  BytecodeParser() {
    // Define this in case we don't see a ModuleGlobalInfo block.
    FirstDerivedTyID = Type::FirstDerivedTyID;
  }

  Module *ParseBytecode(const uchar *Buf, const uchar *EndBuf);
private:          // All of this data is transient across calls to ParseBytecode
  typedef vector<Value *> ValueList;
  typedef vector<ValueList> ValueTable;
  typedef map<const Type *, unsigned> TypeMapType;
  ValueTable Values, LateResolveValues;
  ValueTable ModuleValues, LateResolveModuleValues;
  TypeMapType TypeMap;

  // Information read from the ModuleGlobalInfo section of the file...
  unsigned FirstDerivedTyID;

  // When the ModuleGlobalInfo section is read, we load the type of each method
  // and the 'ModuleValues' slot that it lands in.  We then load a placeholder
  // into its slot to reserve it.  When the method is loaded, this placeholder
  // is replaced.
  //
  list<pair<const MethodType *, unsigned> > MethodSignatureList;

private:
  bool ParseModule            (const uchar * Buf, const uchar *End, Module *&);
  bool ParseModuleGlobalInfo  (const uchar *&Buf, const uchar *End, Module *);
  bool ParseSymbolTable       (const uchar *&Buf, const uchar *End);
  bool ParseMethod            (const uchar *&Buf, const uchar *End, Module *);
  bool ParseBasicBlock    (const uchar *&Buf, const uchar *End, BasicBlock *&);
  bool ParseInstruction   (const uchar *&Buf, const uchar *End, Instruction *&);
  bool ParseRawInst       (const uchar *&Buf, const uchar *End, RawInst &);

  bool ParseConstantPool(const uchar *&Buf, const uchar *EndBuf,
			 SymTabValue::ConstantPoolType &CP, ValueTable &Tab);


  bool parseConstPoolValue(const uchar *&Buf, const uchar *End,
			   const Type *Ty, ConstPoolVal *&V);
  bool parseTypeConstant  (const uchar *&Buf, const uchar *, ConstPoolVal *&);

  Value      *getValue(const Type *Ty, unsigned num, bool Create = true);
  const Type *getType(unsigned ID);

  bool insertValue(Value *D, vector<ValueList> &D);
  bool postResolveValues(ValueTable &ValTab);

  bool getTypeSlot(const Type *Ty, unsigned &Slot);
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
  virtual string getOpcode() const { return "placeholder"; }

  virtual Instruction *clone() const { abort(); return 0; }
};

struct BBPlaceHolderHelper : public BasicBlock {
  BBPlaceHolderHelper(const Type *Ty) : BasicBlock() {
    assert(Ty->isLabelType());
  }
};

struct MethPlaceHolderHelper : public Method {
  MethPlaceHolderHelper(const Type *Ty) 
    : Method((const MethodType*)Ty) {
    assert(Ty->isMethodType() && "Method placeholders must be method types!");
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

#endif
