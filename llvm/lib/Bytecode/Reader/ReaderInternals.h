//===-- ReaderInternals.h - Definitions internal to the reader ---*- C++ -*--=//
//
//  This header file defines various stuff that is used by the bytecode reader.
//
//===----------------------------------------------------------------------===//

#ifndef READER_INTERNALS_H
#define READER_INTERNALS_H

#include "llvm/Bytecode/Primitives.h"
#include "llvm/Function.h"
#include "llvm/BasicBlock.h"
#include "llvm/Instruction.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Constant.h"
#include "Support/NonCopyable.h"
#include <map>
#include <utility>
#include <list>
#include <iostream>

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


class ConstantFwdRefs: public NonCopyable {
  Module* TheModule;
  
  // GlobalRefs - This maintains a mapping between <Type, Slot #>'s and forward
  // references to global values or constants.  Such values may be referenced
  // before they are defined, and if so, the temporary object that they
  // represent is held here.
  //
  typedef std::map<std::pair<const Type *, unsigned>,
                   Value*>  GlobalRefsType;
  GlobalRefsType GlobalRefs;

  Value*       find                   (const Type* Ty, unsigned Slot);
  void         insert                 (const Type* Ty, unsigned Slot, Value* V);
  void         erase                  (const Type* Ty, unsigned Slot);

public:
  // sets the current module pointer: needed to insert placeholder globals
  void         VisitingModule         (Module* M) { TheModule = M; }
  
  // get a forward reference to a global or a constant
  GlobalValue* GetFwdRefToGlobal      (const PointerType* PT, unsigned Slot);
  Constant*    GetFwdRefToConstant    (const Type* Ty,        unsigned Slot);

  // resolve all references to the placeholder (if any) for the given value
  void         ResolveRefsToValue     (Value* val, unsigned Slot);
};


class BytecodeParser : public AbstractTypeUser {
  std::string Error;     // Error message string goes here...
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

  // fwdRefs - This manages forward references to global values.
  ConstantFwdRefs fwdRefs;

  // TypesLoaded - This vector mirrors the Values[TypeTyID] plane.  It is used
  // to deal with forward references to types.
  //
  typedef std::vector<PATypeHandle<Type> > TypeValuesListTy;
  TypeValuesListTy ModuleTypeValues;
  TypeValuesListTy MethodTypeValues;

  // Information read from the ModuleGlobalInfo section of the file...
  unsigned FirstDerivedTyID;

  // When the ModuleGlobalInfo section is read, we load the type of each method
  // and the 'ModuleValues' slot that it lands in.  We then load a placeholder
  // into its slot to reserve it.  When the method is loaded, this placeholder
  // is replaced.
  //
  std::list<std::pair<const PointerType *, unsigned> > MethodSignatureList;

private:
  bool ParseModule          (const uchar * Buf, const uchar *End, Module *&);
  bool ParseModuleGlobalInfo(const uchar *&Buf, const uchar *End, Module *);
  bool ParseSymbolTable   (const uchar *&Buf, const uchar *End, SymbolTable *);
  bool ParseMethod        (const uchar *&Buf, const uchar *End, Module *);
  bool ParseBasicBlock    (const uchar *&Buf, const uchar *End, BasicBlock *&);
  bool ParseInstruction   (const uchar *&Buf, const uchar *End, Instruction *&);
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

  int insertValue(Value *D, std::vector<ValueList> &D);  // -1 = Failure
  bool postResolveValues(ValueTable &ValTab);

  bool getTypeSlot(const Type *Ty, unsigned &Slot);

  // resolveRefsToGlobal   -- resolve forward references to a global
  // resolveRefsToConstant -- resolve forward references to a constant
  // 
  void resolveRefsToGlobal(GlobalValue* GV, unsigned Slot);
  void resolveRefsToConstant(Constant* C, unsigned Slot);
  
  // refineAbstractType - The callback method is invoked when one of the
  // elements of TypeValues becomes more concrete...
  //
  virtual void refineAbstractType(const DerivedType *OldTy, const Type *NewTy);
};

template<class SuperType>
class PlaceholderDef : public SuperType {
  unsigned ID;
  PlaceholderDef();                     // do not implement
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

struct MethPlaceHolderHelper : public Function {
  MethPlaceHolderHelper(const Type *Ty) 
    : Function(cast<const FunctionType>(Ty), true) {
  }
};

struct ConstantPlaceHolderHelper : public Constant {
  ConstantPlaceHolderHelper(const Type *Ty)
    : Constant(Ty) {}
  virtual bool isNullValue() const { return false; }
};

typedef PlaceholderDef<InstPlaceHolderHelper>  DefPHolder;
typedef PlaceholderDef<BBPlaceHolderHelper>    BBPHolder;
typedef PlaceholderDef<MethPlaceHolderHelper>  MethPHolder;
typedef PlaceholderDef<ConstantPlaceHolderHelper>  ConstPHolder;


static inline unsigned getValueIDNumberFromPlaceHolder(Value *Def) {
  if (isa<Constant>(Def))
    return ((ConstPHolder*)Def)->getID();
  
  // else discriminate by type
  switch (Def->getType()->getPrimitiveID()) {
  case Type::LabelTyID:    return ((BBPHolder*)Def)->getID();
  case Type::FunctionTyID: return ((MethPHolder*)Def)->getID();
  default:                 return ((DefPHolder*)Def)->getID();
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


// failure Template - This template function is used as a place to put
// breakpoints in to debug failures of the bytecode parser.
//
template <typename X>
static X failure(X Value) {
  return Value;
}

#endif
