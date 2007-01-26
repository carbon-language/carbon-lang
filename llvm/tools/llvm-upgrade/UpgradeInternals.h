//===-- ParserInternals.h - Definitions internal to the parser --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This header file defines the various variables that are shared among the
//  different components of the parser...
//
//===----------------------------------------------------------------------===//

#ifndef PARSER_INTERNALS_H
#define PARSER_INTERNALS_H

#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/Instructions.h"
#include "llvm/ADT/StringExtras.h"
#include <list>


// Global variables exported from the lexer.
extern int yydebug;
extern void error(const std::string& msg, int line = -1);
extern char* Upgradetext;
extern int   Upgradeleng;
extern int Upgradelineno;

namespace llvm {


class Module;
Module* UpgradeAssembly(const std::string &infile, std::istream& in, 
                        bool debug, bool addAttrs);


class SignedType : public IntegerType {
  const IntegerType *base_type;
  static SignedType *SByteTy;
  static SignedType *SShortTy;
  static SignedType *SIntTy;
  static SignedType *SLongTy;
  SignedType(const IntegerType* ITy);
public:
  static const SignedType *get(const IntegerType* ITy);

  bool isSigned() const { return true; }
  const IntegerType* getBaseType() const {
    return base_type;
  }
  const IntegerType*  resolve() const {
    ForwardType = base_type;
    return base_type;
  }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const SignedType *T) { return true; }
  static inline bool classof(const Type *T); 
};

extern std::istream* LexInput;

// UnEscapeLexed - Run through the specified buffer and change \xx codes to the
// appropriate character.  If AllowNull is set to false, a \00 value will cause
// an error.
//
// If AllowNull is set to true, the return value of the function points to the
// last character of the string in memory.
//
char *UnEscapeLexed(char *Buffer, bool AllowNull = false);

/// InlineAsmDescriptor - This is a simple class that holds info about inline
/// asm blocks, for use by ValID.
struct InlineAsmDescriptor {
  std::string AsmString, Constraints;
  bool HasSideEffects;
  
  InlineAsmDescriptor(const std::string &as, const std::string &c, bool HSE)
    : AsmString(as), Constraints(c), HasSideEffects(HSE) {}
};


// ValID - Represents a reference of a definition of some sort.  This may either
// be a numeric reference or a symbolic (%var) reference.  This is just a
// discriminated union.
//
// Note that I can't implement this class in a straight forward manner with
// constructors and stuff because it goes in a union.
//
struct ValID {
  enum {
    NumberVal, NameVal, ConstSIntVal, ConstUIntVal, ConstFPVal, ConstNullVal,
    ConstUndefVal, ConstZeroVal, ConstantVal, InlineAsmVal
  } Type;

  union {
    int      Num;         // If it's a numeric reference
    char    *Name;        // If it's a named reference.  Memory must be free'd.
    int64_t  ConstPool64; // Constant pool reference.  This is the value
    uint64_t UConstPool64;// Unsigned constant pool reference.
    double   ConstPoolFP; // Floating point constant pool reference
    Constant *ConstantValue; // Fully resolved constant for ConstantVal case.
    InlineAsmDescriptor *IAD;
  };

  static ValID create(int Num) {
    ValID D; D.Type = NumberVal; D.Num = Num; return D;
  }

  static ValID create(char *Name) {
    ValID D; D.Type = NameVal; D.Name = Name; return D;
  }

  static ValID create(int64_t Val) {
    ValID D; D.Type = ConstSIntVal; D.ConstPool64 = Val; return D;
  }

  static ValID create(uint64_t Val) {
    ValID D; D.Type = ConstUIntVal; D.UConstPool64 = Val; return D;
  }

  static ValID create(double Val) {
    ValID D; D.Type = ConstFPVal; D.ConstPoolFP = Val; return D;
  }

  static ValID createNull() {
    ValID D; D.Type = ConstNullVal; return D;
  }

  static ValID createUndef() {
    ValID D; D.Type = ConstUndefVal; return D;
  }

  static ValID createZeroInit() {
    ValID D; D.Type = ConstZeroVal; return D;
  }
  
  static ValID create(Constant *Val) {
    ValID D; D.Type = ConstantVal; D.ConstantValue = Val; return D;
  }
  
  static ValID createInlineAsm(const std::string &AsmString,
                               const std::string &Constraints,
                               bool HasSideEffects) {
    ValID D;
    D.Type = InlineAsmVal;
    D.IAD = new InlineAsmDescriptor(AsmString, Constraints, HasSideEffects);
    return D;
  }

  inline void destroy() const {
    if (Type == NameVal)
      free(Name);    // Free this strdup'd memory.
    else if (Type == InlineAsmVal)
      delete IAD;
  }

  inline ValID copy() const {
    if (Type != NameVal) return *this;
    ValID Result = *this;
    Result.Name = strdup(Name);
    return Result;
  }

  inline std::string getName() const {
    switch (Type) {
    case NumberVal     : return std::string("#") + itostr(Num);
    case NameVal       : return Name;
    case ConstFPVal    : return ftostr(ConstPoolFP);
    case ConstNullVal  : return "null";
    case ConstUndefVal : return "undef";
    case ConstZeroVal  : return "zeroinitializer";
    case ConstUIntVal  :
    case ConstSIntVal  : return std::string("%") + itostr(ConstPool64);
    case ConstantVal:
      if (ConstantValue == ConstantInt::get(Type::Int1Ty, true)) 
        return "true";
      if (ConstantValue == ConstantInt::get(Type::Int1Ty, false))
        return "false";
      return "<constant expression>";
    default:
      assert(0 && "Unknown value!");
      abort();
      return "";
    }
  }

  bool operator<(const ValID &V) const {
    if (Type != V.Type) return Type < V.Type;
    switch (Type) {
    case NumberVal:     return Num < V.Num;
    case NameVal:       return strcmp(Name, V.Name) < 0;
    case ConstSIntVal:  return ConstPool64  < V.ConstPool64;
    case ConstUIntVal:  return UConstPool64 < V.UConstPool64;
    case ConstFPVal:    return ConstPoolFP  < V.ConstPoolFP;
    case ConstNullVal:  return false;
    case ConstUndefVal: return false;
    case ConstZeroVal: return false;
    case ConstantVal:   return ConstantValue < V.ConstantValue;
    default:  assert(0 && "Unknown value type!"); return false;
    }
  }
};


// This structure is used to keep track of obsolete opcodes. The lexer will
// retain the ability to parse obsolete opcode mnemonics. In this case it will
// set "obsolete" to true and the opcode will be the replacement opcode. For
// example if "rem" is encountered then opcode will be set to "urem" and the
// "obsolete" flag will be true. If the opcode is not obsolete then "obsolete"
// will be false. 

enum TermOps {
  RetOp, BrOp, SwitchOp, InvokeOp, UnwindOp, UnreachableOp
};

enum BinaryOps {
  AddOp, SubOp, MulOp,
  DivOp, UDivOp, SDivOp, FDivOp, 
  RemOp, URemOp, SRemOp, FRemOp, 
  AndOp, OrOp, XorOp,
  SetEQ, SetNE, SetLE, SetGE, SetLT, SetGT
};

enum MemoryOps {
  MallocOp, FreeOp, AllocaOp, LoadOp, StoreOp, GetElementPtrOp
};

enum OtherOps {
  PHIOp, CallOp, ShlOp, ShrOp, SelectOp, UserOp1, UserOp2, VAArg,
  ExtractElementOp, InsertElementOp, ShuffleVectorOp,
  ICmpOp, FCmpOp,
  LShrOp, AShrOp
};

enum CastOps {
  CastOp, TruncOp, ZExtOp, SExtOp, FPTruncOp, FPExtOp, FPToUIOp, FPToSIOp,
  UIToFPOp, SIToFPOp, PtrToIntOp, IntToPtrOp, BitCastOp
};

enum Signedness { Signless, Unsigned, Signed };

struct TypeInfo {
  const llvm::Type *T;
  Signedness S;
};

struct PATypeInfo {
  llvm::PATypeHolder* T;
  Signedness S;
};

struct ConstInfo {
  llvm::Constant* C;
  Signedness S;
};

struct ValueInfo {
  llvm::Value* V;
  Signedness S;
};

struct InstrInfo {
  llvm::Instruction *I;
  Signedness S;
};

struct PHIListInfo {
  std::list<std::pair<llvm::Value*, llvm::BasicBlock*> > *P;
  Signedness S;
};

} // End llvm namespace

#endif
