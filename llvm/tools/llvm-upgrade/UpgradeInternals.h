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
#include <iostream>


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

/// This class keeps track of the signedness of a type or value. It allows the
/// signedness of a composite type to be captured in a relatively simple form.
/// This is needed in order to retain the signedness of pre LLVM 2.0 types so
/// they can be upgraded properly. Signedness of composite types must be
/// captured in order to accurately get the signedness of a value through a
/// GEP instruction. 
/// @brief Class to track signedness of types and values.
struct Signedness {
  /// The basic kinds of signedness values.
  enum Kind { 
    Signless, ///< The type doesn't have any sign.
    Unsigned, ///< The type is an unsigned integer.
    Signed,   ///< The type is a signed integer.
    Named,    ///< The type is a named type (probably forward ref or up ref).
    Composite ///< The type is composite (struct, array, pointer). 
  };

private:
  /// @brief Keeps track of Signedness for composite types
  typedef std::vector<Signedness> SignVector;
  Kind kind; ///< The kind of signedness node
  union {
    SignVector *sv;    ///< The vector of Signedness for composite types
    std::string *name; ///< The name of the type for named types.
  };
public:
  /// The Signedness class is used as a member of a union so it cannot have
  /// a constructor or assignment operator. This function suffices.
  /// @brief Copy one signedness value to another
  void copy(const Signedness &that);
  /// The Signedness class is used as a member of a union so it cannot have
  /// a destructor.
  /// @brief Release memory, if any allocated.
  void destroy();

  /// @brief Make a Signless node.
  void makeSignless() { kind = Signless; sv = 0; }
  /// @brief Make a Signed node.
  void makeSigned()   { kind = Signed; sv = 0; }
  /// @brief Make an Unsigned node.
  void makeUnsigned() { kind = Unsigned; sv = 0; }
  /// @brief Make a Named node.
  void makeNamed(const std::string& nm){ 
    kind = Named; name = new std::string(nm); 
  }
  /// @brief Make an empty Composite node.
  void makeComposite() { kind = Composite; sv = new SignVector(); }
  /// @brief Make an Composite node, with the first element given.
  void makeComposite(const Signedness &S) { 
    kind = Composite; 
    sv = new SignVector(); 
    sv->push_back(S);
  }
  /// @brief Add an element to a Composite node.
  void add(const Signedness &S) {
    assert(isComposite() && "Must be composite to use add");
    sv->push_back(S);
  }
  bool operator<(const Signedness &that) const;
  bool operator==(const Signedness &that) const;
  bool isSigned() const { return kind == Signed; }
  bool isUnsigned() const { return kind == Unsigned; }
  bool isSignless() const { return kind == Signless; }
  bool isNamed() const { return kind == Named; }
  bool isComposite() const { return kind == Composite; }
  /// This is used by GetElementPtr to extract the sign of an element.
  /// @brief Get a specific element from a Composite node.
  Signedness get(uint64_t idx) const {
    assert(isComposite() && "Invalid Signedness type for get()");
    assert(sv && idx < sv->size() && "Invalid index");
    return (*sv)[idx];
  }
  /// @brief Get the name from a Named node.
  const std::string& getName() const {
    assert(isNamed() && "Can't get name from non-name Sign");
    return *name;
  }
#ifndef NDEBUG
  void dump() const;
#endif
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
  Signedness S;

  static ValID create(int Num) {
    ValID D; D.Type = NumberVal; D.Num = Num; D.S.makeSignless();
    return D;
  }

  static ValID create(char *Name) {
    ValID D; D.Type = NameVal; D.Name = Name; D.S.makeSignless();
    return D;
  }

  static ValID create(int64_t Val) {
    ValID D; D.Type = ConstSIntVal; D.ConstPool64 = Val; 
    D.S.makeSigned();
    return D;
  }

  static ValID create(uint64_t Val) {
    ValID D; D.Type = ConstUIntVal; D.UConstPool64 = Val; 
    D.S.makeUnsigned();
    return D;
  }

  static ValID create(double Val) {
    ValID D; D.Type = ConstFPVal; D.ConstPoolFP = Val;
    D.S.makeSignless();
    return D;
  }

  static ValID createNull() {
    ValID D; D.Type = ConstNullVal;
    D.S.makeSignless();
    return D;
  }

  static ValID createUndef() {
    ValID D; D.Type = ConstUndefVal;
    D.S.makeSignless();
    return D;
  }

  static ValID createZeroInit() {
    ValID D; D.Type = ConstZeroVal;
    D.S.makeSignless();
    return D;
  }
  
  static ValID create(Constant *Val) {
    ValID D; D.Type = ConstantVal; D.ConstantValue = Val;
    D.S.makeSignless();
    return D;
  }
  
  static ValID createInlineAsm(const std::string &AsmString,
                               const std::string &Constraints,
                               bool HasSideEffects) {
    ValID D;
    D.Type = InlineAsmVal;
    D.IAD = new InlineAsmDescriptor(AsmString, Constraints, HasSideEffects);
    D.S.makeSignless();
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

/// The following enums are used to keep track of prior opcodes. The lexer will
/// retain the ability to parse obsolete opcode mnemonics and generates semantic
/// values containing one of these enumerators.
enum TermOps {
  RetOp, BrOp, SwitchOp, InvokeOp, UnwindOp, UnreachableOp
};

enum BinaryOps {
  AddOp, SubOp, MulOp,
  DivOp, UDivOp, SDivOp, FDivOp, 
  RemOp, URemOp, SRemOp, FRemOp, 
  AndOp, OrOp, XorOp,
  ShlOp, ShrOp, LShrOp, AShrOp,
  SetEQ, SetNE, SetLE, SetGE, SetLT, SetGT
};

enum MemoryOps {
  MallocOp, FreeOp, AllocaOp, LoadOp, StoreOp, GetElementPtrOp
};

enum OtherOps {
  PHIOp, CallOp, SelectOp, UserOp1, UserOp2, VAArg,
  ExtractElementOp, InsertElementOp, ShuffleVectorOp,
  ICmpOp, FCmpOp
};

enum CastOps {
  CastOp, TruncOp, ZExtOp, SExtOp, FPTruncOp, FPExtOp, FPToUIOp, FPToSIOp,
  UIToFPOp, SIToFPOp, PtrToIntOp, IntToPtrOp, BitCastOp
};

// An enumeration for the old calling conventions, ala LLVM 1.9
namespace OldCallingConv {
  enum ID {
    C = 0, CSRet = 1, Fast = 8, Cold = 9, X86_StdCall = 64, X86_FastCall = 65,
    None = 99999
  };
}

/// These structures are used as the semantic values returned from various
/// productions in the grammar. They simply bundle an LLVM IR object with
/// its Signedness value. These help track signedness through the various
/// productions. 
struct TypeInfo {
  const llvm::Type *T;
  Signedness S;
  bool operator<(const TypeInfo& that) const {
    if (this == &that)
      return false;
    if (T < that.T)
      return true;
    if (T == that.T) {
      bool result = S < that.S;
//#define TYPEINFO_DEBUG
#ifdef TYPEINFO_DEBUG
      std::cerr << (result?"true  ":"false ") << T->getDescription() << " (";
      S.dump();
      std::cerr << ") < " << that.T->getDescription() << " (";
      that.S.dump();
      std::cerr << ")\n";
#endif
      return result;
    }
    return false;
  }
  bool operator==(const TypeInfo& that) const {
    if (this == &that)
      return true;
    return T == that.T && S == that.S;
  }
  void destroy() { S.destroy(); }
};

struct PATypeInfo {
  llvm::PATypeHolder* PAT;
  Signedness S;
  void destroy() { S.destroy(); delete PAT; }
};

struct ConstInfo {
  llvm::Constant* C;
  Signedness S;
  void destroy() { S.destroy(); }
};

struct ValueInfo {
  llvm::Value* V;
  Signedness S;
  void destroy() { S.destroy(); }
};

struct InstrInfo {
  llvm::Instruction *I;
  Signedness S;
  void destroy() { S.destroy(); }
};

struct TermInstInfo {
  llvm::TerminatorInst *TI;
  Signedness S;
  void destroy() { S.destroy(); }
};

struct PHIListInfo {
  std::list<std::pair<llvm::Value*, llvm::BasicBlock*> > *P;
  Signedness S;
  void destroy() { S.destroy(); delete P; }
};

} // End llvm namespace

#endif
