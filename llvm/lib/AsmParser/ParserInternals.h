//===-- ParserInternals.h - Definitions internal to the parser ---*- C++ -*--=//
//
//  This header file defines the various variables that are shared among the 
//  different components of the parser...
//
//===----------------------------------------------------------------------===//

#ifndef PARSER_INTERNALS_H
#define PARSER_INTERNALS_H

#include <stdio.h>
#define __STDC_LIMIT_MACROS

#include "llvm/InstrTypes.h"
#include "llvm/BasicBlock.h"
#include "llvm/ConstPoolVals.h"
#include "llvm/iOther.h"
#include "llvm/Method.h"
#include "llvm/Type.h"
#include "llvm/Assembly/Parser.h"
#include "llvm/Support/StringExtras.h"

class Module;

// Global variables exported from the lexer...
extern FILE *llvmAsmin;
extern int llvmAsmlineno;

// Globals exported by the parser...
extern string CurFilename;
Module *RunVMAsmParser(const string &Filename, FILE *F);


// ThrowException - Wrapper around the ParseException class that automatically
// fills in file line number and column number and options info.
//
// This also helps me because I keep typing 'throw new ParseException' instead 
// of just 'throw ParseException'... sigh...
//
static inline void ThrowException(const string &message) {
  // TODO: column number in exception
  throw ParseException(CurFilename, message, llvmAsmlineno);
}

// ValID - Represents a reference of a definition of some sort.  This may either
// be a numeric reference or a symbolic (%var) reference.  This is just a 
// discriminated union.
//
// Note that I can't implement this class in a straight forward manner with 
// constructors and stuff because it goes in a union, and GCC doesn't like 
// putting classes with ctor's in unions.  :(
//
struct ValID {
  int Type;               // 0 = number, 1 = name, 2 = const pool, 
                          // 3 = unsigned const pool, 4 = const string, 
                          // 5 = const fp
  union {
    int      Num;         // If it's a numeric reference
    char    *Name;        // If it's a named reference.  Memory must be free'd.
    int64_t  ConstPool64; // Constant pool reference.  This is the value
    uint64_t UConstPool64;// Unsigned constant pool reference.
    double   ConstPoolFP; // Floating point constant pool reference
  };

  static ValID create(int Num) {
    ValID D; D.Type = 0; D.Num = Num; return D;
  }

  static ValID create(char *Name) {
    ValID D; D.Type = 1; D.Name = Name; return D;
  }

  static ValID create(int64_t Val) {
    ValID D; D.Type = 2; D.ConstPool64 = Val; return D;
  }

  static ValID create(uint64_t Val) {
    ValID D; D.Type = 3; D.UConstPool64 = Val; return D;
  }

  static ValID create_conststr(char *Name) {
    ValID D; D.Type = 4; D.Name = Name; return D;
  }

  static ValID create(double Val) {
    ValID D; D.Type = 5; D.ConstPoolFP = Val; return D;
  }

  inline void destroy() const {
    if (Type == 1 || Type == 4) free(Name);  // Free this strdup'd memory...
  }

  inline ValID copy() const {
    if (Type != 1 && Type != 4) return *this;
    ValID Result = *this;
    Result.Name = strdup(Name);
    return Result;
  }

  inline string getName() const {
    switch (Type) {
    case 0:  return string("#") + itostr(Num);
    case 1:  return Name;
    case 4:  return string("\"") + Name + string("\"");
    case 5:  return ftostr(ConstPoolFP);
    default: return string("%") + itostr(ConstPool64);
    }
  }
};



template<class SuperType>
class PlaceholderDef : public SuperType {
  ValID D;
  // TODO: Placeholder def should hold Line #/Column # of definition in case
  // there is an error resolving the defintition!
public:
  PlaceholderDef(const Type *Ty, const ValID &d) : SuperType(Ty), D(d) {}
  ValID &getDef() { return D; }
};

struct InstPlaceHolderHelper : public Instruction {
  InstPlaceHolderHelper(const Type *Ty) : Instruction(Ty, UserOp1, "") {}

  virtual Instruction *clone() const { abort(); }
  virtual const char *getOpcodeName() const { return "placeholder"; }
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

typedef PlaceholderDef<InstPlaceHolderHelper>  DefPlaceHolder;
typedef PlaceholderDef<BBPlaceHolderHelper>    BBPlaceHolder;
typedef PlaceholderDef<MethPlaceHolderHelper>  MethPlaceHolder;
//typedef PlaceholderDef<ModulePlaceHolderHelper> ModulePlaceHolder;

static inline ValID &getValIDFromPlaceHolder(Value *Def) {
  switch (Def->getType()->getPrimitiveID()) {
  case Type::LabelTyID:  return ((BBPlaceHolder*)Def)->getDef();
  case Type::MethodTyID: return ((MethPlaceHolder*)Def)->getDef();
//case Type::ModuleTyID:  return ((ModulePlaceHolder*)Def)->getDef();
  default:               return ((DefPlaceHolder*)Def)->getDef();
  }
}

#endif
