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
#include "llvm/iOther.h"
#include "llvm/Function.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Assembly/Parser.h"
#include "Support/StringExtras.h"

// Global variables exported from the lexer...
extern std::FILE *llvmAsmin;
extern int llvmAsmlineno;

// Globals exported by the parser...
extern char* llvmAsmtext;
extern int   llvmAsmleng;

namespace llvm {

// Globals exported by the parser...
extern std::string CurFilename;

class Module;
Module *RunVMAsmParser(const std::string &Filename, FILE *F);


// UnEscapeLexed - Run through the specified buffer and change \xx codes to the
// appropriate character.  If AllowNull is set to false, a \00 value will cause
// an exception to be thrown.
//
// If AllowNull is set to true, the return value of the function points to the
// last character of the string in memory.
//
char *UnEscapeLexed(char *Buffer, bool AllowNull = false);


// ThrowException - Wrapper around the ParseException class that automatically
// fills in file line number and column number and options info.
//
// This also helps me because I keep typing 'throw new ParseException' instead 
// of just 'throw ParseException'... sigh...
//
static inline void ThrowException(const std::string &message,
				  int LineNo = -1) {
  if (LineNo == -1) LineNo = llvmAsmlineno;
  // TODO: column number in exception
  throw ParseException(CurFilename, message, LineNo);
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
  enum {
    NumberVal, NameVal, ConstSIntVal, ConstUIntVal, ConstFPVal, ConstNullVal,
    ConstantVal,
  } Type;

  union {
    int      Num;         // If it's a numeric reference
    char    *Name;        // If it's a named reference.  Memory must be free'd.
    int64_t  ConstPool64; // Constant pool reference.  This is the value
    uint64_t UConstPool64;// Unsigned constant pool reference.
    double   ConstPoolFP; // Floating point constant pool reference
    Constant *ConstantValue; // Fully resolved constant for ConstantVal case.
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

  static ValID create(Constant *Val) {
    ValID D; D.Type = ConstantVal; D.ConstantValue = Val; return D;
  }

  inline void destroy() const {
    if (Type == NameVal)
      free(Name);    // Free this strdup'd memory...
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
    case ConstUIntVal  :
    case ConstSIntVal  : return std::string("%") + itostr(ConstPool64);
    case ConstantVal:
      if (ConstantValue == ConstantBool::True) return "true";
      if (ConstantValue == ConstantBool::False) return "false";
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
    case ConstantVal:   return ConstantValue < V.ConstantValue;
    default:  assert(0 && "Unknown value type!"); return false;
    }
  }
};



template<class SuperType>
class PlaceholderValue : public SuperType {
  ValID D;
  int LineNum;
public:
  PlaceholderValue(const Type *Ty, const ValID &d) : SuperType(Ty), D(d) {
    LineNum = llvmAsmlineno;
  }
  ValID &getDef() { return D; }
  int getLineNum() const { return LineNum; }
};

struct InstPlaceHolderHelper : public Instruction {
  InstPlaceHolderHelper(const Type *Ty) : Instruction(Ty, UserOp1, "") {}

  virtual Instruction *clone() const { abort(); return 0; }
  virtual const char *getOpcodeName() const { return "placeholder"; }
};

struct BBPlaceHolderHelper : public BasicBlock {
  BBPlaceHolderHelper(const Type *Ty) : BasicBlock() {
    assert(Ty == Type::LabelTy);
  }
};

typedef PlaceholderValue<InstPlaceHolderHelper>  ValuePlaceHolder;
typedef PlaceholderValue<BBPlaceHolderHelper>    BBPlaceHolder;

static inline ValID &getValIDFromPlaceHolder(const Value *Val) {
  const Type *Ty = Val->getType();
  if (isa<PointerType>(Ty) &&
      isa<FunctionType>(cast<PointerType>(Ty)->getElementType()))
    Ty = cast<PointerType>(Ty)->getElementType();

  switch (Ty->getPrimitiveID()) {
  case Type::LabelTyID:  return ((BBPlaceHolder*)Val)->getDef();
  default:               return ((ValuePlaceHolder*)Val)->getDef();
  }
}

static inline int getLineNumFromPlaceHolder(const Value *Val) {
  const Type *Ty = Val->getType();
  if (isa<PointerType>(Ty) &&
      isa<FunctionType>(cast<PointerType>(Ty)->getElementType()))
    Ty = cast<PointerType>(Ty)->getElementType();

  switch (Ty->getPrimitiveID()) {
  case Type::LabelTyID:  return ((BBPlaceHolder*)Val)->getLineNum();
  default:               return ((ValuePlaceHolder*)Val)->getLineNum();
  }
}

} // End llvm namespace

#endif
