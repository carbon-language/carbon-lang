//===-- ParserInternals.h - Definitions internal to the parser --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
#include "llvm/ParameterAttributes.h"
#include "llvm/Function.h"
#include "llvm/Instructions.h"
#include "llvm/Assembly/Parser.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APSInt.h"
namespace llvm { class MemoryBuffer; }

// Global variables exported from the lexer...

extern llvm::ParseError* TheParseError; /// FIXME: Not threading friendly

// functions exported from the lexer
void InitLLLexer(llvm::MemoryBuffer *MB);
const char *LLLgetTokenStart();
unsigned LLLgetTokenLength();
std::string LLLgetFilename();
unsigned LLLgetLineNo();
void FreeLexer();

namespace llvm {
class Module;

// RunVMAsmParser - Parse a buffer and return Module
Module *RunVMAsmParser(llvm::MemoryBuffer *MB);

// GenerateError - Wrapper around the ParseException class that automatically
// fills in file line number and column number and options info.
//
// This also helps me because I keep typing 'throw new ParseException' instead
// of just 'throw ParseException'... sigh...
//
extern void GenerateError(const std::string &message, int LineNo = -1);

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
    LocalID, GlobalID, LocalName, GlobalName,
    ConstSIntVal, ConstUIntVal, ConstAPInt, ConstFPVal, ConstNullVal,
    ConstUndefVal, ConstZeroVal, ConstantVal, InlineAsmVal
  } Type;

  union {
    unsigned Num;            // If it's a numeric reference like %1234
    std::string *Name;    // If it's a named reference.  Memory must be deleted.
    int64_t  ConstPool64;    // Constant pool reference.  This is the value
    uint64_t UConstPool64;   // Unsigned constant pool reference.
    APSInt *ConstPoolInt;     // Large Integer constant pool reference
    APFloat *ConstPoolFP;    // Floating point constant pool reference
    Constant *ConstantValue; // Fully resolved constant for ConstantVal case.
    InlineAsmDescriptor *IAD;
 };

  static ValID createLocalID(unsigned Num) {
    ValID D; D.Type = LocalID; D.Num = Num; return D;
  }
  static ValID createGlobalID(unsigned Num) {
    ValID D; D.Type = GlobalID; D.Num = Num; return D;
  }
  static ValID createLocalName(const std::string &Name) {
    ValID D; D.Type = LocalName; D.Name = new std::string(Name); return D;
  }
  static ValID createGlobalName(const std::string &Name) {
    ValID D; D.Type = GlobalName; D.Name = new std::string(Name); return D;
  }
  
  static ValID create(int64_t Val) {
    ValID D; D.Type = ConstSIntVal; D.ConstPool64 = Val; return D;
  }

  static ValID create(uint64_t Val) {
    ValID D; D.Type = ConstUIntVal; D.UConstPool64 = Val; return D;
  }

  static ValID create(APFloat *Val) {
    ValID D; D.Type = ConstFPVal; D.ConstPoolFP = Val; return D;
  }
  
  static ValID create(const APInt &Val, bool isSigned) {
    ValID D; D.Type = ConstAPInt;
    D.ConstPoolInt = new APSInt(Val, !isSigned);
    return D;
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
    if (Type == LocalName || Type == GlobalName)
      delete Name;    // Free this strdup'd memory.
    else if (Type == InlineAsmVal)
      delete IAD;
    else if (Type == ConstAPInt)
      delete ConstPoolInt;
  }

  inline ValID copy() const {
    ValID Result = *this;
    if (Type == ConstAPInt)
      Result.ConstPoolInt = new APSInt(*ConstPoolInt);
    
    if (Type != LocalName && Type != GlobalName) return Result;
    Result.Name = new std::string(*Name);
    return Result;
  }

  inline std::string getName() const {
    switch (Type) {
    case LocalID       : return '%' + utostr(Num);
    case GlobalID      : return '@' + utostr(Num);
    case LocalName     : return *Name;
    case GlobalName    : return *Name;
    case ConstAPInt    : return ConstPoolInt->toString();
    case ConstFPVal    : return ftostr(*ConstPoolFP);
    case ConstNullVal  : return "null";
    case ConstUndefVal : return "undef";
    case ConstZeroVal  : return "zeroinitializer";
    case ConstUIntVal  :
    case ConstSIntVal  : return std::string("%") + itostr(ConstPool64);
    case ConstantVal:
      if (ConstantValue == ConstantInt::getTrue()) return "true";
      if (ConstantValue == ConstantInt::getFalse()) return "false";
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
    case LocalID:
    case GlobalID:      return Num < V.Num;
    case LocalName:
    case GlobalName:    return *Name < *V.Name;
    case ConstSIntVal:  return ConstPool64  < V.ConstPool64;
    case ConstUIntVal:  return UConstPool64 < V.UConstPool64;
    case ConstAPInt    : return ConstPoolInt->ult(*V.ConstPoolInt);
    case ConstFPVal:    return ConstPoolFP->compare(*V.ConstPoolFP) ==
                               APFloat::cmpLessThan;
    case ConstNullVal:  return false;
    case ConstUndefVal: return false;
    case ConstZeroVal: return false;
    case ConstantVal:   return ConstantValue < V.ConstantValue;
    default:  assert(0 && "Unknown value type!"); return false;
    }
  }

  bool operator==(const ValID &V) const {
    if (Type != V.Type) return false;
    
    switch (Type) {
    default:  assert(0 && "Unknown value type!");
    case LocalID:
    case GlobalID: return Num == V.Num;
    case LocalName:
    case GlobalName: return *Name == *(V.Name);
    case ConstSIntVal:  return ConstPool64  == V.ConstPool64;
    case ConstUIntVal:  return UConstPool64 == V.UConstPool64;
    case ConstAPInt:    return *ConstPoolInt == *V.ConstPoolInt;
    case ConstFPVal:    return ConstPoolFP->compare(*V.ConstPoolFP) == 
                               APFloat::cmpEqual;
    case ConstantVal:   return ConstantValue == V.ConstantValue;
    case ConstNullVal:  return true;
    case ConstUndefVal: return true;
    case ConstZeroVal:  return true;
    }
  }
};

struct TypeWithAttrs {
  llvm::PATypeHolder *Ty;
  ParameterAttributes Attrs;
};

typedef std::vector<TypeWithAttrs> TypeWithAttrsList; 

struct ArgListEntry {
  ParameterAttributes Attrs;
  llvm::PATypeHolder *Ty;
  std::string *Name;
};

typedef std::vector<struct ArgListEntry> ArgListType;

struct ParamListEntry {
  Value *Val;
  ParameterAttributes Attrs;
};

typedef std::vector<ParamListEntry> ParamList;


} // End llvm namespace

#endif
