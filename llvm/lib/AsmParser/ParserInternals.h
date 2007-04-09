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
#include "llvm/ParameterAttributes.h"
#include "llvm/Function.h"
#include "llvm/Instructions.h"
#include "llvm/Assembly/Parser.h"
#include "llvm/ADT/StringExtras.h"


// Global variables exported from the lexer...

extern int llvmAsmlineno;         /// FIXME: Not threading friendly
extern llvm::ParseError* TheParseError; /// FIXME: Not threading friendly

extern std::string &llvmAsmTextin;

// functions exported from the lexer
void set_scan_file(FILE * F);
void set_scan_string (const char * str);

// Globals exported by the parser...
extern char* llvmAsmtext;
extern int   llvmAsmleng;

namespace llvm {

// Globals exported by the parser...
extern std::string CurFilename;   /// FIXME: Not threading friendly

class Module;
Module *RunVMAsmParser(const std::string &Filename, FILE *F);

// Parse a string directly
Module *RunVMAsmParser(const char * AsmString, Module * M);


// UnEscapeLexed - Run through the specified buffer and change \xx codes to the
// appropriate character.  If AllowNull is set to false, a \00 value will cause
// an error.
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
    ConstSIntVal, ConstUIntVal, ConstFPVal, ConstNullVal,
    ConstUndefVal, ConstZeroVal, ConstantVal, InlineAsmVal
  } Type;

  union {
    unsigned Num;         // If it's a numeric reference like %1234
    char    *Name;        // If it's a named reference.  Memory must be free'd.
    int64_t  ConstPool64; // Constant pool reference.  This is the value
    uint64_t UConstPool64;// Unsigned constant pool reference.
    double   ConstPoolFP; // Floating point constant pool reference
    Constant *ConstantValue; // Fully resolved constant for ConstantVal case.
    InlineAsmDescriptor *IAD;
  };

  static ValID createLocalID(unsigned Num) {
    ValID D; D.Type = LocalID; D.Num = Num; return D;
  }
  static ValID createGlobalID(unsigned Num) {
    ValID D; D.Type = GlobalID; D.Num = Num; return D;
  }
  static ValID createLocalName(char *Name) {
    ValID D; D.Type = LocalName; D.Name = Name; return D;
  }
  static ValID createGlobalName(char *Name) {
    ValID D; D.Type = GlobalName; D.Name = Name; return D;
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
    if (Type == LocalName || Type == GlobalName)
      free(Name);    // Free this strdup'd memory.
    else if (Type == InlineAsmVal)
      delete IAD;
  }

  inline ValID copy() const {
    if (Type != LocalName && Type != GlobalName) return *this;
    ValID Result = *this;
    Result.Name = strdup(Name);
    return Result;
  }

  inline std::string getName() const {
    switch (Type) {
    case LocalID       : return '%' + utostr(Num);
    case GlobalID      : return '@' + utostr(Num);
    case LocalName     : return Name;
    case GlobalName    : return Name;
    case ConstFPVal    : return ftostr(ConstPoolFP);
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
    case GlobalName:    return strcmp(Name, V.Name) < 0;
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

  bool operator==(const ValID &V) const {
    if (Type == V.Type) {
      switch (Type) {
        case LocalID:
        case GlobalID: return Num == V.Num;
        case LocalName:
        case GlobalName: return strcmp(Name, V.Name) == 0;
        case ConstSIntVal:  return ConstPool64  == V.ConstPool64;
        case ConstUIntVal:  return UConstPool64 == V.UConstPool64;
        case ConstFPVal:    return ConstPoolFP  == V.ConstPoolFP;
        case ConstantVal:   return ConstantValue == V.ConstantValue;
        case ConstNullVal:  return true;
        case ConstUndefVal: return true;
        case ConstZeroVal:  return true;
        default:  assert(0 && "Unknown value type!"); return false;
      }
    }
    return false;
  }
};

struct TypeWithAttrs {
  llvm::PATypeHolder *Ty;
  uint16_t Attrs;
};

typedef std::vector<TypeWithAttrs> TypeWithAttrsList; 

struct ArgListEntry {
  uint16_t Attrs;
  llvm::PATypeHolder *Ty;
  char *Name;
};

typedef std::vector<struct ArgListEntry> ArgListType;

struct ValueRefListEntry {
  Value *Val;
  uint16_t Attrs;
};

typedef std::vector<ValueRefListEntry> ValueRefList;


} // End llvm namespace

#endif
