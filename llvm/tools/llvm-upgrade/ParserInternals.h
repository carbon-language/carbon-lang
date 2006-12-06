//===-- ParserInternals.h - Definitions internal to the parser --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencer and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This header file defines the variables that are shared between the lexer,
//  the parser, and the main program.
//
//===----------------------------------------------------------------------===//

#ifndef PARSER_INTERNALS_H
#define PARSER_INTERNALS_H

#include <string>
#include <istream>
#include <vector>

// Global variables exported from the lexer...

extern std::string CurFileName;
extern std::string Textin;
extern int Upgradelineno;
extern std::istream* LexInput;


void UpgradeAssembly(
  const std::string & infile, std::istream& in, std::ostream &out, bool debug);

// Globals exported by the parser...
extern char* Upgradetext;
extern int   Upgradeleng;
extern unsigned SizeOfPointer;

int yyerror(const char *ErrorMsg) ;

/// This enum is used to keep track of the original (1.9) type used to form
/// a type. These are needed for type upgrades and to determine how to upgrade
/// signed instructions with signless operands.
enum Types {
  BoolTy, SByteTy, UByteTy, ShortTy, UShortTy, IntTy, UIntTy, LongTy, ULongTy,
  FloatTy, DoubleTy, PointerTy, PackedTy, ArrayTy, StructTy, OpaqueTy, VoidTy,
  LabelTy, FunctionTy, UnresolvedTy, NumericTy
};

/// This type is used to keep track of the signedness of the obsolete
/// integer types. Instead of creating an llvm::Type directly, the Lexer will
/// create instances of TypeInfo which retains the signedness indication so
/// it can be used by the parser for upgrade decisions.
/// For example if "uint" is encountered then the "first" field will be set 
/// to "int32" and the "second" field will be set to "isUnsigned".  If the 
/// type is not obsolete then "second" will be set to "isSignless".
struct TypeInfo {
  std::string* newTy;
  Types oldTy;
  Types elemTy;

  void destroy() const { delete newTy; }

  TypeInfo clone() const { 
    TypeInfo result = *this; 
    result.newTy = new std::string(*newTy);
    return result;
  }

  Types getElementType() const { return elemTy; }

  bool isSigned() const {
    return oldTy == SByteTy || oldTy == ShortTy || 
           oldTy == IntTy || oldTy == LongTy;
  }

  bool isUnsigned() const {
    return oldTy == UByteTy || oldTy == UShortTy || 
           oldTy == UIntTy || oldTy == ULongTy;
  }

  bool isBool() const {
    return oldTy == BoolTy;
  }

  bool isSignless() const { return !isSigned() && !isUnsigned(); }
  bool isInteger() const { return isSigned() || isUnsigned(); }
  bool isIntegral() const { return oldTy == BoolTy || isInteger(); }
  bool isFloatingPoint() const { return oldTy == DoubleTy || oldTy == FloatTy; }
  bool isPacked() const { return oldTy == PackedTy; }
  bool isPointer() const { return oldTy == PointerTy; }
  bool isOther() const { 
    return !isPacked() && !isPointer() && !isFloatingPoint() && !isIntegral(); }

  unsigned getBitWidth() const {
    switch (oldTy) {
      case LabelTy:
      case VoidTy : return 0;
      case BoolTy : return 1;
      case SByteTy: case UByteTy : return 8;
      case ShortTy: case UShortTy : return 16;
      case IntTy: case UIntTy: case FloatTy: return 32;
      case LongTy: case ULongTy: case DoubleTy : return 64;
      case PointerTy: return SizeOfPointer; // global var
      default:
        return 128; /// Struct/Packed/Array --> doesn't matter
      
    }
  }
};

/// This type is used to keep track of the signedness of values. Instead
/// of creating llvm::Value directly, the parser will create ValueInfo which
/// associates a Value* with a Signedness indication.
struct ValueInfo {
  std::string* val;
  TypeInfo type;
  bool constant;
  bool isConstant() const { return constant; }
  void destroy() { delete val; type.destroy(); }
};

/// This type is used to keep track of the signedness of constants.
struct ConstInfo {
  std::string *cnst;
  TypeInfo type;
  void destroy() { delete cnst; type.destroy(); }
};

typedef std::vector<ValueInfo> ValueList;


#endif
