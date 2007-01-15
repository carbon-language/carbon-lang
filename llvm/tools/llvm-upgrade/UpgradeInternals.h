//===-- UpgradeInternals.h - Internal parser definitionsr -------*- C++ -*-===//
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

#ifndef UPGRADE_INTERNALS_H
#define UPGRADE_INTERNALS_H

#include <llvm/ADT/StringExtras.h>
#include <string>
#include <istream>
#include <vector>
#include <set>
#include <cassert>

// Global variables exported from the lexer.
extern std::string CurFileName;
extern std::string Textin;
extern int Upgradelineno;
extern std::istream* LexInput;

// Global variables exported from the parser.
extern char* Upgradetext;
extern int   Upgradeleng;
extern unsigned SizeOfPointer;

// Functions exported by the parser
void UpgradeAssembly(
  const std::string & infile, std::istream& in, std::ostream &out, bool debug,
  bool addAttrs);
int yyerror(const char *ErrorMsg) ;

/// This enum is used to keep track of the original (1.9) type used to form
/// a type. These are needed for type upgrades and to determine how to upgrade
/// signed instructions with signless operands. The Lexer uses thse in its
/// calls to getTypeInfo
enum Types {
  BoolTy, SByteTy, UByteTy, ShortTy, UShortTy, IntTy, UIntTy, LongTy, ULongTy,
  FloatTy, DoubleTy, PointerTy, PackedTy, ArrayTy, StructTy, PackedStructTy, 
  OpaqueTy, VoidTy, LabelTy, FunctionTy, UnresolvedTy, UpRefTy
};

namespace {
class TypeInfo;
class ValueInfo;
class ConstInfo;
}

typedef std::vector<const TypeInfo*> TypeList;
typedef std::vector<ValueInfo*> ValueList;

/// A function to create a TypeInfo* used in the Lexer.
extern const TypeInfo* getTypeInfo(const std::string& newTy, Types oldTy);

#endif
