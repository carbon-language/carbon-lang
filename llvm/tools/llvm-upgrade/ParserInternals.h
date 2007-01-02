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

#include <llvm/ADT/StringExtras.h>
#include <string>
#include <istream>
#include <vector>
#include <cassert>

// Global variables exported from the lexer...

extern std::string CurFileName;
extern std::string Textin;
extern int Upgradelineno;
extern std::istream* LexInput;

struct TypeInfo;
typedef std::vector<TypeInfo*> TypeList;

void UpgradeAssembly(
  const std::string & infile, std::istream& in, std::ostream &out, bool debug,
  bool addAttrs);

TypeInfo* ResolveType(TypeInfo*& Ty);

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
  FloatTy, DoubleTy, PointerTy, PackedTy, ArrayTy, StructTy, PackedStructTy, 
  OpaqueTy, VoidTy, LabelTy, FunctionTy, UnresolvedTy, NumericTy
};

/// This type is used to keep track of the signedness of values. Instead
/// of creating llvm::Value directly, the parser will create ValueInfo which
/// associates a Value* with a Signedness indication.
struct ValueInfo {
  std::string* val;
  TypeInfo* type;
  bool constant;
  bool isConstant() const { return constant; }
  inline void destroy();
};

/// This type is used to keep track of the signedness of the obsolete
/// integer types. Instead of creating an llvm::Type directly, the Lexer will
/// create instances of TypeInfo which retains the signedness indication so
/// it can be used by the parser for upgrade decisions.
/// For example if "uint" is encountered then the "first" field will be set 
/// to "int32" and the "second" field will be set to "isUnsigned".  If the 
/// type is not obsolete then "second" will be set to "isSignless".
struct TypeInfo {
  TypeInfo() 
    : newTy(0), oldTy(UnresolvedTy), elemTy(0), resultTy(0), elements(0),
      nelems(0) {
  }

  TypeInfo(const char * newType, Types oldType)
    : newTy(0), oldTy(oldType), elemTy(0), resultTy(0), elements(0), nelems(0) {
    newTy = new std::string(newType);
  }

  TypeInfo(std::string *newType, Types oldType, TypeInfo* eTy = 0, 
           TypeInfo *rTy = 0) 
    : newTy(newType), oldTy(oldType), elemTy(eTy), resultTy(rTy), elements(0),
      nelems(0) { 
  }

  TypeInfo(std::string *newType, Types oldType, TypeInfo *eTy, uint64_t elems)
    : newTy(newType), oldTy(oldType), elemTy(eTy), resultTy(0), elements(0), 
      nelems(elems) {
  }

  TypeInfo(std::string *newType, Types oldType, TypeList* TL)
    : newTy(newType), oldTy(oldType), elemTy(0), resultTy(0), elements(TL),
      nelems(0) {
  }

  TypeInfo(std::string *newType, TypeInfo* resTy, TypeList* TL) 
    : newTy(newType), oldTy(FunctionTy), elemTy(0), resultTy(resTy), 
    elements(TL), nelems(0) {
  }

  TypeInfo(const TypeInfo& that)
    : newTy(0), oldTy(that.oldTy), elemTy(0), resultTy(0), elements(0), 
      nelems(0) {
    *this = that;
  }

  TypeInfo& operator=(const TypeInfo& that) {
    oldTy = that.oldTy;
    nelems = that.nelems;
    if (that.newTy)
      newTy = new std::string(*that.newTy);
    if (that.elemTy)
      elemTy = that.elemTy->clone();
    if (that.resultTy)
      resultTy = that.resultTy->clone();
    if (that.elements) {
      elements = new std::vector<TypeInfo*>(that.elements->size());
      *elements = *that.elements;
    }
    return *this;
  }

  ~TypeInfo() { 
    delete newTy; delete elemTy; delete resultTy; delete elements;
  }

  TypeInfo* clone() const { 
    return new TypeInfo(*this);
  }

  Types getElementTy() const { 
    if (elemTy) {
      return elemTy->oldTy;
    }
    return UnresolvedTy;
  }

  const std::string& getNewTy() const { return *newTy; }
  void setOldTy(Types Ty) { oldTy = Ty; }

  TypeInfo* getResultType() const { return resultTy; }
  TypeInfo* getElementType() const { return elemTy; }

  TypeInfo* getPointerType() const {
    std::string* ty = new std::string(*newTy + "*");
    return new TypeInfo(ty, PointerTy, this->clone(), (TypeInfo*)0);
  }

  bool isUnresolved() const { return oldTy == UnresolvedTy; }
  bool isNumeric() const { return oldTy == NumericTy; }
  bool isVoid() const { return oldTy == VoidTy; }
  bool isBool() const { return oldTy == BoolTy; }
  bool isSigned() const {
    return oldTy == SByteTy || oldTy == ShortTy || 
           oldTy == IntTy || oldTy == LongTy;
  }

  bool isUnsigned() const {
    return oldTy == UByteTy || oldTy == UShortTy || 
           oldTy == UIntTy || oldTy == ULongTy;
  }


  bool isSignless() const { return !isSigned() && !isUnsigned(); }
  bool isInteger() const { return isSigned() || isUnsigned(); }
  bool isIntegral() const { return oldTy == BoolTy || isInteger(); }
  bool isFloatingPoint() const { return oldTy == DoubleTy || oldTy == FloatTy; }
  bool isPacked() const { return oldTy == PackedTy; }
  bool isPointer() const { return oldTy == PointerTy; }
  bool isStruct() const { return oldTy == StructTy || oldTy == PackedStructTy; }
  bool isArray() const { return oldTy == ArrayTy; }
  bool isOther() const { 
    return !isPacked() && !isPointer() && !isFloatingPoint() && !isIntegral(); }
  bool isFunction() const { return oldTy == FunctionTy; }
  bool isComposite() const {
    return isStruct() || isPointer() || isArray() || isPacked();
  }

  bool isAttributeCandidate() const {
    return isIntegral() && getBitWidth() < 32;
  }

  unsigned getBitWidth() const {
    switch (oldTy) {
      default:
      case LabelTy:
      case VoidTy : return 0;
      case BoolTy : return 1;
      case SByteTy: case UByteTy : return 8;
      case ShortTy: case UShortTy : return 16;
      case IntTy: case UIntTy: case FloatTy: return 32;
      case LongTy: case ULongTy: case DoubleTy : return 64;
      case PointerTy: return SizeOfPointer; // global var
      case PackedTy: 
      case ArrayTy: 
        return nelems * elemTy->getBitWidth();
      case StructTy:
      case PackedStructTy: {
        uint64_t size = 0;
        for (unsigned i = 0; i < elements->size(); i++) {
          ResolveType((*elements)[i]);
          size += (*elements)[i]->getBitWidth();
        }
        return size;
      }
    }
  }

  TypeInfo* getIndexedType(const ValueInfo&  VI) {
    if (isStruct()) {
      if (VI.isConstant() && VI.type->isInteger()) {
        size_t pos = VI.val->find(' ') + 1;
        if (pos < VI.val->size()) {
          uint64_t idx = atoi(VI.val->substr(pos).c_str());
          return (*elements)[idx];
        } else {
          yyerror("Invalid value for constant integer");
          return 0;
        }
      } else {
        yyerror("Structure requires constant index");
        return 0;
      }
    }
    if (isArray() || isPacked() || isPointer())
      return elemTy;
    yyerror("Invalid type for getIndexedType");
    return 0;
  }

private:
  std::string* newTy;
  Types oldTy;
  TypeInfo *elemTy;
  TypeInfo *resultTy;
  TypeList *elements;
  uint64_t nelems;
};

/// This type is used to keep track of the signedness of constants.
struct ConstInfo {
  std::string *cnst;
  TypeInfo *type;
  void destroy() { delete cnst; delete type; }
};

typedef std::vector<ValueInfo> ValueList;

inline void ValueInfo::destroy() { delete val; delete type; }

#endif
