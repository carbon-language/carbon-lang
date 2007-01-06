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

// Global variables exported from the lexer...

extern std::string CurFileName;
extern std::string Textin;
extern int Upgradelineno;
extern std::istream* LexInput;

struct TypeInfo;
typedef std::vector<const TypeInfo*> TypeList;

void UpgradeAssembly(
  const std::string & infile, std::istream& in, std::ostream &out, bool debug,
  bool addAttrs);

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
  OpaqueTy, VoidTy, LabelTy, FunctionTy, UnresolvedTy, UpRefTy
};

/// This type is used to keep track of the signedness of values. Instead
/// of creating llvm::Value directly, the parser will create ValueInfo which
/// associates a Value* with a Signedness indication.
struct ValueInfo {
  std::string* val;
  const TypeInfo* type;
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

  static const TypeInfo* get(const std::string &newType, Types oldType);
  static const TypeInfo* get(const std::string& newType, Types oldType, 
                             const TypeInfo* eTy, const TypeInfo* rTy);

  static const TypeInfo* get(const std::string& newType, Types oldType, 
                       const TypeInfo *eTy, uint64_t elems);

  static const TypeInfo* get(const std::string& newType, Types oldType, 
                       TypeList* TL);

  static const TypeInfo* get(const std::string& newType, const TypeInfo* resTy, 
                       TypeList* TL);

  const TypeInfo* resolve() const;
  bool operator<(const TypeInfo& that) const;

  bool sameNewTyAs(const TypeInfo* that) const {
    return this->newTy == that->newTy;
  }

  bool sameOldTyAs(const TypeInfo* that) const;

  Types getElementTy() const {
    if (elemTy) {
      return elemTy->oldTy;
    }
    return UnresolvedTy;
  }

  unsigned getUpRefNum() const {
    assert(oldTy == UpRefTy && "Can't getUpRefNum on non upreference");
    return atoi(&((getNewTy().c_str())[1])); // skip the slash
  }

  typedef std::vector<const TypeInfo*> UpRefStack;
  void getSignedness(unsigned &sNum, unsigned &uNum, UpRefStack& stk) const;
  std::string makeUniqueName(const std::string& BaseName) const;

  const std::string& getNewTy() const { return newTy; }
  const TypeInfo* getResultType() const { return resultTy; }
  const TypeInfo* getElementType() const { return elemTy; }

  const TypeInfo* getPointerType() const {
    return get(newTy + "*", PointerTy, this, (TypeInfo*)0);
  }

  bool isUnresolved() const { return oldTy == UnresolvedTy; }
  bool isUpReference() const { return oldTy == UpRefTy; }
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

  bool isUnresolvedDeep() const;

  unsigned getBitWidth() const;

  const TypeInfo* getIndexedType(const ValueInfo&  VI) const;

  unsigned getNumStructElements() const { 
    return (elements ? elements->size() : 0);
  }

  const TypeInfo* getElement(unsigned idx) const {
    if (elements)
      if (idx < elements->size())
        return (*elements)[idx];
    return 0;
  }


private:
  TypeInfo() 
    : newTy(), oldTy(UnresolvedTy), elemTy(0), resultTy(0), elements(0),
      nelems(0) {
  }

  TypeInfo(const TypeInfo& that); // do not implement
  TypeInfo& operator=(const TypeInfo& that); // do not implement

  ~TypeInfo() { delete elements; }

  struct ltfunctor
  {
    bool operator()(const TypeInfo* X, const TypeInfo* Y) const {
      assert(X && "Can't compare null pointer");
      assert(Y && "Can't compare null pointer");
      return *X < *Y;
    }
  };

  typedef std::set<const TypeInfo*, ltfunctor> TypeRegMap;
  static const TypeInfo* add_new_type(TypeInfo* existing);

  std::string newTy;
  Types oldTy;
  TypeInfo *elemTy;
  TypeInfo *resultTy;
  TypeList *elements;
  uint64_t nelems;
  static TypeRegMap registry;
};

/// This type is used to keep track of the signedness of constants.
struct ConstInfo {
  std::string *cnst;
  const TypeInfo *type;
  void destroy() { delete cnst; }
};

typedef std::vector<ValueInfo> ValueList;

inline void ValueInfo::destroy() { delete val; }

#endif
