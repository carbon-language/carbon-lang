//===-- UpgradeParser.y - Upgrade parser for llvm assmbly -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencer and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements the bison parser for LLVM 1.9 assembly language.
//
//===----------------------------------------------------------------------===//

%{
#include "UpgradeInternals.h"
#include <algorithm>
#include <map>
#include <utility>
#include <iostream>

#define YYERROR_VERBOSE 1
#define YYINCLUDED_STDLIB_H
#define YYDEBUG 1

int yylex();                       // declaration" of xxx warnings.
int yyparse();
extern int yydebug;

static std::string CurFilename;
static std::ostream *O = 0;
std::istream* LexInput = 0;
unsigned SizeOfPointer = 32;
static uint64_t unique = 1;

// This bool controls whether attributes are ever added to function declarations
// definitions and calls.
static bool AddAttributes = false;

// This bool is used to communicate between the InstVal and Inst rules about
// whether or not a cast should be deleted. When the flag is set, InstVal has
// determined that the cast is a candidate. However, it can only be deleted if
// the value being casted is the same value name as the instruction. The Inst
// rule makes that comparison if the flag is set and comments out the
// instruction if they match.
static bool deleteUselessCastFlag = false;
static std::string* deleteUselessCastName = 0;

typedef std::vector<const TypeInfo*> TypeVector;
static TypeVector EnumeratedTypes;
typedef std::map<std::string,const TypeInfo*> TypeMap;
static TypeMap NamedTypes;
typedef std::map<const TypeInfo*,std::string> TypePlaneMap;
typedef std::map<std::string,TypePlaneMap> GlobalsTypeMap;
static GlobalsTypeMap Globals;

static void warning(const std::string& msg);

void destroy(ValueList* VL) {
  while (!VL->empty()) {
    ValueInfo& VI = VL->back();
    VI.destroy();
    VL->pop_back();
  }
  delete VL;
}

void UpgradeAssembly(const std::string &infile, std::istream& in, 
                     std::ostream &out, bool debug, bool addAttrs)
{
  Upgradelineno = 1; 
  CurFilename = infile;
  LexInput = &in;
  yydebug = debug;
  AddAttributes = addAttrs;
  O = &out;

  if (yyparse()) {
    std::cerr << "Parse failed.\n";
    out << "llvm-upgrade parse failed.\n";
    exit(1);
  }
}

TypeInfo::TypeRegMap TypeInfo::registry;

const TypeInfo* TypeInfo::get(const std::string &newType, Types oldType) {
  TypeInfo* Ty = new TypeInfo();
  Ty->newTy = newType;
  Ty->oldTy = oldType;
  return add_new_type(Ty);
}

const TypeInfo* TypeInfo::get(const std::string& newType, Types oldType, 
                              const TypeInfo* eTy, const TypeInfo* rTy) {
  TypeInfo* Ty= new TypeInfo();
  Ty->newTy = newType;
  Ty->oldTy = oldType;
  Ty->elemTy = const_cast<TypeInfo*>(eTy);
  Ty->resultTy = const_cast<TypeInfo*>(rTy);
  return add_new_type(Ty);
}

const TypeInfo* TypeInfo::get(const std::string& newType, Types oldType, 
                              const TypeInfo *eTy, uint64_t elems) {
  TypeInfo* Ty = new TypeInfo();
  Ty->newTy = newType;
  Ty->oldTy = oldType;
  Ty->elemTy = const_cast<TypeInfo*>(eTy);
  Ty->nelems = elems;
  return  add_new_type(Ty);
}

const TypeInfo* TypeInfo::get(const std::string& newType, Types oldType, 
                              TypeList* TL) {
  TypeInfo* Ty = new TypeInfo();
  Ty->newTy = newType;
  Ty->oldTy = oldType;
  Ty->elements = TL;
  return add_new_type(Ty);
}

const TypeInfo* TypeInfo::get(const std::string& newType, const TypeInfo* resTy,
                              TypeList* TL) {
  TypeInfo* Ty = new TypeInfo();
  Ty->newTy = newType;
  Ty->oldTy = FunctionTy;
  Ty->resultTy = const_cast<TypeInfo*>(resTy);
  Ty->elements = TL;
  return add_new_type(Ty);
}

const TypeInfo* TypeInfo::resolve() const {
  if (isUnresolved()) {
    if (getNewTy()[0] == '%' && isdigit(newTy[1])) {
      unsigned ref = atoi(&((newTy.c_str())[1])); // skip the %
      if (ref < EnumeratedTypes.size()) {
        return EnumeratedTypes[ref];
      } else {
        std::string msg("Can't resolve numbered type: ");
        msg += getNewTy();
        yyerror(msg.c_str());
      }
    } else {
      TypeMap::iterator I = NamedTypes.find(newTy);
      if (I != NamedTypes.end()) {
        return I->second;
      } else {
        std::string msg("Cannot resolve type: ");
        msg += getNewTy();
        yyerror(msg.c_str());
      }
    }
  }
  // otherwise its already resolved.
  return this;
}

bool TypeInfo::operator<(const TypeInfo& that) const {
  if (this == &that)
    return false;
  if (oldTy != that.oldTy)
    return oldTy < that.oldTy;
  switch (oldTy) {
    case UpRefTy: {
      unsigned thisUp = this->getUpRefNum();
      unsigned thatUp = that.getUpRefNum();
      return thisUp < thatUp;
    }
    case PackedTy:
    case ArrayTy:
      if (this->nelems != that.nelems)
        return nelems < that.nelems;
    case PointerTy: {
      const TypeInfo* thisTy = this->elemTy;
      const TypeInfo* thatTy = that.elemTy;
      return *thisTy < *thatTy;
    }
    case FunctionTy: {
      const TypeInfo* thisTy = this->resultTy;
      const TypeInfo* thatTy = that.resultTy;
      if (!thisTy->sameOldTyAs(thatTy))
        return *thisTy < *thatTy;
      /* FALL THROUGH */
    }
    case StructTy:
    case PackedStructTy: {
      if (elements->size() != that.elements->size())
        return elements->size() < that.elements->size();
      for (unsigned i = 0; i < elements->size(); i++) {
        const TypeInfo* thisTy = (*this->elements)[i];
        const TypeInfo* thatTy = (*that.elements)[i];
        if (!thisTy->sameOldTyAs(thatTy))
          return *thisTy < *thatTy;
      }
      break;
    }
    case UnresolvedTy:
      return this->newTy < that.newTy;
    default:
      break;
  }
  return false; 
}

bool TypeInfo::sameOldTyAs(const TypeInfo* that) const {
  if (that == 0)
    return false;
  if ( this == that ) 
    return true;
  if (oldTy != that->oldTy)
    return false;
  switch (oldTy) {
    case PackedTy:
    case ArrayTy:
      if (nelems != that->nelems)
        return false;
      /* FALL THROUGH */
    case PointerTy: {
      const TypeInfo* thisTy = this->elemTy;
      const TypeInfo* thatTy = that->elemTy;
      return thisTy->sameOldTyAs(thatTy);
    }
    case FunctionTy: {
      const TypeInfo* thisTy = this->resultTy;
      const TypeInfo* thatTy = that->resultTy;
      if (!thisTy->sameOldTyAs(thatTy))
        return false;
      /* FALL THROUGH */
    }
    case StructTy:
    case PackedStructTy: {
      if (elements->size() != that->elements->size())
        return false;
      for (unsigned i = 0; i < elements->size(); i++) {
        const TypeInfo* thisTy = (*this->elements)[i];
        const TypeInfo* thatTy = (*that->elements)[i];
        if (!thisTy->sameOldTyAs(thatTy))
          return false;
      }
      return true;
    }
    case UnresolvedTy:
      return this->newTy == that->newTy;
    default:
      return true; // for all others oldTy == that->oldTy is sufficient
  }
  return true;
}

bool TypeInfo::isUnresolvedDeep() const {
  switch (oldTy) {
    case UnresolvedTy: 
      return true;
    case PackedTy:
    case ArrayTy:
    case PointerTy:
      return elemTy->isUnresolvedDeep();
    case PackedStructTy:
    case StructTy:
      for (unsigned i = 0; i < elements->size(); i++)
        if ((*elements)[i]->isUnresolvedDeep())
          return true;
      return false;
    default:
      return false;
  }
}

unsigned TypeInfo::getBitWidth() const {
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
        size += (*elements)[i]->getBitWidth();
      }
      return size;
    }
  }
}

const TypeInfo* TypeInfo::getIndexedType(const ValueInfo&  VI) const {
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

void TypeInfo::getSignedness(unsigned &sNum, unsigned &uNum, 
                             UpRefStack& stack) const {
  switch (oldTy) {
    default:
    case OpaqueTy: case LabelTy: case VoidTy: case BoolTy: 
    case FloatTy : case DoubleTy: case UpRefTy:
      return;
    case SByteTy: case ShortTy: case LongTy: case IntTy: 
      sNum++;
      return;
    case UByteTy: case UShortTy: case UIntTy: case ULongTy: 
      uNum++;
      return;
    case PointerTy:
    case PackedTy: 
    case ArrayTy:
      stack.push_back(this);
      elemTy->getSignedness(sNum, uNum, stack);
      return;
    case StructTy:
    case PackedStructTy: {
      stack.push_back(this);
      for (unsigned i = 0; i < elements->size(); i++) {
        (*elements)[i]->getSignedness(sNum, uNum, stack);
      }
      return;
    }
    case UnresolvedTy: {
      const TypeInfo* Ty = this->resolve();
      // Let's not recurse.
      UpRefStack::const_iterator I = stack.begin(), E = stack.end();
      for ( ; I != E && *I != Ty; ++I) 
        ;
      if (I == E)
        Ty->getSignedness(sNum, uNum, stack);
      return;
    }
  }
}

std::string AddSuffix(const std::string& Name, const std::string& Suffix) {
  if (Name[Name.size()-1] == '"') {
    std::string Result = Name;
    Result.insert(Result.size()-1, Suffix);
    return Result;
  }
  return Name + Suffix;
}

std::string TypeInfo::makeUniqueName(const std::string& BaseName) const {
  if (BaseName == "\"alloca point\"")
    return BaseName;
  switch (oldTy) {
    default:
      break;
    case OpaqueTy: case LabelTy: case VoidTy: case BoolTy: case UpRefTy:
    case FloatTy : case DoubleTy: case UnresolvedTy:
      return BaseName;
    case SByteTy: case ShortTy: case LongTy: case IntTy: 
      return AddSuffix(BaseName, ".s");
    case UByteTy: case UShortTy: case UIntTy: case ULongTy: 
      return AddSuffix(BaseName, ".u");
  }

  unsigned uNum = 0, sNum = 0;
  std::string Suffix;
  switch (oldTy) {
    case PointerTy:
    case PackedTy: 
    case ArrayTy: {
      TypeInfo::UpRefStack stack;
      elemTy->resolve()->getSignedness(sNum, uNum, stack);
      break;
    }
    case StructTy:
    case PackedStructTy: {
      for (unsigned i = 0; i < elements->size(); i++) {
        TypeInfo::UpRefStack stack;
        (*elements)[i]->resolve()->getSignedness(sNum, uNum, stack);
      }
      break;
    }
    default:
      assert(0 && "Invalid Type");
      break;
  }

  if (sNum == 0 && uNum == 0)
    return BaseName;

  switch (oldTy) {
    default:             Suffix += ".nada"; break;
    case PointerTy:      Suffix += ".pntr"; break;
    case PackedTy:       Suffix += ".pckd"; break;
    case ArrayTy:        Suffix += ".arry"; break;
    case StructTy:       Suffix += ".strc"; break;
    case PackedStructTy: Suffix += ".pstr"; break;
  }

  Suffix += ".s" + llvm::utostr(sNum);
  Suffix += ".u" + llvm::utostr(uNum);
  return AddSuffix(BaseName, Suffix);
}

TypeInfo& TypeInfo::operator=(const TypeInfo& that) {
  oldTy = that.oldTy;
  nelems = that.nelems;
  newTy = that.newTy;
  elemTy = that.elemTy;
  resultTy = that.resultTy;
  if (that.elements) {
    elements = new TypeList(that.elements->size());
    *elements = *that.elements;
  } else {
    elements = 0;
  }
  return *this;
}

const TypeInfo* TypeInfo::add_new_type(TypeInfo* newTy) {
  TypeRegMap::iterator I = registry.find(newTy);
  if (I != registry.end()) {
    delete newTy;
    return *I;
  }
  registry.insert(newTy);
  return newTy;
}

static const char* getCastOpcode(
  std::string& Source, const TypeInfo* SrcTy, const TypeInfo* DstTy) 
{
  unsigned SrcBits = SrcTy->getBitWidth();
  unsigned DstBits = DstTy->getBitWidth();
  const char* opcode = "bitcast";
  // Run through the possibilities ...
  if (DstTy->isIntegral()) {                        // Casting to integral
    if (SrcTy->isIntegral()) {                      // Casting from integral
      if (DstBits < SrcBits)
        opcode = "trunc";
      else if (DstBits > SrcBits) {                // its an extension
        if (SrcTy->isSigned())
          opcode ="sext";                          // signed -> SEXT
        else
          opcode = "zext";                         // unsigned -> ZEXT
      } else {
        opcode = "bitcast";                        // Same size, No-op cast
      }
    } else if (SrcTy->isFloatingPoint()) {          // Casting from floating pt
      if (DstTy->isSigned()) 
        opcode = "fptosi";                         // FP -> sint
      else
        opcode = "fptoui";                         // FP -> uint 
    } else if (SrcTy->isPacked()) {
      assert(DstBits == SrcTy->getBitWidth() &&
               "Casting packed to integer of different width");
        opcode = "bitcast";                        // same size, no-op cast
    } else {
      assert(SrcTy->isPointer() &&
             "Casting from a value that is not first-class type");
      opcode = "ptrtoint";                         // ptr -> int
    }
  } else if (DstTy->isFloatingPoint()) {           // Casting to floating pt
    if (SrcTy->isIntegral()) {                     // Casting from integral
      if (SrcTy->isSigned())
        opcode = "sitofp";                         // sint -> FP
      else
        opcode = "uitofp";                         // uint -> FP
    } else if (SrcTy->isFloatingPoint()) {         // Casting from floating pt
      if (DstBits < SrcBits) {
        opcode = "fptrunc";                        // FP -> smaller FP
      } else if (DstBits > SrcBits) {
        opcode = "fpext";                          // FP -> larger FP
      } else  {
        opcode ="bitcast";                         // same size, no-op cast
      }
    } else if (SrcTy->isPacked()) {
      assert(DstBits == SrcTy->getBitWidth() &&
             "Casting packed to floating point of different width");
        opcode = "bitcast";                        // same size, no-op cast
    } else {
      assert(0 && "Casting pointer or non-first class to float");
    }
  } else if (DstTy->isPacked()) {
    if (SrcTy->isPacked()) {
      assert(DstTy->getBitWidth() == SrcTy->getBitWidth() &&
             "Casting packed to packed of different widths");
      opcode = "bitcast";                          // packed -> packed
    } else if (DstTy->getBitWidth() == SrcBits) {
      opcode = "bitcast";                          // float/int -> packed
    } else {
      assert(!"Illegal cast to packed (wrong type or size)");
    }
  } else if (DstTy->isPointer()) {
    if (SrcTy->isPointer()) {
      opcode = "bitcast";                          // ptr -> ptr
    } else if (SrcTy->isIntegral()) {
      opcode = "inttoptr";                         // int -> ptr
    } else {
      assert(!"Casting invalid type to pointer");
    }
  } else {
    assert(!"Casting to type that is not first-class");
  }
  return opcode;
}

static std::string getCastUpgrade(const std::string& Src, const TypeInfo* SrcTy,
                                  const TypeInfo* DstTy, bool isConst)
{
  std::string Result;
  std::string Source = Src;
  if (SrcTy->isFloatingPoint() && DstTy->isPointer()) {
    // fp -> ptr cast is no longer supported but we must upgrade this
    // by doing a double cast: fp -> int -> ptr
    if (isConst)
      Source = "i64 fptoui(" + Source + " to i64)";
    else {
      *O << "    %cast_upgrade" << unique << " = fptoui " << Source 
         << " to i64\n";
      Source = "i64 %cast_upgrade" + llvm::utostr(unique);
    }
    // Update the SrcTy for the getCastOpcode call below
    SrcTy = TypeInfo::get("i64", ULongTy);
  } else if (DstTy->isBool()) {
    // cast type %x to bool was previously defined as setne type %x, null
    // The cast semantic is now to truncate, not compare so we must retain
    // the original intent by replacing the cast with a setne
    const char* comparator = SrcTy->isPointer() ? ", null" : 
      (SrcTy->isFloatingPoint() ? ", 0.0" : 
       (SrcTy->isBool() ? ", false" : ", 0"));
    const char* compareOp = SrcTy->isFloatingPoint() ? "fcmp one " : "icmp ne ";
    if (isConst) { 
      Result = "(" + Source + comparator + ")";
      Result = compareOp + Result;
    } else
      Result = compareOp + Source + comparator;
    return Result; // skip cast processing below
  }
  SrcTy = SrcTy->resolve();
  DstTy = DstTy->resolve();
  std::string Opcode(getCastOpcode(Source, SrcTy, DstTy));
  if (isConst)
    Result += Opcode + "( " + Source + " to " + DstTy->getNewTy() + ")";
  else
    Result += Opcode + " " + Source + " to " + DstTy->getNewTy();
  return Result;
}

const char* getDivRemOpcode(const std::string& opcode, const TypeInfo* TI) {
  const char* op = opcode.c_str();
  const TypeInfo* Ty = TI->resolve();
  if (Ty->isPacked())
    Ty = Ty->getElementType();
  if (opcode == "div")
    if (Ty->isFloatingPoint())
      op = "fdiv";
    else if (Ty->isUnsigned())
      op = "udiv";
    else if (Ty->isSigned())
      op = "sdiv";
    else
      yyerror("Invalid type for div instruction");
  else if (opcode == "rem")
    if (Ty->isFloatingPoint())
      op = "frem";
    else if (Ty->isUnsigned())
      op = "urem";
    else if (Ty->isSigned())
      op = "srem";
    else
      yyerror("Invalid type for rem instruction");
  return op;
}

std::string 
getCompareOp(const std::string& setcc, const TypeInfo* TI) {
  assert(setcc.length() == 5);
  char cc1 = setcc[3];
  char cc2 = setcc[4];
  assert(cc1 == 'e' || cc1 == 'n' || cc1 == 'l' || cc1 == 'g');
  assert(cc2 == 'q' || cc2 == 'e' || cc2 == 'e' || cc2 == 't');
  std::string result("xcmp xxx");
  result[6] = cc1;
  result[7] = cc2;
  if (TI->isFloatingPoint()) {
    result[0] = 'f';
    result[5] = 'o';
    if (cc1 == 'n')
      result[5] = 'u'; // NE maps to unordered
    else
      result[5] = 'o'; // everything else maps to ordered
  } else if (TI->isIntegral() || TI->isPointer()) {
    result[0] = 'i';
    if ((cc1 == 'e' && cc2 == 'q') || (cc1 == 'n' && cc2 == 'e'))
      result.erase(5,1);
    else if (TI->isSigned())
      result[5] = 's';
    else if (TI->isUnsigned() || TI->isPointer() || TI->isBool())
      result[5] = 'u';
    else
      yyerror("Invalid integral type for setcc");
  }
  return result;
}

static const TypeInfo* getFunctionReturnType(const TypeInfo* PFTy) {
  PFTy = PFTy->resolve();
  if (PFTy->isPointer()) {
    const TypeInfo* ElemTy = PFTy->getElementType();
    ElemTy = ElemTy->resolve();
    if (ElemTy->isFunction())
      return ElemTy->getResultType();
  } else if (PFTy->isFunction()) {
    return PFTy->getResultType();
  }
  return PFTy;
}

static const TypeInfo* ResolveUpReference(const TypeInfo* Ty, 
                                          TypeInfo::UpRefStack* stack) {
  assert(Ty->isUpReference() && "Can't resolve a non-upreference");
  unsigned upref = Ty->getUpRefNum();
  assert(upref < stack->size() && "Invalid up reference");
  return (*stack)[upref - stack->size() - 1];
}

static const TypeInfo* getGEPIndexedType(const TypeInfo* PTy, ValueList* idxs) {
  const TypeInfo* Result = PTy = PTy->resolve();
  assert(PTy->isPointer() && "GEP Operand is not a pointer?");
  TypeInfo::UpRefStack stack;
  for (unsigned i = 0; i < idxs->size(); ++i) {
    if (Result->isComposite()) {
      Result = Result->getIndexedType((*idxs)[i]);
      Result = Result->resolve();
      stack.push_back(Result);
    } else
      yyerror("Invalid type for index");
  }
  // Resolve upreferences so we can return a more natural type
  if (Result->isPointer()) {
    if (Result->getElementType()->isUpReference()) {
      stack.push_back(Result);
      Result = ResolveUpReference(Result->getElementType(), &stack);
    }
  } else if (Result->isUpReference()) {
    Result = ResolveUpReference(Result->getElementType(), &stack);
  }
  return Result->getPointerType();
}


// This function handles appending .u or .s to integer value names that
// were previously unsigned or signed, respectively. This avoids name
// collisions since the unsigned and signed type planes have collapsed
// into a single signless type plane.
static std::string getUniqueName(const std::string *Name, const TypeInfo* Ty,
                                 bool isGlobal = false, bool isDef = false) {

  // If its not a symbolic name, don't modify it, probably a constant val.
  if ((*Name)[0] != '%' && (*Name)[0] != '"')
    return *Name;

  // If its a numeric reference, just leave it alone.
  if (isdigit((*Name)[1]))
    return *Name;

  // Resolve the type
  Ty = Ty->resolve(); 

  // If its a global name, get its uniquified name, if any
  GlobalsTypeMap::iterator GI = Globals.find(*Name);
  if (GI != Globals.end()) {
    TypePlaneMap::iterator TPI = GI->second.begin();
    TypePlaneMap::iterator TPE = GI->second.end();
    for ( ; TPI != TPE ; ++TPI) {
      if (TPI->first->sameNewTyAs(Ty)) 
        return TPI->second;
    }
  }

  if (isGlobal) {
    // We didn't find a global name, but if its supposed to be global then all 
    // we can do is return the name. This is probably a forward reference of a 
    // global value that hasn't been defined yet. Since we have no definition
    // we don't know its linkage class. Just assume its an external and the name
    // shouldn't change.
    return *Name;
  }

  // Default the result to the current name
  std::string Result = Ty->makeUniqueName(*Name);

  return Result;
}

static unsigned UniqueNameCounter = 0;

std::string getGlobalName(const std::string* Name, const std::string Linkage,
                          const TypeInfo* Ty, bool isConstant) {
  // Default to given name
  std::string Result = *Name; 
  // Look up the name in the Globals Map
  GlobalsTypeMap::iterator GI = Globals.find(*Name);
  // Did we see this global name before?
  if (GI != Globals.end()) {
    if (Ty->isUnresolvedDeep()) {
      // The Gval's type is unresolved. Consequently, we can't disambiguate it
      // by type. We'll just change its name and emit a warning.
      warning("Cannot disambiguate global value '" + *Name + 
              "' because type '" + Ty->getNewTy() + "'is unresolved.\n");
      Result = *Name + ".unique";
      UniqueNameCounter++;
      Result += llvm::utostr(UniqueNameCounter);
      return Result;
    } else {
      TypePlaneMap::iterator TPI = GI->second.find(Ty);
      if (TPI != GI->second.end()) {
        // We found an existing name of the same old type. This isn't allowed 
        // in LLVM 2.0. Consequently, we must alter the name of the global so it
        // can at least compile. References to the global will yield the first
        // definition, which is okay. We also must warn about this.
        Result = *Name + ".unique";
        UniqueNameCounter++;
        Result += llvm::utostr(UniqueNameCounter);
        warning(std::string("Global variable '") + *Name + "' was renamed to '"+
                Result + "'");
      } else { 
        // There isn't an existing definition for this name according to the
        // old types. Now search the TypePlanMap for types with the same new
        // name. 
        TypePlaneMap::iterator TPI = GI->second.begin();
        TypePlaneMap::iterator TPE = GI->second.end();
        for ( ; TPI != TPE; ++TPI) {
          if (TPI->first->sameNewTyAs(Ty)) {
            // The new types are the same but the old types are different so 
            // this is a global name collision resulting from type planes 
            // collapsing. 
            if (Linkage == "external" || Linkage == "dllimport" || 
                Linkage == "extern_weak" || Linkage == "") {
              // The linkage of this gval is external so we can't reliably 
              // rename it because it could potentially create a linking 
              // problem.  However, we can't leave the name conflict in the 
              // output either or it won't assemble with LLVM 2.0.  So, all we 
              // can do is rename this one to something unique and emit a 
              // warning about the problem.
              Result = *Name + ".unique";
              UniqueNameCounter++;
              Result += llvm::utostr(UniqueNameCounter);
              warning("Renaming global value '" + *Name + "' to '" + Result + 
                      "' may cause linkage errors.");
              return Result;
            } else {
              // Its linkage is internal and its type is known so we can 
              // disambiguate the name collision successfully based on the type.
              Result = getUniqueName(Name, Ty);
              TPI->second = Result;
              return Result;
            }
          }
        }
        // We didn't find an entry in the type plane with the same new type and
        // the old types differ so this is a new type plane for this global 
        // variable. We just fall through to the logic below which inserts
        // the global.
      }
    }
  }

  // Its a new global name, if it is external we can't change it
  if (isConstant || Linkage == "external" || Linkage == "dllimport" || 
      Linkage == "extern_weak" || Linkage == "") {
    Globals[Result][Ty] = Result;
    return Result;
  }

  // Its a new global name, and it is internal, change the name to make it
  // unique for its type.
  // Result = getUniqueName(Name, Ty);
  Globals[*Name][Ty] = Result;
  return Result;
}
%}

// %file-prefix="UpgradeParser"

%union {
  std::string*    String;
  const TypeInfo* Type;
  ValueInfo       Value;
  ConstInfo       Const;
  ValueList*      ValList;
  TypeList*       TypeVec;
}

%token <Type>   VOID BOOL SBYTE UBYTE SHORT USHORT INT UINT LONG ULONG
%token <Type>   FLOAT DOUBLE LABEL 
%token <String> OPAQUE ESINT64VAL EUINT64VAL SINTVAL UINTVAL FPVAL
%token <String> NULL_TOK UNDEF ZEROINITIALIZER TRUETOK FALSETOK
%token <String> TYPE VAR_ID LABELSTR STRINGCONSTANT
%token <String> IMPLEMENTATION BEGINTOK ENDTOK
%token <String> DECLARE GLOBAL CONSTANT SECTION VOLATILE
%token <String> TO DOTDOTDOT CONST INTERNAL LINKONCE WEAK 
%token <String> DLLIMPORT DLLEXPORT EXTERN_WEAK APPENDING
%token <String> EXTERNAL TARGET TRIPLE ENDIAN POINTERSIZE LITTLE BIG
%token <String> ALIGN UNINITIALIZED
%token <String> DEPLIBS CALL TAIL ASM_TOK MODULE SIDEEFFECT
%token <String> CC_TOK CCC_TOK CSRETCC_TOK FASTCC_TOK COLDCC_TOK
%token <String> X86_STDCALLCC_TOK X86_FASTCALLCC_TOK
%token <String> DATALAYOUT
%token <String> RET BR SWITCH INVOKE EXCEPT UNWIND UNREACHABLE
%token <String> ADD SUB MUL DIV UDIV SDIV FDIV REM UREM SREM FREM AND OR XOR
%token <String> SETLE SETGE SETLT SETGT SETEQ SETNE  // Binary Comparators
%token <String> ICMP FCMP EQ NE SLT SGT SLE SGE OEQ ONE OLT OGT OLE OGE 
%token <String> ORD UNO UEQ UNE ULT UGT ULE UGE
%token <String> MALLOC ALLOCA FREE LOAD STORE GETELEMENTPTR
%token <String> PHI_TOK SELECT SHL SHR ASHR LSHR VAARG
%token <String> EXTRACTELEMENT INSERTELEMENT SHUFFLEVECTOR
%token <String> CAST TRUNC ZEXT SEXT FPTRUNC FPEXT FPTOUI FPTOSI UITOFP SITOFP 
%token <String> PTRTOINT INTTOPTR BITCAST

%type <String> OptAssign OptLinkage OptCallingConv OptAlign OptCAlign 
%type <String> SectionString OptSection GlobalVarAttributes GlobalVarAttribute
%type <String> ConstExpr DefinitionList
%type <String> ConstPool TargetDefinition LibrariesDefinition LibList OptName
%type <String> ArgVal ArgListH ArgList FunctionHeaderH BEGIN FunctionHeader END
%type <String> Function FunctionProto BasicBlock 
%type <String> InstructionList BBTerminatorInst JumpTable Inst
%type <String> OptTailCall OptVolatile Unwind
%type <String> SymbolicValueRef OptSideEffect GlobalType
%type <String> FnDeclareLinkage BasicBlockList BigOrLittle AsmBlock
%type <String> Name ConstValueRef ConstVector External
%type <String> ShiftOps SetCondOps LogicalOps ArithmeticOps CastOps
%type <String> IPredicates FPredicates

%type <ValList> ValueRefList ValueRefListE IndexList
%type <TypeVec> TypeListI ArgTypeListI

%type <Type> IntType SIntType UIntType FPType TypesV Types 
%type <Type> PrimType UpRTypesV UpRTypes

%type <String> IntVal EInt64Val 
%type <Const>  ConstVal

%type <Value> ValueRef ResolvedVal InstVal PHIList MemoryInst

%start Module

%%

// Handle constant integer size restriction and conversion...
IntVal : SINTVAL | UINTVAL ;
EInt64Val : ESINT64VAL | EUINT64VAL;

// Operations that are notably excluded from this list include:
// RET, BR, & SWITCH because they end basic blocks and are treated specially.
ArithmeticOps: ADD | SUB | MUL | DIV | UDIV | SDIV | FDIV 
             | REM | UREM | SREM | FREM;
LogicalOps   : AND | OR | XOR;
SetCondOps   : SETLE | SETGE | SETLT | SETGT | SETEQ | SETNE;
IPredicates  : EQ | NE | SLT | SGT | SLE | SGE | ULT | UGT | ULE | UGE;
FPredicates  : OEQ | ONE | OLT | OGT | OLE | OGE | ORD | UNO | UEQ | UNE
             | ULT | UGT | ULE | UGE | TRUETOK | FALSETOK;
ShiftOps     : SHL | SHR | ASHR | LSHR;
CastOps      : TRUNC | ZEXT | SEXT | FPTRUNC | FPEXT | FPTOUI | FPTOSI | 
               UITOFP | SITOFP | PTRTOINT | INTTOPTR | BITCAST | CAST
             ;

// These are some types that allow classification if we only want a particular 
// thing... for example, only a signed, unsigned, or integral type.
SIntType :  LONG |  INT |  SHORT | SBYTE;
UIntType : ULONG | UINT | USHORT | UBYTE;
IntType  : SIntType | UIntType;
FPType   : FLOAT | DOUBLE;

// OptAssign - Value producing statements have an optional assignment component
OptAssign : Name '=' {
    $$ = $1;
  }
  | /*empty*/ {
    $$ = new std::string(""); 
  };

OptLinkage 
  : INTERNAL | LINKONCE | WEAK | APPENDING | DLLIMPORT | DLLEXPORT 
  | EXTERN_WEAK 
  | /*empty*/   { $$ = new std::string(""); } ;

OptCallingConv 
  : CCC_TOK | CSRETCC_TOK | FASTCC_TOK | COLDCC_TOK | X86_STDCALLCC_TOK 
  | X86_FASTCALLCC_TOK 
  | CC_TOK EUINT64VAL { 
    *$1 += *$2; 
    delete $2;
    $$ = $1; 
    }
  | /*empty*/ { $$ = new std::string(""); } ;

// OptAlign/OptCAlign - An optional alignment, and an optional alignment with
// a comma before it.
OptAlign 
  : /*empty*/        { $$ = new std::string(); }
  | ALIGN EUINT64VAL { *$1 += " " + *$2; delete $2; $$ = $1; };

OptCAlign 
  : /*empty*/            { $$ = new std::string(); } 
  | ',' ALIGN EUINT64VAL { 
    $2->insert(0, ", "); 
    *$2 += " " + *$3;
    delete $3;
    $$ = $2;
  };

SectionString 
  : SECTION STRINGCONSTANT { 
    *$1 += " " + *$2;
    delete $2;
    $$ = $1;
  };

OptSection : /*empty*/     { $$ = new std::string(); } 
           | SectionString;

GlobalVarAttributes 
    : /* empty */ { $$ = new std::string(); } 
    | ',' GlobalVarAttribute GlobalVarAttributes  {
      $2->insert(0, ", ");
      if (!$3->empty())
        *$2 += " " + *$3;
      delete $3;
      $$ = $2;
    };

GlobalVarAttribute 
    : SectionString 
    | ALIGN EUINT64VAL {
      *$1 += " " + *$2;
      delete $2;
      $$ = $1;
    };

//===----------------------------------------------------------------------===//
// Types includes all predefined types... except void, because it can only be
// used in specific contexts (function returning void for example).  To have
// access to it, a user must explicitly use TypesV.
//

// TypesV includes all of 'Types', but it also includes the void type.
TypesV    : Types    | VOID ;
UpRTypesV : UpRTypes | VOID ; 
Types     : UpRTypes ;

// Derived types are added later...
//
PrimType : BOOL | SBYTE | UBYTE | SHORT  | USHORT | INT   | UINT ;
PrimType : LONG | ULONG | FLOAT | DOUBLE | LABEL;
UpRTypes 
  : OPAQUE { 
    $$ = TypeInfo::get(*$1, OpaqueTy);
  } 
  | SymbolicValueRef { 
    $$ = TypeInfo::get(*$1, UnresolvedTy);
  }
  | PrimType { 
    $$ = $1; 
  }
  | '\\' EUINT64VAL {                   // Type UpReference
    $2->insert(0, "\\");
    $$ = TypeInfo::get(*$2, UpRefTy);
  }
  | UpRTypesV '(' ArgTypeListI ')' {           // Function derived type?
    std::string newTy( $1->getNewTy() + "(");
    for (unsigned i = 0; i < $3->size(); ++i) {
      if (i != 0)
        newTy +=  ", ";
      if ((*$3)[i]->isVoid())
        newTy += "...";
      else
        newTy += (*$3)[i]->getNewTy();
    }
    newTy += ")";
    $$ = TypeInfo::get(newTy, $1, $3);
  }
  | '[' EUINT64VAL 'x' UpRTypes ']' {          // Sized array type?
    uint64_t elems = atoi($2->c_str());
    $2->insert(0,"[ ");
    *$2 += " x " + $4->getNewTy() + " ]";
    $$ = TypeInfo::get(*$2, ArrayTy, $4, elems);
  }
  | '<' EUINT64VAL 'x' UpRTypes '>' {          // Packed array type?
    uint64_t elems = atoi($2->c_str());
    $2->insert(0,"< ");
    *$2 += " x " + $4->getNewTy() + " >";
    $$ = TypeInfo::get(*$2, PackedTy, $4, elems);
  }
  | '{' TypeListI '}' {                        // Structure type?
    std::string newTy("{");
    for (unsigned i = 0; i < $2->size(); ++i) {
      if (i != 0)
        newTy +=  ", ";
      newTy += (*$2)[i]->getNewTy();
    }
    newTy += "}";
    $$ = TypeInfo::get(newTy, StructTy, $2);
  }
  | '{' '}' {                                  // Empty structure type?
    $$ = TypeInfo::get("{}", StructTy, new TypeList());
  }
  | '<' '{' TypeListI '}' '>' {                // Packed Structure type?
    std::string newTy("<{");
    for (unsigned i = 0; i < $3->size(); ++i) {
      if (i != 0)
        newTy +=  ", ";
      newTy += (*$3)[i]->getNewTy();
    }
    newTy += "}>";
    $$ = TypeInfo::get(newTy, PackedStructTy, $3);
  }
  | '<' '{' '}' '>' {                          // Empty packed structure type?
    $$ = TypeInfo::get("<{}>", PackedStructTy, new TypeList());
  }
  | UpRTypes '*' {                             // Pointer type?
    $$ = $1->getPointerType();
  };

// TypeList - Used for struct declarations and as a basis for function type 
// declaration type lists
//
TypeListI 
  : UpRTypes {
    $$ = new TypeList();
    $$->push_back($1);
  }
  | TypeListI ',' UpRTypes {
    $$ = $1;
    $$->push_back($3);
  };

// ArgTypeList - List of types for a function type declaration...
ArgTypeListI 
  : TypeListI 
  | TypeListI ',' DOTDOTDOT {
    $$ = $1;
    $$->push_back(TypeInfo::get("void",VoidTy));
    delete $3;
  }
  | DOTDOTDOT {
    $$ = new TypeList();
    $$->push_back(TypeInfo::get("void",VoidTy));
    delete $1;
  }
  | /*empty*/ {
    $$ = new TypeList();
  };

// ConstVal - The various declarations that go into the constant pool.  This
// production is used ONLY to represent constants that show up AFTER a 'const',
// 'constant' or 'global' token at global scope.  Constants that can be inlined
// into other expressions (such as integers and constexprs) are handled by the
// ResolvedVal, ValueRef and ConstValueRef productions.
//
ConstVal: Types '[' ConstVector ']' { // Nonempty unsized arr
    $$.type = $1;
    $$.cnst = new std::string($1->getNewTy());
    *$$.cnst += " [ " + *$3 + " ]";
    delete $3;
  }
  | Types '[' ']' {
    $$.type = $1;
    $$.cnst = new std::string($1->getNewTy());
    *$$.cnst += "[ ]";
  }
  | Types 'c' STRINGCONSTANT {
    $$.type = $1;
    $$.cnst = new std::string($1->getNewTy());
    *$$.cnst += " c" + *$3;
    delete $3;
  }
  | Types '<' ConstVector '>' { // Nonempty unsized arr
    $$.type = $1;
    $$.cnst = new std::string($1->getNewTy());
    *$$.cnst += " < " + *$3 + " >";
    delete $3;
  }
  | Types '{' ConstVector '}' {
    $$.type = $1;
    $$.cnst = new std::string($1->getNewTy());
    *$$.cnst += " { " + *$3 + " }";
    delete $3;
  }
  | Types '{' '}' {
    $$.type = $1;
    $$.cnst = new std::string($1->getNewTy());
    *$$.cnst += " {}";
  }
  | Types NULL_TOK {
    $$.type = $1;
    $$.cnst = new std::string($1->getNewTy());
    *$$.cnst +=  " " + *$2;
    delete $2;
  }
  | Types UNDEF {
    $$.type = $1;
    $$.cnst = new std::string($1->getNewTy());
    *$$.cnst += " " + *$2;
    delete $2;
  }
  | Types SymbolicValueRef {
    std::string Name = getUniqueName($2, $1->resolve(), true);
    $$.type = $1;
    $$.cnst = new std::string($1->getNewTy());
    *$$.cnst += " " + Name;
    delete $2;
  }
  | Types ConstExpr {
    $$.type = $1;
    $$.cnst = new std::string($1->getNewTy());
    *$$.cnst += " " + *$2;
    delete $2;
  }
  | Types ZEROINITIALIZER {
    $$.type = $1;
    $$.cnst = new std::string($1->getNewTy());
    *$$.cnst += " " + *$2;
    delete $2;
  }
  | SIntType EInt64Val {      // integral constants
    $$.type = $1;
    $$.cnst = new std::string($1->getNewTy());
    *$$.cnst += " " + *$2;
    delete $2;
  }
  | UIntType EInt64Val {            // integral constants
    $$.type = $1;
    $$.cnst = new std::string($1->getNewTy());
    *$$.cnst += " " + *$2;
    delete $2;
  }
  | BOOL TRUETOK {                      // Boolean constants
    $$.type = $1;
    $$.cnst = new std::string($1->getNewTy());
    *$$.cnst += " " + *$2;
    delete $2;
  }
  | BOOL FALSETOK {                     // Boolean constants
    $$.type = $1;
    $$.cnst = new std::string($1->getNewTy());
    *$$.cnst += " " + *$2;
    delete $2;
  }
  | FPType FPVAL {                   // Float & Double constants
    $$.type = $1;
    $$.cnst = new std::string($1->getNewTy());
    *$$.cnst += " " + *$2;
    delete $2;
  };


ConstExpr: CastOps '(' ConstVal TO Types ')' {
    std::string source = *$3.cnst;
    const TypeInfo* SrcTy = $3.type->resolve();
    const TypeInfo* DstTy = $5->resolve(); 
    if (*$1 == "cast") {
      // Call getCastUpgrade to upgrade the old cast
      $$ = new std::string(getCastUpgrade(source, SrcTy, DstTy, true));
    } else {
      // Nothing to upgrade, just create the cast constant expr
      $$ = new std::string(*$1);
      *$$ += "( " + source + " to " + $5->getNewTy() + ")";
    }
    delete $1; $3.destroy(); delete $4;
  }
  | GETELEMENTPTR '(' ConstVal IndexList ')' {
    *$1 += "(" + *$3.cnst;
    for (unsigned i = 0; i < $4->size(); ++i) {
      ValueInfo& VI = (*$4)[i];
      *$1 += ", " + *VI.val;
      VI.destroy();
    }
    *$1 += ")";
    $$ = $1;
    $3.destroy();
    delete $4;
  }
  | SELECT '(' ConstVal ',' ConstVal ',' ConstVal ')' {
    *$1 += "(" + *$3.cnst + "," + *$5.cnst + "," + *$7.cnst + ")";
    $3.destroy(); $5.destroy(); $7.destroy();
    $$ = $1;
  }
  | ArithmeticOps '(' ConstVal ',' ConstVal ')' {
    const char* op = getDivRemOpcode(*$1, $3.type); 
    $$ = new std::string(op);
    *$$ += "(" + *$3.cnst + "," + *$5.cnst + ")";
    delete $1; $3.destroy(); $5.destroy();
  }
  | LogicalOps '(' ConstVal ',' ConstVal ')' {
    *$1 += "(" + *$3.cnst + "," + *$5.cnst + ")";
    $3.destroy(); $5.destroy();
    $$ = $1;
  }
  | SetCondOps '(' ConstVal ',' ConstVal ')' {
    *$1 = getCompareOp(*$1, $3.type);
    *$1 += "(" + *$3.cnst + "," + *$5.cnst + ")";
    $3.destroy(); $5.destroy();
    $$ = $1;
  }
  | ICMP IPredicates '(' ConstVal ',' ConstVal ')' {
    *$1 += " " + *$2 + " (" +  *$4.cnst + "," + *$6.cnst + ")";
    delete $2; $4.destroy(); $6.destroy();
    $$ = $1;
  }
  | FCMP FPredicates '(' ConstVal ',' ConstVal ')' {
    *$1 += " " + *$2 + " (" + *$4.cnst + "," + *$6.cnst + ")";
    delete $2; $4.destroy(); $6.destroy();
    $$ = $1;
  }
  | ShiftOps '(' ConstVal ',' ConstVal ')' {
    const char* shiftop = $1->c_str();
    if (*$1 == "shr")
      shiftop = ($3.type->isUnsigned()) ? "lshr" : "ashr";
    $$ = new std::string(shiftop);
    *$$ += "(" + *$3.cnst + "," + *$5.cnst + ")";
    delete $1; $3.destroy(); $5.destroy();
  }
  | EXTRACTELEMENT '(' ConstVal ',' ConstVal ')' {
    *$1 += "(" + *$3.cnst + "," + *$5.cnst + ")";
    $3.destroy(); $5.destroy();
    $$ = $1;
  }
  | INSERTELEMENT '(' ConstVal ',' ConstVal ',' ConstVal ')' {
    *$1 += "(" + *$3.cnst + "," + *$5.cnst + "," + *$7.cnst + ")";
    $3.destroy(); $5.destroy(); $7.destroy();
    $$ = $1;
  }
  | SHUFFLEVECTOR '(' ConstVal ',' ConstVal ',' ConstVal ')' {
    *$1 += "(" + *$3.cnst + "," + *$5.cnst + "," + *$7.cnst + ")";
    $3.destroy(); $5.destroy(); $7.destroy();
    $$ = $1;
  };


// ConstVector - A list of comma separated constants.

ConstVector 
  : ConstVector ',' ConstVal {
    *$1 += ", " + *$3.cnst;
    $3.destroy();
    $$ = $1;
  }
  | ConstVal { $$ = new std::string(*$1.cnst); $1.destroy(); }
  ;


// GlobalType - Match either GLOBAL or CONSTANT for global declarations...
GlobalType : GLOBAL | CONSTANT ;


//===----------------------------------------------------------------------===//
//                             Rules to match Modules
//===----------------------------------------------------------------------===//

// Module rule: Capture the result of parsing the whole file into a result
// variable...
//
Module : DefinitionList {
};

// DefinitionList - Top level definitions
//
DefinitionList : DefinitionList Function {
    $$ = 0;
  } 
  | DefinitionList FunctionProto {
    *O << *$2 << '\n';
    delete $2;
    $$ = 0;
  }
  | DefinitionList MODULE ASM_TOK AsmBlock {
    *O << "module asm " << ' ' << *$4 << '\n';
    $$ = 0;
  }  
  | DefinitionList IMPLEMENTATION {
    *O << "implementation\n";
    $$ = 0;
  }
  | ConstPool { $$ = 0; }

External : EXTERNAL | UNINITIALIZED { $$ = $1; *$$ = "external"; }

// ConstPool - Constants with optional names assigned to them.
ConstPool : ConstPool OptAssign TYPE TypesV {
    EnumeratedTypes.push_back($4);
    if (!$2->empty()) {
      NamedTypes[*$2] = $4;
      *O << *$2 << " = ";
    }
    *O << "type " << $4->getNewTy() << '\n';
    delete $2; delete $3;
    $$ = 0;
  }
  | ConstPool FunctionProto {       // Function prototypes can be in const pool
    *O << *$2 << '\n';
    delete $2;
    $$ = 0;
  }
  | ConstPool MODULE ASM_TOK AsmBlock {  // Asm blocks can be in the const pool
    *O << *$2 << ' ' << *$3 << ' ' << *$4 << '\n';
    delete $2; delete $3; delete $4; 
    $$ = 0;
  }
  | ConstPool OptAssign OptLinkage GlobalType ConstVal  GlobalVarAttributes {
    if (!$2->empty()) {
      std::string Name = getGlobalName($2,*$3, $5.type->getPointerType(),
                                       *$4 == "constant");
      *O << Name << " = ";
    }
    *O << *$3 << ' ' << *$4 << ' ' << *$5.cnst << ' ' << *$6 << '\n';
    delete $2; delete $3; delete $4; delete $6; 
    $$ = 0;
  }
  | ConstPool OptAssign External GlobalType Types  GlobalVarAttributes {
    if (!$2->empty()) {
      std::string Name = getGlobalName($2,*$3,$5->getPointerType(),
                                       *$4 == "constant");
      *O << Name << " = ";
    }
    *O <<  *$3 << ' ' << *$4 << ' ' << $5->getNewTy() << ' ' << *$6 << '\n';
    delete $2; delete $3; delete $4; delete $6;
    $$ = 0;
  }
  | ConstPool OptAssign DLLIMPORT GlobalType Types GlobalVarAttributes {
    if (!$2->empty()) {
      std::string Name = getGlobalName($2,*$3,$5->getPointerType(),
                                       *$4 == "constant");
      *O << Name << " = ";
    }
    *O << *$3 << ' ' << *$4 << ' ' << $5->getNewTy() << ' ' << *$6 << '\n';
    delete $2; delete $3; delete $4; delete $6;
    $$ = 0;
  }
  | ConstPool OptAssign EXTERN_WEAK GlobalType Types  GlobalVarAttributes {
    if (!$2->empty()) {
      std::string Name = getGlobalName($2,*$3,$5->getPointerType(),
                                       *$4 == "constant");
      *O << Name << " = ";
    }
    *O << *$3 << ' ' << *$4 << ' ' << $5->getNewTy() << ' ' << *$6 << '\n';
    delete $2; delete $3; delete $4; delete $6;
    $$ = 0;
  }
  | ConstPool TARGET TargetDefinition { 
    *O << *$2 << ' ' << *$3 << '\n';
    delete $2; delete $3;
    $$ = 0;
  }
  | ConstPool DEPLIBS '=' LibrariesDefinition {
    *O << *$2 << " = " << *$4 << '\n';
    delete $2; delete $4;
    $$ = 0;
  }
  | /* empty: end of list */ { 
    $$ = 0;
  };


AsmBlock : STRINGCONSTANT ;

BigOrLittle : BIG | LITTLE 

TargetDefinition 
  : ENDIAN '=' BigOrLittle {
    *$1 += " = " + *$3;
    delete $3;
    $$ = $1;
  }
  | POINTERSIZE '=' EUINT64VAL {
    *$1 += " = " + *$3;
    if (*$3 == "64")
      SizeOfPointer = 64;
    delete $3;
    $$ = $1;
  }
  | TRIPLE '=' STRINGCONSTANT {
    *$1 += " = " + *$3;
    delete $3;
    $$ = $1;
  }
  | DATALAYOUT '=' STRINGCONSTANT {
    *$1 += " = " + *$3;
    delete $3;
    $$ = $1;
  };

LibrariesDefinition 
  : '[' LibList ']' {
    $2->insert(0, "[ ");
    *$2 += " ]";
    $$ = $2;
  };

LibList 
  : LibList ',' STRINGCONSTANT {
    *$1 += ", " + *$3;
    delete $3;
    $$ = $1;
  }
  | STRINGCONSTANT 
  | /* empty: end of list */ {
    $$ = new std::string();
  };

//===----------------------------------------------------------------------===//
//                       Rules to match Function Headers
//===----------------------------------------------------------------------===//

Name : VAR_ID | STRINGCONSTANT;
OptName : Name | /*empty*/ { $$ = new std::string(); };

ArgVal : Types OptName {
  $$ = new std::string($1->getNewTy());
  if (!$2->empty()) {
    std::string Name = getUniqueName($2, $1->resolve());
    *$$ += " " + Name;
  }
  delete $2;
};

ArgListH : ArgListH ',' ArgVal {
    *$1 += ", " + *$3;
    delete $3;
  }
  | ArgVal {
    $$ = $1;
  };

ArgList : ArgListH {
    $$ = $1;
  }
  | ArgListH ',' DOTDOTDOT {
    *$1 += ", ...";
    $$ = $1;
    delete $3;
  }
  | DOTDOTDOT {
    $$ = $1;
  }
  | /* empty */ { $$ = new std::string(); };

FunctionHeaderH 
  : OptCallingConv TypesV Name '(' ArgList ')' OptSection OptAlign {
    if (!$1->empty()) {
      *$1 += " ";
    }
    *$1 += $2->getNewTy() + " " + *$3 + "(" + *$5 + ")";
    if (!$7->empty()) {
      *$1 += " " + *$7;
    }
    if (!$8->empty()) {
      *$1 += " " + *$8;
    }
    delete $3;
    delete $5;
    delete $7;
    delete $8;
    $$ = $1;
  };

BEGIN : BEGINTOK { $$ = new std::string("{"); delete $1; }
  | '{' { $$ = new std::string ("{"); }

FunctionHeader 
  : OptLinkage FunctionHeaderH BEGIN {
    *O << "define ";
    if (!$1->empty()) {
      *O << *$1 << ' ';
    }
    *O << *$2 << ' ' << *$3 << '\n';
    delete $1; delete $2; delete $3;
    $$ = 0;
  }
  ;

END : ENDTOK { $$ = new std::string("}"); delete $1; }
    | '}' { $$ = new std::string("}"); };

Function : FunctionHeader BasicBlockList END {
  if ($2)
    *O << *$2;
  *O << *$3 << "\n\n";
  delete $1; delete $2; delete $3;
  $$ = 0;
};

FnDeclareLinkage
  : /*default*/ { $$ = new std::string(); }
  | DLLIMPORT    
  | EXTERN_WEAK 
  ;
  
FunctionProto 
  : DECLARE FnDeclareLinkage FunctionHeaderH { 
    if (!$2->empty())
      *$1 += " " + *$2;
    *$1 += " " + *$3;
    delete $2;
    delete $3;
    $$ = $1;
  };

//===----------------------------------------------------------------------===//
//                        Rules to match Basic Blocks
//===----------------------------------------------------------------------===//

OptSideEffect : /* empty */ { $$ = new std::string(); }
  | SIDEEFFECT;

ConstValueRef 
  : ESINT64VAL | EUINT64VAL | FPVAL | TRUETOK | FALSETOK | NULL_TOK | UNDEF
  | ZEROINITIALIZER 
  | '<' ConstVector '>' { 
    $2->insert(0, "<");
    *$2 += ">";
    $$ = $2;
  }
  | ConstExpr 
  | ASM_TOK OptSideEffect STRINGCONSTANT ',' STRINGCONSTANT {
    if (!$2->empty()) {
      *$1 += " " + *$2;
    }
    *$1 += " " + *$3 + ", " + *$5;
    delete $2; delete $3; delete $5;
    $$ = $1;
  };

SymbolicValueRef : IntVal | Name ;

// ValueRef - A reference to a definition... either constant or symbolic
ValueRef 
  : SymbolicValueRef {
    $$.val = $1;
    $$.constant = false;
    $$.type = 0;
  }
  | ConstValueRef {
    $$.val = $1;
    $$.constant = true;
    $$.type = 0;
  }
  ;

// ResolvedVal - a <type> <value> pair.  This is used only in cases where the
// type immediately preceeds the value reference, and allows complex constant
// pool references (for things like: 'ret [2 x int] [ int 12, int 42]')
ResolvedVal : Types ValueRef {
    $1 = $1->resolve();
    std::string Name = getUniqueName($2.val, $1);
    $$ = $2;
    delete $$.val;
    $$.val = new std::string($1->getNewTy() + " " + Name);
    $$.type = $1;
  };

BasicBlockList : BasicBlockList BasicBlock {
    $$ = 0;
  }
  | BasicBlock { // Do not allow functions with 0 basic blocks   
    $$ = 0;
  };


// Basic blocks are terminated by branching instructions: 
// br, br/cc, switch, ret
//
BasicBlock : InstructionList BBTerminatorInst  {
    $$ = 0;
  };

InstructionList : InstructionList Inst {
    *O << "    " << *$2 << '\n';
    delete $2;
    $$ = 0;
  }
  | /* empty */ {
    $$ = 0;
  }
  | LABELSTR {
    *O << *$1 << '\n';
    delete $1;
    $$ = 0;
  };

Unwind : UNWIND | EXCEPT { $$ = $1; *$$ = "unwind"; }

BBTerminatorInst : RET ResolvedVal {              // Return with a result...
    *O << "    " << *$1 << ' ' << *$2.val << '\n';
    delete $1; $2.destroy();
    $$ = 0;
  }
  | RET VOID {                                       // Return with no result...
    *O << "    " << *$1 << ' ' << $2->getNewTy() << '\n';
    delete $1;
    $$ = 0;
  }
  | BR LABEL ValueRef {                         // Unconditional Branch...
    *O << "    " << *$1 << ' ' << $2->getNewTy() << ' ' << *$3.val << '\n';
    delete $1; $3.destroy();
    $$ = 0;
  }                                                  // Conditional Branch...
  | BR BOOL ValueRef ',' LABEL ValueRef ',' LABEL ValueRef {  
    std::string Name = getUniqueName($3.val, $2);
    *O << "    " << *$1 << ' ' << $2->getNewTy() << ' ' << Name << ", " 
       << $5->getNewTy() << ' ' << *$6.val << ", " << $8->getNewTy() << ' ' 
       << *$9.val << '\n';
    delete $1; $3.destroy(); $6.destroy(); $9.destroy();
    $$ = 0;
  }
  | SWITCH IntType ValueRef ',' LABEL ValueRef '[' JumpTable ']' {
    std::string Name = getUniqueName($3.val, $2);
    *O << "    " << *$1 << ' ' << $2->getNewTy() << ' ' << Name << ", " 
       << $5->getNewTy() << ' ' << *$6.val << " [" << *$8 << " ]\n";
    delete $1; $3.destroy(); $6.destroy(); 
    delete $8;
    $$ = 0;
  }
  | SWITCH IntType ValueRef ',' LABEL ValueRef '[' ']' {
    std::string Name = getUniqueName($3.val, $2);
    *O << "    " << *$1 << ' ' << $2->getNewTy() << ' ' << Name << ", " 
       << $5->getNewTy() << ' ' << *$6.val << "[]\n";
    delete $1; $3.destroy(); $6.destroy();
    $$ = 0;
  }
  | OptAssign INVOKE OptCallingConv TypesV ValueRef '(' ValueRefListE ')'
    TO LABEL ValueRef Unwind LABEL ValueRef {
    const TypeInfo* ResTy = getFunctionReturnType($4);
    *O << "    ";
    if (!$1->empty()) {
      std::string Name = getUniqueName($1, ResTy);
      *O << Name << " = ";
    }
    *O << *$2 << ' ' << *$3 << ' ' << $4->getNewTy() << ' ' << *$5.val << " (";
    for (unsigned i = 0; i < $7->size(); ++i) {
      ValueInfo& VI = (*$7)[i];
      *O << *VI.val;
      if (i+1 < $7->size())
        *O << ", ";
      VI.destroy();
    }
    *O << ") " << *$9 << ' ' << $10->getNewTy() << ' ' << *$11.val << ' ' 
       << *$12 << ' ' << $13->getNewTy() << ' ' << *$14.val << '\n';
    delete $1; delete $2; delete $3; $5.destroy(); delete $7; 
    delete $9; $11.destroy(); delete $12; $14.destroy(); 
    $$ = 0;
  }
  | Unwind {
    *O << "    " << *$1 << '\n';
    delete $1;
    $$ = 0;
  }
  | UNREACHABLE {
    *O << "    " << *$1 << '\n';
    delete $1;
    $$ = 0;
  };

JumpTable : JumpTable IntType ConstValueRef ',' LABEL ValueRef {
    *$1 += " " + $2->getNewTy() + " " + *$3 + ", " + $5->getNewTy() + " " + 
           *$6.val;
    delete $3; $6.destroy();
    $$ = $1;
  }
  | IntType ConstValueRef ',' LABEL ValueRef {
    $2->insert(0, $1->getNewTy() + " " );
    *$2 += ", " + $4->getNewTy() + " " + *$5.val;
    $5.destroy();
    $$ = $2;
  };

Inst 
  : OptAssign InstVal {
    if (!$1->empty()) {
      // Get a unique name for this value, based on its type.
      std::string Name = getUniqueName($1, $2.type);
      *$1 = Name + " = ";
      if (deleteUselessCastFlag && *deleteUselessCastName == Name) {
        // don't actually delete it, just comment it out
        $1->insert(0, "; USELSS BITCAST: "); 
        delete deleteUselessCastName;
      }
    }
    *$1 += *$2.val;
    $2.destroy();
    deleteUselessCastFlag = false;
    $$ = $1; 
  };

PHIList 
  : Types '[' ValueRef ',' ValueRef ']' {    // Used for PHI nodes
    std::string Name = getUniqueName($3.val, $1);
    Name.insert(0, $1->getNewTy() + "[");
    Name += "," + *$5.val + "]";
    $$.val = new std::string(Name);
    $$.type = $1;
    $3.destroy(); $5.destroy();
  }
  | PHIList ',' '[' ValueRef ',' ValueRef ']' {
    std::string Name = getUniqueName($4.val, $1.type);
    *$1.val += ", [" + Name + "," + *$6.val + "]";
    $4.destroy(); $6.destroy();
    $$ = $1;
  };


ValueRefList 
  : ResolvedVal {
    $$ = new ValueList();
    $$->push_back($1);
  }
  | ValueRefList ',' ResolvedVal {
    $$ = $1;
    $$->push_back($3);
  };

// ValueRefListE - Just like ValueRefList, except that it may also be empty!
ValueRefListE 
  : ValueRefList  { $$ = $1; }
  | /*empty*/ { $$ = new ValueList(); }
  ;

OptTailCall 
  : TAIL CALL {
    *$1 += " " + *$2;
    delete $2;
    $$ = $1;
  }
  | CALL 
  ;

InstVal : ArithmeticOps Types ValueRef ',' ValueRef {
    const char* op = getDivRemOpcode(*$1, $2); 
    std::string Name1 = getUniqueName($3.val, $2);
    std::string Name2 = getUniqueName($5.val, $2);
    $$.val = new std::string(op);
    *$$.val += " " + $2->getNewTy() + " " + Name1 + ", " + Name2;
    $$.type = $2;
    delete $1; $3.destroy(); $5.destroy();
  }
  | LogicalOps Types ValueRef ',' ValueRef {
    std::string Name1 = getUniqueName($3.val, $2);
    std::string Name2 = getUniqueName($5.val, $2);
    *$1 += " " + $2->getNewTy() + " " + Name1 + ", " + Name2;
    $$.val = $1;
    $$.type = $2;
    $3.destroy(); $5.destroy();
  }
  | SetCondOps Types ValueRef ',' ValueRef {
    std::string Name1 = getUniqueName($3.val, $2);
    std::string Name2 = getUniqueName($5.val, $2);
    *$1 = getCompareOp(*$1, $2);
    *$1 += " " + $2->getNewTy() + " " + Name1 + ", " + Name2;
    $$.val = $1;
    $$.type = TypeInfo::get("bool",BoolTy);
    $3.destroy(); $5.destroy();
  }
  | ICMP IPredicates Types ValueRef ',' ValueRef {
    std::string Name1 = getUniqueName($4.val, $3);
    std::string Name2 = getUniqueName($6.val, $3);
    *$1 += " " + *$2 + " " + $3->getNewTy() + " " + Name1 + "," + Name2;
    $$.val = $1;
    $$.type = TypeInfo::get("bool",BoolTy);
    delete $2; $4.destroy(); $6.destroy();
  }
  | FCMP FPredicates Types ValueRef ',' ValueRef {
    std::string Name1 = getUniqueName($4.val, $3);
    std::string Name2 = getUniqueName($6.val, $3);
    *$1 += " " + *$2 + " " + $3->getNewTy() + " " + Name1 + "," + Name2;
    $$.val = $1;
    $$.type = TypeInfo::get("bool",BoolTy);
    delete $2; $4.destroy(); $6.destroy();
  }
  | ShiftOps ResolvedVal ',' ResolvedVal {
    const char* shiftop = $1->c_str();
    if (*$1 == "shr")
      shiftop = ($2.type->isUnsigned()) ? "lshr" : "ashr";
    $$.val = new std::string(shiftop);
    *$$.val += " " + *$2.val + ", " + *$4.val;
    $$.type = $2.type;
    delete $1; $2.destroy(); $4.destroy();
  }
  | CastOps ResolvedVal TO Types {
    std::string source = *$2.val;
    const TypeInfo* SrcTy = $2.type->resolve();
    const TypeInfo* DstTy = $4->resolve();
    $$.val = new std::string();
    $$.type = DstTy;
    if (*$1 == "cast") {
      *$$.val += getCastUpgrade(source, SrcTy, DstTy, false);
    } else {
      *$$.val += *$1 + " " + source + " to " + DstTy->getNewTy();
    }
    // Check to see if this is a useless cast of a value to the same name
    // and the same type. Such casts will probably cause redefinition errors
    // when assembled and perform no code gen action so just remove them.
    if (*$1 == "cast" || *$1 == "bitcast")
      if (SrcTy->isInteger() && DstTy->isInteger() &&
          SrcTy->getBitWidth() == DstTy->getBitWidth()) {
        deleteUselessCastFlag = true; // Flag the "Inst" rule
        deleteUselessCastName = new std::string(*$2.val); // save the name
        size_t pos = deleteUselessCastName->find_first_of("%\"",0);
        if (pos != std::string::npos) {
          // remove the type portion before val
          deleteUselessCastName->erase(0, pos);
        }
      }
    delete $1; $2.destroy();
    delete $3;
  }
  | SELECT ResolvedVal ',' ResolvedVal ',' ResolvedVal {
    *$1 += " " + *$2.val + ", " + *$4.val + ", " + *$6.val;
    $$.val = $1;
    $$.type = $4.type;
    $2.destroy(); $4.destroy(); $6.destroy();
  }
  | VAARG ResolvedVal ',' Types {
    *$1 += " " + *$2.val + ", " + $4->getNewTy();
    $$.val = $1;
    $$.type = $4;
    $2.destroy();
  }
  | EXTRACTELEMENT ResolvedVal ',' ResolvedVal {
    *$1 += " " + *$2.val + ", " + *$4.val;
    $$.val = $1;
    $2.type = $2.type->resolve();;
    $$.type = $2.type->getElementType();
    $2.destroy(); $4.destroy();
  }
  | INSERTELEMENT ResolvedVal ',' ResolvedVal ',' ResolvedVal {
    *$1 += " " + *$2.val + ", " + *$4.val + ", " + *$6.val;
    $$.val = $1;
    $$.type = $2.type;
    $2.destroy(); $4.destroy(); $6.destroy();
  }
  | SHUFFLEVECTOR ResolvedVal ',' ResolvedVal ',' ResolvedVal {
    *$1 += " " + *$2.val + ", " + *$4.val + ", " + *$6.val;
    $$.val = $1;
    $$.type = $2.type;
    $2.destroy(); $4.destroy(); $6.destroy();
  }
  | PHI_TOK PHIList {
    *$1 += " " + *$2.val;
    $$.val = $1;
    $$.type = $2.type;
    delete $2.val;
  }
  | OptTailCall OptCallingConv TypesV ValueRef '(' ValueRefListE ')'  {
    // map llvm.isunordered to "fcmp uno" 
    if (*$4.val == "%llvm.isunordered.f32" ||
        *$4.val == "%llvm.isunordered.f64") {
      $$.val = new std::string( "fcmp uno " + *(*$6)[0].val + ", ");
      size_t pos = (*$6)[1].val->find(' ');
      assert(pos != std::string::npos && "no space?");
      *$$.val += (*$6)[1].val->substr(pos+1);
      $$.type = TypeInfo::get("bool", BoolTy);
    } else {
      if (!$2->empty())
        *$1 += " " + *$2;
      if (!$1->empty())
        *$1 += " ";
      *$1 += $3->getNewTy() + " " + *$4.val + "(";
      for (unsigned i = 0; i < $6->size(); ++i) {
        ValueInfo& VI = (*$6)[i];
        *$1 += *VI.val;
        if (i+1 < $6->size())
          *$1 += ", ";
        VI.destroy();
      }
      *$1 += ")";
      $$.val = $1;
      $$.type = getFunctionReturnType($3);
    }
    delete $2; $4.destroy(); delete $6;
  }
  | MemoryInst ;


// IndexList - List of indices for GEP based instructions...
IndexList 
  : ',' ValueRefList { $$ = $2; }
  | /* empty */ {  $$ = new ValueList(); }
  ;

OptVolatile 
  : VOLATILE 
  | /* empty */ { $$ = new std::string(); }
  ;

MemoryInst : MALLOC Types OptCAlign {
    *$1 += " " + $2->getNewTy();
    if (!$3->empty())
      *$1 += " " + *$3;
    $$.val = $1;
    $$.type = $2->getPointerType();
    delete $3;
  }
  | MALLOC Types ',' UINT ValueRef OptCAlign {
    std::string Name = getUniqueName($5.val, $4);
    *$1 += " " + $2->getNewTy() + ", " + $4->getNewTy() + " " + Name;
    if (!$6->empty())
      *$1 += " " + *$6;
    $$.val = $1;
    $$.type = $2->getPointerType();
    $5.destroy(); delete $6;
  }
  | ALLOCA Types OptCAlign {
    *$1 += " " + $2->getNewTy();
    if (!$3->empty())
      *$1 += " " + *$3;
    $$.val = $1;
    $$.type = $2->getPointerType();
    delete $3;
  }
  | ALLOCA Types ',' UINT ValueRef OptCAlign {
    std::string Name = getUniqueName($5.val, $4);
    *$1 += " " + $2->getNewTy() + ", " + $4->getNewTy() + " " + Name;
    if (!$6->empty())
      *$1 += " " + *$6;
    $$.val = $1;
    $$.type = $2->getPointerType();
    $5.destroy(); delete $6;
  }
  | FREE ResolvedVal {
    *$1 += " " + *$2.val;
    $$.val = $1;
    $$.type = TypeInfo::get("void", VoidTy); 
    $2.destroy();
  }
  | OptVolatile LOAD Types ValueRef {
    std::string Name = getUniqueName($4.val, $3);
    if (!$1->empty())
      *$1 += " ";
    *$1 += *$2 + " " + $3->getNewTy() + " " + Name;
    $$.val = $1;
    $$.type = $3->getElementType();
    delete $2; $4.destroy();
  }
  | OptVolatile STORE ResolvedVal ',' Types ValueRef {
    std::string Name = getUniqueName($6.val, $5);
    if (!$1->empty())
      *$1 += " ";
    *$1 += *$2 + " " + *$3.val + ", " + $5->getNewTy() + " " + Name;
    $$.val = $1;
    $$.type = TypeInfo::get("void", VoidTy);
    delete $2; $3.destroy(); $6.destroy();
  }
  | GETELEMENTPTR Types ValueRef IndexList {
    std::string Name = getUniqueName($3.val, $2);
    // Upgrade the indices
    for (unsigned i = 0; i < $4->size(); ++i) {
      ValueInfo& VI = (*$4)[i];
      if (VI.type->isUnsigned() && !VI.isConstant() && 
          VI.type->getBitWidth() < 64) {
        *O << "    %gep_upgrade" << unique << " = zext " << *VI.val 
           << " to i64\n";
        *VI.val = "i64 %gep_upgrade" + llvm::utostr(unique++);
        VI.type = TypeInfo::get("i64",ULongTy);
      }
    }
    *$1 += " " + $2->getNewTy() + " " + Name;
    for (unsigned i = 0; i < $4->size(); ++i) {
      ValueInfo& VI = (*$4)[i];
      *$1 += ", " + *VI.val;
    }
    $$.val = $1;
    $$.type = getGEPIndexedType($2,$4); 
    $3.destroy(); delete $4;
  };

%%

int yyerror(const char *ErrorMsg) {
  std::string where 
    = std::string((CurFilename == "-") ? std::string("<stdin>") : CurFilename)
                  + ":" + llvm::utostr((unsigned) Upgradelineno) + ": ";
  std::string errMsg = where + "error: " + std::string(ErrorMsg) + 
                       " while reading ";
  if (yychar == YYEMPTY || yychar == 0)
    errMsg += "end-of-file.";
  else
    errMsg += "token: '" + std::string(Upgradetext, Upgradeleng) + "'";
  std::cerr << "llvm-upgrade: " << errMsg << '\n';
  *O << "llvm-upgrade parse failed.\n";
  exit(1);
}

static void warning(const std::string& ErrorMsg) {
  std::string where 
    = std::string((CurFilename == "-") ? std::string("<stdin>") : CurFilename)
                  + ":" + llvm::utostr((unsigned) Upgradelineno) + ": ";
  std::string errMsg = where + "warning: " + std::string(ErrorMsg) + 
                       " while reading ";
  if (yychar == YYEMPTY || yychar == 0)
    errMsg += "end-of-file.";
  else
    errMsg += "token: '" + std::string(Upgradetext, Upgradeleng) + "'";
  std::cerr << "llvm-upgrade: " << errMsg << '\n';
}
