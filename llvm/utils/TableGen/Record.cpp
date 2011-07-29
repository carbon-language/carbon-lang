//===- Record.cpp - Record implementation ---------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Implement the tablegen record classes.
//
//===----------------------------------------------------------------------===//

#include "Record.h"
#include "Error.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/Format.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"

using namespace llvm;

//===----------------------------------------------------------------------===//
//    std::string wrapper for DenseMap purposes
//===----------------------------------------------------------------------===//

/// TableGenStringKey - This is a wrapper for std::string suitable for
/// using as a key to a DenseMap.  Because there isn't a particularly
/// good way to indicate tombstone or empty keys for strings, we want
/// to wrap std::string to indicate that this is a "special" string
/// not expected to take on certain values (those of the tombstone and
/// empty keys).  This makes things a little safer as it clarifies
/// that DenseMap is really not appropriate for general strings.

class TableGenStringKey {
public:
  TableGenStringKey(const std::string &str) : data(str) {}
  TableGenStringKey(const char *str) : data(str) {}

  const std::string &str() const { return data; }
  
private:
  std::string data;
};

/// Specialize DenseMapInfo for TableGenStringKey.
namespace llvm {

template<> struct DenseMapInfo<TableGenStringKey> {
  static inline TableGenStringKey getEmptyKey() {
    TableGenStringKey Empty("<<<EMPTY KEY>>>");
    return Empty;
  }
  static inline TableGenStringKey getTombstoneKey() {
    TableGenStringKey Tombstone("<<<TOMBSTONE KEY>>>");
    return Tombstone;
  }
  static unsigned getHashValue(const TableGenStringKey& Val) {
    return HashString(Val.str());
  }
  static bool isEqual(const TableGenStringKey& LHS,
                      const TableGenStringKey& RHS) {
    return LHS.str() == RHS.str();
  }
};

}

//===----------------------------------------------------------------------===//
//    Type implementations
//===----------------------------------------------------------------------===//

BitRecTy BitRecTy::Shared;
IntRecTy IntRecTy::Shared;
StringRecTy StringRecTy::Shared;
CodeRecTy CodeRecTy::Shared;
DagRecTy DagRecTy::Shared;

void RecTy::dump() const { print(errs()); }

ListRecTy *RecTy::getListTy() {
  if (!ListTy)
    ListTy = new ListRecTy(this);
  return ListTy;
}

const Init *BitRecTy::convertValue(const BitsInit *BI) {
  if (BI->getNumBits() != 1) return 0; // Only accept if just one bit!
  return BI->getBit(0);
}

bool BitRecTy::baseClassOf(const BitsRecTy *RHS) const {
  return RHS->getNumBits() == 1;
}

const Init *BitRecTy::convertValue(const IntInit *II) {
  int64_t Val = II->getValue();
  if (Val != 0 && Val != 1) return 0;  // Only accept 0 or 1 for a bit!

  return new BitInit(Val != 0);
}

const Init *BitRecTy::convertValue(const TypedInit *VI) {
  if (dynamic_cast<BitRecTy*>(VI->getType()))
    return VI;  // Accept variable if it is already of bit type!
  return 0;
}

BitsRecTy *BitsRecTy::get(unsigned Sz) {
  static std::vector<BitsRecTy*> Shared;
  if (Sz >= Shared.size())
    Shared.resize(Sz + 1);
  BitsRecTy *&Ty = Shared[Sz];
  if (!Ty)
    Ty = new BitsRecTy(Sz);
  return Ty;
}

std::string BitsRecTy::getAsString() const {
  return "bits<" + utostr(Size) + ">";
}

const Init *BitsRecTy::convertValue(const UnsetInit *UI) {
  SmallVector<const Init *, 16> NewBits(Size);

  for (unsigned i = 0; i != Size; ++i)
    NewBits[i] = new UnsetInit();

  return new BitsInit(ArrayRef<const Init *>(NewBits));
}

const Init *BitsRecTy::convertValue(const BitInit *UI) {
  if (Size != 1) return 0;  // Can only convert single bit.
  return new BitsInit(ArrayRef<const Init *>(UI));
}

/// canFitInBitfield - Return true if the number of bits is large enough to hold
/// the integer value.
static bool canFitInBitfield(int64_t Value, unsigned NumBits) {
  // For example, with NumBits == 4, we permit Values from [-7 .. 15].
  return (NumBits >= sizeof(Value) * 8) ||
         (Value >> NumBits == 0) || (Value >> (NumBits-1) == -1);
}

/// convertValue from Int initializer to bits type: Split the integer up into the
/// appropriate bits.
///
const Init *BitsRecTy::convertValue(const IntInit *II) {
  int64_t Value = II->getValue();
  // Make sure this bitfield is large enough to hold the integer value.
  if (!canFitInBitfield(Value, Size))
    return 0;

  SmallVector<const Init *, 16> NewBits(Size);

  for (unsigned i = 0; i != Size; ++i)
    NewBits[i] = new BitInit(Value & (1LL << i));

  return new BitsInit(ArrayRef<const Init *>(NewBits));
}

const Init *BitsRecTy::convertValue(const BitsInit *BI) {
  // If the number of bits is right, return it.  Otherwise we need to expand or
  // truncate.
  if (BI->getNumBits() == Size) return BI;
  return 0;
}

const Init *BitsRecTy::convertValue(const TypedInit *VI) {
  if (BitsRecTy *BRT = dynamic_cast<BitsRecTy*>(VI->getType()))
    if (BRT->Size == Size) {
      SmallVector<const Init *, 16> NewBits(Size);
 
      for (unsigned i = 0; i != Size; ++i)
        NewBits[i] = new VarBitInit(VI, i);
      return new BitsInit(ArrayRef<const Init *>(NewBits));
    }

  if (Size == 1 && dynamic_cast<BitRecTy*>(VI->getType()))
    return new BitsInit(ArrayRef<const Init *>(VI));

  if (const TernOpInit *Tern = dynamic_cast<const TernOpInit*>(VI)) {
    if (Tern->getOpcode() == TernOpInit::IF) {
      const Init *LHS = Tern->getLHS();
      const Init *MHS = Tern->getMHS();
      const Init *RHS = Tern->getRHS();

      const IntInit *MHSi = dynamic_cast<const IntInit*>(MHS);
      const IntInit *RHSi = dynamic_cast<const IntInit*>(RHS);

      if (MHSi && RHSi) {
        int64_t MHSVal = MHSi->getValue();
        int64_t RHSVal = RHSi->getValue();

        if (canFitInBitfield(MHSVal, Size) && canFitInBitfield(RHSVal, Size)) {
          SmallVector<const Init *, 16> NewBits(Size);

          for (unsigned i = 0; i != Size; ++i)
            NewBits[i] =
              new TernOpInit(TernOpInit::IF, LHS,
                             new IntInit((MHSVal & (1LL << i)) ? 1 : 0),
                             new IntInit((RHSVal & (1LL << i)) ? 1 : 0),
                             VI->getType());

          return new BitsInit(ArrayRef<const Init *>(NewBits));
        }
      } else {
        const BitsInit *MHSbs = dynamic_cast<const BitsInit*>(MHS);
        const BitsInit *RHSbs = dynamic_cast<const BitsInit*>(RHS);

        if (MHSbs && RHSbs) {
          SmallVector<const Init *, 16> NewBits(Size);

          for (unsigned i = 0; i != Size; ++i)
            NewBits[i] = new TernOpInit(TernOpInit::IF, LHS,
                                        MHSbs->getBit(i),
                                        RHSbs->getBit(i),
                                        VI->getType());

          return new BitsInit(ArrayRef<const Init *>(NewBits));
        }
      }
    }
  }

  return 0;
}

const Init *IntRecTy::convertValue(const BitInit *BI) {
  return new IntInit(BI->getValue());
}

const Init *IntRecTy::convertValue(const BitsInit *BI) {
  int64_t Result = 0;
  for (unsigned i = 0, e = BI->getNumBits(); i != e; ++i)
    if (const BitInit *Bit = dynamic_cast<const BitInit*>(BI->getBit(i))) {
      Result |= Bit->getValue() << i;
    } else {
      return 0;
    }
  return new IntInit(Result);
}

const Init *IntRecTy::convertValue(const TypedInit *TI) {
  if (TI->getType()->typeIsConvertibleTo(this))
    return TI;  // Accept variable if already of the right type!
  return 0;
}

const Init *StringRecTy::convertValue(const UnOpInit *BO) {
  if (BO->getOpcode() == UnOpInit::CAST) {
    const Init *L = BO->getOperand()->convertInitializerTo(this);
    if (L == 0) return 0;
    if (L != BO->getOperand())
      return new UnOpInit(UnOpInit::CAST, L, new StringRecTy);
    return BO;
  }

  return convertValue((const TypedInit*)BO);
}

const Init *StringRecTy::convertValue(const BinOpInit *BO) {
  if (BO->getOpcode() == BinOpInit::STRCONCAT) {
    const Init *L = BO->getLHS()->convertInitializerTo(this);
    const Init *R = BO->getRHS()->convertInitializerTo(this);
    if (L == 0 || R == 0) return 0;
    if (L != BO->getLHS() || R != BO->getRHS())
      return new BinOpInit(BinOpInit::STRCONCAT, L, R, new StringRecTy);
    return BO;
  }

  return convertValue((const TypedInit*)BO);
}


const Init *StringRecTy::convertValue(const TypedInit *TI) {
  if (dynamic_cast<StringRecTy*>(TI->getType()))
    return TI;  // Accept variable if already of the right type!
  return 0;
}

std::string ListRecTy::getAsString() const {
  return "list<" + Ty->getAsString() + ">";
}

const Init *ListRecTy::convertValue(const ListInit *LI) {
  std::vector<const Init*> Elements;

  // Verify that all of the elements of the list are subclasses of the
  // appropriate class!
  for (unsigned i = 0, e = LI->getSize(); i != e; ++i)
    if (const Init *CI = LI->getElement(i)->convertInitializerTo(Ty))
      Elements.push_back(CI);
    else
      return 0;

  ListRecTy *LType = dynamic_cast<ListRecTy*>(LI->getType());
  if (LType == 0) {
    return 0;
  }

  return new ListInit(Elements, this);
}

const Init *ListRecTy::convertValue(const TypedInit *TI) {
  // Ensure that TI is compatible with our class.
  if (ListRecTy *LRT = dynamic_cast<ListRecTy*>(TI->getType()))
    if (LRT->getElementType()->typeIsConvertibleTo(getElementType()))
      return TI;
  return 0;
}

const Init *CodeRecTy::convertValue(const TypedInit *TI) {
  if (TI->getType()->typeIsConvertibleTo(this))
    return TI;
  return 0;
}

const Init *DagRecTy::convertValue(const TypedInit *TI) {
  if (TI->getType()->typeIsConvertibleTo(this))
    return TI;
  return 0;
}

const Init *DagRecTy::convertValue(const UnOpInit *BO) {
  if (BO->getOpcode() == UnOpInit::CAST) {
    const Init *L = BO->getOperand()->convertInitializerTo(this);
    if (L == 0) return 0;
    if (L != BO->getOperand())
      return new UnOpInit(UnOpInit::CAST, L, new DagRecTy);
    return BO;
  }
  return 0;
}

const Init *DagRecTy::convertValue(const BinOpInit *BO) {
  if (BO->getOpcode() == BinOpInit::CONCAT) {
    const Init *L = BO->getLHS()->convertInitializerTo(this);
    const Init *R = BO->getRHS()->convertInitializerTo(this);
    if (L == 0 || R == 0) return 0;
    if (L != BO->getLHS() || R != BO->getRHS())
      return new BinOpInit(BinOpInit::CONCAT, L, R, new DagRecTy);
    return BO;
  }
  return 0;
}

RecordRecTy *RecordRecTy::get(Record *R) {
  return &dynamic_cast<RecordRecTy&>(*R->getDefInit()->getType());
}

std::string RecordRecTy::getAsString() const {
  return Rec->getName();
}

const Init *RecordRecTy::convertValue(const DefInit *DI) {
  // Ensure that DI is a subclass of Rec.
  if (!DI->getDef()->isSubClassOf(Rec))
    return 0;
  return DI;
}

const Init *RecordRecTy::convertValue(const TypedInit *TI) {
  // Ensure that TI is compatible with Rec.
  if (RecordRecTy *RRT = dynamic_cast<RecordRecTy*>(TI->getType()))
    if (RRT->getRecord()->isSubClassOf(getRecord()) ||
        RRT->getRecord() == getRecord())
      return TI;
  return 0;
}

bool RecordRecTy::baseClassOf(const RecordRecTy *RHS) const {
  if (Rec == RHS->getRecord() || RHS->getRecord()->isSubClassOf(Rec))
    return true;

  const std::vector<Record*> &SC = Rec->getSuperClasses();
  for (unsigned i = 0, e = SC.size(); i != e; ++i)
    if (RHS->getRecord()->isSubClassOf(SC[i]))
      return true;

  return false;
}


/// resolveTypes - Find a common type that T1 and T2 convert to.
/// Return 0 if no such type exists.
///
RecTy *llvm::resolveTypes(RecTy *T1, RecTy *T2) {
  if (!T1->typeIsConvertibleTo(T2)) {
    if (!T2->typeIsConvertibleTo(T1)) {
      // If one is a Record type, check superclasses
      RecordRecTy *RecTy1 = dynamic_cast<RecordRecTy*>(T1);
      if (RecTy1) {
        // See if T2 inherits from a type T1 also inherits from
        const std::vector<Record *> &T1SuperClasses =
          RecTy1->getRecord()->getSuperClasses();
        for(std::vector<Record *>::const_iterator i = T1SuperClasses.begin(),
              iend = T1SuperClasses.end();
            i != iend;
            ++i) {
          RecordRecTy *SuperRecTy1 = RecordRecTy::get(*i);
          RecTy *NewType1 = resolveTypes(SuperRecTy1, T2);
          if (NewType1 != 0) {
            if (NewType1 != SuperRecTy1) {
              delete SuperRecTy1;
            }
            return NewType1;
          }
        }
      }
      RecordRecTy *RecTy2 = dynamic_cast<RecordRecTy*>(T2);
      if (RecTy2) {
        // See if T1 inherits from a type T2 also inherits from
        const std::vector<Record *> &T2SuperClasses =
          RecTy2->getRecord()->getSuperClasses();
        for (std::vector<Record *>::const_iterator i = T2SuperClasses.begin(),
              iend = T2SuperClasses.end();
            i != iend;
            ++i) {
          RecordRecTy *SuperRecTy2 = RecordRecTy::get(*i);
          RecTy *NewType2 = resolveTypes(T1, SuperRecTy2);
          if (NewType2 != 0) {
            if (NewType2 != SuperRecTy2) {
              delete SuperRecTy2;
            }
            return NewType2;
          }
        }
      }
      return 0;
    }
    return T2;
  }
  return T1;
}


//===----------------------------------------------------------------------===//
//    Initializer implementations
//===----------------------------------------------------------------------===//

void Init::dump() const { return print(errs()); }

const Init *
BitsInit::convertInitializerBitRange(const std::vector<unsigned> &Bits) const {
  SmallVector<const Init *, 16> NewBits(Bits.size());

  for (unsigned i = 0, e = Bits.size(); i != e; ++i) {
    if (Bits[i] >= getNumBits())
      return 0;
    NewBits[i] = getBit(Bits[i]);
  }
  return new BitsInit(ArrayRef<const Init *>(NewBits));
}

std::string BitsInit::getAsString() const {
  std::string Result = "{ ";
  for (unsigned i = 0, e = getNumBits(); i != e; ++i) {
    if (i) Result += ", ";
    if (const Init *Bit = getBit(e-i-1))
      Result += Bit->getAsString();
    else
      Result += "*";
  }
  return Result + " }";
}

// resolveReferences - If there are any field references that refer to fields
// that have been filled in, we can propagate the values now.
//
const Init *BitsInit::resolveReferences(Record &R, const RecordVal *RV) const {
  bool Changed = false;
  SmallVector<const Init *, 16> NewBits(getNumBits());

  for (unsigned i = 0, e = Bits.size(); i != e; ++i) {
    const Init *B;
    const Init *CurBit = getBit(i);

    do {
      B = CurBit;
      CurBit = CurBit->resolveReferences(R, RV);
      Changed |= B != CurBit;
    } while (B != CurBit);
    NewBits[i] = CurBit;
  }

  if (Changed)
    return new BitsInit(ArrayRef<const Init *>(NewBits));

  return this;
}

std::string IntInit::getAsString() const {
  return itostr(Value);
}

const Init *
IntInit::convertInitializerBitRange(const std::vector<unsigned> &Bits) const {
  SmallVector<const Init *, 16> NewBits(Bits.size());

  for (unsigned i = 0, e = Bits.size(); i != e; ++i) {
    if (Bits[i] >= 64)
      return 0;

    NewBits[i] = new BitInit(Value & (INT64_C(1) << Bits[i]));
  }
  return new BitsInit(ArrayRef<const Init *>(NewBits));
}

const Init *
ListInit::convertInitListSlice(const std::vector<unsigned> &Elements) const {
  std::vector<const Init*> Vals;
  for (unsigned i = 0, e = Elements.size(); i != e; ++i) {
    if (Elements[i] >= getSize())
      return 0;
    Vals.push_back(getElement(Elements[i]));
  }
  return new ListInit(Vals, getType());
}

Record *ListInit::getElementAsRecord(unsigned i) const {
  assert(i < Values.size() && "List element index out of range!");
  const DefInit *DI = dynamic_cast<const DefInit*>(Values[i]);
  if (DI == 0) throw "Expected record in list!";
  return DI->getDef();
}

const Init *ListInit::resolveReferences(Record &R, const RecordVal *RV) const {
  std::vector<const Init*> Resolved;
  Resolved.reserve(getSize());
  bool Changed = false;

  for (unsigned i = 0, e = getSize(); i != e; ++i) {
    const Init *E;
    const Init *CurElt = getElement(i);

    do {
      E = CurElt;
      CurElt = CurElt->resolveReferences(R, RV);
      Changed |= E != CurElt;
    } while (E != CurElt);
    Resolved.push_back(E);
  }

  if (Changed)
    return new ListInit(Resolved, getType());
  return this;
}

const Init *ListInit::resolveListElementReference(Record &R,
                                                  const RecordVal *IRV,
                                                  unsigned Elt) const {
  if (Elt >= getSize())
    return 0;  // Out of range reference.
  const Init *E = getElement(Elt);
  // If the element is set to some value, or if we are resolving a reference
  // to a specific variable and that variable is explicitly unset, then
  // replace the VarListElementInit with it.
  if (IRV || !dynamic_cast<const UnsetInit*>(E))
    return E;
  return 0;
}

std::string ListInit::getAsString() const {
  std::string Result = "[";
  for (unsigned i = 0, e = Values.size(); i != e; ++i) {
    if (i) Result += ", ";
    Result += Values[i]->getAsString();
  }
  return Result + "]";
}

const Init *OpInit::resolveBitReference(Record &R, const RecordVal *IRV,
                                        unsigned Bit) const {
  const Init *Folded = Fold(&R, 0);

  if (Folded != this) {
    const TypedInit *Typed = dynamic_cast<const TypedInit *>(Folded);
    if (Typed) {
      return Typed->resolveBitReference(R, IRV, Bit);
    }
  }

  return 0;
}

const Init *OpInit::resolveListElementReference(Record &R, const RecordVal *IRV,
                                                unsigned Elt) const {
  const Init *Folded = Fold(&R, 0);

  if (Folded != this) {
    const TypedInit *Typed = dynamic_cast<const TypedInit *>(Folded);
    if (Typed) {
      return Typed->resolveListElementReference(R, IRV, Elt);
    }
  }

  return 0;
}

const Init *UnOpInit::Fold(Record *CurRec, MultiClass *CurMultiClass) const {
  switch (getOpcode()) {
  default: assert(0 && "Unknown unop");
  case CAST: {
    if (getType()->getAsString() == "string") {
      const StringInit *LHSs = dynamic_cast<const StringInit*>(LHS);
      if (LHSs) {
        return LHSs;
      }

      const DefInit *LHSd = dynamic_cast<const DefInit*>(LHS);
      if (LHSd) {
        return new StringInit(LHSd->getDef()->getName());
      }
    } else {
      const StringInit *LHSs = dynamic_cast<const StringInit*>(LHS);
      if (LHSs) {
        std::string Name = LHSs->getValue();

        // From TGParser::ParseIDValue
        if (CurRec) {
          if (const RecordVal *RV = CurRec->getValue(Name)) {
            if (RV->getType() != getType())
              throw "type mismatch in cast";
            return new VarInit(Name, RV->getType());
          }

          std::string TemplateArgName = CurRec->getName()+":"+Name;
          if (CurRec->isTemplateArg(TemplateArgName)) {
            const RecordVal *RV = CurRec->getValue(TemplateArgName);
            assert(RV && "Template arg doesn't exist??");

            if (RV->getType() != getType())
              throw "type mismatch in cast";

            return new VarInit(TemplateArgName, RV->getType());
          }
        }

        if (CurMultiClass) {
          std::string MCName = CurMultiClass->Rec.getName()+"::"+Name;
          if (CurMultiClass->Rec.isTemplateArg(MCName)) {
            const RecordVal *RV = CurMultiClass->Rec.getValue(MCName);
            assert(RV && "Template arg doesn't exist??");

            if (RV->getType() != getType())
              throw "type mismatch in cast";

            return new VarInit(MCName, RV->getType());
          }
        }

        if (Record *D = (CurRec->getRecords()).getDef(Name))
          return DefInit::get(D);

        throw TGError(CurRec->getLoc(), "Undefined reference:'" + Name + "'\n");
      }
    }
    break;
  }
  case HEAD: {
    const ListInit *LHSl = dynamic_cast<const ListInit*>(LHS);
    if (LHSl) {
      if (LHSl->getSize() == 0) {
        assert(0 && "Empty list in car");
        return 0;
      }
      return LHSl->getElement(0);
    }
    break;
  }
  case TAIL: {
    const ListInit *LHSl = dynamic_cast<const ListInit*>(LHS);
    if (LHSl) {
      if (LHSl->getSize() == 0) {
        assert(0 && "Empty list in cdr");
        return 0;
      }
      ListInit::const_iterator begin = LHSl->begin()+1;
      ListInit::const_iterator end   = LHSl->end();
      // We can't pass these iterators directly to ArrayRef because
      // they are not convertible to Init **.  Fortunately,
      // RandomAccessIterator::operator * is guaranteed to return an
      // lvalue.
      ListInit *Result =
        new ListInit(ArrayRef<const Init *>(&*begin, end - begin),
                     LHSl->getType());
      return Result;
    }
    break;
  }
  case EMPTY: {
    const ListInit *LHSl = dynamic_cast<const ListInit*>(LHS);
    if (LHSl) {
      if (LHSl->getSize() == 0) {
        return new IntInit(1);
      } else {
        return new IntInit(0);
      }
    }
    const StringInit *LHSs = dynamic_cast<const StringInit*>(LHS);
    if (LHSs) {
      if (LHSs->getValue().empty()) {
        return new IntInit(1);
      } else {
        return new IntInit(0);
      }
    }

    break;
  }
  }
  return this;
}

const Init *UnOpInit::resolveReferences(Record &R, const RecordVal *RV) const {
  const Init *lhs = LHS->resolveReferences(R, RV);

  if (LHS != lhs)
    return (new UnOpInit(getOpcode(), lhs, getType()))->Fold(&R, 0);
  return Fold(&R, 0);
}

std::string UnOpInit::getAsString() const {
  std::string Result;
  switch (Opc) {
  case CAST: Result = "!cast<" + getType()->getAsString() + ">"; break;
  case HEAD: Result = "!head"; break;
  case TAIL: Result = "!tail"; break;
  case EMPTY: Result = "!empty"; break;
  }
  return Result + "(" + LHS->getAsString() + ")";
}

const Init *BinOpInit::Fold(Record *CurRec, MultiClass *CurMultiClass) const {
  switch (getOpcode()) {
  default: assert(0 && "Unknown binop");
  case CONCAT: {
    const DagInit *LHSs = dynamic_cast<const DagInit*>(LHS);
    const DagInit *RHSs = dynamic_cast<const DagInit*>(RHS);
    if (LHSs && RHSs) {
      const DefInit *LOp = dynamic_cast<const DefInit*>(LHSs->getOperator());
      const DefInit *ROp = dynamic_cast<const DefInit*>(RHSs->getOperator());
      if (LOp == 0 || ROp == 0 || LOp->getDef() != ROp->getDef())
        throw "Concated Dag operators do not match!";
      std::vector<const Init*> Args;
      std::vector<std::string> ArgNames;
      for (unsigned i = 0, e = LHSs->getNumArgs(); i != e; ++i) {
        Args.push_back(LHSs->getArg(i));
        ArgNames.push_back(LHSs->getArgName(i));
      }
      for (unsigned i = 0, e = RHSs->getNumArgs(); i != e; ++i) {
        Args.push_back(RHSs->getArg(i));
        ArgNames.push_back(RHSs->getArgName(i));
      }
      return new DagInit(LHSs->getOperator(), "", Args, ArgNames);
    }
    break;
  }
  case STRCONCAT: {
    const StringInit *LHSs = dynamic_cast<const StringInit*>(LHS);
    const StringInit *RHSs = dynamic_cast<const StringInit*>(RHS);
    if (LHSs && RHSs)
      return new StringInit(LHSs->getValue() + RHSs->getValue());
    break;
  }
  case EQ: {
    // try to fold eq comparison for 'bit' and 'int', otherwise fallback
    // to string objects.
    const IntInit* L =
      dynamic_cast<const IntInit*>(LHS->convertInitializerTo(IntRecTy::get()));
    const IntInit* R =
      dynamic_cast<const IntInit*>(RHS->convertInitializerTo(IntRecTy::get()));

    if (L && R)
      return new IntInit(L->getValue() == R->getValue());

    const StringInit *LHSs = dynamic_cast<const StringInit*>(LHS);
    const StringInit *RHSs = dynamic_cast<const StringInit*>(RHS);

    // Make sure we've resolved
    if (LHSs && RHSs)
      return new IntInit(LHSs->getValue() == RHSs->getValue());

    break;
  }
  case SHL:
  case SRA:
  case SRL: {
    const IntInit *LHSi = dynamic_cast<const IntInit*>(LHS);
    const IntInit *RHSi = dynamic_cast<const IntInit*>(RHS);
    if (LHSi && RHSi) {
      int64_t LHSv = LHSi->getValue(), RHSv = RHSi->getValue();
      int64_t Result;
      switch (getOpcode()) {
      default: assert(0 && "Bad opcode!");
      case SHL: Result = LHSv << RHSv; break;
      case SRA: Result = LHSv >> RHSv; break;
      case SRL: Result = (uint64_t)LHSv >> (uint64_t)RHSv; break;
      }
      return new IntInit(Result);
    }
    break;
  }
  }
  return this;
}

const Init *BinOpInit::resolveReferences(Record &R, const RecordVal *RV) const {
  const Init *lhs = LHS->resolveReferences(R, RV);
  const Init *rhs = RHS->resolveReferences(R, RV);

  if (LHS != lhs || RHS != rhs)
    return (new BinOpInit(getOpcode(), lhs, rhs, getType()))->Fold(&R, 0);
  return Fold(&R, 0);
}

std::string BinOpInit::getAsString() const {
  std::string Result;
  switch (Opc) {
  case CONCAT: Result = "!con"; break;
  case SHL: Result = "!shl"; break;
  case SRA: Result = "!sra"; break;
  case SRL: Result = "!srl"; break;
  case EQ: Result = "!eq"; break;
  case STRCONCAT: Result = "!strconcat"; break;
  }
  return Result + "(" + LHS->getAsString() + ", " + RHS->getAsString() + ")";
}

static const Init *ForeachHelper(const Init *LHS, const Init *MHS,
                                 const Init *RHS, RecTy *Type,
                                 Record *CurRec, MultiClass *CurMultiClass);

static const Init *EvaluateOperation(const OpInit *RHSo, const Init *LHS,
                                     const Init *Arg,
                                     RecTy *Type, Record *CurRec,
                                     MultiClass *CurMultiClass) {
  std::vector<const Init *> NewOperands;

  const TypedInit *TArg = dynamic_cast<const TypedInit*>(Arg);

  // If this is a dag, recurse
  if (TArg && TArg->getType()->getAsString() == "dag") {
    const Init *Result = ForeachHelper(LHS, Arg, RHSo, Type,
                                 CurRec, CurMultiClass);
    if (Result != 0) {
      return Result;
    } else {
      return 0;
    }
  }

  for (int i = 0; i < RHSo->getNumOperands(); ++i) {
    const OpInit *RHSoo = dynamic_cast<const OpInit*>(RHSo->getOperand(i));

    if (RHSoo) {
      const Init *Result = EvaluateOperation(RHSoo, LHS, Arg,
                                       Type, CurRec, CurMultiClass);
      if (Result != 0) {
        NewOperands.push_back(Result);
      } else {
        NewOperands.push_back(Arg);
      }
    } else if (LHS->getAsString() == RHSo->getOperand(i)->getAsString()) {
      NewOperands.push_back(Arg);
    } else {
      NewOperands.push_back(RHSo->getOperand(i));
    }
  }

  // Now run the operator and use its result as the new leaf
  const OpInit *NewOp = RHSo->clone(NewOperands);
  const Init *NewVal = NewOp->Fold(CurRec, CurMultiClass);
  if (NewVal != NewOp) {
    delete NewOp;
    return NewVal;
  }
  return 0;
}

static const Init *ForeachHelper(const Init *LHS, const Init *MHS,
                                 const Init *RHS, RecTy *Type,
                                 Record *CurRec,
                                 MultiClass *CurMultiClass) {
  const DagInit *MHSd = dynamic_cast<const DagInit*>(MHS);
  const ListInit *MHSl = dynamic_cast<const ListInit*>(MHS);

  DagRecTy *DagType = dynamic_cast<DagRecTy*>(Type);
  ListRecTy *ListType = dynamic_cast<ListRecTy*>(Type);

  const OpInit *RHSo = dynamic_cast<const OpInit*>(RHS);

  if (!RHSo) {
    throw TGError(CurRec->getLoc(), "!foreach requires an operator\n");
  }

  const TypedInit *LHSt = dynamic_cast<const TypedInit*>(LHS);

  if (!LHSt) {
    throw TGError(CurRec->getLoc(), "!foreach requires typed variable\n");
  }

  if ((MHSd && DagType) || (MHSl && ListType)) {
    if (MHSd) {
      const Init *Val = MHSd->getOperator();
      const Init *Result = EvaluateOperation(RHSo, LHS, Val,
                                       Type, CurRec, CurMultiClass);
      if (Result != 0) {
        Val = Result;
      }

      std::vector<std::pair<const Init *, std::string> > args;
      for (unsigned int i = 0; i < MHSd->getNumArgs(); ++i) {
        const Init *Arg;
        std::string ArgName;
        Arg = MHSd->getArg(i);
        ArgName = MHSd->getArgName(i);

        // Process args
        const Init *Result = EvaluateOperation(RHSo, LHS, Arg, Type,
                                         CurRec, CurMultiClass);
        if (Result != 0) {
          Arg = Result;
        }

        // TODO: Process arg names
        args.push_back(std::make_pair(Arg, ArgName));
      }

      return new DagInit(Val, "", args);
    }
    if (MHSl) {
      std::vector<const Init *> NewOperands;
      std::vector<const Init *> NewList(MHSl->begin(), MHSl->end());

      for (std::vector<const Init *>::iterator li = NewList.begin(),
             liend = NewList.end();
           li != liend;
           ++li) {
        const Init *Item = *li;
        NewOperands.clear();
        for(int i = 0; i < RHSo->getNumOperands(); ++i) {
          // First, replace the foreach variable with the list item
          if (LHS->getAsString() == RHSo->getOperand(i)->getAsString()) {
            NewOperands.push_back(Item);
          } else {
            NewOperands.push_back(RHSo->getOperand(i));
          }
        }

        // Now run the operator and use its result as the new list item
        const OpInit *NewOp = RHSo->clone(NewOperands);
        const Init *NewItem = NewOp->Fold(CurRec, CurMultiClass);
        if (NewItem != NewOp) {
          *li = NewItem;
          delete NewOp;
        }
      }
      return new ListInit(NewList, MHSl->getType());
    }
  }
  return 0;
}

const Init *TernOpInit::Fold(Record *CurRec, MultiClass *CurMultiClass) const {
  switch (getOpcode()) {
  default: assert(0 && "Unknown binop");
  case SUBST: {
    const DefInit *LHSd = dynamic_cast<const DefInit*>(LHS);
    const VarInit *LHSv = dynamic_cast<const VarInit*>(LHS);
    const StringInit *LHSs = dynamic_cast<const StringInit*>(LHS);

    const DefInit *MHSd = dynamic_cast<const DefInit*>(MHS);
    const VarInit *MHSv = dynamic_cast<const VarInit*>(MHS);
    const StringInit *MHSs = dynamic_cast<const StringInit*>(MHS);

    const DefInit *RHSd = dynamic_cast<const DefInit*>(RHS);
    const VarInit *RHSv = dynamic_cast<const VarInit*>(RHS);
    const StringInit *RHSs = dynamic_cast<const StringInit*>(RHS);

    if ((LHSd && MHSd && RHSd)
        || (LHSv && MHSv && RHSv)
        || (LHSs && MHSs && RHSs)) {
      if (RHSd) {
        Record *Val = RHSd->getDef();
        if (LHSd->getAsString() == RHSd->getAsString()) {
          Val = MHSd->getDef();
        }
        return DefInit::get(Val);
      }
      if (RHSv) {
        std::string Val = RHSv->getName();
        if (LHSv->getAsString() == RHSv->getAsString()) {
          Val = MHSv->getName();
        }
        return new VarInit(Val, getType());
      }
      if (RHSs) {
        std::string Val = RHSs->getValue();

        std::string::size_type found;
        std::string::size_type idx = 0;
        do {
          found = Val.find(LHSs->getValue(), idx);
          if (found != std::string::npos) {
            Val.replace(found, LHSs->getValue().size(), MHSs->getValue());
          }
          idx = found +  MHSs->getValue().size();
        } while (found != std::string::npos);

        return new StringInit(Val);
      }
    }
    break;
  }

  case FOREACH: {
    const Init *Result = ForeachHelper(LHS, MHS, RHS, getType(),
                                 CurRec, CurMultiClass);
    if (Result != 0) {
      return Result;
    }
    break;
  }

  case IF: {
    const IntInit *LHSi = dynamic_cast<const IntInit*>(LHS);
    if (const Init *I = LHS->convertInitializerTo(IntRecTy::get()))
      LHSi = dynamic_cast<const IntInit*>(I);
    if (LHSi) {
      if (LHSi->getValue()) {
        return MHS;
      } else {
        return RHS;
      }
    }
    break;
  }
  }

  return this;
}

const Init *TernOpInit::resolveReferences(Record &R,
                                          const RecordVal *RV) const {
  const Init *lhs = LHS->resolveReferences(R, RV);

  if (Opc == IF && lhs != LHS) {
    const IntInit *Value = dynamic_cast<const IntInit*>(lhs);
    if (const Init *I = lhs->convertInitializerTo(IntRecTy::get()))
      Value = dynamic_cast<const IntInit*>(I);
    if (Value != 0) {
      // Short-circuit
      if (Value->getValue()) {
        const Init *mhs = MHS->resolveReferences(R, RV);
        return (new TernOpInit(getOpcode(), lhs, mhs,
                               RHS, getType()))->Fold(&R, 0);
      } else {
        const Init *rhs = RHS->resolveReferences(R, RV);
        return (new TernOpInit(getOpcode(), lhs, MHS,
                               rhs, getType()))->Fold(&R, 0);
      }
    }
  }

  const Init *mhs = MHS->resolveReferences(R, RV);
  const Init *rhs = RHS->resolveReferences(R, RV);

  if (LHS != lhs || MHS != mhs || RHS != rhs)
    return (new TernOpInit(getOpcode(), lhs, mhs, rhs, getType()))->Fold(&R, 0);
  return Fold(&R, 0);
}

std::string TernOpInit::getAsString() const {
  std::string Result;
  switch (Opc) {
  case SUBST: Result = "!subst"; break;
  case FOREACH: Result = "!foreach"; break;
  case IF: Result = "!if"; break;
 }
  return Result + "(" + LHS->getAsString() + ", " + MHS->getAsString() + ", "
    + RHS->getAsString() + ")";
}

RecTy *TypedInit::getFieldType(const std::string &FieldName) const {
  RecordRecTy *RecordType = dynamic_cast<RecordRecTy *>(getType());
  if (RecordType) {
    RecordVal *Field = RecordType->getRecord()->getValue(FieldName);
    if (Field) {
      return Field->getType();
    }
  }
  return 0;
}

const Init *
TypedInit::convertInitializerBitRange(const std::vector<unsigned> &Bits) const {
  BitsRecTy *T = dynamic_cast<BitsRecTy*>(getType());
  if (T == 0) return 0;  // Cannot subscript a non-bits variable.
  unsigned NumBits = T->getNumBits();

  SmallVector<const Init *, 16> NewBits(Bits.size());
  for (unsigned i = 0, e = Bits.size(); i != e; ++i) {
    if (Bits[i] >= NumBits)
      return 0;

    NewBits[i] = new VarBitInit(this, Bits[i]);
  }
  return new BitsInit(ArrayRef<const Init *>(NewBits));
}

const Init *
TypedInit::convertInitListSlice(const std::vector<unsigned> &Elements) const {
  ListRecTy *T = dynamic_cast<ListRecTy*>(getType());
  if (T == 0) return 0;  // Cannot subscript a non-list variable.

  if (Elements.size() == 1)
    return new VarListElementInit(this, Elements[0]);

  std::vector<const Init*> ListInits;
  ListInits.reserve(Elements.size());
  for (unsigned i = 0, e = Elements.size(); i != e; ++i)
    ListInits.push_back(new VarListElementInit(this, Elements[i]));
  return new ListInit(ListInits, T);
}


const Init *VarInit::resolveBitReference(Record &R, const RecordVal *IRV,
                                         unsigned Bit) const {
  if (R.isTemplateArg(getName())) return 0;
  if (IRV && IRV->getName() != getName()) return 0;

  RecordVal *RV = R.getValue(getName());
  assert(RV && "Reference to a non-existent variable?");
  assert(dynamic_cast<const BitsInit*>(RV->getValue()));
  const BitsInit *BI = (const BitsInit*)RV->getValue();

  assert(Bit < BI->getNumBits() && "Bit reference out of range!");
  const Init *B = BI->getBit(Bit);

  // If the bit is set to some value, or if we are resolving a reference to a
  // specific variable and that variable is explicitly unset, then replace the
  // VarBitInit with it.
  if (IRV || !dynamic_cast<const UnsetInit*>(B))
    return B;
  return 0;
}

const Init *VarInit::resolveListElementReference(Record &R,
                                                 const RecordVal *IRV,
                                                 unsigned Elt) const {
  if (R.isTemplateArg(getName())) return 0;
  if (IRV && IRV->getName() != getName()) return 0;

  RecordVal *RV = R.getValue(getName());
  assert(RV && "Reference to a non-existent variable?");
  const ListInit *LI = dynamic_cast<const ListInit*>(RV->getValue());
  if (!LI) {
    const VarInit *VI = dynamic_cast<const VarInit*>(RV->getValue());
    assert(VI && "Invalid list element!");
    return new VarListElementInit(VI, Elt);
  }

  if (Elt >= LI->getSize())
    return 0;  // Out of range reference.
  const Init *E = LI->getElement(Elt);
  // If the element is set to some value, or if we are resolving a reference
  // to a specific variable and that variable is explicitly unset, then
  // replace the VarListElementInit with it.
  if (IRV || !dynamic_cast<const UnsetInit*>(E))
    return E;
  return 0;
}


RecTy *VarInit::getFieldType(const std::string &FieldName) const {
  if (RecordRecTy *RTy = dynamic_cast<RecordRecTy*>(getType()))
    if (const RecordVal *RV = RTy->getRecord()->getValue(FieldName))
      return RV->getType();
  return 0;
}

const Init *VarInit::getFieldInit(Record &R, const RecordVal *RV,
                                  const std::string &FieldName) const {
  if (dynamic_cast<RecordRecTy*>(getType()))
    if (const RecordVal *Val = R.getValue(VarName)) {
      if (RV != Val && (RV || dynamic_cast<const UnsetInit*>(Val->getValue())))
        return 0;
      const Init *TheInit = Val->getValue();
      assert(TheInit != this && "Infinite loop detected!");
      if (const Init *I = TheInit->getFieldInit(R, RV, FieldName))
        return I;
      else
        return 0;
    }
  return 0;
}

/// resolveReferences - This method is used by classes that refer to other
/// variables which may not be defined at the time the expression is formed.
/// If a value is set for the variable later, this method will be called on
/// users of the value to allow the value to propagate out.
///
const Init *VarInit::resolveReferences(Record &R, const RecordVal *RV) const {
  if (RecordVal *Val = R.getValue(VarName))
    if (RV == Val || (RV == 0 && !dynamic_cast<const UnsetInit*>(Val->getValue())))
      return Val->getValue();
  return this;
}

std::string VarBitInit::getAsString() const {
   return TI->getAsString() + "{" + utostr(Bit) + "}";
}

const Init *VarBitInit::resolveReferences(Record &R,
                                          const RecordVal *RV) const {
  if (const Init *I = getVariable()->resolveBitReference(R, RV, getBitNum()))
    return I;
  return this;
}

std::string VarListElementInit::getAsString() const {
  return TI->getAsString() + "[" + utostr(Element) + "]";
}

const Init *VarListElementInit::resolveReferences(Record &R,
                                                  const RecordVal *RV) const {
  if (const Init *I = getVariable()->resolveListElementReference(R, RV,
                                                           getElementNum()))
    return I;
  return this;
}

const Init *VarListElementInit::resolveBitReference(Record &R,
                                                    const RecordVal *RV,
                                                    unsigned Bit) const {
  // FIXME: This should be implemented, to support references like:
  // bit B = AA[0]{1};
  return 0;
}

const Init *VarListElementInit::
resolveListElementReference(Record &R, const RecordVal *RV,
                            unsigned Elt) const {
  // FIXME: This should be implemented, to support references like:
  // int B = AA[0][1];
  return 0;
}

DefInit *DefInit::get(Record *R) {
  return R->getDefInit();
}

RecTy *DefInit::getFieldType(const std::string &FieldName) const {
  if (const RecordVal *RV = Def->getValue(FieldName))
    return RV->getType();
  return 0;
}

const Init *DefInit::getFieldInit(Record &R, const RecordVal *RV,
                                  const std::string &FieldName) const {
  return Def->getValue(FieldName)->getValue();
}


std::string DefInit::getAsString() const {
  return Def->getName();
}

const Init *FieldInit::resolveBitReference(Record &R, const RecordVal *RV,
                                           unsigned Bit) const {
  if (const Init *BitsVal = Rec->getFieldInit(R, RV, FieldName))
    if (const BitsInit *BI = dynamic_cast<const BitsInit*>(BitsVal)) {
      assert(Bit < BI->getNumBits() && "Bit reference out of range!");
      const Init *B = BI->getBit(Bit);

      if (dynamic_cast<const BitInit*>(B))  // If the bit is set.
        return B;                     // Replace the VarBitInit with it.
    }
  return 0;
}

const Init *FieldInit::resolveListElementReference(Record &R,
                                                   const RecordVal *RV,
                                                   unsigned Elt) const {
  if (const Init *ListVal = Rec->getFieldInit(R, RV, FieldName))
    if (const ListInit *LI = dynamic_cast<const ListInit*>(ListVal)) {
      if (Elt >= LI->getSize()) return 0;
      const Init *E = LI->getElement(Elt);

      // If the element is set to some value, or if we are resolving a
      // reference to a specific variable and that variable is explicitly
      // unset, then replace the VarListElementInit with it.
      if (RV || !dynamic_cast<const UnsetInit*>(E))
        return E;
    }
  return 0;
}

const Init *FieldInit::resolveReferences(Record &R, const RecordVal *RV) const {
  const Init *NewRec = RV ? Rec->resolveReferences(R, RV) : Rec;

  const Init *BitsVal = NewRec->getFieldInit(R, RV, FieldName);
  if (BitsVal) {
    const Init *BVR = BitsVal->resolveReferences(R, RV);
    return BVR->isComplete() ? BVR : this;
  }

  if (NewRec != Rec) {
    return new FieldInit(NewRec, FieldName);
  }
  return this;
}

const Init *DagInit::resolveReferences(Record &R, const RecordVal *RV) const {
  std::vector<const Init*> NewArgs;
  for (unsigned i = 0, e = Args.size(); i != e; ++i)
    NewArgs.push_back(Args[i]->resolveReferences(R, RV));

  const Init *Op = Val->resolveReferences(R, RV);

  if (Args != NewArgs || Op != Val)
    return new DagInit(Op, ValName, NewArgs, ArgNames);

  return this;
}


std::string DagInit::getAsString() const {
  std::string Result = "(" + Val->getAsString();
  if (!ValName.empty())
    Result += ":" + ValName;
  if (Args.size()) {
    Result += " " + Args[0]->getAsString();
    if (!ArgNames[0].empty()) Result += ":$" + ArgNames[0];
    for (unsigned i = 1, e = Args.size(); i != e; ++i) {
      Result += ", " + Args[i]->getAsString();
      if (!ArgNames[i].empty()) Result += ":$" + ArgNames[i];
    }
  }
  return Result + ")";
}


//===----------------------------------------------------------------------===//
//    Other implementations
//===----------------------------------------------------------------------===//

RecordVal::RecordVal(const std::string &N, RecTy *T, unsigned P)
  : Name(N), Ty(T), Prefix(P) {
  Value = Ty->convertValue(new UnsetInit());
  assert(Value && "Cannot create unset value for current type!");
}

void RecordVal::dump() const { errs() << *this; }

void RecordVal::print(raw_ostream &OS, bool PrintSem) const {
  if (getPrefix()) OS << "field ";
  OS << *getType() << " " << getName();

  if (getValue())
    OS << " = " << *getValue();

  if (PrintSem) OS << ";\n";
}

unsigned Record::LastID = 0;

DefInit *Record::getDefInit() {
  if (!TheInit)
    TheInit = new DefInit(this, new RecordRecTy(this));
  return TheInit;
}

void Record::setName(const std::string &Name) {
  if (TrackedRecords.getDef(getName()) == this) {
    TrackedRecords.removeDef(getName());
    this->Name = Name;
    TrackedRecords.addDef(this);
  } else {
    TrackedRecords.removeClass(getName());
    this->Name = Name;
    TrackedRecords.addClass(this);
  }
}

/// resolveReferencesTo - If anything in this record refers to RV, replace the
/// reference to RV with the RHS of RV.  If RV is null, we resolve all possible
/// references.
void Record::resolveReferencesTo(const RecordVal *RV) {
  for (unsigned i = 0, e = Values.size(); i != e; ++i) {
    if (const Init *V = Values[i].getValue())
      Values[i].setValue(V->resolveReferences(*this, RV));
  }
}

void Record::dump() const { errs() << *this; }

raw_ostream &llvm::operator<<(raw_ostream &OS, const Record &R) {
  OS << R.getName();

  const std::vector<std::string> &TArgs = R.getTemplateArgs();
  if (!TArgs.empty()) {
    OS << "<";
    for (unsigned i = 0, e = TArgs.size(); i != e; ++i) {
      if (i) OS << ", ";
      const RecordVal *RV = R.getValue(TArgs[i]);
      assert(RV && "Template argument record not found??");
      RV->print(OS, false);
    }
    OS << ">";
  }

  OS << " {";
  const std::vector<Record*> &SC = R.getSuperClasses();
  if (!SC.empty()) {
    OS << "\t//";
    for (unsigned i = 0, e = SC.size(); i != e; ++i)
      OS << " " << SC[i]->getName();
  }
  OS << "\n";

  const std::vector<RecordVal> &Vals = R.getValues();
  for (unsigned i = 0, e = Vals.size(); i != e; ++i)
    if (Vals[i].getPrefix() && !R.isTemplateArg(Vals[i].getName()))
      OS << Vals[i];
  for (unsigned i = 0, e = Vals.size(); i != e; ++i)
    if (!Vals[i].getPrefix() && !R.isTemplateArg(Vals[i].getName()))
      OS << Vals[i];

  return OS << "}\n";
}

/// getValueInit - Return the initializer for a value with the specified name,
/// or throw an exception if the field does not exist.
///
const Init *Record::getValueInit(StringRef FieldName) const {
  const RecordVal *R = getValue(FieldName);
  if (R == 0 || R->getValue() == 0)
    throw "Record `" + getName() + "' does not have a field named `" +
      FieldName.str() + "'!\n";
  return R->getValue();
}


/// getValueAsString - This method looks up the specified field and returns its
/// value as a string, throwing an exception if the field does not exist or if
/// the value is not a string.
///
std::string Record::getValueAsString(StringRef FieldName) const {
  const RecordVal *R = getValue(FieldName);
  if (R == 0 || R->getValue() == 0)
    throw "Record `" + getName() + "' does not have a field named `" +
          FieldName.str() + "'!\n";

  if (const StringInit *SI = dynamic_cast<const StringInit*>(R->getValue()))
    return SI->getValue();
  throw "Record `" + getName() + "', field `" + FieldName.str() +
        "' does not have a string initializer!";
}

/// getValueAsBitsInit - This method looks up the specified field and returns
/// its value as a BitsInit, throwing an exception if the field does not exist
/// or if the value is not the right type.
///
const BitsInit *Record::getValueAsBitsInit(StringRef FieldName) const {
  const RecordVal *R = getValue(FieldName);
  if (R == 0 || R->getValue() == 0)
    throw "Record `" + getName() + "' does not have a field named `" +
          FieldName.str() + "'!\n";

  if (const BitsInit *BI = dynamic_cast<const BitsInit*>(R->getValue()))
    return BI;
  throw "Record `" + getName() + "', field `" + FieldName.str() +
        "' does not have a BitsInit initializer!";
}

/// getValueAsListInit - This method looks up the specified field and returns
/// its value as a ListInit, throwing an exception if the field does not exist
/// or if the value is not the right type.
///
const ListInit *Record::getValueAsListInit(StringRef FieldName) const {
  const RecordVal *R = getValue(FieldName);
  if (R == 0 || R->getValue() == 0)
    throw "Record `" + getName() + "' does not have a field named `" +
          FieldName.str() + "'!\n";

  if (const ListInit *LI = dynamic_cast<const ListInit*>(R->getValue()))
    return LI;
  throw "Record `" + getName() + "', field `" + FieldName.str() +
        "' does not have a list initializer!";
}

/// getValueAsListOfDefs - This method looks up the specified field and returns
/// its value as a vector of records, throwing an exception if the field does
/// not exist or if the value is not the right type.
///
std::vector<Record*>
Record::getValueAsListOfDefs(StringRef FieldName) const {
  const ListInit *List = getValueAsListInit(FieldName);
  std::vector<Record*> Defs;
  for (unsigned i = 0; i < List->getSize(); i++) {
    if (const DefInit *DI = dynamic_cast<const DefInit*>(List->getElement(i))) {
      Defs.push_back(DI->getDef());
    } else {
      throw "Record `" + getName() + "', field `" + FieldName.str() +
            "' list is not entirely DefInit!";
    }
  }
  return Defs;
}

/// getValueAsInt - This method looks up the specified field and returns its
/// value as an int64_t, throwing an exception if the field does not exist or if
/// the value is not the right type.
///
int64_t Record::getValueAsInt(StringRef FieldName) const {
  const RecordVal *R = getValue(FieldName);
  if (R == 0 || R->getValue() == 0)
    throw "Record `" + getName() + "' does not have a field named `" +
          FieldName.str() + "'!\n";

  if (const IntInit *II = dynamic_cast<const IntInit*>(R->getValue()))
    return II->getValue();
  throw "Record `" + getName() + "', field `" + FieldName.str() +
        "' does not have an int initializer!";
}

/// getValueAsListOfInts - This method looks up the specified field and returns
/// its value as a vector of integers, throwing an exception if the field does
/// not exist or if the value is not the right type.
///
std::vector<int64_t>
Record::getValueAsListOfInts(StringRef FieldName) const {
  const ListInit *List = getValueAsListInit(FieldName);
  std::vector<int64_t> Ints;
  for (unsigned i = 0; i < List->getSize(); i++) {
    if (const IntInit *II = dynamic_cast<const IntInit*>(List->getElement(i))) {
      Ints.push_back(II->getValue());
    } else {
      throw "Record `" + getName() + "', field `" + FieldName.str() +
            "' does not have a list of ints initializer!";
    }
  }
  return Ints;
}

/// getValueAsListOfStrings - This method looks up the specified field and
/// returns its value as a vector of strings, throwing an exception if the
/// field does not exist or if the value is not the right type.
///
std::vector<std::string>
Record::getValueAsListOfStrings(StringRef FieldName) const {
  const ListInit *List = getValueAsListInit(FieldName);
  std::vector<std::string> Strings;
  for (unsigned i = 0; i < List->getSize(); i++) {
    if (const StringInit *II = dynamic_cast<const StringInit*>(List->getElement(i))) {
      Strings.push_back(II->getValue());
    } else {
      throw "Record `" + getName() + "', field `" + FieldName.str() +
            "' does not have a list of strings initializer!";
    }
  }
  return Strings;
}

/// getValueAsDef - This method looks up the specified field and returns its
/// value as a Record, throwing an exception if the field does not exist or if
/// the value is not the right type.
///
Record *Record::getValueAsDef(StringRef FieldName) const {
  const RecordVal *R = getValue(FieldName);
  if (R == 0 || R->getValue() == 0)
    throw "Record `" + getName() + "' does not have a field named `" +
      FieldName.str() + "'!\n";

  if (const DefInit *DI = dynamic_cast<const DefInit*>(R->getValue()))
    return DI->getDef();
  throw "Record `" + getName() + "', field `" + FieldName.str() +
        "' does not have a def initializer!";
}

/// getValueAsBit - This method looks up the specified field and returns its
/// value as a bit, throwing an exception if the field does not exist or if
/// the value is not the right type.
///
bool Record::getValueAsBit(StringRef FieldName) const {
  const RecordVal *R = getValue(FieldName);
  if (R == 0 || R->getValue() == 0)
    throw "Record `" + getName() + "' does not have a field named `" +
      FieldName.str() + "'!\n";

  if (const BitInit *BI = dynamic_cast<const BitInit*>(R->getValue()))
    return BI->getValue();
  throw "Record `" + getName() + "', field `" + FieldName.str() +
        "' does not have a bit initializer!";
}

/// getValueAsDag - This method looks up the specified field and returns its
/// value as an Dag, throwing an exception if the field does not exist or if
/// the value is not the right type.
///
const DagInit *Record::getValueAsDag(StringRef FieldName) const {
  const RecordVal *R = getValue(FieldName);
  if (R == 0 || R->getValue() == 0)
    throw "Record `" + getName() + "' does not have a field named `" +
      FieldName.str() + "'!\n";

  if (const DagInit *DI = dynamic_cast<const DagInit*>(R->getValue()))
    return DI;
  throw "Record `" + getName() + "', field `" + FieldName.str() +
        "' does not have a dag initializer!";
}

std::string Record::getValueAsCode(StringRef FieldName) const {
  const RecordVal *R = getValue(FieldName);
  if (R == 0 || R->getValue() == 0)
    throw "Record `" + getName() + "' does not have a field named `" +
      FieldName.str() + "'!\n";

  if (const CodeInit *CI = dynamic_cast<const CodeInit*>(R->getValue()))
    return CI->getValue();
  throw "Record `" + getName() + "', field `" + FieldName.str() +
    "' does not have a code initializer!";
}


void MultiClass::dump() const {
  errs() << "Record:\n";
  Rec.dump();

  errs() << "Defs:\n";
  for (RecordVector::const_iterator r = DefPrototypes.begin(),
         rend = DefPrototypes.end();
       r != rend;
       ++r) {
    (*r)->dump();
  }
}


void RecordKeeper::dump() const { errs() << *this; }

raw_ostream &llvm::operator<<(raw_ostream &OS, const RecordKeeper &RK) {
  OS << "------------- Classes -----------------\n";
  const std::map<std::string, Record*> &Classes = RK.getClasses();
  for (std::map<std::string, Record*>::const_iterator I = Classes.begin(),
         E = Classes.end(); I != E; ++I)
    OS << "class " << *I->second;

  OS << "------------- Defs -----------------\n";
  const std::map<std::string, Record*> &Defs = RK.getDefs();
  for (std::map<std::string, Record*>::const_iterator I = Defs.begin(),
         E = Defs.end(); I != E; ++I)
    OS << "def " << *I->second;
  return OS;
}


/// getAllDerivedDefinitions - This method returns all concrete definitions
/// that derive from the specified class name.  If a class with the specified
/// name does not exist, an error is printed and true is returned.
std::vector<Record*>
RecordKeeper::getAllDerivedDefinitions(const std::string &ClassName) const {
  Record *Class = getClass(ClassName);
  if (!Class)
    throw "ERROR: Couldn't find the `" + ClassName + "' class!\n";

  std::vector<Record*> Defs;
  for (std::map<std::string, Record*>::const_iterator I = getDefs().begin(),
         E = getDefs().end(); I != E; ++I)
    if (I->second->isSubClassOf(Class))
      Defs.push_back(I->second);

  return Defs;
}

