//===- Record.cpp - Record implementation ---------------------------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#include "Record.h"
#include "llvm/Support/DataTypes.h"
using namespace llvm;

//===----------------------------------------------------------------------===//
//    Type implementations
//===----------------------------------------------------------------------===//

void RecTy::dump() const { print(std::cerr); }

Init *BitRecTy::convertValue(BitsInit *BI) {
  if (BI->getNumBits() != 1) return 0; // Only accept if just one bit!
  return BI->getBit(0);
}

bool BitRecTy::baseClassOf(const BitsRecTy *RHS) const {
  return RHS->getNumBits() == 1;
}

Init *BitRecTy::convertValue(IntInit *II) {
  int Val = II->getValue();
  if (Val != 0 && Val != 1) return 0;  // Only accept 0 or 1 for a bit!
  
  return new BitInit(Val != 0); 
}

Init *BitRecTy::convertValue(TypedInit *VI) {
  if (dynamic_cast<BitRecTy*>(VI->getType()))
    return VI;  // Accept variable if it is already of bit type!
  return 0;
}

Init *BitsRecTy::convertValue(UnsetInit *UI) {
  BitsInit *Ret = new BitsInit(Size);

  for (unsigned i = 0; i != Size; ++i)
    Ret->setBit(i, new UnsetInit());
  return Ret;
}

Init *BitsRecTy::convertValue(BitInit *UI) {
  if (Size != 1) return 0;  // Can only convert single bit...
  BitsInit *Ret = new BitsInit(1);
  Ret->setBit(0, UI);
  return Ret;
}

// convertValue from Int initializer to bits type: Split the integer up into the
// appropriate bits...
//
Init *BitsRecTy::convertValue(IntInit *II) {
  int64_t Value = II->getValue();
  // Make sure this bitfield is large enough to hold the integer value...
  if (Value >= 0) {
    if (Value & ~((1LL << Size)-1))
      return 0;
  } else {
    if ((Value >> Size) != -1 || ((Value & (1 << (Size-1))) == 0))
      return 0;
  }

  BitsInit *Ret = new BitsInit(Size);
  for (unsigned i = 0; i != Size; ++i)
    Ret->setBit(i, new BitInit(Value & (1 << i)));

  return Ret;
}

Init *BitsRecTy::convertValue(BitsInit *BI) {
  // If the number of bits is right, return it.  Otherwise we need to expand or
  // truncate...
  if (BI->getNumBits() == Size) return BI;
  return 0;
}

Init *BitsRecTy::convertValue(TypedInit *VI) {
  if (BitsRecTy *BRT = dynamic_cast<BitsRecTy*>(VI->getType()))
    if (BRT->Size == Size) {
      BitsInit *Ret = new BitsInit(Size);
      for (unsigned i = 0; i != Size; ++i)
	Ret->setBit(i, new VarBitInit(VI, i));
      return Ret;
    }
  if (Size == 1 && dynamic_cast<BitRecTy*>(VI->getType())) {
    BitsInit *Ret = new BitsInit(1);
    Ret->setBit(0, VI);
    return Ret;
  }
      
  return 0;
}

Init *IntRecTy::convertValue(BitInit *BI) {
  return new IntInit(BI->getValue());
}

Init *IntRecTy::convertValue(BitsInit *BI) {
  int Result = 0;
  for (unsigned i = 0, e = BI->getNumBits(); i != e; ++i) 
    if (BitInit *Bit = dynamic_cast<BitInit*>(BI->getBit(i))) {
      Result |= Bit->getValue() << i;
    } else {
      return 0;
    }
  return new IntInit(Result);
}

Init *IntRecTy::convertValue(TypedInit *TI) {
  if (TI->getType()->typeIsConvertibleTo(this))
    return TI;  // Accept variable if already of the right type!
  return 0;
}

Init *StringRecTy::convertValue(TypedInit *TI) {
  if (dynamic_cast<StringRecTy*>(TI->getType()))
    return TI;  // Accept variable if already of the right type!
  return 0;
}

void ListRecTy::print(std::ostream &OS) const {
  OS << "list<" << *Ty << ">";
}

Init *ListRecTy::convertValue(ListInit *LI) {
  std::vector<Init*> Elements;

  // Verify that all of the elements of the list are subclasses of the
  // appropriate class!
  for (unsigned i = 0, e = LI->getSize(); i != e; ++i)
    if (Init *CI = LI->getElement(i)->convertInitializerTo(Ty))
      Elements.push_back(CI);
    else
      return 0;

  return new ListInit(Elements);
}

Init *ListRecTy::convertValue(TypedInit *TI) {
  // Ensure that TI is compatible with our class.
  if (ListRecTy *LRT = dynamic_cast<ListRecTy*>(TI->getType()))
    if (LRT->getElementType()->typeIsConvertibleTo(getElementType()))
      return TI;
  return 0;
}

Init *CodeRecTy::convertValue(TypedInit *TI) {
  if (TI->getType()->typeIsConvertibleTo(this))
    return TI;
  return 0;
}

Init *DagRecTy::convertValue(TypedInit *TI) {
  if (TI->getType()->typeIsConvertibleTo(this))
    return TI;
  return 0;
}


void RecordRecTy::print(std::ostream &OS) const {
  OS << Rec->getName();
}

Init *RecordRecTy::convertValue(DefInit *DI) {
  // Ensure that DI is a subclass of Rec.
  if (!DI->getDef()->isSubClassOf(Rec))
    return 0;
  return DI;
}

Init *RecordRecTy::convertValue(TypedInit *TI) {
  // Ensure that TI is compatible with Rec.
  if (RecordRecTy *RRT = dynamic_cast<RecordRecTy*>(TI->getType()))
    if (RRT->getRecord()->isSubClassOf(getRecord()) ||
        RRT->getRecord() == getRecord())
      return TI;
  return 0;
}

bool RecordRecTy::baseClassOf(const RecordRecTy *RHS) const {
  return Rec == RHS->getRecord() || RHS->getRecord()->isSubClassOf(Rec);
}


//===----------------------------------------------------------------------===//
//    Initializer implementations
//===----------------------------------------------------------------------===//

void Init::dump() const { return print(std::cerr); }

Init *BitsInit::convertInitializerBitRange(const std::vector<unsigned> &Bits) {
  BitsInit *BI = new BitsInit(Bits.size());
  for (unsigned i = 0, e = Bits.size(); i != e; ++i) {
    if (Bits[i] >= getNumBits()) {
      delete BI;
      return 0;
    }
    BI->setBit(i, getBit(Bits[i]));
  }
  return BI;
}

void BitsInit::print(std::ostream &OS) const {
  //if (!printInHex(OS)) return;
  //if (!printAsVariable(OS)) return;
  //if (!printAsUnset(OS)) return;

  OS << "{ ";
  for (unsigned i = 0, e = getNumBits(); i != e; ++i) {
    if (i) OS << ", ";
    if (Init *Bit = getBit(e-i-1))
      Bit->print(OS);
    else
      OS << "*";
  }
  OS << " }";
}

bool BitsInit::printInHex(std::ostream &OS) const {
  // First, attempt to convert the value into an integer value...
  int Result = 0;
  for (unsigned i = 0, e = getNumBits(); i != e; ++i) 
    if (BitInit *Bit = dynamic_cast<BitInit*>(getBit(i))) {
      Result |= Bit->getValue() << i;
    } else {
      return true;
    }

  OS << "0x" << std::hex << Result << std::dec;
  return false;
}

bool BitsInit::printAsVariable(std::ostream &OS) const {
  // Get the variable that we may be set equal to...
  assert(getNumBits() != 0);
  VarBitInit *FirstBit = dynamic_cast<VarBitInit*>(getBit(0));
  if (FirstBit == 0) return true;
  TypedInit *Var = FirstBit->getVariable();

  // Check to make sure the types are compatible.
  BitsRecTy *Ty = dynamic_cast<BitsRecTy*>(FirstBit->getVariable()->getType());
  if (Ty == 0) return true;
  if (Ty->getNumBits() != getNumBits()) return true; // Incompatible types!

  // Check to make sure all bits are referring to the right bits in the variable
  for (unsigned i = 0, e = getNumBits(); i != e; ++i) {
    VarBitInit *Bit = dynamic_cast<VarBitInit*>(getBit(i));
    if (Bit == 0 || Bit->getVariable() != Var || Bit->getBitNum() != i)
      return true;
  }

  Var->print(OS);
  return false;
}

bool BitsInit::printAsUnset(std::ostream &OS) const {
  for (unsigned i = 0, e = getNumBits(); i != e; ++i) 
    if (!dynamic_cast<UnsetInit*>(getBit(i)))
      return true;
  OS << "?";
  return false;
}

// resolveReferences - If there are any field references that refer to fields
// that have been filled in, we can propagate the values now.
//
Init *BitsInit::resolveReferences(Record &R, const RecordVal *RV) {
  bool Changed = false;
  BitsInit *New = new BitsInit(getNumBits());

  for (unsigned i = 0, e = Bits.size(); i != e; ++i) {
    Init *B;
    Init *CurBit = getBit(i);

    do {
      B = CurBit;
      CurBit = CurBit->resolveReferences(R, RV);
      Changed |= B != CurBit;
    } while (B != CurBit);
    New->setBit(i, CurBit);
  }

  if (Changed)
    return New;
  delete New;
  return this;
}

Init *IntInit::getBinaryOp(BinaryOp Op, Init *RHS) {
  IntInit *RHSi = dynamic_cast<IntInit*>(RHS);
  if (RHSi == 0) return 0;

  int NewValue;
  switch (Op) {
  default: assert(0 && "Unknown binop");
  case SHL: NewValue = Value << RHSi->getValue(); break;
  case SRA: NewValue = Value >> RHSi->getValue(); break;
  case SRL: NewValue = (unsigned)Value >> (unsigned)RHSi->getValue(); break;
  }
  return new IntInit(NewValue);
}


Init *IntInit::convertInitializerBitRange(const std::vector<unsigned> &Bits) {
  BitsInit *BI = new BitsInit(Bits.size());

  for (unsigned i = 0, e = Bits.size(); i != e; ++i) {
    if (Bits[i] >= 32) {
      delete BI;
      return 0;
    }
    BI->setBit(i, new BitInit(Value & (1 << Bits[i])));
  }
  return BI;
}

Init *ListInit::convertInitListSlice(const std::vector<unsigned> &Elements) {
  std::vector<Init*> Vals;
  for (unsigned i = 0, e = Elements.size(); i != e; ++i) {
    if (Elements[i] >= getSize())
      return 0;
    Vals.push_back(getElement(Elements[i]));
  }
  return new ListInit(Vals);
}

Init *ListInit::resolveReferences(Record &R, const RecordVal *RV) {
  std::vector<Init*> Resolved;
  Resolved.reserve(getSize());
  bool Changed = false;

  for (unsigned i = 0, e = getSize(); i != e; ++i) {
    Init *E;
    Init *CurElt = getElement(i);

    do {
      E = CurElt;
      CurElt = CurElt->resolveReferences(R, RV);
      Changed |= E != CurElt;
    } while (E != CurElt);
    Resolved.push_back(E);
  }

  if (Changed)
    return new ListInit(Resolved);
  return this;
}

void ListInit::print(std::ostream &OS) const {
  OS << "[";
  for (unsigned i = 0, e = Values.size(); i != e; ++i) {
    if (i) OS << ", ";
    OS << *Values[i];
  }
  OS << "]";
}

Init *TypedInit::convertInitializerBitRange(const std::vector<unsigned> &Bits) {
  BitsRecTy *T = dynamic_cast<BitsRecTy*>(getType());
  if (T == 0) return 0;  // Cannot subscript a non-bits variable...
  unsigned NumBits = T->getNumBits();

  BitsInit *BI = new BitsInit(Bits.size());
  for (unsigned i = 0, e = Bits.size(); i != e; ++i) {
    if (Bits[i] >= NumBits) {
      delete BI;
      return 0;
    }
    BI->setBit(i, new VarBitInit(this, Bits[i]));
  }
  return BI;
}

Init *TypedInit::convertInitListSlice(const std::vector<unsigned> &Elements) {
  ListRecTy *T = dynamic_cast<ListRecTy*>(getType());
  if (T == 0) return 0;  // Cannot subscript a non-list variable...

  if (Elements.size() == 1)
    return new VarListElementInit(this, Elements[0]);

  std::vector<Init*> ListInits;
  ListInits.reserve(Elements.size());
  for (unsigned i = 0, e = Elements.size(); i != e; ++i)
    ListInits.push_back(new VarListElementInit(this, Elements[i]));
  return new ListInit(ListInits);
}


Init *VarInit::resolveBitReference(Record &R, const RecordVal *IRV,
                                   unsigned Bit) {
  if (R.isTemplateArg(getName())) return 0;
  if (IRV && IRV->getName() != getName()) return 0;

  RecordVal *RV = R.getValue(getName());
  assert(RV && "Reference to a non-existant variable?");
  assert(dynamic_cast<BitsInit*>(RV->getValue()));
  BitsInit *BI = (BitsInit*)RV->getValue();
  
  assert(Bit < BI->getNumBits() && "Bit reference out of range!");
  Init *B = BI->getBit(Bit);

  if (!dynamic_cast<UnsetInit*>(B))  // If the bit is not set...
    return B;                        // Replace the VarBitInit with it.
  return 0;
}

Init *VarInit::resolveListElementReference(Record &R, const RecordVal *IRV,
                                           unsigned Elt) {
  if (R.isTemplateArg(getName())) return 0;
  if (IRV && IRV->getName() != getName()) return 0;

  RecordVal *RV = R.getValue(getName());
  assert(RV && "Reference to a non-existant variable?");
  ListInit *LI = dynamic_cast<ListInit*>(RV->getValue());
  assert(LI && "Invalid list element!");

  if (Elt >= LI->getSize())
    return 0;  // Out of range reference.
  Init *E = LI->getElement(Elt);
  if (!dynamic_cast<UnsetInit*>(E))  // If the element is set
    return E;                        // Replace the VarListElementInit with it.
  return 0;
}


RecTy *VarInit::getFieldType(const std::string &FieldName) const {
  if (RecordRecTy *RTy = dynamic_cast<RecordRecTy*>(getType()))
    if (const RecordVal *RV = RTy->getRecord()->getValue(FieldName))
      return RV->getType();
  return 0;
}

Init *VarInit::getFieldInit(Record &R, const std::string &FieldName) const {
  if (RecordRecTy *RTy = dynamic_cast<RecordRecTy*>(getType()))
    if (const RecordVal *RV = R.getValue(VarName)) {
      Init *TheInit = RV->getValue();
      assert(TheInit != this && "Infinite loop detected!");
      if (Init *I = TheInit->getFieldInit(R, FieldName))
        return I;
      else
        return 0;
    }
  return 0;
}

/// resolveReferences - This method is used by classes that refer to other
/// variables which may not be defined at the time they expression is formed.
/// If a value is set for the variable later, this method will be called on
/// users of the value to allow the value to propagate out.
///
Init *VarInit::resolveReferences(Record &R, const RecordVal *RV) {
  if (RecordVal *Val = R.getValue(VarName))
    if (RV == Val || (RV == 0 && !dynamic_cast<UnsetInit*>(Val->getValue())))
      return Val->getValue();
  return this;
}
  

Init *VarBitInit::resolveReferences(Record &R, const RecordVal *RV) {
  if (Init *I = getVariable()->resolveBitReference(R, RV, getBitNum()))
    return I;
  return this;
}

Init *VarListElementInit::resolveReferences(Record &R, const RecordVal *RV) {
  if (Init *I = getVariable()->resolveListElementReference(R, RV,
                                                           getElementNum()))
    return I;
  return this;
}

Init *VarListElementInit::resolveBitReference(Record &R, const RecordVal *RV,
                                              unsigned Bit) {
  // FIXME: This should be implemented, to support references like:
  // bit B = AA[0]{1};
  return 0;
}

Init *VarListElementInit::
resolveListElementReference(Record &R, const RecordVal *RV, unsigned Elt) {
  // FIXME: This should be implemented, to support references like:
  // int B = AA[0][1];
  return 0;
}

RecTy *DefInit::getFieldType(const std::string &FieldName) const {
  if (const RecordVal *RV = Def->getValue(FieldName))
    return RV->getType();
  return 0;
}

Init *DefInit::getFieldInit(Record &R, const std::string &FieldName) const {
  return Def->getValue(FieldName)->getValue();
}


void DefInit::print(std::ostream &OS) const {
  OS << Def->getName();
}

Init *FieldInit::resolveBitReference(Record &R, const RecordVal *RV,
                                     unsigned Bit) {
  if (Init *BitsVal = Rec->getFieldInit(R, FieldName))
    if (BitsInit *BI = dynamic_cast<BitsInit*>(BitsVal)) {
      assert(Bit < BI->getNumBits() && "Bit reference out of range!");
      Init *B = BI->getBit(Bit);
      
      if (dynamic_cast<BitInit*>(B))  // If the bit is set...
        return B;                     // Replace the VarBitInit with it.
    }
  return 0;
}

Init *FieldInit::resolveListElementReference(Record &R, const RecordVal *RV,
                                             unsigned Elt) {
  if (Init *ListVal = Rec->getFieldInit(R, FieldName))
    if (ListInit *LI = dynamic_cast<ListInit*>(ListVal)) {
      if (Elt >= LI->getSize()) return 0;
      Init *E = LI->getElement(Elt);

      if (!dynamic_cast<UnsetInit*>(E))  // If the bit is set...
        return E;                  // Replace the VarListElementInit with it.
    }
  return 0;
}

Init *FieldInit::resolveReferences(Record &R, const RecordVal *RV) {
  Init *NewRec = RV ? Rec->resolveReferences(R, RV) : Rec;

  Init *BitsVal = NewRec->getFieldInit(R, FieldName);
  if (BitsVal) {
    Init *BVR = BitsVal->resolveReferences(R, RV);
    return BVR->isComplete() ? BVR : this;
  }

  if (NewRec != Rec) {
    dump();
    NewRec->dump(); std::cerr << "\n";
    return new FieldInit(NewRec, FieldName);
  }
  return this;
}


void DagInit::print(std::ostream &OS) const {
  OS << "(" << NodeTypeDef->getName();
  if (Args.size()) {
    OS << " " << *Args[0];
    if (!ArgNames[0].empty()) OS << ":$" << ArgNames[0];
    for (unsigned i = 1, e = Args.size(); i != e; ++i) {
      OS << ", " << *Args[i];
      if (!ArgNames[i].empty()) OS << ":$" << ArgNames[i];
    }
  }
  OS << ")";
}


//===----------------------------------------------------------------------===//
//    Other implementations
//===----------------------------------------------------------------------===//

RecordVal::RecordVal(const std::string &N, RecTy *T, unsigned P)
  : Name(N), Ty(T), Prefix(P) {
  Value = Ty->convertValue(new UnsetInit());
  assert(Value && "Cannot create unset value for current type!");
}

void RecordVal::dump() const { std::cerr << *this; }

void RecordVal::print(std::ostream &OS, bool PrintSem) const {
  if (getPrefix()) OS << "field ";
  OS << *getType() << " " << getName();

  if (getValue())
    OS << " = " << *getValue();

  if (PrintSem) OS << ";\n";
}

/// resolveReferencesTo - If anything in this record refers to RV, replace the
/// reference to RV with the RHS of RV.  If RV is null, we resolve all possible
/// references.
void Record::resolveReferencesTo(const RecordVal *RV) {
  for (unsigned i = 0, e = Values.size(); i != e; ++i) {
    if (Init *V = Values[i].getValue())
      Values[i].setValue(V->resolveReferences(*this, RV));
  }
}


void Record::dump() const { std::cerr << *this; }

std::ostream &llvm::operator<<(std::ostream &OS, const Record &R) {
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
Init *Record::getValueInit(const std::string &FieldName) const {
  const RecordVal *R = getValue(FieldName);
  if (R == 0 || R->getValue() == 0)
    throw "Record `" + getName() + "' does not have a field named `" +
      FieldName + "'!\n";
  return R->getValue();
}


/// getValueAsString - This method looks up the specified field and returns its
/// value as a string, throwing an exception if the field does not exist or if
/// the value is not a string.
///
std::string Record::getValueAsString(const std::string &FieldName) const {
  const RecordVal *R = getValue(FieldName);
  if (R == 0 || R->getValue() == 0)
    throw "Record `" + getName() + "' does not have a field named `" +
          FieldName + "'!\n";

  if (const StringInit *SI = dynamic_cast<const StringInit*>(R->getValue()))
    return SI->getValue();
  throw "Record `" + getName() + "', field `" + FieldName +
        "' does not have a string initializer!";
}

/// getValueAsBitsInit - This method looks up the specified field and returns
/// its value as a BitsInit, throwing an exception if the field does not exist
/// or if the value is not the right type.
///
BitsInit *Record::getValueAsBitsInit(const std::string &FieldName) const {
  const RecordVal *R = getValue(FieldName);
  if (R == 0 || R->getValue() == 0)
    throw "Record `" + getName() + "' does not have a field named `" +
          FieldName + "'!\n";

  if (BitsInit *BI = dynamic_cast<BitsInit*>(R->getValue()))
    return BI;
  throw "Record `" + getName() + "', field `" + FieldName +
        "' does not have a BitsInit initializer!";
}

/// getValueAsListInit - This method looks up the specified field and returns
/// its value as a ListInit, throwing an exception if the field does not exist
/// or if the value is not the right type.
///
ListInit *Record::getValueAsListInit(const std::string &FieldName) const {
  const RecordVal *R = getValue(FieldName);
  if (R == 0 || R->getValue() == 0)
    throw "Record `" + getName() + "' does not have a field named `" +
          FieldName + "'!\n";

  if (ListInit *LI = dynamic_cast<ListInit*>(R->getValue()))
    return LI;
  throw "Record `" + getName() + "', field `" + FieldName +
        "' does not have a list initializer!";
}

/// getValueAsInt - This method looks up the specified field and returns its
/// value as an int, throwing an exception if the field does not exist or if
/// the value is not the right type.
///
int Record::getValueAsInt(const std::string &FieldName) const {
  const RecordVal *R = getValue(FieldName);
  if (R == 0 || R->getValue() == 0)
    throw "Record `" + getName() + "' does not have a field named `" +
          FieldName + "'!\n";

  if (IntInit *II = dynamic_cast<IntInit*>(R->getValue()))
    return II->getValue();
  throw "Record `" + getName() + "', field `" + FieldName +
        "' does not have a list initializer!";
}

/// getValueAsDef - This method looks up the specified field and returns its
/// value as a Record, throwing an exception if the field does not exist or if
/// the value is not the right type.
///
Record *Record::getValueAsDef(const std::string &FieldName) const {
  const RecordVal *R = getValue(FieldName);
  if (R == 0 || R->getValue() == 0)
    throw "Record `" + getName() + "' does not have a field named `" +
      FieldName + "'!\n";

  if (DefInit *DI = dynamic_cast<DefInit*>(R->getValue()))
    return DI->getDef();
  throw "Record `" + getName() + "', field `" + FieldName +
        "' does not have a list initializer!";
}

/// getValueAsBit - This method looks up the specified field and returns its
/// value as a bit, throwing an exception if the field does not exist or if
/// the value is not the right type.
///
bool Record::getValueAsBit(const std::string &FieldName) const {
  const RecordVal *R = getValue(FieldName);
  if (R == 0 || R->getValue() == 0)
    throw "Record `" + getName() + "' does not have a field named `" +
      FieldName + "'!\n";

  if (BitInit *BI = dynamic_cast<BitInit*>(R->getValue()))
    return BI->getValue();
  throw "Record `" + getName() + "', field `" + FieldName +
        "' does not have a bit initializer!";
}

/// getValueAsDag - This method looks up the specified field and returns its
/// value as an Dag, throwing an exception if the field does not exist or if
/// the value is not the right type.
///
DagInit *Record::getValueAsDag(const std::string &FieldName) const {
  const RecordVal *R = getValue(FieldName);
  if (R == 0 || R->getValue() == 0)
    throw "Record `" + getName() + "' does not have a field named `" +
      FieldName + "'!\n";

  if (DagInit *DI = dynamic_cast<DagInit*>(R->getValue()))
    return DI;
  throw "Record `" + getName() + "', field `" + FieldName +
        "' does not have a dag initializer!";
}


void RecordKeeper::dump() const { std::cerr << *this; }

std::ostream &llvm::operator<<(std::ostream &OS, const RecordKeeper &RK) {
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
  Record *Class = Records.getClass(ClassName);
  if (!Class)
    throw "ERROR: Couldn't find the `" + ClassName + "' class!\n";

  std::vector<Record*> Defs;
  for (std::map<std::string, Record*>::const_iterator I = getDefs().begin(),
         E = getDefs().end(); I != E; ++I)
    if (I->second->isSubClassOf(Class))
      Defs.push_back(I->second);

  return Defs;
}

