//===- Record.cpp - Record implementation ---------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#include "Record.h"

//===----------------------------------------------------------------------===//
//    Type implementations
//===----------------------------------------------------------------------===//

void RecTy::dump() const { print(std::cerr); }

Init *BitRecTy::convertValue(BitsInit *BI) {
  if (BI->getNumBits() != 1) return 0; // Only accept if just one bit!
  return BI->getBit(0);
}

Init *BitRecTy::convertValue(IntInit *II) {
  int Val = II->getValue();
  if (Val != 0 && Val != 1) return 0;  // Only accept 0 or 1 for a bit!
  delete II;
  
  return new BitInit(Val != 0); 
}

Init *BitRecTy::convertValue(VarInit *VI) {
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
  int Value = II->getValue();
  delete II;

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

Init *BitsRecTy::convertValue(VarInit *VI) {
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

Init *IntRecTy::convertValue(VarInit *VI) {
  if (dynamic_cast<IntRecTy*>(VI->getType()))
    return VI;  // Accept variable if already of the right type!
  return 0;
}

Init *StringRecTy::convertValue(VarInit *VI) {
  if (dynamic_cast<StringRecTy*>(VI->getType()))
    return VI;  // Accept variable if already of the right type!
  return 0;
}

void ListRecTy::print(std::ostream &OS) const {
  OS << "list<" << Class->getName() << ">";
}

Init *ListRecTy::convertValue(ListInit *LI) {
  // Verify that all of the elements of the list are subclasses of the
  // appopriate class!
  for (unsigned i = 0, e = LI->getSize(); i != e; ++i)
    if (!LI->getElement(i)->isSubClassOf(Class))
      return 0;
  return LI;
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
  if (!printAsVariable(OS)) return;
  if (!printAsUnset(OS)) return;

  OS << "{ ";
  for (unsigned i = 0, e = getNumBits(); i != e; ++i) {
    if (i) OS << ", ";
    getBit(e-i-1)->print(OS);
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
  VarInit *Var = FirstBit->getVariable();

  // Check to make sure the types are compatible.
  BitsRecTy *Ty = dynamic_cast<BitsRecTy*>(Var->getType());
  if (Ty == 0) return true;
  if (Ty->getNumBits() != getNumBits()) return true; // Incompatible types!

  // Check to make sure all bits are referring to the right bits in the variable
  for (unsigned i = 0, e = getNumBits(); i != e; ++i) {
    VarBitInit *Bit = dynamic_cast<VarBitInit*>(getBit(i));
    if (Bit == 0 || Bit->getVariable() != Var || Bit->getBitNum() != i)
      return true;
  }

  OS << Var->getName();
  return false;
}

bool BitsInit::printAsUnset(std::ostream &OS) const {
  for (unsigned i = 0, e = getNumBits(); i != e; ++i) 
    if (!dynamic_cast<UnsetInit*>(getBit(i)))
      return true;
  OS << "?";
  return false;
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

void ListInit::print(std::ostream &OS) const {
  OS << "[";
  for (unsigned i = 0, e = Records.size(); i != e; ++i) {
    if (i) OS << ", ";
    OS << Records[i]->getName();
  }
  OS << "]";
}

Init *VarInit::convertInitializerBitRange(const std::vector<unsigned> &Bits) {
  BitsRecTy *T = dynamic_cast<BitsRecTy*>(Ty);
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

Init *BitsInit::resolveReferences(Record &R) {
  bool Changed = false;
  BitsInit *New = new BitsInit(getNumBits());

  for (unsigned i = 0, e = Bits.size(); i != e; ++i) {
    Init *B;
    New->setBit(i, getBit(i));
    do {
      B = New->getBit(i);
      New->setBit(i, B->resolveReferences(R));
      Changed |= B != New->getBit(i);
    } while (B != New->getBit(i));
  }

  if (Changed)
    return New;
  delete New;
  return this;
}



Init *VarBitInit::resolveReferences(Record &R) {
  if (R.isTemplateArg(getVariable()->getName()))
    return this;

  RecordVal *RV = R.getValue(getVariable()->getName());
  assert(RV && "Reference to a non-existant variable?");
  assert(dynamic_cast<BitsInit*>(RV->getValue()));
  BitsInit *BI = (BitsInit*)RV->getValue();
  
  assert(getBitNum() < BI->getNumBits() && "Bit reference out of range!");
  Init *B = BI->getBit(getBitNum());

  if (!dynamic_cast<UnsetInit*>(B))  // If the bit is not set...
    return B;                        // Replace the VarBitInit with it.
  return this;
}

void DefInit::print(std::ostream &OS) const {
  OS << Def->getName();
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
  if (getValue()) {
    OS << " = " << *getValue();
  }
  if (PrintSem) OS << ";\n";
}

// resolveReferences - If there are any field references that refer to fields
// that have been filled in, we can propagate the values now.
//
void Record::resolveReferences() {
  for (unsigned i = 0, e = Values.size(); i != e; ++i)
    Values[i].setValue(Values[i].getValue()->resolveReferences(*this));
}

void Record::dump() const { std::cerr << *this; }

std::ostream &operator<<(std::ostream &OS, const Record &R) {
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

void RecordKeeper::dump() const { std::cerr << *this; }

std::ostream &operator<<(std::ostream &OS, const RecordKeeper &RK) {
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
