#include "Record.h"
#include "CodeEmitterGen.h"
#include <ostream>

void CodeEmitterGen::createEmitter(std::ostream &o) {
  std::vector<Record*> Insts;

  const std::map<std::string, Record*> &Defs = Records.getDefs();
  Record *Inst = Records.getClass("Instruction");
  assert(Inst && "Couldn't find Instruction class!");

  for (std::map<std::string, Record*>::const_iterator I = Defs.begin(),
	 E = Defs.end(); I != E; ++I)
    if (I->second->isSubClassOf(Inst))
      Insts.push_back(I->second);

  std::string Namespace = "V9::";
  std::string ClassName = "SparcV9CodeEmitter::";

  //const std::string &Namespace = Inst->getValue("Namespace")->getName();
  o << "unsigned " << ClassName
    << "getBinaryCodeForInstr(MachineInstr &MI) {\n"
    << "  unsigned Value = 0;\n"
    << "  std::cerr << MI;\n"
    << "  switch (MI.getOpcode()) {\n";
  for (std::vector<Record*>::iterator I = Insts.begin(), E = Insts.end();
       I != E; ++I)
  {
    Record *R = *I;
    o << "    case " << Namespace << R->getName() << ": {\n"
      << "      std::cerr << \"Emitting " << R->getName() << "\\n\";\n";

    const RecordVal *InstVal = R->getValue("Inst");
    Init *InitVal = InstVal->getValue();

    assert(dynamic_cast<BitsInit*>(InitVal) &&
           "Can only handle undefined bits<> types!");
    BitsInit *BI = (BitsInit*)InitVal;

    unsigned Value = 0;
    const std::vector<RecordVal> &Vals = R->getValues();

    o << "      // prefilling: ";
    // Start by filling in fixed values...
    for (unsigned i = 0, e = BI->getNumBits(); i != e; ++i) {
      if (BitInit *B = dynamic_cast<BitInit*>(BI->getBit(e-i-1))) {
        Value |= B->getValue() << (e-i-1);
        o << B->getValue();
      } else {
        o << "0";
      }
    }
    o << "\n";

    o << "      // " << *InstVal << "\n";
    o << "      Value = " << Value << "U;\n\n";
    
    // Loop over all of the fields in the instruction adding in any
    // contributions to this value (due to bit references).
    //
    unsigned op = 0;
    std::map<const std::string,unsigned> OpOrder;
    for (unsigned i = 0, e = Vals.size(); i != e; ++i) {
      if (Vals[i].getName() != "Inst" && 
          !Vals[i].getValue()->isComplete() &&
          Vals[i].getName() != "annul" && 
          Vals[i].getName() != "cc" &&
          Vals[i].getName() != "predict")
      {
        o << "      // op" << op << ": " << Vals[i].getName() << "\n"
          << "      int64_t op" << op 
          <<" = getMachineOpValue(MI.getOperand("<<op<<"));\n";
        //<< "      MachineOperand &op" << op <<" = MI.getOperand("<<op<<");\n";
        OpOrder[Vals[i].getName()] = op++;
      }
    }

    unsigned Offset = 31;
    for (int f = Vals.size()-1; f >= 0; --f) {
      if (Vals[f].getPrefix()) {
        BitsInit *FieldInitializer = (BitsInit*)Vals[f].getValue();

        // Scan through the field looking for bit initializers of the current
        // variable...
        for (int i = FieldInitializer->getNumBits()-1; i >= 0; --i) {
          if (BitInit *BI=dynamic_cast<BitInit*>(FieldInitializer->getBit(i))){
            --Offset;
          } else if (UnsetInit *UI = 
                     dynamic_cast<UnsetInit*>(FieldInitializer->getBit(i))) {
            --Offset;
          } else if (VarBitInit *VBI =
                     dynamic_cast<VarBitInit*>(FieldInitializer->getBit(i))) {
            TypedInit *TI = VBI->getVariable();
            if (VarInit *VI = dynamic_cast<VarInit*>(TI)) {
              o << "      Value |= getValueBit(op" << OpOrder[VI->getName()]
                << ", " << VBI->getBitNum()
                << ")" << " << " << Offset << ";\n";
              --Offset;
            } else if (FieldInit *FI = dynamic_cast<FieldInit*>(TI)) {
              // FIXME: implement this!
              o << "FIELD INIT not implemented yet!\n";
            } else {
              o << "Error: UNIMPLEMENTED\n";
            }
          }
        } 
      } else {
        if (Vals[f].getName() == "annul" || Vals[f].getName() == "cc" ||
            Vals[f].getName() == "predict")
          --Offset;
      }
    }

    o << "      break;\n"
      << "    }\n";
  }
  o << "  default:\n"
    << "    std::cerr << \"Not supported instr: \" << MI << \"\\n\";\n"
    << "    abort();\n"
    << "  }\n"
    << "  return Value;\n"
    << "}\n";
}
