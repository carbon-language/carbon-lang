//===- SimpleInstrSelEmitter.cpp - Generate a Simple Instruction Selector ------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This tablegen backend is responsible for emitting an instruction selector
// 
//
//===----------------------------------------------------------------------===//
#include "InstrInfoEmitter.h"
#include "SimpleInstrSelEmitter.h"
#include "CodeGenWrappers.h"
#include "Record.h"
#include "Support/Debug.h"
#include "Support/StringExtras.h"
#include <set>


#include "Record.h"
#include "Support/CommandLine.h"
#include "llvm/System/Signals.h"
#include "Support/FileUtilities.h"
#include "CodeEmitterGen.h"
#include "RegisterInfoEmitter.h"
#include "InstrInfoEmitter.h"
#include "InstrSelectorEmitter.h"
#include "SimpleInstrSelEmitter.h"
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <vector>

namespace llvm {

  std::string FnDecs;

// run - Emit the main instruction description records for the target...
void SimpleInstrSelEmitter::run(std::ostream &OS) {


  //  EmitSourceFileHeader("Mark's Instruction Selector for the X86 target", OS);


//   OS << "#include \"llvm/CodeGen/MachineInstrBuilder.h\"\n";

//   OS << "#include \"llvm/Constants.h\"\n";
//   OS << "#include \"llvm/DerivedTypes.h\"\n";
//   OS << "#include \"llvm/Function.h\"\n";
//   OS << "#include \"llvm/Instructions.h\"\n";
//   OS << "#include \"llvm/Pass.h\"\n";
//   OS << "#include \"llvm/CodeGen/IntrinsicLowering.h\"\n";
//   OS << "#include \"llvm/CodeGen/MachineConstantPool.h\"\n";
//   OS << "#include \"llvm/CodeGen/MachineFrameInfo.h\"\n";
//   OS << "#include \"llvm/CodeGen/MachineFunction.h\"\n";
//   OS << "#include \"llvm/CodeGen/MachineInstrBuilder.h\"\n";
//   OS << "#include \"llvm/CodeGen/SSARegMap.h\"\n";
//   OS << "#include \"llvm/Target/MRegisterInfo.h\"\n";
//   OS << "#include \"llvm/Target/TargetMachine.h\"\n";
//   OS << "#include \"llvm/Support/InstVisitor.h\"\n";

//  OS << "using namespace llvm;\n\n";


  FnDecs = "";

  // for each InstrClass

  std::vector<Record*> Recs = Records.getAllDerivedDefinitions("InstrClass");
  for (unsigned i = 0, e = Recs.size(); i != e; ++i) {
    std::string InstrClassName = Recs[i]->getName();
    OS << "// Generate BMI instructions for " << InstrClassName << "\n\n";
    OS << "void ISel::visit";
    OS << Recs[i]->getValueAsString("FunctionName");
    OS << "(" <<  Recs[i]->getValueAsString("InstructionName") << " &I)\n{" << "\n";
    // for each supported InstrSubclass

    OS << spacing() << "unsigned DestReg = getReg(I);\n";
    OS << spacing() << "unsigned Op0Reg  = getReg(I.getOperand(0));\n";
    OS << spacing() << "unsigned Op1Reg  = getReg(I.getOperand(1));\n";
    OS << spacing() << "Value   *Op0Val  = I.getOperand(0);\n";
    OS << spacing() << "Value   *Op1Val  = I.getOperand(1);\n";

    OS << spacing() << "MachineBasicBlock::iterator IP = BB->end();\n";

    OS << std::endl;

    ListInit *SupportedSubclasses = Recs[i]->getValueAsListInit("Supports");

    //OS << spacing() << InstrClassName << "Prep();" << "\n";
    //FnDecs += "void ISel::" + InstrClassName + "Prep() {\n\n}\n\n";

    std::vector<std::string> vi;

    // generate subclasses nested switch statements
    InstrSubclasses(OS, InstrClassName, InstrClassName, SupportedSubclasses, vi, 0);

    //OS << spacing() << InstrClassName << "Post();\n";
    //FnDecs += "void ISel::" + InstrClassName + "Post() {\n\n}\n\n";

    OS << "}\n";
    OS << "\n\n\n";

  } // for each instrclass

  //  OS << "} //namespace\n";


#if 0
  // print out function stubs
  OS << "\n\n\n//Functions\n\n" << FnDecs;

  // print out getsubclass() definitions
  std::vector<Record*> SubclassColRec = Records.getAllDerivedDefinitions("InstrSubclassCollection");
  for (unsigned j=0, m=SubclassColRec.getSize(); j!=m; ++j) {
    std::string SubclassName = SubclassColRec[j]->getName();
    FnDecs += "unsigned ISel::get" + SubclassName + "() {\n\n";

    ListInit* list = dynamic_cast<ListInit*>(SubclassColRec[j].getValueAsListInit("List"));
    
    for (unsigned k=0; n=list.getSize(); k!=n; ++k) {

    FnDecs += "}\n\n";
  }
#endif

} //run

  

// find instructions that match all the subclasses (only support for 1 now)
Record* SimpleInstrSelEmitter::findInstruction(std::ostream &OS, std::string cl, std::vector<std::string>& vi) {
  std::vector<Record*> Recs = Records.getAllDerivedDefinitions("TargInstrSet");

  for (unsigned i = 0, e = Recs.size(); i != e; ++i) {
    Record* thisClass = Recs[i]->getValueAsDef("Class");

    if (thisClass->getName() == cl) {

      // get the Subclasses this supports
      ListInit* SubclassList = Recs[i]->getValueAsListInit("List");

      bool Match = true;

      if (SubclassList->getSize() != vi.size())
	Match = false;
      
      // match the instruction's supported subclasses with the subclasses we are looking for

      for (unsigned j=0, f=SubclassList->getSize(); j!=f; ++j) {
	DefInit* SubclassDef = dynamic_cast<DefInit*>(SubclassList->getElement(j));
	Record* thisSubclass = SubclassDef->getDef();

	std::string searchingFor = vi[j];

	if (thisSubclass->getName() != searchingFor) {
	  Match = false;
	}

      } // for each subclass list

      if (Match == true) { return Recs[i]; }
  
    } //if instrclass matches

  } // for all instructions

  // if no instructions found, return NULL
  return NULL;

} //findInstruction
  


Record* SimpleInstrSelEmitter::findRegister(std::ostream &OS, std::string regname) {
  std::vector<Record*> Recs = Records.getAllDerivedDefinitions("Register");

  for (unsigned i = 0, e = Recs.size(); i != e; ++i) {
    Record* thisReg = Recs[i];

    if (thisReg->getName() == regname) return Recs[i];
  }
  
  return NULL;

}

// handle "::" and "+" etc
std::string SimpleInstrSelEmitter::formatRegister(std::ostream &OS, std::string regname) {
  std::string Reg;
  std::string suffix;

  int x = std::strcspn(regname.c_str(),"+-");

  // operate on text before "+" or "-", append it back at the end
  Reg = regname.substr(0,x);
  suffix = regname.substr(x,regname.length());

  unsigned int y = std::strcspn(Reg.c_str(),":");

  if (y == Reg.length()) { // does not contain "::"
    
    Record* RegRec = findRegister(OS,Reg);

    assert(RegRec && "Register not found!");

    if (RegRec->getValueAsString("Namespace") != "Virtual") {
      Reg = RegRec->getValueAsString("Namespace") + "::" + RegRec->getName();
    } else {
      Reg = RegRec->getName();
    }
  } // regular case

  // append + or - at the end again (i.e. X86::EAX+1)
  Reg = Reg + suffix;

  return Reg;
}


// take information in the instruction class and generate the correct BMI call
void SimpleInstrSelEmitter::generateBMIcall(std::ostream &OS, std::string MBB, std::string IP, std::string Opcode, int NumOperands, ListInit &instroperands, ListInit &operands) {

  // find Destination Register
  StringInit* DestRegStr = dynamic_cast<StringInit*>(operands.getElement(0));
  std::string DestReg = formatRegister(OS,DestRegStr->getValue());

  OS << "BuildMI(";
  OS << MBB << ", ";
  OS << IP << ", ";
  OS << Opcode << ", ";
  OS << NumOperands;

  if (DestReg != "Pseudo") {
    OS << ", " << DestReg << ")";
  } else {
    OS << ")";
  }
  
  // handle the .add stuff
  for (unsigned i=0, e=instroperands.getSize(); i!=e; ++i) {
    DefInit* OpDef = dynamic_cast<DefInit*>(instroperands.getElement(i));
    StringInit* RegStr = dynamic_cast<StringInit*>(operands.getElement(i+1));

    Record* Op = OpDef->getDef();

    std::string opstr = Op->getValueAsString("Name");

    std::string regname;

    if (opstr == "Register") {
      regname = formatRegister(OS,RegStr->getValue());
    } else {
      regname = RegStr->getValue();
    }

    OS << ".add" << opstr << "(" << regname << ")";
  }
  
  OS << ";\n";
  
} //generateBMIcall

  
 std::string SimpleInstrSelEmitter::spacing() {
   return globalSpacing;
 }

 std::string SimpleInstrSelEmitter::addspacing() {
   globalSpacing += "  ";
   return globalSpacing;
 }

 std::string SimpleInstrSelEmitter::remspacing() {
   globalSpacing = globalSpacing.substr(0,globalSpacing.length()-2);
   return globalSpacing;
 }


 // recursively print out the subclasses of an instruction
 //
 void SimpleInstrSelEmitter::InstrSubclasses(std::ostream &OS, std::string prefix, std::string InstrClassName, ListInit* SupportedSubclasses, std::vector<std::string>& vi, unsigned depth) {

   
   if (depth >= SupportedSubclasses->getSize()) {
      return;
   }

   // get the subclass collection
   
   DefInit* InstrSubclassColl = dynamic_cast<DefInit*>(SupportedSubclasses->getElement(depth));

   Record* InstrSubclassRec = InstrSubclassColl->getDef();

   std::string SubclassName = InstrSubclassRec->getName();
   
   
   if (InstrSubclassRec->getValueAsString("PreCode") != "") {
     //	  OS << spacing() << prefix << "_" << Subclass->getName() << "_Prep();\n";
     OS << spacing() << InstrSubclassRec->getValueAsString("PreCode") << "\n\n";
   }
   
   
   OS << spacing() << "// Looping through " << SubclassName << "\n";
   
   OS << spacing() << "switch (" <<  SubclassName <<") {\n";
   addspacing();
   
   ListInit* SubclassList = InstrSubclassRec->getValueAsListInit("List");
   
   for (unsigned k=0, g = SubclassList->getSize(); k!=g; ++k) {
     
     DefInit* SubclassDef = dynamic_cast<DefInit*>(SubclassList->getElement(k));
     
     Record* Subclass = SubclassDef->getDef();
     
     OS << spacing() << "// " << prefix << "_" << Subclass->getName() << "\n";
     OS << spacing() << "case " << Subclass->getName() << ":\n";
     addspacing();
     OS << spacing() << "{\n";
     
     
     vi.push_back(Subclass->getName());
     
     // go down hierarchy
     InstrSubclasses(OS, prefix + "_" + Subclass->getName(), InstrClassName, SupportedSubclasses, vi, depth+1);
     
     // find the record that matches this
     Record *theInstructionSet = findInstruction(OS, InstrClassName, vi);
     
     // only print out the assertion if this is a leaf
     if ( (theInstructionSet == NULL) && (depth == (SupportedSubclasses->getSize() - 1)) ) {
       
       OS << spacing() << "assert(0 && \"No instructions defined for " << InstrClassName << " instructions of subclasses " << prefix << "_" << Subclass->getName() << "!\");" << "\n";
       
     } else if (theInstructionSet != NULL) {
       
       if (theInstructionSet->getValueAsString("PreCode") != "") {
	 OS << spacing() << theInstructionSet->getValueAsString("PreCode") << "\n\n";
       }
       
       ListInit *theInstructions = theInstructionSet->getValueAsListInit("Instructions");
       
       ListInit *registerlists = theInstructionSet->getValueAsListInit("Operands"); // not necessarily registers anymore, but the name will stay for now
       
       for (unsigned l=0, h=theInstructions->getSize(); l!=h; ++l) {
	 
	 DefInit *theInstructionDef = dynamic_cast<DefInit*>(theInstructions->getElement(l));
	 Record *theInstruction = theInstructionDef->getDef();
	 
	 ListInit *operands = theInstruction->getValueAsListInit("Params");
	 
	 OS << spacing();
	 
	 ListInit* registers = dynamic_cast<ListInit*>(registerlists->getElement(l));
	 
	 // handle virtual instructions here before going to generateBMIcall
	 
	 if (theInstruction->getValueAsString("Namespace") == "Virtual") {
	   
	   // create reg for different sizes
	   std::string Instr = theInstruction->getName();
	   StringInit* DestRegInit = dynamic_cast<StringInit*>(registers->getElement(0));
	   std::string DestReg = DestRegInit->getValue();
	   std::string theType;
	   
	   if (Instr == "NullInstruction") { } // do nothing
	   else if (Instr == "CreateRegByte")
	     OS << "unsigned " << DestReg << " = makeAnotherReg(Type::SByteTy);\n";
	   else if (Instr == "CreateRegShort")
	     OS << "unsigned " << DestReg << " = makeAnotherReg(Type::SShortTy);\n";
	   else if (Instr == "CreateRegInt")
	     OS << "unsigned " << DestReg << " = makeAnotherReg(Type::SIntTy);\n";
	   else if (Instr == "CreateRegLong")
	     OS << "unsigned " << DestReg << " = makeAnotherReg(Type::SLongTy);\n";
	   else if (Instr == "CreateRegUByte")
	     OS << "unsigned " << DestReg << " = makeAnotherReg(Type::UByteTy);\n";
	   else if (Instr == "CreateRegUShort")
	     OS << "unsigned " << DestReg << " = makeAnotherReg(Type::UShortTy);\n";
	   else if (Instr == "CreateRegUInt")
	     OS << "unsigned " << DestReg << " = makeAnotherReg(Type::UIntTy);\n";
	   else if (Instr == "CreateRegULong") 
	     OS << "unsigned " << DestReg << " = makeAnotherReg(Type::ULongTy);\n";
	   else if (Instr == "CreateRegFloat") 
	     OS << "unsigned " << DestReg << " = makeAnotherReg(Type::FloatTy);\n";
	   else if (Instr == "CreateRegDouble") 
	     OS << "unsigned " << DestReg << " = makeAnotherReg(Type::DoubleTy);\n";
	   else if (Instr == "CreateRegPointer") 
	     OS << "unsigned " << DestReg << " = makeAnotherReg(Type::PointerTy_;\n";
	   else 
	     OS << "unsigned " << DestReg << " = makeAnotherReg(Type::SByteTy);\n"; // create a byte by default
	   
	   
	 } else {
	   std::string InstrName;
	   
	   if (theInstruction->getValueAsString("Namespace") != "Virtual") {
	     InstrName = theInstruction->getValueAsString("Namespace") + "::" + theInstruction->getValueAsString("Name");
	   } else {
	     // shouldn't ever happen, virtual instrs should be caught before this
	     InstrName = theInstruction->getValueAsString("Name");
	   }
	   
	   generateBMIcall(OS, "*BB","IP",InstrName,theInstruction->getValueAsInt("NumOperands"),*operands,*registers);
	 }
	 
       }
       
       if (theInstructionSet->getValueAsString("PostCode") != "") {
	 OS << spacing() << theInstructionSet->getValueAsString("PostCode") << "\n\n";
       }
       
     }
     
     
     
     if (InstrSubclassRec->getValueAsString("PostCode") != "") {
       //OS << spacing() << "// " << prefix << "_" << Subclass->getName() << "_Prep();\n";
       OS << spacing() << InstrSubclassRec->getValueAsString("PostCode") << "\n\n";
     }
     
     
     OS << spacing() << "break;\n";

     OS << spacing() << "}\n\n";
     
     remspacing();
     
     vi.pop_back();
   }
   
   // provide a default case for the switch
   
   OS << spacing() << "default:\n";
   OS << spacing() << "  assert(0 && \"No instructions defined for " << InstrClassName << " instructions of subclasses " << prefix << "_" << SubclassName << "!\");" << "\n";
   OS << spacing() << "  break;\n\n";
   
   remspacing();
   OS << spacing() << "}\n";
   
 }


 // ret br switch invoke unwind
 // add sub mul div rem setcc (eq ne lt gt le ge)
 // and or xor sbl sbr
 // malloc free alloca load store
 // getelementptr phi cast call vanext vaarg
 
} // End llvm namespace
