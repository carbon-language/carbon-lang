//===-- Writer.cpp - Library for Printing VM assembly files ------*- C++ -*--=//
//
// This library implements the functionality defined in llvm/Assembly/Writer.h
//
// This library uses the Analysis library to figure out offsets for
// variables in the method tables...
//
// TODO: print out the type name instead of the full type if a particular type
//       is in the symbol table...
//
//===----------------------------------------------------------------------===//

#include "llvm/Assembly/Writer.h"
#include "llvm/Analysis/SlotCalculator.h"
#include "llvm/Module.h"
#include "llvm/Method.h"
#include "llvm/BasicBlock.h"
#include "llvm/ConstPoolVals.h"
#include "llvm/iOther.h"
#include "llvm/iMemory.h"

void DebugValue(const Value *V) {
  cerr << V << endl;
}

// WriteAsOperand - Write the name of the specified value out to the specified
// ostream.  This can be useful when you just want to print int %reg126, not the
// whole instruction that generated it.
//
ostream &WriteAsOperand(ostream &Out, const Value *V, bool PrintType, 
			bool PrintName, SlotCalculator *Table) {
  if (PrintType)
    Out << " " << V->getType();
  
  if (V->hasName() && PrintName) {
    Out << " %" << V->getName();
  } else {
    if (const ConstPoolVal *CPV = V->castConstant()) {
      Out << " " << CPV->getStrValue();
    } else {
      int Slot;
      if (Table) {
	Slot = Table->getValSlot(V);
      } else {
	if (const Type *Ty = V->castType()) {
	  return Out << " " << Ty;
	} else if (const MethodArgument *MA = V->castMethodArgument()) {
	  Table = new SlotCalculator(MA->getParent(), true);
	} else if (const Instruction *I = V->castInstruction()) {
	  Table = new SlotCalculator(I->getParent()->getParent(), true);
	} else if (const BasicBlock *BB = V->castBasicBlock()) {
	  Table = new SlotCalculator(BB->getParent(), true);
	} else if (const Method *Meth = V->castMethod()) {
	  Table = new SlotCalculator(Meth, true);
	} else if (const Module *Mod  = V->castModule()) {
	  Table = new SlotCalculator(Mod, true);
	} else {
	  return Out << "BAD VALUE TYPE!";
	}
	Slot = Table->getValSlot(V);
	delete Table;
      }
      if (Slot >= 0)  Out << " %" << Slot;
      else if (PrintName)
        Out << "<badref>";     // Not embeded into a location?
    }
  }
  return Out;
}



class AssemblyWriter : public ModuleAnalyzer {
  ostream &Out;
  SlotCalculator &Table;
public:
  inline AssemblyWriter(ostream &o, SlotCalculator &Tab) : Out(o), Table(Tab) {
  }

  inline void write(const Module *M)         { processModule(M);      }
  inline void write(const Method *M)         { processMethod(M);      }
  inline void write(const BasicBlock *BB)    { processBasicBlock(BB); }
  inline void write(const Instruction *I)    { processInstruction(I); }
  inline void write(const ConstPoolVal *CPV) { processConstant(CPV);  }

protected:
  virtual bool visitMethod(const Method *M);
  virtual bool processConstPool(const ConstantPool &CP, bool isMethod);
  virtual bool processConstant(const ConstPoolVal *CPV);
  virtual bool processMethod(const Method *M);
  virtual bool processMethodArgument(const MethodArgument *MA);
  virtual bool processBasicBlock(const BasicBlock *BB);
  virtual bool processInstruction(const Instruction *I);

private :
  void writeOperand(const Value *Op, bool PrintType, bool PrintName = true);
};



// visitMethod - This member is called after the above two steps, visting each
// method, because they are effectively values that go into the constant pool.
//
bool AssemblyWriter::visitMethod(const Method *M) {
  return false;
}

bool AssemblyWriter::processConstPool(const ConstantPool &CP, bool isMethod) {
  // Done printing arguments...
  if (isMethod) {
    const MethodType *MT = CP.getParentV()->castMethodAsserting()->getType()->
                                            isMethodType();
    if (MT->isVarArg()) {
      if (MT->getParamTypes().size())
	Out << ", ";
      Out << "...";  // Output varargs portion of signature!
    }
    Out << ")\n";
  }

  ModuleAnalyzer::processConstPool(CP, isMethod);
  
  if (isMethod) {
    if (!CP.getParentV()->castMethodAsserting()->isExternal())
      Out << "begin";
  } else {
    Out << "implementation\n";
  }
  return false;
}


// processConstant - Print out a constant pool entry...
//
bool AssemblyWriter::processConstant(const ConstPoolVal *CPV) {
  if (!CPV->hasName())
    return false;    // Don't print out unnamed constants, they will be inlined

  // Print out name...
  Out << "\t%" << CPV->getName() << " = ";

  // Print out the constant type...
  Out << CPV->getType();

  // Write the value out now...
  writeOperand(CPV, false, false);

  if (!CPV->hasName() && CPV->getType() != Type::VoidTy) {
    int Slot = Table.getValSlot(CPV); // Print out the def slot taken...
    Out << "\t\t; <" << CPV->getType() << ">:";
    if (Slot >= 0) Out << Slot;
    else Out << "<badref>";
  } 

  Out << endl;
  return false;
}

// processMethod - Process all aspects of a method.
//
bool AssemblyWriter::processMethod(const Method *M) {
  // Print out the return type and name...
  Out << "\n" << (M->isExternal() ? "declare " : "") 
      << M->getReturnType() << " \"" << M->getName() << "\"(";
  Table.incorporateMethod(M);
  ModuleAnalyzer::processMethod(M);
  Table.purgeMethod();
  if (!M->isExternal())
    Out << "end\n";
  return false;
}

// processMethodArgument - This member is called for every argument that 
// is passed into the method.  Simply print it out
//
bool AssemblyWriter::processMethodArgument(const MethodArgument *Arg) {
  // Insert commas as we go... the first arg doesn't get a comma
  if (Arg != Arg->getParent()->getArgumentList().front()) Out << ", ";

  // Output type...
  Out << Arg->getType();
  
  // Output name, if available...
  if (Arg->hasName())
    Out << " %" << Arg->getName();
  else if (Table.getValSlot(Arg) < 0)
    Out << "<badref>";
  
  return false;
}

// processBasicBlock - This member is called for each basic block in a methd.
//
bool AssemblyWriter::processBasicBlock(const BasicBlock *BB) {
  if (BB->hasName()) {              // Print out the label if it exists...
    Out << "\n" << BB->getName() << ":";
  } else {
    int Slot = Table.getValSlot(BB);
    Out << "\n; <label>:";
    if (Slot >= 0) 
      Out << Slot;         // Extra newline seperates out label's
    else 
      Out << "<badref>"; 
  }
  Out << "\t\t\t\t\t;[#uses=" << BB->use_size() << "]\n";  // Output # uses

  ModuleAnalyzer::processBasicBlock(BB);
  return false;
}

// processInstruction - This member is called for each Instruction in a methd.
//
bool AssemblyWriter::processInstruction(const Instruction *I) {
  Out << "\t";

  // Print out name if it exists...
  if (I && I->hasName())
    Out << "%" << I->getName() << " = ";

  // Print out the opcode...
  Out << I->getOpcodeName();

  // Print out the type of the operands...
  const Value *Operand = I->getNumOperands() ? I->getOperand(0) : 0;

  // Special case conditional branches to swizzle the condition out to the front
  if (I->getOpcode() == Instruction::Br && I->getNumOperands() > 1) {
    writeOperand(I->getOperand(2), true);
    Out << ",";
    writeOperand(Operand, true);
    Out << ",";
    writeOperand(I->getOperand(1), true);

  } else if (I->getOpcode() == Instruction::Switch) {
    // Special case switch statement to get formatting nice and correct...
    writeOperand(Operand         , true); Out << ",";
    writeOperand(I->getOperand(1), true); Out << " [";

    for (unsigned op = 2, Eop = I->getNumOperands(); op < Eop; op += 2) {
      Out << "\n\t\t";
      writeOperand(I->getOperand(op  ), true); Out << ",";
      writeOperand(I->getOperand(op+1), true);
    }
    Out << "\n\t]";
  } else if (I->isPHINode()) {
    Out << " " << Operand->getType();

    Out << " [";  writeOperand(Operand, false); Out << ",";
    writeOperand(I->getOperand(1), false); Out << " ]";
    for (unsigned op = 2, Eop = I->getNumOperands(); op < Eop; op += 2) {
      Out << ", [";  
      writeOperand(I->getOperand(op  ), false); Out << ",";
      writeOperand(I->getOperand(op+1), false); Out << " ]";
    }
  } else if (I->getOpcode() == Instruction::Ret && !Operand) {
    Out << " void";
  } else if (I->getOpcode() == Instruction::Call) {
    writeOperand(Operand, true);
    Out << "(";
    if (I->getNumOperands() > 1) writeOperand(I->getOperand(1), true);
    for (unsigned op = 2, Eop = I->getNumOperands(); op < Eop; ++op) {
      Out << ",";
      writeOperand(I->getOperand(op), true);
    }

    Out << " )";
  } else if (I->getOpcode() == Instruction::Malloc || 
	     I->getOpcode() == Instruction::Alloca) {
    Out << " " << ((const PointerType*)I->getType())->getValueType();
    if (I->getNumOperands()) {
      Out << ",";
      writeOperand(I->getOperand(0), true);
    }
  } else if (I->getOpcode() == Instruction::Cast) {
    writeOperand(Operand, true);
    Out << " to " << I->getType();
  } else if (Operand) {   // Print the normal way...

    // PrintAllTypes - Instructions who have operands of all the same type 
    // omit the type from all but the first operand.  If the instruction has
    // different type operands (for example br), then they are all printed.
    bool PrintAllTypes = false;
    const Type *TheType = Operand->getType();

    for (unsigned i = 1, E = I->getNumOperands(); i != E; ++i) {
      Operand = I->getOperand(i);
      if (Operand->getType() != TheType) {
	PrintAllTypes = true;       // We have differing types!  Print them all!
	break;
      }
    }

    if (!PrintAllTypes)
      Out << " " << I->getOperand(0)->getType();

    for (unsigned i = 0, E = I->getNumOperands(); i != E; ++i) {
      if (i) Out << ",";
      writeOperand(I->getOperand(i), PrintAllTypes);
    }
  }

  // Print a little comment after the instruction indicating which slot it
  // occupies.
  //
  if (I->getType() != Type::VoidTy) {
    Out << "\t\t; <" << I->getType() << ">";

    if (!I->hasName()) {
      int Slot = Table.getValSlot(I); // Print out the def slot taken...
      if (Slot >= 0) Out << ":" << Slot;
      else Out << ":<badref>";
    }
    Out << "\t[#uses=" << I->use_size() << "]";  // Output # uses
  }
  Out << endl;

  return false;
}


void AssemblyWriter::writeOperand(const Value *Operand, bool PrintType, 
				  bool PrintName) {
  WriteAsOperand(Out, Operand, PrintType, PrintName, &Table);
}


//===----------------------------------------------------------------------===//
//                       External Interface declarations
//===----------------------------------------------------------------------===//



void WriteToAssembly(const Module *M, ostream &o) {
  if (M == 0) { o << "<null> module\n"; return; }
  SlotCalculator SlotTable(M, true);
  AssemblyWriter W(o, SlotTable);

  W.write(M);
}

void WriteToAssembly(const Method *M, ostream &o) {
  if (M == 0) { o << "<null> method\n"; return; }
  SlotCalculator SlotTable(M->getParent(), true);
  AssemblyWriter W(o, SlotTable);

  W.write(M);
}


void WriteToAssembly(const BasicBlock *BB, ostream &o) {
  if (BB == 0) { o << "<null> basic block\n"; return; }

  SlotCalculator SlotTable(BB->getParent(), true);
  AssemblyWriter W(o, SlotTable);

  W.write(BB);
}

void WriteToAssembly(const ConstPoolVal *CPV, ostream &o) {
  if (CPV == 0) { o << "<null> constant pool value\n"; return; }
  WriteAsOperand(o, CPV, true, true, 0);
}

void WriteToAssembly(const Instruction *I, ostream &o) {
  if (I == 0) { o << "<null> instruction\n"; return; }

  SlotCalculator SlotTable(I->getParent() ? I->getParent()->getParent() : 0, 
			   true);
  AssemblyWriter W(o, SlotTable);

  W.write(I);
}
