//===- MappingInfo.cpp - create LLVM info and output to .s file -----------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file contains a FunctionPass called MappingInfoAsmPrinter,
// which creates two maps: one between LLVM Instructions and MachineInstrs
// (the "LLVM I TO MI MAP"), and another between MachineBasicBlocks and
// MachineInstrs (the "BB TO MI MAP").
//
// As a side effect, it outputs this information as .byte directives to
// the assembly file. The output is designed to survive the SPARC assembler,
// in order that the Reoptimizer may read it in from memory later when the
// binary is loaded. Therefore, it may contain some hidden SPARC-architecture
// dependencies. Currently this question is purely theoretical as the
// Reoptimizer works only on the SPARC.
//
// The LLVM I TO MI MAP consists of a set of information for each
// BasicBlock in a Function, ordered from begin() to end(). The information
// for a BasicBlock consists of
//  1) its (0-based) index in the Function,
//  2) the number of LLVM Instructions it contains, and
//  3) information for each Instruction, in sequence from the begin()
//     to the end() of the BasicBlock. The information for an Instruction
//     consists of
//     1) its (0-based) index in the BasicBlock,
//     2) the number of MachineInstrs that correspond to that Instruction
//        (as reported by MachineCodeForInstruction), and
//     3) the MachineInstr number calculated by create_MI_to_number_Key,
//        for each of the MachineInstrs that correspond to that Instruction.
//
// The BB TO MI MAP consists of a three-element tuple for each
// MachineBasicBlock in a function, ordered from begin() to end() of
// its MachineFunction: first, the index of the MachineBasicBlock in the
// function; second, the number of the MachineBasicBlock in the function
// as computed by create_BB_to_MInumber_Key; and third, the number of
// MachineInstrs in the MachineBasicBlock.
//
//===--------------------------------------------------------------------===//

#include "MappingInfo.h"
#include "llvm/Pass.h"
#include "llvm/Module.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineCodeForInstruction.h"
#include "Support/StringExtras.h"

namespace llvm {

namespace {
  class MappingInfoAsmPrinter : public FunctionPass { 
    std::ostream &Out;
  public:
    MappingInfoAsmPrinter(std::ostream &out) : Out(out){}
    const char *getPassName () const { return "Instr. Mapping Info Collector"; }
    bool runOnFunction(Function &FI);
    typedef std::map<const MachineInstr*, unsigned> InstructionKey;
  private:
    MappingInfo *currentOutputMap;
    std::map<Function *, unsigned> Fkey; // Function # for all functions.
    bool doInitialization(Module &M);
    void create_BB_to_MInumber_Key(Function &FI, InstructionKey &key);
    void create_MI_to_number_Key(Function &FI, InstructionKey &key);
    void buildBBMIMap (Function &FI, MappingInfo &Map);
    void buildLMIMap (Function &FI, MappingInfo &Map);
    void writeNumber(unsigned X);
    void selectOutputMap (MappingInfo &m) { currentOutputMap = &m; }
    void outByte (unsigned char b) { currentOutputMap->outByte (b); }
    bool doFinalization (Module &M);
  };
}

/// getMappingInfoAsmPrinterPass - Static factory method: returns a new
/// MappingInfoAsmPrinter Pass object, which uses OUT as its output
/// stream for assembly output.
///
Pass *getMappingInfoAsmPrinterPass(std::ostream &out){
  return (new MappingInfoAsmPrinter(out));
}

/// runOnFunction - Builds up the maps for the given function FI and then
/// writes them out as assembly code to the current output stream OUT.
/// This is an entry point to the pass, called by the PassManager.
///
bool MappingInfoAsmPrinter::runOnFunction(Function &FI) {
  unsigned num = Fkey[&FI]; // Function number for the current function.

  // Create objects to hold the maps.
  MappingInfo LMIMap ("LLVM I TO MI MAP", "LMIMap", num);
  MappingInfo BBMIMap ("BB TO MI MAP", "BBMIMap", num);

  // Now, build the maps.
  buildLMIMap (FI, LMIMap);
  buildBBMIMap (FI, BBMIMap);

  // Now, write out the maps.
  LMIMap.dumpAssembly (Out);
  BBMIMap.dumpAssembly (Out);

  return false; 
}  

/// writeNumber - Write out the number X as a sequence of .byte
/// directives to the current output stream Out. This method performs a
/// run-length encoding of the unsigned integers X that are output.
///
void MappingInfoAsmPrinter::writeNumber(unsigned X) {
  unsigned i=0;
  do {
    unsigned tmp = X & 127;
    X >>= 7;
    if (X) tmp |= 128;
    outByte (tmp);
    ++i;
  } while(X);
}

/// doInitialization - Assign a number to each Function, as follows:
/// Functions are numbered starting at 0 at the begin() of each Module.
/// Functions which are External (and thus have 0 basic blocks) are not
/// inserted into the maps, and are not assigned a number.  The side-effect
/// of this method is to fill in Fkey to contain the mapping from Functions
/// to numbers. (This method is called automatically by the PassManager.)
///
bool MappingInfoAsmPrinter::doInitialization(Module &M) {
  unsigned i = 0;
  for (Module::iterator FI = M.begin(), FE = M.end(); FI != FE; ++FI) {
    if (FI->isExternal()) continue;
    Fkey[FI] = i;
    ++i;
  }
  return false; // Success.
}

/// create_BB_to_MInumber_Key -- Assign a number to each MachineBasicBlock
/// in the given Function, as follows: Numbering starts at zero in each
/// Function. MachineBasicBlocks are numbered from begin() to end()
/// in the Function's corresponding MachineFunction. Each successive
/// MachineBasicBlock increments the numbering by the number of instructions
/// it contains. The side-effect of this method is to fill in the parameter
/// KEY with the mapping of MachineBasicBlocks to numbers. KEY
/// is keyed on MachineInstrs, so each MachineBasicBlock is represented
/// therein by its first MachineInstr.
///
void MappingInfoAsmPrinter::create_BB_to_MInumber_Key(Function &FI,
                                                     InstructionKey &key) {
  unsigned i = 0;
  MachineFunction &MF = MachineFunction::get(&FI);
  for (MachineFunction::iterator BI = MF.begin(), BE = MF.end();
       BI != BE; ++BI) {
    MachineBasicBlock &miBB = *BI;
    key[&miBB.front()] = i;
    i = i+(miBB.size());
  }
}

/// create_MI_to_number_Key - Assign a number to each MachineInstr
/// in the given Function with respect to its enclosing MachineBasicBlock, as
/// follows: Numberings start at 0 in each MachineBasicBlock. MachineInstrs
/// are numbered from begin() to end() in their MachineBasicBlock. Each
/// MachineInstr is numbered, then the numbering is incremented by 1. The
/// side-effect of this method is to fill in the parameter KEY
/// with the mapping from MachineInstrs to numbers.
///
void MappingInfoAsmPrinter::create_MI_to_number_Key(Function &FI,
                                                   InstructionKey &key) {
  MachineFunction &MF = MachineFunction::get(&FI);
  for (MachineFunction::iterator BI=MF.begin(), BE=MF.end(); BI != BE; ++BI) {
    MachineBasicBlock &miBB = *BI;
    unsigned j = 0;
    for(MachineBasicBlock::iterator miI = miBB.begin(), miE = miBB.end();
        miI != miE; ++miI, ++j) {
      key[miI] = j;
    }
  }
}

/// buildBBMIMap - Build the BB TO MI MAP for the function FI,
/// and save it into the parameter MAP.
///
void MappingInfoAsmPrinter::buildBBMIMap(Function &FI, MappingInfo &Map) {
  unsigned bb = 0;

  // First build temporary table used to write out the map.
  InstructionKey BBkey;
  create_BB_to_MInumber_Key(FI, BBkey);

  selectOutputMap (Map);
  MachineFunction &MF = MachineFunction::get(&FI);  
  for (MachineFunction::iterator BI = MF.begin(), BE = MF.end();
       BI != BE; ++BI, ++bb) {
    MachineBasicBlock &miBB = *BI;
    writeNumber(bb);
    writeNumber(BBkey[&miBB.front()]);
    writeNumber(miBB.size());
  }
}

/// buildLMIMap - Build the LLVM I TO MI MAP for the function FI,
/// and save it into the parameter MAP.
///
void MappingInfoAsmPrinter::buildLMIMap(Function &FI, MappingInfo &Map) {
  unsigned bb = 0;
  // First build temporary table used to write out the map.
  InstructionKey MIkey;
  create_MI_to_number_Key(FI, MIkey);

  selectOutputMap (Map);
  for (Function::iterator BI = FI.begin(), BE = FI.end(); 
       BI != BE; ++BI, ++bb) {
    unsigned li = 0;
    writeNumber(bb);
    writeNumber(BI->size());
    for (BasicBlock::iterator II = BI->begin(), IE = BI->end(); II != IE;
         ++II, ++li) {
      MachineCodeForInstruction& miI = MachineCodeForInstruction::get(II);
      writeNumber(li);
      writeNumber(miI.size());
      for (MachineCodeForInstruction::iterator miII = miI.begin(), 
           miIE = miI.end(); miII != miIE; ++miII) {
	     writeNumber(MIkey[*miII]);
      }
    }
  } 
}

void MappingInfo::byteVector::dumpAssembly (std::ostream &Out) {
  for (iterator i = begin (), e = end (); i != e; ++i)
	Out << ".byte " << (int)*i << "\n";
}

static void writePrologue (std::ostream &Out, const std::string &comment,
			   const std::string &symName) {
  // Prologue:
  // Output a comment describing the object.
  Out << "!" << comment << "\n";   
  // Switch the current section to .rodata in the assembly output:
  Out << "\t.section \".rodata\"\n\t.align 8\n";  
  // Output a global symbol naming the object:
  Out << "\t.global " << symName << "\n";    
  Out << "\t.type " << symName << ",#object\n"; 
  Out << symName << ":\n"; 
}

static void writeEpilogue (std::ostream &Out, const std::string &symName) {
  // Epilogue:
  // Output a local symbol marking the end of the object:
  Out << ".end_" << symName << ":\n";    
  // Output size directive giving the size of the object:
  Out << "\t.size " << symName << ", .end_" << symName << "-" << symName
      << "\n";
}

void MappingInfo::dumpAssembly (std::ostream &Out) {
  const std::string &name (symbolPrefix + utostr (functionNumber));
  writePrologue (Out, comment, name);
  // The LMIMap and BBMIMap are supposed to start with a length word:
  Out << "\t.word .end_" << name << "-" << name << "\n";
  bytes.dumpAssembly (Out);
  writeEpilogue (Out, name);
}

/// doFinalization - This method writes out two tables, named
/// FunctionBB and FunctionLI, which map Function numbers (as in
/// doInitialization) to the BBMIMap and LMIMap tables. (This used to
/// be the "FunctionInfo" pass.)
///
bool MappingInfoAsmPrinter::doFinalization (Module &M) {
  unsigned f;
  
  writePrologue(Out, "FUNCTION TO BB MAP", "FunctionBB");
  f=0;
  for(Module::iterator FI = M.begin (), FE = M.end (); FE != FI; ++FI) {
    if (FI->isExternal ())
      continue;
    Out << "\t.xword BBMIMap" << f << "\n";
    ++f;
  }
  writeEpilogue(Out, "FunctionBB");
  
  writePrologue(Out, "FUNCTION TO LI MAP", "FunctionLI");
  f=0;
  for(Module::iterator FI = M.begin (), FE = M.end (); FE != FI; ++FI) {
    if (FI->isExternal ())
      continue;
    Out << "\t.xword LMIMap" << f << "\n";
    ++f;
  }
  writeEpilogue(Out, "FunctionLI");
  
  return false;
}

} // End llvm namespace

