//===- MappingInfo.cpp - create LLVM info and output to .s file ---------===//
//
// This file contains a FunctionPass called getMappingInfoForFunction,
// which creates two maps: one between LLVM Instructions and MachineInstrs,
// and another between MachineBasicBlocks and MachineInstrs (the "BB TO
// MI MAP").
//
// As a side effect, it outputs this information as .byte directives to
// the assembly file. The output is designed to survive the SPARC assembler,
// in order that the Reoptimizer may read it in from memory later when the
// binary is loaded. Therefore, it may contain some hidden SPARC-architecture
// dependencies. Currently this question is purely theoretical as the
// Reoptimizer works only on the SPARC.
//
//===--------------------------------------------------------------------===//

#include "llvm/Reoptimizer/Mapping/MappingInfo.h"
#include "llvm/Pass.h"
#include "llvm/Module.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineCodeForInstruction.h"
#include <map>
using std::vector;

namespace {
  class getMappingInfoForFunction : public FunctionPass { 
    std::ostream &Out;
  public:
    getMappingInfoForFunction(std::ostream &out) : Out(out){}
    const char* getPassName() const{return "Sparc MappingInformation";}
    bool runOnFunction(Function &FI);
  private:
    std::map<const Function*, unsigned> Fkey; //key of F to num
    std::map<const MachineInstr*, unsigned> BBkey; //key BB to num
    std::map<const MachineInstr*, unsigned> MIkey; //key MI to num
    void writePrologue(const std::string &comment,
                       const std::string &symbolPrefix, unsigned num);
    void writeEpilogue(const std::string &symbolPrefix, unsigned num);

    bool doInitialization(Module &M);
    void create_BB_to_MInumber_Key(Function &FI);    
    void create_MI_to_number_Key(Function &FI);
    void writeBBToMImap(Function &FI, unsigned num);
    void writeLLVMToMImap(Function &FI, unsigned num);
    void writeNumber(unsigned X);
  };
}

/// MappingInfoForFunction -- Static factory method: returns a new
/// getMappingInfoForFunction Pass object, which uses OUT as its
/// output stream for assembly output. 
Pass *MappingInfoForFunction(std::ostream &out){
  return (new getMappingInfoForFunction(out));
}

/// runOnFunction -- Builds up the maps for the given function FI and then
/// writes them out as assembly code to the current output stream OUT.
/// This is an entry point to the pass, called by the PassManager.
bool getMappingInfoForFunction::runOnFunction(Function &FI) {
  // First we build temporary tables used to write out the maps.
  create_BB_to_MInumber_Key(FI);
  create_MI_to_number_Key(FI);
  unsigned num = Fkey[&FI]; // Function number for the current function.

  // Now, write out the maps.
  writeBBToMImap(FI, num);
  writeLLVMToMImap(FI, num);

  return false; 
}  

/// writePrologue -- Output a COMMENT describing the map, then output a
/// global symbol to start the map named by concatenating SYMBOLPREFIX
/// and NUM, then output a word containing the length of the map, to the
/// current output stream Out. This also switches the current section to
/// .rodata in the assembly output.
void getMappingInfoForFunction::writePrologue(const std::string &comment,
                                              const std::string &symbolPrefix,
                                              unsigned num) {
  // Comment:
  Out << "!" << comment << "\n";   
  // Switch sections:
  Out << "\t.section \".rodata\"\n\t.align 8\n";  
  // Global symbol naming the map:
  Out << "\t.global " << symbolPrefix << num << "\n";    
  Out << "\t.type " << symbolPrefix << num << ",#object\n"; 
  Out << symbolPrefix << num << ":\n"; 
  // Length word:
  Out << "\t.word .end_" << symbolPrefix << num << "-"
      << symbolPrefix << num << "\n";
}

/// writeEpilogue -- Outputs a local symbol to end the map named by
/// concatenating SYMBOLPREFIX and NUM, followed by a .size directive that
/// gives the size of the map, to the current output stream Out.
void getMappingInfoForFunction::writeEpilogue(const std::string &symbolPrefix,
                                              unsigned num) {
  // Local symbol ending the map:
  Out << ".end_" << symbolPrefix << num << ":\n";    
  // Size directive:
  Out << "\t.size " << symbolPrefix << num << ", .end_" 
      << symbolPrefix << num << "-" << symbolPrefix 
      << num << "\n\n\n\n";
}

/// outByte -- NOT DONE YET.
void outByte (unsigned char b) {
  Out << "\t.byte " << tmp << "\n";
}


/// writeNumber -- Write out the number X as a sequence of .byte
/// directives to the current output stream Out. This method performs a
/// run-length encoding of the unsigned integers X that are output.
void getMappingInfoForFunction::writeNumber(unsigned X) {
  unsigned i=0;
  do {
    unsigned tmp = X & 127;
    X >>= 7;
    if (X) tmp |= 128;
    outByte (tmp);
    ++i;
  } while(X);
}

/// doInitialization -- Assign a number to each Function, as follows:
/// Functions are numbered starting at 0 at the begin() of each Module.
/// Functions which are External (and thus have 0 basic blocks) are not
/// inserted into the maps, and are not assigned a number.  The side-effect
/// of this method is to fill in Fkey to contain the mapping from Functions
/// to numbers. (This method is called automatically by the PassManager.)
bool getMappingInfoForFunction::doInitialization(Module &M) {
  unsigned i = 0;
  for (Module::iterator FI = M.begin(), FE = M.end(); FI != FE; ++FI) {
    if (FI->isExternal()) continue;
    Fkey[FI] = i;
    ++i;
  }
  return false;
}

/// create_BB_to_MInumber_Key -- Assign a number to each MachineBasicBlock
/// in the given Function, as follows: Numbering starts at zero in each
/// Function. MachineBasicBlocks are numbered from begin() to end()
/// in the Function's corresponding MachineFunction. Each successive
/// MachineBasicBlock increments the numbering by the number of instructions
/// it contains. The side-effect of this method is to fill in the instance
/// variable BBkey with the mapping of MachineBasicBlocks to numbers. BBkey
/// is keyed on MachineInstrs, so each MachineBasicBlock is represented
/// therein by its first MachineInstr.
void getMappingInfoForFunction::create_BB_to_MInumber_Key(Function &FI) {
  unsigned i = 0;
  MachineFunction &MF = MachineFunction::get(&FI);
  for (MachineFunction::iterator BI = MF.begin(), BE = MF.end();
       BI != BE; ++BI) {
    MachineBasicBlock &miBB = *BI;
    BBkey[miBB[0]] = i;
    i = i+(miBB.size());
  }
}

/// create_MI_to_number_Key -- Assign a number to each MachineInstr
/// in the given Function with respect to its enclosing MachineBasicBlock, as
/// follows: Numberings start at 0 in each MachineBasicBlock. MachineInstrs
/// are numbered from begin() to end() in their MachineBasicBlock. Each
/// MachineInstr is numbered, then the numbering is incremented by 1. The
/// side-effect of this method is to fill in the instance variable MIkey
/// with the mapping from MachineInstrs to numbers.
void getMappingInfoForFunction::create_MI_to_number_Key(Function &FI) {
  MachineFunction &MF = MachineFunction::get(&FI);
  for (MachineFunction::iterator BI=MF.begin(), BE=MF.end(); BI != BE; ++BI) {
    MachineBasicBlock &miBB = *BI;
    unsigned j = 0;
    for(MachineBasicBlock::iterator miI=miBB.begin(), miE=miBB.end();
	miI!=miE; ++miI, ++j) {
      MIkey[*miI]=j;
    }
  }
}

/// writeBBToMImap -- Output the BB TO MI MAP for the given function as
/// assembly code to the current output stream. The BB TO MI MAP consists
/// of a three-element tuple for each MachineBasicBlock in a function:
/// first, the index of the MachineBasicBlock in the function; second,
/// the number of the MachineBasicBlock in the function as computed by
/// create_BB_to_MInumber_Key; and third, the number of MachineInstrs in
/// the MachineBasicBlock.
void getMappingInfoForFunction::writeBBToMImap(Function &FI, unsigned num){
  unsigned bb = 0;
  const std::string MapComment = "BB TO MI MAP";
  const std::string MapSymbolPrefix = "BBMIMap";
  writePrologue(MapComment, MapSymbolPrefix, num);
  MachineFunction &MF = MachineFunction::get(&FI);  
  for (MachineFunction::iterator BI = MF.begin(), BE = MF.end();
       BI != BE; ++BI, ++bb) {
    MachineBasicBlock &miBB = *BI;
    writeNumber(bb);
    writeNumber(BBkey[miBB[0]]);
    writeNumber(miBB.size());
  }
  writeEpilogue(MapSymbolPrefix, num);  
}

/// writeLLVMToMImap -- Output the LLVM I TO MI MAP for the given function
/// as assembly code to the current output stream. The LLVM I TO MI MAP
/// consists of a set of information for each BasicBlock in a Function,
/// ordered from begin() to end(). The information for a BasicBlock consists
/// of 1) its (0-based) index in the Function, 2) the number of LLVM
/// Instructions it contains, and 3) information for each Instruction, in
/// sequence from the begin() to the end() of the BasicBlock. The information
/// for an Instruction consists of 1) its (0-based) index in the BasicBlock,
/// 2) the number of MachineInstrs that correspond to that Instruction
/// (as reported by MachineCodeForInstruction), and 3) the MachineInstr
/// number calculated by create_MI_to_number_Key, for each of the
/// MachineInstrs that correspond to that Instruction.
void getMappingInfoForFunction::writeLLVMToMImap(Function &FI, unsigned num) {
  unsigned bb = 0;
  const std::string MapComment = "LLVM I TO MI MAP";
  const std::string MapSymbolPrefix = "LMIMap";
  writePrologue(MapComment, MapSymbolPrefix, num);
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
  writeEpilogue(MapSymbolPrefix, num); 
}
