//===- MappingInfo.cpp - create LLVM info and output to .s file ---------===//
//
// Create Map from LLVM BB and Instructions and Machine Instructions
// and output the information as .byte directives to the .s file
// Currently Sparc specific but will be extended for others later
//
//===--------------------------------------------------------------------===//

#include "llvm/Reoptimizer/Mapping/MappingInfo.h"
#include "llvm/Pass.h"
#include "llvm/Module.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineCodeForBasicBlock.h"
#include "llvm/CodeGen/MachineCodeForInstruction.h"
#include <map>
using std::vector;


// MappingInfo - This method collects mapping info 
// for the mapping from LLVM to machine code.
//
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
    
    bool doInitialization(Module &M);
    void create_BB_to_MInumber_Key(Function &FI);    
    void create_MI_to_number_Key(Function &FI);
    void writeBBToMImap(Function &FI);
    void writeLLVMToMImap(Function &FI);
    void getMappingInfoForFunction::writePrologue(const char * area,    
						  const char *label,
						  unsigned FunctionNo);
    void getMappingInfoForFunction::writeEpilogue(const char *area, 
						  const char *label,
						  unsigned FunctionNo);
    unsigned writeNumber(unsigned X);
  };
}


//pass definition
Pass *MappingInfoForFunction(std::ostream &out){
  return (new getMappingInfoForFunction(out));
}

//function definitions :
//create and output maps to the .s file
bool getMappingInfoForFunction::runOnFunction(Function &FI) {
  
  
  //first create reference maps
  //createFunctionKey(M);
  create_BB_to_MInumber_Key(FI);
  create_MI_to_number_Key(FI);
  unsigned FunctionNo = Fkey[&(FI)];

  //now print out the maps
  writePrologue("BB TO MI MAP", "BBMIMap", FunctionNo);
  writeBBToMImap(FI);
  writeEpilogue("BB TO MI MAP", "BBMIMap", FunctionNo);  
  
  writePrologue("LLVM I TO MI MAP", "LMIMap", FunctionNo);
  writeLLVMToMImap(FI);
  writeEpilogue("LLVM I TO MI MAP", "LMIMap", FunctionNo); 
  return false; 
}  

void getMappingInfoForFunction::writePrologue(const char *area,
					      const char *label, 
					      unsigned FunctionNo){
  Out << "!" << area << "\n";   
  Out << "\t.section \".rodata\"\n\t.align 8\n";  
  Out << "\t.global " << label << FunctionNo << "\n";    
  Out << "\t.type " << label << FunctionNo << ",#object\n"; 
  Out << label << FunctionNo << ":\n"; 
  Out << "\t.word .end_" << label << FunctionNo << "-"
      << label << FunctionNo << "\n";
}

void getMappingInfoForFunction::writeEpilogue(const char *area,
					      const char *label,
					      unsigned FunctionNo){
  Out << ".end_" << label << FunctionNo << ":\n";    
  Out << "\t.size " << label << FunctionNo << ", .end_" 
      << label << FunctionNo << "-" << label 
      << FunctionNo << "\n\n\n\n";
}

//write out information as .byte directives
unsigned getMappingInfoForFunction::writeNumber(unsigned X) {
  unsigned i=0;
  do {
    unsigned tmp = X & 127;
    X >>= 7;
    if (X) tmp |= 128;
    Out << "\t.byte " << tmp << "\n";
    ++i;
  } while(X);
  return i;
}

//Assign a number to each Function 
bool getMappingInfoForFunction::doInitialization(Module &M){
  unsigned i = 0;
  for (Module::iterator FI = M.begin(), FE = M.end();
       FI != FE; ++FI){
    //dont count F with 0 BBs
    if(FI->isExternal()) continue;
    Fkey[FI] = i;
    ++i;
  }
  return false;
}
     
//Assign a Number to each BB
void getMappingInfoForFunction::create_BB_to_MInumber_Key(Function &FI){
  unsigned i = 0;
  for (Function::iterator BI = FI.begin(), BE = FI.end(); 
       BI != BE; ++BI){
    MachineCodeForBasicBlock &miBB = MachineCodeForBasicBlock::get(BI);
    BBkey[miBB[0]] = i;
    i = i+(miBB.size());
  }
}

//Assign a number to each MI wrt beginning of the BB
void getMappingInfoForFunction::create_MI_to_number_Key(Function &FI){
  for (Function::iterator BI=FI.begin(), BE=FI.end(); 
       BI != BE; ++BI){
    MachineCodeForBasicBlock &miBB = MachineCodeForBasicBlock::get(BI);
    unsigned j = 0;
    for(MachineCodeForBasicBlock::iterator miI=miBB.begin(), miE=miBB.end();
	miI!=miE; ++miI, ++j){
      MIkey[*miI]=j;
    }
  }
}

//BBtoMImap: contains F#, BB#, 
//              MI#[wrt beginning of F], #MI in BB
void getMappingInfoForFunction::writeBBToMImap(Function &FI){
  unsigned bb=0;
  for (Function::iterator BI = FI.begin(), 
	 BE = FI.end(); BI != BE; ++BI, ++bb){
    MachineCodeForBasicBlock &miBB = MachineCodeForBasicBlock::get(BI);
    writeNumber(bb);
    //Out << " BB: "<<(void *)BI<<"\n";
    //for(int i=0; i<miBB.size(); ++i)
    //Out<<*miBB[i]<<"\n";
    writeNumber( BBkey[ miBB[0] ]);
    writeNumber(miBB.size());
  }
}

//LLVMtoMImap: contains F#, BB#, LLVM#, 
//                           MIs[wrt to beginning of BB] 
void getMappingInfoForFunction::writeLLVMToMImap(Function &FI){

  unsigned bb =0;
  for (Function::iterator BI = FI.begin(),  BE = FI.end(); 
       BI != BE; ++BI, ++bb){
    unsigned li = 0;
    writeNumber(bb);
    //std::cerr<<"BasicBlockNumber= "<<bb<<"\n";

    //Out << "BB: "<<(void *)BI<<"\n";
    writeNumber(BI->size());
    //std::cerr<<"BasicBlockSize  = "<<BI->size()<<"\n";

    for (BasicBlock::iterator II = BI->begin(), 
	   IE = BI->end(); II != IE; ++II, ++li){
    //Out << "I: "<<*II<<"\n";
      MachineCodeForInstruction& miI = 
	MachineCodeForInstruction::get(II);
      
      //do for each corr. MI
      writeNumber(li);
      //std::cerr<<"InstructionNumber= "<<li<<"\n";

      writeNumber(miI.size());
      //std::cerr<<"InstructionSize  = "<<miI.size()<<"\n";
   
      for (MachineCodeForInstruction::iterator miII = miI.begin(), 
	     miIE = miI.end(); miII != miIE; ++miII){
	//Out << "MI: "<<**miII<<"\n";
	writeNumber(MIkey[*miII]);
        //std::cerr<<"MachineInstruction= "<<MIkey[*miII]<<"\n";
      }
    }
  } 
}
