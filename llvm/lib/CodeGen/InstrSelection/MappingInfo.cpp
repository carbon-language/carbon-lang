//===- MappingInfo.cpp - create LLVM info and output to .s file ---------===//
//
// Create Map from LLVM BB and Instructions and Machine Instructions
// and output the information as .byte directives to the .s file
// Currently Sparc specific but will be extended for others later
//
//===--------------------------------------------------------------------===//

#include "llvm/CodeGen/MappingInfo.h"
#include "llvm/Pass.h"
#include "llvm/Module.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineCodeForBasicBlock.h"
#include "llvm/CodeGen/MachineCodeForInstruction.h"
#include <map>
#include <vector>
using std::vector;


// MappingInfo - This method collects mapping info 
// for the mapping from LLVM to machine code.
//
namespace {
  class getMappingInfoForFunction : public Pass { 
    std::ostream &Out;
  private:
    std::map<const Function*, unsigned> Fkey; //key of function to num
    std::map<const MachineInstr*, unsigned> BBkey; //key basic block to num
    std::map<const MachineInstr*, unsigned> MIkey; //key machine instruction to num
    vector<vector<unsigned> > BBmap;
    vector<vector<unsigned> > MImap;
 
    void createFunctionKey(Module &M);
    void createBasicBlockKey(Module &M);    
    void createMachineInstructionKey(Module &M);
    void createBBToMImap(Module &M);
    void createLLVMToMImap(Module &M);
    unsigned writeNumber(unsigned X);
    
  public:
    getMappingInfoForFunction(std::ostream &out) : Out(out){}

    const char* getPassName() const {
      return "Sparc CollectMappingInfoForInstruction";
    }
    
    bool run(Module &M);
  };
}


//pass definition
Pass *MappingInfoForFunction(std::ostream &out){
  return (new getMappingInfoForFunction(out));
}

//function definitions :
//create and output maps to the .s file
bool getMappingInfoForFunction::run(Module &M) {

  //  Module *M = &m;

  //map for Function to Function number
  createFunctionKey(M);
      
  //map for BB to LLVM instruction number
  createBasicBlockKey(M);
      
  //map from Machine Instruction to Machine Instruction number
  createMachineInstructionKey(M);
      
  //map of Basic Block to first Machine Instruction and number 
  // of instructions go thro each function
  createBBToMImap(M);
  
  //map of LLVM Instruction to Machine Instruction 
  createLLVMToMImap(M);
  
  //unsigned r =0;
  //for (Module::iterator FI = M.begin(), FE = M.end(); 
  //FI != FE; ++FI){
  //unsigned r = 0;
  //  if(FI->isExternal()) continue;
  //for (Function::iterator BI = FI->begin(), BE = FI->end(); 
  // BI != BE; ++BI){
  //r++;
  //}
  //Out <<"#BB in F: "<<r<<"\n";
  //}
  //Out <<"#BB: "<< r <<"\n";
  //Out <<"BBkey.size() "<<BBkey.size()<<"\n";
  //Out <<"BBmap.size() "<<BBmap.size()<<"\n";
  // Write map to the sparc assembly stream
  // Start by writing out the basic block to first and last
  // machine instruction map to the .s file
  Out << "\n\n!BB TO MI MAP\n";
  Out << "\t.section \".data\"\n\t.align 8\n";
  Out << "\t.global BBMIMap\n";
  Out << "BBMIMap:\n";
  //add stream object here that will contain info about the map
  //add object to write this out to the .s file
  //int x=0;
  unsigned sizeBBmap=0;
  unsigned sizeLImap=0;
  for (vector<vector<unsigned> >::iterator BBmapI = 
	 BBmap.begin(), BBmapE = BBmap.end(); BBmapI != BBmapE;
       ++BBmapI){
    sizeBBmap += writeNumber((*BBmapI)[0]);
    sizeBBmap += writeNumber((*BBmapI)[1]);
    sizeBBmap += writeNumber((*BBmapI)[2]);
    sizeBBmap += writeNumber((*BBmapI)[3]);
    //x++;
  }
  //Out <<"sizeOutputed = "<<x<<"\n";
  
  Out << "\t.type BBMIMap,#object\n";
  Out << "\t.size BBMIMap,"<<BBmap.size() << "\n";
  
  //output length info
  Out <<"\n\n!LLVM BB MAP Length\n\t.section \".bbdata";
  Out << "\",#alloc,#write\n\t.global BBMIMap_length\n\t.align 4\n\t.type BBMIMap_length,";
  Out <<"#object\n\t.size BBMIMap_length,4\nBBMIMap_length:\n\t.word "
      << sizeBBmap <<"\n\n\n\n";
 

  //Now write out the LLVM instruction to the corresponding
  //machine instruction map
  Out << "!LLVM I TO MI MAP\n";
  Out << "\t.section\".data\"\n\t.align 8\n";
  Out << "\t.global LMIMap\n";
  Out << "LMIMap:\n";
  //add stream object here that will contain info about the map
  //add object to write this out to the .s file
  for (vector<vector<unsigned> >::iterator MImapI = 
	 MImap.begin(), MImapE = MImap.end(); MImapI != MImapE;
       ++MImapI){
    sizeLImap += writeNumber((*MImapI)[0]);
    sizeLImap += writeNumber((*MImapI)[1]);
    sizeLImap += writeNumber((*MImapI)[2]);
    sizeLImap += writeNumber((*MImapI)[3]);
  }
  Out << "\t.type LMIMap,#object\n";
  Out << "\t.size LMIMap,"<<MImap.size() << "\n";
  //output length info
  Out <<"\n\n!LLVM MI MAP Length\n\t.section\".llvmdata";
  Out << "\",#alloc,#write\n\t.global LMIMap_length\n\t.align 4\n\t.type LMIMap_length,";
  Out <<"#object\n\t.size LMIMap_length,4\nLMIMap_length:\n\t.word "
      << ((MImap.size())*4)<<"\n\n\n\n";

  return false; 
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
void getMappingInfoForFunction::createFunctionKey(Module &M){
  unsigned i = 0;
  unsigned j = 0;
  for (Module::iterator FI = M.begin(), FE = M.end();
       FI != FE; ++FI){
    //dont count F with 0 BBs
    if(FI->isExternal()) continue;
    Fkey[FI] = i;
    ++i;
  }
}
     
//Assign a Number to each BB
void getMappingInfoForFunction::createBasicBlockKey(Module &M){
  //unsigned i = 0;
  for (Module::iterator FI = M.begin(), FE = M.end(); 
       FI != FE; ++FI){
    unsigned i = 0;
    if(FI->isExternal()) continue;
    for (Function::iterator BI = FI->begin(), BE = FI->end(); 
	 BI != BE; ++BI){
      MachineCodeForBasicBlock &miBB = MachineCodeForBasicBlock::get(BI);
      BBkey[miBB[0]] = i;
      i = i+(miBB.size());
    }
  }
}

//Assign a number to each MI wrt beginning of the BB
void getMappingInfoForFunction::createMachineInstructionKey(Module &M){
  for (Module::iterator FI = M.begin(), FE = M.end(); 
       FI != FE; ++FI){
    if(FI->isExternal()) continue;
    for (Function::iterator BI=FI->begin(), BE=FI->end(); 
	 BI != BE; ++BI){
      MachineCodeForBasicBlock &miBB = MachineCodeForBasicBlock::get(BI);
      unsigned j = 0;
      for (MachineCodeForBasicBlock::iterator miI = miBB.begin(),
	     miE = miBB.end(); miI != miE; ++miI, ++j){
	MIkey[*miI] = j;
      }
    }
  }
}

//BBtoMImap: contains F#, BB#, 
//              MI#[wrt beginning of F], #MI in BB
void getMappingInfoForFunction::createBBToMImap(Module &M){

  for (Module::iterator FI = M.begin(), FE = M.end();
       FI != FE; ++FI){	
    if(FI->isExternal())continue;
    unsigned i = 0;
    for (Function::iterator BI = FI->begin(), 
	   BE = FI->end(); BI != BE; ++BI){
      MachineCodeForBasicBlock &miBB = MachineCodeForBasicBlock::get(BI);
     //add record into the map
      BBmap.push_back(vector<unsigned>());
      vector<unsigned> &oneBB = BBmap.back();
      oneBB.reserve(4);

      //add F#
      oneBB.push_back(Fkey[FI]);
      //add BB#
      oneBB.push_back( i );
      //add the MI#[wrt the beginning of F]
      oneBB.push_back( BBkey[ miBB[0] ]);
      //add the # of MI
      oneBB.push_back(miBB.size());
      ++i;

    }
  }
}

//LLVMtoMImap: contains F#, BB#, LLVM#, 
//                           MIs[wrt to beginning of BB] 
void getMappingInfoForFunction::createLLVMToMImap(Module &M){
  
  for (Module::iterator FI = M.begin(), FE = M.end();
       FI != FE; ++FI){
    if(FI->isExternal()) continue;
    unsigned i =0;
    for (Function::iterator BI = FI->begin(),  BE = FI->end(); 
	 BI != BE; ++BI, ++i){
      unsigned j = 0;
      for (BasicBlock::iterator II = BI->begin(), 
	     IE = BI->end(); II != IE; ++II, ++j){
	MachineCodeForInstruction& miI = 
	  MachineCodeForInstruction::get(II);
	//do for each corr. MI
	for (MachineCodeForInstruction::iterator miII = miI.begin(), 
	       miIE = miI.end(); miII != miIE; ++miII){

	  MImap.push_back(vector<unsigned>());
	  vector<unsigned> &oneMI = MImap.back();
	  oneMI.reserve(4);
	  
	  //add F#
	  oneMI.push_back(Fkey[FI]);
	  //add BB#
	  oneMI.push_back(i);
	  //add LLVM Instr#
	  oneMI.push_back(j);
	  //add MI#[wrt to beginning of BB]
	  oneMI.push_back(MIkey[*miII]);
	}
      }
    } 
  }
}


