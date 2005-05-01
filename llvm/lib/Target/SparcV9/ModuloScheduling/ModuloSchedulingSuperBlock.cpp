//===-- ModuloSchedulingSuperBlock.cpp - ModuloScheduling--------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This ModuloScheduling pass is based on the Swing Modulo Scheduling
//  algorithm, but has been extended to support SuperBlocks (multiple
//  basic block, single entry, multipl exit loops).
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "ModuloSchedSB"

#include "DependenceAnalyzer.h"
#include "ModuloSchedulingSuperBlock.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/GraphWriter.h"
#include "llvm/Instructions.h"
#include "../MachineCodeForInstruction.h"
#include "../SparcV9RegisterInfo.h"
#include "../SparcV9Internals.h"
#include "../SparcV9TmpInstr.h"
#include <fstream>
#include <sstream>

using namespace llvm;
/// Create ModuloSchedulingSBPass
///
FunctionPass *llvm::createModuloSchedulingSBPass(TargetMachine & targ) {
  DEBUG(std::cerr << "Created ModuloSchedulingSBPass\n");
  return new ModuloSchedulingSBPass(targ);
}

//Graph Traits for printing out the dependence graph
template<typename GraphType>
static void WriteGraphToFile(std::ostream &O, const std::string &GraphName,
                             const GraphType &GT) {
  std::string Filename = GraphName + ".dot";
  O << "Writing '" << Filename << "'...";
  std::ofstream F(Filename.c_str());

  if (F.good())
    WriteGraph(F, GT);
  else
    O << "  error opening file for writing!";
  O << "\n";
};

namespace llvm {
  Statistic<> NumLoops("moduloschedSB-numLoops", "Total Number of Loops");
  Statistic<> NumSB("moduloschedSB-numSuperBlocks", "Total Number of SuperBlocks");
  Statistic<> BBWithCalls("modulosched-BBCalls", "Basic Blocks rejected due to calls");
  Statistic<> BBWithCondMov("modulosched-loopCondMov", "Basic Blocks rejected due to conditional moves");
  bool ModuloSchedulingSBPass::runOnFunction(Function &F) {
    bool Changed = false;
    
    //Get MachineFunction
    MachineFunction &MF = MachineFunction::get(&F);

    //Get Loop Info & Dependence Anaysis info
    LoopInfo &LI = getAnalysis<LoopInfo>();
    DependenceAnalyzer &DA = getAnalysis<DependenceAnalyzer>();

    //Worklist of superblocks
    std::vector<std::vector<const MachineBasicBlock*> > Worklist;
    FindSuperBlocks(F, LI, Worklist);
 
    DEBUG(if(Worklist.size() == 0) std::cerr << "No superblocks in function to ModuloSchedule\n");
    
    //Loop over worklist and ModuloSchedule each SuperBlock
    for(std::vector<std::vector<const MachineBasicBlock*> >::iterator SB = Worklist.begin(),
	  SBE = Worklist.end(); SB != SBE; ++SB) {
      
      //Print out Superblock
      DEBUG(std::cerr << "ModuloScheduling SB: \n";
	    for(std::vector<const MachineBasicBlock*>::const_iterator BI = SB->begin(), 
		  BE = SB->end(); BI != BE; ++BI) {
	      (*BI)->print(std::cerr);});
      
      if(!CreateDefMap(*SB)) {
	defaultInst = 0;
	defMap.clear();
	continue;
      }

      MSchedGraph *MSG = new MSchedGraph(*SB, target, indVarInstrs[*SB], DA, 
					 machineTollvm[*SB]);

      //Write Graph out to file
      DEBUG(WriteGraphToFile(std::cerr, F.getName(), MSG));
      DEBUG(MSG->print(std::cerr));

    }
    return Changed;
    
  }

  void ModuloSchedulingSBPass::FindSuperBlocks(Function &F, LoopInfo &LI,
		      std::vector<std::vector<const MachineBasicBlock*> > &Worklist) {

    //Get MachineFunction
    MachineFunction &MF = MachineFunction::get(&F);
    
    //Map of LLVM BB to machine BB
    std::map<BasicBlock*, MachineBasicBlock*> bbMap;

    for (MachineFunction::iterator BI = MF.begin(); BI != MF.end(); ++BI) {
      BasicBlock *llvmBB = (BasicBlock*) BI->getBasicBlock();
      assert(!bbMap.count(llvmBB) && "LLVM BB already in map!");
      bbMap[llvmBB] = &*BI;
    }

    //Iterate over the loops, and find super blocks
    for(LoopInfo::iterator LB = LI.begin(), LE = LI.end(); LB != LE; ++LB) {
      Loop *L = *LB;
      ++NumLoops;

      //If loop is not single entry, try the next one
      if(!L->getLoopPreheader())
	continue;
    
      //Check size of this loop, we don't want SBB loops
      if(L->getBlocks().size() == 1)
	continue;
      
      //Check if this loop contains no sub loops
      if(L->getSubLoops().size() == 0) {
	
	std::vector<const MachineBasicBlock*> superBlock;
	
	//Get Loop Headers
	BasicBlock *header = L->getHeader();

	//Follow the header and make sure each BB only has one entry and is valid
	BasicBlock *current = header;
	assert(bbMap.count(current) && "LLVM BB must have corresponding Machine BB\n");
	MachineBasicBlock *currentMBB = bbMap[header];
	bool done = false;
	bool success = true;

	while(!done) {
	  
	  if(MachineBBisValid(currentMBB)) {

	    //Loop over successors of this BB, they should be in the loop block
	    //and be valid
	    BasicBlock *next = 0;
	    for(succ_iterator I = succ_begin(current), E = succ_end(current);
		I != E; ++I) {
	      if(L->contains(*I)) {
		if(!next) 
		  next = *I;
		else {
		  done = true;
		  success = false;
		  break;
		}
	      }
	    }
	    if(success) {
	      superBlock.push_back(currentMBB);
	      if(next == header)
		done = true;
	      else if(!next->getSinglePredecessor()) {
		done = true;
		success = false;
	      }
	      else {
		//Check that the next BB only has one entry
		current = next;
		assert(bbMap.count(current) && "LLVM BB must have corresponding Machine BB");
		currentMBB = bbMap[current];
	      }
	    }
	  }
	  else {
	    done = true;
	    success = false;
	  }
	}

	if(success) {
	  ++NumSB;
	  Worklist.push_back(superBlock);
	}

      }

    }
  }
  
  /// This function checks if a Machine Basic Block is valid for modulo
  /// scheduling. This means that it has no control flow (if/else or
  /// calls) in the block.  Currently ModuloScheduling only works on
  /// single basic block loops.
  bool ModuloSchedulingSBPass::MachineBBisValid(const MachineBasicBlock *BI) {
    
    //Check size of our basic block.. make sure we have more then just the terminator in it
    if(BI->getBasicBlock()->size() == 1)
      return false;
    
    //Get Target machine instruction info
    const TargetInstrInfo *TMI = target.getInstrInfo();

    //Check each instruction and look for calls, keep map to get index later
    std::map<const MachineInstr*, unsigned> indexMap;

    unsigned count = 0;
    for(MachineBasicBlock::const_iterator I = BI->begin(), E = BI->end(); I != E; ++I) {
      //Get opcode to check instruction type
      MachineOpCode OC = I->getOpcode();

      //Look for calls
      if(TMI->isCall(OC)) {
	++BBWithCalls;
	return false;
      }
    
      //Look for conditional move
      if(OC == V9::MOVRZr || OC == V9::MOVRZi || OC == V9::MOVRLEZr || OC == V9::MOVRLEZi
	 || OC == V9::MOVRLZr || OC == V9::MOVRLZi || OC == V9::MOVRNZr || OC == V9::MOVRNZi
	 || OC == V9::MOVRGZr || OC == V9::MOVRGZi || OC == V9::MOVRGEZr
	 || OC == V9::MOVRGEZi || OC == V9::MOVLEr || OC == V9::MOVLEi || OC == V9::MOVLEUr
	 || OC == V9::MOVLEUi || OC == V9::MOVFLEr || OC == V9::MOVFLEi
	 || OC == V9::MOVNEr || OC == V9::MOVNEi || OC == V9::MOVNEGr || OC == V9::MOVNEGi
	 || OC == V9::MOVFNEr || OC == V9::MOVFNEi) {
	++BBWithCondMov;
	return false;
      }

      indexMap[I] = count;

      if(TMI->isNop(OC))
	continue;

      ++count;
    }

    return true;
  }
}

bool ModuloSchedulingSBPass::CreateDefMap(std::vector<const MachineBasicBlock*> &SB) {
  defaultInst = 0;

  for(std::vector<const MachineBasicBlock*>::iterator BI = SB.begin(), 
	BE = SB.end(); BI != BE; ++BI) {

    for(MachineBasicBlock::const_iterator I = (*BI)->begin(), E = (*BI)->end(); I != E; ++I) {
      for(unsigned opNum = 0; opNum < I->getNumOperands(); ++opNum) {
	const MachineOperand &mOp = I->getOperand(opNum);
	if(mOp.getType() == MachineOperand::MO_VirtualRegister && mOp.isDef()) {
	  
	  //assert if this is the second def we have seen
	  assert(!defMap.count(mOp.getVRegValue()) && "Def already in the map");
	  defMap[mOp.getVRegValue()] = (MachineInstr*) &*I;
	}
	
	//See if we can use this Value* as our defaultInst
	if(!defaultInst && mOp.getType() == MachineOperand::MO_VirtualRegister) {
	  Value *V = mOp.getVRegValue();
	  if(!isa<TmpInstruction>(V) && !isa<Argument>(V) && !isa<Constant>(V) && !isa<PHINode>(V))
	    defaultInst = (Instruction*) V;
	}
      }
    }
  }
    
  if(!defaultInst)
    return false;
  
  return true;

}
