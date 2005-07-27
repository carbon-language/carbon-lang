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
#include "llvm/Constants.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/GraphWriter.h"
#include "llvm/Support/Timer.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/Instructions.h"
#include "../MachineCodeForInstruction.h"
#include "../SparcV9RegisterInfo.h"
#include "../SparcV9Internals.h"
#include "../SparcV9TmpInstr.h"
#include <fstream>
#include <sstream>
#include <cmath>
#include <utility>

using namespace llvm;
/// Create ModuloSchedulingSBPass
///
FunctionPass *llvm::createModuloSchedulingSBPass(TargetMachine & targ) {
  DEBUG(std::cerr << "Created ModuloSchedulingSBPass\n");
  return new ModuloSchedulingSBPass(targ);
}


#if 1
#define TIME_REGION(VARNAME, DESC) \
   NamedRegionTimer VARNAME(DESC)
#else
#define TIME_REGION(VARNAME, DESC)
#endif


//Graph Traits for printing out the dependence graph
template<typename GraphType>
static void WriteGraphToFileSB(std::ostream &O, const std::string &GraphName,
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
  Statistic<> BBWithCondMov("modulosched-loopCondMov", 
                            "Basic Blocks rejected due to conditional moves");
  Statistic<> SBResourceConstraint("modulosched-resourceConstraint", 
                                 "Loops constrained by resources");
  Statistic<> SBRecurrenceConstraint("modulosched-recurrenceConstraint", 
                                   "Loops constrained by recurrences");
  Statistic<> SBFinalIISum("modulosched-finalIISum", "Sum of all final II");
  Statistic<> SBIISum("modulosched-IISum", "Sum of all theoretical II");
  Statistic<> SBMSLoops("modulosched-schedLoops", "Number of loops successfully modulo-scheduled");
  Statistic<> SBNoSched("modulosched-noSched", "No schedule");
  Statistic<> SBSameStage("modulosched-sameStage", "Max stage is 0");
  Statistic<> SBBLoops("modulosched-SBBLoops", "Number single basic block loops");
  Statistic<> SBInvalid("modulosched-SBInvalid", "Number invalid superblock loops");
  Statistic<> SBValid("modulosched-SBValid", "Number valid superblock loops");
 Statistic<> SBSize("modulosched-SBSize", "Total size of all valid superblocks");

  template<>
  struct DOTGraphTraits<MSchedGraphSB*> : public DefaultDOTGraphTraits {
    static std::string getGraphName(MSchedGraphSB *F) {
      return "Dependence Graph";
    }

    static std::string getNodeLabel(MSchedGraphSBNode *Node, MSchedGraphSB *Graph) {
      if(!Node->isPredicate()) {
        if (Node->getInst()) {
          std::stringstream ss;
          ss << *(Node->getInst());
          return ss.str(); //((MachineInstr*)Node->getInst());
        }
        else
          return "No Inst";
      }
      else
        return "Pred Node";
    }
    static std::string getEdgeSourceLabel(MSchedGraphSBNode *Node,
                                          MSchedGraphSBNode::succ_iterator I) {
      //Label each edge with the type of dependence
      std::string edgelabel = "";
      switch (I.getEdge().getDepOrderType()) {
        
      case MSchedGraphSBEdge::TrueDep:
        edgelabel = "True";
        break;

      case MSchedGraphSBEdge::AntiDep:
        edgelabel =  "Anti";
        break;
        
      case MSchedGraphSBEdge::OutputDep:
        edgelabel = "Output";
        break;
        
      case MSchedGraphSBEdge::NonDataDep:
        edgelabel = "Pred";
        break;

      default:
        edgelabel = "Unknown";
        break;
      }

      //FIXME
      int iteDiff = I.getEdge().getIteDiff();
      std::string intStr = "(IteDiff: ";
      intStr += itostr(iteDiff);

      intStr += ")";
      edgelabel += intStr;

      return edgelabel;
    }
  };

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

      MSchedGraphSB *MSG = new MSchedGraphSB(*SB, target, indVarInstrs[*SB], DA, 
                                         machineTollvm[*SB]);

      //Write Graph out to file
      DEBUG(WriteGraphToFileSB(std::cerr, F.getName(), MSG));
      
      //Calculate Resource II
      int ResMII = calculateResMII(*SB);

      //Calculate Recurrence II
      int RecMII = calculateRecMII(MSG, ResMII);
      
      DEBUG(std::cerr << "Number of reccurrences found: " << recurrenceList.size() << "\n");
      
      //Our starting initiation interval is the maximum of RecMII and ResMII
      if(RecMII < ResMII)
        ++SBRecurrenceConstraint;
      else
        ++SBResourceConstraint;
      
      II = std::max(RecMII, ResMII);
      int mII = II;
    
      
      //Print out II, RecMII, and ResMII
      DEBUG(std::cerr << "II starts out as " << II << " ( RecMII=" << RecMII << " and ResMII=" << ResMII << ")\n");
     
      //Calculate Node Properties
      calculateNodeAttributes(MSG, ResMII);
      
      //Dump node properties if in debug mode
      DEBUG(for(std::map<MSchedGraphSBNode*, MSNodeSBAttributes>::iterator I =  nodeToAttributesMap.begin(),
                  E = nodeToAttributesMap.end(); I !=E; ++I) {
              std::cerr << "Node: " << *(I->first) << " ASAP: " << I->second.ASAP << " ALAP: "
                        << I->second.ALAP << " MOB: " << I->second.MOB << " Depth: " << I->second.depth
                        << " Height: " << I->second.height << "\n";
            });
      

      //Put nodes in order to schedule them
      computePartialOrder();
 
      //Dump out partial order
      DEBUG(for(std::vector<std::set<MSchedGraphSBNode*> >::iterator I = partialOrder.begin(),
                  E = partialOrder.end(); I !=E; ++I) {
              std::cerr << "Start set in PO\n";
              for(std::set<MSchedGraphSBNode*>::iterator J = I->begin(), JE = I->end(); J != JE; ++J)
                std::cerr << "PO:" << **J << "\n";
            });

      //Place nodes in final order
      orderNodes();
      
      //Dump out order of nodes
      DEBUG(for(std::vector<MSchedGraphSBNode*>::iterator I = FinalNodeOrder.begin(), E = FinalNodeOrder.end(); I != E; ++I) {
              std::cerr << "FO:" << **I << "\n";
            });
      

      //Finally schedule nodes
      bool haveSched = computeSchedule(*SB, MSG);
      
      //Print out final schedule
      DEBUG(schedule.print(std::cerr));
      
      //Final scheduling step is to reconstruct the loop only if we actual have
      //stage > 0
      if(haveSched) {
        //schedule.printSchedule(std::cerr);
        reconstructLoop(*SB);
        ++SBMSLoops;
        //Changed = true;
        SBIISum += mII;
        SBFinalIISum += II;
        
      if(schedule.getMaxStage() == 0)
        ++SBSameStage;
      }
      else
        ++SBNoSched;
      
      //Clear out our maps for the next basic block that is processed
      nodeToAttributesMap.clear();
      partialOrder.clear();
      recurrenceList.clear();
      FinalNodeOrder.clear();
      schedule.clear();
      defMap.clear();
      
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
        unsigned offset = 0;
        std::map<const MachineInstr*, unsigned> indexMap;

        while(!done) {
          //Loop over successors of this BB, they should be in the
          //loop block and be valid
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


          


        if(success) {
          ++NumSB;

          //Loop over all the blocks in the superblock
          for(std::vector<const MachineBasicBlock*>::iterator currentMBB = superBlock.begin(), MBBEnd = superBlock.end(); currentMBB != MBBEnd; ++currentMBB) {
            if(!MachineBBisValid(*currentMBB, indexMap, offset)) {
              success = false;
              break;
            }
          }
        }
        
        if(success) {
          if(getIndVar(superBlock, bbMap, indexMap)) {
            ++SBValid;
            Worklist.push_back(superBlock);
            SBSize += superBlock.size();
          }
          else
            ++SBInvalid;
        }
      }
    }
  }
  
  
  bool ModuloSchedulingSBPass::getIndVar(std::vector<const MachineBasicBlock*> &superBlock, std::map<BasicBlock*, MachineBasicBlock*> &bbMap, 
                                  std::map<const MachineInstr*, unsigned> &indexMap) {
    //See if we can get induction var instructions
    std::set<const BasicBlock*> llvmSuperBlock;

    for(unsigned i =0; i < superBlock.size(); ++i)
      llvmSuperBlock.insert(superBlock[i]->getBasicBlock());

    //Get Target machine instruction info
    const TargetInstrInfo *TMI = target.getInstrInfo();
    
    //Get the loop back branch
    BranchInst *b = dyn_cast<BranchInst>(((BasicBlock*) (superBlock[superBlock.size()-1])->getBasicBlock())->getTerminator());
     std::set<Instruction*> indVar;

    if(b->isConditional()) {
      //Get the condition for the branch 
      Value *cond = b->getCondition();
    
      DEBUG(std::cerr << "Condition: " << *cond << "\n");
    
      //List of instructions associated with induction variable
      std::vector<Instruction*> stack;
    
      //Add branch
      indVar.insert(b);
    
      if(Instruction *I = dyn_cast<Instruction>(cond))
        if(bbMap.count(I->getParent())) {
          if (!assocIndVar(I, indVar, stack, bbMap, superBlock[(superBlock.size()-1)]->getBasicBlock(), llvmSuperBlock))
            return false;
        }
        else
          return false;
      else
        return false;
    }
    else {
      indVar.insert(b);
    }

    //Dump out instructions associate with indvar for debug reasons
    DEBUG(for(std::set<Instruction*>::iterator N = indVar.begin(), NE = indVar.end(); 
              N != NE; ++N) {
            std::cerr << **N << "\n";
          });
    
    //Create map of machine instr to llvm instr
    std::map<MachineInstr*, Instruction*> mllvm;
    for(std::vector<const MachineBasicBlock*>::iterator MBB = superBlock.begin(), MBE = superBlock.end(); MBB != MBE; ++MBB) {
      BasicBlock *BB = (BasicBlock*) (*MBB)->getBasicBlock();
      for(BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; ++I) {
        MachineCodeForInstruction & tempMvec = MachineCodeForInstruction::get(I);
        for (unsigned j = 0; j < tempMvec.size(); j++) {
          mllvm[tempMvec[j]] = I;
        }
      }
    }

      //Convert list of LLVM Instructions to list of Machine instructions
      std::map<const MachineInstr*, unsigned> mIndVar;
      for(std::set<Instruction*>::iterator N = indVar.begin(), 
            NE = indVar.end(); N != NE; ++N) {
              
        //If we have a load, we can't handle this loop because
        //there is no way to preserve dependences between loads
        //and stores
        if(isa<LoadInst>(*N))
          return false;

        MachineCodeForInstruction & tempMvec = MachineCodeForInstruction::get(*N);
        for (unsigned j = 0; j < tempMvec.size(); j++) {
          MachineOpCode OC = (tempMvec[j])->getOpcode();
          if(TMI->isNop(OC))
            continue;
          if(!indexMap.count(tempMvec[j]))
            continue;
          mIndVar[(MachineInstr*) tempMvec[j]] = indexMap[(MachineInstr*) tempMvec[j]];
          DEBUG(std::cerr << *(tempMvec[j]) << " at index " << indexMap[(MachineInstr*) tempMvec[j]] << "\n");
        }
      }
      
      //Put into a map for future access
      indVarInstrs[superBlock] = mIndVar;
      machineTollvm[superBlock] = mllvm;
      
      return true;
      
  }

  bool ModuloSchedulingSBPass::assocIndVar(Instruction *I, 
                                           std::set<Instruction*> &indVar,
                                           std::vector<Instruction*> &stack, 
                                       std::map<BasicBlock*, MachineBasicBlock*> &bbMap, 
                                           const BasicBlock *last, std::set<const BasicBlock*> &llvmSuperBlock) {

    stack.push_back(I);
    
    //If this is a phi node, check if its the canonical indvar
    if(PHINode *PN = dyn_cast<PHINode>(I)) {
      if(llvmSuperBlock.count(PN->getParent())) {
        if (Instruction *Inc =
            dyn_cast<Instruction>(PN->getIncomingValueForBlock(last)))
          if (Inc->getOpcode() == Instruction::Add && Inc->getOperand(0) == PN)
            if (ConstantInt *CI = dyn_cast<ConstantInt>(Inc->getOperand(1)))
              if (CI->equalsInt(1)) {
                //We have found the indvar, so add the stack, and inc instruction to the set
                indVar.insert(stack.begin(), stack.end());
                indVar.insert(Inc);
                stack.pop_back();
                return true;
              }
        return false;
      }
    }
    else {
      //Loop over each of the instructions operands, check if they are an instruction and in this BB
      for(unsigned i = 0; i < I->getNumOperands(); ++i) {
        if(Instruction *N =  dyn_cast<Instruction>(I->getOperand(i))) {
          if(bbMap.count(N->getParent()))
            if(!assocIndVar(N, indVar, stack, bbMap, last, llvmSuperBlock))
              return false;
        }
      }
    }
    
    stack.pop_back();
    return true;
  }
  

  /// This function checks if a Machine Basic Block is valid for modulo
  /// scheduling. This means that it has no control flow (if/else or
  /// calls) in the block.  Currently ModuloScheduling only works on
  /// single basic block loops.
  bool ModuloSchedulingSBPass::MachineBBisValid(const MachineBasicBlock *BI,     
                        std::map<const MachineInstr*, unsigned> &indexMap, 
                                                unsigned &offset) {
    
    //Check size of our basic block.. make sure we have more then just the terminator in it
    if(BI->getBasicBlock()->size() == 1)
      return false;
    
    //Get Target machine instruction info
    const TargetInstrInfo *TMI = target.getInstrInfo();

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

      indexMap[I] = count + offset;

      if(TMI->isNop(OC))
        continue;

      ++count;
    }

    offset += count;

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
          Value *V = mOp.getVRegValue();
          //assert if this is the second def we have seen
          if(defMap.count(V) && isa<PHINode>(V))
            DEBUG(std::cerr << "FIXME: Dup def for phi!\n");
          else {
            //assert(!defMap.count(V) && "Def already in the map");
            if(defMap.count(V))
              return false;
            defMap[V] = (MachineInstr*) &*I;
          }
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


//ResMII is calculated by determining the usage count for each resource
//and using the maximum.
//FIXME: In future there should be a way to get alternative resources
//for each instruction
int ModuloSchedulingSBPass::calculateResMII(std::vector<const MachineBasicBlock*> &superBlock) {

  TIME_REGION(X, "calculateResMII");

  const TargetInstrInfo *mii = target.getInstrInfo();
  const TargetSchedInfo *msi = target.getSchedInfo();

  int ResMII = 0;

  //Map to keep track of usage count of each resource
  std::map<unsigned, unsigned> resourceUsageCount;

  for(std::vector<const MachineBasicBlock*>::iterator BI = superBlock.begin(), BE = superBlock.end(); BI != BE; ++BI) {
    for(MachineBasicBlock::const_iterator I = (*BI)->begin(), E = (*BI)->end(); I != E; ++I) {

      //Get resource usage for this instruction
      InstrRUsage rUsage = msi->getInstrRUsage(I->getOpcode());
      std::vector<std::vector<resourceId_t> > resources = rUsage.resourcesByCycle;

      //Loop over resources in each cycle and increments their usage count
      for(unsigned i=0; i < resources.size(); ++i)
        for(unsigned j=0; j < resources[i].size(); ++j) {
          if(!resourceUsageCount.count(resources[i][j])) {
            resourceUsageCount[resources[i][j]] = 1;
          }
          else {
            resourceUsageCount[resources[i][j]] =  resourceUsageCount[resources[i][j]] + 1;
          }
        }
    }
  }

  //Find maximum usage count

  //Get max number of instructions that can be issued at once. (FIXME)
  int issueSlots = msi->maxNumIssueTotal;

  for(std::map<unsigned,unsigned>::iterator RB = resourceUsageCount.begin(), RE = resourceUsageCount.end(); RB != RE; ++RB) {

    //Get the total number of the resources in our cpu
    int resourceNum = CPUResource::getCPUResource(RB->first)->maxNumUsers;

    //Get total usage count for this resources
    unsigned usageCount = RB->second;

    //Divide the usage count by either the max number we can issue or the number of
    //resources (whichever is its upper bound)
    double finalUsageCount;
    DEBUG(std::cerr << "Resource Num: " << RB->first << " Usage: " << usageCount << " TotalNum: " << resourceNum << "\n");

    if( resourceNum <= issueSlots)
      finalUsageCount = ceil(1.0 * usageCount / resourceNum);
    else
      finalUsageCount = ceil(1.0 * usageCount / issueSlots);


    //Only keep track of the max
    ResMII = std::max( (int) finalUsageCount, ResMII);

  }

  return ResMII;
  
}

/// calculateRecMII - Calculates the value of the highest recurrence
/// By value we mean the total latency/distance
int ModuloSchedulingSBPass::calculateRecMII(MSchedGraphSB *graph, int MII) {
  
  TIME_REGION(X, "calculateRecMII");
  
  findAllCircuits(graph, MII);
  int RecMII = 0;
  
  for(std::set<std::pair<int, std::vector<MSchedGraphSBNode*> > >::iterator I = recurrenceList.begin(), E=recurrenceList.end(); I !=E; ++I) {
    RecMII = std::max(RecMII, I->first);
  }
  
  return MII;
}

int CircCountSB;

void ModuloSchedulingSBPass::unblock(MSchedGraphSBNode *u, std::set<MSchedGraphSBNode*> &blocked,
             std::map<MSchedGraphSBNode*, std::set<MSchedGraphSBNode*> > &B) {

  //Unblock u
  DEBUG(std::cerr << "Unblocking: " << *u << "\n");
  blocked.erase(u);

  //std::set<MSchedGraphSBNode*> toErase;
  while (!B[u].empty()) {
    MSchedGraphSBNode *W = *B[u].begin();
    B[u].erase(W);
    //toErase.insert(*W);
    DEBUG(std::cerr << "Removed: " << *W << "from B-List\n");
    if(blocked.count(W))
      unblock(W, blocked, B);
  }

}

void ModuloSchedulingSBPass::addSCC(std::vector<MSchedGraphSBNode*> &SCC, std::map<MSchedGraphSBNode*, MSchedGraphSBNode*> &newNodes) {

  int totalDelay = 0;
  int totalDistance = 0;
  std::vector<MSchedGraphSBNode*> recc;
  MSchedGraphSBNode *start = 0;
  MSchedGraphSBNode *end = 0;

  //Loop over recurrence, get delay and distance
  for(std::vector<MSchedGraphSBNode*>::iterator N = SCC.begin(), NE = SCC.end(); N != NE; ++N) {
    DEBUG(std::cerr << **N << "\n");
    totalDelay += (*N)->getLatency();
    
    for(unsigned i = 0; i < (*N)->succ_size(); ++i) {
      MSchedGraphSBEdge *edge = (*N)->getSuccessor(i);
      if(find(SCC.begin(), SCC.end(), edge->getDest()) != SCC.end()) {
        totalDistance += edge->getIteDiff();
        if(edge->getIteDiff() > 0)
          if(!start && !end) {
            start = *N;
            end = edge->getDest();
          }
            
      }
    }


    //Get the original node
    recc.push_back(newNodes[*N]);


  }

  DEBUG(std::cerr << "End Recc\n");
  

  assert( (start && end) && "Must have start and end node to ignore edge for SCC");

  if(start && end) {    
    //Insert reccurrence into the list
    DEBUG(std::cerr << "Ignore Edge from!!: " << *start << " to " << *end << "\n");
    edgesToIgnore.insert(std::make_pair(newNodes[start], (newNodes[end])->getInEdgeNum(newNodes[start])));
  }

  int lastII = totalDelay / totalDistance;


  recurrenceList.insert(std::make_pair(lastII, recc));

}

bool ModuloSchedulingSBPass::circuit(MSchedGraphSBNode *v, std::vector<MSchedGraphSBNode*> &stack,
             std::set<MSchedGraphSBNode*> &blocked, std::vector<MSchedGraphSBNode*> &SCC,
             MSchedGraphSBNode *s, std::map<MSchedGraphSBNode*, std::set<MSchedGraphSBNode*> > &B,
                                   int II, std::map<MSchedGraphSBNode*, MSchedGraphSBNode*> &newNodes) {
  bool f = false;

  DEBUG(std::cerr << "Finding Circuits Starting with: ( " << v << ")"<< *v << "\n");

  //Push node onto the stack
  stack.push_back(v);

  //block this node
  blocked.insert(v);

  //Loop over all successors of node v that are in the scc, create Adjaceny list
  std::set<MSchedGraphSBNode*> AkV;
  for(MSchedGraphSBNode::succ_iterator I = v->succ_begin(), E = v->succ_end(); I != E; ++I) {
    if((std::find(SCC.begin(), SCC.end(), *I) != SCC.end())) {
      AkV.insert(*I);
    }
  }

  for(std::set<MSchedGraphSBNode*>::iterator I = AkV.begin(), E = AkV.end(); I != E; ++I) {
    if(*I == s) {
      //We have a circuit, so add it to our list
      addRecc(stack, newNodes);
      f = true;
    }
    else if(!blocked.count(*I)) {
      if(circuit(*I, stack, blocked, SCC, s, B, II, newNodes))
        f = true;
    }
    else
      DEBUG(std::cerr << "Blocked: " << **I << "\n");
  }


  if(f) {
    unblock(v, blocked, B);
  }
  else {
    for(std::set<MSchedGraphSBNode*>::iterator I = AkV.begin(), E = AkV.end(); I != E; ++I)
      B[*I].insert(v);

  }

  //Pop v
  stack.pop_back();

  return f;

}

void ModuloSchedulingSBPass::addRecc(std::vector<MSchedGraphSBNode*> &stack, std::map<MSchedGraphSBNode*, MSchedGraphSBNode*> &newNodes) {
  std::vector<MSchedGraphSBNode*> recc;
  //Dump recurrence for now
  DEBUG(std::cerr << "Starting Recc\n");
        
  int totalDelay = 0;
  int totalDistance = 0;
  MSchedGraphSBNode *lastN = 0;
  MSchedGraphSBNode *start = 0;
  MSchedGraphSBNode *end = 0;

  //Loop over recurrence, get delay and distance
  for(std::vector<MSchedGraphSBNode*>::iterator N = stack.begin(), NE = stack.end(); N != NE; ++N) {
    DEBUG(std::cerr << **N << "\n");
    totalDelay += (*N)->getLatency();
    if(lastN) {
      int iteDiff = (*N)->getInEdge(lastN).getIteDiff();
      totalDistance += iteDiff;

      if(iteDiff > 0) {
        start = lastN;
        end = *N;
      }
    }
    //Get the original node
    lastN = *N;
    recc.push_back(newNodes[*N]);


  }

  //Get the loop edge
  totalDistance += lastN->getIteDiff(*stack.begin());

  DEBUG(std::cerr << "End Recc\n");
  CircCountSB++;

  if(start && end) {    
    //Insert reccurrence into the list
    DEBUG(std::cerr << "Ignore Edge from!!: " << *start << " to " << *end << "\n");
    edgesToIgnore.insert(std::make_pair(newNodes[start], (newNodes[end])->getInEdgeNum(newNodes[start])));
  }
  else {
    //Insert reccurrence into the list
    DEBUG(std::cerr << "Ignore Edge from: " << *lastN << " to " << **stack.begin() << "\n");
    edgesToIgnore.insert(std::make_pair(newNodes[lastN], newNodes[(*stack.begin())]->getInEdgeNum(newNodes[lastN])));

  }
  //Adjust II until we get close to the inequality delay - II*distance <= 0
  int RecMII = II; //Starting value
  int value = totalDelay-(RecMII * totalDistance);
  int lastII = II;
  while(value < 0) {
          
    lastII = RecMII;
    RecMII--;
    value = totalDelay-(RecMII * totalDistance);
  }

  recurrenceList.insert(std::make_pair(lastII, recc));

}


void ModuloSchedulingSBPass::findAllCircuits(MSchedGraphSB *g, int II) {

  CircCountSB = 0;

  //Keep old to new node mapping information
  std::map<MSchedGraphSBNode*, MSchedGraphSBNode*> newNodes;

  //copy the graph
  MSchedGraphSB *MSG = new MSchedGraphSB(*g, newNodes);

  DEBUG(std::cerr << "Finding All Circuits\n");

  //Set of blocked nodes
  std::set<MSchedGraphSBNode*> blocked;

  //Stack holding current circuit
  std::vector<MSchedGraphSBNode*> stack;

  //Map for B Lists
  std::map<MSchedGraphSBNode*, std::set<MSchedGraphSBNode*> > B;

  //current node
  MSchedGraphSBNode *s;


  //Iterate over the graph until its down to one node or empty
  while(MSG->size() > 1) {

    //Write Graph out to file
    //WriteGraphToFile(std::cerr, "Graph" + utostr(MSG->size()), MSG);

    DEBUG(std::cerr << "Graph Size: " << MSG->size() << "\n");
    DEBUG(std::cerr << "Finding strong component Vk with least vertex\n");

    //Iterate over all the SCCs in the graph
    std::set<MSchedGraphSBNode*> Visited;
    std::vector<MSchedGraphSBNode*> Vk;
    MSchedGraphSBNode* s = 0;
    int numEdges = 0;

    //Find scc with the least vertex
    for (MSchedGraphSB::iterator GI = MSG->begin(), E = MSG->end(); GI != E; ++GI)
      if (Visited.insert(GI->second).second) {
        for (scc_iterator<MSchedGraphSBNode*> SCCI = scc_begin(GI->second),
               E = scc_end(GI->second); SCCI != E; ++SCCI) {
          std::vector<MSchedGraphSBNode*> &nextSCC = *SCCI;

          if (Visited.insert(nextSCC[0]).second) {
            Visited.insert(nextSCC.begin()+1, nextSCC.end());

            if(nextSCC.size() > 1) {
              DEBUG(std::cerr << "SCC size: " << nextSCC.size() << "\n");
              
              for(unsigned i = 0; i < nextSCC.size(); ++i) {
                //Loop over successor and see if in scc, then count edge
                MSchedGraphSBNode *node = nextSCC[i];
                for(MSchedGraphSBNode::succ_iterator S = node->succ_begin(), SE = node->succ_end(); S != SE; ++S) {
                  if(find(nextSCC.begin(), nextSCC.end(), *S) != nextSCC.end())
                    numEdges++;
                }
              }
              DEBUG(std::cerr << "Num Edges: " << numEdges << "\n");
            }
            
            //Ignore self loops
            if(nextSCC.size() > 1) {

              //Get least vertex in Vk
              if(!s) {
                s = nextSCC[0];
                Vk = nextSCC;
              }

              for(unsigned i = 0; i < nextSCC.size(); ++i) {
                if(nextSCC[i] < s) {
                  s = nextSCC[i];
                  Vk = nextSCC;
                }
              }
            }
          }
        }
      }



    //Process SCC
    DEBUG(for(std::vector<MSchedGraphSBNode*>::iterator N = Vk.begin(), NE = Vk.end();
              N != NE; ++N) { std::cerr << *((*N)->getInst()); });

    //Iterate over all nodes in this scc
    for(std::vector<MSchedGraphSBNode*>::iterator N = Vk.begin(), NE = Vk.end();
        N != NE; ++N) {
      blocked.erase(*N);
      B[*N].clear();
    }
    if(Vk.size() > 1) {
      if(numEdges < 98)
        circuit(s, stack, blocked, Vk, s, B, II, newNodes);
      else
        addSCC(Vk, newNodes);


      //Delete nodes from the graph
      //Find all nodes up to s and delete them
      std::vector<MSchedGraphSBNode*> nodesToRemove;
      nodesToRemove.push_back(s);
      for(MSchedGraphSB::iterator N = MSG->begin(), NE = MSG->end(); N != NE; ++N) {
        if(N->second < s )
            nodesToRemove.push_back(N->second);
      }
      for(std::vector<MSchedGraphSBNode*>::iterator N = nodesToRemove.begin(), NE = nodesToRemove.end(); N != NE; ++N) {
        DEBUG(std::cerr << "Deleting Node: " << **N << "\n");
        MSG->deleteNode(*N);
      }
    }
    else
      break;
  }    
  DEBUG(std::cerr << "Num Circuits found: " << CircCountSB << "\n");
}
/// calculateNodeAttributes - The following properties are calculated for
/// each node in the dependence graph: ASAP, ALAP, Depth, Height, and
/// MOB.
void ModuloSchedulingSBPass::calculateNodeAttributes(MSchedGraphSB *graph, int MII) {

  TIME_REGION(X, "calculateNodeAttributes");

  assert(nodeToAttributesMap.empty() && "Node attribute map was not cleared");

  //Loop over the nodes and add them to the map
  for(MSchedGraphSB::iterator I = graph->begin(), E = graph->end(); I != E; ++I) {

    DEBUG(std::cerr << "Inserting node into attribute map: " << *I->second << "\n");

    //Assert if its already in the map
    assert(nodeToAttributesMap.count(I->second) == 0 &&
           "Node attributes are already in the map");

    //Put into the map with default attribute values
    nodeToAttributesMap[I->second] = MSNodeSBAttributes();
  }

  //Create set to deal with reccurrences
  std::set<MSchedGraphSBNode*> visitedNodes;

  //Now Loop over map and calculate the node attributes
  for(std::map<MSchedGraphSBNode*, MSNodeSBAttributes>::iterator I = nodeToAttributesMap.begin(), E = nodeToAttributesMap.end(); I != E; ++I) {
    calculateASAP(I->first, MII, (MSchedGraphSBNode*) 0);
    visitedNodes.clear();
  }

  int maxASAP = findMaxASAP();
  //Calculate ALAP which depends on ASAP being totally calculated
  for(std::map<MSchedGraphSBNode*, MSNodeSBAttributes>::iterator I = nodeToAttributesMap.begin(), E = nodeToAttributesMap.end(); I != E; ++I) {
    calculateALAP(I->first, MII, maxASAP, (MSchedGraphSBNode*) 0);
    visitedNodes.clear();
  }

  //Calculate MOB which depends on ASAP being totally calculated, also do depth and height
  for(std::map<MSchedGraphSBNode*, MSNodeSBAttributes>::iterator I = nodeToAttributesMap.begin(), E = nodeToAttributesMap.end(); I != E; ++I) {
    (I->second).MOB = std::max(0,(I->second).ALAP - (I->second).ASAP);

    DEBUG(std::cerr << "MOB: " << (I->second).MOB << " (" << *(I->first) << ")\n");
    calculateDepth(I->first, (MSchedGraphSBNode*) 0);
    calculateHeight(I->first, (MSchedGraphSBNode*) 0);
  }


}

/// ignoreEdge - Checks to see if this edge of a recurrence should be ignored or not
bool ModuloSchedulingSBPass::ignoreEdge(MSchedGraphSBNode *srcNode, MSchedGraphSBNode *destNode) {
  if(destNode == 0 || srcNode ==0)
    return false;

  bool findEdge = edgesToIgnore.count(std::make_pair(srcNode, destNode->getInEdgeNum(srcNode)));

  DEBUG(std::cerr << "Ignoring edge? from: " << *srcNode << " to " << *destNode << "\n");

  return findEdge;
}


/// calculateASAP - Calculates the
int  ModuloSchedulingSBPass::calculateASAP(MSchedGraphSBNode *node, int MII, MSchedGraphSBNode *destNode) {

  DEBUG(std::cerr << "Calculating ASAP for " << *node << "\n");

  //Get current node attributes
  MSNodeSBAttributes &attributes = nodeToAttributesMap.find(node)->second;

  if(attributes.ASAP != -1)
    return attributes.ASAP;

  int maxPredValue = 0;

  //Iterate over all of the predecessors and find max
  for(MSchedGraphSBNode::pred_iterator P = node->pred_begin(), E = node->pred_end(); P != E; ++P) {

    //Only process if we are not ignoring the edge
    if(!ignoreEdge(*P, node)) {
      int predASAP = -1;
      predASAP = calculateASAP(*P, MII, node);

      assert(predASAP != -1 && "ASAP has not been calculated");
      int iteDiff = node->getInEdge(*P).getIteDiff();

      int currentPredValue = predASAP + (*P)->getLatency() - (iteDiff * MII);
      DEBUG(std::cerr << "pred ASAP: " << predASAP << ", iteDiff: " << iteDiff << ", PredLatency: " << (*P)->getLatency() << ", Current ASAP pred: " << currentPredValue << "\n");
      maxPredValue = std::max(maxPredValue, currentPredValue);
    }
  }

  attributes.ASAP = maxPredValue;

  DEBUG(std::cerr << "ASAP: " << attributes.ASAP << " (" << *node << ")\n");

  return maxPredValue;
}


int ModuloSchedulingSBPass::calculateALAP(MSchedGraphSBNode *node, int MII,
                                        int maxASAP, MSchedGraphSBNode *srcNode) {

  DEBUG(std::cerr << "Calculating ALAP for " << *node << "\n");

  MSNodeSBAttributes &attributes = nodeToAttributesMap.find(node)->second;

  if(attributes.ALAP != -1)
    return attributes.ALAP;

  if(node->hasSuccessors()) {

    //Trying to deal with the issue where the node has successors, but
    //we are ignoring all of the edges to them. So this is my hack for
    //now.. there is probably a more elegant way of doing this (FIXME)
    bool processedOneEdge = false;

    //FIXME, set to something high to start
    int minSuccValue = 9999999;

    //Iterate over all of the predecessors and fine max
    for(MSchedGraphSBNode::succ_iterator P = node->succ_begin(),
          E = node->succ_end(); P != E; ++P) {

      //Only process if we are not ignoring the edge
      if(!ignoreEdge(node, *P)) {
        processedOneEdge = true;
        int succALAP = -1;
        succALAP = calculateALAP(*P, MII, maxASAP, node);
        
        assert(succALAP != -1 && "Successors ALAP should have been caclulated");
        
        int iteDiff = P.getEdge().getIteDiff();
        
        int currentSuccValue = succALAP - node->getLatency() + iteDiff * MII;
        
        DEBUG(std::cerr << "succ ALAP: " << succALAP << ", iteDiff: " << iteDiff << ", SuccLatency: " << (*P)->getLatency() << ", Current ALAP succ: " << currentSuccValue << "\n");

        minSuccValue = std::min(minSuccValue, currentSuccValue);
      }
    }

    if(processedOneEdge)
        attributes.ALAP = minSuccValue;

    else
      attributes.ALAP = maxASAP;
  }
  else
    attributes.ALAP = maxASAP;

  DEBUG(std::cerr << "ALAP: " << attributes.ALAP << " (" << *node << ")\n");

  if(attributes.ALAP < 0)
    attributes.ALAP = 0;

  return attributes.ALAP;
}

int ModuloSchedulingSBPass::findMaxASAP() {
  int maxASAP = 0;

  for(std::map<MSchedGraphSBNode*, MSNodeSBAttributes>::iterator I = nodeToAttributesMap.begin(),
        E = nodeToAttributesMap.end(); I != E; ++I)
    maxASAP = std::max(maxASAP, I->second.ASAP);
  return maxASAP;
}


int ModuloSchedulingSBPass::calculateHeight(MSchedGraphSBNode *node,MSchedGraphSBNode *srcNode) {

  MSNodeSBAttributes &attributes = nodeToAttributesMap.find(node)->second;

  if(attributes.height != -1)
    return attributes.height;

  int maxHeight = 0;

  //Iterate over all of the predecessors and find max
  for(MSchedGraphSBNode::succ_iterator P = node->succ_begin(),
        E = node->succ_end(); P != E; ++P) {


    if(!ignoreEdge(node, *P)) {
      int succHeight = calculateHeight(*P, node);

      assert(succHeight != -1 && "Successors Height should have been caclulated");

      int currentHeight = succHeight + node->getLatency();
      maxHeight = std::max(maxHeight, currentHeight);
    }
  }
  attributes.height = maxHeight;
  DEBUG(std::cerr << "Height: " << attributes.height << " (" << *node << ")\n");
  return maxHeight;
}


int ModuloSchedulingSBPass::calculateDepth(MSchedGraphSBNode *node,
                                          MSchedGraphSBNode *destNode) {

  MSNodeSBAttributes &attributes = nodeToAttributesMap.find(node)->second;

  if(attributes.depth != -1)
    return attributes.depth;

  int maxDepth = 0;

  //Iterate over all of the predecessors and fine max
  for(MSchedGraphSBNode::pred_iterator P = node->pred_begin(), E = node->pred_end(); P != E; ++P) {

    if(!ignoreEdge(*P, node)) {
      int predDepth = -1;
      predDepth = calculateDepth(*P, node);

      assert(predDepth != -1 && "Predecessors ASAP should have been caclulated");

      int currentDepth = predDepth + (*P)->getLatency();
      maxDepth = std::max(maxDepth, currentDepth);
    }
  }
  attributes.depth = maxDepth;

  DEBUG(std::cerr << "Depth: " << attributes.depth << " (" << *node << "*)\n");
  return maxDepth;
}

void ModuloSchedulingSBPass::computePartialOrder() {

  TIME_REGION(X, "calculatePartialOrder");
  
  DEBUG(std::cerr << "Computing Partial Order\n");

  //Steps to add a recurrence to the partial order 1) Find reccurrence
  //with the highest RecMII. Add it to the partial order.  2) For each
  //recurrence with decreasing RecMII, add it to the partial order
  //along with any nodes that connect this recurrence to recurrences
  //already in the partial order
  for(std::set<std::pair<int, std::vector<MSchedGraphSBNode*> > >::reverse_iterator 
        I = recurrenceList.rbegin(), E=recurrenceList.rend(); I !=E; ++I) {

    std::set<MSchedGraphSBNode*> new_recurrence;

    //Loop through recurrence and remove any nodes already in the partial order
    for(std::vector<MSchedGraphSBNode*>::const_iterator N = I->second.begin(),
          NE = I->second.end(); N != NE; ++N) {

      bool found = false;
      for(std::vector<std::set<MSchedGraphSBNode*> >::iterator PO = partialOrder.begin(),
            PE = partialOrder.end(); PO != PE; ++PO) {
        if(PO->count(*N))
          found = true;
      }

      //Check if its a branch, and remove to handle special
      if(!found) {
        new_recurrence.insert(*N);
      }

    }


    if(new_recurrence.size() > 0) {

      std::vector<MSchedGraphSBNode*> path;
      std::set<MSchedGraphSBNode*> nodesToAdd;

      //Dump recc we are dealing with (minus nodes already in PO)
      DEBUG(std::cerr << "Recc: ");
      DEBUG(for(std::set<MSchedGraphSBNode*>::iterator R = new_recurrence.begin(), RE = new_recurrence.end(); R != RE; ++R) { std::cerr << **R ; });

      //Add nodes that connect this recurrence to recurrences in the partial path
      for(std::set<MSchedGraphSBNode*>::iterator N = new_recurrence.begin(),
          NE = new_recurrence.end(); N != NE; ++N)
        searchPath(*N, path, nodesToAdd, new_recurrence);

      //Add nodes to this recurrence if they are not already in the partial order
      for(std::set<MSchedGraphSBNode*>::iterator N = nodesToAdd.begin(), NE = nodesToAdd.end();
          N != NE; ++N) {
        bool found = false;
        for(std::vector<std::set<MSchedGraphSBNode*> >::iterator PO = partialOrder.begin(),
              PE = partialOrder.end(); PO != PE; ++PO) {
          if(PO->count(*N))
            found = true;
        }
        if(!found) {
          assert("FOUND CONNECTOR");
          new_recurrence.insert(*N);
        }
      }

      partialOrder.push_back(new_recurrence);
    }
  }

  //Add any nodes that are not already in the partial order
  //Add them in a set, one set per connected component
  std::set<MSchedGraphSBNode*> lastNodes;
  std::set<MSchedGraphSBNode*> noPredNodes;
  for(std::map<MSchedGraphSBNode*, MSNodeSBAttributes>::iterator I = nodeToAttributesMap.begin(),
        E = nodeToAttributesMap.end(); I != E; ++I) {

    bool found = false;

    //Check if its already in our partial order, if not add it to the final vector
    for(std::vector<std::set<MSchedGraphSBNode*> >::iterator PO = partialOrder.begin(),
          PE = partialOrder.end(); PO != PE; ++PO) {
      if(PO->count(I->first))
        found = true;
    }
    if(!found)
      lastNodes.insert(I->first);
  }

  //For each node w/out preds, see if there is a path to one of the
  //recurrences, and if so add them to that current recc
  /*for(std::set<MSchedGraphSBNode*>::iterator N = noPredNodes.begin(), NE = noPredNodes.end();
      N != NE; ++N) {
    DEBUG(std::cerr << "No Pred Path from: " << **N << "\n");
    for(std::vector<std::set<MSchedGraphSBNode*> >::iterator PO = partialOrder.begin(),
          PE = partialOrder.end(); PO != PE; ++PO) {
      std::vector<MSchedGraphSBNode*> path;
      pathToRecc(*N, path, *PO, lastNodes);
    }
    }*/


  //Break up remaining nodes that are not in the partial order
  ///into their connected compoenents
    while(lastNodes.size() > 0) {
      std::set<MSchedGraphSBNode*> ccSet;
      connectedComponentSet(*(lastNodes.begin()),ccSet, lastNodes);
      if(ccSet.size() > 0)
        partialOrder.push_back(ccSet);
    }

}

void ModuloSchedulingSBPass::connectedComponentSet(MSchedGraphSBNode *node, std::set<MSchedGraphSBNode*> &ccSet, std::set<MSchedGraphSBNode*> &lastNodes) {
  
  //Add to final set
  if( !ccSet.count(node) && lastNodes.count(node)) {
    lastNodes.erase(node);
    ccSet.insert(node);
  }
  else
    return;

  //Loop over successors and recurse if we have not seen this node before
  for(MSchedGraphSBNode::succ_iterator node_succ = node->succ_begin(), end=node->succ_end(); node_succ != end; ++node_succ) {
    connectedComponentSet(*node_succ, ccSet, lastNodes);
  }

}

void ModuloSchedulingSBPass::searchPath(MSchedGraphSBNode *node,
                                      std::vector<MSchedGraphSBNode*> &path,
                                      std::set<MSchedGraphSBNode*> &nodesToAdd,
                                     std::set<MSchedGraphSBNode*> &new_reccurrence) {
  //Push node onto the path
  path.push_back(node);

  //Loop over all successors and see if there is a path from this node to
  //a recurrence in the partial order, if so.. add all nodes to be added to recc
  for(MSchedGraphSBNode::succ_iterator S = node->succ_begin(), SE = node->succ_end(); S != SE;
      ++S) {

    //Check if we should ignore this edge first
    if(ignoreEdge(node,*S))
      continue;
    
    //check if successor is in this recurrence, we will get to it eventually
    if(new_reccurrence.count(*S))
      continue;

    //If this node exists in a recurrence already in the partial
    //order, then add all nodes in the path to the set of nodes to add
     //Check if its already in our partial order, if not add it to the
     //final vector
    bool found = false;
    for(std::vector<std::set<MSchedGraphSBNode*> >::iterator PO = partialOrder.begin(),
          PE = partialOrder.end(); PO != PE; ++PO) {

      if(PO->count(*S)) {
        found = true;
        break;
      }
    }

    if(!found) {
      nodesToAdd.insert(*S);
      searchPath(*S, path, nodesToAdd, new_reccurrence);
    }
  }

  //Pop Node off the path
  path.pop_back();
}

void dumpIntersection(std::set<MSchedGraphSBNode*> &IntersectCurrent) {
  std::cerr << "Intersection (";
  for(std::set<MSchedGraphSBNode*>::iterator I = IntersectCurrent.begin(), E = IntersectCurrent.end(); I != E; ++I)
    std::cerr << **I << ", ";
  std::cerr << ")\n";
}

void ModuloSchedulingSBPass::orderNodes() {

  TIME_REGION(X, "orderNodes");

  int BOTTOM_UP = 0;
  int TOP_DOWN = 1;

  //Set default order
  int order = BOTTOM_UP;

  //Loop over and find all pred nodes and schedule them first
  /*for(std::vector<std::set<MSchedGraphSBNode*> >::iterator CurrentSet = partialOrder.begin(), E= partialOrder.end(); CurrentSet != E; ++CurrentSet) {
    for(std::set<MSchedGraphSBNode*>::iterator N = CurrentSet->begin(), NE = CurrentSet->end(); N != NE; ++N)
      if((*N)->isPredicate()) {
        FinalNodeOrder.push_back(*N);
        CurrentSet->erase(*N);
      }
      }*/



  //Loop over all the sets and place them in the final node order
  for(std::vector<std::set<MSchedGraphSBNode*> >::iterator CurrentSet = partialOrder.begin(), E= partialOrder.end(); CurrentSet != E; ++CurrentSet) {

    DEBUG(std::cerr << "Processing set in S\n");
    DEBUG(dumpIntersection(*CurrentSet));

    //Result of intersection
    std::set<MSchedGraphSBNode*> IntersectCurrent;

    predIntersect(*CurrentSet, IntersectCurrent);

    //If the intersection of predecessor and current set is not empty
    //sort nodes bottom up
    if(IntersectCurrent.size() != 0) {
      DEBUG(std::cerr << "Final Node Order Predecessors and Current Set interesection is NOT empty\n");
      order = BOTTOM_UP;
    }
    //If empty, use successors
    else {
      DEBUG(std::cerr << "Final Node Order Predecessors and Current Set interesection is empty\n");

      succIntersect(*CurrentSet, IntersectCurrent);

      //sort top-down
      if(IntersectCurrent.size() != 0) {
         DEBUG(std::cerr << "Final Node Order Successors and Current Set interesection is NOT empty\n");
        order = TOP_DOWN;
      }
      else {
        DEBUG(std::cerr << "Final Node Order Successors and Current Set interesection is empty\n");
        //Find node with max ASAP in current Set
        MSchedGraphSBNode *node;
        int maxASAP = 0;
        DEBUG(std::cerr << "Using current set of size " << CurrentSet->size() << "to find max ASAP\n");
        for(std::set<MSchedGraphSBNode*>::iterator J = CurrentSet->begin(), JE = CurrentSet->end(); J != JE; ++J) {
          //Get node attributes
          MSNodeSBAttributes nodeAttr= nodeToAttributesMap.find(*J)->second;
          //assert(nodeAttr != nodeToAttributesMap.end() && "Node not in attributes map!");
        
          if(maxASAP <= nodeAttr.ASAP) {
            maxASAP = nodeAttr.ASAP;
            node = *J;
          }
        }
        assert(node != 0 && "In node ordering node should not be null");
        IntersectCurrent.insert(node);
        order = BOTTOM_UP;
      }
    }

    //Repeat until all nodes are put into the final order from current set
    while(IntersectCurrent.size() > 0) {

      if(order == TOP_DOWN) {
        DEBUG(std::cerr << "Order is TOP DOWN\n");

        while(IntersectCurrent.size() > 0) {
          DEBUG(std::cerr << "Intersection is not empty, so find heighest height\n");
        
          int MOB = 0;
          int height = 0;
          MSchedGraphSBNode *highestHeightNode = *(IntersectCurrent.begin());
                
          //Find node in intersection with highest heigh and lowest MOB
          for(std::set<MSchedGraphSBNode*>::iterator I = IntersectCurrent.begin(),
                E = IntersectCurrent.end(); I != E; ++I) {
        
            //Get current nodes properties
            MSNodeSBAttributes nodeAttr= nodeToAttributesMap.find(*I)->second;

            if(height < nodeAttr.height) {
              highestHeightNode = *I;
              height = nodeAttr.height;
              MOB = nodeAttr.MOB;
            }
            else if(height ==  nodeAttr.height) {
              if(MOB > nodeAttr.height) {
                highestHeightNode = *I;
                height =  nodeAttr.height;
                MOB = nodeAttr.MOB;
              }
            }
          }
        
          //Append our node with greatest height to the NodeOrder
          if(std::find(FinalNodeOrder.begin(), FinalNodeOrder.end(), highestHeightNode) == FinalNodeOrder.end()) {
            DEBUG(std::cerr << "Adding node to Final Order: " << *highestHeightNode << "\n");
            FinalNodeOrder.push_back(highestHeightNode);
          }

          //Remove V from IntersectOrder
          IntersectCurrent.erase(std::find(IntersectCurrent.begin(),
                                      IntersectCurrent.end(), highestHeightNode));


          //Intersect V's successors with CurrentSet
          for(MSchedGraphSBNode::succ_iterator P = highestHeightNode->succ_begin(),
                E = highestHeightNode->succ_end(); P != E; ++P) {
            //if(lower_bound(CurrentSet->begin(),
            //     CurrentSet->end(), *P) != CurrentSet->end()) {
            if(std::find(CurrentSet->begin(), CurrentSet->end(), *P) != CurrentSet->end()) {
              if(ignoreEdge(highestHeightNode, *P))
                continue;
              //If not already in Intersect, add
              if(!IntersectCurrent.count(*P))
                IntersectCurrent.insert(*P);
            }
          }
        } //End while loop over Intersect Size

        //Change direction
        order = BOTTOM_UP;

        //Reset Intersect to reflect changes in OrderNodes
        IntersectCurrent.clear();
        predIntersect(*CurrentSet, IntersectCurrent);
        
      } //End If TOP_DOWN
        
        //Begin if BOTTOM_UP
      else {
        DEBUG(std::cerr << "Order is BOTTOM UP\n");
        while(IntersectCurrent.size() > 0) {
          DEBUG(std::cerr << "Intersection of size " << IntersectCurrent.size() << ", finding highest depth\n");

          //dump intersection
          DEBUG(dumpIntersection(IntersectCurrent));
          //Get node with highest depth, if a tie, use one with lowest
          //MOB
          int MOB = 0;
          int depth = 0;
          MSchedGraphSBNode *highestDepthNode = *(IntersectCurrent.begin());
        
          for(std::set<MSchedGraphSBNode*>::iterator I = IntersectCurrent.begin(),
                E = IntersectCurrent.end(); I != E; ++I) {
            //Find node attribute in graph
            MSNodeSBAttributes nodeAttr= nodeToAttributesMap.find(*I)->second;
        
            if(depth < nodeAttr.depth) {
              highestDepthNode = *I;
              depth = nodeAttr.depth;
              MOB = nodeAttr.MOB;
            }
            else if(depth == nodeAttr.depth) {
              if(MOB > nodeAttr.MOB) {
                highestDepthNode = *I;
                depth = nodeAttr.depth;
                MOB = nodeAttr.MOB;
              }
            }
          }
        
        

          //Append highest depth node to the NodeOrder
           if(std::find(FinalNodeOrder.begin(), FinalNodeOrder.end(), highestDepthNode) == FinalNodeOrder.end()) {
             DEBUG(std::cerr << "Adding node to Final Order: " << *highestDepthNode << "\n");
             FinalNodeOrder.push_back(highestDepthNode);
           }
          //Remove heightestDepthNode from IntersectOrder
           IntersectCurrent.erase(highestDepthNode);
        

          //Intersect heightDepthNode's pred with CurrentSet
          for(MSchedGraphSBNode::pred_iterator P = highestDepthNode->pred_begin(),
                E = highestDepthNode->pred_end(); P != E; ++P) {
            if(CurrentSet->count(*P)) {
              if(ignoreEdge(*P, highestDepthNode))
                continue;
        
            //If not already in Intersect, add
            if(!IntersectCurrent.count(*P))
              IntersectCurrent.insert(*P);
            }
          }
        
        } //End while loop over Intersect Size
        
          //Change order
        order = TOP_DOWN;
        
        //Reset IntersectCurrent to reflect changes in OrderNodes
        IntersectCurrent.clear();
        succIntersect(*CurrentSet, IntersectCurrent);
        } //End if BOTTOM_DOWN
        
      DEBUG(std::cerr << "Current Intersection Size: " << IntersectCurrent.size() << "\n");
    }
    //End Wrapping while loop
    DEBUG(std::cerr << "Ending Size of Current Set: " << CurrentSet->size() << "\n");
  }//End for over all sets of nodes

  //FIXME: As the algorithm stands it will NEVER add an instruction such as ba (with no
  //data dependencies) to the final order. We add this manually. It will always be
  //in the last set of S since its not part of a recurrence
    //Loop over all the sets and place them in the final node order
  std::vector<std::set<MSchedGraphSBNode*> > ::reverse_iterator LastSet = partialOrder.rbegin();
  for(std::set<MSchedGraphSBNode*>::iterator CurrentNode = LastSet->begin(), LastNode = LastSet->end();
      CurrentNode != LastNode; ++CurrentNode) {
    if((*CurrentNode)->getInst()->getOpcode() == V9::BA)
      FinalNodeOrder.push_back(*CurrentNode);
  }
  //Return final Order
  //return FinalNodeOrder;
}


void ModuloSchedulingSBPass::predIntersect(std::set<MSchedGraphSBNode*> &CurrentSet, std::set<MSchedGraphSBNode*> &IntersectResult) {

  for(unsigned j=0; j < FinalNodeOrder.size(); ++j) {
    for(MSchedGraphSBNode::pred_iterator P = FinalNodeOrder[j]->pred_begin(),
          E = FinalNodeOrder[j]->pred_end(); P != E; ++P) {

      //Check if we are supposed to ignore this edge or not
      if(ignoreEdge(*P,FinalNodeOrder[j]))
        continue;
        
      if(CurrentSet.count(*P))
        if(std::find(FinalNodeOrder.begin(), FinalNodeOrder.end(), *P) == FinalNodeOrder.end())
          IntersectResult.insert(*P);
    }
  }
}

void ModuloSchedulingSBPass::succIntersect(std::set<MSchedGraphSBNode*> &CurrentSet, std::set<MSchedGraphSBNode*> &IntersectResult) {

  for(unsigned j=0; j < FinalNodeOrder.size(); ++j) {
    for(MSchedGraphSBNode::succ_iterator P = FinalNodeOrder[j]->succ_begin(),
          E = FinalNodeOrder[j]->succ_end(); P != E; ++P) {

      //Check if we are supposed to ignore this edge or not
      if(ignoreEdge(FinalNodeOrder[j],*P))
        continue;

      if(CurrentSet.count(*P))
        if(std::find(FinalNodeOrder.begin(), FinalNodeOrder.end(), *P) == FinalNodeOrder.end())
          IntersectResult.insert(*P);
    }
  }
}



bool ModuloSchedulingSBPass::computeSchedule(std::vector<const MachineBasicBlock*> &SB, MSchedGraphSB *MSG) {

  TIME_REGION(X, "computeSchedule");

  bool success = false;

  //FIXME: Should be set to max II of the original loop
  //Cap II in order to prevent infinite loop
  int capII = MSG->totalDelay();

  while(!success) {

    //Keep track of branches, but do not insert into the schedule
    std::vector<MSchedGraphSBNode*> branches;

    //Loop over the final node order and process each node
    for(std::vector<MSchedGraphSBNode*>::iterator I = FinalNodeOrder.begin(),
          E = FinalNodeOrder.end(); I != E; ++I) {

      //CalculateEarly and Late start
      bool initialLSVal = false;
      bool initialESVal = false;
      int EarlyStart = 0;
      int LateStart = 0; 
      bool hasSucc = false;
      bool hasPred = false;
      bool sched;

      if((*I)->isBranch())
        if((*I)->hasPredecessors())
          sched = true;
        else
          sched = false;
      else
        sched = true;

      if(sched) {
        //Loop over nodes in the schedule and determine if they are predecessors
        //or successors of the node we are trying to schedule
        for(MSScheduleSB::schedule_iterator nodesByCycle = schedule.begin(), nodesByCycleEnd = schedule.end();
            nodesByCycle != nodesByCycleEnd; ++nodesByCycle) {
        
          //For this cycle, get the vector of nodes schedule and loop over it
          for(std::vector<MSchedGraphSBNode*>::iterator schedNode = nodesByCycle->second.begin(), SNE = nodesByCycle->second.end(); schedNode != SNE; ++schedNode) {
        
            if((*I)->isPredecessor(*schedNode)) {
              int diff = (*I)->getInEdge(*schedNode).getIteDiff();
              int ES_Temp = nodesByCycle->first + (*schedNode)->getLatency() - diff * II;
              DEBUG(std::cerr << "Diff: " << diff << " Cycle: " << nodesByCycle->first << "\n");
              DEBUG(std::cerr << "Temp EarlyStart: " << ES_Temp << " Prev EarlyStart: " << EarlyStart << "\n");
              if(initialESVal)
                EarlyStart = std::max(EarlyStart, ES_Temp);
              else {
                EarlyStart = ES_Temp;
                initialESVal = true;
              }
              hasPred = true;
            }
            if((*I)->isSuccessor(*schedNode)) {
              int diff = (*schedNode)->getInEdge(*I).getIteDiff();
              int LS_Temp = nodesByCycle->first - (*I)->getLatency() + diff * II;
              DEBUG(std::cerr << "Diff: " << diff << " Cycle: " << nodesByCycle->first << "\n");
              DEBUG(std::cerr << "Temp LateStart: " << LS_Temp << " Prev LateStart: " << LateStart << "\n");
              if(initialLSVal)
                LateStart = std::min(LateStart, LS_Temp);
              else {
                LateStart = LS_Temp;
                initialLSVal = true;
              }
              hasSucc = true;
            }
          }
        }
      }
      else {
        branches.push_back(*I);
        continue;
      }

      //Check if the node has no pred or successors and set Early Start to its ASAP
      if(!hasSucc && !hasPred)
        EarlyStart = nodeToAttributesMap.find(*I)->second.ASAP;

      DEBUG(std::cerr << "Has Successors: " << hasSucc << ", Has Pred: " << hasPred << "\n");
      DEBUG(std::cerr << "EarlyStart: " << EarlyStart << ", LateStart: " << LateStart << "\n");

      //Now, try to schedule this node depending upon its pred and successor in the schedule
      //already
      if(!hasSucc && hasPred)
        success = scheduleNode(*I, EarlyStart, (EarlyStart + II -1));
      else if(!hasPred && hasSucc)
        success = scheduleNode(*I, LateStart, (LateStart - II +1));
      else if(hasPred && hasSucc) {
        if(EarlyStart > LateStart) {
        success = false;
          //LateStart = EarlyStart;
          DEBUG(std::cerr << "Early Start can not be later then the late start cycle, schedule fails\n");
        }
        else
          success = scheduleNode(*I, EarlyStart, std::min(LateStart, (EarlyStart + II -1)));
      }
      else
        success = scheduleNode(*I, EarlyStart, EarlyStart + II - 1);

      if(!success) {
        ++II; 
        schedule.clear();
        break;
      }

    }

    if(success) {
      DEBUG(std::cerr << "Constructing Schedule Kernel\n");
      success = schedule.constructKernel(II, branches, indVarInstrs[SB]);
      DEBUG(std::cerr << "Done Constructing Schedule Kernel\n");
      if(!success) {
        ++II;
        schedule.clear();
      }
      DEBUG(std::cerr << "Final II: " << II << "\n");
     
    }

    if(II >= capII) {
      DEBUG(std::cerr << "Maximum II reached, giving up\n");
      return false;
    }

    assert(II < capII && "The II should not exceed the original loop number of cycles");
  }
  return true;
}


bool ModuloSchedulingSBPass::scheduleNode(MSchedGraphSBNode *node,
                                      int start, int end) {
  bool success = false;

  DEBUG(std::cerr << *node << " (Start Cycle: " << start << ", End Cycle: " << end << ")\n");

  //Make sure start and end are not negative
  //if(start < 0) {
  //start = 0;

  //}
  //if(end < 0)
  //end = 0;

  bool forward = true;
  if(start > end)
    forward = false;

  bool increaseSC = true;
  int cycle = start ;


  while(increaseSC) {

    increaseSC = false;

    increaseSC = schedule.insert(node, cycle, II);

    if(!increaseSC)
      return true;

    //Increment cycle to try again
    if(forward) {
      ++cycle;
      DEBUG(std::cerr << "Increase cycle: " << cycle << "\n");
      if(cycle > end)
        return false;
    }
    else {
      --cycle;
      DEBUG(std::cerr << "Decrease cycle: " << cycle << "\n");
      if(cycle < end)
        return false;
    }
  }

  return success;
}

void ModuloSchedulingSBPass::reconstructLoop(std::vector<const MachineBasicBlock*> &SB) {

  TIME_REGION(X, "reconstructLoop");


  DEBUG(std::cerr << "Reconstructing Loop\n");

  //First find the value *'s that we need to "save"
  std::map<const Value*, std::pair<const MachineInstr*, int> > valuesToSave;

  //Keep track of instructions we have already seen and their stage because
  //we don't want to "save" values if they are used in the kernel immediately
  std::map<const MachineInstr*, int> lastInstrs;


  std::set<MachineBasicBlock*> seenBranchesBB;
  const TargetInstrInfo *MTI = target.getInstrInfo();
  std::map<MachineBasicBlock*, std::vector<std::pair<MachineInstr*, int> > > instrsMovedDown;
  std::map<MachineBasicBlock*, int> branchStage;

  //Loop over kernel and only look at instructions from a stage > 0
  //Look at its operands and save values *'s that are read
  for(MSScheduleSB::kernel_iterator I = schedule.kernel_begin(), E = schedule.kernel_end(); I != E; ++I) {
    
    if(I->second !=0) {
      //For this instruction, get the Value*'s that it reads and put them into the set.
      //Assert if there is an operand of another type that we need to save
      const MachineInstr *inst = I->first;
      lastInstrs[inst] = I->second;

      for(unsigned i=0; i < inst->getNumOperands(); ++i) {
        //get machine operand
        const MachineOperand &mOp = inst->getOperand(i);
        
        if(mOp.getType() == MachineOperand::MO_VirtualRegister && mOp.isUse()) {
          //find the value in the map
          if (const Value* srcI = mOp.getVRegValue()) {

            if(isa<Constant>(srcI) || isa<Argument>(srcI))
              continue;

            //Before we declare this Value* one that we should save
            //make sure its def is not of the same stage as this instruction
            //because it will be consumed before its used
            Instruction *defInst = (Instruction*) srcI;
        
            //Should we save this value?
            bool save = true;

            //Continue if not in the def map, loop invariant code does not need to be saved
            if(!defMap.count(srcI))
              continue;

            MachineInstr *defInstr = defMap[srcI];
        

            if(lastInstrs.count(defInstr)) {
              if(lastInstrs[defInstr] == I->second) {
                save = false;
                
              }
            }
        
            if(save)
              valuesToSave[srcI] = std::make_pair(I->first, i);
          }     
        }
        
        if(mOp.getType() != MachineOperand::MO_VirtualRegister && mOp.isUse()) {
          assert("Our assumption is wrong. We have another type of register that needs to be saved\n");
        }
      }
    }
    
    
    //Do a check to see if instruction was moved below its original branch
    if(MTI->isBranch(I->first->getOpcode())) {
      seenBranchesBB.insert(I->first->getParent());
      branchStage[I->first->getParent()] = I->second;
    }
    else {
      instrsMovedDown[I->first->getParent()].push_back(std::make_pair(I->first, I->second));
      //assert(seenBranchesBB.count(I->first->getParent()) && "Instruction moved below branch\n");
    }

  }

  //The new loop will consist of one or more prologues, the kernel, and one or more epilogues.

  //Map to keep track of old to new values
  std::map<Value*, std::map<int, Value*> > newValues;

  //Map to keep track of old to new values in kernel
  std::map<Value*, std::map<int, Value*> > kernelPHIs;

  //Another map to keep track of what machine basic blocks these new value*s are in since
  //they have no llvm instruction equivalent
  std::map<Value*, MachineBasicBlock*> newValLocation;

  std::vector<std::vector<MachineBasicBlock*> > prologues;
  std::vector<std::vector<BasicBlock*> > llvm_prologues;

  //Map to keep track of where the inner branches go
  std::map<const MachineBasicBlock*, Value*> sideExits;

  
  //Write prologue
  if(schedule.getMaxStage() != 0)
    writePrologues(prologues, SB, llvm_prologues, valuesToSave, newValues, newValLocation);

  std::vector<BasicBlock*> llvmKernelBBs;
  std::vector<MachineBasicBlock*> machineKernelBBs;
  Function *parent = (Function*) SB[0]->getBasicBlock()->getParent();

  for(unsigned i = 0; i < SB.size(); ++i) {
    llvmKernelBBs.push_back(new BasicBlock("Kernel", parent));
    
    machineKernelBBs.push_back(new MachineBasicBlock(llvmKernelBBs[i]));
    (((MachineBasicBlock*)SB[0])->getParent())->getBasicBlockList().push_back(machineKernelBBs[i]);
  }

  writeKernel(llvmKernelBBs, machineKernelBBs, valuesToSave, newValues, newValLocation, kernelPHIs);


  std::vector<std::vector<MachineBasicBlock*> > epilogues;
  std::vector<std::vector<BasicBlock*> > llvm_epilogues;

  //Write epilogues
  if(schedule.getMaxStage() != 0)
    writeEpilogues(epilogues, SB, llvm_epilogues, valuesToSave, newValues, newValLocation, kernelPHIs);


  //Fix our branches
  fixBranches(prologues, llvm_prologues, machineKernelBBs, llvmKernelBBs, epilogues, llvm_epilogues, SB, sideExits);

  //Print out epilogues and prologue
  DEBUG(for(std::vector<std::vector<MachineBasicBlock*> >::iterator PI = prologues.begin(), PE = prologues.end();
      PI != PE; ++PI) {
          std::cerr << "PROLOGUE\n";
          for(std::vector<MachineBasicBlock*>::iterator I = PI->begin(), E = PI->end(); I != E; ++I)
            (*I)->print(std::cerr);
        });

  DEBUG(std::cerr << "KERNEL\n");
  DEBUG(for(std::vector<MachineBasicBlock*>::iterator I = machineKernelBBs.begin(), E = machineKernelBBs.end(); I != E; ++I) { (*I)->print(std::cerr);});

  DEBUG(for(std::vector<std::vector<MachineBasicBlock*> >::iterator EI = epilogues.begin(), EE = epilogues.end();
      EI != EE; ++EI) {
    std::cerr << "EPILOGUE\n";
    for(std::vector<MachineBasicBlock*>::iterator I = EI->begin(), E = EI->end(); I != E; ++I)
      (*I)->print(std::cerr);
  });


  //Remove phis
  removePHIs(SB, prologues, epilogues, machineKernelBBs, newValLocation);

  //Print out epilogues and prologue
  DEBUG(for(std::vector<std::vector<MachineBasicBlock*> >::iterator PI = prologues.begin(), PE = prologues.end();
      PI != PE; ++PI) {
          std::cerr << "PROLOGUE\n";
          for(std::vector<MachineBasicBlock*>::iterator I = PI->begin(), E = PI->end(); I != E; ++I)
            (*I)->print(std::cerr);
        });

  DEBUG(std::cerr << "KERNEL\n");
  DEBUG(for(std::vector<MachineBasicBlock*>::iterator I = machineKernelBBs.begin(), E = machineKernelBBs.end(); I != E; ++I) { (*I)->print(std::cerr);});

  DEBUG(for(std::vector<std::vector<MachineBasicBlock*> >::iterator EI = epilogues.begin(), EE = epilogues.end();
      EI != EE; ++EI) {
    std::cerr << "EPILOGUE\n";
    for(std::vector<MachineBasicBlock*>::iterator I = EI->begin(), E = EI->end(); I != E; ++I)
      (*I)->print(std::cerr);
  });

  writeSideExits(prologues, llvm_prologues, epilogues, llvm_epilogues, sideExits, instrsMovedDown, SB, machineKernelBBs, branchStage);


  DEBUG(std::cerr << "New Machine Function" << "\n");
}


void ModuloSchedulingSBPass::fixBranches(std::vector<std::vector<MachineBasicBlock*> > &prologues, std::vector<std::vector<BasicBlock*> > &llvm_prologues, std::vector<MachineBasicBlock*> &machineKernelBB, std::vector<BasicBlock*> &llvmKernelBB, std::vector<std::vector<MachineBasicBlock*> > &epilogues, std::vector<std::vector<BasicBlock*> > &llvm_epilogues, std::vector<const MachineBasicBlock*> &SB, std::map<const MachineBasicBlock*, Value*> &sideExits) {

  const TargetInstrInfo *TMI = target.getInstrInfo();

  //Get exit BB
  BasicBlock *last = (BasicBlock*) SB[SB.size()-1]->getBasicBlock();
  BasicBlock *kernel_exit = 0;
  bool sawFirst = false;

  for(succ_iterator I = succ_begin(last),
        E = succ_end(last); I != E; ++I) {
    if (*I != SB[0]->getBasicBlock()) {
      kernel_exit = *I;
      break;
    }
    else
      sawFirst = true;
  }
  if(!kernel_exit && sawFirst) {
    kernel_exit = (BasicBlock*) SB[0]->getBasicBlock();
  }

  assert(kernel_exit && "Kernel Exit can not be null");

  if(schedule.getMaxStage() != 0) {
    //Fix prologue branches
    for(unsigned i = 0; i <  prologues.size(); ++i) {

      for(unsigned j = 0; j < prologues[i].size(); ++j) {

        MachineBasicBlock *currentMBB = prologues[i][j];
       
        //Find terminator since getFirstTerminator does not work!
        for(MachineBasicBlock::reverse_iterator mInst = currentMBB->rbegin(), mInstEnd = currentMBB->rend(); mInst != mInstEnd; ++mInst) {
          MachineOpCode OC = mInst->getOpcode();
          //If its a branch update its branchto
          if(TMI->isBranch(OC)) {
            for(unsigned opNum = 0; opNum < mInst->getNumOperands(); ++opNum) {
              MachineOperand &mOp = mInst->getOperand(opNum);
              if (mOp.getType() == MachineOperand::MO_PCRelativeDisp) {
                //Check if we are branching to the kernel, if not branch to epilogue
                if(mOp.getVRegValue() == SB[0]->getBasicBlock()) {
                  if(i >= prologues.size()-1)
                    mOp.setValueReg(llvmKernelBB[0]);
                  else
                    mOp.setValueReg(llvm_prologues[i+1][0]);
                }
                else if( (mOp.getVRegValue() == kernel_exit) && (j == prologues[i].size()-1)) {
                  mOp.setValueReg(llvm_epilogues[i][0]);
                }
                else if(mOp.getVRegValue() == SB[j+1]->getBasicBlock()) {
                  mOp.setValueReg(llvm_prologues[i][j+1]);
                }
                
              }
            }
            
            DEBUG(std::cerr << "New Prologue Branch: " << *mInst << "\n");
          }
        }

        //Update llvm basic block with our new branch instr
        DEBUG(std::cerr << SB[i]->getBasicBlock()->getTerminator() << "\n");
        
        const BranchInst *branchVal = dyn_cast<BranchInst>(SB[i]->getBasicBlock()->getTerminator());

        //Check for inner branch
        if(j < prologues[i].size()-1) {
          //Find our side exit LLVM basic block
          BasicBlock *sideExit = 0;
          for(unsigned s = 0; s < branchVal->getNumSuccessors(); ++s) {
            if(branchVal->getSuccessor(s) != SB[i+1]->getBasicBlock())
              sideExit = branchVal->getSuccessor(s);
          }
          assert(sideExit && "Must have side exit llvm basic block");
          TerminatorInst *newBranch = new BranchInst(sideExit,
                                        llvm_prologues[i][j+1],
                                        branchVal->getCondition(),
                                        llvm_prologues[i][j]);
        }
        else {
          //If last prologue
          if(i == prologues.size()-1) {
            TerminatorInst *newBranch = new BranchInst(llvmKernelBB[0],
                                                       llvm_epilogues[i][0],
                                                       branchVal->getCondition(),
                                                       llvm_prologues[i][j]);
          }
          else {
            TerminatorInst *newBranch = new BranchInst(llvm_prologues[i+1][0],
                                                       llvm_epilogues[i][0],
                                                       branchVal->getCondition(),
                                                       llvm_prologues[i][j]);
          }
        }
      }
    }
  }

  //Fix up kernel machine branches
  for(unsigned i = 0; i < machineKernelBB.size(); ++i) {
    MachineBasicBlock *currentMBB = machineKernelBB[i];

    for(MachineBasicBlock::reverse_iterator mInst = currentMBB->rbegin(), mInstEnd = currentMBB->rend(); mInst != mInstEnd; ++mInst) {
      MachineOpCode OC = mInst->getOpcode();
      if(TMI->isBranch(OC)) {
        for(unsigned opNum = 0; opNum < mInst->getNumOperands(); ++opNum) {
          MachineOperand &mOp = mInst->getOperand(opNum);
        
          if(mOp.getType() == MachineOperand::MO_PCRelativeDisp) {
            //Deal with inner kernel branches
            if(i < machineKernelBB.size()-1) {
              if(mOp.getVRegValue() == SB[i+1]->getBasicBlock())
                mOp.setValueReg(llvmKernelBB[i+1]);
              //Side exit!
              else {
                sideExits[SB[i]] = mOp.getVRegValue();
              }
            }
            else {
              if(mOp.getVRegValue() == SB[0]->getBasicBlock())
                mOp.setValueReg(llvmKernelBB[0]);
              else {
                if(llvm_epilogues.size() > 0)
                  mOp.setValueReg(llvm_epilogues[0][0]);
              }
            }
          }
        }
      }
    }

    //Update kernelLLVM branches
    const BranchInst *branchVal = dyn_cast<BranchInst>(SB[0]->getBasicBlock()->getTerminator());
  
    //deal with inner branch
    if(i < machineKernelBB.size()-1) {
      
      //Find our side exit LLVM basic block
      BasicBlock *sideExit = 0;
      for(unsigned s = 0; s < branchVal->getNumSuccessors(); ++s) {
        if(branchVal->getSuccessor(s) != SB[i+1]->getBasicBlock())
          sideExit = branchVal->getSuccessor(s);
      }
      assert(sideExit && "Must have side exit llvm basic block");
      TerminatorInst *newBranch = new BranchInst(sideExit,
                                                 llvmKernelBB[i+1],
                                                 branchVal->getCondition(),
                                                 llvmKernelBB[i]);
    }
    else {
      //Deal with outter branches
      if(epilogues.size() > 0) {
        TerminatorInst *newBranch = new BranchInst(llvmKernelBB[0],
                                                   llvm_epilogues[0][0],
                                                   branchVal->getCondition(),
                                                   llvmKernelBB[i]);
      }
      else {
        TerminatorInst *newBranch = new BranchInst(llvmKernelBB[0],
                                                   kernel_exit,
                                                   branchVal->getCondition(),
                                                   llvmKernelBB[i]);
      }
    }
  }

  if(schedule.getMaxStage() != 0) {
  
    //Lastly add unconditional branches for the epilogues
    for(unsigned i = 0; i <  epilogues.size(); ++i) {

      for(unsigned j=0; j < epilogues[i].size(); ++j) {
        //Now since we don't have fall throughs, add a unconditional
        //branch to the next prologue
        
        //Before adding these, we need to check if the epilogue already has
        //a branch in it
        bool hasBranch = false;
        /*if(j < epilogues[i].size()-1) {
          MachineBasicBlock *currentMBB = epilogues[i][j];
          for(MachineBasicBlock::reverse_iterator mInst = currentMBB->rbegin(), mInstEnd = currentMBB->rend(); mInst != mInstEnd; ++mInst) {
            
            MachineOpCode OC = mInst->getOpcode();
            
            //If its a branch update its branchto
            if(TMI->isBranch(OC)) {
              hasBranch = true;
              for(unsigned opNum = 0; opNum < mInst->getNumOperands(); ++opNum) {
                MachineOperand &mOp = mInst->getOperand(opNum);
                if (mOp.getType() == MachineOperand::MO_PCRelativeDisp) {
                 
                  if(mOp.getVRegValue() != sideExits[SB[j]]) {
                    mOp.setValueReg(llvm_epilogues[i][j+1]);
                  }
                  
                }
              }
              
              
              DEBUG(std::cerr << "New Epilogue Branch: " << *mInst << "\n");
            }
          }
          if(hasBranch) {
            const BranchInst *branchVal = dyn_cast<BranchInst>(SB[j]->getBasicBlock()->getTerminator());
            TerminatorInst *newBranch = new BranchInst((BasicBlock*)sideExits[SB[j]],
                                                       llvm_epilogues[i][j+1],
                                                       branchVal->getCondition(),
                                                       llvm_epilogues[i][j]);
          }
          }*/

        if(!hasBranch) {
        
          //Handle inner branches
          if(j < epilogues[i].size()-1) {
            BuildMI(epilogues[i][j], V9::BA, 1).addPCDisp(llvm_epilogues[i][j+1]);
            TerminatorInst *newBranch = new BranchInst(llvm_epilogues[i][j+1],
                                                       llvm_epilogues[i][j]);
          }
          else {
            
            //Check if this is the last epilogue
            if(i != epilogues.size()-1) {
              BuildMI(epilogues[i][j], V9::BA, 1).addPCDisp(llvm_epilogues[i+1][0]);
              //Add unconditional branch to end of epilogue
              TerminatorInst *newBranch = new BranchInst(llvm_epilogues[i+1][0],
                                                         llvm_epilogues[i][j]);
              
            }
            else {
              BuildMI(epilogues[i][j], V9::BA, 1).addPCDisp(kernel_exit);
              TerminatorInst *newBranch = new BranchInst(kernel_exit, llvm_epilogues[i][j]);
            }
          }
          
          //Add one more nop!
          BuildMI(epilogues[i][j], V9::NOP, 0);
          
        }
      }
    }
  }

  //Find all llvm basic blocks that branch to the loop entry and
  //change to our first prologue.
  const BasicBlock *llvmBB = SB[0]->getBasicBlock();
  
  std::vector<const BasicBlock*>Preds (pred_begin(llvmBB), pred_end(llvmBB));
  
  for(std::vector<const BasicBlock*>::iterator P = Preds.begin(), 
        PE = Preds.end(); P != PE; ++P) {
    if(*P == SB[SB.size()-1]->getBasicBlock())
       continue;
     else {
       DEBUG(std::cerr << "Found our entry BB\n");
       DEBUG((*P)->print(std::cerr));
       //Get the Terminator instruction for this basic block and print it out
       //DEBUG(std::cerr << *((*P)->getTerminator()) << "\n");
       
       //Update the terminator
       TerminatorInst *term = ((BasicBlock*)*P)->getTerminator();
       for(unsigned i=0; i < term->getNumSuccessors(); ++i) {
         if(term->getSuccessor(i) == llvmBB) {
           DEBUG(std::cerr << "Replacing successor bb\n");
           if(llvm_prologues.size() > 0) {
             term->setSuccessor(i, llvm_prologues[0][0]);

             DEBUG(std::cerr << "New Term" << *((*P)->getTerminator()) << "\n");

             //Also update its corresponding machine instruction
             MachineCodeForInstruction & tempMvec =
               MachineCodeForInstruction::get(term);
             for (unsigned j = 0; j < tempMvec.size(); j++) {
               MachineInstr *temp = tempMvec[j];
               MachineOpCode opc = temp->getOpcode();
               if(TMI->isBranch(opc)) {
                 DEBUG(std::cerr << *temp << "\n");
                 //Update branch
                 for(unsigned opNum = 0; opNum < temp->getNumOperands(); ++opNum) {
                   MachineOperand &mOp = temp->getOperand(opNum);
                   if (mOp.getType() == MachineOperand::MO_PCRelativeDisp) {
                     if(mOp.getVRegValue() == llvmBB)
                       mOp.setValueReg(llvm_prologues[0][0]);
                   }
                 }
               }
             }
           }
           else {
             term->setSuccessor(i, llvmKernelBB[0]);

             //Also update its corresponding machine instruction
             MachineCodeForInstruction & tempMvec =
               MachineCodeForInstruction::get(term);
             for(unsigned j = 0; j < tempMvec.size(); j++) {
               MachineInstr *temp = tempMvec[j];
               MachineOpCode opc = temp->getOpcode();
               if(TMI->isBranch(opc)) {
                 DEBUG(std::cerr << *temp << "\n");
                 //Update branch
                 for(unsigned opNum = 0; opNum < temp->getNumOperands(); ++opNum) {
                   MachineOperand &mOp = temp->getOperand(opNum);
                   if(mOp.getType() == MachineOperand::MO_PCRelativeDisp) {
                     if(mOp.getVRegValue() == llvmBB)
                       mOp.setValueReg(llvmKernelBB[0]);
                   }
                 }
               }
             }
           }
         }
       }
       break;
     }
  }

}


void ModuloSchedulingSBPass::writePrologues(std::vector<std::vector<MachineBasicBlock *> > &prologues, std::vector<const MachineBasicBlock*> &origSB, std::vector<std::vector<BasicBlock*> > &llvm_prologues, std::map<const Value*, std::pair<const MachineInstr*, int> > &valuesToSave, std::map<Value*, std::map<int, Value*> > &newValues, std::map<Value*, MachineBasicBlock*> &newValLocation) {

  //Keep a map to easily know whats in the kernel
  std::map<int, std::set<const MachineInstr*> > inKernel;
  int maxStageCount = 0;

  //Keep a map of new values we consumed in case they need to be added back
  std::map<Value*, std::map<int, Value*> > consumedValues;

  DEBUG(schedule.print(std::cerr));

  for(MSScheduleSB::kernel_iterator I = schedule.kernel_begin(), E = schedule.kernel_end(); I != E; ++I) {
    maxStageCount = std::max(maxStageCount, I->second);

    //Put int the map so we know what instructions in each stage are in the kernel
    DEBUG(std::cerr << "Inserting instruction " << *(I->first) << " into map at stage " << I->second << "\n");
    inKernel[I->second].insert(I->first);
  }

  //Get target information to look at machine operands
  const TargetInstrInfo *mii = target.getInstrInfo();

 //Now write the prologues
  for(int i = 0; i < maxStageCount; ++i) {
    std::vector<MachineBasicBlock*> current_prologue;
    std::vector<BasicBlock*> current_llvm_prologue;

    for(std::vector<const MachineBasicBlock*>::iterator MB = origSB.begin(), 
          MBE = origSB.end(); MB != MBE; ++MB) {
      const MachineBasicBlock *MBB = *MB;
      //Create new llvm and machine bb
      BasicBlock *llvmBB = new BasicBlock("PROLOGUE", (Function*) (MBB->getBasicBlock()->getParent()));
      MachineBasicBlock *machineBB = new MachineBasicBlock(llvmBB);

      DEBUG(std::cerr << "i=" << i << "\n");

      for(int j = i; j >= 0; --j) {
        //iterate over instructions in original bb
        for(MachineBasicBlock::const_iterator MI = MBB->begin(), 
              ME = MBB->end(); ME != MI; ++MI) {
          if(inKernel[j].count(&*MI)) {
            MachineInstr *instClone = MI->clone();
            machineBB->push_back(instClone);
            
            //If its a branch, insert a nop
            if(mii->isBranch(instClone->getOpcode()))
              BuildMI(machineBB, V9::NOP, 0);
        
            
            DEBUG(std::cerr << "Cloning: " << *MI << "\n");
            
            //After cloning, we may need to save the value that this instruction defines
            for(unsigned opNum=0; opNum < MI->getNumOperands(); ++opNum) {
              Instruction *tmp;
              
              //get machine operand
              MachineOperand &mOp = instClone->getOperand(opNum);
              if(mOp.getType() == MachineOperand::MO_VirtualRegister 
                 && mOp.isDef()) {

                //Check if this is a value we should save
                if(valuesToSave.count(mOp.getVRegValue())) {
                  //Save copy in tmpInstruction
                  tmp = new TmpInstruction(mOp.getVRegValue());
                  
                  //Add TmpInstruction to safe LLVM Instruction MCFI
                  MachineCodeForInstruction & tempMvec = MachineCodeForInstruction::get(defaultInst);
                  tempMvec.addTemp((Value*) tmp);

                  DEBUG(std::cerr << "Value: " << *(mOp.getVRegValue()) 
                        << " New Value: " << *tmp << " Stage: " << i << "\n");
                
                newValues[mOp.getVRegValue()][i]= tmp;
                newValLocation[tmp] = machineBB;

                DEBUG(std::cerr << "Machine Instr Operands: " 
                      << *(mOp.getVRegValue()) << ", 0, " << *tmp << "\n");
                
                //Create machine instruction and put int machineBB
                MachineInstr *saveValue;
                if(mOp.getVRegValue()->getType() == Type::FloatTy)
                  saveValue = BuildMI(machineBB, V9::FMOVS, 3).addReg(mOp.getVRegValue()).addRegDef(tmp);
                else if(mOp.getVRegValue()->getType() == Type::DoubleTy)
                  saveValue = BuildMI(machineBB, V9::FMOVD, 3).addReg(mOp.getVRegValue()).addRegDef(tmp);
                else
                  saveValue = BuildMI(machineBB, V9::ORr, 3).addReg(mOp.getVRegValue()).addImm(0).addRegDef(tmp);
        

                DEBUG(std::cerr << "Created new machine instr: " << *saveValue << "\n");
                }
              }

              //We may also need to update the value that we use if
              //its from an earlier prologue
              if(j != 0) {
                if(mOp.getType() == MachineOperand::MO_VirtualRegister && mOp.isUse()) {
                  if(newValues.count(mOp.getVRegValue())) {
                    if(newValues[mOp.getVRegValue()].count(i-1)) {
                      Value *oldV =  mOp.getVRegValue();
                      DEBUG(std::cerr << "Replaced this value: " << mOp.getVRegValue() << " With:" << (newValues[mOp.getVRegValue()][i-1]) << "\n");
                      //Update the operand with the right value
                      mOp.setValueReg(newValues[mOp.getVRegValue()][i-1]);
                      
                      //Remove this value since we have consumed it
                      //NOTE: Should this only be done if j != maxStage?
                      consumedValues[oldV][i-1] = (newValues[oldV][i-1]);
                      DEBUG(std::cerr << "Deleted value: " << consumedValues[oldV][i-1] << "\n");
                      newValues[oldV].erase(i-1);
                    }
                  }
                  else
                    if(consumedValues.count(mOp.getVRegValue()))
                      assert(!consumedValues[mOp.getVRegValue()].count(i-1) && "Found a case where we need the value");
                }
              }
            }
          }
        }
      }
        (((MachineBasicBlock*)MBB)->getParent())->getBasicBlockList().push_back(machineBB);
        current_prologue.push_back(machineBB);
        current_llvm_prologue.push_back(llvmBB);
    }
    prologues.push_back(current_prologue);
    llvm_prologues.push_back(current_llvm_prologue);

  }
}


void ModuloSchedulingSBPass::writeEpilogues(std::vector<std::vector<MachineBasicBlock*> > &epilogues, std::vector<const MachineBasicBlock*> &origSB, std::vector<std::vector<BasicBlock*> > &llvm_epilogues, std::map<const Value*, std::pair<const MachineInstr*, int> > &valuesToSave, std::map<Value*, std::map<int, Value*> > &newValues,std::map<Value*, MachineBasicBlock*> &newValLocation, std::map<Value*, std::map<int, Value*> > &kernelPHIs ) {

  std::map<int, std::set<const MachineInstr*> > inKernel;
   const TargetInstrInfo *MTI = target.getInstrInfo();

  for(MSScheduleSB::kernel_iterator I = schedule.kernel_begin(), E = schedule.kernel_end(); I != E; ++I) {

    //Put int the map so we know what instructions in each stage are in the kernel
    inKernel[I->second].insert(I->first);
  }

  std::map<Value*, Value*> valPHIs;

  //some debug stuff, will remove later
  DEBUG(for(std::map<Value*, std::map<int, Value*> >::iterator V = newValues.begin(), E = newValues.end(); V !=E; ++V) {
    std::cerr << "Old Value: " << *(V->first) << "\n";
    for(std::map<int, Value*>::iterator I = V->second.begin(), IE = V->second.end(); I != IE; ++I)
      std::cerr << "Stage: " << I->first << " Value: " << *(I->second) << "\n";
  });


  //Now write the epilogues
  for(int i = schedule.getMaxStage()-1; i >= 0; --i) {
    std::vector<MachineBasicBlock*> current_epilogue;
    std::vector<BasicBlock*> current_llvm_epilogue;
    
    for(std::vector<const MachineBasicBlock*>::iterator MB = origSB.begin(), MBE = origSB.end(); MB != MBE; ++MB) {
      const MachineBasicBlock *MBB = *MB;

      BasicBlock *llvmBB = new BasicBlock("EPILOGUE", (Function*) (MBB->getBasicBlock()->getParent()));
      MachineBasicBlock *machineBB = new MachineBasicBlock(llvmBB);
      
      DEBUG(std::cerr << " Epilogue #: " << i << "\n");
      
      std::map<Value*, int> inEpilogue;
      
      for(MachineBasicBlock::const_iterator MI = MBB->begin(), ME = MBB->end(); ME != MI; ++MI) {
        for(int j=schedule.getMaxStage(); j > i; --j) {
          if(inKernel[j].count(&*MI)) {
            DEBUG(std::cerr << "Cloning instruction " << *MI << "\n");
            MachineInstr *clone = MI->clone();
            
            //Update operands that need to use the result from the phi
            for(unsigned opNum=0; opNum < clone->getNumOperands(); ++opNum) {
              //get machine operand
              const MachineOperand &mOp = clone->getOperand(opNum);
              
              if((mOp.getType() == MachineOperand::MO_VirtualRegister && mOp.isUse())) {
                
                DEBUG(std::cerr << "Writing PHI for " << (mOp.getVRegValue()) << "\n");
                
                //If this is the last instructions for the max iterations ago, don't update operands
                if(inEpilogue.count(mOp.getVRegValue()))
                  if(inEpilogue[mOp.getVRegValue()] == i)
                    continue;
                
                //Quickly write appropriate phis for this operand
                if(newValues.count(mOp.getVRegValue())) {
                  if(newValues[mOp.getVRegValue()].count(i)) {
                    Instruction *tmp = new TmpInstruction(newValues[mOp.getVRegValue()][i]);
                    
                    //Get machine code for this instruction
                    MachineCodeForInstruction & tempMvec = MachineCodeForInstruction::get(defaultInst);
                    tempMvec.addTemp((Value*) tmp);
                    
                    //assert of no kernelPHI for this value
                    assert(kernelPHIs[mOp.getVRegValue()][i] !=0 && "Must have final kernel phi to construct epilogue phi");
                    
                    MachineInstr *saveValue = BuildMI(machineBB, V9::PHI, 3).addReg(newValues[mOp.getVRegValue()][i]).addReg(kernelPHIs[mOp.getVRegValue()][i]).addRegDef(tmp);
                    DEBUG(std::cerr << "Resulting PHI: " << *saveValue << "\n");
                    valPHIs[mOp.getVRegValue()] = tmp;
                  }
                }
                
                if(valPHIs.count(mOp.getVRegValue())) {
                  //Update the operand in the cloned instruction
                  clone->getOperand(opNum).setValueReg(valPHIs[mOp.getVRegValue()]);
                }
              }
              else if((mOp.getType() == MachineOperand::MO_VirtualRegister && mOp.isDef())) {
                inEpilogue[mOp.getVRegValue()] = i;
              }
             
            }
            machineBB->push_back(clone);
            //if(MTI->isBranch(clone->getOpcode()))
            //BuildMI(machineBB, V9::NOP, 0);
          }
        }
      }
      (((MachineBasicBlock*)MBB)->getParent())->getBasicBlockList().push_back(machineBB);
      current_epilogue.push_back(machineBB);
      current_llvm_epilogue.push_back(llvmBB);
    }
     
    DEBUG(std::cerr << "EPILOGUE #" << i << "\n");
    DEBUG(for(std::vector<MachineBasicBlock*>::iterator B = current_epilogue.begin(), BE = current_epilogue.end(); B != BE; ++B) {
            (*B)->print(std::cerr);});
    
    epilogues.push_back(current_epilogue);
    llvm_epilogues.push_back(current_llvm_epilogue);
  }
}

void ModuloSchedulingSBPass::writeKernel(std::vector<BasicBlock*> &llvmBB, std::vector<MachineBasicBlock*> &machineBB, std::map<const Value*, std::pair<const MachineInstr*, int> > &valuesToSave, std::map<Value*, std::map<int, Value*> > &newValues, std::map<Value*, MachineBasicBlock*> &newValLocation, std::map<Value*, std::map<int, Value*> > &kernelPHIs) {

  //Keep track of operands that are read and saved from a previous iteration. The new clone
  //instruction will use the result of the phi instead.
  std::map<Value*, Value*> finalPHIValue;
  std::map<Value*, Value*> kernelValue;

  //Branches are a special case
  std::vector<MachineInstr*> branches;

  //Get target information to look at machine operands
  const TargetInstrInfo *mii = target.getInstrInfo();
  unsigned index = 0;
  int numBr = 0;
  bool seenBranch = false;

  //Create TmpInstructions for the final phis
  for(MSScheduleSB::kernel_iterator I = schedule.kernel_begin(), E = schedule.kernel_end(); I != E; ++I) {

   DEBUG(std::cerr << "Stage: " << I->second << " Inst: " << *(I->first) << "\n";);

   //Clone instruction
   const MachineInstr *inst = I->first;
   MachineInstr *instClone = inst->clone();

   if(seenBranch && !mii->isBranch(instClone->getOpcode())) {
     index++;
     seenBranch = false;
     numBr = 0;
   }
   else if(seenBranch && (numBr == 2)) {
     index++;
     numBr = 0;
   }

   //Insert into machine basic block
   assert(index < machineBB.size() && "Must have a valid index into kernel MBBs");
   machineBB[index]->push_back(instClone);

   if(mii->isBranch(instClone->getOpcode())) {
     BuildMI(machineBB[index], V9::NOP, 0);

     seenBranch = true;
     numBr++;
   }
   
   DEBUG(std::cerr <<  "Cloned Inst: " << *instClone << "\n");

   //Loop over Machine Operands
   for(unsigned i=0; i < inst->getNumOperands(); ++i) {
     //get machine operand
     const MachineOperand &mOp = inst->getOperand(i);

     if(I->second != 0) {
       if(mOp.getType() == MachineOperand::MO_VirtualRegister && mOp.isUse()) {

         //Check to see where this operand is defined if this instruction is from max stage
         if(I->second == schedule.getMaxStage()) {
           DEBUG(std::cerr << "VREG: " << *(mOp.getVRegValue()) << "\n");
         }

         //If its in the value saved, we need to create a temp instruction and use that instead
         if(valuesToSave.count(mOp.getVRegValue())) {

           //Check if we already have a final PHI value for this
           if(!finalPHIValue.count(mOp.getVRegValue())) {
             //Only create phi if the operand def is from a stage before this one
             if(schedule.defPreviousStage(mOp.getVRegValue(), I->second)) {
             TmpInstruction *tmp = new TmpInstruction(mOp.getVRegValue());
        
             //Get machine code for this instruction
             MachineCodeForInstruction & tempMvec = MachineCodeForInstruction::get(defaultInst);
             tempMvec.addTemp((Value*) tmp);
        
             //Update the operand in the cloned instruction
             instClone->getOperand(i).setValueReg(tmp);
        
             //save this as our final phi
             finalPHIValue[mOp.getVRegValue()] = tmp;
             newValLocation[tmp] = machineBB[index];
             }
           }
           else {
             //Use the previous final phi value
             instClone->getOperand(i).setValueReg(finalPHIValue[mOp.getVRegValue()]);
           }
         }
       }
     }
     if(I->second != schedule.getMaxStage()) {
       if(mOp.getType() == MachineOperand::MO_VirtualRegister && mOp.isDef()) {
         if(valuesToSave.count(mOp.getVRegValue())) {
        
           TmpInstruction *tmp = new TmpInstruction(mOp.getVRegValue());
        
           //Get machine code for this instruction
           MachineCodeForInstruction & tempVec = MachineCodeForInstruction::get(defaultInst);
           tempVec.addTemp((Value*) tmp);

           //Create new machine instr and put in MBB
           MachineInstr *saveValue;
           if(mOp.getVRegValue()->getType() == Type::FloatTy)
             saveValue = BuildMI(machineBB[index], V9::FMOVS, 3).addReg(mOp.getVRegValue()).addRegDef(tmp);
           else if(mOp.getVRegValue()->getType() == Type::DoubleTy)
             saveValue = BuildMI(machineBB[index], V9::FMOVD, 3).addReg(mOp.getVRegValue()).addRegDef(tmp);
           else
             saveValue = BuildMI(machineBB[index], V9::ORr, 3).addReg(mOp.getVRegValue()).addImm(0).addRegDef(tmp);
        
        
           //Save for future cleanup
           kernelValue[mOp.getVRegValue()] = tmp;
           newValLocation[tmp] = machineBB[index];
           kernelPHIs[mOp.getVRegValue()][schedule.getMaxStage()-1] = tmp;
         }
       }
     }
   }

 }

  //Loop over each value we need to generate phis for
 for(std::map<Value*, std::map<int, Value*> >::iterator V = newValues.begin(),
       E = newValues.end(); V != E; ++V) {


   DEBUG(std::cerr << "Writing phi for" << *(V->first));
   DEBUG(std::cerr << "\nMap of Value* for this phi\n");
   DEBUG(for(std::map<int, Value*>::iterator I = V->second.begin(),
               IE = V->second.end(); I != IE; ++I) {
     std::cerr << "Stage: " << I->first;
     std::cerr << " Value: " << *(I->second) << "\n";
   });

   //If we only have one current iteration live, its safe to set
   //lastPhi = to kernel value
   if(V->second.size() == 1) {
     assert(kernelValue[V->first] != 0 && "Kernel value* must exist to create phi");
     MachineInstr *saveValue = BuildMI(*machineBB[0], machineBB[0]->begin(),V9::PHI, 3).addReg(V->second.begin()->second).addReg(kernelValue[V->first]).addRegDef(finalPHIValue[V->first]);
     DEBUG(std::cerr << "Resulting PHI (one live): " << *saveValue << "\n");
     kernelPHIs[V->first][V->second.begin()->first] = kernelValue[V->first];
     DEBUG(std::cerr << "Put kernel phi in at stage: " << schedule.getMaxStage()-1 << " (map stage = " << V->second.begin()->first << ")\n");
   }
   else {

     //Keep track of last phi created.
     Instruction *lastPhi = 0;

     unsigned count = 1;
     //Loop over the the map backwards to generate phis
     for(std::map<int, Value*>::reverse_iterator I = V->second.rbegin(), IE = V->second.rend();
         I != IE; ++I) {

       if(count < (V->second).size()) {
         if(lastPhi == 0) {
           lastPhi = new TmpInstruction(I->second);

           //Get machine code for this instruction
           MachineCodeForInstruction & tempMvec = MachineCodeForInstruction::get(defaultInst);
           tempMvec.addTemp((Value*) lastPhi);

           MachineInstr *saveValue = BuildMI(*machineBB[0], machineBB[0]->begin(), V9::PHI, 3).addReg(kernelValue[V->first]).addReg(I->second).addRegDef(lastPhi);
           DEBUG(std::cerr << "Resulting PHI: " << *saveValue << "\n");
           newValLocation[lastPhi] = machineBB[0];
         }
         else {
           Instruction *tmp = new TmpInstruction(I->second);

           //Get machine code for this instruction
           MachineCodeForInstruction & tempMvec = MachineCodeForInstruction::get(defaultInst);
           tempMvec.addTemp((Value*) tmp);
        

           MachineInstr *saveValue = BuildMI(*machineBB[0], machineBB[0]->begin(), V9::PHI, 3).addReg(lastPhi).addReg(I->second).addRegDef(tmp);
           DEBUG(std::cerr << "Resulting PHI: " << *saveValue << "\n");
           lastPhi = tmp;
           kernelPHIs[V->first][I->first] = lastPhi;
           newValLocation[lastPhi] = machineBB[0];
         }
       }
       //Final phi value
       else {
         //The resulting value must be the Value* we created earlier
         assert(lastPhi != 0 && "Last phi is NULL!\n");
         MachineInstr *saveValue = BuildMI(*machineBB[0], machineBB[0]->begin(), V9::PHI, 3).addReg(lastPhi).addReg(I->second).addRegDef(finalPHIValue[V->first]);
         DEBUG(std::cerr << "Resulting PHI: " << *saveValue << "\n");
         kernelPHIs[V->first][I->first] = finalPHIValue[V->first];
       }

       ++count;
     }

   }
 }
}


void ModuloSchedulingSBPass::removePHIs(std::vector<const MachineBasicBlock*> &SB, std::vector<std::vector<MachineBasicBlock*> > &prologues, std::vector<std::vector<MachineBasicBlock*> > &epilogues, std::vector<MachineBasicBlock*> &kernelBB, std::map<Value*, MachineBasicBlock*> &newValLocation) {

  //Worklist to delete things
  std::vector<std::pair<MachineBasicBlock*, MachineBasicBlock::iterator> > worklist;

  //Worklist of TmpInstructions that need to be added to a MCFI
  std::vector<Instruction*> addToMCFI;

  //Worklist to add OR instructions to end of kernel so not to invalidate the iterator
  //std::vector<std::pair<Instruction*, Value*> > newORs;

  const TargetInstrInfo *TMI = target.getInstrInfo();

  //Start with the kernel and for each phi insert a copy for the phi
  //def and for each arg
  //phis are only in the first BB in the kernel
  for(MachineBasicBlock::iterator I = kernelBB[0]->begin(), E = kernelBB[0]->end(); 
      I != E; ++I) {

    DEBUG(std::cerr << "Looking at Instr: " << *I << "\n");
    
    //Get op code and check if its a phi
    if(I->getOpcode() == V9::PHI) {

      DEBUG(std::cerr << "Replacing PHI: " << *I << "\n");
      Instruction *tmp = 0;

      for(unsigned i = 0; i < I->getNumOperands(); ++i) {
        
        //Get Operand
        const MachineOperand &mOp = I->getOperand(i);
        assert(mOp.getType() == MachineOperand::MO_VirtualRegister 
               && "Should be a Value*\n");
        
        if(!tmp) {
          tmp = new TmpInstruction(mOp.getVRegValue());
          addToMCFI.push_back(tmp);
        }

        //Now for all our arguments we read, OR to the new
        //TmpInstruction that we created
        if(mOp.isUse()) {
          DEBUG(std::cerr << "Use: " << mOp << "\n");
          //Place a copy at the end of its BB but before the branches
          assert(newValLocation.count(mOp.getVRegValue()) && "We must know where this value is located\n");
          //Reverse iterate to find the branches, we can safely assume no instructions have been
          //put in the nop positions
          for(MachineBasicBlock::iterator inst = --(newValLocation[mOp.getVRegValue()])->end(), endBB = (newValLocation[mOp.getVRegValue()])->begin(); inst != endBB; --inst) {
            MachineOpCode opc = inst->getOpcode();
            if(TMI->isBranch(opc) || TMI->isNop(opc))
              continue;
            else {
              if(mOp.getVRegValue()->getType() == Type::FloatTy)
                BuildMI(*(newValLocation[mOp.getVRegValue()]), ++inst, V9::FMOVS, 3).addReg(mOp.getVRegValue()).addRegDef(tmp);
              else if(mOp.getVRegValue()->getType() == Type::DoubleTy)
                BuildMI(*(newValLocation[mOp.getVRegValue()]), ++inst, V9::FMOVD, 3).addReg(mOp.getVRegValue()).addRegDef(tmp);
              else
                BuildMI(*(newValLocation[mOp.getVRegValue()]), ++inst, V9::ORr, 3).addReg(mOp.getVRegValue()).addImm(0).addRegDef(tmp);
        
              break;
            }
        
          }

        }
        else {
          //Remove the phi and replace it with an OR
          DEBUG(std::cerr << "Def: " << mOp << "\n");
          //newORs.push_back(std::make_pair(tmp, mOp.getVRegValue()));
          if(tmp->getType() == Type::FloatTy)
            BuildMI(*kernelBB[0], I, V9::FMOVS, 3).addReg(tmp).addRegDef(mOp.getVRegValue());
          else if(tmp->getType() == Type::DoubleTy)
            BuildMI(*kernelBB[0], I, V9::FMOVD, 3).addReg(tmp).addRegDef(mOp.getVRegValue());
          else
            BuildMI(*kernelBB[0], I, V9::ORr, 3).addReg(tmp).addImm(0).addRegDef(mOp.getVRegValue());
        
        
          worklist.push_back(std::make_pair(kernelBB[0], I));
        }
        
      }

    }


  }

  //Add TmpInstructions to some MCFI
  if(addToMCFI.size() > 0) {
    MachineCodeForInstruction & tempMvec = MachineCodeForInstruction::get(defaultInst);
    for(unsigned x = 0; x < addToMCFI.size(); ++x) {
      tempMvec.addTemp(addToMCFI[x]);
    }
    addToMCFI.clear();
  }


  //Remove phis from epilogue
  for(std::vector<std::vector<MachineBasicBlock*> >::iterator MB = epilogues.begin(), 
        ME = epilogues.end(); MB != ME; ++MB) {
    
    for(std::vector<MachineBasicBlock*>::iterator currentMBB = MB->begin(), currentME = MB->end(); currentMBB != currentME; ++currentMBB) {
      
      for(MachineBasicBlock::iterator I = (*currentMBB)->begin(), 
            E = (*currentMBB)->end(); I != E; ++I) {

        DEBUG(std::cerr << "Looking at Instr: " << *I << "\n");
        //Get op code and check if its a phi
        if(I->getOpcode() == V9::PHI) {
          Instruction *tmp = 0;

          for(unsigned i = 0; i < I->getNumOperands(); ++i) {
            //Get Operand
            const MachineOperand &mOp = I->getOperand(i);
            assert(mOp.getType() == MachineOperand::MO_VirtualRegister && "Should be a Value*\n");
        
            if(!tmp) {
              tmp = new TmpInstruction(mOp.getVRegValue());
              addToMCFI.push_back(tmp);
            }
        
            //Now for all our arguments we read, OR to the new TmpInstruction that we created
            if(mOp.isUse()) {
              DEBUG(std::cerr << "Use: " << mOp << "\n");
              //Place a copy at the end of its BB but before the branches
              assert(newValLocation.count(mOp.getVRegValue()) && "We must know where this value is located\n");
              //Reverse iterate to find the branches, we can safely assume no instructions have been
              //put in the nop positions
              for(MachineBasicBlock::iterator inst = --(newValLocation[mOp.getVRegValue()])->end(), endBB = (newValLocation[mOp.getVRegValue()])->begin(); inst != endBB; --inst) {
                MachineOpCode opc = inst->getOpcode();
                if(TMI->isBranch(opc) || TMI->isNop(opc))
                  continue;
                else {
                  if(mOp.getVRegValue()->getType() == Type::FloatTy)
                    BuildMI(*(newValLocation[mOp.getVRegValue()]), ++inst, V9::FMOVS, 3).addReg(mOp.getVRegValue()).addRegDef(tmp);
                  else if(mOp.getVRegValue()->getType() == Type::DoubleTy)
                    BuildMI(*(newValLocation[mOp.getVRegValue()]), ++inst, V9::FMOVD, 3).addReg(mOp.getVRegValue()).addRegDef(tmp);
                  else
                    BuildMI(*(newValLocation[mOp.getVRegValue()]), ++inst, V9::ORr, 3).addReg(mOp.getVRegValue()).addImm(0).addRegDef(tmp);
                  
                  
                  break;
                }
                
              }
              
            }
            else {
              //Remove the phi and replace it with an OR
              DEBUG(std::cerr << "Def: " << mOp << "\n");
              if(tmp->getType() == Type::FloatTy)
                BuildMI(**currentMBB, I, V9::FMOVS, 3).addReg(tmp).addRegDef(mOp.getVRegValue());
              else if(tmp->getType() == Type::DoubleTy)
                BuildMI(**currentMBB, I, V9::FMOVD, 3).addReg(tmp).addRegDef(mOp.getVRegValue());
              else
                BuildMI(**currentMBB, I, V9::ORr, 3).addReg(tmp).addImm(0).addRegDef(mOp.getVRegValue());
              
              worklist.push_back(std::make_pair(*currentMBB,I));
            }
          }
        }
      }
    }
  }


  if(addToMCFI.size() > 0) {
    MachineCodeForInstruction & tempMvec = MachineCodeForInstruction::get(defaultInst);
    for(unsigned x = 0; x < addToMCFI.size(); ++x) {
      tempMvec.addTemp(addToMCFI[x]);
    }
    addToMCFI.clear();
  }
  
    //Delete the phis
  for(std::vector<std::pair<MachineBasicBlock*, MachineBasicBlock::iterator> >::iterator I =  worklist.begin(), E = worklist.end(); I != E; ++I) {
    DEBUG(std::cerr << "Deleting PHI " << *I->second << "\n");
    I->first->erase(I->second);
                
  }


  assert((addToMCFI.size() == 0) && "We should have added all TmpInstructions to some MachineCodeForInstruction");
}




void ModuloSchedulingSBPass::writeSideExits(std::vector<std::vector<MachineBasicBlock *> > &prologues, std::vector<std::vector<BasicBlock*> > &llvm_prologues, std::vector<std::vector<MachineBasicBlock *> > &epilogues, std::vector<std::vector<BasicBlock*> > &llvm_epilogues, std::map<const MachineBasicBlock*, Value*> &sideExits, std::map<MachineBasicBlock*, std::vector<std::pair<MachineInstr*, int> > > &instrsMovedDown, std::vector<const MachineBasicBlock*> &SB, std::vector<MachineBasicBlock*> &kernelMBBs, std::map<MachineBasicBlock*, int> branchStage) {

  const TargetInstrInfo *TMI = target.getInstrInfo();

  //Repeat for each side exit
  for(unsigned sideExitNum = 0; sideExitNum < SB.size()-1; ++sideExitNum) {

    std::vector<std::vector<BasicBlock*> > side_llvm_epilogues;
    std::vector<std::vector<MachineBasicBlock*> > side_epilogues;
    MachineBasicBlock* sideMBB;
    BasicBlock* sideBB;

    //Create side exit blocks
    //Get the LLVM basic block
    BasicBlock *bb = (BasicBlock*) SB[sideExitNum]->getBasicBlock();
    MachineBasicBlock *mbb = (MachineBasicBlock*) SB[sideExitNum];
    
    int stage = branchStage[mbb];

    //Create new basic blocks for our side exit instructios that were moved below the branch
    sideBB = new BasicBlock("SideExit", (Function*) bb->getParent());
    sideMBB = new MachineBasicBlock(sideBB);
    (((MachineBasicBlock*)SB[0])->getParent())->getBasicBlockList().push_back(sideMBB);

    
    if(instrsMovedDown.count(mbb)) {
      for(std::vector<std::pair<MachineInstr*, int> >::iterator I = instrsMovedDown[mbb].begin(), E = instrsMovedDown[mbb].end(); I != E; ++I) {
        if(branchStage[mbb] == I->second)
          sideMBB->push_back((I->first)->clone());
      }
        
      //Add unconditional branches to original exits
      BuildMI(sideMBB, V9::BA, 1).addPCDisp(sideExits[mbb]);
      BuildMI(sideMBB, V9::NOP, 0);
      
      //Add unconditioal branch to llvm BB
      BasicBlock *extBB = dyn_cast<BasicBlock>(sideExits[mbb]);
      assert(extBB && "Side exit basicblock can not be null");
      TerminatorInst *newBranch = new BranchInst(extBB, sideBB);
    }

     //Clone epilogues and update their branches, one cloned epilogue set per side exit
    //only clone epilogues that are from a greater stage!
    for(unsigned i = 0; i < epilogues.size()-stage; ++i) {
      std::vector<MachineBasicBlock*> MB = epilogues[i];
      
      std::vector<MachineBasicBlock*> newEp;
      std::vector<BasicBlock*> newLLVMEp;
    
      for(std::vector<MachineBasicBlock*>::iterator currentMBB = MB.begin(), 
            lastMBB = MB.end(); currentMBB != lastMBB; ++currentMBB) {
        BasicBlock *tmpBB = new BasicBlock("SideEpilogue", (Function*) (*currentMBB)->getBasicBlock()->getParent());
        MachineBasicBlock *tmp = new MachineBasicBlock(tmpBB);
      
        //Clone instructions and insert into new MBB
        for(MachineBasicBlock::iterator I = (*currentMBB)->begin(), 
              E = (*currentMBB)->end(); I != E; ++I) {
        
          MachineInstr *clone = I->clone();
          if(clone->getOpcode() == V9::BA && (currentMBB+1 == lastMBB)) {
            //update branch to side exit
            for(unsigned i = 0; i < clone->getNumOperands(); ++i) {
              MachineOperand &mOp = clone->getOperand(i);
              if (mOp.getType() == MachineOperand::MO_PCRelativeDisp) {
                mOp.setValueReg(sideBB);
              }
            }
          }
        
          tmp->push_back(clone);
        
        }
      
        //Add llvm branch
        TerminatorInst *newBranch = new BranchInst(sideBB, tmpBB);
      
        newEp.push_back(tmp);
        (((MachineBasicBlock*)SB[0])->getParent())->getBasicBlockList().push_back(tmp);

        newLLVMEp.push_back(tmpBB);
      
      }
      side_llvm_epilogues.push_back(newLLVMEp);
      side_epilogues.push_back(newEp);
    }
  
  //Now stich up all the branches
  
  //Loop over prologues, and if its an inner branch and branches to our original side exit
  //then have it branch to the appropriate epilogue first (if it exists)
    for(unsigned P = 0; P < prologues.size(); ++P) {

      //Get BB side exit we are dealing with
      MachineBasicBlock *currentMBB = prologues[P][sideExitNum];
      if(P >= (unsigned) stage) {
        //Iterate backwards of machine instructions to find the branch we need to update
        for(MachineBasicBlock::reverse_iterator mInst = currentMBB->rbegin(), mInstEnd = currentMBB->rend(); mInst != mInstEnd; ++mInst) {
          MachineOpCode OC = mInst->getOpcode();
          
          //If its a branch update its branchto
          if(TMI->isBranch(OC)) {
            for(unsigned opNum = 0; opNum < mInst->getNumOperands(); ++opNum) {
              MachineOperand &mOp = mInst->getOperand(opNum);
              if (mOp.getType() == MachineOperand::MO_PCRelativeDisp) {
                //Check if we branch to side exit
                if(mOp.getVRegValue() == sideExits[mbb]) {
                  mOp.setValueReg(side_llvm_epilogues[P][0]);
                }
              }
            }
            DEBUG(std::cerr << "New Prologue Branch: " << *mInst << "\n");
          }
        }
        
        //Update llvm branch
        TerminatorInst *branchVal = ((BasicBlock*) currentMBB->getBasicBlock())->getTerminator();
        DEBUG(std::cerr << *branchVal << "\n");
        
        for(unsigned i=0; i < branchVal->getNumSuccessors(); ++i) {
          if(branchVal->getSuccessor(i) == sideExits[mbb]) {
            DEBUG(std::cerr << "Replacing successor bb\n");
            branchVal->setSuccessor(i, side_llvm_epilogues[P][0]);
          }
        }
      }
      else {
        //must add BA branch because another prologue or kernel has the actual side exit branch
         //Add unconditional branches to original exits
        assert( (sideExitNum+1) < prologues[P].size() && "must have valid prologue to branch to");
        BuildMI(prologues[P][sideExitNum], V9::BA, 1).addPCDisp((BasicBlock*)(prologues[P][sideExitNum+1])->getBasicBlock());
        BuildMI(prologues[P][sideExitNum], V9::NOP, 0);

        TerminatorInst *newBranch = new BranchInst((BasicBlock*) (prologues[P][sideExitNum+1])->getBasicBlock(), (BasicBlock*) (prologues[P][sideExitNum])->getBasicBlock());

      }
    }


    //Update side exits in kernel
    MachineBasicBlock *currentMBB = kernelMBBs[sideExitNum];
    //Iterate backwards of machine instructions to find the branch we need to update
    for(MachineBasicBlock::reverse_iterator mInst = currentMBB->rbegin(), mInstEnd = currentMBB->rend(); mInst != mInstEnd; ++mInst) {
      MachineOpCode OC = mInst->getOpcode();
      
      //If its a branch update its branchto
      if(TMI->isBranch(OC)) {
        for(unsigned opNum = 0; opNum < mInst->getNumOperands(); ++opNum) {
          MachineOperand &mOp = mInst->getOperand(opNum);
          if (mOp.getType() == MachineOperand::MO_PCRelativeDisp) {
            //Check if we branch to side exit
            if(mOp.getVRegValue() == sideExits[mbb]) {
              if(side_llvm_epilogues.size() > 0)
                mOp.setValueReg(side_llvm_epilogues[0][0]);
              else
                mOp.setValueReg(sideBB);
            }
          }
        }
        DEBUG(std::cerr << "New Prologue Branch: " << *mInst << "\n");
      }
    }

    //Update llvm branch
    //Update llvm branch
    TerminatorInst *branchVal = ((BasicBlock*)currentMBB->getBasicBlock())->getTerminator();
    DEBUG(std::cerr << *branchVal << "\n");
    
    for(unsigned i=0; i < branchVal->getNumSuccessors(); ++i) {
      if(branchVal->getSuccessor(i) == sideExits[mbb]) {
        DEBUG(std::cerr << "Replacing successor bb\n");
        if(side_llvm_epilogues.size() > 0)
          branchVal->setSuccessor(i, side_llvm_epilogues[0][0]);
        else
          branchVal->setSuccessor(i, sideBB);
      }
    }
  }
}

