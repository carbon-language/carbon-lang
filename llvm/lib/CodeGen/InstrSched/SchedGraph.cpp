// $Id$
//***************************************************************************
// File:
//	SchedGraph.cpp
// 
// Purpose:
//	Scheduling graph based on SSA graph plus extra dependence edges
//	capturing dependences due to machine resources (machine registers,
//	CC registers, and any others).
// 
// History:
//	7/20/01	 -  Vikram Adve  -  Created
//**************************************************************************/

#include "SchedGraph.h"
#include "llvm/InstrTypes.h"
#include "llvm/Instruction.h"
#include "llvm/BasicBlock.h"
#include "llvm/Method.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/Target/MachineInstrInfo.h"
#include "llvm/Target/MachineRegInfo.h"
#include "llvm/Support/StringExtras.h"
#include <algorithm>


//*********************** Internal Data Structures *************************/

typedef vector< pair<SchedGraphNode*, unsigned int> > RefVec;

// The following needs to be a class, not a typedef, so we can use
// an opaque declaration in SchedGraph.h
class RegToRefVecMap: public hash_map<int, RefVec> {
  typedef hash_map<int, RefVec>::      iterator iterator;
  typedef hash_map<int, RefVec>::const_iterator const_iterator;
};

// 
// class SchedGraphEdge
// 

/*ctor*/
SchedGraphEdge::SchedGraphEdge(SchedGraphNode* _src,
			       SchedGraphNode* _sink,
			       SchedGraphEdgeDepType _depType,
			       DataDepOrderType _depOrderType,
			       int _minDelay)
  : src(_src),
    sink(_sink),
    depType(_depType),
    depOrderType(_depOrderType),
    val(NULL),
    minDelay((_minDelay >= 0)? _minDelay : _src->getLatency())
{
  src->addOutEdge(this);
  sink->addInEdge(this);
}


/*ctor*/
SchedGraphEdge::SchedGraphEdge(SchedGraphNode*  _src,
			       SchedGraphNode*  _sink,
			       const Value*     _val,
			       DataDepOrderType _depOrderType,
			       int              _minDelay)
  : src(_src),
    sink(_sink),
    depType(DefUseDep),
    depOrderType(_depOrderType),
    val(_val),
    minDelay((_minDelay >= 0)? _minDelay : _src->getLatency())
{
  src->addOutEdge(this);
  sink->addInEdge(this);
}


/*ctor*/
SchedGraphEdge::SchedGraphEdge(SchedGraphNode*  _src,
			       SchedGraphNode*  _sink,
			       unsigned int     _regNum,
			       DataDepOrderType _depOrderType,
			       int             _minDelay)
  : src(_src),
    sink(_sink),
    depType(MachineRegister),
    depOrderType(_depOrderType),
    minDelay((_minDelay >= 0)? _minDelay : _src->getLatency()),
    machineRegNum(_regNum)
{
  src->addOutEdge(this);
  sink->addInEdge(this);
}


/*ctor*/
SchedGraphEdge::SchedGraphEdge(SchedGraphNode* _src,
			       SchedGraphNode* _sink,
			       ResourceId      _resourceId,
			       int             _minDelay)
  : src(_src),
    sink(_sink),
    depType(MachineResource),
    depOrderType(NonDataDep),
    minDelay((_minDelay >= 0)? _minDelay : _src->getLatency()),
    resourceId(_resourceId)
{
  src->addOutEdge(this);
  sink->addInEdge(this);
}

/*dtor*/
SchedGraphEdge::~SchedGraphEdge()
{
}

void SchedGraphEdge::dump(int indent=0) const {
  printIndent(indent); cout << *this; 
}


// 
// class SchedGraphNode
// 

/*ctor*/
SchedGraphNode::SchedGraphNode(unsigned int _nodeId,
			       const Instruction* _instr,
			       const MachineInstr* _minstr,
			       const TargetMachine& target)
  : nodeId(_nodeId),
    instr(_instr),
    minstr(_minstr),
    latency(0)
{
  if (minstr)
    {
      MachineOpCode mopCode = minstr->getOpCode();
      latency = target.getInstrInfo().hasResultInterlock(mopCode)
	? target.getInstrInfo().minLatency(mopCode)
	: target.getInstrInfo().maxLatency(mopCode);
    }
}


/*dtor*/
SchedGraphNode::~SchedGraphNode()
{
}

void SchedGraphNode::dump(int indent=0) const {
  printIndent(indent); cout << *this; 
}


inline void
SchedGraphNode::addInEdge(SchedGraphEdge* edge)
{
  inEdges.push_back(edge);
}


inline void
SchedGraphNode::addOutEdge(SchedGraphEdge* edge)
{
  outEdges.push_back(edge);
}

inline void
SchedGraphNode::removeInEdge(const SchedGraphEdge* edge)
{
  assert(edge->getSink() == this);
  
  for (iterator I = beginInEdges(); I != endInEdges(); ++I)
    if ((*I) == edge)
      {
	inEdges.erase(I);
	break;
      }
}

inline void
SchedGraphNode::removeOutEdge(const SchedGraphEdge* edge)
{
  assert(edge->getSrc() == this);
  
  for (iterator I = beginOutEdges(); I != endOutEdges(); ++I)
    if ((*I) == edge)
      {
	outEdges.erase(I);
	break;
      }
}


// 
// class SchedGraph
// 


/*ctor*/
SchedGraph::SchedGraph(const BasicBlock* bb,
		       const TargetMachine& target)
{
  bbVec.push_back(bb);
  this->buildGraph(target);
}


/*dtor*/
SchedGraph::~SchedGraph()
{
  for (iterator I=begin(); I != end(); ++I)
    {
      SchedGraphNode* node = (*I).second;
      
      // for each node, delete its out-edges
      for (SchedGraphNode::iterator I = node->beginOutEdges();
	   I != node->endOutEdges(); ++I)
	delete *I;
      
      // then delete the node itself.
      delete node;
    }
}


void
SchedGraph::dump() const
{
  cout << "  Sched Graph for Basic Blocks: ";
  for (unsigned i=0, N=bbVec.size(); i < N; i++)
    {
      cout << (bbVec[i]->hasName()? bbVec[i]->getName() : "block")
	   << " (" << bbVec[i] << ")"
	   << ((i == N-1)? "" : ", ");
    }
  
  cout << endl << endl << "    Actual Root nodes : ";
  for (unsigned i=0, N=graphRoot->outEdges.size(); i < N; i++)
    cout << graphRoot->outEdges[i]->getSink()->getNodeId()
	 << ((i == N-1)? "" : ", ");
  
  cout << endl << "    Graph Nodes:" << endl;
  for (const_iterator I=begin(); I != end(); ++I)
    cout << endl << * (*I).second;
  
  cout << endl;
}


void
SchedGraph::eraseIncomingEdges(SchedGraphNode* node, bool addDummyEdges)
{
  // Delete and disconnect all in-edges for the node
  for (SchedGraphNode::iterator I = node->beginInEdges();
       I != node->endInEdges(); ++I)
    {
      SchedGraphNode* srcNode = (*I)->getSrc();
      srcNode->removeOutEdge(*I);
      delete *I;
      
      if (addDummyEdges &&
	  srcNode != getRoot() &&
	  srcNode->beginOutEdges() == srcNode->endOutEdges())
	{ // srcNode has no more out edges, so add an edge to dummy EXIT node
	  assert(node != getLeaf() && "Adding edge that was just removed?");
	  (void) new SchedGraphEdge(srcNode, getLeaf(),
		    SchedGraphEdge::CtrlDep, SchedGraphEdge::NonDataDep, 0);
	}
    }
  
  node->inEdges.clear();
}

void
SchedGraph::eraseOutgoingEdges(SchedGraphNode* node, bool addDummyEdges)
{
  // Delete and disconnect all out-edges for the node
  for (SchedGraphNode::iterator I = node->beginOutEdges();
       I != node->endOutEdges(); ++I)
    {
      SchedGraphNode* sinkNode = (*I)->getSink();
      sinkNode->removeInEdge(*I);
      delete *I;
      
      if (addDummyEdges &&
	  sinkNode != getLeaf() &&
	  sinkNode->beginInEdges() == sinkNode->endInEdges())
	{ //sinkNode has no more in edges, so add an edge from dummy ENTRY node
	  assert(node != getRoot() && "Adding edge that was just removed?");
	  (void) new SchedGraphEdge(getRoot(), sinkNode,
		    SchedGraphEdge::CtrlDep, SchedGraphEdge::NonDataDep, 0);
	}
    }
  
  node->outEdges.clear();
}

void
SchedGraph::eraseIncidentEdges(SchedGraphNode* node, bool addDummyEdges)
{
  this->eraseIncomingEdges(node, addDummyEdges);	
  this->eraseOutgoingEdges(node, addDummyEdges);	
}


void
SchedGraph::addDummyEdges()
{
  assert(graphRoot->outEdges.size() == 0);
  
  for (const_iterator I=begin(); I != end(); ++I)
    {
      SchedGraphNode* node = (*I).second;
      assert(node != graphRoot && node != graphLeaf);
      if (node->beginInEdges() == node->endInEdges())
	(void) new SchedGraphEdge(graphRoot, node, SchedGraphEdge::CtrlDep,
				  SchedGraphEdge::NonDataDep, 0);
      if (node->beginOutEdges() == node->endOutEdges())
	(void) new SchedGraphEdge(node, graphLeaf, SchedGraphEdge::CtrlDep,
				  SchedGraphEdge::NonDataDep, 0);
    }
}


void
SchedGraph::addCDEdges(const TerminatorInst* term,
		       const TargetMachine& target)
{
  const MachineInstrInfo& mii = target.getInstrInfo();
  MachineCodeForVMInstr& termMvec = term->getMachineInstrVec();
  
  // Find the first branch instr in the sequence of machine instrs for term
  // 
  unsigned first = 0;
  while (! mii.isBranch(termMvec[first]->getOpCode()))
    ++first;
  assert(first < termMvec.size() &&
	 "No branch instructions for BR?  Ok, but weird!  Delete assertion.");
  if (first == termMvec.size())
    return;
  
  SchedGraphNode* firstBrNode = this->getGraphNodeForInstr(termMvec[first]);
  
  // Add CD edges from each instruction in the sequence to the
  // *last preceding* branch instr. in the sequence 
  // 
  for (int i = (int) termMvec.size()-1; i > (int) first; i--) 
    {
      SchedGraphNode* toNode = this->getGraphNodeForInstr(termMvec[i]);
      assert(toNode && "No node for instr generated for branch?");
      
      for (int j = i-1; j >= 0; j--) 
	if (mii.isBranch(termMvec[j]->getOpCode()))
	  {
	    SchedGraphNode* brNode = this->getGraphNodeForInstr(termMvec[j]);
	    assert(brNode && "No node for instr generated for branch?");
	    (void) new SchedGraphEdge(brNode, toNode, SchedGraphEdge::CtrlDep,
				      SchedGraphEdge::NonDataDep, 0);
	    break;			// only one incoming edge is enough
	  }
    }
  
  // Add CD edges from each instruction preceding the first branch
  // to the first branch
  // 
  for (int i = first-1; i >= 0; i--) 
    {
      SchedGraphNode* fromNode = this->getGraphNodeForInstr(termMvec[i]);
      assert(fromNode && "No node for instr generated for branch?");
      (void) new SchedGraphEdge(fromNode, firstBrNode, SchedGraphEdge::CtrlDep,
				SchedGraphEdge::NonDataDep, 0);
    }
  
  // Now add CD edges to the first branch instruction in the sequence
  // from all preceding instructions in the basic block.
  // 
  const BasicBlock* bb = term->getParent();
  for (BasicBlock::const_iterator II = bb->begin(); II != bb->end(); ++II)
    {
      if ((*II) == (const Instruction*) term)	// special case, handled above
	continue;
      
      assert(! (*II)->isTerminator() && "Two terminators in basic block?");
      
      const MachineCodeForVMInstr& mvec = (*II)->getMachineInstrVec();
      for (unsigned i=0, N=mvec.size(); i < N; i++) 
	{
	  SchedGraphNode* fromNode = this->getGraphNodeForInstr(mvec[i]);
	  if (fromNode == NULL)
	    continue;			// dummy instruction, e.g., PHI
	  
	  (void) new SchedGraphEdge(fromNode, firstBrNode,
				    SchedGraphEdge::CtrlDep,
				    SchedGraphEdge::NonDataDep, 0);
	  
	  // If we find any other machine instructions (other than due to
	  // the terminator) that also have delay slots, add an outgoing edge
	  // from the instruction to the instructions in the delay slots.
	  // 
	  unsigned d = mii.getNumDelaySlots(mvec[i]->getOpCode());
	  assert(i+d < N && "Insufficient delay slots for instruction?");
	  
	  for (unsigned j=1; j <= d; j++)
	    {
	      SchedGraphNode* toNode = this->getGraphNodeForInstr(mvec[i+j]);
	      assert(toNode && "No node for machine instr in delay slot?");
	      (void) new SchedGraphEdge(fromNode, toNode,
					SchedGraphEdge::CtrlDep,
				      SchedGraphEdge::NonDataDep, 0);
	    }
	}
    }
}


void
SchedGraph::addMemEdges(const vector<const Instruction*>& memVec,
			const TargetMachine& target)
{
  const MachineInstrInfo& mii = target.getInstrInfo();
  
  for (unsigned im=0, NM=memVec.size(); im < NM; im++)
    {
      const Instruction* fromInstr = memVec[im];
      bool fromIsLoad = fromInstr->getOpcode() == Instruction::Load;
      
      for (unsigned jm=im+1; jm < NM; jm++)
	{
	  const Instruction* toInstr = memVec[jm];
	  bool toIsLoad = toInstr->getOpcode() == Instruction::Load;
	  SchedGraphEdge::DataDepOrderType depOrderType;
	  
	  if (fromIsLoad)
	    {
	      if (toIsLoad) continue;	// both instructions are loads
	      depOrderType = SchedGraphEdge::AntiDep;
	    }
	  else
	    {
	      depOrderType = (toIsLoad)? SchedGraphEdge::TrueDep
		: SchedGraphEdge::OutputDep;
	    }
	  
	  MachineCodeForVMInstr& fromInstrMvec=fromInstr->getMachineInstrVec();
	  MachineCodeForVMInstr& toInstrMvec = toInstr->getMachineInstrVec();
	  
	  // We have two VM memory instructions, and at least one is a store.
	  // Add edges between all machine load/store instructions.
	  // 
	  for (unsigned i=0, N=fromInstrMvec.size(); i < N; i++) 
	    {
	      MachineOpCode fromOpCode = fromInstrMvec[i]->getOpCode();
	      if (mii.isLoad(fromOpCode) || mii.isStore(fromOpCode))
		{
		  SchedGraphNode* fromNode =
		    this->getGraphNodeForInstr(fromInstrMvec[i]);
		  assert(fromNode && "No node for memory instr?");
		  
		  for (unsigned j=0, M=toInstrMvec.size(); j < M; j++) 
		    {
		      MachineOpCode toOpCode = toInstrMvec[j]->getOpCode();
		      if (mii.isLoad(toOpCode) || mii.isStore(toOpCode))
			{
			  SchedGraphNode* toNode =
			    this->getGraphNodeForInstr(toInstrMvec[j]);
			  assert(toNode && "No node for memory instr?");
			  
			  (void) new SchedGraphEdge(fromNode, toNode,
						    SchedGraphEdge::MemoryDep,
						    depOrderType, 1);
			}
		    }
		}
	    }
	}
    }
}


void
SchedGraph::addMachineRegEdges(RegToRefVecMap& regToRefVecMap,
			       const TargetMachine& target)
{
  assert(bbVec.size() == 1 && "Only handling a single basic block here");
  
  // This assumes that such hardwired registers are never allocated
  // to any LLVM value (since register allocation happens later), i.e.,
  // any uses or defs of this register have been made explicit!
  // Also assumes that two registers with different numbers are
  // not aliased!
  // 
  for (RegToRefVecMap::iterator I = regToRefVecMap.begin();
       I != regToRefVecMap.end(); ++I)
    {
      int regNum        = (*I).first;
      RefVec& regRefVec = (*I).second;
      
      // regRefVec is ordered by control flow order in the basic block
      for (unsigned i=0; i < regRefVec.size(); ++i)
	{
	  SchedGraphNode* node = regRefVec[i].first;
	  unsigned int opNum   = regRefVec[i].second;
	  bool isDef = node->getMachineInstr()->operandIsDefined(opNum);
	        
          for (unsigned p=0; p < i; ++p)
            {
              SchedGraphNode* prevNode = regRefVec[p].first;
              if (prevNode != node)
                {
                  unsigned int prevOpNum = regRefVec[p].second;
                  bool prevIsDef =
                    prevNode->getMachineInstr()->operandIsDefined(prevOpNum);
                  
                  if (isDef)
                    new SchedGraphEdge(prevNode, node, regNum,
                                       (prevIsDef)? SchedGraphEdge::OutputDep
                                                  : SchedGraphEdge::AntiDep);
                  else if (prevIsDef)
                    new SchedGraphEdge(prevNode, node, regNum,
                                       SchedGraphEdge::TrueDep);
                }
            }
        }
    }
}


void
SchedGraph::addSSAEdge(SchedGraphNode* node,
		       const Value* val,
		       const TargetMachine& target)
{
  if (!val->isInstruction()) return;
  
  const Instruction* thisVMInstr = node->getInstr();
  const Instruction* defVMInstr  = cast<const Instruction>(val);
  
  // Phi instructions are the only ones that produce a value but don't get
  // any non-dummy machine instructions.  Return here as an optimization.
  // 
  if (defVMInstr->isPHINode())
    return;
  
  // Now add the graph edge for the appropriate machine instruction(s).
  // Note that multiple machine instructions generated for the
  // def VM instruction may modify the register for the def value.
  // 
  MachineCodeForVMInstr& defMvec = defVMInstr->getMachineInstrVec();
  const MachineInstrInfo& mii = target.getInstrInfo();
  
  for (unsigned i=0, N=defMvec.size(); i < N; i++)
    for (int o=0, N = mii.getNumOperands(defMvec[i]->getOpCode()); o < N; o++)
      {
	const MachineOperand& defOp = defMvec[i]->getOperand(o); 
	
	if (defOp.opIsDef()
	    && (defOp.getOperandType() == MachineOperand::MO_VirtualRegister
		|| defOp.getOperandType() == MachineOperand::MO_CCRegister)
	    && (defOp.getVRegValue() == val))
	  {
	    // this instruction does define value `val'.
	    // if there is a node for it in the same graph, add an edge.
	    SchedGraphNode* defNode = this->getGraphNodeForInstr(defMvec[i]);
	    if (defNode != NULL && defNode != node)
              (void) new SchedGraphEdge(defNode, node, val);
	  }
      }
}


void
SchedGraph::addEdgesForInstruction(const MachineInstr& minstr,
				   RegToRefVecMap& regToRefVecMap,
				   const TargetMachine& target)
{
  SchedGraphNode* node = this->getGraphNodeForInstr(&minstr);
  if (node == NULL)
    return;
  
  assert(node->getInstr() && "Should be no dummy nodes here!");
  const Instruction& instr = * node->getInstr();
  
  // Add edges for all operands of the machine instruction.
  // Also, record all machine register references to add reg. deps. later.
  // 
  for (unsigned i=0, numOps=minstr.getNumOperands(); i < numOps; i++)
    {
      const MachineOperand& mop = minstr.getOperand(i);
      
      // if this writes to a machine register other than the hardwired
      // "zero" register, record the reference.
      if (mop.getOperandType() == MachineOperand::MO_MachineRegister
	  && (mop.getMachineRegNum()
	      != (unsigned) target.getRegInfo().getZeroRegNum()))
	{
	  regToRefVecMap[mop.getMachineRegNum()].push_back(make_pair(node, i));
	}
      
      // ignore all other def operands
      if (minstr.operandIsDefined(i))
	continue;
      
      switch(mop.getOperandType())
	{
	case MachineOperand::MO_VirtualRegister:
	case MachineOperand::MO_CCRegister:
	  if (mop.getVRegValue())
	    addSSAEdge(node, mop.getVRegValue(), target);
	  break;
	  
	case MachineOperand::MO_MachineRegister:
	  break; 
	  
	case MachineOperand::MO_SignExtendedImmed:
	case MachineOperand::MO_UnextendedImmed:
	case MachineOperand::MO_PCRelativeDisp:
	  break;	// nothing to do for immediate fields
	  
	default:
	  assert(0 && "Unknown machine operand type in SchedGraph builder");
	  break;
	}
    }

  // Add edges for values implicitly used by the machine instruction sequence
  // for the VM instruction but not made explicit operands.  Examples include
  // function arguments to a Call instructions or the return value of a Ret
  // instruction.  We'll conservatively add the dependences to every machine
  // machine instruction in the instruction sequence for this VM instr
  // (at least for now, there is never more than one machine instr).
  // 
  const vector<const Value*>& implicitUses =
    instr.getMachineInstrVec().getImplicitUses();
  for (unsigned i=0; i < implicitUses.size(); ++i)
    addSSAEdge(node, implicitUses[i], target);
}


void
SchedGraph::addNonSSAEdgesForValue(const Instruction* instr,
                                   const TargetMachine& target)
{
  assert(instr->isInstruction());
  if (instr->isPHINode())
    return;

  MachineCodeForVMInstr& mvec = instr->getMachineInstrVec();
  const MachineInstrInfo& mii = target.getInstrInfo();
  RefVec refVec;
  
  for (unsigned i=0, N=mvec.size(); i < N; i++)
    for (int o=0, N = mii.getNumOperands(mvec[i]->getOpCode()); o < N; o++)
      {
	const MachineOperand& op = mvec[i]->getOperand(o); 
	
	if ((op.getOperandType() == MachineOperand::MO_VirtualRegister ||
             op.getOperandType() == MachineOperand::MO_CCRegister)
	    && op.getVRegValue() == (Value*) instr)
          {
	    // this operand is a definition or use of value `instr'
	    SchedGraphNode* node = this->getGraphNodeForInstr(mvec[i]);
            assert(node && "No node for machine instruction in this BB?");
            refVec.push_back(make_pair(node, o));
          }
      }
  
  // refVec is ordered by control flow order of the machine instructions
  for (unsigned i=0; i < refVec.size(); ++i)
    {
      SchedGraphNode* node = refVec[i].first;
      unsigned int   opNum = refVec[i].second;
      bool isDef = node->getMachineInstr()->operandIsDefined(opNum);
      
      if (isDef)
        // add output and/or anti deps to this definition
        for (unsigned p=0; p < i; ++p)
          {
            SchedGraphNode* prevNode = refVec[p].first;
            if (prevNode != node)
              {
                bool prevIsDef = prevNode->getMachineInstr()->
                  operandIsDefined(refVec[p].second);
                new SchedGraphEdge(prevNode, node, SchedGraphEdge::DefUseDep,
                                   (prevIsDef)? SchedGraphEdge::OutputDep
                                              : SchedGraphEdge::AntiDep);
              }
          }
    }
}


void
SchedGraph::buildNodesforVMInstr(const TargetMachine& target,
                                 const Instruction* instr)
{
  const MachineInstrInfo& mii = target.getInstrInfo();
  const MachineCodeForVMInstr& mvec = instr->getMachineInstrVec();
  for (unsigned i=0; i < mvec.size(); i++)
    if (! mii.isDummyPhiInstr(mvec[i]->getOpCode()))
      {
        SchedGraphNode* node = new SchedGraphNode(getNumNodes(),
                                                  instr, mvec[i], target);
        this->noteGraphNodeForInstr(mvec[i], node);
      }
}


void
SchedGraph::buildGraph(const TargetMachine& target)
{
  const MachineInstrInfo& mii = target.getInstrInfo();
  const BasicBlock* bb = bbVec[0];
  
  assert(bbVec.size() == 1 && "Only handling a single basic block here");
  
  // Use this data structures to note all LLVM memory instructions.
  // We use this to add memory dependence edges without a second full walk.
  // 
  vector<const Instruction*> memVec;
  
  // Use this data structures to note any uses or definitions of
  // machine registers so we can add edges for those later without
  // extra passes over the nodes.
  // The vector holds an ordered list of references to the machine reg,
  // ordered according to control-flow order.  This only works for a
  // single basic block, hence the assertion.  Each reference is identified
  // by the pair: <node, operand-number>.
  // 
  RegToRefVecMap regToRefVecMap;
  
  // Make a dummy root node.  We'll add edges to the real roots later.
  graphRoot = new SchedGraphNode(0, NULL, NULL, target);
  graphLeaf = new SchedGraphNode(1, NULL, NULL, target);

  //----------------------------------------------------------------
  // First add nodes for all the machine instructions in the basic block
  // because this greatly simplifies identifying which edges to add.
  // Do this one VM instruction at a time since the SchedGraphNode needs that.
  // Also, remember the load/store instructions to add memory deps later.
  //----------------------------------------------------------------
  
  for (BasicBlock::const_iterator II = bb->begin(); II != bb->end(); ++II)
    {
      const Instruction *instr = *II;

      // Build graph nodes for this VM instruction
      buildNodesforVMInstr(target, instr);
      
      // Remember the load/store instructions to add memory deps later.
      if (instr->getOpcode() == Instruction::Load ||
	  instr->getOpcode() == Instruction::Store) 
	memVec.push_back(instr);
    } 
  
  //----------------------------------------------------------------
  // Now add edges for the following (all are incoming edges except (4)):
  // (1) operands of the machine instruction, including hidden operands
  // (2) machine register dependences
  // (3) memory load/store dependences
  // (3) other resource dependences for the machine instruction, if any
  // (4) output dependences when multiple machine instructions define the
  //     same value; all must have been generated from a single VM instrn
  // (5) control dependences to branch instructions generated for the
  //     terminator instruction of the BB. Because of delay slots and
  //     2-way conditional branches, multiple CD edges are needed
  //     (see addCDEdges for details).
  // Also, note any uses or defs of machine registers.
  // 
  //----------------------------------------------------------------
      
  // First, add edges to the terminator instruction of the basic block.
  this->addCDEdges(bb->getTerminator(), target);
      
  // Then add memory dep edges: store->load, load->store, and store->store
  this->addMemEdges(memVec, target);
      
  // Then add other edges for all instructions in the block.
  // Do this in machine code order and find all references to machine regs.
  MachineCodeForBasicBlock& mvec = bb->getMachineInstrVec();
  for (unsigned i=0, N=mvec.size(); i < N; i++)
    addEdgesForInstruction(*mvec[i], regToRefVecMap, target);
  
  // Since the code is no longer in SSA form, add output dep. edges
  // between machine instructions that define the same Value, and anti-dep.
  // edges from those to other machine instructions for the same VM instr.
  // We assume that all machine instructions that define a value are
  // generated from the VM instruction corresponding to that value.
  // 
  for (BasicBlock::const_iterator II = bb->begin(); II != bb->end(); ++II)
    {
      const Instruction *instr = *II;
      this->addNonSSAEdgesForValue(instr, target);
    }
  
  // Then add edges for dependences on machine registers
  this->addMachineRegEdges(regToRefVecMap, target);
  
  // Finally, add edges from the dummy root and to dummy leaf
  this->addDummyEdges();		
}


// 
// class SchedGraphSet
// 

/*ctor*/
SchedGraphSet::SchedGraphSet(const Method* _method,
			     const TargetMachine& target) :
  method(_method)
{
  buildGraphsForMethod(method, target);
}


/*dtor*/
SchedGraphSet::~SchedGraphSet()
{
  // delete all the graphs
  for (iterator I=begin(); I != end(); ++I)
    delete (*I).second;
}


void
SchedGraphSet::dump() const
{
  cout << "======== Sched graphs for method `"
       << (method->hasName()? method->getName() : "???")
       << "' ========" << endl << endl;
  
  for (const_iterator I=begin(); I != end(); ++I)
    (*I).second->dump();
  
  cout << endl << "====== End graphs for method `"
       << (method->hasName()? method->getName() : "")
       << "' ========" << endl << endl;
}


void
SchedGraphSet::buildGraphsForMethod(const Method *method,
				    const TargetMachine& target)
{
  for (Method::const_iterator BI = method->begin(); BI != method->end(); ++BI)
    {
      SchedGraph* graph = new SchedGraph(*BI, target);
      this->noteGraphForBlock(*BI, graph);
    }   
}



ostream&
operator<<(ostream& os, const SchedGraphEdge& edge)
{
  os << "edge [" << edge.src->getNodeId() << "] -> ["
     << edge.sink->getNodeId() << "] : ";
  
  switch(edge.depType) {
  case SchedGraphEdge::CtrlDep:		os<< "Control Dep"; break;
  case SchedGraphEdge::DefUseDep:	os<< "Reg Value " << edge.val; break;
  case SchedGraphEdge::MemoryDep:	os<< "Mem Value " << edge.val; break;
  case SchedGraphEdge::MachineRegister: os<< "Reg " <<edge.machineRegNum;break;
  case SchedGraphEdge::MachineResource: os<<"Resource "<<edge.resourceId;break;
  default: assert(0); break;
  }
  
  os << " : delay = " << edge.minDelay << endl;
  
  return os;
}

ostream&
operator<<(ostream& os, const SchedGraphNode& node)
{
  printIndent(4, os);
  os << "Node " << node.nodeId << " : "
     << "latency = " << node.latency << endl;
  
  printIndent(6, os);
  
  if (node.getMachineInstr() == NULL)
    os << "(Dummy node)" << endl;
  else
    {
      os << *node.getMachineInstr() << endl;
  
      printIndent(6, os);
      os << node.inEdges.size() << " Incoming Edges:" << endl;
      for (unsigned i=0, N=node.inEdges.size(); i < N; i++)
	{
	  printIndent(8, os);
	  os << * node.inEdges[i];
	}
  
      printIndent(6, os);
      os << node.outEdges.size() << " Outgoing Edges:" << endl;
      for (unsigned i=0, N=node.outEdges.size(); i < N; i++)
	{
	  printIndent(8, os);
	  os << * node.outEdges[i];
	}
    }
  
  return os;
}
