//===- IPModRef.cpp - Compute IP Mod/Ref information ------------*- C++ -*-===//
//
// See high-level comments in include/llvm/Analysis/IPModRef.h
// 
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/IPModRef.h"
#include "llvm/Analysis/DataStructure.h"
#include "llvm/Analysis/DSGraph.h"
#include "llvm/Module.h"
#include "llvm/iOther.h"
#include "Support/Statistic.h"
#include "Support/STLExtras.h"
#include "Support/StringExtras.h"

//----------------------------------------------------------------------------
// Private constants and data
//----------------------------------------------------------------------------

static RegisterAnalysis<IPModRef>
Z("ipmodref", "Interprocedural mod/ref analysis");


//----------------------------------------------------------------------------
// class ModRefInfo
//----------------------------------------------------------------------------

void ModRefInfo::print(std::ostream &O) const
{
  O << std::endl << "Modified   nodes = " << modNodeSet;
  O              << "Referenced nodes = " << refNodeSet << std::endl;
}

void ModRefInfo::dump() const
{
  print(std::cerr);
}

//----------------------------------------------------------------------------
// class FunctionModRefInfo
//----------------------------------------------------------------------------


// This constructor computes a node numbering for the TD graph.
// 
FunctionModRefInfo::FunctionModRefInfo(const Function& func,
                                       IPModRef& ipmro,
                                       const DSGraph& tdg,
                                       const DSGraph& ldg)
  : F(func), IPModRefObj(ipmro), 
    funcTDGraph(tdg),
    funcLocalGraph(ldg),
    funcModRefInfo(tdg.getGraphSize())
{
  for (unsigned i=0, N = funcTDGraph.getGraphSize(); i < N; ++i)
    NodeIds[funcTDGraph.getNodes()[i]] = i;
}


FunctionModRefInfo::~FunctionModRefInfo()
{
  for(std::map<const CallInst*, ModRefInfo*>::iterator
        I=callSiteModRefInfo.begin(), E=callSiteModRefInfo.end(); I != E; ++I)
    delete(I->second);

  // Empty map just to make problems easier to track down
  callSiteModRefInfo.clear();
}

unsigned FunctionModRefInfo::getNodeId(const Value* value) const {
  return getNodeId(funcTDGraph.getNodeForValue(const_cast<Value*>(value))
                   .getNode());
}



// Compute Mod/Ref bit vectors for the entire function.
// These are simply copies of the Read/Write flags from the nodes of
// the top-down DS graph.
// 
void FunctionModRefInfo::computeModRef(const Function &func)
{
  // Mark all nodes in the graph that are marked MOD as being mod
  // and all those marked REF as being ref.
  for (unsigned i = 0, N = funcTDGraph.getGraphSize(); i < N; ++i)
    {
      if (funcTDGraph.getNodes()[i]->isModified())
        funcModRefInfo.setNodeIsMod(i);
      if (funcTDGraph.getNodes()[i]->isRead())
        funcModRefInfo.setNodeIsRef(i);
    }

  // Compute the Mod/Ref info for all call sites within the function
  // Use the Local DSgraph, which includes all the call sites in the
  // original program.
  const std::vector<DSCallSite>& callSites = funcLocalGraph.getFunctionCalls();
  for (unsigned i = 0, N = callSites.size(); i < N; ++i)
    computeModRef(callSites[i].getCallInst());
}

// ResolveCallSiteModRefInfo - This method performs the following actions:
//
//  1. It clones the top-down graph for the current function
//  2. It clears all of the mod/ref bits in the cloned graph
//  3. It then merges the bottom-up graph(s) for the specified call-site into
//     the clone (bringing new mod/ref bits).
//  4. It returns the clone, and a mapping of nodes from the original TDGraph to
//     the cloned graph with Mod/Ref info for the callsite.
//
// NOTE: Because this clones a dsgraph and returns it, the caller is responsible
//       for deleting the returned graph!
// NOTE: This method may return a null pointer if it is unable to determine the
//       requested information (because the call site calls an external
//       function or we cannot determine the complete set of functions invoked).
//
DSGraph *FunctionModRefInfo::ResolveCallSiteModRefInfo(CallInst &CI,
                               std::map<const DSNode*, DSNodeHandle> &NodeMap) {

  // Step #1: Clone the top-down graph...
  std::map<const DSNode*, DSNode*> RawNodeMap;
  DSGraph *Result = new DSGraph(funcTDGraph, RawNodeMap);

  // Convert the NodeMap from a map to DSNode* to be a map to DSNodeHandle's
  NodeMap.insert(RawNodeMap.begin(), RawNodeMap.end());

  // We are now done with the old map... so free it's memory...
  RawNodeMap.clear();

  // Step #2: Clear Mod/Ref information...
  Result->maskNodeTypes(~(DSNode::Modified | DSNode::Read));

  // Step #3: clone the bottom up graphs for the callees into the caller graph
  if (const Function *F = CI.getCalledFunction()) {
    if (F->isExternal()) {
      delete Result;
      return 0;   // We cannot compute Mod/Ref info for this callsite...
    }

    // Build up a DSCallSite for our invocation point here...

    // If the call returns a value, make sure to merge the nodes...
    DSNodeHandle RetVal;
    if (DS::isPointerType(CI.getType()))
      RetVal = Result->getNodeForValue(&CI);

    // Populate the arguments list...
    std::vector<DSNodeHandle> Args;
    for (unsigned i = 1, e = CI.getNumOperands(); i != e; ++i)
      if (DS::isPointerType(CI.getOperand(i)->getType()))
        Args.push_back(Result->getNodeForValue(CI.getOperand(i)));

    // Build the call site...
    DSCallSite CS(CI, RetVal, 0, Args);

    // Perform the merging now of the graph for the callee, which will come with
    // mod/ref bits set...
    Result->mergeInGraph(CS, IPModRefObj.getBUDSGraph(*F),
                         DSGraph::StripAllocaBit);

  } else {
    std::cerr << "IP Mod/Ref indirect call not implemented yet: "
              << "Being conservative\n";
    delete Result;
    return 0;
  }

  // Remove trivial dead nodes... don't aggressively prune graph though... the
  // graph is short lived anyway.
  Result->removeTriviallyDeadNodes(false);

  // Step #4: Return the clone + the mapping (by ref)
  return Result;
}

// Compute Mod/Ref bit vectors for a single call site.
// These are copies of the Read/Write flags from the nodes of
// the graph produced by clearing all flags in teh caller's TD graph
// and then inlining the callee's BU graph into the caller's TD graph.
// 
void
FunctionModRefInfo::computeModRef(const CallInst& callInst)
{
  // Allocate the mod/ref info for the call site.  Bits automatically cleared.
  ModRefInfo* callModRefInfo = new ModRefInfo(funcTDGraph.getGraphSize());
  callSiteModRefInfo[&callInst] = callModRefInfo;

  // Get a copy of the graph for the callee with the callee inlined
  std::map<const DSNode*, DSNodeHandle> NodeMap;
  DSGraph* csgp =
    ResolveCallSiteModRefInfo(const_cast<CallInst&>(callInst), NodeMap);

  assert(csgp && "FIXME: Cannot handle case where call site mod/ref info"
         " is not available yet!");

  // For all nodes in the graph, extract the mod/ref information
  const std::vector<DSNode*>& csgNodes = csgp->getNodes();
  const std::vector<DSNode*>& origNodes = funcTDGraph.getNodes();
  assert(csgNodes.size() == origNodes.size());
  for (unsigned i=0, N = csgNodes.size(); i < N; ++i)
    { 
      if (csgNodes[i]->isModified())
        callModRefInfo->setNodeIsMod(getNodeId(origNodes[i]));
      if (csgNodes[i]->isRead())
        callModRefInfo->setNodeIsRef(getNodeId(origNodes[i]));
    }

  // Drop nodemap before we delete the graph...
  NodeMap.clear();
  delete csgp;
}


// Print the results of the pass.
// Currently this just prints bit-vectors and is not very readable.
// 
void FunctionModRefInfo::print(std::ostream &O) const
{
  O << "---------- Mod/ref information for function "
    << F.getName() << "---------- \n\n";

  O << "Mod/ref info for function body:\n";
  funcModRefInfo.print(O);

  for (std::map<const CallInst*, ModRefInfo*>::const_iterator
         CI = callSiteModRefInfo.begin(), CE = callSiteModRefInfo.end();
       CI != CE; ++CI)
    {
      O << "Mod/ref info for call site " << CI->first << ":\n";
      CI->second->print(O);
    }

  O << "\n";
}

void FunctionModRefInfo::dump() const
{
  print(std::cerr);
}


//----------------------------------------------------------------------------
// class IPModRef: An interprocedural pass that computes IP Mod/Ref info.
//----------------------------------------------------------------------------

// Free the FunctionModRefInfo objects cached in funcToModRefInfoMap.
// 
void IPModRef::releaseMemory()
{
  for(std::map<const Function*, FunctionModRefInfo*>::iterator
        I=funcToModRefInfoMap.begin(), E=funcToModRefInfoMap.end(); I != E; ++I)
    delete(I->second);

  // Clear map so memory is not re-released if we are called again
  funcToModRefInfoMap.clear();
}

// Run the "interprocedural" pass on each function.  This needs to do
// NO real interprocedural work because all that has been done the
// data structure analysis.
// 
bool IPModRef::run(Module &theModule)
{
  M = &theModule;

  for (Module::const_iterator FI = M->begin(), FE = M->end(); FI != FE; ++FI)
    if (! FI->isExternal())
      getFuncInfo(*FI, /*computeIfMissing*/ true);
  return true;
}


FunctionModRefInfo& IPModRef::getFuncInfo(const Function& func,
                                          bool computeIfMissing)
{
  FunctionModRefInfo*& funcInfo = funcToModRefInfoMap[&func];
  assert (funcInfo != NULL || computeIfMissing);
  if (funcInfo == NULL)
    { // Create a new FunctionModRefInfo object
      funcInfo = new FunctionModRefInfo(func, *this, // inserts into map
                              getAnalysis<TDDataStructures>().getDSGraph(func),
                          getAnalysis<LocalDataStructures>().getDSGraph(func));
      funcInfo->computeModRef(func);            // computes the mod/ref info
    }
  return *funcInfo;
}

/// getBUDSGraph - This method returns the BU data structure graph for F through
/// the use of the BUDataStructures object.
///
const DSGraph &IPModRef::getBUDSGraph(const Function &F) {
  return getAnalysis<BUDataStructures>().getDSGraph(F);
}


// getAnalysisUsage - This pass requires top-down data structure graphs.
// It modifies nothing.
// 
void IPModRef::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequired<LocalDataStructures>();
  AU.addRequired<BUDataStructures>();
  AU.addRequired<TDDataStructures>();
}


void IPModRef::print(std::ostream &O) const
{
  O << "\n========== Results of Interprocedural Mod/Ref Analysis ==========\n";
  
  for (std::map<const Function*, FunctionModRefInfo*>::const_iterator
         mapI = funcToModRefInfoMap.begin(), mapE = funcToModRefInfoMap.end();
       mapI != mapE; ++mapI)
    mapI->second->print(O);

  O << "\n";
}


void IPModRef::dump() const
{
  print(std::cerr);
}
