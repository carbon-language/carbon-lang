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



// Dummy function that will be replaced with one that inlines
// the callee's BU graph into the caller's TD graph.
// 
static const DSGraph* ResolveGraphForCallSite(const DSGraph& funcTDGraph,
                                       const CallInst& callInst)
{
  return &funcTDGraph;                    // TEMPORARY
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
  const DSGraph* csgp = ResolveGraphForCallSite(funcTDGraph, callInst);
  assert(csgp && "Unable to compute callee mod/ref information");

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
