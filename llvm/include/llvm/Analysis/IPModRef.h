//===- IPModRef.h - Compute IP Mod/Ref information --------------*- C++ -*-===//
//
// class IPModRef:
// 
// class IPModRef is an interprocedural analysis pass that computes
// flow-insensitive IP Mod and Ref information for every function
// (the GMOD and GREF problems) and for every call site (MOD and REF).
// 
// In practice, this needs to do NO real interprocedural work because
// all that is needed is done by the data structure analysis.
// This uses the top-down DS graph for a function and the bottom-up DS graph
// for each callee (including the Mod/Ref flags in the bottom-up graph)
// to compute the set of nodes that are Mod and Ref for the function and
// for each of its call sites.
//
// 
// class FunctionModRefInfo:
// 
// The results of IPModRef are encapsulated in the class FunctionModRefInfo.
// The results are stored as bit vectors: bit i represents node i
// in the TD DSGraph for the current function.  (This node numbering is
// implemented by class FunctionModRefInfo.)  Each FunctionModRefInfo
// includes:
// -- 2 bit vectors for the function (GMOD and GREF), and
// -- 2 bit vectors for each call site (MOD and REF).
//
// 
// IPModRef vs. Alias Analysis for Clients:
// 
// The IPModRef pass does not provide simpler query interfaces for specific
// LLVM values, instructions, or pointers because those results should be
// obtained through alias analysis (e.g., class DSAliasAnalysis).
// class IPModRef is primarily meant for other analysis passes that need to
// use Mod/Ref information efficiently for more complicated purposes;
// the bit-vector representations make propagation very efficient.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_IPMODREF_H
#define LLVM_ANALYSIS_IPMODREF_H

#include "llvm/Pass.h"
#include "Support/BitSetVector.h"
#include "Support/hash_map"

class Module;
class Function;
class CallInst;
class DSNode;
class DSGraph;
class DSNodeHandle;
class ModRefInfo;               // Result of IP Mod/Ref for one entity
class FunctionModRefInfo;       // ModRefInfo for a func and all calls in it
class IPModRef;                 // Pass that computes IP Mod/Ref info

//---------------------------------------------------------------------------
// class ModRefInfo 
// 
// Purpose:
//   Representation of Mod/Ref information for a single function or callsite.
//   This is represented as a pair of bit vectors, one each for Mod and Ref.
//   Each bit vector is indexed by the node id of the DS graph node index.
//---------------------------------------------------------------------------

class ModRefInfo {
  BitSetVector   modNodeSet;            // set of modified nodes
  BitSetVector   refNodeSet;            // set of referenced nodes
  
public:
  // 
  // Methods to construct ModRefInfo objects.
  // 
  ModRefInfo(unsigned int numNodes)
    : modNodeSet(numNodes),
      refNodeSet(numNodes) { }

  unsigned getSize() const {
    assert(modNodeSet.size() == refNodeSet.size() &&
           "Mod & Ref different size?");
    return modNodeSet.size();
  }

  void setNodeIsMod (unsigned nodeId)   { modNodeSet[nodeId] = true; }
  void setNodeIsRef (unsigned nodeId)   { refNodeSet[nodeId] = true; }

  //
  // Methods to query the mod/ref info
  // 
  bool nodeIsMod (unsigned nodeId) const  { return modNodeSet.test(nodeId); }
  bool nodeIsRef (unsigned nodeId) const  { return refNodeSet.test(nodeId); }
  bool nodeIsKill(unsigned nodeId) const  { return false; }

  const BitSetVector&  getModSet() const  { return modNodeSet; }
        BitSetVector&  getModSet()        { return modNodeSet; }

  const BitSetVector&  getRefSet() const  { return refNodeSet; }
        BitSetVector&  getRefSet()        { return refNodeSet; }

  // Debugging support methods
  void print(std::ostream &O, const std::string& prefix=std::string("")) const;
  void dump() const;
};


//----------------------------------------------------------------------------
// class FunctionModRefInfo
// 
// Representation of the results of IP Mod/Ref analysis for a function
// and for each of the call sites within the function.
// Each of these are represented as bit vectors of size = the number of
// nodes in the top-dwon DS graph of the function.  Nodes are identified by
// their nodeId, in the range [0 .. funcTDGraph.size()-1].
//----------------------------------------------------------------------------

class FunctionModRefInfo {
  const Function&       F;                  // The function
  IPModRef&             IPModRefObj;        // The IPModRef Object owning this
  DSGraph*              funcTDGraph;        // Top-down DS graph for function
  ModRefInfo            funcModRefInfo;     // ModRefInfo for the function body
  std::map<const CallInst*, ModRefInfo*>
                        callSiteModRefInfo; // ModRefInfo for each callsite
  std::map<const DSNode*, unsigned> NodeIds;

  friend class IPModRef;

  void          computeModRef   (const Function &func);
  void          computeModRef   (const CallInst& callInst);
  DSGraph *ResolveCallSiteModRefInfo(CallInst &CI,
                                hash_map<const DSNode*, DSNodeHandle> &NodeMap);

public:
  /* ctor */    FunctionModRefInfo      (const Function& func,
                                         IPModRef&       IPModRefObj,
                                         DSGraph*        tdgClone);
  /* dtor */    ~FunctionModRefInfo     ();

  // Identify the function and its relevant DS graph
  // 
  const Function& getFunction() const   { return F; }
  const DSGraph&  getFuncGraph() const  { return *funcTDGraph; }

  // Retrieve Mod/Ref results for a single call site and for the function body
  // 
  const ModRefInfo*     getModRefInfo  (const Function& func) const {
    return &funcModRefInfo;
  }
  const ModRefInfo*     getModRefInfo  (const CallInst& callInst) const {
    std::map<const CallInst*, ModRefInfo*>::const_iterator I = 
      callSiteModRefInfo.find(&callInst);
    return (I == callSiteModRefInfo.end())? NULL : I->second;
  }

  // Get the nodeIds used to index all Mod/Ref information for current function
  //
  unsigned              getNodeId       (const DSNode* node) const {
    std::map<const DSNode*, unsigned>::const_iterator iter = NodeIds.find(node);
    assert(iter != NodeIds.end() && iter->second < funcModRefInfo.getSize());
    return iter->second;
  }

  unsigned              getNodeId       (const Value* value) const;

  // Debugging support methods
  void print(std::ostream &O) const;
  void dump() const;
};


//----------------------------------------------------------------------------
// class IPModRef
// 
// Purpose:
// An interprocedural pass that computes IP Mod/Ref info for functions and
// for individual call sites.
// 
// Given the DSGraph of a function, this class can be queried for
// a ModRefInfo object describing all the nodes in the DSGraph that are
// (a) modified, and (b) referenced during an execution of the function
// from an arbitrary callsite, or during an execution of a single call-site
// within the function.
//----------------------------------------------------------------------------

class IPModRef : public Pass {
  std::map<const Function*, FunctionModRefInfo*> funcToModRefInfoMap;
  Module* M;

  FunctionModRefInfo& getFuncInfo(const Function& func,
                                  bool computeIfMissing = false);
public:
  IPModRef() : M(NULL)  { }
  ~IPModRef()           { }

  // Driver function to run IP Mod/Ref on a Module.
  // This initializes the module reference, and then computes IPModRef
  // results immediately if demand-driven analysis was *not* specified.
  // 
  virtual bool run(Module &M);

  // Retrieve the Mod/Ref information for a single function
  // 
  const FunctionModRefInfo& getFunctionModRefInfo(const Function& func) {
    return getFuncInfo(func);
  }

  /// getBUDSGraph - This method returns the BU data structure graph for F
  /// through the use of the BUDataStructures object.
  ///
  const DSGraph &getBUDSGraph(const Function &F);

  // Debugging support methods
  // 
  void print(std::ostream &O) const;
  void dump() const;

  // Release memory held by this pass when the pass pipeline is done
  // 
  virtual void releaseMemory();

  // getAnalysisUsage - This pass requires top-down data structure graphs.
  // It modifies nothing.
  // 
  virtual void getAnalysisUsage(AnalysisUsage &AU) const;
};

//===----------------------------------------------------------------------===//

#endif
