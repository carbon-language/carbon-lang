//===- MemoryDepAnalysis.cpp - Compute dep graph for memory ops --*-C++-*--===//
//
// This file implements a pass (MemoryDepAnalysis) that computes memory-based
// data dependences between instructions for each function in a module.  
// Memory-based dependences occur due to load and store operations, but
// also the side-effects of call instructions.
//
// The result of this pass is a DependenceGraph for each function
// representing the memory-based data dependences between instructions.
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/MemoryDepAnalysis.h"
#include "llvm/Analysis/IPModRef.h"
#include "llvm/Analysis/DataStructure.h"
#include "llvm/Analysis/DSGraph.h"
#include "llvm/Module.h"
#include "llvm/iMemory.h"
#include "llvm/iOther.h"
#include "llvm/Support/InstVisitor.h"
#include "llvm/Support/CFG.h"
#include "Support/TarjanSCCIterator.h"
#include "Support/Statistic.h"
#include "Support/STLExtras.h"
#include "Support/hash_map"
#include "Support/hash_set"


///--------------------------------------------------------------------------
/// struct ModRefTable:
/// 
/// A data structure that tracks ModRefInfo for instructions:
///   -- modRefMap is a map of Instruction* -> ModRefInfo for the instr.
///   -- definers  is a vector of instructions that define    any node
///   -- users     is a vector of instructions that reference any node
///   -- numUsersBeforeDef is a vector indicating that the number of users
///                seen before definers[i] is numUsersBeforeDef[i].
/// 
/// numUsersBeforeDef[] effectively tells us the exact interleaving of
/// definers and users within the ModRefTable.
/// This is only maintained when constructing the table for one SCC, and
/// not copied over from one table to another since it is no longer useful.
///--------------------------------------------------------------------------

struct ModRefTable {
  typedef hash_map<Instruction*, ModRefInfo> ModRefMap;
  typedef ModRefMap::const_iterator                 const_map_iterator;
  typedef ModRefMap::      iterator                        map_iterator;
  typedef std::vector<Instruction*>::const_iterator const_ref_iterator;
  typedef std::vector<Instruction*>::      iterator       ref_iterator;

  ModRefMap                 modRefMap;
  std::vector<Instruction*> definers;
  std::vector<Instruction*> users;
  std::vector<unsigned>     numUsersBeforeDef;

  // Iterators to enumerate all the defining instructions
  const_ref_iterator defsBegin()  const {  return definers.begin(); }
        ref_iterator defsBegin()        {  return definers.begin(); }
  const_ref_iterator defsEnd()    const {  return definers.end(); }
        ref_iterator defsEnd()          {  return definers.end(); }

  // Iterators to enumerate all the user instructions
  const_ref_iterator usersBegin() const {  return users.begin(); }
        ref_iterator usersBegin()       {  return users.begin(); }
  const_ref_iterator usersEnd()   const {  return users.end(); }
        ref_iterator usersEnd()         {  return users.end(); }

  // Iterator identifying the last user that was seen *before* a
  // specified def.  In particular, all users in the half-closed range
  //    [ usersBegin(), usersBeforeDef_End(defPtr) )
  // were seen *before* the specified def.  All users in the half-closed range
  //    [ usersBeforeDef_End(defPtr), usersEnd() )
  // were seen *after* the specified def.
  // 
  ref_iterator usersBeforeDef_End(const_ref_iterator defPtr) {
    unsigned defIndex = (unsigned) (defPtr - defsBegin());
    assert(defIndex < numUsersBeforeDef.size());
    assert(usersBegin() + numUsersBeforeDef[defIndex] <= usersEnd()); 
    return usersBegin() + numUsersBeforeDef[defIndex]; 
  }
  const_ref_iterator usersBeforeDef_End(const_ref_iterator defPtr) const {
    return const_cast<ModRefTable*>(this)->usersBeforeDef_End(defPtr);
  }

  // 
  // Modifier methods
  // 
  void AddDef(Instruction* D) {
    definers.push_back(D);
    numUsersBeforeDef.push_back(users.size());
  }
  void AddUse(Instruction* U) {
    users.push_back(U);
  }
  void Insert(const ModRefTable& fromTable) {
    modRefMap.insert(fromTable.modRefMap.begin(), fromTable.modRefMap.end());
    definers.insert(definers.end(),
                    fromTable.definers.begin(), fromTable.definers.end());
    users.insert(users.end(),
                 fromTable.users.begin(), fromTable.users.end());
    numUsersBeforeDef.clear(); /* fromTable.numUsersBeforeDef is ignored */
  }
};


///--------------------------------------------------------------------------
/// class ModRefInfoBuilder:
/// 
/// A simple InstVisitor<> class that retrieves the Mod/Ref info for
/// Load/Store/Call instructions and inserts this information in
/// a ModRefTable.  It also records all instructions that Mod any node
/// and all that use any node.
///--------------------------------------------------------------------------

class ModRefInfoBuilder : public InstVisitor<ModRefInfoBuilder> {
  const DSGraph&            funcGraph;
  const FunctionModRefInfo& funcModRef;
  ModRefTable&              modRefTable;

  ModRefInfoBuilder();                         // DO NOT IMPLEMENT
  ModRefInfoBuilder(const ModRefInfoBuilder&); // DO NOT IMPLEMENT
  void operator=(const ModRefInfoBuilder&);    // DO NOT IMPLEMENT

public:
  /*ctor*/      ModRefInfoBuilder(const DSGraph&  _funcGraph,
                                  const FunctionModRefInfo& _funcModRef,
                                  ModRefTable&    _modRefTable)
    : funcGraph(_funcGraph), funcModRef(_funcModRef), modRefTable(_modRefTable)
  {
  }

  // At a call instruction, retrieve the ModRefInfo using IPModRef results.
  // Add the call to the defs list if it modifies any nodes and to the uses
  // list if it refs any nodes.
  // 
  void          visitCallInst   (CallInst& callInst) {
    ModRefInfo safeModRef(funcGraph.getGraphSize());
    const ModRefInfo* callModRef = funcModRef.getModRefInfo(callInst);
    if (callModRef == NULL)
      { // call to external/unknown function: mark all nodes as Mod and Ref
        safeModRef.getModSet().set();
        safeModRef.getRefSet().set();
        callModRef = &safeModRef;
      }

    modRefTable.modRefMap.insert(std::make_pair(&callInst,
                                                ModRefInfo(*callModRef)));
    if (callModRef->getModSet().any())
      modRefTable.AddDef(&callInst);
    if (callModRef->getRefSet().any())
      modRefTable.AddUse(&callInst);
  }

  // At a store instruction, add to the mod set the single node pointed to
  // by the pointer argument of the store.  Interestingly, if there is no
  // such node, that would be a null pointer reference!
  void          visitStoreInst  (StoreInst& storeInst) {
    const DSNodeHandle& ptrNode =
      funcGraph.getNodeForValue(storeInst.getPointerOperand());
    if (const DSNode* target = ptrNode.getNode())
      {
        unsigned nodeId = funcModRef.getNodeId(target);
        ModRefInfo& minfo =
          modRefTable.modRefMap.insert(
            std::make_pair(&storeInst,
                           ModRefInfo(funcGraph.getGraphSize()))).first->second;
        minfo.setNodeIsMod(nodeId);
        modRefTable.AddDef(&storeInst);
      }
    else
      std::cerr << "Warning: Uninitialized pointer reference!\n";
  }

  // At a load instruction, add to the ref set the single node pointed to
  // by the pointer argument of the load.  Interestingly, if there is no
  // such node, that would be a null pointer reference!
  void          visitLoadInst  (LoadInst& loadInst) {
    const DSNodeHandle& ptrNode =
      funcGraph.getNodeForValue(loadInst.getPointerOperand());
    if (const DSNode* target = ptrNode.getNode())
      {
        unsigned nodeId = funcModRef.getNodeId(target);
        ModRefInfo& minfo =
          modRefTable.modRefMap.insert(
            std::make_pair(&loadInst,
                           ModRefInfo(funcGraph.getGraphSize()))).first->second;
        minfo.setNodeIsRef(nodeId);
        modRefTable.AddUse(&loadInst);
      }
    else
      std::cerr << "Warning: Uninitialized pointer reference!\n";
  }
};


//----------------------------------------------------------------------------
// class MemoryDepAnalysis: A dep. graph for load/store/call instructions
//----------------------------------------------------------------------------


/// getAnalysisUsage - This does not modify anything.  It uses the Top-Down DS
/// Graph and IPModRef.
///
void MemoryDepAnalysis::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequired<TDDataStructures>();
  AU.addRequired<IPModRef>();
}


/// Basic dependence gathering algorithm, using TarjanSCCIterator on CFG:
/// 
/// for every SCC S in the CFG in PostOrder on the SCC DAG
///     {
///       for every basic block BB in S in *postorder*
///         for every instruction I in BB in reverse
///           Add (I, ModRef[I]) to ModRefCurrent
///           if (Mod[I] != NULL)
///               Add I to DefSetCurrent:  { I \in S : Mod[I] != NULL }
///           if (Ref[I] != NULL)
///               Add I to UseSetCurrent:  { I       : Ref[I] != NULL }
/// 
///       for every def D in DefSetCurrent
/// 
///           // NOTE: D comes after itself iff S contains a loop
///           if (HasLoop(S) && D & D)
///               Add output-dep: D -> D2
/// 
///           for every def D2 *after* D in DefSetCurrent
///               // NOTE: D2 comes before D in execution order
///               if (D & D2)
///                   Add output-dep: D2 -> D
///                   if (HasLoop(S))
///                       Add output-dep: D -> D2
/// 
///           for every use U in UseSetCurrent that was seen *before* D
///               // NOTE: U comes after D in execution order
///               if (U & D)
///                   if (U != D || HasLoop(S))
///                       Add true-dep: D -> U
///                   if (HasLoop(S))
///                       Add anti-dep: U -> D
/// 
///           for every use U in UseSetCurrent that was seen *after* D
///               // NOTE: U comes before D in execution order
///               if (U & D)
///                   if (U != D || HasLoop(S))
///                       Add anti-dep: U -> D
///                   if (HasLoop(S))
///                       Add true-dep: D -> U
/// 
///           for every def Dnext in DefSetAfter
///               // NOTE: Dnext comes after D in execution order
///               if (Dnext & D)
///                   Add output-dep: D -> Dnext
/// 
///           for every use Unext in UseSetAfter
///               // NOTE: Unext comes after D in execution order
///               if (Unext & D)
///                   Add true-dep: D -> Unext
/// 
///       for every use U in UseSetCurrent
///           for every def Dnext in DefSetAfter
///               // NOTE: Dnext comes after U in execution order
///               if (Dnext & D)
///                   Add anti-dep: U -> Dnext
/// 
///       Add ModRefCurrent to ModRefAfter: { (I, ModRef[I] ) }
///       Add DefSetCurrent to DefSetAfter: { I : Mod[I] != NULL }
///       Add UseSetCurrent to UseSetAfter: { I : Ref[I] != NULL }
///     }
///         
///
void MemoryDepAnalysis::ProcessSCC(std::vector<BasicBlock*> &S,
                                   ModRefTable& ModRefAfter, bool hasLoop) {
  ModRefTable ModRefCurrent;
  ModRefTable::ModRefMap& mapCurrent = ModRefCurrent.modRefMap;
  ModRefTable::ModRefMap& mapAfter   = ModRefAfter.modRefMap;

  // Builder class fills out a ModRefTable one instruction at a time.
  // To use it, we just invoke it's visit function for each basic block:
  // 
  //   for each basic block BB in the SCC in *postorder*
  //       for each instruction  I in BB in *reverse*
  //           ModRefInfoBuilder::visit(I)
  //           : Add (I, ModRef[I]) to ModRefCurrent.modRefMap
  //           : Add I  to ModRefCurrent.definers if it defines any node
  //           : Add I  to ModRefCurrent.users    if it uses any node
  // 
  ModRefInfoBuilder builder(*funcGraph, *funcModRef, ModRefCurrent);
  for (std::vector<BasicBlock*>::iterator BI = S.begin(), BE = S.end();
       BI != BE; ++BI)
    // Note: BBs in the SCC<> created by TarjanSCCIterator are in postorder.
    for (BasicBlock::reverse_iterator II=(*BI)->rbegin(), IE=(*BI)->rend();
         II != IE; ++II)
      builder.visit(*II);

  ///       for every def D in DefSetCurrent
  /// 
  for (ModRefTable::ref_iterator II=ModRefCurrent.defsBegin(),
         IE=ModRefCurrent.defsEnd(); II != IE; ++II)
    {
      ///           // NOTE: D comes after itself iff S contains a loop
      ///           if (HasLoop(S))
      ///               Add output-dep: D -> D2
      if (hasLoop)
        funcDepGraph->AddSimpleDependence(**II, **II, OutputDependence);

      ///           for every def D2 *after* D in DefSetCurrent
      ///               // NOTE: D2 comes before D in execution order
      ///               if (D2 & D)
      ///                   Add output-dep: D2 -> D
      ///                   if (HasLoop(S))
      ///                       Add output-dep: D -> D2
      for (ModRefTable::ref_iterator JI=II+1; JI != IE; ++JI)
        if (!Disjoint(mapCurrent.find(*II)->second.getModSet(),
                      mapCurrent.find(*JI)->second.getModSet()))
          {
            funcDepGraph->AddSimpleDependence(**JI, **II, OutputDependence);
            if (hasLoop)
              funcDepGraph->AddSimpleDependence(**II, **JI, OutputDependence);
          }
  
      ///           for every use U in UseSetCurrent that was seen *before* D
      ///               // NOTE: U comes after D in execution order
      ///               if (U & D)
      ///                   if (U != D || HasLoop(S))
      ///                       Add true-dep: U -> D
      ///                   if (HasLoop(S))
      ///                       Add anti-dep: D -> U
      ModRefTable::ref_iterator JI=ModRefCurrent.usersBegin();
      ModRefTable::ref_iterator JE = ModRefCurrent.usersBeforeDef_End(II);
      for ( ; JI != JE; ++JI)
        if (!Disjoint(mapCurrent.find(*II)->second.getModSet(),
                      mapCurrent.find(*JI)->second.getRefSet()))
          {
            if (*II != *JI || hasLoop)
              funcDepGraph->AddSimpleDependence(**II, **JI, TrueDependence);
            if (hasLoop)
              funcDepGraph->AddSimpleDependence(**JI, **II, AntiDependence);
          }

      ///           for every use U in UseSetCurrent that was seen *after* D
      ///               // NOTE: U comes before D in execution order
      ///               if (U & D)
      ///                   if (U != D || HasLoop(S))
      ///                       Add anti-dep: U -> D
      ///                   if (HasLoop(S))
      ///                       Add true-dep: D -> U
      for (/*continue JI*/ JE = ModRefCurrent.usersEnd(); JI != JE; ++JI)
        if (!Disjoint(mapCurrent.find(*II)->second.getModSet(),
                      mapCurrent.find(*JI)->second.getRefSet()))
          {
            if (*II != *JI || hasLoop)
              funcDepGraph->AddSimpleDependence(**JI, **II, AntiDependence);
            if (hasLoop)
              funcDepGraph->AddSimpleDependence(**II, **JI, TrueDependence);
          }

      ///           for every def Dnext in DefSetPrev
      ///               // NOTE: Dnext comes after D in execution order
      ///               if (Dnext & D)
      ///                   Add output-dep: D -> Dnext
      for (ModRefTable::ref_iterator JI=ModRefAfter.defsBegin(),
             JE=ModRefAfter.defsEnd(); JI != JE; ++JI)
        if (!Disjoint(mapCurrent.find(*II)->second.getModSet(),
                      mapAfter.find(*JI)->second.getModSet()))
          funcDepGraph->AddSimpleDependence(**II, **JI, OutputDependence);

      ///           for every use Unext in UseSetAfter
      ///               // NOTE: Unext comes after D in execution order
      ///               if (Unext & D)
      ///                   Add true-dep: D -> Unext
      for (ModRefTable::ref_iterator JI=ModRefAfter.usersBegin(),
             JE=ModRefAfter.usersEnd(); JI != JE; ++JI)
        if (!Disjoint(mapCurrent.find(*II)->second.getModSet(),
                      mapAfter.find(*JI)->second.getRefSet()))
          funcDepGraph->AddSimpleDependence(**II, **JI, TrueDependence);
    }

  /// 
  ///       for every use U in UseSetCurrent
  ///           for every def Dnext in DefSetAfter
  ///               // NOTE: Dnext comes after U in execution order
  ///               if (Dnext & D)
  ///                   Add anti-dep: U -> Dnext
  for (ModRefTable::ref_iterator II=ModRefCurrent.usersBegin(),
         IE=ModRefCurrent.usersEnd(); II != IE; ++II)
    for (ModRefTable::ref_iterator JI=ModRefAfter.defsBegin(),
           JE=ModRefAfter.defsEnd(); JI != JE; ++JI)
      if (!Disjoint(mapCurrent.find(*II)->second.getRefSet(),
                    mapAfter.find(*JI)->second.getModSet()))
        funcDepGraph->AddSimpleDependence(**II, **JI, AntiDependence);
    
  ///       Add ModRefCurrent to ModRefAfter: { (I, ModRef[I] ) }
  ///       Add DefSetCurrent to DefSetAfter: { I : Mod[I] != NULL }
  ///       Add UseSetCurrent to UseSetAfter: { I : Ref[I] != NULL }
  ModRefAfter.Insert(ModRefCurrent);
}


/// Debugging support methods
/// 
void MemoryDepAnalysis::print(std::ostream &O) const
{
  // TEMPORARY LOOP
  for (hash_map<Function*, DependenceGraph*>::const_iterator
         I = funcMap.begin(), E = funcMap.end(); I != E; ++I)
    {
      Function* func = I->first;
      DependenceGraph* depGraph = I->second;

  O << "\n================================================================\n";
  O << "DEPENDENCE GRAPH FOR MEMORY OPERATIONS IN FUNCTION " << func->getName();
  O << "\n================================================================\n\n";
  depGraph->print(*func, O);

    }
}


/// 
/// Run the pass on a function
/// 
bool MemoryDepAnalysis::runOnFunction(Function &F) {
  assert(!F.isExternal());

  // Get the FunctionModRefInfo holding IPModRef results for this function.
  // Use the TD graph recorded within the FunctionModRefInfo object, which
  // may not be the same as the original TD graph computed by DS analysis.
  // 
  funcModRef = &getAnalysis<IPModRef>().getFunctionModRefInfo(F);
  funcGraph  = &funcModRef->getFuncGraph();

  // TEMPORARY: ptr to depGraph (later just becomes "this").
  assert(!funcMap.count(&F) && "Analyzing function twice?");
  funcDepGraph = funcMap[&F] = new DependenceGraph();

  ModRefTable ModRefAfter;

  SCC<Function*>* nextSCC;
  for (TarjanSCC_iterator<Function*> I = tarj_begin(&F), E = tarj_end(&F);
       I != E; ++I)
    ProcessSCC(*I, ModRefAfter, I.hasLoop());

  return true;
}


//-------------------------------------------------------------------------
// TEMPORARY FUNCTIONS TO MAKE THIS A MODULE PASS ---
// These functions will go away once this class becomes a FunctionPass.
// 

// Driver function to compute dependence graphs for every function.
// This is temporary and will go away once this is a FunctionPass.
// 
bool MemoryDepAnalysis::run(Module& M)
{
  for (Module::iterator FI = M.begin(), FE = M.end(); FI != FE; ++FI)
    if (! FI->isExternal())
      runOnFunction(*FI); // automatically inserts each depGraph into funcMap
  return true;
}
  
// Release all the dependence graphs in the map.
void MemoryDepAnalysis::releaseMemory()
{
  for (hash_map<Function*, DependenceGraph*>::const_iterator
         I = funcMap.begin(), E = funcMap.end(); I != E; ++I)
    delete I->second;
  funcMap.clear();

  // Clear pointers because the pass constructor will not be invoked again.
  funcDepGraph = NULL;
  funcGraph = NULL;
  funcModRef = NULL;
}

MemoryDepAnalysis::~MemoryDepAnalysis()
{
  releaseMemory();
}

//----END TEMPORARY FUNCTIONS----------------------------------------------


void MemoryDepAnalysis::dump() const
{
  this->print(std::cerr);
}

static RegisterAnalysis<MemoryDepAnalysis>
Z("memdep", "Memory Dependence Analysis");

