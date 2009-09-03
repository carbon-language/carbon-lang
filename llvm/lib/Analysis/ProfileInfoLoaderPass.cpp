//===- ProfileInfoLoaderPass.cpp - LLVM Pass to load profile info ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a concrete implementation of profiling information that
// loads the information from a profile dump file.
//
//===----------------------------------------------------------------------===//
#define DEBUG_TYPE "profile-loader"
#include "llvm/BasicBlock.h"
#include "llvm/InstrTypes.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/Analysis/ProfileInfo.h"
#include "llvm/Analysis/ProfileInfoLoader.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Format.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/SmallSet.h"
#include <set>
using namespace llvm;

STATISTIC(NumEdgesRead, "The # of edges read.");

static cl::opt<std::string>
ProfileInfoFilename("profile-info-file", cl::init("llvmprof.out"),
                    cl::value_desc("filename"),
                    cl::desc("Profile file loaded by -profile-loader"));

namespace {
  class VISIBILITY_HIDDEN LoaderPass : public ModulePass, public ProfileInfo {
    std::string Filename;
    std::set<Edge> SpanningTree;
    std::set<const BasicBlock*> BBisUnvisited;
  public:
    static char ID; // Class identification, replacement for typeinfo
    explicit LoaderPass(const std::string &filename = "")
      : ModulePass(&ID), Filename(filename) {
      if (filename.empty()) Filename = ProfileInfoFilename;
    }

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesAll();
    }

    virtual const char *getPassName() const {
      return "Profiling information loader";
    }

    // recurseBasicBlock() - Calculates the edge weights for as much basic
    // blocks as possbile.
    virtual void recurseBasicBlock(const BasicBlock *BB);
    virtual void readEdgeOrRemember(Edge, Edge&, unsigned &, unsigned &);
    virtual void readOrRememberEdge(ProfileInfo::Edge, unsigned,
                                    unsigned, Function*);

    /// run - Load the profile information from the specified file.
    virtual bool runOnModule(Module &M);
  };
}  // End of anonymous namespace

char LoaderPass::ID = 0;
static RegisterPass<LoaderPass>
X("profile-loader", "Load profile information from llvmprof.out", false, true);

static RegisterAnalysisGroup<ProfileInfo> Y(X);

ModulePass *llvm::createProfileLoaderPass() { return new LoaderPass(); }

/// createProfileLoaderPass - This function returns a Pass that loads the
/// profiling information for the module from the specified filename, making it
/// available to the optimizers.
Pass *llvm::createProfileLoaderPass(const std::string &Filename) {
  return new LoaderPass(Filename);
}

void LoaderPass::readEdgeOrRemember(Edge edge, Edge &tocalc, 
                                    unsigned &uncalc, unsigned &count) {
  double w;
  if ((w = getEdgeWeight(edge)) == MissingValue) {
    tocalc = edge;
    uncalc++;
  } else {
    count+=w;
  }
}

// recurseBasicBlock - Visits all neighbours of a block and then tries to
// calculate the missing edge values.
void LoaderPass::recurseBasicBlock(const BasicBlock *BB) {

  // break recursion if already visited
  if (BBisUnvisited.find(BB) == BBisUnvisited.end()) return;
  BBisUnvisited.erase(BB);
  if (!BB) return;

  for (succ_const_iterator bbi = succ_begin(BB), bbe = succ_end(BB);
       bbi != bbe; ++bbi) {
    recurseBasicBlock(*bbi);
  }
  for (pred_const_iterator bbi = pred_begin(BB), bbe = pred_end(BB);
       bbi != bbe; ++bbi) {
    recurseBasicBlock(*bbi);
  }

  Edge edgetocalc;
  unsigned uncalculated = 0;

  // collect weights of all incoming and outgoing edges, rememer edges that
  // have no value
  unsigned incount = 0;
  SmallSet<const BasicBlock*,8> pred_visited;
  pred_const_iterator bbi = pred_begin(BB), bbe = pred_end(BB);
  if (bbi==bbe) {
    readEdgeOrRemember(getEdge(0, BB),edgetocalc,uncalculated,incount);
  }
  for (;bbi != bbe; ++bbi) {
    if (pred_visited.insert(*bbi)) {
      readEdgeOrRemember(getEdge(*bbi, BB),edgetocalc,uncalculated,incount);
    }
  }

  unsigned outcount = 0;
  SmallSet<const BasicBlock*,8> succ_visited;
  succ_const_iterator sbbi = succ_begin(BB), sbbe = succ_end(BB);
  if (sbbi==sbbe) {
    readEdgeOrRemember(getEdge(BB, 0),edgetocalc,uncalculated,outcount);
  }
  for (;sbbi != sbbe; ++sbbi) {
    if (succ_visited.insert(*sbbi)) {
      readEdgeOrRemember(getEdge(BB, *sbbi),edgetocalc,uncalculated,outcount);
    }
  }

  // if exactly one edge weight was missing, calculate it and remove it from
  // spanning tree
  if (uncalculated == 1) {
    if (incount < outcount) {
      EdgeInformation[BB->getParent()][edgetocalc] = outcount-incount;
    } else {
      EdgeInformation[BB->getParent()][edgetocalc] = incount-outcount;
    }
    DEBUG(errs() << "--Calc Edge Counter for " << edgetocalc << ": "
                 << format("%g", getEdgeWeight(edgetocalc)) << "\n");
    SpanningTree.erase(edgetocalc);
  }
}

void LoaderPass::readOrRememberEdge(ProfileInfo::Edge e,
                                    unsigned weight, unsigned ei,
                                    Function *F) {
  if (weight != (unsigned)MissingValue) {
    EdgeInformation[F][e] += weight;
    DEBUG(errs()<<"--Read Edge Counter for " << e 
                <<" (# "<<ei<<"): "<<(unsigned)getEdgeWeight(e)<<"\n");
  } else {
    SpanningTree.insert(e);
  }
}

bool LoaderPass::runOnModule(Module &M) {
  ProfileInfoLoader PIL("profile-loader", Filename, M);

  EdgeInformation.clear();
  std::vector<unsigned> ECs = PIL.getRawEdgeCounts();
  if (ECs.size() > 0) {
    unsigned ei = 0;
    for (Module::iterator F = M.begin(), E = M.end(); F != E; ++F) {
      if (F->isDeclaration()) continue;
      if (ei < ECs.size())
        EdgeInformation[F][ProfileInfo::getEdge(0, &F->getEntryBlock())] +=
          ECs[ei++];
      for (Function::iterator BB = F->begin(), E = F->end(); BB != E; ++BB) {
        // Okay, we have to add a counter of each outgoing edge.  If the
        // outgoing edge is not critical don't split it, just insert the counter
        // in the source or destination of the edge.
        TerminatorInst *TI = BB->getTerminator();
        for (unsigned s = 0, e = TI->getNumSuccessors(); s != e; ++s) {
          if (ei < ECs.size())
            EdgeInformation[F][ProfileInfo::getEdge(BB, TI->getSuccessor(s))] +=
              ECs[ei++];
        }
      }
    }
    if (ei != ECs.size()) {
      errs() << "WARNING: profile information is inconsistent with "
             << "the current program!\n";
    }
    NumEdgesRead = ei;
  }

  ECs = PIL.getRawOptimalEdgeCounts();
  if (ECs.size() > 0) {
    unsigned ei = 0;
    for (Module::iterator F = M.begin(), E = M.end(); F != E; ++F) {
      if (F->isDeclaration()) continue;
      DEBUG(errs()<<"Working on "<<F->getNameStr()<<"\n");
      if (ei < ECs.size()) {
        readOrRememberEdge(getEdge(0,&F->getEntryBlock()), ECs[ei], ei, F); 
        ei++;
      }
      for (Function::iterator BB = F->begin(), E = F->end(); BB != E; ++BB) {
        TerminatorInst *TI = BB->getTerminator();
        if (TI->getNumSuccessors() == 0) {
          if (ei < ECs.size()) {
            readOrRememberEdge(getEdge(BB,0), ECs[ei], ei, F); ei++;
          }
        }
        for (unsigned s = 0, e = TI->getNumSuccessors(); s != e; ++s) {
          if (ei < ECs.size()) {
            readOrRememberEdge(getEdge(BB,TI->getSuccessor(s)), ECs[ei], ei, F);
            ei++;
          }
        }
      }
      while (SpanningTree.size() > 0) {
#if 0
        unsigned size = SpanningTree.size();
#endif
        BBisUnvisited.clear();
        for (std::set<Edge>::iterator ei = SpanningTree.begin(),
             ee = SpanningTree.end(); ei != ee; ++ei) {
          BBisUnvisited.insert(ei->first);
          BBisUnvisited.insert(ei->second);
        }
        while (BBisUnvisited.size() > 0) {
          recurseBasicBlock(*BBisUnvisited.begin());
        }
#if 0
        if (SpanningTree.size() == size) {
          DEBUG(errs()<<"{");
          for (std::set<Edge>::iterator ei = SpanningTree.begin(),
               ee = SpanningTree.end(); ei != ee; ++ei) {
            DEBUG(errs()<<"("<<(ei->first?ei->first->getName():"0")<<","
                        <<(ei->second?ei->second->getName():"0")<<"),");
          }
          assert(0 && "No edge calculated!");
        }
#endif
      }
    }
    if (ei != ECs.size()) {
      errs() << "WARNING: profile information is inconsistent with "
             << "the current program!\n";
    }
    NumEdgesRead = ei;
  }

  BlockInformation.clear();
  std::vector<unsigned> BCs = PIL.getRawBlockCounts();
  if (BCs.size() > 0) {
    unsigned bi = 0;
    for (Module::iterator F = M.begin(), E = M.end(); F != E; ++F) {
      if (F->isDeclaration()) continue;
      for (Function::iterator BB = F->begin(), E = F->end(); BB != E; ++BB)
        if (bi < BCs.size())
          BlockInformation[F][BB] = BCs[bi++];
    }
    if (bi != BCs.size()) {
      errs() << "WARNING: profile information is inconsistent with "
             << "the current program!\n";
    }
  }

  FunctionInformation.clear();
  std::vector<unsigned> FCs = PIL.getRawFunctionCounts();
  if (FCs.size() > 0) {
    unsigned fi = 0;
    for (Module::iterator F = M.begin(), E = M.end(); F != E; ++F) {
      if (F->isDeclaration()) continue;
      if (fi < FCs.size())
        FunctionInformation[F] = FCs[fi++];
    }
    if (fi != FCs.size()) {
      errs() << "WARNING: profile information is inconsistent with "
             << "the current program!\n";
    }
  }

  return false;
}
