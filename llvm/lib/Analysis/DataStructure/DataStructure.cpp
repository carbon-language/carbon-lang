//===- DataStructure.cpp - Analysis for data structure identification -------=//
//
// Implement the LLVM data structure analysis library.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/DataStructure.h"
#include "llvm/Module.h"
#include <fstream>
#include <algorithm>

//===----------------------------------------------------------------------===//
// DataStructure Class Implementation
//

AnalysisID DataStructure::ID(AnalysisID::create<DataStructure>());

// releaseMemory - If the pass pipeline is done with this pass, we can release
// our memory... here...
void DataStructure::releaseMemory() {
  for (InfoMap::iterator I = DSInfo.begin(), E = DSInfo.end(); I != E; ++I) {
    delete I->second.first;
    delete I->second.second;
  }

  // Empty map so next time memory is released, data structures are not
  // re-deleted.
  DSInfo.clear();
}

// FIXME REMOVE
#include <sys/time.h>
#include "Support/CommandLine.h"

cl::Flag   Time("t", "Print analysis time...");


// print - Print out the analysis results...
void DataStructure::print(std::ostream &O, Module *M) const {
  if (Time) {
    timeval TV1, TV2;
    gettimeofday(&TV1, 0);
    for (Module::iterator I = M->begin(), E = M->end(); I != E; ++I)
      if (!I->isExternal() && I->getName() == "main") {
        //getDSGraph(*I);
        getClosedDSGraph(I);
      }
    gettimeofday(&TV2, 0);
    std::cerr << "Analysis took "
         << (TV2.tv_sec-TV1.tv_sec)*1000000+(TV2.tv_usec-TV1.tv_usec)
         << " microseconds.\n";
  }

  for (Module::iterator I = M->begin(), E = M->end(); I != E; ++I)
    if (!I->isExternal()) {

      std::string Filename = "ds." + I->getName() + ".dot";
      O << "Writing '" << Filename << "'...";
      std::ofstream F(Filename.c_str());

      if (F.good()) {
        F << "digraph DataStructures {\n"
          << "\tnode [shape=Mrecord];\n"
          << "\tedge [arrowtail=\"dot\"];\n"
          << "\tsize=\"10,7.5\";\n"
          << "\trotate=\"90\";\n";

        getDSGraph(I).printFunction(F, "Local");
        getClosedDSGraph(I).printFunction(F, "Closed");

        F << "}\n";
      } else {
        O << "  error opening file for writing!\n";
      }

      if (Time) 
        O << " [" << getDSGraph(I).getGraphSize() << ", "
          << getClosedDSGraph(I).getGraphSize() << "]\n";
      else
        O << "\n";
    }
}


//===----------------------------------------------------------------------===//
// PointerVal Class Implementation
//

void PointerVal::print(std::ostream &O) const {
  if (Node) {
    O << "  Node: " << Node->getCaption() << "[" << Index << "]\n";
  } else {
    O << "  NULL NODE\n";
  }
}

//===----------------------------------------------------------------------===//
// PointerValSet Class Implementation
//

void PointerValSet::addRefs() {
  for (unsigned i = 0, e = Vals.size(); i != e; ++i)
    Vals[i].Node->addReferrer(this);
}

void PointerValSet::dropRefs() {
  for (unsigned i = 0, e = Vals.size(); i != e; ++i)
    Vals[i].Node->removeReferrer(this);
}

const PointerValSet &PointerValSet::operator=(const PointerValSet &PVS) {
  dropRefs();
  Vals.clear();
  Vals = PVS.Vals;
  addRefs();
  return *this;
}

// operator< - Allow insertion into a map...
bool PointerValSet::operator<(const PointerValSet &PVS) const {
  if (Vals.size() < PVS.Vals.size()) return true;
  if (Vals.size() > PVS.Vals.size()) return false;
  if (Vals.size() == 1) return Vals[0] < PVS.Vals[0];  // Most common case

  std::vector<PointerVal> S1(Vals), S2(PVS.Vals);
  sort(S1.begin(), S1.end());
  sort(S2.begin(), S2.end());
  return S1 < S2;
}

bool PointerValSet::operator==(const PointerValSet &PVS) const {
  if (Vals.size() != PVS.Vals.size()) return false;
  if (Vals.size() == 1) return Vals[0] == PVS.Vals[0];  // Most common case...

  std::vector<PointerVal> S1(Vals), S2(PVS.Vals);
  sort(S1.begin(), S1.end());
  sort(S2.begin(), S2.end());
  return S1 == S2;
}


bool PointerValSet::add(const PointerVal &PV, Value *Pointer) {
  if (std::find(Vals.begin(), Vals.end(), PV) != Vals.end())
    return false;
  Vals.push_back(PV);
  if (Pointer) PV.Node->addPointer(Pointer);
  PV.Node->addReferrer(this);
  return true;
}

// removePointerTo - Remove a single pointer val that points to the specified
// node...
void PointerValSet::removePointerTo(DSNode *Node) {
  std::vector<PointerVal>::iterator I = std::find(Vals.begin(), Vals.end(), Node);
  assert(I != Vals.end() && "Couldn't remove nonexistent edge!");
  Vals.erase(I);
  Node->removeReferrer(this);
}


void PointerValSet::print(std::ostream &O) const {
  for (unsigned i = 0, e = Vals.size(); i != e; ++i)
    Vals[i].print(O);
}

