//===- DataStructure.cpp - Analysis for data structure identification -------=//
//
// Implement the LLVM data structure analysis library.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/DataStructure.h"
#include "llvm/Module.h"
#include "llvm/Function.h"
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


// print - Print out the analysis results...
void DataStructure::print(std::ostream &O, Module *M) const {
  for (Module::iterator I = M->begin(), E = M->end(); I != E; ++I)
    if (!(*I)->isExternal()) {

      string Filename = "ds." + (*I)->getName() + ".dot";
      O << "Writing '" << Filename << "'...\n";
      ofstream F(Filename.c_str());
      if (F.good()) {
        F << "digraph DataStructures {\n"
          << "\tnode [shape=Mrecord];\n"
          << "\tedge [arrowtail=\"dot\"];\n"
          << "\tsize=\"10,7.5\";\n"
          << "\trotate=\"90\";\n";

        getDSGraph(*I).printFunction(F, "Local");
        getClosedDSGraph(*I).printFunction(F, "Closed");

        F << "}\n";
      } else {
        O << "  error opening file for writing!\n";
      }
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
  vector<PointerVal>::iterator I = std::find(Vals.begin(), Vals.end(), Node);
  assert(I != Vals.end() && "Couldn't remove nonexistent edge!");
  Vals.erase(I);
  Node->removeReferrer(this);
}


void PointerValSet::print(std::ostream &O) const {
  for (unsigned i = 0, e = Vals.size(); i != e; ++i)
    Vals[i].print(O);
}

