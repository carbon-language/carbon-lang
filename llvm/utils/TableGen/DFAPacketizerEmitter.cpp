//===- DFAPacketizerEmitter.cpp - Packetization DFA for a VLIW machine-----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class parses the Schedule.td file and produces an API that can be used
// to reason about whether an instruction can be added to a packet on a VLIW
// architecture. The class internally generates a deterministic finite
// automaton (DFA) that models all possible mappings of machine instructions
// to functional units as instructions are added to a packet.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCInstrItineraries.h"
#include "llvm/TableGen/Record.h"
#include "CodeGenTarget.h"
#include "DFAPacketizerEmitter.h"
#include <list>

using namespace llvm;

//
//
// State represents the usage of machine resources if the packet contains
// a set of instruction classes.
//
// Specifically, currentState is a set of bit-masks.
// The nth bit in a bit-mask indicates whether the nth resource is being used
// by this state. The set of bit-masks in a state represent the different
// possible outcomes of transitioning to this state.
// For example: consider a two resource architecture: resource L and resource M
// with three instruction classes: L, M, and L_or_M.
// From the initial state (currentState = 0x00), if we add instruction class
// L_or_M we will transition to a state with currentState = [0x01, 0x10]. This
// represents the possible resource states that can result from adding a L_or_M
// instruction
//
// Another way of thinking about this transition is we are mapping a NDFA with
// two states [0x01] and [0x10] into a DFA with a single state [0x01, 0x10].
//
//
namespace {
class State {
 public:
  static int currentStateNum;
  int stateNum;
  bool isInitial;
  std::set<unsigned> stateInfo;

  State();
  State(const State& S);

  //
  // canAddInsnClass - Returns true if an instruction of type InsnClass is a
  // valid transition from this state, i.e., can an instruction of type InsnClass
  // be added to the packet represented by this state.
  //
  // PossibleStates is the set of valid resource states that ensue from valid
  // transitions.
  //
  bool canAddInsnClass(unsigned InsnClass, std::set<unsigned>& PossibleStates);
};
} // End anonymous namespace.


namespace {
struct Transition {
 public:
  static int currentTransitionNum;
  int transitionNum;
  State* from;
  unsigned input;
  State* to;

  Transition(State* from_, unsigned input_, State* to_);
};
} // End anonymous namespace.


//
// Comparators to keep set of states sorted.
//
namespace {
struct ltState {
  bool operator()(const State* s1, const State* s2) const;
};
} // End anonymous namespace.


//
// class DFA: deterministic finite automaton for processor resource tracking.
//
namespace {
class DFA {
public:
  DFA();

  // Set of states. Need to keep this sorted to emit the transition table.
  std::set<State*, ltState> states;

  // Map from a state to the list of transitions with that state as source.
  std::map<State*, SmallVector<Transition*, 16>, ltState> stateTransitions;
  State* currentState;

  // Highest valued Input seen.
  unsigned LargestInput;

  //
  // Modify the DFA.
  //
  void initialize();
  void addState(State*);
  void addTransition(Transition*);

  //
  // getTransition -  Return the state when a transition is made from
  // State From with Input I. If a transition is not found, return NULL.
  //
  State* getTransition(State*, unsigned);

  //
  // isValidTransition: Predicate that checks if there is a valid transition
  // from state From on input InsnClass.
  //
  bool isValidTransition(State* From, unsigned InsnClass);

  //
  // writeTable: Print out a table representing the DFA.
  //
  void writeTableAndAPI(raw_ostream &OS, const std::string& ClassName);
};
} // End anonymous namespace.


//
// Constructors for State, Transition, and DFA
//
State::State() :
  stateNum(currentStateNum++), isInitial(false) {}


State::State(const State& S) :
  stateNum(currentStateNum++), isInitial(S.isInitial),
  stateInfo(S.stateInfo) {}


Transition::Transition(State* from_, unsigned input_, State* to_) :
  transitionNum(currentTransitionNum++), from(from_), input(input_),
  to(to_) {}


DFA::DFA() :
  LargestInput(0) {}


bool ltState::operator()(const State* s1, const State* s2) const {
    return (s1->stateNum < s2->stateNum);
}


//
// canAddInsnClass - Returns true if an instruction of type InsnClass is a
// valid transition from this state i.e., can an instruction of type InsnClass
// be added to the packet represented by this state.
//
// PossibleStates is the set of valid resource states that ensue from valid
// transitions.
//
bool State::canAddInsnClass(unsigned InsnClass,
                            std::set<unsigned>& PossibleStates) {
  //
  // Iterate over all resource states in currentState.
  //
  bool AddedState = false;

  for (std::set<unsigned>::iterator SI = stateInfo.begin();
       SI != stateInfo.end(); ++SI) {
    unsigned thisState = *SI;

    //
    // Iterate over all possible resources used in InsnClass.
    // For ex: for InsnClass = 0x11, all resources = {0x01, 0x10}.
    //

    DenseSet<unsigned> VisitedResourceStates;
    for (unsigned int j = 0; j < sizeof(InsnClass) * 8; ++j) {
      if ((0x1 << j) & InsnClass) {
        //
        // For each possible resource used in InsnClass, generate the
        // resource state if that resource was used.
        //
        unsigned ResultingResourceState = thisState | (0x1 << j);
        //
        // Check if the resulting resource state can be accommodated in this
        // packet.
        // We compute ResultingResourceState OR thisState.
        // If the result of the OR is different than thisState, it implies
        // that there is at least one resource that can be used to schedule
        // InsnClass in the current packet.
        // Insert ResultingResourceState into PossibleStates only if we haven't
        // processed ResultingResourceState before.
        //
        if ((ResultingResourceState != thisState) &&
            (VisitedResourceStates.count(ResultingResourceState) == 0)) {
          VisitedResourceStates.insert(ResultingResourceState);
          PossibleStates.insert(ResultingResourceState);
          AddedState = true;
        }
      }
    }
  }

  return AddedState;
}


void DFA::initialize() {
  currentState->isInitial = true;
}


void DFA::addState(State* S) {
  assert(!states.count(S) && "State already exists");
  states.insert(S);
}


void DFA::addTransition(Transition* T) {
  // Update LargestInput.
  if (T->input > LargestInput)
    LargestInput = T->input;

  // Add the new transition.
  stateTransitions[T->from].push_back(T);
}


//
// getTransition - Return the state when a transition is made from
// State From with Input I. If a transition is not found, return NULL.
//
State* DFA::getTransition(State* From, unsigned I) {
  // Do we have a transition from state From?
  if (!stateTransitions.count(From))
    return NULL;

  // Do we have a transition from state From with Input I?
  for (SmallVector<Transition*, 16>::iterator VI =
         stateTransitions[From].begin();
         VI != stateTransitions[From].end(); ++VI)
    if ((*VI)->input == I)
      return (*VI)->to;

  return NULL;
}


bool DFA::isValidTransition(State* From, unsigned InsnClass) {
  return (getTransition(From, InsnClass) != NULL);
}


int State::currentStateNum = 0;
int Transition::currentTransitionNum = 0;

DFAGen::DFAGen(RecordKeeper& R):
  TargetName(CodeGenTarget(R).getName()),
  allInsnClasses(), Records(R) {}


//
// writeTableAndAPI - Print out a table representing the DFA and the
// associated API to create a DFA packetizer.
//
// Format:
// DFAStateInputTable[][2] = pairs of <Input, Transition> for all valid
//                           transitions.
// DFAStateEntryTable[i] = Index of the first entry in DFAStateInputTable for
//                         the ith state.
//
//
void DFA::writeTableAndAPI(raw_ostream &OS, const std::string& TargetName) {
  std::set<State*, ltState>::iterator SI = states.begin();
  // This table provides a map to the beginning of the transitions for State s
  // in DFAStateInputTable.
  std::vector<int> StateEntry(states.size());

  OS << "namespace llvm {\n\n";
  OS << "const int " << TargetName << "DFAStateInputTable[][2] = {\n";

  // Tracks the total valid transitions encountered so far. It is used
  // to construct the StateEntry table.
  int ValidTransitions = 0;
  for (unsigned i = 0; i < states.size(); ++i, ++SI) {
    StateEntry[i] = ValidTransitions;
    for (unsigned j = 0; j <= LargestInput; ++j) {
      assert (((*SI)->stateNum == (int) i) && "Mismatch in state numbers");
      if (!isValidTransition(*SI, j))
        continue;

      OS << "{" << j << ", "
         << getTransition(*SI, j)->stateNum
         << "},    ";
      ++ValidTransitions;
    }

    // If there are no valid transitions from this stage, we need a sentinel
    // transition.
    if (ValidTransitions == StateEntry[i])
      OS << "{-1, -1},";

    OS << "\n";
  }
  OS << "};\n\n";
  OS << "const unsigned int " << TargetName << "DFAStateEntryTable[] = {\n";

  // Multiply i by 2 since each entry in DFAStateInputTable is a set of
  // two numbers.
  for (unsigned i = 0; i < states.size(); ++i)
    OS << StateEntry[i] << ", ";

  OS << "\n};\n";
  OS << "} // namespace\n";


  //
  // Emit DFA Packetizer tables if the target is a VLIW machine.
  //
  std::string SubTargetClassName = TargetName + "GenSubtargetInfo";
  OS << "\n" << "#include \"llvm/CodeGen/DFAPacketizer.h\"\n";
  OS << "namespace llvm {\n";
  OS << "DFAPacketizer* " << SubTargetClassName << "::"
     << "createDFAPacketizer(const InstrItineraryData *IID) const {\n"
     << "   return new DFAPacketizer(IID, " << TargetName
     << "DFAStateInputTable, " << TargetName << "DFAStateEntryTable);\n}\n\n";
  OS << "} // End llvm namespace \n";
}


//
// collectAllInsnClasses - Populate allInsnClasses which is a set of units
// used in each stage.
//
void DFAGen::collectAllInsnClasses(const std::string &Name,
                                  Record *ItinData,
                                  unsigned &NStages,
                                  raw_ostream &OS) {
  // Collect processor itineraries.
  std::vector<Record*> ProcItinList =
    Records.getAllDerivedDefinitions("ProcessorItineraries");

  // If just no itinerary then don't bother.
  if (ProcItinList.size() < 2)
    return;
  std::map<std::string, unsigned> NameToBitsMap;

  // Parse functional units for all the itineraries.
  for (unsigned i = 0, N = ProcItinList.size(); i < N; ++i) {
    Record *Proc = ProcItinList[i];
    std::vector<Record*> FUs = Proc->getValueAsListOfDefs("FU");

    // Convert macros to bits for each stage.
    for (unsigned i = 0, N = FUs.size(); i < N; ++i)
      NameToBitsMap[FUs[i]->getName()] = (unsigned) (1U << i);
  }

  const std::vector<Record*> &StageList =
    ItinData->getValueAsListOfDefs("Stages");

  // The number of stages.
  NStages = StageList.size();

  // For each unit.
  unsigned UnitBitValue = 0;

  // Compute the bitwise or of each unit used in this stage.
  for (unsigned i = 0; i < NStages; ++i) {
    const Record *Stage = StageList[i];

    // Get unit list.
    const std::vector<Record*> &UnitList =
      Stage->getValueAsListOfDefs("Units");

    for (unsigned j = 0, M = UnitList.size(); j < M; ++j) {
      // Conduct bitwise or.
      std::string UnitName = UnitList[j]->getName();
      assert(NameToBitsMap.count(UnitName));
      UnitBitValue |= NameToBitsMap[UnitName];
    }

    if (UnitBitValue != 0)
      allInsnClasses.insert(UnitBitValue);
  }
}


//
// Run the worklist algorithm to generate the DFA.
//
void DFAGen::run(raw_ostream &OS) {
  EmitSourceFileHeader("Target DFA Packetizer Tables", OS);

  // Collect processor iteraries.
  std::vector<Record*> ProcItinList =
    Records.getAllDerivedDefinitions("ProcessorItineraries");

  //
  // Collect the instruction classes.
  //
  for (unsigned i = 0, N = ProcItinList.size(); i < N; i++) {
    Record *Proc = ProcItinList[i];

    // Get processor itinerary name.
    const std::string &Name = Proc->getName();

    // Skip default.
    if (Name == "NoItineraries")
      continue;

    // Sanity check for at least one instruction itinerary class.
    unsigned NItinClasses =
      Records.getAllDerivedDefinitions("InstrItinClass").size();
    if (NItinClasses == 0)
      return;

    // Get itinerary data list.
    std::vector<Record*> ItinDataList = Proc->getValueAsListOfDefs("IID");

    // Collect instruction classes for all itinerary data.
    for (unsigned j = 0, M = ItinDataList.size(); j < M; j++) {
      Record *ItinData = ItinDataList[j];
      unsigned NStages;
      collectAllInsnClasses(Name, ItinData, NStages, OS);
    }
  }


  //
  // Run a worklist algorithm to generate the DFA.
  //
  DFA D;
  State* Initial = new State;
  Initial->isInitial = true;
  Initial->stateInfo.insert(0x0);
  D.addState(Initial);
  SmallVector<State*, 32> WorkList;
  std::map<std::set<unsigned>, State*> Visited;

  WorkList.push_back(Initial);

  //
  // Worklist algorithm to create a DFA for processor resource tracking.
  // C = {set of InsnClasses}
  // Begin with initial node in worklist. Initial node does not have
  // any consumed resources,
  //     ResourceState = 0x0
  // Visited = {}
  // While worklist != empty
  //    S = first element of worklist
  //    For every instruction class C
  //      if we can accommodate C in S:
  //          S' = state with resource states = {S Union C}
  //          Add a new transition: S x C -> S'
  //          If S' is not in Visited:
  //             Add S' to worklist
  //             Add S' to Visited
  //
  while (!WorkList.empty()) {
    State* current = WorkList.pop_back_val();
    for (DenseSet<unsigned>::iterator CI = allInsnClasses.begin(),
           CE = allInsnClasses.end(); CI != CE; ++CI) {
      unsigned InsnClass = *CI;

      std::set<unsigned> NewStateResources;
      //
      // If we haven't already created a transition for this input
      // and the state can accommodate this InsnClass, create a transition.
      //
      if (!D.getTransition(current, InsnClass) &&
          current->canAddInsnClass(InsnClass, NewStateResources)) {
        State* NewState = NULL;

        //
        // If we have seen this state before, then do not create a new state.
        //
        //
        std::map<std::set<unsigned>, State*>::iterator VI;
        if ((VI = Visited.find(NewStateResources)) != Visited.end())
          NewState = VI->second;
        else {
          NewState = new State;
          NewState->stateInfo = NewStateResources;
          D.addState(NewState);
          Visited[NewStateResources] = NewState;
          WorkList.push_back(NewState);
        }

        Transition* NewTransition = new Transition(current, InsnClass,
                                                   NewState);
        D.addTransition(NewTransition);
      }
    }
  }

  // Print out the table.
  D.writeTableAndAPI(OS, TargetName);
}
