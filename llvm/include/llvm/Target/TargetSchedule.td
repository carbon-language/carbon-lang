//===- TargetSchedule.td - Target Independent Scheduling ---*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the target-independent scheduling interfaces which should
// be implemented by each target which is using TableGen based scheduling.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Processor functional unit - These values represent the function units
// available across all chip sets for the target.  Eg., IntUnit, FPUnit, ...
// These may be independent values for each chip set or may be shared across
// all chip sets of the target.  Each functional unit is treated as a resource
// during scheduling and has an affect instruction order based on availability
// during a time interval.
//
class FuncUnit;

//===----------------------------------------------------------------------===//
// Pipeline bypass / forwarding - These values specifies the symbolic names of
// pipeline bypasses which can be used to forward results of instructions
// that are forwarded to uses.
class Bypass;
def NoBypass : Bypass;

class ReservationKind<bits<1> val> {
  int Value = val;
}

def Required : ReservationKind<0>;
def Reserved : ReservationKind<1>;

//===----------------------------------------------------------------------===//
// Instruction stage - These values represent a non-pipelined step in
// the execution of an instruction.  Cycles represents the number of
// discrete time slots needed to complete the stage.  Units represent
// the choice of functional units that can be used to complete the
// stage.  Eg. IntUnit1, IntUnit2. NextCycles indicates how many
// cycles should elapse from the start of this stage to the start of
// the next stage in the itinerary.  For example:
//
// A stage is specified in one of two ways:
//
//   InstrStage<1, [FU_x, FU_y]>     - TimeInc defaults to Cycles
//   InstrStage<1, [FU_x, FU_y], 0>  - TimeInc explicit
//

class InstrStage<int cycles, list<FuncUnit> units,
                 int timeinc = -1,
                 ReservationKind kind = Required> {
  int Cycles          = cycles;       // length of stage in machine cycles
  list<FuncUnit> Units = units;       // choice of functional units
  int TimeInc         = timeinc;      // cycles till start of next stage
  int Kind            = kind.Value;   // kind of FU reservation
}

//===----------------------------------------------------------------------===//
// Instruction itinerary - An itinerary represents a sequential series of steps
// required to complete an instruction.  Itineraries are represented as lists of
// instruction stages.
//

//===----------------------------------------------------------------------===//
// Instruction itinerary classes - These values represent 'named' instruction
// itinerary.  Using named itineraries simplifies managing groups of
// instructions across chip sets.  An instruction uses the same itinerary class
// across all chip sets.  Thus a new chip set can be added without modifying
// instruction information.
//
// NumMicroOps represents the number of micro-operations that each instruction
// in the class are decoded to. If the number is zero, then it means the
// instruction can decode into variable number of micro-ops and it must be
// determined dynamically.
//
class InstrItinClass<int ops = 1> {
  int NumMicroOps = ops;
}
def NoItinerary : InstrItinClass;

//===----------------------------------------------------------------------===//
// Instruction itinerary data - These values provide a runtime map of an
// instruction itinerary class (name) to its itinerary data.
//
// OperandCycles are optional "cycle counts". They specify the cycle after
// instruction issue the values which correspond to specific operand indices
// are defined or read. Bypasses are optional "pipeline forwarding pathes", if
// a def by an instruction is available on a specific bypass and the use can
// read from the same bypass, then the operand use latency is reduced by one.
//
//  InstrItinData<IIC_iLoad_i , [InstrStage<1, [A9_Pipe1]>,
//                               InstrStage<1, [A9_AGU]>],
//                              [3, 1], [A9_LdBypass]>,
//  InstrItinData<IIC_iMVNr   , [InstrStage<1, [A9_Pipe0, A9_Pipe1]>],
//                              [1, 1], [NoBypass, A9_LdBypass]>,
//
// In this example, the instruction of IIC_iLoadi reads its input on cycle 1
// (after issue) and the result of the load is available on cycle 3. The result
// is available via forwarding path A9_LdBypass. If it's used by the first
// source operand of instructions of IIC_iMVNr class, then the operand latency
// is reduced by 1.
class InstrItinData<InstrItinClass Class, list<InstrStage> stages,
                    list<int> operandcycles = [],
                    list<Bypass> bypasses = []> {
  InstrItinClass TheClass = Class;
  list<InstrStage> Stages = stages;
  list<int> OperandCycles = operandcycles;
  list<Bypass> Bypasses = bypasses;
}

//===----------------------------------------------------------------------===//
// Processor itineraries - These values represent the set of all itinerary
// classes for a given chip set.
//
// Set property values to -1 to use the default.
// See InstrItineraryProps for comments and defaults.
class ProcessorItineraries<list<FuncUnit> fu, list<Bypass> bp,
                           list<InstrItinData> iid> {
  int IssueWidth = -1; // Max instructions that may be scheduled per cycle.
  int MinLatency = -1; // Determines which instrucions are allowed in a group.
                       // (-1) inorder (0) ooo, (1): inorder +var latencies.
  int LoadLatency = -1; // Cycles for loads to access the cache.
  int HighLatency = -1; // Approximation of cycles for "high latency" ops.

  list<FuncUnit> FU = fu;
  list<Bypass> BP = bp;
  list<InstrItinData> IID = iid;
}

// NoItineraries - A marker that can be used by processors without schedule
// info. Subtargets using NoItineraries can bypass the scheduler's
// expensive HazardRecognizer because no reservation table is needed.
def NoItineraries : ProcessorItineraries<[], [], []>;

// Processor itineraries with non-unit issue width. This allows issue
// width to be explicity specified at the beginning of the itinerary.
class MultiIssueItineraries<int issuewidth, int minlatency,
                            int loadlatency, int highlatency,
                            list<FuncUnit> fu, list<Bypass> bp,
                            list<InstrItinData> iid>
  : ProcessorItineraries<fu, bp, iid> {
  let IssueWidth = issuewidth;
  let MinLatency = minlatency;
  let LoadLatency = loadlatency;
  let HighLatency = highlatency;
}
