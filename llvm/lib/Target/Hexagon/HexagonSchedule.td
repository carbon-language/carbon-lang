//===- HexagonSchedule.td - Hexagon Scheduling Definitions -*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// Functional Units
def LSUNIT    : FuncUnit; // SLOT0
def LUNIT     : FuncUnit; // SLOT1
def MUNIT     : FuncUnit; // SLOT2
def SUNIT     : FuncUnit; // SLOT3
def LOOPUNIT  : FuncUnit;

// Itinerary classes
def ALU32     : InstrItinClass;
def ALU64     : InstrItinClass;
def CR        : InstrItinClass;
def J         : InstrItinClass;
def JR        : InstrItinClass;
def LD        : InstrItinClass;
def LD0       : InstrItinClass;
def M         : InstrItinClass;
def ST        : InstrItinClass;
def ST0       : InstrItinClass;
def S         : InstrItinClass;
def SYS       : InstrItinClass;
def ENDLOOP   : InstrItinClass;
def PSEUDO    : InstrItinClass;
def PSEUDOM   : InstrItinClass;

def HexagonItineraries :
      ProcessorItineraries<[LSUNIT, LUNIT, MUNIT, SUNIT, LOOPUNIT], [], [
        InstrItinData<ALU32  , [InstrStage<1, [LUNIT, LSUNIT, MUNIT, SUNIT]>]>,
        InstrItinData<ALU64  , [InstrStage<1, [MUNIT, SUNIT]>]>,
        InstrItinData<CR     , [InstrStage<1, [SUNIT]>]>,
        InstrItinData<J      , [InstrStage<1, [SUNIT, MUNIT]>]>,
        InstrItinData<JR     , [InstrStage<1, [MUNIT]>]>,
        InstrItinData<LD     , [InstrStage<1, [LUNIT, LSUNIT]>]>,
        InstrItinData<LD0    , [InstrStage<1, [LSUNIT]>]>,
        InstrItinData<M      , [InstrStage<1, [MUNIT, SUNIT]>]>,
        InstrItinData<ST     , [InstrStage<1, [LSUNIT]>]>,
        InstrItinData<ST0    , [InstrStage<1, [LSUNIT]>]>,
        InstrItinData<S      , [InstrStage<1, [SUNIT, MUNIT]>]>,
        InstrItinData<SYS    , [InstrStage<1, [LSUNIT]>]>,
        InstrItinData<ENDLOOP, [InstrStage<1, [LOOPUNIT]>]>,
        InstrItinData<PSEUDO , [InstrStage<1, [LUNIT, LSUNIT, MUNIT, SUNIT]>]>,
        InstrItinData<PSEUDOM, [InstrStage<1, [MUNIT, SUNIT], 0>,
                                InstrStage<1, [MUNIT, SUNIT]>]>
      ]>;

def HexagonModel : SchedMachineModel {
  // Max issue per cycle == bundle width.
  let IssueWidth = 4;
  let Itineraries = HexagonItineraries;
  let LoadLatency = 1;
}

//===----------------------------------------------------------------------===//
// V4 Machine Info +
//===----------------------------------------------------------------------===//

include "HexagonScheduleV4.td"

//===----------------------------------------------------------------------===//
// V4 Machine Info -
//===----------------------------------------------------------------------===//
