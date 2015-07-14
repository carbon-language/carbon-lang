//===-- FreeBSDSignals.cpp --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "FreeBSDSignals.h"

using namespace lldb_private;

FreeBSDSignals::FreeBSDSignals()
    : UnixSignals()
{
    Reset();
}

void
FreeBSDSignals::Reset()
{
    UnixSignals::Reset();

    //        SIGNO   NAME           SHORT NAME  SUPPRESS STOP   NOTIFY DESCRIPTION 
    //        ======  ============   ==========  ======== ====== ====== ===================================================
    AddSignal (32,    "SIGTHR",      "THR",      false,   true , true , "thread interrupt");
    AddSignal (33,    "SIGLIBRT",    "LIBRT",    false,   true , true , "reserved by real-time library");
    AddSignal (65,    "SIGRTMIN",    "RTMIN",    false,   true , true , "real time signal 0");
    AddSignal (66,    "SIGRTMIN+1",  "RTMIN+1",  false,   true , true , "real time signal 1");
    AddSignal (67,    "SIGRTMIN+2",  "RTMIN+2",  false,   true , true , "real time signal 2");
    AddSignal (68,    "SIGRTMIN+3",  "RTMIN+3",  false,   true , true , "real time signal 3");
    AddSignal (69,    "SIGRTMIN+4",  "RTMIN+4",  false,   true , true , "real time signal 4");
    AddSignal (70,    "SIGRTMIN+5",  "RTMIN+5",  false,   true , true , "real time signal 5");
    AddSignal (71,    "SIGRTMIN+6",  "RTMIN+6",  false,   true , true , "real time signal 6");
    AddSignal (72,    "SIGRTMIN+7",  "RTMIN+7",  false,   true , true , "real time signal 7");
    AddSignal (73,    "SIGRTMIN+8",  "RTMIN+8",  false,   true , true , "real time signal 8");
    AddSignal (74,    "SIGRTMIN+9",  "RTMIN+9",  false,   true , true , "real time signal 9");
    AddSignal (75,    "SIGRTMIN+10", "RTMIN+10", false,   true , true , "real time signal 10");
    AddSignal (76,    "SIGRTMIN+11", "RTMIN+11", false,   true , true , "real time signal 11");
    AddSignal (77,    "SIGRTMIN+12", "RTMIN+12", false,   true , true , "real time signal 12");
    AddSignal (78,    "SIGRTMIN+13", "RTMIN+13", false,   true , true , "real time signal 13");
    AddSignal (79,    "SIGRTMIN+14", "RTMIN+14", false,   true , true , "real time signal 14");
    AddSignal (80,    "SIGRTMIN+15", "RTMIN+15", false,   true , true , "real time signal 15");
    AddSignal (81,    "SIGRTMIN+16", "RTMIN+16", false,   true , true , "real time signal 16");
    AddSignal (82,    "SIGRTMIN+17", "RTMIN+17", false,   true , true , "real time signal 17");
    AddSignal (83,    "SIGRTMIN+18", "RTMIN+18", false,   true , true , "real time signal 18");
    AddSignal (84,    "SIGRTMIN+19", "RTMIN+19", false,   true , true , "real time signal 19");
    AddSignal (85,    "SIGRTMIN+20", "RTMIN+20", false,   true , true , "real time signal 20");
    AddSignal (86,    "SIGRTMIN+21", "RTMIN+21", false,   true , true , "real time signal 21");
    AddSignal (87,    "SIGRTMIN+22", "RTMIN+22", false,   true , true , "real time signal 22");
    AddSignal (88,    "SIGRTMIN+23", "RTMIN+23", false,   true , true , "real time signal 23");
    AddSignal (89,    "SIGRTMIN+24", "RTMIN+24", false,   true , true , "real time signal 24");
    AddSignal (90,    "SIGRTMIN+25", "RTMIN+25", false,   true , true , "real time signal 25");
    AddSignal (91,    "SIGRTMIN+26", "RTMIN+26", false,   true , true , "real time signal 26");
    AddSignal (92,    "SIGRTMIN+27", "RTMIN+27", false,   true , true , "real time signal 27");
    AddSignal (93,    "SIGRTMIN+28", "RTMIN+28", false,   true , true , "real time signal 28");
    AddSignal (94,    "SIGRTMIN+29", "RTMIN+29", false,   true , true , "real time signal 29");
    AddSignal (95,    "SIGRTMIN+30", "RTMIN+30", false,   true , true , "real time signal 30");
    AddSignal (96,    "SIGRTMAX-30", "RTMAX-30", false,   true , true , "real time signal 31");
    AddSignal (97,    "SIGRTMAX-29", "RTMAX-29", false,   true , true , "real time signal 32");
    AddSignal (98,    "SIGRTMAX-28", "RTMAX-28", false,   true , true , "real time signal 33");
    AddSignal (99,    "SIGRTMAX-27", "RTMAX-27", false,   true , true , "real time signal 34");
    AddSignal (100,   "SIGRTMAX-26", "RTMAX-26", false,   true , true , "real time signal 35");
    AddSignal (101,   "SIGRTMAX-25", "RTMAX-25", false,   true , true , "real time signal 36");
    AddSignal (102,   "SIGRTMAX-24", "RTMAX-24", false,   true , true , "real time signal 37");
    AddSignal (103,   "SIGRTMAX-23", "RTMAX-23", false,   true , true , "real time signal 38");
    AddSignal (104,   "SIGRTMAX-22", "RTMAX-22", false,   true , true , "real time signal 39");
    AddSignal (105,   "SIGRTMAX-21", "RTMAX-21", false,   true , true , "real time signal 40");
    AddSignal (106,   "SIGRTMAX-20", "RTMAX-20", false,   true , true , "real time signal 41");
    AddSignal (107,   "SIGRTMAX-19", "RTMAX-19", false,   true , true , "real time signal 42");
    AddSignal (108,   "SIGRTMAX-18", "RTMAX-18", false,   true , true , "real time signal 43");
    AddSignal (109,   "SIGRTMAX-17", "RTMAX-17", false,   true , true , "real time signal 44");
    AddSignal (110,   "SIGRTMAX-16", "RTMAX-16", false,   true , true , "real time signal 45");
    AddSignal (111,   "SIGRTMAX-15", "RTMAX-15", false,   true , true , "real time signal 46");
    AddSignal (112,   "SIGRTMAX-14", "RTMAX-14", false,   true , true , "real time signal 47");
    AddSignal (113,   "SIGRTMAX-13", "RTMAX-13", false,   true , true , "real time signal 48");
    AddSignal (114,   "SIGRTMAX-12", "RTMAX-12", false,   true , true , "real time signal 49");
    AddSignal (115,   "SIGRTMAX-11", "RTMAX-11", false,   true , true , "real time signal 50");
    AddSignal (116,   "SIGRTMAX-10", "RTMAX-10", false,   true , true , "real time signal 51");
    AddSignal (117,   "SIGRTMAX-9",  "RTMAX-9",  false,   true , true , "real time signal 52");
    AddSignal (118,   "SIGRTMAX-8",  "RTMAX-8",  false,   true , true , "real time signal 53");
    AddSignal (119,   "SIGRTMAX-7",  "RTMAX-7",  false,   true , true , "real time signal 54");
    AddSignal (120,   "SIGRTMAX-6",  "RTMAX-6",  false,   true , true , "real time signal 55");
    AddSignal (121,   "SIGRTMAX-5",  "RTMAX-5",  false,   true , true , "real time signal 56");
    AddSignal (122,   "SIGRTMAX-4",  "RTMAX-4",  false,   true , true , "real time signal 57");
    AddSignal (123,   "SIGRTMAX-3",  "RTMAX-3",  false,   true , true , "real time signal 58");
    AddSignal (124,   "SIGRTMAX-2",  "RTMAX-2",  false,   true , true , "real time signal 59");
    AddSignal (125,   "SIGRTMAX-1",  "RTMAX-1",  false,   true , true , "real time signal 60");
    AddSignal (126,   "SIGRTMAX",    "RTMAX",    false,   true , true , "real time signal 61");
}
