//===- FuzzerDFSan.cpp - DFSan-based fuzzer mutator -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// DataFlowSanitizer (DFSan) is a tool for
// generalised dynamic data flow (taint) analysis:
// http://clang.llvm.org/docs/DataFlowSanitizer.html .
//
// This file implements a mutation algorithm based on taint
// analysis feedback from DFSan.
//
// The approach has some similarity to "Taint-based Directed Whitebox Fuzzing"
// by Vijay Ganesh & Tim Leek & Martin Rinard:
// http://dspace.mit.edu/openaccess-disseminate/1721.1/59320,
// but it uses a full blown LLVM IR taint analysis and separate instrumentation
// to analyze all of the "attack points" at once.
//
// Workflow:
//   * lib/Fuzzer/Fuzzer*.cpp is compiled w/o any instrumentation.
//   * The code under test is compiled with DFSan *and* with special extra hooks
//     that are inserted before dfsan. Currently supported hooks:
//     - __sanitizer_cov_trace_cmp: inserted before every ICMP instruction,
//       receives the type, size and arguments of ICMP.
//   * Every call to HOOK(a,b) is replaced by DFSan with
//     __dfsw_HOOK(a, b, label(a), label(b)) so that __dfsw_HOOK
//     gets all the taint labels for the arguments.
//   * At the Fuzzer startup we assign a unique DFSan label
//     to every byte of the input string (Fuzzer::CurrentUnit) so that for any
//     chunk of data we know which input bytes it has derived from.
//   * The __dfsw_* functions (implemented in this file) record the
//     parameters (i.e. the application data and the corresponding taint labels)
//     in a global state.
//   * Fuzzer::MutateWithDFSan() tries to use the data recorded by __dfsw_*
//     hooks to guide the fuzzing towards new application states.
//     For example if 4 bytes of data that derive from input bytes {4,5,6,7}
//     are compared with a constant 12345 and the comparison always yields
//     the same result, we try to insert 12345, 12344, 12346 into bytes
//     {4,5,6,7} of the next fuzzed inputs.
//
// This code does not function when DFSan is not linked in.
// Instead of using ifdefs and thus requiring a separate build of lib/Fuzzer
// we redeclare the dfsan_* interface functions as weak and check if they
// are nullptr before calling.
// If this approach proves to be useful we may add attribute(weak) to the
// dfsan declarations in dfsan_interface.h
//
// This module is in the "proof of concept" stage.
// It is capable of solving only the simplest puzzles
// like test/dfsan/DFSanSimpleCmpTest.cpp.
//===----------------------------------------------------------------------===//

/* Example of manual usage:
(
  cd $LLVM/lib/Fuzzer/
  clang  -fPIC -c -g -O2 -std=c++11 Fuzzer*.cpp
  clang++ -O0 -std=c++11 -fsanitize-coverage=3  \
    -mllvm -sanitizer-coverage-experimental-trace-compares=1 \
    -fsanitize=dataflow -fsanitize-blacklist=./dfsan_fuzzer_abi.list  \
    test/dfsan/DFSanSimpleCmpTest.cpp Fuzzer*.o
  ./a.out
)
*/

#include "FuzzerInternal.h"
#include <sanitizer/dfsan_interface.h>

#include <cstring>
#include <iostream>
#include <unordered_map>

extern "C" {
__attribute__((weak))
dfsan_label dfsan_create_label(const char *desc, void *userdata);
__attribute__((weak))
void dfsan_set_label(dfsan_label label, void *addr, size_t size);
__attribute__((weak))
void dfsan_add_label(dfsan_label label, void *addr, size_t size);
__attribute__((weak))
const struct dfsan_label_info *dfsan_get_label_info(dfsan_label label);
}  // extern "C"

namespace {

// These values are copied from include/llvm/IR/InstrTypes.h.
// We do not include the LLVM headers here to remain independent.
// If these values ever change, an assertion in ComputeCmp will fail.
enum Predicate {
  ICMP_EQ = 32,  ///< equal
  ICMP_NE = 33,  ///< not equal
  ICMP_UGT = 34, ///< unsigned greater than
  ICMP_UGE = 35, ///< unsigned greater or equal
  ICMP_ULT = 36, ///< unsigned less than
  ICMP_ULE = 37, ///< unsigned less or equal
  ICMP_SGT = 38, ///< signed greater than
  ICMP_SGE = 39, ///< signed greater or equal
  ICMP_SLT = 40, ///< signed less than
  ICMP_SLE = 41, ///< signed less or equal
};

template <class U, class S>
bool ComputeCmp(size_t CmpType, U Arg1, U Arg2) {
  switch(CmpType) {
    case ICMP_EQ : return Arg1 == Arg2;
    case ICMP_NE : return Arg1 != Arg2;
    case ICMP_UGT: return Arg1 > Arg2;
    case ICMP_UGE: return Arg1 >= Arg2;
    case ICMP_ULT: return Arg1 < Arg2;
    case ICMP_ULE: return Arg1 <= Arg2;
    case ICMP_SGT: return (S)Arg1 > (S)Arg2;
    case ICMP_SGE: return (S)Arg1 >= (S)Arg2;
    case ICMP_SLT: return (S)Arg1 < (S)Arg2;
    case ICMP_SLE: return (S)Arg1 <= (S)Arg2;
    default: assert(0 && "unsupported CmpType");
  }
  return false;
}

static bool ComputeCmp(size_t CmpSize, size_t CmpType, uint64_t Arg1,
                       uint64_t Arg2) {
  if (CmpSize == 8) return ComputeCmp<uint64_t, int64_t>(CmpType, Arg1, Arg2);
  if (CmpSize == 4) return ComputeCmp<uint32_t, int32_t>(CmpType, Arg1, Arg2);
  if (CmpSize == 2) return ComputeCmp<uint16_t, int16_t>(CmpType, Arg1, Arg2);
  if (CmpSize == 1) return ComputeCmp<uint8_t, int8_t>(CmpType, Arg1, Arg2);
  assert(0 && "unsupported type size");
  return true;
}

// As a simplification we use the range of input bytes instead of a set of input
// bytes.
struct LabelRange {
  uint16_t Beg, End;  // Range is [Beg, End), thus Beg==End is an empty range.

  LabelRange(uint16_t Beg = 0, uint16_t End = 0) : Beg(Beg), End(End) {}

  static LabelRange Join(LabelRange LR1, LabelRange LR2) {
    if (LR1.Beg == LR1.End) return LR2;
    if (LR2.Beg == LR2.End) return LR1;
    return {std::min(LR1.Beg, LR2.Beg), std::max(LR1.End, LR2.End)};
  }
  LabelRange &Join(LabelRange LR) {
    return *this = Join(*this, LR);
  }
  static LabelRange Singleton(const dfsan_label_info *LI) {
    uint16_t Idx = (uint16_t)(uintptr_t)LI->userdata;
    assert(Idx > 0);
    return {(uint16_t)(Idx - 1), Idx};
  }
};

std::ostream &operator<<(std::ostream &os, const LabelRange &LR) {
  return os << "[" << LR.Beg << "," << LR.End << ")";
}

class DFSanState {
 public:
   DFSanState(const fuzzer::Fuzzer::FuzzingOptions &Options)
       : Options(Options) {}

  struct CmpSiteInfo {
    size_t ResCounters[2] = {0, 0};
    size_t CmpSize = 0;
    LabelRange LR;
    std::unordered_map<uint64_t, size_t> CountedConstants;
  };

  LabelRange GetLabelRange(dfsan_label L);
  void DFSanCmpCallback(uintptr_t PC, size_t CmpSize, size_t CmpType,
                        uint64_t Arg1, uint64_t Arg2, dfsan_label L1,
                        dfsan_label L2);
  bool Mutate(fuzzer::Unit *U);

 private:
  std::unordered_map<uintptr_t, CmpSiteInfo> PcToCmpSiteInfoMap;
  LabelRange LabelRanges[1 << (sizeof(dfsan_label) * 8)] = {};
  const fuzzer::Fuzzer::FuzzingOptions &Options;
};

LabelRange DFSanState::GetLabelRange(dfsan_label L) {
  LabelRange &LR = LabelRanges[L];
  if (LR.Beg < LR.End || L == 0)
    return LR;
  const dfsan_label_info *LI = dfsan_get_label_info(L);
  if (LI->l1 || LI->l2)
    return LR = LabelRange::Join(GetLabelRange(LI->l1), GetLabelRange(LI->l2));
  return LR = LabelRange::Singleton(LI);
}

void DFSanState::DFSanCmpCallback(uintptr_t PC, size_t CmpSize, size_t CmpType,
                                  uint64_t Arg1, uint64_t Arg2, dfsan_label L1,
                                  dfsan_label L2) {
  if (L1 == 0 && L2 == 0)
    return;  // Not actionable.
  if (L1 != 0 && L2 != 0)
    return;  // Probably still actionable.
  bool Res = ComputeCmp(CmpSize, CmpType, Arg1, Arg2);
  CmpSiteInfo &CSI = PcToCmpSiteInfoMap[PC];
  CSI.CmpSize = CmpSize;
  CSI.LR.Join(GetLabelRange(L1)).Join(GetLabelRange(L2));
  if (!L1) CSI.CountedConstants[Arg1]++;
  if (!L2) CSI.CountedConstants[Arg2]++;
  size_t Counter = CSI.ResCounters[Res]++;

  if (Options.Verbosity >= 2  &&
      (Counter & (Counter - 1)) == 0 &&
      CSI.ResCounters[!Res] == 0)
    std::cerr << "DFSAN:"
              << " PC " << std::hex << PC << std::dec
              << " S " << CmpSize
              << " T " << CmpType
              << " A1 " << Arg1 << " A2 " << Arg2 << " R " << Res
              << " L" << L1 << GetLabelRange(L1)
              << " L" << L2 << GetLabelRange(L2)
              << " LR " << CSI.LR
              << "\n";
}

bool DFSanState::Mutate(fuzzer::Unit *U) {
  for (auto &PCToCmp : PcToCmpSiteInfoMap) {
    auto &CSI = PCToCmp.second;
    if (CSI.ResCounters[0] * CSI.ResCounters[1] != 0) continue;
    if (CSI.ResCounters[0] + CSI.ResCounters[1] < 1000) continue;
    if (CSI.CountedConstants.size() != 1) continue;
    uintptr_t C = CSI.CountedConstants.begin()->first;
    if (U->size() >= CSI.CmpSize) {
      size_t RangeSize = CSI.LR.End - CSI.LR.Beg;
      size_t Idx = CSI.LR.Beg + rand() % RangeSize;
      if (Idx + CSI.CmpSize > U->size()) continue;
      C += rand() % 5 - 2;
      memcpy(U->data() + Idx, &C, CSI.CmpSize);
      return true;
    }
  }
  return false;
}

static DFSanState *DFSan;

}  // namespace

namespace fuzzer {

bool Fuzzer::MutateWithDFSan(Unit *U) {
  if (!&dfsan_create_label || !DFSan) return false;
  return DFSan->Mutate(U);
}

void Fuzzer::InitializeDFSan() {
  if (!&dfsan_create_label || !Options.UseDFSan) return;
  DFSan = new DFSanState(Options);
  CurrentUnit.resize(Options.MaxLen);
  for (size_t i = 0; i < static_cast<size_t>(Options.MaxLen); i++) {
    dfsan_label L = dfsan_create_label("input", (void*)(i + 1));
    // We assume that no one else has called dfsan_create_label before.
    assert(L == i + 1);
    dfsan_set_label(L, &CurrentUnit[i], 1);
  }
}

}  // namespace fuzzer

extern "C" {
void __dfsw___sanitizer_cov_trace_cmp(uint64_t SizeAndType, uint64_t Arg1,
                                      uint64_t Arg2, dfsan_label L0,
                                      dfsan_label L1, dfsan_label L2) {
  assert(L0 == 0);
  uintptr_t PC = reinterpret_cast<uintptr_t>(__builtin_return_address(0));
  uint64_t CmpSize = (SizeAndType >> 32) / 8;
  uint64_t Type = (SizeAndType << 32) >> 32;
  DFSan->DFSanCmpCallback(PC, CmpSize, Type, Arg1, Arg2, L1, L2);
}
}  // extern "C"
