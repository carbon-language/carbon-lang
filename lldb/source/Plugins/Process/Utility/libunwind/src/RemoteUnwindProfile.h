/* -*- mode: C++; c-basic-offset: 4; tab-width: 4 vi:set tabstop=4 expandtab: -*/
//===-- RemoteUnwindProfile.h -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef __UNWIND_PROFILE_H__
#define __UNWIND_PROFILE_H__
#if defined (SUPPORT_REMOTE_UNWINDING)

#include <vector>

// The architecture-independent profile of a function's prologue

namespace lldb_private 
{

class RemoteUnwindProfile {
public:
    RemoteUnwindProfile () : fRegistersSaved(32, 0), fRegSizes(10, 0) { }
    struct CFALocation {
        int regno;
        int offset;
    };
    enum RegisterSavedWhere { kRegisterOffsetFromCFA, kRegisterIsCFA };
    enum RegisterType { kGeneralPurposeRegister = 0, kFloatingPointRegister, kVectorRegister };
    struct SavedReg {
        int regno;
        RegisterSavedWhere location;
        int64_t value;
        int adj;    // Used in kRegisterInRegister e.g. when we recover the caller's rsp by
                    // taking the contents of rbp and subtracting 16.
        RegisterType type;
    };
    // In the following maps the key is the address after which this change has effect.
    //
    //  0  push %rbp
    //  1  mov  %rsp, %rbp
    //  2  sub  $16, %rsp
    //
    // At saved_registers<2> we'll find the record stating that rsp is now stored in rbp.

    std::map<uint64_t, CFALocation> cfa;
    std::map<uint64_t, std::vector<SavedReg> > saved_registers;

    struct CFALocation initial_cfa;  // At entry to the function

    std::vector<uint8_t> fRegistersSaved;
    std::vector<uint8_t> fRegSizes;
    SavedReg returnAddress;
    uint64_t fStart, fEnd;           // low and high pc values for this function.
                                     // END is the addr of the first insn outside the function.
    uint64_t fFirstInsnPastPrologue;
};

class RemoteProcInfo;

bool AssemblyParse (RemoteProcInfo *procinfo, unw_accessors_t *as, unw_addr_space_t as, uint64_t start, uint64_t end, RemoteUnwindProfile &profile, void *arg);


class FuncBounds {
    public:
    FuncBounds (uint64_t low, uint64_t high) : fStart(low), fEnd(high) { }
    uint64_t fStart;
    uint64_t fEnd;
};

inline bool operator<(const FuncBounds &ap1, const FuncBounds &ap2) {
    if (ap1.fStart < ap2.fStart)
        return true;
    if (ap1.fStart == ap2.fStart && ap1.fEnd < ap2.fEnd)
        return true;
    return false;
}


};
#endif


#endif // __UNWIND_PROFILE_H__
