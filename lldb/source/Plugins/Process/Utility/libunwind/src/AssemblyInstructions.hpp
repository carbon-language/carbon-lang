/* -*- mode: C++; c-basic-offset: 4; tab-width: 4 vi:set tabstop=4 expandtab: -*/
//===-- AssemblyInstructions.hpp --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef __ASSEMBLY_INSTRUCTIONS_HPP
#define __ASSEMBLY_INSTRUCTIONS_HPP

#if defined (SUPPORT_REMOTE_UNWINDING)

#ifndef __STDC_LIMIT_MACROS
#define __STDC_LIMIT_MACROS
#endif
#ifndef __STDC_CONSTANT_MACROS
#define __STDC_CONSTANT_MACROS
#endif

#include <limits.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <map>

#include "libunwind.h"
#include "AssemblyParser.hpp"
#include "AddressSpace.hpp"
#include "Registers.hpp"
#include "RemoteUnwindProfile.h"

namespace lldb_private
{

// A debug function to dump the contents of an RemoteUnwindProfile to
// stdout in a human readable form.

template <typename A, typename R>
void printProfile (A& addressSpace, uint64_t pc, RemoteUnwindProfile* profile, R& registers) {
    RemoteProcInfo *procinfo = addressSpace.getRemoteProcInfo();
    RemoteRegisterMap *regmap = procinfo->getRegisterMap();

    procinfo->logDebug ("Print profile: given pc of 0x%llx, profile has range 0x%llx - 0x%llx", pc, profile->fStart, profile->fEnd);
    procinfo->logDebug ("CFA locations:");
    std::map<uint64_t, RemoteUnwindProfile::CFALocation>::iterator i;
    for (i = profile->cfa.begin(); i != profile->cfa.end(); ++i) {
        procinfo->logDebug ("   as of 0x%llx cfa is based off of reg %d (%s) offset %d", i->first, i->second.regno, regmap->unwind_regno_to_name(i->second.regno), i->second.offset);
    }
    procinfo->logDebug ("Caller's saved IP is at %d bytes offset from the cfa", (int)profile->returnAddress.value);
    procinfo->logDebug ("Register saves:");
    std::map<uint64_t, std::vector<RemoteUnwindProfile::SavedReg> >::iterator j;
    for (j = profile->saved_registers.begin(); j != profile->saved_registers.end(); ++j) {
        char *tbuf1, *tbuf2, *tbuf3;
        asprintf (&tbuf1, "  at pc 0x%llx there are %d registers saved ", j->first, (int) j->second.size());
        std::vector<RemoteUnwindProfile::SavedReg>::iterator k;
        for (k = j->second.begin(); k != j->second.end(); ++k) {
            if (k->location == RemoteUnwindProfile::kRegisterOffsetFromCFA) {
                asprintf (&tbuf2, "[reg %d (%s) is %d bytes from cfa] ", k->regno, regmap->unwind_regno_to_name(k->regno), (int) k->value);
                int newlen = strlen (tbuf1) + strlen (tbuf2) + 1;
                tbuf3 = (char *) malloc (newlen);
                strcpy (tbuf3, tbuf1);
                strcat (tbuf3, tbuf2);
                free (tbuf1);
                free (tbuf2);
                tbuf1 = tbuf3;
            }
            if (k->location == RemoteUnwindProfile::kRegisterIsCFA) {
                asprintf (&tbuf2, "[reg %d (%s) is the same as the cfa] ", k->regno, regmap->unwind_regno_to_name(k->regno));
                int newlen = strlen (tbuf1) + strlen (tbuf2) + 1;
                tbuf3 = (char *) malloc (newlen);
                strcpy (tbuf3, tbuf1);
                strcat (tbuf3, tbuf2);
                free (tbuf1);
                free (tbuf2);
                tbuf1 = tbuf3;
            }
        }
        procinfo->logDebug ("%s", tbuf1);
        free (tbuf1);
    }
}

template <typename A, typename R>
int stepWithAssembly (A& addressSpace, uint64_t pc, RemoteUnwindProfile* profile, R& registers) {
    R newRegisters(registers);
    RemoteProcInfo *procinfo = addressSpace.getRemoteProcInfo();
    if (pc > profile->fEnd)
        ABORT("stepWithAssembly called with pc not in RemoteUnwindProfile's bounds");

    if (procinfo && (procinfo->getDebugLoggingLevel() & UNW_LOG_LEVEL_DEBUG))
        printProfile (addressSpace, pc, profile, registers);

    std::map<uint64_t, RemoteUnwindProfile::CFALocation>::iterator i = profile->cfa.lower_bound (pc);
    if (i == profile->cfa.begin() && i == profile->cfa.end())
        return UNW_EINVAL;
    if (i == profile->cfa.end()) {
        --i;
    } else {
        if (i != profile->cfa.begin() && i->first != pc)
          --i;
    }

    uint64_t cfa = registers.getRegister (i->second.regno) + i->second.offset;
    
    std::map<uint64_t, std::vector<RemoteUnwindProfile::SavedReg> >::iterator j;

    for (j = profile->saved_registers.begin(); j != profile->saved_registers.end() && j->first <= pc; ++j) {
        std::vector<RemoteUnwindProfile::SavedReg>::iterator k = j->second.begin();
        for (; k != j->second.end(); ++k) {
            RemoteUnwindProfile::SavedReg sr = *k;
            if (sr.type == RemoteUnwindProfile::kGeneralPurposeRegister) {
                uint64_t result;
                int err = 0;
                switch (sr.location) {
                    case RemoteUnwindProfile::kRegisterOffsetFromCFA: 
                        result = addressSpace.getP(cfa + sr.value, err);
                        break;
                    case RemoteUnwindProfile::kRegisterIsCFA:
                            result = cfa;
                        break;
                    default:
                        ABORT("Unknown saved register location in stepWithAssembly.");
                }
                // If we failed to read remote memory, stop unwinding.
                if (err)
                    return UNW_STEP_END;
                newRegisters.setRegister (sr.regno, result);
            }
        }
    }
    newRegisters.setSP(cfa);
    uint64_t ip = addressSpace.getP(cfa + profile->returnAddress.value);
    if (ip == 0) 
      return UNW_STEP_END;
    newRegisters.setIP(ip);
    registers = newRegisters;
    return UNW_STEP_SUCCESS;
}

}; // namespace lldb_private

#endif // SUPPORT_REMOTE_UNWINDING
#endif  //ASSEMBLY_INSTRUCTIONS_HPP
