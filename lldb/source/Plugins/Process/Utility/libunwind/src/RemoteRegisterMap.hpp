/* -*- mode: C++; c-basic-offset: 4; tab-width: 4 vi:set tabstop=4 expandtab: -*/
//===-- RemoteRegisterMap.hpp -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// Provide conversions between reigster names, the libunwind internal enums, 
// and the register numbers the program calling libunwind are using.

#ifndef __REMOTE_REGISTER_MAP_HPP__
#define __REMOTE_REGISTER_MAP_HPP__

#if defined (SUPPORT_REMOTE_UNWINDING)

#ifndef __STDC_LIMIT_MACROS
#define __STDC_LIMIT_MACROS
#endif
#ifndef __STDC_CONSTANT_MACROS
#define __STDC_CONSTANT_MACROS
#endif

#include "libunwind.h"
#include <vector>

namespace lldb_private
{
class RemoteRegisterMap {
public:
    RemoteRegisterMap (unw_accessors_t *accessors, unw_targettype_t target);
    ~RemoteRegisterMap ();
    void initialize_x86_64 ();
    void initialize_i386 ();
    bool name_to_caller_regno (const char *name, int& callerr);
    bool name_to_unwind_regno (const char *name, int& unwindr);
    bool unwind_regno_to_caller_regno (int unwindr, int& callerr);
    bool nonvolatile_reg_p (int unwind_regno);
    bool argument_regnum_p (int unwind_regno);
    const char *ip_register_name();
    const char *sp_register_name();
    int caller_regno_for_ip ();
    int caller_regno_for_sp ();
    int unwind_regno_for_frame_pointer ();
    int unwind_regno_for_stack_pointer ();
    int wordsize ()                         { return fWordSize; }
    void scan_caller_regs (unw_addr_space_t as, void *arg);

    bool unwind_regno_to_machine_regno (int unwindr, int& machiner);
    bool machine_regno_to_unwind_regno (int machr, int& unwindr);
    bool caller_regno_to_unwind_regno (int callerr, int& unwindr);
    const char* unwind_regno_to_name (int unwindr);
    int byte_size_for_regtype (unw_regtype_t type);

private:

    // A structure that collects everything we need to know about a
    // given register in one place.
    struct reg {
        int unwind_regno;    // What libunwind-remote uses internally
        int caller_regno;    // What the libunwind-remote driver program uses
        int eh_frame_regno;  // What the eh_frame section uses
        int machine_regno;   // What the actual bits/bytes are in instructions
        char *name;
        unw_regtype_t type;
        reg () : unwind_regno(-1), machine_regno(-1), caller_regno(-1), 
                 eh_frame_regno(-1), name(NULL), type(UNW_NOT_A_REG) { }
    };

    unw_accessors_t fAccessors;
    unw_targettype_t fTarget;
    std::vector<RemoteRegisterMap::reg> fRegMap;
    int fWordSize;
};

void RemoteRegisterMap::initialize_x86_64 () {
#define DEFREG(ureg, ehno, machno, regn) {RemoteRegisterMap::reg r; r.unwind_regno = ureg; r.name = regn; r.eh_frame_regno = ehno; r.machine_regno = machno; r.type = UNW_INTEGER_REG; fRegMap.push_back(r); }
    DEFREG (UNW_X86_64_RAX, 0,  0,  strdup ("rax"));
    DEFREG (UNW_X86_64_RDX, 1,  2,  strdup ("rdx"));
    DEFREG (UNW_X86_64_RCX, 2,  1,  strdup ("rcx"));
    DEFREG (UNW_X86_64_RBX, 3,  3,  strdup ("rbx"));
    DEFREG (UNW_X86_64_RSI, 4,  6,  strdup ("rsi"));
    DEFREG (UNW_X86_64_RDI, 5,  7,  strdup ("rdi"));
    DEFREG (UNW_X86_64_RBP, 6,  5,  strdup ("rbp"));
    DEFREG (UNW_X86_64_RSP, 7,  4,  strdup ("rsp"));
    DEFREG (UNW_X86_64_R8,  8,  8,  strdup ("r8"));
    DEFREG (UNW_X86_64_R9,  9,  9,  strdup ("r9"));
    DEFREG (UNW_X86_64_R10, 10, 10, strdup ("r10"));
    DEFREG (UNW_X86_64_R11, 11, 11, strdup ("r11"));
    DEFREG (UNW_X86_64_R12, 12, 12, strdup ("r12"));
    DEFREG (UNW_X86_64_R13, 13, 13, strdup ("r13"));
    DEFREG (UNW_X86_64_R14, 14, 14, strdup ("r14"));
    DEFREG (UNW_X86_64_R15, 15, 15, strdup ("r15"));
#undef DEFREG
    RemoteRegisterMap::reg r;
    r.name = strdup ("rip");
    r.type = UNW_INTEGER_REG; 
    r.eh_frame_regno = 16;
    fRegMap.push_back(r);
}

void RemoteRegisterMap::initialize_i386 () {
#define DEFREG(ureg, ehno, machno, regn) {RemoteRegisterMap::reg r; r.unwind_regno = ureg; r.name = regn; r.eh_frame_regno = ehno; r.machine_regno = machno; r.type = UNW_INTEGER_REG; fRegMap.push_back(r); }
    DEFREG (UNW_X86_EAX, 0,  0,  strdup ("eax"));
    DEFREG (UNW_X86_ECX, 1,  1,  strdup ("ecx"));
    DEFREG (UNW_X86_EDX, 2,  2,  strdup ("edx"));
    DEFREG (UNW_X86_EBX, 3,  3,  strdup ("ebx"));
    // i386 EH frame info has the next two swapped,
    // v. gcc/config/i386/darwin.h:DWARF2_FRAME_REG_OUT.
    DEFREG (UNW_X86_EBP, 4,  5,  strdup ("ebp"));
    DEFREG (UNW_X86_ESP, 5,  4,  strdup ("esp"));
    DEFREG (UNW_X86_ESI, 6,  6,  strdup ("esi"));
    DEFREG (UNW_X86_EDI, 7,  7,  strdup ("edi"));
#undef DEFREG
    RemoteRegisterMap::reg r;
    r.name = strdup ("eip");
    r.type = UNW_INTEGER_REG; 
    r.eh_frame_regno = 8;
    fRegMap.push_back(r);
}


RemoteRegisterMap::RemoteRegisterMap (unw_accessors_t *accessors, unw_targettype_t target) {
    fAccessors = *accessors;
    fTarget = target;
    switch (target) {
        case UNW_TARGET_X86_64:
            this->initialize_x86_64();
            fWordSize = 8;
            break;
        case UNW_TARGET_I386:
            this->initialize_i386();
            fWordSize = 4;
            break;
        default:
            ABORT("RemoteRegisterMap called with unknown target");
    }
}

RemoteRegisterMap::~RemoteRegisterMap () {
    std::vector<RemoteRegisterMap::reg>::iterator j;
    for (j = fRegMap.begin(); j != fRegMap.end(); ++j)
        free (j->name);
}

bool RemoteRegisterMap::name_to_caller_regno (const char *name, int& callerr) {
    if (name == NULL)
        return false;
    for (std::vector<RemoteRegisterMap::reg>::iterator j = fRegMap.begin(); j != fRegMap.end(); ++j)
        if (strcasecmp (j->name, name) == 0) {
            callerr = j->caller_regno;
            return true;
        }
    return false;
}

bool RemoteRegisterMap::unwind_regno_to_caller_regno (int unwindr, int& callerr) {
    if (unwindr == UNW_REG_IP) {
      callerr = this->caller_regno_for_ip ();
      return true;
    }
    if (unwindr == UNW_REG_SP) {
      callerr = this->caller_regno_for_sp ();
      return true;
    }
    for (std::vector<RemoteRegisterMap::reg>::iterator j = fRegMap.begin(); j != fRegMap.end(); ++j)
        if (j->unwind_regno == unwindr && j->caller_regno != -1) {
            callerr = j->caller_regno;
            return true;
        }
    return false;
}

bool RemoteRegisterMap::nonvolatile_reg_p (int unwind_regno) {
    if (fTarget == UNW_TARGET_X86_64) {
        switch (unwind_regno) {
            case UNW_X86_64_RBX:
            case UNW_X86_64_RSP:
            case UNW_X86_64_RBP:  // not actually a nonvolatile but often treated as such by convention
            case UNW_X86_64_R12:
            case UNW_X86_64_R13:
            case UNW_X86_64_R14:
            case UNW_X86_64_R15:
            case UNW_REG_IP:
            case UNW_REG_SP:
                return true;
                break;
            default:
                return false;
        }
    }
    if (fTarget == UNW_TARGET_I386) {
        switch (unwind_regno) {
            case UNW_X86_EBX:
            case UNW_X86_EBP:  // not actually a nonvolatile but often treated as such by convention
            case UNW_X86_ESI:
            case UNW_X86_EDI:
            case UNW_X86_ESP:
            case UNW_REG_IP:
            case UNW_REG_SP:
                return true;
                break;
            default:
                return false;
        }
    }
    return false;
}


bool RemoteRegisterMap::argument_regnum_p (int unwind_regno) {
    if (fTarget == UNW_TARGET_X86_64) {
        switch (unwind_regno) {
            case UNW_X86_64_RDI: /* arg 1 */
            case UNW_X86_64_RSI: /* arg 2 */
            case UNW_X86_64_RDX: /* arg 3 */
            case UNW_X86_64_RCX: /* arg 4 */
            case UNW_X86_64_R8:  /* arg 5 */
            case UNW_X86_64_R9:  /* arg 6 */
                return true;
                break;
            default:
                return false;
        }
    }
    return false;
}

const char *RemoteRegisterMap::ip_register_name () {
    switch (fTarget) {
        case UNW_TARGET_X86_64:
            return "rip";
        case UNW_TARGET_I386:
            return "eip";
        default:
            ABORT("unsupported architecture");
    }
    return NULL;
}

const char *RemoteRegisterMap::sp_register_name () {
    switch (fTarget) {
        case UNW_TARGET_X86_64:
            return "rsp";
        case UNW_TARGET_I386:
            return "esp";
        default:
            ABORT("unsupported architecture");
    }
    return NULL;
}

int RemoteRegisterMap::caller_regno_for_ip () {
    int callerr;
    if (this->name_to_caller_regno (this->ip_register_name(), callerr))
        return callerr;
    return -1;
}

int RemoteRegisterMap::caller_regno_for_sp () {
    int callerr;
    if (this->name_to_caller_regno (this->sp_register_name(), callerr))
        return callerr;
    return -1;
}

int RemoteRegisterMap::unwind_regno_for_frame_pointer () {
    switch (fTarget) {
        case UNW_TARGET_X86_64:
            return UNW_X86_64_RBP;
        case UNW_TARGET_I386:
            return UNW_X86_EBP;
        default:
            ABORT("cannot be reached");
    }
    return -1;
}

int RemoteRegisterMap::unwind_regno_for_stack_pointer () {
    switch (fTarget) {
        case UNW_TARGET_X86_64:
            return UNW_X86_64_RSP;
        case UNW_TARGET_I386:
            return UNW_X86_ESP;
        default:
            ABORT("cannot be reached");
    }
    return -1;
}

// This call requires a "arg" which specifies a given process/thread to
// complete unlike the rest of the RegisterMap functions.  Ideally this
// would be in the ctor but the register map is created when an 
// AddressSpace is created and we don't have a process/thread yet.

void RemoteRegisterMap::scan_caller_regs (unw_addr_space_t as, void *arg) {
    for (int i = 0; i < 256; i++) {
        unw_regtype_t type;
        char namebuf[16];
        if (fAccessors.reg_info (as, i, &type, namebuf, sizeof (namebuf), arg) == UNW_ESUCCESS
            && type != UNW_NOT_A_REG) {
            std::vector<RemoteRegisterMap::reg>::iterator j;
            for (j = fRegMap.begin(); j != fRegMap.end(); ++j) {
                if (strcasecmp (j->name, namebuf) == 0) {
                    j->caller_regno = i;
                    // if we haven't picked up a reg type yet it will be UNW_NOT_A_REG via the ctor
                    if (j->type == UNW_NOT_A_REG)
                        j->type = type;
                    if (j->type != type) {
                        ABORT("Caller and libunwind disagree about type of register");
                    break;
                    }
                }
            }
            // caller knows about a register we don't have a libunwind entry for
            if (j == fRegMap.end()) {
                RemoteRegisterMap::reg r;
                r.name = strdup (namebuf);
                r.caller_regno = i;
                r.type = type;
                fRegMap.push_back(r);
            }
        }
    }
}


bool RemoteRegisterMap::name_to_unwind_regno (const char *name, int& unwindr) {
    if (name == NULL)
        return false;
    for (std::vector<RemoteRegisterMap::reg>::iterator j = fRegMap.begin(); j != fRegMap.end(); ++j)
        if (strcasecmp (j->name, name) == 0) {
            unwindr = j->unwind_regno;
            return true;
        }
    return false;
}

bool RemoteRegisterMap::unwind_regno_to_machine_regno (int unwindr, int& machiner) {
    if (unwindr == UNW_REG_IP)
      unwindr = this->caller_regno_for_ip ();
    if (unwindr == UNW_REG_SP)
      unwindr = this->caller_regno_for_sp ();
    for (std::vector<RemoteRegisterMap::reg>::iterator j = fRegMap.begin(); j != fRegMap.end(); ++j)
        if (j->unwind_regno == unwindr && j->machine_regno != -1) {
            machiner = j->machine_regno;
            return true;
        }
    return false;
}
bool RemoteRegisterMap::machine_regno_to_unwind_regno (int machr, int& unwindr) {
    for (std::vector<RemoteRegisterMap::reg>::iterator j = fRegMap.begin(); j != fRegMap.end(); ++j)
        if (j->machine_regno == machr && j->unwind_regno != -1) {
            unwindr = j->unwind_regno;
            return true;
        }
    return false;
}
bool RemoteRegisterMap::caller_regno_to_unwind_regno (int callerr, int& unwindr) {
    if (this->caller_regno_for_ip() == callerr) {
        unwindr = UNW_REG_IP;
        return true;
    }
    if (this->caller_regno_for_sp() == callerr) {
        unwindr = UNW_REG_SP;
        return true;
    }
    for (std::vector<RemoteRegisterMap::reg>::iterator j = fRegMap.begin(); j != fRegMap.end(); ++j)
        if (j->caller_regno == callerr && j->unwind_regno != -1) {
            unwindr = j->unwind_regno;
            return true;
        }
    return false;
}

const char* RemoteRegisterMap::unwind_regno_to_name (int unwindr) {
    for (std::vector<RemoteRegisterMap::reg>::iterator j = fRegMap.begin(); j != fRegMap.end(); ++j)
        if (j->unwind_regno == unwindr && j->name != NULL) {
            return j->name;
        }
    return NULL;
}

int RemoteRegisterMap::byte_size_for_regtype (unw_regtype_t type) {
    switch (type) {
        case UNW_TARGET_X86_64:
        case UNW_TARGET_I386:
            if (type == UNW_INTEGER_REG)        return fWordSize;
            if (type == UNW_FLOATING_POINT_REG) return 8;
            if (type == UNW_VECTOR_REG)         return 16;
        default:
            ABORT("cannot be reached");
    }
    return -1;
}


}; // namespace lldb_private

#endif // SUPPORT_REMOTE_UNWINDING

#endif // __REMOTE_REGISTER_MAP_HPP__

