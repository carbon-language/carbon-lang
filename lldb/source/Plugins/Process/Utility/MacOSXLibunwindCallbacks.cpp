//===-- MacOSXLibunwindCallbacks.cpp ----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_MacOSXLibunwindCallbacks_cpp_
#define liblldb_MacOSXLibunwindCallbacks_cpp_
#if defined(__cplusplus)

#include "lldb/Core/DataExtractor.h"
#include "lldb/Core/FileSpec.h"
#include "lldb/Core/DataBufferHeap.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Symbol/SymbolContext.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"

#include "lldb/lldb-enumerations.h"
#include "llvm-c/EnhancedDisassembly.h"

#include "libunwind/include/libunwind.h"

using namespace lldb;

namespace lldb_private {

/* Don't implement (libunwind does not use)
      find_proc_info
      put_unwind_info
      get_dyn_info_list_addr
      access_mem
      resume
*/
/*
  Should implement (not needed yet)
      access_fpreg
      access_vecreg
      proc_is_sigtramp
      proc_is_inferior_function_call
      access_reg_inf_func_call
*/
  
static int 
access_reg (lldb_private::unw_addr_space_t as, lldb_private::unw_regnum_t regnum, lldb_private::unw_word_t *valp, int write, void *arg)
{
    if (arg == 0)
        return -1;
    Thread *th = (Thread *) arg;
    /* FIXME Only support reading for now.  */
    if (write == 1)
        return -1;
    if (th->GetRegisterContext()->GetRegisterInfoAtIndex(regnum) == NULL)
        return -1;
    DataExtractor de;
    if (!th->GetRegisterContext()->ReadRegisterBytes (regnum, de))
        return -1;
    memcpy (valp, de.GetDataStart(), de.GetByteSize());
    return UNW_ESUCCESS;
}

static int 
get_proc_name (lldb_private::unw_addr_space_t as, lldb_private::unw_word_t ip, char *bufp, size_t buf_len, lldb_private::unw_word_t *offp, void *arg)
{
    if (arg == 0)
        return -1;
    Thread *thread = (Thread *) arg;
    Target &target = thread->GetProcess().GetTarget();
    Address addr;
    if (!target.GetSectionLoadList().ResolveLoadAddress(ip, addr))
        return -1;
    
    SymbolContext sc;
    if (!target.GetImages().ResolveSymbolContextForAddress (addr, eSymbolContextFunction, sc))
        return -1;
    if (!sc.symbol)
        return -1;
    strlcpy (bufp, sc.symbol->GetMangled().GetMangledName().AsCString(""), buf_len);
    if (offp)
        *offp = addr.GetLoadAddress(&target) - sc.symbol->GetValue().GetLoadAddress(&target);
    return UNW_ESUCCESS;
}

static int 
find_image_info (lldb_private::unw_addr_space_t as, lldb_private::unw_word_t load_addr, lldb_private::unw_word_t *mh, 
                 lldb_private::unw_word_t *text_start, lldb_private::unw_word_t *text_end, 
                 lldb_private::unw_word_t *eh_frame, lldb_private::unw_word_t *eh_frame_len, 
                 lldb_private::unw_word_t *compact_unwind_start, lldb_private::unw_word_t *compact_unwind_len, void *arg)
{
    if (arg == 0)
        return -1;
    Thread *thread = (Thread *) arg;
    Target &target = thread->GetProcess().GetTarget();
    Address addr;
    if (!target.GetSectionLoadList().ResolveLoadAddress(load_addr, addr))
        return -1;
    
    SymbolContext sc;
    if (!target.GetImages().ResolveSymbolContextForAddress (addr, eSymbolContextModule, sc))
        return -1;
    
    SectionList *sl = sc.module_sp->GetObjectFile()->GetSectionList();
    static ConstString g_segment_name_TEXT("__TEXT");
    SectionSP text_segment_sp(sl->FindSectionByName(g_segment_name_TEXT));
    if (!text_segment_sp)
        return -1;
    
    *mh = text_segment_sp->GetLoadBaseAddress (&target);
    *text_start = text_segment_sp->GetLoadBaseAddress (&target);
    *text_end = *text_start + text_segment_sp->GetByteSize();
    
    static ConstString g_section_name_eh_frame ("__eh_frame");
    SectionSP eh_frame_section_sp = text_segment_sp->GetChildren().FindSectionByName(g_section_name_eh_frame);
    if (eh_frame_section_sp.get()) {
        *eh_frame = eh_frame_section_sp->GetLoadBaseAddress (&target);
        *eh_frame_len = eh_frame_section_sp->GetByteSize();
    } else {
        *eh_frame = 0;
        *eh_frame_len = 0;
    }
    
    static ConstString g_section_name_unwind_info ("__unwind_info");
    SectionSP unwind_info_section_sp = text_segment_sp->GetChildren().FindSectionByName(g_section_name_unwind_info);
    if (unwind_info_section_sp.get()) {
        *compact_unwind_start = unwind_info_section_sp->GetLoadBaseAddress (&target);
        *compact_unwind_len = unwind_info_section_sp->GetByteSize();
    } else {
        *compact_unwind_start = 0;
        *compact_unwind_len = 0;
    }
    return UNW_ESUCCESS;
}

static int 
get_proc_bounds (lldb_private::unw_addr_space_t as, lldb_private::unw_word_t ip, lldb_private::unw_word_t *low, lldb_private::unw_word_t *high, void *arg)
{
    if (arg == 0)
        return -1;
    Thread *thread = (Thread *) arg;
    Target &target = thread->GetProcess().GetTarget();
    Address addr;
    if (!target.GetSectionLoadList().ResolveLoadAddress(ip, addr))
        return -1;
    SymbolContext sc;
    if (!target.GetImages().ResolveSymbolContextForAddress (addr, eSymbolContextFunction | eSymbolContextSymbol, sc))
        return -1;
    if (sc.function)
    {
        lldb::addr_t start, len;
        start = sc.function->GetAddressRange().GetBaseAddress().GetLoadAddress(&target);
        len = sc.function->GetAddressRange().GetByteSize();
        if (start == LLDB_INVALID_ADDRESS || len == LLDB_INVALID_ADDRESS)
            return -1;
        *low = start;
        *high = start + len;
        return UNW_ESUCCESS;
    }
    if (sc.symbol)
    {
        lldb::addr_t start, len;
        start = sc.symbol->GetAddressRangeRef().GetBaseAddress().GetLoadAddress(&target);
        len = sc.symbol->GetAddressRangeRef().GetByteSize();
        if (start == LLDB_INVALID_ADDRESS)
            return -1;
        *low = start;
        if (len != LLDB_INVALID_ADDRESS)
            *high = start + len;
        else
            *high = 0;
        return UNW_ESUCCESS;
    }
    return -1;
}

static int 
access_raw (lldb_private::unw_addr_space_t as, lldb_private::unw_word_t addr, lldb_private::unw_word_t extent, uint8_t *valp, int write, void *arg)
{
    if (arg == 0)
        return -1;
    Thread *th = (Thread *) arg;
    /* FIXME Only support reading for now.  */
    if (write == 1)
        return -1;
    
    Error error;
    if (th->GetProcess().ReadMemory (addr, valp, extent, error) != extent)
        return -1;
    return UNW_ESUCCESS;
}


static int 
reg_info 
(
    lldb_private::unw_addr_space_t as, 
    lldb_private::unw_regnum_t regnum, 
    lldb_private::unw_regtype_t *type, 
    char *buf, 
    size_t buflen, 
    void *arg
)
{
    if (arg == 0)
        return -1;
    Thread *th = (Thread *) arg;
    RegisterContext *regc = th->GetRegisterContext();
    if (regnum > regc->GetRegisterCount())
    {
        *type = UNW_NOT_A_REG;
        return UNW_ESUCCESS;
    }
    
    const char *name = regc->GetRegisterName (regnum);
    if (name == NULL)
    {
        *type = UNW_NOT_A_REG;
        return UNW_ESUCCESS;
    }
    strlcpy (buf, name, buflen);
    
    const lldb::RegisterInfo *reginfo = regc->GetRegisterInfoAtIndex (regnum);
    if (reginfo == NULL || reginfo->encoding == eEncodingInvalid)
    {
        *type = UNW_NOT_A_REG;
        return UNW_ESUCCESS;
    }
    if (reginfo->encoding == eEncodingUint || reginfo->encoding == eEncodingSint)
        *type = UNW_INTEGER_REG;
    if (reginfo->encoding == eEncodingIEEE754)
        *type = UNW_FLOATING_POINT_REG;
    if (reginfo->encoding == eEncodingVector)
        *type = UNW_VECTOR_REG;
    
    return UNW_ESUCCESS;
}


static int
read_byte_for_edis (uint8_t *buf, uint64_t addr, void *arg)
{
    if (arg == 0)
        return -1;
    Thread *th = (Thread *) arg;
    DataBufferHeap onebyte(1, 0);
    Error error;
    if (th->GetProcess().ReadMemory (addr, onebyte.GetBytes(), onebyte.GetByteSize(), error) != 1)
        return -1;
    *buf = onebyte.GetBytes()[0];
    return UNW_ESUCCESS;
}

static int 
instruction_length (lldb_private::unw_addr_space_t as, lldb_private::unw_word_t addr, int *length, void *arg)
{
    EDDisassemblerRef disasm;
    EDInstRef         cur_insn;
    
    if (arg == 0)
        return -1;
    Thread *thread = (Thread *) arg;
    Target &target = thread->GetProcess().GetTarget();

    const ArchSpec::CPU arch_cpu = target.GetArchitecture ().GetGenericCPUType();

    if (arch_cpu == ArchSpec::eCPU_i386)
    {
        if (EDGetDisassembler (&disasm, "i386-apple-darwin", kEDAssemblySyntaxX86ATT) != 0)
            return -1;
    }
    else if (arch_cpu == ArchSpec::eCPU_x86_64)
    {
        if (EDGetDisassembler (&disasm, "x86_64-apple-darwin", kEDAssemblySyntaxX86ATT) != 0)
            return -1;
    }
    else
    {
        return -1;
    }
    
    if (EDCreateInsts (&cur_insn, 1, disasm, read_byte_for_edis, addr, arg) != 1)
        return -1;
    *length = EDInstByteSize (cur_insn);
    EDReleaseInst (cur_insn);
    return UNW_ESUCCESS;
}

lldb_private::unw_accessors_t 
get_macosx_libunwind_callbacks () {
  lldb_private::unw_accessors_t ap;
  bzero (&ap, sizeof (lldb_private::unw_accessors_t));
  ap.find_proc_info = NULL;
  ap.put_unwind_info = NULL;
  ap.get_dyn_info_list_addr = NULL;
  ap.find_image_info = find_image_info;
  ap.access_mem = NULL;
  ap.access_reg = access_reg;
  ap.access_fpreg = NULL;
  ap.access_vecreg = NULL;
  ap.resume = NULL;
  ap.get_proc_name = get_proc_name;
  ap.get_proc_bounds = get_proc_bounds;
  ap.access_raw = access_raw;
  ap.reg_info = reg_info;
  ap.proc_is_sigtramp = NULL;
  ap.proc_is_inferior_function_call = NULL;
  ap.access_reg_inf_func_call = NULL;
  ap.instruction_length = instruction_length;
  return ap;
}


} // namespace lldb_private

#endif  // #if defined(__cplusplus)
#endif // #ifndef liblldb_MacOSXLibunwindCallbacks_cpp_
