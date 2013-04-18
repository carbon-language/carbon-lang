//===-- ObjectContainerUniversalMachO.cpp -----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ObjectContainerUniversalMachO.h"
#include "lldb/Core/ArchSpec.h"
#include "lldb/Core/DataBuffer.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/Stream.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Target/Target.h"

using namespace lldb;
using namespace lldb_private;
using namespace llvm::MachO;

void
ObjectContainerUniversalMachO::Initialize()
{
    PluginManager::RegisterPlugin (GetPluginNameStatic(),
                                   GetPluginDescriptionStatic(),
                                   CreateInstance);
}

void
ObjectContainerUniversalMachO::Terminate()
{
    PluginManager::UnregisterPlugin (CreateInstance);
}


const char *
ObjectContainerUniversalMachO::GetPluginNameStatic()
{
    return "object-container.mach-o";
}

const char *
ObjectContainerUniversalMachO::GetPluginDescriptionStatic()
{
    return "Universal mach-o object container reader.";
}


ObjectContainer *
ObjectContainerUniversalMachO::CreateInstance
(
    const lldb::ModuleSP &module_sp,
    DataBufferSP& data_sp,
    lldb::offset_t data_offset,
    const FileSpec *file,
    lldb::offset_t file_offset,
    lldb::offset_t length
)
{
    // We get data when we aren't trying to look for cached container information,
    // so only try and look for an architecture slice if we get data
    if (data_sp)
    {
        DataExtractor data;
        data.SetData (data_sp, data_offset, length);
        if (ObjectContainerUniversalMachO::MagicBytesMatch(data))
        {
            std::unique_ptr<ObjectContainerUniversalMachO> container_ap(new ObjectContainerUniversalMachO (module_sp, data_sp, data_offset, file, file_offset, length));
            if (container_ap->ParseHeader())
            {
                return container_ap.release();
            }
        }
    }
    return NULL;
}

bool
ObjectContainerUniversalMachO::MagicBytesMatch (const DataExtractor &data)
{
    lldb::offset_t offset = 0;
    uint32_t magic = data.GetU32(&offset);
    return magic == UniversalMagic || magic == UniversalMagicSwapped;
}

ObjectContainerUniversalMachO::ObjectContainerUniversalMachO
(
    const lldb::ModuleSP &module_sp,
    DataBufferSP& data_sp,
    lldb::offset_t data_offset,
    const FileSpec *file,
    lldb::offset_t file_offset,
    lldb::offset_t length
) :
    ObjectContainer (module_sp, file, file_offset, length, data_sp, data_offset),
    m_header(),
    m_fat_archs()
{
    memset(&m_header, 0, sizeof(m_header));
}


ObjectContainerUniversalMachO::~ObjectContainerUniversalMachO()
{
}

bool
ObjectContainerUniversalMachO::ParseHeader ()
{
    bool success = false;
    // Store the file offset for this universal file as we could have a universal .o file
    // in a BSD archive, or be contained in another kind of object.
    lldb::offset_t offset = 0;
    // Universal mach-o files always have their headers in big endian.
    m_data.SetByteOrder (eByteOrderBig);
    m_header.magic = m_data.GetU32(&offset);

    if (m_header.magic == UniversalMagic)
    {
        m_data.SetAddressByteSize(4);

        m_header.nfat_arch = m_data.GetU32(&offset);

        // Now we should have enough data for all of the fat headers, so lets index
        // them so we know how many architectures that this universal binary contains.
        uint32_t arch_idx = 0;
        for (arch_idx = 0; arch_idx < m_header.nfat_arch; ++arch_idx)
        {
            if (m_data.ValidOffsetForDataOfSize(offset, sizeof(fat_arch)))
            {
                fat_arch arch;
                if (m_data.GetU32(&offset, &arch, sizeof(fat_arch)/sizeof(uint32_t)))
                {
                    m_fat_archs.push_back(arch);
                }
            }
        }
        success = true;
    }
    else
    {
        memset(&m_header, 0, sizeof(m_header));
    }

    // We no longer need any data, we parsed all we needed to parse
    // and cached it in m_header and m_fat_archs
    m_data.Clear();
    return success;
}

void
ObjectContainerUniversalMachO::Dump (Stream *s) const
{
    s->Printf("%p: ", this);
    s->Indent();
    const size_t num_archs = GetNumArchitectures();
    const size_t num_objects = GetNumObjects();
    s->Printf("ObjectContainerUniversalMachO, num_archs = %lu, num_objects = %lu", num_archs, num_objects);
    uint32_t i;
    ArchSpec arch;
    s->IndentMore();
    for (i=0; i<num_archs; i++)
    {
        s->Indent();
        GetArchitectureAtIndex(i, arch);
        s->Printf("arch[%u] = %s\n", i, arch.GetArchitectureName());
    }
    for (i=0; i<num_objects; i++)
    {
        s->Indent();
        s->Printf("object[%u] = %s\n", i, GetObjectNameAtIndex (i));
    }
    s->IndentLess();
    s->EOL();
}

size_t
ObjectContainerUniversalMachO::GetNumArchitectures () const
{
    return m_header.nfat_arch;
}

bool
ObjectContainerUniversalMachO::GetArchitectureAtIndex (uint32_t idx, ArchSpec& arch) const
{
    if (idx < m_header.nfat_arch)
    {
        arch.SetArchitecture (eArchTypeMachO, m_fat_archs[idx].cputype, m_fat_archs[idx].cpusubtype);
        return true;
    }
    return false;
}

ObjectFileSP
ObjectContainerUniversalMachO::GetObjectFile (const FileSpec *file)
{
    uint32_t arch_idx = 0;
    ArchSpec arch;
    // If the module hasn't specified an architecture yet, set it to the default 
    // architecture:
    ModuleSP module_sp (GetModule());
    if (module_sp)
    {
        if (!module_sp->GetArchitecture().IsValid())
        {
            arch = Target::GetDefaultArchitecture ();
            if (!arch.IsValid())
                arch.SetTriple (LLDB_ARCH_DEFAULT);
        }
        else
            arch = module_sp->GetArchitecture();
            
        ArchSpec curr_arch;
        // First, try to find an exact match for the Arch of the Target.
        for (arch_idx = 0; arch_idx < m_header.nfat_arch; ++arch_idx)
        {
            if (GetArchitectureAtIndex (arch_idx, curr_arch) && arch.IsExactMatch(curr_arch))
                break;
        }

        // Failing an exact match, try to find a compatible Arch of the Target.
        if (arch_idx >= m_header.nfat_arch)
        {
            for (arch_idx = 0; arch_idx < m_header.nfat_arch; ++arch_idx)
            {
                if (GetArchitectureAtIndex (arch_idx, curr_arch) && arch.IsCompatibleMatch(curr_arch))
                    break;
            }
        }

        if (arch_idx < m_header.nfat_arch)
        {
            DataBufferSP data_sp;
            lldb::offset_t data_offset = 0;
            return ObjectFile::FindPlugin (module_sp,
                                           file,
                                           m_offset + m_fat_archs[arch_idx].offset,
                                           m_fat_archs[arch_idx].size,
                                           data_sp,
                                           data_offset);
        }
    }
    return ObjectFileSP();
}


//------------------------------------------------------------------
// PluginInterface protocol
//------------------------------------------------------------------
const char *
ObjectContainerUniversalMachO::GetPluginName()
{
    return "ObjectContainerUniversalMachO";
}

const char *
ObjectContainerUniversalMachO::GetShortPluginName()
{
    return GetPluginNameStatic();
}

uint32_t
ObjectContainerUniversalMachO::GetPluginVersion()
{
    return 1;
}


