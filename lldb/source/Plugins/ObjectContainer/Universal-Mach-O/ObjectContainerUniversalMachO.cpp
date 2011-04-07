//===-- ObjectContainerUniversalMachO.cpp -----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ObjectContainerUniversalMachO.h"
#include "lldb/Core/Stream.h"
#include "lldb/Core/ArchSpec.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
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
    Module* module,
    DataBufferSP& dataSP,
    const FileSpec *file,
    addr_t offset,
    addr_t length
)
{
    if (ObjectContainerUniversalMachO::MagicBytesMatch(dataSP))
    {
        std::auto_ptr<ObjectContainerUniversalMachO> container_ap(new ObjectContainerUniversalMachO (module, dataSP, file, offset, length));
        if (container_ap->ParseHeader())
        {
            return container_ap.release();
        }
    }
    return NULL;
}



bool
ObjectContainerUniversalMachO::MagicBytesMatch (DataBufferSP& dataSP)
{
    DataExtractor data(dataSP, lldb::endian::InlHostByteOrder(), 4);
    uint32_t offset = 0;
    uint32_t magic = data.GetU32(&offset);
    return magic == UniversalMagic || magic == UniversalMagicSwapped;
}

ObjectContainerUniversalMachO::ObjectContainerUniversalMachO
(
    Module* module,
    DataBufferSP& dataSP,
    const FileSpec *file,
    addr_t offset,
    addr_t length
) :
    ObjectContainer (module, file, offset, length, dataSP),
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
    // Store the file offset for this universal file as we could have a universal .o file
    // in a BSD archive, or be contained in another kind of object.
    uint32_t offset = 0;
    // Universal mach-o files always have their headers in big endian.
    m_data.SetByteOrder (eByteOrderBig);
    m_header.magic = m_data.GetU32(&offset);

    if (m_header.magic == UniversalMagic)
    {
        m_data.SetAddressByteSize(4);

        m_header.nfat_arch = m_data.GetU32(&offset);

        const size_t nfat_arch_size = sizeof(fat_arch) * m_header.nfat_arch;
        // See if the current data we have is enough for all of the fat headers?
        if (!m_data.ValidOffsetForDataOfSize(offset, nfat_arch_size))
        {
            // The fat headers are larger than the number of bytes we have been
            // given when this class was constructed. We will read the exact number
            // of bytes that we need.
            DataBufferSP data_sp(m_file.ReadFileContents(m_offset, nfat_arch_size + sizeof(fat_header)));
            m_data.SetData (data_sp);
        }

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
        // Now that we have indexed the universal headers, we no longer need any cached data.
        m_data.Clear();

        return true;
    }
    else
    {
        memset(&m_header, 0, sizeof(m_header));
    }

    return false;
}

void
ObjectContainerUniversalMachO::Dump (Stream *s) const
{
    s->Printf("%.*p: ", (int)sizeof(void*) * 2, this);
    s->Indent();
    const size_t num_archs = GetNumArchitectures();
    const size_t num_objects = GetNumObjects();
    s->Printf("ObjectContainerUniversalMachO, num_archs = %u, num_objects = %u", num_archs, num_objects);
    uint32_t i;
    ArchSpec arch;
    s->IndentMore();
    for (i=0; i<num_archs; i++)
    {
        s->Indent();
        GetArchitectureAtIndex(i, arch);
        s->Printf("arch[%u] = %s\n", arch.GetArchitectureName());
    }
    for (i=0; i<num_objects; i++)
    {
        s->Indent();
        s->Printf("object[%u] = %s\n", GetObjectNameAtIndex (i));
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

ObjectFile *
ObjectContainerUniversalMachO::GetObjectFile (const FileSpec *file)
{
    uint32_t arch_idx = 0;
    ArchSpec arch;
    // If the module hasn't specified an architecture yet, set it to the default 
    // architecture:
    if (!m_module->GetArchitecture().IsValid())
    {
        arch = Target::GetDefaultArchitecture ();
        if (!arch.IsValid())
            arch.SetTriple (LLDB_ARCH_DEFAULT, NULL);
    }
    else
        arch = m_module->GetArchitecture();
        
    ArchSpec curr_arch;
    for (arch_idx = 0; arch_idx < m_header.nfat_arch; ++arch_idx)
    {
        if (GetArchitectureAtIndex (arch_idx, curr_arch))
        {
            if (arch == curr_arch)
            {
                return ObjectFile::FindPlugin (m_module, file, m_offset + m_fat_archs[arch_idx].offset, m_fat_archs[arch_idx].size);
            }
        }
    }
    return NULL;
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


