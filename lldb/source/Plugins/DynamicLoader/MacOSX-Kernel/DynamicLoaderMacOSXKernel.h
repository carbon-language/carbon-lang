//===-- DynamicLoaderMacOSXKernel.h -----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_DynamicLoaderMacOSXKernel_h_
#define liblldb_DynamicLoaderMacOSXKernel_h_

// C Includes
// C++ Includes
#include <map>
#include <vector>
#include <string>

// Other libraries and framework includes
#include "llvm/Support/MachO.h"

#include "lldb/Target/DynamicLoader.h"
#include "lldb/Host/FileSpec.h"
#include "lldb/Host/TimeValue.h"
#include "lldb/Core/UUID.h"
#include "lldb/Host/Mutex.h"
#include "lldb/Target/Process.h"

class DynamicLoaderMacOSXKernel : public lldb_private::DynamicLoader
{
public:
    //------------------------------------------------------------------
    // Static Functions
    //------------------------------------------------------------------
    static void
    Initialize();

    static void
    Terminate();

    static const char *
    GetPluginNameStatic();

    static const char *
    GetPluginDescriptionStatic();

    static lldb_private::DynamicLoader *
    CreateInstance (lldb_private::Process *process, bool force);

    DynamicLoaderMacOSXKernel (lldb_private::Process *process);

    virtual
    ~DynamicLoaderMacOSXKernel ();
    //------------------------------------------------------------------
    /// Called after attaching a process.
    ///
    /// Allow DynamicLoader plug-ins to execute some code after
    /// attaching to a process.
    //------------------------------------------------------------------
    virtual void
    DidAttach ();

    virtual void
    DidLaunch ();

    virtual lldb::ThreadPlanSP
    GetStepThroughTrampolinePlan (lldb_private::Thread &thread,
                                  bool stop_others);

    virtual lldb_private::Error
    CanLoadImage ();

    //------------------------------------------------------------------
    // PluginInterface protocol
    //------------------------------------------------------------------
    virtual const char *
    GetPluginName();

    virtual const char *
    GetShortPluginName();

    virtual uint32_t
    GetPluginVersion();

protected:
    void
    PrivateInitialize (lldb_private::Process *process);

    void
    PrivateProcessStateChanged (lldb_private::Process *process,
                                lldb::StateType state);
    
    void
    UpdateIfNeeded();

    void
    LoadKernelModuleIfNeeded ();

    void
    Clear (bool clear_process);

    void
    PutToLog (lldb_private::Log *log) const;

    static bool
    BreakpointHitCallback (void *baton,
                           lldb_private::StoppointCallbackContext *context,
                           lldb::user_id_t break_id,
                           lldb::user_id_t break_loc_id);

    bool
    BreakpointHit (lldb_private::StoppointCallbackContext *context, 
                   lldb::user_id_t break_id, 
                   lldb::user_id_t break_loc_id);
    uint32_t
    AddrByteSize()
    {
        switch (m_kernel.header.magic)
        {
            case llvm::MachO::HeaderMagic32:
            case llvm::MachO::HeaderMagic32Swapped:
                return 4;

            case llvm::MachO::HeaderMagic64:
            case llvm::MachO::HeaderMagic64Swapped:
                return 8;

            default:
                break;
        }
        return 0;
    }

    static lldb::ByteOrder
    GetByteOrderFromMagic (uint32_t magic)
    {
        switch (magic)
        {
            case llvm::MachO::HeaderMagic32:
            case llvm::MachO::HeaderMagic64:
                return lldb::endian::InlHostByteOrder();

            case llvm::MachO::HeaderMagic32Swapped:
            case llvm::MachO::HeaderMagic64Swapped:
                if (lldb::endian::InlHostByteOrder() == lldb::eByteOrderBig)
                    return lldb::eByteOrderLittle;
                else
                    return lldb::eByteOrderBig;

            default:
                break;
        }
        return lldb::eByteOrderInvalid;
    }

    class Segment
    {
    public:

        Segment() :
            name(),
            vmaddr(LLDB_INVALID_ADDRESS),
            vmsize(0),
            fileoff(0),
            filesize(0),
            maxprot(0),
            initprot(0),
            nsects(0),
            flags(0)
        {
        }

        lldb_private::ConstString name;
        lldb::addr_t vmaddr;
        lldb::addr_t vmsize;
        lldb::addr_t fileoff;
        lldb::addr_t filesize;
        uint32_t maxprot;
        uint32_t initprot;
        uint32_t nsects;
        uint32_t flags;

        bool
        operator==(const Segment& rhs) const
        {
            return name == rhs.name && vmaddr == rhs.vmaddr && vmsize == rhs.vmsize;
        }

        void
        PutToLog (lldb_private::Log *log,
                  lldb::addr_t slide) const;

    };

    enum { KERNEL_MODULE_MAX_NAME = 64u };
    
    struct OSKextLoadedKextSummary
    {
        char                     name[KERNEL_MODULE_MAX_NAME];
        lldb::ModuleSP           module_sp;
        uint32_t                 module_create_stop_id;
        lldb_private::UUID       uuid;            // UUID for this dylib if it has one, else all zeros
        lldb_private::Address    so_address;        // The section offset address for this kext in case it can be read from object files
        uint64_t                 address;
        uint64_t                 size;
        uint64_t                 version;
        uint32_t                 load_tag;
        uint32_t                 flags;
        uint64_t                 reference_list;
        llvm::MachO::mach_header header;    // The mach header for this image
        std::vector<Segment>     segments;      // All segment vmaddr and vmsize pairs for this executable (from memory of inferior)

        OSKextLoadedKextSummary() :
            module_sp (),
            module_create_stop_id (UINT32_MAX),
            uuid (),
            so_address (),
            address (LLDB_INVALID_ADDRESS),
            size (0),
            version (0),
            load_tag (0),
            flags (0),
            reference_list (0),
            header(),
            segments()
        {
            name[0] = '\0';
        }

        void
        Clear (bool load_cmd_data_only)
        {
            if (!load_cmd_data_only)
            {
                so_address.Clear();
                address = LLDB_INVALID_ADDRESS;
                size = 0;
                version = 0;
                load_tag = 0;
                flags = 0;
                reference_list = 0;
                name[0] = '\0';
                ::memset (&header, 0, sizeof(header));
            }
            module_sp.reset();
            module_create_stop_id = UINT32_MAX;
            uuid.Clear();
            segments.clear();
        }

        bool
        operator == (const OSKextLoadedKextSummary& rhs) const
        {
            return  address == rhs.address
                    && size == rhs.size
            //&& module_sp.get() == rhs.module_sp.get()
                    && uuid == rhs.uuid
                    && version == rhs.version
                    && load_tag == rhs.load_tag
                    && flags == rhs.flags
                    && reference_list == rhs.reference_list
                    && strncmp (name, rhs.name, KERNEL_MODULE_MAX_NAME) == 0
                    && memcmp(&header, &rhs.header, sizeof(header)) == 0
                    && segments == rhs.segments;
        }

        bool
        UUIDValid() const
        {
            return uuid.IsValid();
        }

        uint32_t
        GetAddressByteSize ()
        {
            if (header.cputype)
            {
                if (header.cputype & llvm::MachO::CPUArchABI64)
                    return 8;
                else
                    return 4;
            }
            return 0;
        }

        lldb::ByteOrder
        GetByteOrder()
        {
            switch (header.magic)
            {
            case llvm::MachO::HeaderMagic32:        // MH_MAGIC
            case llvm::MachO::HeaderMagic64:        // MH_MAGIC_64
                return lldb::endian::InlHostByteOrder();

            case llvm::MachO::HeaderMagic32Swapped: // MH_CIGAM
            case llvm::MachO::HeaderMagic64Swapped: // MH_CIGAM_64
                if (lldb::endian::InlHostByteOrder() == lldb::eByteOrderLittle)
                    return lldb::eByteOrderBig;
                else
                    return lldb::eByteOrderLittle;
            default:
                assert (!"invalid header.magic value");
                break;
            }
            return lldb::endian::InlHostByteOrder();
        }

        lldb_private::ArchSpec
        GetArchitecture () const
        {
            return lldb_private::ArchSpec (lldb_private::eArchTypeMachO, header.cputype, header.cpusubtype);
        }

        const Segment *
        FindSegment (const lldb_private::ConstString &name) const;

        void
        PutToLog (lldb_private::Log *log) const;

        typedef std::vector<OSKextLoadedKextSummary> collection;
        typedef collection::iterator iterator;
        typedef collection::const_iterator const_iterator;
    };

    struct OSKextLoadedKextSummaryHeader
    {
        uint32_t version;
        uint32_t entry_size;
        uint32_t entry_count;
        uint32_t reserved; /* explicit alignment for gdb  */
        lldb::addr_t image_infos_addr;

        OSKextLoadedKextSummaryHeader() :
            version (0),
            entry_size (0),
            entry_count (0),
            reserved (0),
            image_infos_addr (LLDB_INVALID_ADDRESS)
        {
        }

        void
        Clear()
        {
            version = 0;
            entry_size = 0;
            entry_count = 0;
            reserved = 0;
            image_infos_addr = LLDB_INVALID_ADDRESS;
        }

        bool
        IsValid() const
        {
            return version >= 1 || version <= 2;
        }
    };

    bool
    ReadMachHeader (OSKextLoadedKextSummary& kext_summary,
                    lldb_private::DataExtractor *load_command_data);

    void
    RegisterNotificationCallbacks();

    void
    UnregisterNotificationCallbacks();

    uint32_t
    ParseLoadCommands (const lldb_private::DataExtractor& data,
                       OSKextLoadedKextSummary& dylib_info);

    bool
    UpdateImageLoadAddress(OSKextLoadedKextSummary& info);

    bool
    FindTargetModule (OSKextLoadedKextSummary &image_info,
                      bool can_create,
                      bool *did_create_ptr);

    void
    SetNotificationBreakpointIfNeeded ();

    bool
    ReadAllKextSummaries ();

    bool
    ReadKextSummaryHeader ();
    
    bool
    ParseKextSummaries (const lldb_private::Address &kext_summary_addr, 
                        uint32_t count);
    
    bool
    AddModulesUsingImageInfos (OSKextLoadedKextSummary::collection &image_infos);
    
    void
    UpdateImageInfosHeaderAndLoadCommands(OSKextLoadedKextSummary::collection &image_infos, 
                                          uint32_t infos_count, 
                                          bool update_executable);

    bool
    UpdateCommPageLoadAddress (lldb_private::Module *module);

    uint32_t
    ReadKextSummaries (const lldb_private::Address &kext_summary_addr,
                       uint32_t image_infos_count, 
                       OSKextLoadedKextSummary::collection &image_infos);
    
    bool
    UnloadImageLoadAddress (OSKextLoadedKextSummary& info);

    OSKextLoadedKextSummary m_kernel; // Info about the current kernel image being used
    lldb_private::Address m_kext_summary_header_ptr_addr;
    lldb_private::Address m_kext_summary_header_addr;
    OSKextLoadedKextSummaryHeader m_kext_summary_header;
    OSKextLoadedKextSummary::collection m_kext_summaries;
    mutable lldb_private::Mutex m_mutex;
    lldb::user_id_t m_break_id;

private:
    DISALLOW_COPY_AND_ASSIGN (DynamicLoaderMacOSXKernel);
};

#endif  // liblldb_DynamicLoaderMacOSXKernel_h_
