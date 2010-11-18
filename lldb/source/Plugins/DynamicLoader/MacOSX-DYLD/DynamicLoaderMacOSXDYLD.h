//===-- DynamicLoaderMacOSXDYLD.h -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_DynamicLoaderMacOSXDYLD_h_
#define liblldb_DynamicLoaderMacOSXDYLD_h_

// C Includes
// C++ Includes
#include <map>
#include <vector>
#include <string>

// Other libraries and framework includes
#include "llvm/Support/MachO.h"

#include "lldb/Target/DynamicLoader.h"
#include "lldb/Core/FileSpec.h"
#include "lldb/Core/UUID.h"
#include "lldb/Host/Mutex.h"
#include "lldb/Target/Process.h"

class DynamicLoaderMacOSXDYLD : public lldb_private::DynamicLoader
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
    CreateInstance (lldb_private::Process *process);

    DynamicLoaderMacOSXDYLD (lldb_private::Process *process);

    virtual
    ~DynamicLoaderMacOSXDYLD ();
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

    //------------------------------------------------------------------
    // Process::Notifications callback functions
    //------------------------------------------------------------------
    static void
    Initialize (void *baton,
                lldb_private::Process *process);

    static void
    ProcessStateChanged (void *baton,
                         lldb_private::Process *process,
                         lldb::StateType state);

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

    virtual void
    GetPluginCommandHelp (const char *command, lldb_private::Stream *strm);

    virtual lldb_private::Error
    ExecutePluginCommand (lldb_private::Args &command, lldb_private::Stream *strm);

    virtual lldb_private::Log *
    EnablePluginLogging (lldb_private::Stream *strm, lldb_private::Args &command);



protected:
    void
    PrivateInitialize (lldb_private::Process *process);

    void
    PrivateProcessStateChanged (lldb_private::Process *process,
                                lldb::StateType state);
    bool
    LocateDYLD ();

    bool
    DidSetNotificationBreakpoint () const;

    void
    Clear (bool clear_process);

    void
    PutToLog (lldb_private::Log *log) const;

    bool
    ReadDYLDInfoFromMemoryAndSetNotificationCallback (lldb::addr_t addr);

    uint32_t
    UpdateAllImageInfos ();

    static bool
    NotifyBreakpointHit (void *baton,
                         lldb_private::StoppointCallbackContext *context,
                         lldb::user_id_t break_id,
                         lldb::user_id_t break_loc_id);
    void
    UpdateAllImageInfosHeaderAndLoadCommands ();

    bool
    UpdateCommPageLoadAddress (lldb_private::Module *module);

    uint32_t
    AddrByteSize()
    {
        switch (m_dyld.header.magic)
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
                return lldb::eByteOrderHost;

            case llvm::MachO::HeaderMagic32Swapped:
            case llvm::MachO::HeaderMagic64Swapped:
                if (lldb::eByteOrderHost == lldb::eByteOrderBig)
                    return lldb::eByteOrderLittle;
                else
                    return lldb::eByteOrderBig;

            default:
                break;
        }
        return lldb::eByteOrderInvalid;
    }

    bool
    ReadMachHeader (lldb::addr_t addr,
                    llvm::MachO::mach_header *header,
                    lldb_private::DataExtractor *load_command_data);
    class Segment
    {
    public:

        Segment() :
            name(),
            addr(LLDB_INVALID_ADDRESS),
            size(0)
        {
        }

        lldb_private::ConstString name;
        lldb::addr_t addr;
        lldb::addr_t size;

        bool
        operator==(const Segment& rhs) const
        {
            return name == rhs.name && addr == rhs.addr && size == rhs.size;
        }

        void
        PutToLog (lldb_private::Log *log,
                  lldb::addr_t slide) const;

    };

    struct DYLDImageInfo
    {
        lldb::addr_t address;           // Address of mach header for this dylib
        lldb::addr_t slide;             // The amount to slide all segments by if there is a global slide.
        lldb::addr_t mod_date;          // Modification date for this dylib
        lldb_private::FileSpec file_spec;       // Resolved path for this dylib
        lldb_private::UUID uuid;                // UUID for this dylib if it has one, else all zeros
        llvm::MachO::mach_header header;      // The mach header for this image
        std::vector<Segment> segments;  // All segment vmaddr and vmsize pairs for this executable (from memory of inferior)

        DYLDImageInfo() :
            address(LLDB_INVALID_ADDRESS),
            slide(0),
            mod_date(0),
            file_spec(),
            uuid(),
            header(),
            segments()
        {
        }

        void
        Clear(bool load_cmd_data_only)
        {
            if (!load_cmd_data_only)
            {
                address = LLDB_INVALID_ADDRESS;
                slide = 0;
                mod_date = 0;
                file_spec.Clear();
                ::bzero (&header, sizeof(header));
            }
            uuid.Clear();
            segments.clear();
        }

        bool
        operator == (const DYLDImageInfo& rhs) const
        {
            return  address == rhs.address
                && slide == rhs.slide
                && mod_date == rhs.mod_date
                && file_spec == rhs.file_spec
                && uuid == rhs.uuid
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
                return lldb::eByteOrderHost;

            case llvm::MachO::HeaderMagic32Swapped: // MH_CIGAM
            case llvm::MachO::HeaderMagic64Swapped: // MH_CIGAM_64
                if (lldb::eByteOrderHost == lldb::eByteOrderLittle)
                    return lldb::eByteOrderBig;
                else
                    return lldb::eByteOrderLittle;
            default:
                assert (!"invalid header.magic value");
                break;
            }
            return lldb::eByteOrderHost;
        }

        const Segment *
        FindSegment (const lldb_private::ConstString &name) const;

        void
        PutToLog (lldb_private::Log *log) const;

        typedef std::vector<DYLDImageInfo> collection;
        typedef collection::iterator iterator;
        typedef collection::const_iterator const_iterator;
    };

    struct DYLDAllImageInfos
    {
        uint32_t version;
        uint32_t dylib_info_count;              // Version >= 1
        lldb::addr_t dylib_info_addr;           // Version >= 1
        lldb::addr_t notification;              // Version >= 1
        bool processDetachedFromSharedRegion;   // Version >= 1
        bool libSystemInitialized;              // Version >= 2
        lldb::addr_t dyldImageLoadAddress;      // Version >= 2

        DYLDAllImageInfos() :
            version (0),
            dylib_info_count (0),
            dylib_info_addr (LLDB_INVALID_ADDRESS),
            notification (LLDB_INVALID_ADDRESS),
            processDetachedFromSharedRegion (false),
            libSystemInitialized (false),
            dyldImageLoadAddress (LLDB_INVALID_ADDRESS)
        {
        }

        void
        Clear()
        {
            version = 0;
            dylib_info_count = 0;
            dylib_info_addr = LLDB_INVALID_ADDRESS;
            notification = LLDB_INVALID_ADDRESS;
            processDetachedFromSharedRegion = false;
            libSystemInitialized = false;
            dyldImageLoadAddress = LLDB_INVALID_ADDRESS;
        }

        bool
        IsValid() const
        {
            return version >= 1 || version <= 6;
        }
    };

    void
    RegisterNotificationCallbacks();

    void
    UnregisterNotificationCallbacks();

    uint32_t
    ParseLoadCommands (const lldb_private::DataExtractor& data,
                       struct DYLDImageInfo& dylib_info,
                       lldb_private::FileSpec *lc_id_dylinker);

    bool
    UpdateImageLoadAddress(lldb_private::Module *module,
                           struct DYLDImageInfo& info);

    bool
    UnloadImageLoadAddress (lldb_private::Module *module,
                            struct DYLDImageInfo& info);

    bool
    NeedToLocateDYLD () const;

    bool
    SetNotificationBreakpoint ();

    bool
    ReadAllImageInfosStructure ();

    DYLDImageInfo m_dyld;               // Info about the curent dyld being used
    lldb::addr_t m_dyld_all_image_infos_addr;
    DYLDAllImageInfos m_dyld_all_image_infos;
    uint32_t m_dyld_all_image_infos_stop_id;
    lldb::user_id_t m_break_id;
    DYLDImageInfo::collection m_dyld_image_infos;   // Current shared libraries information
    uint32_t m_dyld_image_infos_stop_id;    // The process stop ID that "m_dyld_image_infos" is valid for
    mutable lldb_private::Mutex m_mutex;
    lldb_private::Process::Notifications m_notification_callbacks;

private:
    DISALLOW_COPY_AND_ASSIGN (DynamicLoaderMacOSXDYLD);
};

#endif  // liblldb_DynamicLoaderMacOSXDYLD_h_
