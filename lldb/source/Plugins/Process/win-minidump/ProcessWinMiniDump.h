//===-- ProcessWinMiniDump.h ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ProcessWinMiniDump_h_
#define liblldb_ProcessWinMiniDump_h_

#include <list>
#include <vector>

#include "lldb/Core/ConstString.h"
#include "lldb/Core/Error.h"
#include "lldb/Target/Process.h"

struct ThreadData;

class ProcessWinMiniDump : public lldb_private::Process
{
public:
    static lldb::ProcessSP
    CreateInstance (lldb_private::Target& target,
                    lldb_private::Listener &listener,
                    const lldb_private::FileSpec *crash_file_path);

    static void
    Initialize();

    static void
    Terminate();

    static lldb_private::ConstString
    GetPluginNameStatic();

    static const char *
    GetPluginDescriptionStatic();

    ProcessWinMiniDump(lldb_private::Target& target,
                       lldb_private::Listener &listener,
                       const lldb_private::FileSpec &core_file);

    virtual
    ~ProcessWinMiniDump();

    bool
    CanDebug(lldb_private::Target &target, bool plugin_specified_by_name) override;

    lldb_private::Error
    DoLoadCore() override;

    lldb_private::DynamicLoader *
    GetDynamicLoader() override;

    lldb_private::ConstString
    GetPluginName() override;

    uint32_t
    GetPluginVersion() override;

    lldb_private::Error
    DoDestroy() override;

    void
    RefreshStateAfterStop() override;

    bool
    IsAlive() override;

    size_t
    ReadMemory(lldb::addr_t addr, void *buf, size_t size, lldb_private::Error &error) override;

    size_t
    DoReadMemory(lldb::addr_t addr, void *buf, size_t size, lldb_private::Error &error) override;

    lldb::addr_t
    GetImageInfoAddress() override;

    lldb_private::ArchSpec
    GetArchitecture();

protected:
    void
    Clear();

    bool
    UpdateThreadList(lldb_private::ThreadList &old_thread_list,
                     lldb_private::ThreadList &new_thread_list) override;

private:
    lldb_private::Error
    MapMiniDumpIntoMemory(const char *file);

    lldb_private::ArchSpec
    DetermineArchitecture();

    // Isolate the data to keep Windows-specific types out of this header.  Can't
    // use the typical pimpl idiom because the implementation of this class also
    // needs access to public and protected members of the base class.
    class Data;
    std::unique_ptr<Data> m_data_up;
};

#endif  // liblldb_ProcessWinMiniDump_h_
