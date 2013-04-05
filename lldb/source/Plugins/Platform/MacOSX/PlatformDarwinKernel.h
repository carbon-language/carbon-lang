//===-- PlatformDarwinKernel.h ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_PlatformDarwinKernel_h_
#define liblldb_PlatformDarwinKernel_h_

#if defined (__APPLE__)  // This Plugin uses the Mac-specific source/Host/macosx/cfcpp utilities


// C Includes
// C++ Includes
// Other libraries and framework includes
#include "lldb/Host/FileSpec.h"

// Project includes
#include "PlatformDarwin.h"

class PlatformDarwinKernel : public PlatformDarwin
{
public:

    //------------------------------------------------------------
    // Class Functions
    //------------------------------------------------------------
    static lldb_private::Platform*
    CreateInstance (bool force, const lldb_private::ArchSpec *arch);

    static void
    Initialize ();

    static void
    Terminate ();

    static const char *
    GetPluginNameStatic ();

    static const char *
    GetShortPluginNameStatic();

    static const char *
    GetDescriptionStatic();

    //------------------------------------------------------------
    // Class Methods
    //------------------------------------------------------------
    PlatformDarwinKernel (lldb_private::LazyBool is_ios_debug_session);

    virtual
    ~PlatformDarwinKernel();

    //------------------------------------------------------------
    // lldb_private::PluginInterface functions
    //------------------------------------------------------------
    virtual const char *
    GetPluginName()
    {
        return GetPluginNameStatic();
    }

    virtual const char *
    GetShortPluginName()
    {
        return GetShortPluginNameStatic();
    }

    virtual uint32_t
    GetPluginVersion()
    {
        return 1;
    }

    //------------------------------------------------------------
    // lldb_private::Platform functions
    //------------------------------------------------------------
    virtual const char *
    GetDescription ()
    {
        return GetDescriptionStatic();
    }

    virtual void
    GetStatus (lldb_private::Stream &strm);

    virtual lldb_private::Error
    GetSharedModule (const lldb_private::ModuleSpec &module_spec,
                     lldb::ModuleSP &module_sp,
                     const lldb_private::FileSpecList *module_search_paths_ptr,
                     lldb::ModuleSP *old_module_sp_ptr,
                     bool *did_create_ptr);

    virtual bool
    GetSupportedArchitectureAtIndex (uint32_t idx, 
                                     lldb_private::ArchSpec &arch);

protected:

    // Map from kext bundle ID ("com.apple.filesystems.exfat") to FileSpec for the kext bundle on 
    // the host ("/System/Library/Extensions/exfat.kext/Contents/Info.plist").
    typedef std::multimap<lldb_private::ConstString, lldb_private::FileSpec> BundleIDToKextMap;
    typedef BundleIDToKextMap::iterator BundleIDToKextIterator;

    
    // Array of directories that were searched for kext bundles (used only for reporting to user)
    typedef std::vector<lldb_private::FileSpec> DirectoriesSearchedCollection;
    typedef DirectoriesSearchedCollection::iterator DirectoriesSearchedIterator;


    static lldb_private::FileSpec::EnumerateDirectoryResult
    GetKextDirectoriesInSDK (void *baton,
                             lldb_private::FileSpec::FileType file_type,
                             const lldb_private::FileSpec &file_spec);

    static lldb_private::FileSpec::EnumerateDirectoryResult 
    GetKextsInDirectory (void *baton,
                         lldb_private::FileSpec::FileType file_type,
                         const lldb_private::FileSpec &file_spec);

    void
    SearchForKexts();

    // Directories where we may find iOS SDKs with kext bundles in them
    void
    GetiOSSDKDirectoriesToSearch (std::vector<lldb_private::FileSpec> &directories);

    // Directories where we may find Mac OS X SDKs with kext bundles in them
    void
    GetMacSDKDirectoriesToSearch (std::vector<lldb_private::FileSpec> &directories);

    // Directories where we may find Mac OS X or iOS SDKs with kext bundles in them
    void
    GetGenericSDKDirectoriesToSearch (std::vector<lldb_private::FileSpec> &directories);

    // Directories where we may find iOS kext bundles
    void
    GetiOSDirectoriesToSearch (std::vector<lldb_private::FileSpec> &directories);

    // Directories where we may find MacOSX kext bundles
    void
    GetMacDirectoriesToSearch (std::vector<lldb_private::FileSpec> &directories);

    // Directories where we may find iOS or MacOSX kext bundles
    void
    GetGenericDirectoriesToSearch (std::vector<lldb_private::FileSpec> &directories);

    // Search through a vector of SDK FileSpecs, add any directories that may contain kexts
    // to the vector of kext dir FileSpecs
    void
    SearchSDKsForKextDirectories (std::vector<lldb_private::FileSpec> sdk_dirs, std::vector<lldb_private::FileSpec> &kext_dirs);

    // Search through all of the directories passed in, find all .kext bundles in those directories,
    // get the CFBundleIDs out of the Info.plists and add the bundle ID and kext path to m_name_to_kext_path_map.
    void
    IndexKextsInDirectories (std::vector<lldb_private::FileSpec> kext_dirs);

    lldb_private::Error
    ExamineKextForMatchingUUID (const lldb_private::FileSpec &kext_bundle_path, const lldb_private::UUID &uuid, lldb::ModuleSP &exe_module_sp);

private:

    BundleIDToKextMap m_name_to_kext_path_map; 
    DirectoriesSearchedCollection m_directories_searched;
    lldb_private::LazyBool m_ios_debug_session;

    DISALLOW_COPY_AND_ASSIGN (PlatformDarwinKernel);

};

#endif // __APPLE__

#endif  // liblldb_PlatformDarwinKernel_h_
