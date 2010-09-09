//===-- RNBServices.cpp -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  Created by Christopher Friesen on 3/21/08.
//
//===----------------------------------------------------------------------===//

#import "RNBServices.h"

#import <CoreFoundation/CoreFoundation.h>
#import <unistd.h>
#import "DNBLog.h"
#include "MacOSX/CFUtils.h"

#if defined (__arm__)
#import <SpringBoardServices/SpringBoardServices.h>
#endif

int
ListApplications(std::string& plist, bool opt_runningApps, bool opt_debuggable)
{
#if defined (__arm__)
    int result = -1;

    CFAllocatorRef alloc = kCFAllocatorDefault;

    // Create a mutable array that we can populate. Specify zero so it can be of any size.
    CFReleaser<CFMutableArrayRef> plistMutableArray (::CFArrayCreateMutable (alloc, 0, &kCFTypeArrayCallBacks));

    CFReleaser<CFStringRef> sbsFrontAppID (::SBSCopyFrontmostApplicationDisplayIdentifier ());
    CFReleaser<CFArrayRef> sbsAppIDs (::SBSCopyApplicationDisplayIdentifiers (opt_runningApps, opt_debuggable));

    CFIndex count = ::CFArrayGetCount (sbsAppIDs.get());
    CFIndex i = 0;
    for (i = 0; i < count; i++)
    {
        CFStringRef displayIdentifier = (CFStringRef)::CFArrayGetValueAtIndex (sbsAppIDs.get(), i);

        // Create a new mutable dictionary for each application
        CFReleaser<CFMutableDictionaryRef> appInfoDict (::CFDictionaryCreateMutable (alloc, 0, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks));

        // Get the process id for the app (if there is one)
        pid_t pid = INVALID_NUB_PROCESS;
        if (::SBSProcessIDForDisplayIdentifier ((CFStringRef)displayIdentifier, &pid) == true)
        {
            CFReleaser<CFNumberRef> pidCFNumber (::CFNumberCreate (alloc,  kCFNumberSInt32Type, &pid));
            ::CFDictionarySetValue (appInfoDict.get(), DTSERVICES_APP_PID_KEY, pidCFNumber.get());
        }

        // Set the a boolean to indicate if this is the front most
        if (sbsFrontAppID.get() && displayIdentifier && (::CFStringCompare (sbsFrontAppID.get(), displayIdentifier, 0) == kCFCompareEqualTo))
            ::CFDictionarySetValue (appInfoDict.get(), DTSERVICES_APP_FRONTMOST_KEY, kCFBooleanTrue);
        else
            ::CFDictionarySetValue (appInfoDict.get(), DTSERVICES_APP_FRONTMOST_KEY, kCFBooleanFalse);


        CFReleaser<CFStringRef> executablePath (::SBSCopyExecutablePathForDisplayIdentifier (displayIdentifier));
        if (executablePath.get() != NULL)
        {
            ::CFDictionarySetValue (appInfoDict.get(), DTSERVICES_APP_PATH_KEY, executablePath.get());
        }

        CFReleaser<CFStringRef> iconImagePath (::SBSCopyIconImagePathForDisplayIdentifier (displayIdentifier)) ;
        if (iconImagePath.get() != NULL)
        {
            ::CFDictionarySetValue (appInfoDict.get(), DTSERVICES_APP_ICON_PATH_KEY, iconImagePath.get());
        }

        CFReleaser<CFStringRef> localizedDisplayName (::SBSCopyLocalizedApplicationNameForDisplayIdentifier (displayIdentifier));
        if (localizedDisplayName.get() != NULL)
        {
            ::CFDictionarySetValue (appInfoDict.get(), DTSERVICES_APP_DISPLAY_NAME_KEY, localizedDisplayName.get());
        }

        // Append the application info to the plist array
        ::CFArrayAppendValue (plistMutableArray.get(), appInfoDict.get());
    }

    CFReleaser<CFDataRef> plistData (::CFPropertyListCreateXMLData (alloc, plistMutableArray.get()));

    // write plist to service port
    if (plistData.get() != NULL)
    {
        CFIndex size = ::CFDataGetLength (plistData.get());
        const UInt8 *bytes = ::CFDataGetBytePtr (plistData.get());
        if (bytes != NULL && size > 0)
        {
            plist.assign((char *)bytes, size);
            return 0;   // Success
        }
        else
        {
            DNBLogError("empty application property list.");
            result = -2;
        }
    }
    else
    {
        DNBLogError("serializing task list.");
        result = -3;
    }

    return result;
#else
    // TODO: list all current processes
    DNBLogError("SBS doesn't support getting application list.");
    return -1;
#endif
}


bool
IsSBProcess (nub_process_t pid)
{
#if defined (__arm__)
    bool opt_runningApps = true;
    bool opt_debuggable = false;

    CFReleaser<CFArrayRef> sbsAppIDs (::SBSCopyApplicationDisplayIdentifiers (opt_runningApps, opt_debuggable));
    if (sbsAppIDs.get() != NULL)
    {
        CFIndex count = ::CFArrayGetCount (sbsAppIDs.get());
        CFIndex i = 0;
        for (i = 0; i < count; i++)
        {
            CFStringRef displayIdentifier = (CFStringRef)::CFArrayGetValueAtIndex (sbsAppIDs.get(), i);

            // Get the process id for the app (if there is one)
            pid_t sbs_pid = INVALID_NUB_PROCESS;
            if (::SBSProcessIDForDisplayIdentifier ((CFStringRef)displayIdentifier, &sbs_pid) == TRUE)
            {
                if (sbs_pid == pid)
                    return true;
            }
        }
    }
#endif
    return false;
}

