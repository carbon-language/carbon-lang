Design Of lib/System
====================

The software in this directory is designed to completely shield LLVM from any
and all operating system specific functionality. It is not intended to be a
complete operating system wrapper (such as ACE), but only to provide the
functionality necessary to support LLVM.

The software located here, of necessity, has very specific and stringent design
rules. Violation of these rules means that cracks in the shield could form and
the primary goal of the library is defeated. By consistently using this library,
LLVM becomes more easily ported to new platforms since (hopefully) the only thing
requiring porting is this library.

Complete documentation for the library can be found in the file:
  llvm/docs/SystemLibrary.html 
or at this URL:
  http://llvm.org/docs/SystemLibrary.html

However, for the impatient, here's a high level summary of the design rules:

1. No functions are declared with throw specifications. This is on purpose to 
   make sure that additional exception handling code is not introduced by the
   compiler.

2. On error only an instance of std::string that explains the error and possibly
   the context of the error may be thrown.

3. Error messages should do whatever is necessary to get a readable message from
   the operating system about the error. For example, on UNIX the strerror_r
   function ought to be used.

4. Entry points into the library should be fairly high level and aimed at
   completing some task needed by LLVM. There should *not* be a 1-to-1
   relationship between operating system calls and the library's interface.
   Certain implementations of the

5. The implementation of an lib/System interface can vary drastically between
   platforms. That's okay as long as the end result of the interface function is
   the same. For example, a function to create a directory is pretty straight
   forward on all operating system. System V IPC on the other hand isn't even
   supported on all platforms. Instead of "supporting" System V IPC, lib/System
   should provide an interface to the basic concept of inter-process 
   communications. The implementations might use System V IPC if that was
   available or named pipes, or whatever gets the job done effectively for a
   given operating system.

6. Implementations are separated first by the general class of operating system
   as provided by the configure script's $build variable. This variable is used
   to create a link from $BUILD_OBJ_ROOT/lib/System/platform to a directory in
   $BUILD_SRC_ROOT/lib/System directory with the same name as the $build
   variable. This provides a retargetable include mechanism. By using the link's
   name (platform) we can actually include the operating specific
   implementation. For example, support $build is "Darwin" for MacOS X. If we
   place:
     #include "platform/File.cpp"
   into a a file in lib/System, it will actually include
   lib/System/Darwin/File.cpp. What this does is quickly differentiate the basic
   class of operating system that will provide the implementation.

7. Implementation files in lib/System need may only do two things: (1) define 
   functions and data that is *TRULY* generic (completely platform agnostic) and
   (2) #include the platform specific implementation with:

      #include "platform/Impl.cpp"

   where Impl is the name of the implementation files.

8. Platform specific implementation files (platform/Impl.cpp) may only #include
   other Impl.cpp files found in directories under lib/System. The order of
   inclusion is very important (from most generic to most specific) so that we
   don't inadvertently place an implementation in the wrong place. For example,
   consider a fictitious implementation file named DoIt.cpp. Here's how the
   #includes should work for a Linux platform

   lib/System/DoIt.cpp
     #include "platform/DoIt.cpp"        // platform specific impl. of Doit
     DoIt

   lib/System/Linux/DoIt.cpp             // impl that works on all Linux 
     #include "../Unix/DoIt.cpp"         // generic Unix impl. of DoIt
     #include "../Unix/SUS/DoIt.cpp      // SUS specific impl. of DoIt
     #include "../Unix/SUS/v3/DoIt.cpp   // SUSv3 specific impl. of DoIt

   Note that the #includes in lib/System/Linux/DoIt.cpp are all optional but
   should be used where the implementation of some functionality can be shared
   across some set of Unix variants. We don't want to duplicate code across
   variants if their implementation could be shared.

9. The library does not attempt to shield LLVM from the C++ standard library or
   standard template library. These libraries are considered to be platform
   agnostic already.

10. LLVM should not include *any* system headers anywhere except in lib/System.

11. lib/System must *not* expose *any* system headers through its interface.
