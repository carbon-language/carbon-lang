Experimental DynamoRIO-MSAN plugin (codename "MSanDR").
Supports Linux/x86_64 only.

Building:
  1. First, download and build DynamoRIO:
     (svn co https://dynamorio.googlecode.com/svn/trunk dr && \
      cd dr && mkdir build && cd build && \
      cmake -DDR_EXT_DRMGR_STATIC=ON -DDR_EXT_DRSYMS_STATIC=ON \
            -DDR_EXT_DRUTIL_STATIC=ON -DDR_EXT_DRWRAP_STATIC=ON .. && \
      make -j10 && make install)

  2. Download and build DrMemory (for DrSyscall extension)
     (svn co http://drmemory.googlecode.com/svn/trunk/ drmemory && \
      cd drmemory && mkdir build && cd build && \
      cmake -DDynamoRIO_DIR=`pwd`/../../dr/exports/cmake .. && \
      make -j10 && make install)

  NOTE: The line above will build a shared DrSyscall library in a non-standard
  location. This will require the use of LD_LIBRARY_PATH when running MSanDR.
  To build a static DrSyscall library (and link it into MSanDR), add
  -DDR_EXT_DRSYSCALL_STATIC=ON to the CMake invocation above, but
  beware: DrSyscall is LGPL.

  3. Now, build LLVM with two extra CMake flags:
       -DDynamoRIO_DIR=<path_to_dynamorio>/exports/cmake
       -DDrMemoryFramework_DIR=<path_to_drmemory>/exports64/drmf

  This will build a lib/clang/$VERSION/lib/linux/libclang_rt.msandr-x86_64.so

Running:
  <path_to_dynamorio>/exports/bin64/drrun -c lib/clang/$VERSION/lib/linux/libclang_rt.msandr-x86_64.so -- test_binary

MSan unit tests contain several tests for MSanDR (use MemorySanitizerDr.* gtest filter).
