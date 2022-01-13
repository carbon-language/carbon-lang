LLD - The LLVM Linker
=====================

LLD is a linker from the LLVM project that is a drop-in replacement
for system linkers and runs much faster than them. It also provides
features that are useful for toolchain developers.

The linker supports ELF (Unix), PE/COFF (Windows), Mach-O (macOS) and
WebAssembly in descending order of completeness. Internally, LLD consists of
several different linkers. The ELF port is the one that will be described in
this document. The PE/COFF port is complete, including
Windows debug info (PDB) support. The WebAssembly port is still a work in
progress (See :doc:`WebAssembly`).  The Mach-O port is built based on a
different architecture than the others. For the details about Mach-O, please
read :doc:`AtomLLD`.

Features
--------

- LLD is a drop-in replacement for the GNU linkers that accepts the
  same command line arguments and linker scripts as GNU.

  We are currently working closely with the FreeBSD project to make
  LLD default system linker in future versions of the operating
  system, so we are serious about addressing compatibility issues. As
  of February 2017, LLD is able to link the entire FreeBSD/amd64 base
  system including the kernel. With a few work-in-progress patches it
  can link approximately 95% of the ports collection on AMD64. For the
  details, see `FreeBSD quarterly status report
  <https://www.freebsd.org/news/status/report-2016-10-2016-12.html#Using-LLVM%27s-LLD-Linker-as-FreeBSD%27s-System-Linker>`_.

- LLD is very fast. When you link a large program on a multicore
  machine, you can expect that LLD runs more than twice as fast as the GNU
  gold linker. Your mileage may vary, though.

- It supports various CPUs/ABIs including AArch64, AMDGPU, ARM, Hexagon, MIPS
  32/64 big/little-endian, PowerPC, PowerPC64, RISC-V, SPARC V9, x86-32 and
  x86-64. Among these, AArch64, ARM (>= v6), PowerPC, PowerPC64, x86-32 and
  x86-64 have production quality. MIPS seems decent too.

- It is always a cross-linker, meaning that it always supports all the
  above targets however it was built. In fact, we don't provide a
  build-time option to enable/disable each target. This should make it
  easy to use our linker as part of a cross-compile toolchain.

- You can embed LLD in your program to eliminate dependencies on
  external linkers. All you have to do is to construct object files
  and command line arguments just like you would do to invoke an
  external linker and then call the linker's main function,
  ``lld::elf::link``, from your code.

- It is small. We are using LLVM libObject library to read from object
  files, so it is not a completely fair comparison, but as of February
  2017, LLD/ELF consists only of 21k lines of C++ code while GNU gold
  consists of 198k lines of C++ code.

- Link-time optimization (LTO) is supported by default. Essentially,
  all you have to do to do LTO is to pass the ``-flto`` option to clang.
  Then clang creates object files not in the native object file format
  but in LLVM bitcode format. LLD reads bitcode object files, compile
  them using LLVM and emit an output file. Because in this way LLD can
  see the entire program, it can do the whole program optimization.

- Some very old features for ancient Unix systems (pre-90s or even
  before that) have been removed. Some default settings have been
  tuned for the 21st century. For example, the stack is marked as
  non-executable by default to tighten security.

Performance
-----------

This is a link time comparison on a 2-socket 20-core 40-thread Xeon
E5-2680 2.80 GHz machine with an SSD drive. We ran gold and lld with
or without multi-threading support. To disable multi-threading, we
added ``-no-threads`` to the command lines.

============  ===========  ============  ====================  ==================  ===============  =============
Program       Output size  GNU ld        GNU gold w/o threads  GNU gold w/threads  lld w/o threads  lld w/threads
ffmpeg dbg    92 MiB       1.72s         1.16s                 1.01s               0.60s            0.35s
mysqld dbg    154 MiB      8.50s         2.96s                 2.68s               1.06s            0.68s
clang dbg     1.67 GiB     104.03s       34.18s                23.49s              14.82s           5.28s
chromium dbg  1.14 GiB     209.05s [1]_  64.70s                60.82s              27.60s           16.70s
============  ===========  ============  ====================  ==================  ===============  =============

As you can see, lld is significantly faster than GNU linkers.
Note that this is just a benchmark result of our environment.
Depending on number of available cores, available amount of memory or
disk latency/throughput, your results may vary.

.. [1] Since GNU ld doesn't support the ``-icf=all`` and
       ``-gdb-index`` options, we removed them from the command line
       for GNU ld. GNU ld would have been slower than this if it had
       these options.

Build
-----

If you have already checked out LLVM using SVN, you can check out LLD
under ``tools`` directory just like you probably did for clang. For the
details, see `Getting Started with the LLVM System
<https://llvm.org/docs/GettingStarted.html>`_.

If you haven't checked out LLVM, the easiest way to build LLD is to
check out the entire LLVM projects/sub-projects from a git mirror and
build that tree. You need `cmake` and of course a C++ compiler.

.. code-block:: console

  $ git clone https://github.com/llvm/llvm-project llvm-project
  $ mkdir build
  $ cd build
  $ cmake -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_PROJECTS=lld -DCMAKE_INSTALL_PREFIX=/usr/local ../llvm-project/llvm
  $ make install

Using LLD
---------

LLD is installed as ``ld.lld``. On Unix, linkers are invoked by
compiler drivers, so you are not expected to use that command
directly. There are a few ways to tell compiler drivers to use ld.lld
instead of the default linker.

The easiest way to do that is to overwrite the default linker. After
installing LLD to somewhere on your disk, you can create a symbolic
link by doing ``ln -s /path/to/ld.lld /usr/bin/ld`` so that
``/usr/bin/ld`` is resolved to LLD.

If you don't want to change the system setting, you can use clang's
``-fuse-ld`` option. In this way, you want to set ``-fuse-ld=lld`` to
LDFLAGS when building your programs.

LLD leaves its name and version number to a ``.comment`` section in an
output. If you are in doubt whether you are successfully using LLD or
not, run ``readelf --string-dump .comment <output-file>`` and examine the
output. If the string "Linker: LLD" is included in the output, you are
using LLD.

History
-------

Here is a brief project history of the ELF and COFF ports.

- May 2015: We decided to rewrite the COFF linker and did that.
  Noticed that the new linker is much faster than the MSVC linker.

- July 2015: The new ELF port was developed based on the COFF linker
  architecture.

- September 2015: The first patches to support MIPS and AArch64 landed.

- October 2015: Succeeded to self-host the ELF port. We have noticed
  that the linker was faster than the GNU linkers, but we weren't sure
  at the time if we would be able to keep the gap as we would add more
  features to the linker.

- July 2016: Started working on improving the linker script support.

- December 2016: Succeeded to build the entire FreeBSD base system
  including the kernel. We had widen the performance gap against the
  GNU linkers.

Internals
---------

For the internals of the linker, please read :doc:`NewLLD`. It is a bit
outdated but the fundamental concepts remain valid. We'll update the
document soon.

.. toctree::
   :maxdepth: 1

   NewLLD
   AtomLLD
   WebAssembly
   windows_support
   missingkeyfunction
   error_handling_script
   Partitions
   ReleaseNotes
   ELF/linker_script
   ELF/warn_backrefs
