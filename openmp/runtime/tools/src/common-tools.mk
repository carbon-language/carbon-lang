# common-tools.mk #

#
#//===----------------------------------------------------------------------===//
#//
#//                     The LLVM Compiler Infrastructure
#//
#// This file is dual licensed under the MIT and the University of Illinois Open
#// Source Licenses. See LICENSE.txt for details.
#//
#//===----------------------------------------------------------------------===//
#

# --------------------------------------------------------------------------------------------------
# Dev tools and general options (like -fpic, -O2 or -g).
# --------------------------------------------------------------------------------------------------

# c       -- C compiler.
# cxx     -- C++ compiler.
# cpp     -- C preprocessor.
# fort    -- Fortran compiler.
# as      -- Assembler.
# ar      -- Librarian (static library maker).
# ld      -- Linker (dynamic library maker).
# *-out   -- Flag denoting output file. If space between flag and file name required, add explicit
#            space to variable, e. g.: "c-out = -o$(space)".
# *-flags -- Flags to appropriate program, e. g. c-flags -- flags for C compiler, etc.

# --- Common definitions ---

# Add current directory (it contains generated files).
# Note: It is important to specify current dir as "./" (not just "."). Otherwise Intel compiler
# on Windows* OS generates such a dependency: "kmp_runtime.obj: .\kmp_i18n.inc", and make complains
# "No rule to build .\kmp_i18n.inc". Using "./" solves the problem.
cpp-flags += -I ./
# For non-x86 architecture
ifeq "$(filter 32 32e 64 mic,$(arch))" ""
    cpp-flags += $(shell pkg-config --cflags libffi)
endif
# Add all VPATH directories to path for searching include files.
cpp-flags += $(foreach i,$(VPATH),-I $(i))


# Shouldn't this be being set from the command line somehow?
cpp-flags += -D USE_ITT_BUILD

ifeq "$(OPTIMIZATION)" "on"
    cpp-flags += -D NDEBUG
else
    cpp-flags += -D _DEBUG -D BUILD_DEBUG
    ifeq "$(os)" "win"
        # This is forced since VS2010 tool produces inconsistent directives
        # between objects, resulting in a link failure.
        cpp-flags += -D _ITERATOR_DEBUG_LEVEL=0
    endif
endif

# --- Linux* OS, Intel(R) Many Integrated Core Architecture and OS X* definitions ---

ifneq "$(filter lin mac,$(os))" ""
    # --- C/C++ ---
    ifeq "$(c)" ""
        c = icc
    endif
    # C++ compiler is a complement to C compiler.
    ifeq "$(c)" "icc"
        cxx = icpc
    endif
    ifeq "$(c)" "gcc"
        cxx = g++
    endif
    ifeq "$(c)" "clang"
        cxx = clang++
    endif
    # Output file flag.
    c-out   = -o$(space)
    cxx-out = -o$(space)
    # Compile only, no link.
    c-flags   += -c
    cxx-flags += -c
    # Generating dependecy file.
    c-flags-m   += -M -MG
    cxx-flags-m += -M -MG
    # Enable C99 language.
    ifneq "$(CPLUSPLUS)" "on"
        c-flags += -std=gnu99
    endif
    # Generate position-independent code (a must for shared objects).
    ifeq "$(LINK_TYPE)" "dyna"
        c-flags   += -fPIC
        cxx-flags += -fPIC
    endif
    # Emit debugging information.
    ifeq "$(DEBUG_INFO)" "on"
        c-flags   += -g
        cxx-flags += -g
    endif
    # Instrument program for profiling, gather extra information.
    ifeq "$(COVERAGE)" "on"
        ifeq "$(c)" "icc"
            c-flags   += -prof_genx
        endif
        ifeq "$(cxx)" "icpc"
            cxx-flags += -prof_genx
        endif
    endif
    # Turn optimization on or off.
    ifeq "$(OPTIMIZATION)" "on"
        # -inline-min-size=1 improves performance of PARALLEL EPCC up to 10% on fxi64lin01,
        # doesn't change performance on fxe64lin01.
        # Presence of the -inline-min-size=1 switch should only help
        # to promote performance stability between changes,
        # even if it has no observable impact right now.
	ifneq "$(filter icl icl.exe,$(c))" ""
            c-flags   += -O2 -inline-min-size=1
	else
            c-flags   += -O2
	endif
        ifneq "$(filter icl icl.exe,$(cxx))" ""
            cxx-flags += -O2 -inline-min-size=1
	else
            cxx-flags += -O2
	endif
    else
        c-flags   += -O0
        cxx-flags += -O0
    endif
    # --- Assembler ---
    ifeq "$(c)" "icc"
        as        = icc
    endif
    ifeq "$(c)" "gcc"
        as        = gcc
    endif
    ifeq "$(c)" "clang"
        as        = clang
    endif
    as-out    = -o$(space)
    as-flags += $(cpp-flags)
    # Compile only, no link.
    as-flags += -c
    as-flags += -x assembler-with-cpp
    # --- Fortran ---
    ifeq "$(c)" "icc"
        fort = ifort
    endif
    ifeq "$(c)" "gcc"
        fort = gfortran
    endif
    ifeq "$(c)" "clang"
        fort = gfortran
    endif
    ifeq "$(fort)" ""
        fort = ifort
    endif
    fort-out    = -o$(space)
    fort-flags += -c
endif

# --- Linux* OS definitions ---

ifeq "$(os)" "lin"
ifneq "$(arch)" "mic"
    # --- C/C++ ---
    # On lin_32, we want to maintain stack alignment to be conpatible with GNU binaries built with
    # compiler.
    ifeq "$(c)" "icc"
        ifeq "$(arch)" "32"
            c-flags   += -falign-stack=maintain-16-byte
            cxx-flags += -falign-stack=maintain-16-byte
        endif
        # Generate code that will run on any Pentium or later processor.
        ifeq "$(arch)" "32"
            c-flags   += -mia32
            cxx-flags += -mia32
        endif
    endif
    ifeq "$(c)" "gcc"
        ifeq "$(arch)" "arm"
            c-flags   += -marm
        endif
    endif
    # --- Librarian ---
    ar        = ar
    ar-out    = $(empty)
    ar-flags += cr
    # --- Linker ---
    # Use ld by default, however, makefile may specify ld=$(c) before including devtools.mk.
    ifeq "$(ld)" ""
        ld = $(c)
    endif
    ld-flags-dll += -shared
    ifeq "$(ld)" "ld"
        ld-out = -o$(space)
        ifeq "$(arch)" "32"
            ld-flags += -m elf_i386
        endif
        ifeq "$(arch)" "32e"
            ld-flags += -m elf_x86_64
        endif
        ld-flags     += -x -lc -ldl
        ld-flags     += -z noexecstack
        ld-flags-dll += -soname=$(@F)
    endif
    ifeq "$(ld)" "$(c)"
        ld-out    = $(c-out)
        ld-flags += -Wl,-z,noexecstack
        ld-flags-dll += -Wl,-soname=$(@F)
    endif
    ifeq "$(ld)" "$(cxx)"
        ld-out    = $(cxx-out)
        ld-flags += -Wl,-z,noexecstack
        ld-flags-dll += -Wl,-soname=$(@F)
    endif
endif
endif

# --- Intel(R) Many Integrated Core Architecture definitions ---

ifeq "$(arch)" "mic"
    # --- C/C++ ---
    # Intel(R) Many Integrated Core Architecture specific options, need clarification for purpose:
    #c-flags     += -mmic -mP2OPT_intrin_disable_name=memcpy -mP2OPT_intrin_disable_name=memset -mGLOB_freestanding -mGLOB_nonstandard_lib -nostdlib -fno-builtin
    #cxx-flags   += -mmic -mP2OPT_intrin_disable_name=memcpy -mP2OPT_intrin_disable_name=memset -mGLOB_freestanding -mGLOB_nonstandard_lib -nostdlib -fno-builtin
    # icc for mic has a bug: it generates dependencies for target like file.obj, while real object
    # files are named file.o. -MT is a workaround for the problem.
    c-flags-m   += -MT $(basename $@).o
    cxx-flags-m += -MT $(basename $@).o
    # --- Librarian ---
    ar        = ar
    ar-out    = $(empty)
    ar-flags += cr
    # --- Linker ---
    # Use $(c) by default, however, makefile may specify another linker (e.g. ld=ld) before including devtools.mk.
    ifeq "$(ld)" ""
        ld = $(c)
    endif
    ifeq "$(ld)" "ld"
        ld-out   = -o$(space)
        ld-flags += -m elf_l1om_fbsd
        ld-flags-dll += -shared -x -lc
        ld-flags-dll += -soname=$(@F)
        # Now find out path to libraries.
            ld-flags-L := $(shell $(c) -Wl,-v -\# 2>&1 | grep -e "-L")
            $(call debug,ld-flags-L)
            # Remove continuation characters; first add a space to the end (" -Lpath1 /" -> "-Lpath1 / ")
            ld-flags-L := $(filter-out \,$(ld-flags-L))
            $(call debug,ld-flags-L)
            # Linker treats backslash ('\') as an escape symbol, so replace it with forward slash.
            ld-flags-L := $(subst \,/,$(ld-flags-L))
            $(call debug,ld-flags-L)
        ld-flags += $(ld-flags-L)
    endif
    ifeq "$(ld)" "$(c)"
        ld-out        = $(c-out)
        ld-flags-dll += -shared -Wl,-x -Wl,-soname=$(@F)
    endif
    ifeq "$(ld)" "$(cxx)"
        ld-out        = $(cxx-out)
        ld-flags-dll += -shared -Wl,-x -Wl,-soname=$(@F)
    endif
endif

# --- OS X* definitions ---

ifeq "$(os)" "mac"
    # --- Librarian ---
    ar        = libtool
    ar-out    = -o$(space)
    ar-flags += -static
    # --- Linker ---
    # Use C compiler as linker by default, however, makefile may specify ld=$(libtool) before
    # including devtools.mk.
    ifeq "$(ld)" ""
        ld = $(c)
    endif
    ifeq "$(ld)" "libtool"
        ld-out        = -o$(space)
        ld-flags-dll += -dynamic
        ld-flags     += -lc -ldl
    endif
    ifeq "$(ld)" "$(c)"
        ld-out        = $(c-out)
        ld-flags-dll += -dynamiclib
    endif
    ifeq "$(ld)" "$(cxx)"
        ld-out        = $(cxx-out)
        ld-flags-dll += -dynamiclib
    endif
    # These options suitable for any linker, either C compiler or libtool.
    ld-flags-dll += -headerpad_max_install_names
    ld-flags-dll += -install_name $(@F)
endif

# --- Windows* OS definitions ---

ifeq "$(os)" "win"
    # Disable warning "function "..." (declared at line ... of ...) was declared deprecated...".
    cpp-flags += -D_CRT_SECURE_NO_WARNINGS -D_CRT_SECURE_NO_DEPRECATE
    # --- C/C++ ---
    ifeq "$(c)" ""
        c = icl.exe
    endif
    cxx    = $(c)
    # Often default icl.cfg file in compiler bin/ directory contains options -Qvc and
    # -Qlocation,link. Setting ICLCFG (and IFORTCFG) to specially prepared empty config file
    # overrides default config.
    ICLCFG   = $(tools_dir)icc.cfg
    IFORTCFG = $(tools_dir)icc.cfg
    export ICLCFG
    export IFORTCFG
    # Output file.
    c-out   = -o$(space)
    cxx-out = -o$(space)
    # Disable annoying compiler logo.
    c-flags   += -nologo
    cxx-flags += -nologo
    # Generate code that will run on any Pentium or later processor.
    ifeq "$(arch)" "32"
        c-flags   += -arch:ia32
        cxx-flags += -arch:ia32
    endif
    # Compile only, no link.
    c-flags   += -c
    cxx-flags += -c
    # -QM  -- Generate dependency file.
    # -QMM -- do not include system headers. On Windows* OS, system headers may be located in
    #         "C:\Program Files\...", but path with space confuses make, so we exclude system
    #         headers.
    # -QMG -- Treat missed headers as generated. We do have some generated include files.
    c-flags-m   += -QM -QMM -QMG
    cxx-flags-m += -QM -QMM -QMG
    # Enable C99 language.
    ifneq "$(CPLUSPLUS)" "on"
    	c-flags   += -Qstd=gnu99
    endif
    # Enable C++ exception handling.
    # ??? Why we disable it on Linux* OS?
    cxx-flags += -EHsc
    ifeq "$(arch)" "32"
        ifneq "$(filter icl icl.exe,$(c))" ""
            c-flags   += -Qsafeseh
        endif
        ifneq "$(filter icl icl.exe,$(cxx))" ""
            cxx-flags += -Qsafeseh
        endif
    endif
    # Emit debugging information.
    ifeq "$(DEBUG_INFO)" "on"
        c-flags   += -Zi
        cxx-flags += -Zi
    endif
    # Instrument program for profiling, gather extra information.
    ifeq "$(COVERAGE)" "on"
        c-flags   += -Qprof_genx
        cxx-flags += -Qprof_genx
    endif
    # Turn optimization on or off.
    ifeq "$(OPTIMIZATION)" "on"
        # Presence of the -inline-min-size=1 switch should only help
        # to promote performance stability between changes,
        # even if it has no observable impact right now.
        # See the Linux* OS section above.
	ifneq "$(filter icl icl.exe,$(c))" ""
	    c-flags   += -O2 -Qinline-min-size=1
	else
	    c-flags   += -O2
	endif 
        ifneq "$(filter icl icl.exe,$(cxx))" ""
            cxx-flags += -O2 -Qinline-min-size=1
	else
            cxx-flags += -O2
	endif
    else
        c-flags   += -Od
        cxx-flags += -Od
        # Enable stack frame runtime error checking.
        # !!! 0Obsolete option. Should use /RTC instead.
        c-flags   += -RTC1
        cxx-flags += -RTC1
    endif
    # SDL (Security Development Lifecycle) flags:
    #   GS - Stack-based Buffer Overrun Detection
    #   DynamicBase - Image Randomization
    c-flags   += -GS -DynamicBase  
    cxx-flags += -GS -DynamicBase  
    # --- Assembler ---
    ifeq "$(arch)" "32"
        as   = ml
    endif
    ifeq "$(arch)" "32e"
        as   = ml64
    endif
    ifeq "$(as)" "ias"
        as-out   = -o$(space)
    endif
    ifneq "$(filter ml ml64,$(as))" ""
        as-out   = -Fo
        as-flags += -nologo -c
        # SDL (Security Development Lifecycle) flags:
        #   DynamicBase - Image Randomization
	as-flags += -DynamicBase 
    endif
    # --- Fortran ---
    fort        = ifort
    fort-out    = -o$(space)
    fort-flags += -nologo
    fort-flags += -c
    # SDL (Security Development Lifecycle) flags:
    #   GS - Stack-based Buffer Overrun Detection
    #   DynamicBase - Image Randomization
    fort-flags += -GS -DynamicBase 
    # --- Librarian ---
    ar     = link.exe
    ar-out = -out:
    # Generate static library. Must be the first option.
    ar-flags += -lib
    # Turn off tool banner.
    ar-flags += -nologo
    # --- Linker ---
    ld       = link.exe
    ld-out   = -out:
    # Generate dynamic library.
    ld-flags-dll += -dll
    # Turn off tool banner.
    ld-flags += -nologo
    # Generate pdb (Program DataBase, debug information) file.
    # If DEBUG_INFO is on, generate normal (full-featured) pdb file. Otherwise, we need only
    # stripped pdb. But stripped pdb cannot be generated alone, we have to generate normal *and*
    # stripped pdb. After generating both pdb files we rename stripped pdb to normal pdb name (see
    # rules.mk).
    ifeq "$(DEBUG_INFO)" "on"
        ld-flags += $(if $(pdb_file),-debug -pdb:$(pdb_file))
    else
        ld-flags += $(if $(pdb_file),-debug -pdb:$(pdb_file) -pdbstripped:$(pdb_file).stripped)
    endif
    # Use def file, if $(def_file) is specified.
    ld-flags += $(if $(def_file),-def:$(def_file))
    # Generate import library, if $(imp_file) is specified.
    ld-flags += $(if $(imp_file),-implib:$(imp_file))
    # Specify architecture.
    ifeq "$(arch)" "32"
        ar-flags += -machine:i386
        ld-flags += -machine:i386
    endif
    ifeq "$(arch)" "32e"
        ar-flags += -machine:amd64
        ld-flags += -machine:amd64
    endif
    # SAFESEH
    ifeq "$(arch)" "32"
        as-flags += -safeseh
        ld-flags += -safeseh
    endif
    # SDL (Security Development Lifecycle) flags:
    #   NXCompat - Data Execution Prevention
    ld-flags += -NXCompat -DynamicBase
endif

# end of file #
