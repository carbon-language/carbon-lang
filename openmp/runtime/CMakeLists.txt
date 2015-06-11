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

################
# CMAKE libomp
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

#########
# GLOBALS 
set(GLOBAL_DEBUG 0)

# Add cmake directory to search for custom cmake functions
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})

#######################################################################
# Standalone build or part of LLVM?
set(LIBOMP_STANDALONE_BUILD FALSE)
if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_CURRENT_SOURCE_DIR}" OR 
   "${CMAKE_SOURCE_DIR}/runtime" STREQUAL "${CMAKE_CURRENT_SOURCE_DIR}")
    project(libomp C CXX)
    set(LIBOMP_STANDALONE_BUILD TRUE)
endif()

# These include files are in cmake/ subdirectory except for FindPerl which is a cmake standard module
include(HelperFunctions)
include(Definitions) # -D definitions when compiling 
include(CommonFlags) # compiler, assembler, fortran, linker flags common for all compilers
include(SourceFiles) # source files to compile 
include(PerlFlags)   # Perl flags for generate-def.pl and expand-vars.pl
include(FindPerl) # Standard cmake module to check for Perl
include(GetArchitecture) # get_architecture()

####################################################################
#                         CONFIGURATION
#
# * Any variable/value that is CACHE-ed can be changed after the initial run of cmake
# through the file, CMakeCache.txt which is in the build directory.
# * If you change any value in CMakeCache.txt, then just run cmake ..
# and the changed will be picked up.  One can also use -DVARIABLE=VALUE
# when calling cmake to changed configuration values.  
# * CMAKE_C_COMPILER, CMAKE_CXX_COMPILER, CMAKE_ASM_MASM_COMPILER, CMAKE_ASM_COMPILER,
# CMAKE_Fortran_COMPILER can only by specified on the initial run of cmake. 
# This means you cannot specify -DCMAKE_C_COMPILER= on a subsequent run of cmake
# in the same build directory until that build directory is emptied.
# If you want to change the compiler, then empty the build directory and rerun cmake.

# Build Configuration
set(os_possible_values lin mac win)
set(arch_possible_values 32e 32 arm ppc64 ppc64le aarch64 mic)
set(build_type_possible_values release debug relwithdebinfo)
set(omp_version_possible_values 41 40 30)
set(lib_type_possible_values normal profile stubs)
set(mic_arch_possible_values knf knc)

# Below, cmake will try and determine the operating system and architecture for you.
# These values are set in CMakeCache.txt when cmake is first run (-Dvar_name=... will take precedence)
#  parameter  | default value             
# ----------------------------
# Right now, this build system considers os=lin to mean "Unix-like that is not MAC"
# Apple goes first because CMake considers Mac to be a Unix based 
# operating system, while libomp considers it a special case
if(${APPLE})
    set(temp_os mac)
elseif(${UNIX})
    set(temp_os lin)
elseif(${WIN32})       
    set(temp_os win)
else()
    set(temp_os lin)
endif()

# If adding a new architecture, take a look at cmake/GetArchitecture.cmake
get_architecture(detected_arch)

set(LIBOMP_OS ${temp_os} CACHE STRING
    "The operating system to build for (lin/mac/win)")
set(LIBOMP_ARCH ${detected_arch} CACHE STRING
    "The architecture to build for (32e/32/arm/ppc64/ppc64le/aarch64/mic).  32e is Intel(R) 64 architecture, 32 is IA-32 architecture")
set(LIBOMP_LIB_TYPE normal CACHE STRING
    "Performance,Profiling,Stubs library (normal/profile/stubs)")
set(LIBOMP_VERSION 5 CACHE STRING 
    "Produce libguide (version 4) or libomp (version 5)")
set(LIBOMP_OMP_VERSION 41 CACHE STRING 
    "The OpenMP version (41/40/30)")
set(LIBOMP_MIC_ARCH knc CACHE STRING 
    "Intel(R) Many Integrated Core Architecture (Intel(R) MIC Architecture) (knf/knc).  Ignored if not Intel(R) MIC Architecture build.")
set(LIBOMP_FORTRAN_MODULES false CACHE BOOL
    "Create Fortran module files? (requires fortran compiler)")

# - These tests are little tests performed after the library is formed.
# - The library won't be copied to the exports directory 
#   until it has passed/skipped all below tests
# - To skip these tests, just pass -DLIBOMP_MICRO_TESTS=OFF to cmake
set(LIBOMP_TEST_TOUCH true CACHE BOOL
    "Perform a small touch test?")
set(LIBOMP_TEST_RELO true CACHE BOOL
    "Perform a relocation test for dynamic libraries?")
set(LIBOMP_TEST_EXECSTACK true CACHE BOOL
    "Perform a execstack test for linux dynamic libraries?")
set(LIBOMP_TEST_INSTR true CACHE BOOL
    "Perform an instruction test for Intel(R) MIC Architecture libraries?")
set(LIBOMP_TEST_DEPS true CACHE BOOL
    "Perform a library dependency test?")
set(LIBOMP_MICRO_TESTS false CACHE BOOL
    "Perform touch, relo, execstack, instr, and deps tests?")

# - stats-gathering enables OpenMP stats where things like the number of 
# parallel regions, clock ticks spent in particular openmp regions are recorded.
set(LIBOMP_STATS false CACHE BOOL
    "Stats-Gathering functionality?")

# - Support for universal fat binary builds on Mac
# - Having this extra variable allows people to build this library as a universal library 
#   without forcing a universal build of the llvm/clang compiler.
set(LIBOMP_OSX_ARCHITECTURES "${CMAKE_OSX_ARCHITECTURES}" CACHE STRING
    "For Mac builds, semicolon separated list of architectures to build for universal fat binary.")
set(CMAKE_OSX_ARCHITECTURES ${LIBOMP_OSX_ARCHITECTURES})

# OMPT-support
set(LIBOMP_OMPT_SUPPORT false CACHE BOOL
    "OMPT-support?")
set(LIBOMP_OMPT_BLAME true CACHE BOOL
    "OMPT-blame?")
set(LIBOMP_OMPT_TRACE true CACHE BOOL
    "OMPT-trace?")

# User specified flags.  These are appended to the predetermined flags found 
# in CommonFlags.cmake and ${CMAKE_C_COMPILER_ID}/*Flags.cmake (e.g., GNU/CFlags.cmake)
set(LIBOMP_CFLAGS "" CACHE STRING
    "Appended user specified C compiler flags.")
set(LIBOMP_CXXFLAGS "" CACHE STRING
    "Appended user specified C++ compiler flags.")
set(LIBOMP_CPPFLAGS "" CACHE STRING
    "Appended user specified C preprocessor flags.")
set(LIBOMP_ASMFLAGS "" CACHE STRING
    "Appended user specified assembler flags.")
set(LIBOMP_LDFLAGS "" CACHE STRING
    "Appended user specified linker flags.")
set(LIBOMP_LIBFLAGS "" CACHE STRING
    "Appended user specified linked libs flags. (e.g., -lm)")
set(LIBOMP_FFLAGS "" CACHE STRING 
    "Appended user specified Fortran compiler flags.  These are only used if LIBOMP_FORTRAN_MODULES==true.")

# Should the libomp library and generated headers be copied into the original source exports/ directory
# Turning this to false aids parallel builds to not interfere with each other.
set(LIBOMP_COPY_EXPORTS true CACHE STRING
    "Should exports be copied into source exports/ directory?")

# - Allow three build types: Release, Debug, RelWithDebInfo (these relate to build.pl's release, debug, and diag settings respectively)
# - default is Release (when CMAKE_BUILD_TYPE is not defined)
# - CMAKE_BUILD_TYPE affects the -O and -g flags (CMake magically includes correct version of them on per compiler basis)
# - typical: Release = -O3 -DNDEBUG
#            RelWithDebInfo = -O2 -g -DNDEBUG
#            Debug = -g
if(CMAKE_BUILD_TYPE)
    # CMAKE_BUILD_TYPE was defined, check for validity 
    string(TOLOWER "${CMAKE_BUILD_TYPE}" cmake_build_type_lowercase)
    check_variable(cmake_build_type_lowercase  "${build_type_possible_values}")
else()
    # CMAKE_BUILD_TYPE was not defined, set default to Release
    unset(CMAKE_BUILD_TYPE CACHE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING
        "Choose the type of build, options are: Release/Debug/RelWithDebInfo")
    string(TOLOWER "${CMAKE_BUILD_TYPE}" cmake_build_type_lowercase)
    check_variable(cmake_build_type_lowercase  "${build_type_possible_values}")
endif()

if(${LIBOMP_STANDALONE_BUILD})
    # Allow user to choose a suffix for the installation directory, or if part of
    # LLVM build then just use LLVM_LIBDIR_SUFFIX
    set(LIBOMP_LIBDIR_SUFFIX "" CACHE STRING
        "suffix of lib installation directory e.g., 64 => lib64")
    # Should assertions be enabled?  They are on by default, or it part of
    # LLVM build then just use LLVM_ENABLE_ASSERTIONS
    set(LIBOMP_ENABLE_ASSERTIONS TRUE CACHE BOOL
        "enable assertions?")
else()
    set(LIBOMP_LIBDIR_SUFFIX ${LLVM_LIBDIR_SUFFIX})
    set(LIBOMP_ENABLE_ASSERTIONS ${LLVM_ENABLE_ASSERTIONS})
endif()

# Check valid values
check_variable(LIBOMP_OS "${os_possible_values}")
check_variable(LIBOMP_ARCH "${arch_possible_values}")
check_variable(LIBOMP_OMP_VERSION "${omp_version_possible_values}")
check_variable(LIBOMP_LIB_TYPE "${lib_type_possible_values}")
if("${LIBOMP_ARCH}" STREQUAL "mic")
    check_variable(LIBOMP_MIC_ARCH "${mic_arch_possible_values}")
endif()
# Get the build number from kmp_version.c
get_build_number("${CMAKE_CURRENT_SOURCE_DIR}" build_number)

# Getting time and date 
# As of now, no timestamp will be created.
set(date "No Timestamp")

#################################################################
# Set some useful flags variables for other parts of cmake to use
# Operating System
set(LINUX FALSE)
set(MAC FALSE)
set(WINDOWS FALSE)
set(MIC FALSE)
set(FREEBSD FALSE)
if("${LIBOMP_OS}" STREQUAL "lin")
    set(LINUX TRUE)
    set(real_os lin)
elseif("${LIBOMP_OS}" STREQUAL "mac")
    set(MAC TRUE)
    set(real_os mac)
elseif("${LIBOMP_OS}" STREQUAL "win")
    set(WINDOWS TRUE)
    set(real_os win)
endif()
if("${CMAKE_SYSTEM_NAME}" STREQUAL "FreeBSD")
    set(FREEBSD TRUE)
endif()

# Architecture
set(IA32 FALSE)
set(INTEL64 FALSE)
set(ARM FALSE)
set(AARCH64 FALSE)
set(PPC64BE FALSE)
set(PPC64LE FALSE)
set(PPC64 FALSE)
if("${LIBOMP_ARCH}" STREQUAL "32")      # IA-32 architecture
    set(IA32 TRUE)
elseif("${LIBOMP_ARCH}" STREQUAL "32e") # Intel(R) 64 architecture
    set(INTEL64 TRUE)
elseif("${LIBOMP_ARCH}" STREQUAL "arm") # ARM architecture
    set(ARM TRUE)
elseif("${LIBOMP_ARCH}" STREQUAL "ppc64") # PPC64BE architecture
    set(PPC64BE TRUE)
    set(PPC64 TRUE)
elseif("${LIBOMP_ARCH}" STREQUAL "ppc64le") # PPC64LE architecture
    set(PPC64LE TRUE)
    set(PPC64 TRUE)
elseif("${LIBOMP_ARCH}" STREQUAL "aarch64") # AARCH64 architecture
    set(AARCH64 TRUE)
elseif("${LIBOMP_ARCH}" STREQUAL "mic") # Intel(R) Many Integrated Core Architecture
    set(MIC TRUE)
endif()

# Set some flags based on build_type
# cmake_build_type_lowercase is based off of CMAKE_BUILD_TYPE, just put in lowercase.
set(RELEASE_BUILD FALSE)
set(DEBUG_BUILD FALSE)
set(RELWITHDEBINFO_BUILD FALSE)
if("${cmake_build_type_lowercase}" STREQUAL "release")
    set(RELEASE_BUILD TRUE)
elseif("${cmake_build_type_lowercase}" STREQUAL "debug")
    set(DEBUG_BUILD TRUE)
elseif("${cmake_build_type_lowercase}" STREQUAL "relwithdebinfo")
    set(RELWITHDEBINFO_BUILD TRUE)
endif()

# Include itt notify interface? Right now, always.
set(LIBOMP_USE_ITT_NOTIFY TRUE)

# normal, profile, stubs library.
set(NORMAL_LIBRARY FALSE)
set(STUBS_LIBRARY FALSE)
set(PROFILE_LIBRARY FALSE)
if("${LIBOMP_LIB_TYPE}" STREQUAL "normal")
    set(NORMAL_LIBRARY TRUE)
elseif("${LIBOMP_LIB_TYPE}" STREQUAL "profile")
    set(PROFILE_LIBRARY TRUE)
elseif("${LIBOMP_LIB_TYPE}" STREQUAL "stubs")
    set(STUBS_LIBRARY TRUE)
endif()

###############################################
# Features for compilation and build in general

# - Does the compiler support a 128-bit floating point data type? Default is false
# - If a compiler does, then change it in the CMakeCache.txt file (or using the cmake GUI)
#   or send to cmake -DCOMPILER_SUPPORTS_QUAD_PRECISION=true
# - If COMPILER_SUPPORTS_QUAD_PRECISION is true, then a corresponding COMPILER_QUAD_TYPE must be given
#   This is the compiler's quad-precision data type.
# ** TODO: This isn't complete yet. Finish it. Requires changing macros in kmp_os.h **
set(LIBOMP_COMPILER_SUPPORTS_QUAD_PRECISION false CACHE BOOL
    "*INCOMPLETE* Does the compiler support a 128-bit floating point type?")
set(LIBOMP_COMPILER_QUAD_TYPE "" CACHE STRING
    "*INCOMPLETE* The quad precision data type (e.g., for gcc, __float128)")

# - Should the orignal build rules for builds be used? (cmake/OriginalBuildRules.cmake).  This setting is off by default.
# - This always compiles with -g.  And if it is a release build, the debug info is stripped out via objcopy and put into libomp.dbg.
set(LIBOMP_USE_BUILDPL_RULES false CACHE BOOL
    "Should the build follow build.pl rules/recipes?")

# - Should the build use the predefined linker flags (OS-dependent) in CommonFlags.cmake?
# - these predefined linker flags should work for Windows, Mac, and True Linux for the most popular compilers/linkers
set(LIBOMP_USE_PREDEFINED_LINKER_FLAGS true CACHE BOOL
    "Should the build use the predefined linker flags in CommonFlags.cmake?")

# - On multinode systems, larger alignment is desired to avoid false sharing
set(LIBOMP_USE_INTERNODE_ALIGNMENT false CACHE BOOL
    "Should larger alignment (4096 bytes) be used for some locks and data structures?")

# - libgomp drop-in compatibility
if(${LINUX} AND NOT ${PPC64})
    set(LIBOMP_USE_VERSION_SYMBOLS true CACHE BOOL
        "Should version symbols be used? These provide binary compatibility with libgomp.")
else()
    set(LIBOMP_USE_VERSION_SYMBOLS false)
endif()

# - TSX based locks have __asm code which can be troublesome for some compilers.  This feature is also x86 specific.
if(${IA32} OR ${INTEL64})
    set(LIBOMP_USE_ADAPTIVE_LOCKS true CACHE BOOL
        "Should TSX-based lock be compiled (adaptive lock in kmp_lock.cpp).  These are x86 specific.")
else()
    set(LIBOMP_USE_ADAPTIVE_LOCKS false)
endif()

##################################
# Error checking the configuration 
if(${LIBOMP_STATS} AND (${WINDOWS} OR ${MAC}))
    error_say("Stats-gathering functionality is only supported on x86-Linux and Intel(R) MIC Architecture")
endif()
if(${LIBOMP_STATS} AND NOT (${IA32} OR ${INTEL64} OR ${MIC}))
    error_say("Stats-gathering functionality is only supported on x86-Linux and Intel(R) MIC Architecture")
endif()
if(${LIBOMP_USE_ADAPTIVE_LOCKS} AND NOT(${IA32} OR ${INTEL64}))
    error_say("Adaptive locks (TSX) functionality is only supported on x86 Architecture")
endif()

###############################################
# - Create the suffix for the export directory
# - Only add to suffix when not a default value
# - Example suffix: .deb.30.s1 
#   final export directory: exports/lin_32e.deb.30.s1/lib
# - These suffixes imply the build is a Debug, OpenMP 3.0, Stats-Gathering version of the library
if(NOT "${cmake_build_type_lowercase}" STREQUAL "release")
    string(SUBSTRING "${cmake_build_type_lowercase}" 0 3 build_type_suffix)
    set(suffix "${suffix}.${build_type_suffix}")
endif()
if(NOT "${LIBOMP_OMP_VERSION}" STREQUAL "41")
    set(suffix "${suffix}.${LIBOMP_OMP_VERSION}")
endif()
if(${LIBOMP_STATS})
    set(suffix "${suffix}.s1")
endif()
if(${LIBOMP_OMPT_SUPPORT})
    set(suffix "${suffix}.ompt")
    if(NOT ${LIBOMP_OMPT_BLAME})
        set(suffix "${suffix}.no-ompt-blame")
    endif()
    if(NOT ${LIBOMP_OMPT_TRACE})
        set(suffix "${suffix}.no-ompt-trace")
    endif()
endif()

####################################
# Setting file extensions / suffixes
set(obj ${CMAKE_C_OUTPUT_EXTENSION})
set(lib ${CMAKE_STATIC_LIBRARY_SUFFIX})
set(dll ${CMAKE_SHARED_LIBRARY_SUFFIX})
set(exe ${CMAKE_EXECUTABLE_SUFFIX})

######################
# Find perl executable
# Perl is used to create omp.h (and other headers) along with kmp_i18n_id.inc and kmp_i18n_default.inc (see below in Rules section)
if(NOT "${PERL_FOUND}") # variable is defined in FindPerl Standard CMake Module
    error_say("Error: Could not find valid perl")
endif()

#########################
# Setting directory names
if(${MIC})
    set(platform "${real_os}_${LIBOMP_MIC_ARCH}") # e.g., lin_knf, lin_knc
else()
    set(platform "${real_os}_${LIBOMP_ARCH}") # e.g., lin_32e, mac_32
endif()
# build directory (Where CMakeCache.txt is created, build files generated)
set(build_dir "${CMAKE_CURRENT_BINARY_DIR}")
set(src_dir "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(tools_dir "${CMAKE_CURRENT_SOURCE_DIR}/tools") 
set(export_dir "${CMAKE_CURRENT_SOURCE_DIR}/exports") 
set(export_ptf_dir "${export_dir}/${platform}${suffix}")
set(export_cmn_dir "${export_dir}/common${suffix}/include") 
set(export_inc_dir "${export_ptf_dir}/include") 
set(export_mod_dir "${export_ptf_dir}/include_compat")
_export_lib_dir(${platform} export_lib_dir)  # set exports directory (relative to build_dir) e.g., ../exports/lin_32e/lib/
                                             # or ../exports/mac_32e/lib.thin/ for mac
if(${MAC})
    # macs use lib.thin/ subdirectory for non-fat libraries that only contain one architecture
    # macs use lib/ subdirectory for fat libraries that contain both IA-32 architecture and Intel(R) 64 architecture code.
    _export_lib_fat_dir(${platform} export_lib_fat_dir)
endif()
set(inc_dir "${src_dir}/include/${LIBOMP_OMP_VERSION}")

############################
# Setting final library name
set(lib_item "libomp")
if(${PROFILE_LIBRARY})
    set(lib_item "${lib_item}prof")
endif()
if(${STUBS_LIBRARY})
    set(lib_item "${lib_item}stubs")
endif()
if(${WINDOWS})
    set(lib_item "${lib_item}md")
endif()
set(LIBOMP_LIB_NAME "${lib_item}" CACHE STRING "OMP library name")
set(lib_ext "${dll}")
# ${lib_file} is real library name:
# libomp.so    for Linux
# libomp.dylib for Mac
# libompmd.dll   for Windows
set(lib_file "${LIBOMP_LIB_NAME}${lib_ext}")

########################################
# Setting export file names
if(${WINDOWS})
    set(imp_file "${lib_item}${lib}") # this is exported (libomp.lib)
    set(def_file "${lib_item}.def") # this is not exported
    if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug" OR 
        "${CMAKE_BUILD_TYPE}" STREQUAL "RelWithDebInfo" OR 
        ${LIBOMP_USE_BUILDPL_RULES})
        set(pdb_file "${lib_file}.pdb") # this is exported if it exists (libompmd.dll.pdb)
    endif()
endif()
set(export_lib_files "${lib_file}" "${imp_file}" "${pdb_file}") 
set(export_mod_files "omp_lib.mod" "omp_lib_kinds.mod")
set(export_cmn_files "omp.h" "omp_lib.h" "omp_lib.f" "omp_lib.f90")

if(${LIBOMP_OMPT_SUPPORT})
    set(export_cmn_files ${export_cmn_files} "ompt.h")
endif()

if("${export_lib_fat_dir}")
    set(export_lib_fat_files "${lib_file}" "${imp_file}")
endif()

#########################
# Getting legal type/arch
set_legal_type(legal_type)
set_legal_arch(legal_arch)

#################################################
# Preprocessor Definitions (cmake/Definitions.cmake)
# Preprocessor Includes 
# Compiler (C/C++) Flags (cmake/CommonFlags.cmake)
# Assembler Flags (cmake/CommonFlags.cmake)
# Fortran   Flags (cmake/CommonFlags.cmake)
# Linker    Flags (cmake/CommonFlags.cmake)
# Archiver  Flags (cmake/CommonFlags.cmake)
# Helper Perl Script Flags (cmake/PerlFlags.cmake)
# * Inside the cmake/CommonFlags.cmake file, the LIBOMP_*FLAGS are added.
# * Cannot use CMAKE_*_FLAGS directly because -x c++ is put in the linker command and mangles the linking phase.

# Grab compiler-dependent flags
# Cmake will look for cmake/${CMAKE_C_COMPILER_ID}/CFlags.cmake to append additional c, cxx, and linker flags.
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/${CMAKE_C_COMPILER_ID} ${CMAKE_MODULE_PATH})
find_file(compiler_specific_include_file_found CFlags.cmake ${CMAKE_MODULE_PATH})
if(compiler_specific_include_file_found)
    include(CFlags) # COMPILER_SUPPORTS_QUAD_PRECISION changed in here
    append_compiler_specific_c_and_cxx_flags(C_FLAGS CXX_FLAGS)
    append_compiler_specific_linker_flags(LD_FLAGS LD_LIB_FLAGS)
else()
    warning_say("Could not find cmake/${CMAKE_C_COMPILER_ID}/CFlags.cmake: will only use default flags")
endif()
# Grab assembler-dependent flags
# CMake will look for cmake/${CMAKE_ASM_COMPILER_ID}/AsmFlags.cmake to append additional assembler flags.
if(${WINDOWS})
    # Windows based systems use CMAKE_ASM_MASM_COMPILER
    # The windows assembly files are in MASM format, and they require a tool that can handle MASM syntax (ml.exe or ml64.exe typically)
    enable_language(ASM_MASM) 
    set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/${CMAKE_ASM_MASM_COMPILER_ID} ${CMAKE_MODULE_PATH})
    find_file(assembler_specific_include_file_found AsmFlags.cmake ${CMAKE_MODULE_PATH})
    if(assembler_specific_include_file_found)
        include(AsmFlags)
        append_assembler_specific_asm_flags(ASM_FLAGS)
    else()
        warning_say("Could not find cmake/${CMAKE_ASM_MASM_COMPILER_ID}/AsmFlags.cmake: will only use default flags")
    endif()
else()
    # Unix (including Mac) based systems use CMAKE_ASM_COMPILER
    # Unix assembly files can be handled by compiler usually.
    find_file(assembler_specific_include_file_found AsmFlags.cmake ${CMAKE_MODULE_PATH})
    if(assembler_specific_include_file_found)
        include(AsmFlags)
        append_assembler_specific_asm_flags(ASM_FLAGS)
    else()
        warning_say("Could not find cmake/${CMAKE_ASM_COMPILER_ID}/AsmFlags.cmake: will only use default flags")
    endif()
endif()

# Grab all the compiler-independent flags
append_c_and_cxx_flags_common(C_FLAGS CXX_FLAGS) 
append_asm_flags_common(ASM_FLAGS)
append_fort_flags_common(F_FLAGS)
append_linker_flags_common(LD_FLAGS LD_LIB_FLAGS)
append_archiver_flags_common(AR_FLAGS)
append_cpp_flags(DEFINITIONS_FLAGS)

# Setup the flags correctly for cmake (covert to string)
# Pretty them up (STRIP any beginning and trailing whitespace)
list_to_string("${DEFINITIONS_FLAGS}" DEFINITIONS_FLAGS)
list_to_string("${C_FLAGS}" C_FLAGS)
list_to_string("${CXX_FLAGS}" CXX_FLAGS)
list_to_string("${ASM_FLAGS}" ASM_FLAGS)
list_to_string("${LD_FLAGS}" LD_FLAGS)
list_to_string("${LD_LIB_FLAGS}" LD_LIB_FLAGS)
# Windows specific for creating import library
list_to_string("${AR_FLAGS}" AR_FLAGS)
string(STRIP "${DEFINITIONS_FLAGS}" DEFINITIONS_FLAGS)
string(STRIP "${C_FLAGS}" C_FLAGS)
string(STRIP "${CXX_FLAGS}" CXX_FLAGS)
string(STRIP "${ASM_FLAGS}" ASM_FLAGS)
string(STRIP "${LD_FLAGS}" LD_FLAGS)
string(STRIP "${LD_LIB_FLAGS}" LD_LIB_FLAGS)
# Windows specific for creating import library
string(STRIP "${AR_FLAGS}" AR_FLAGS)

# Grab the Perl flags
set_ev_flags(ev_flags) # expand-vars.pl flags
set_gd_flags(gd_flags) # generate-def.pl flags (Windows only)
set(oa_opts "--os=${real_os}" "--arch=${LIBOMP_ARCH}") # sent to the perl scripts

#########################################################
# Getting correct source files (cmake/SourceFiles.cmake)
set_c_files(lib_c_items)
set_cpp_files(lib_cxx_items)
set_asm_files(lib_asm_items)
set_imp_c_files(imp_c_items) # Windows-specific
set(lib_src_files "${lib_c_items}" "${lib_cxx_items}" "${lib_asm_items}")
set(imp_src_files "${imp_c_items}")

#####################################################################
# Debug print outs.  Will print "variable = ${variable}" if GLOBAL_DEBUG == 1
if(GLOBAL_DEBUG)
    include(CMakePrintSystemInformation)
endif()
debug_say_var(CMAKE_ASM_COMPILE_OBJECT)
debug_say_var(CMAKE_RC_COMPILER)
debug_say_var(CMAKE_C_COMPILER_ID)
debug_say_var(date)
debug_say_var(LIBOMP_STATS)
debug_say_var(lib_file)
debug_say_var(export_lib_files)
debug_say_var(DEFINITIONS_FLAGS)
debug_say_var(C_FLAGS)
debug_say_var(CXX_FLAGS)
debug_say_var(ASM_FLAGS)
debug_say_var(F_FLAGS)
debug_say_var(LD_FLAGS)
debug_say_var(LD_LIB_FLAGS)
debug_say_var(AR_FLAGS)
debug_say_var(ev_flags)
debug_say_var(gd_flags)
debug_say_var(oa_opts)
debug_say_var(lib_c_items)
debug_say_var(lib_cxx_items)
debug_say_var(lib_asm_items)
debug_say_var(imp_c_items)
debug_say_var(lib_src_files)
debug_say_var(imp_src_files)

####################################################################
# Print configuration after all variables are set.
if(${LIBOMP_STANDALONE_BUILD})
    say("LIBOMP: Operating System     -- ${LIBOMP_OS}")
    say("LIBOMP: Target Architecture  -- ${LIBOMP_ARCH}")
    if(${MIC})
        say("LIBOMP: Intel(R) MIC Architecture    -- ${LIBOMP_MIC_ARCH}")
    endif()
    say("LIBOMP: Build Type           -- ${CMAKE_BUILD_TYPE}")
    say("LIBOMP: OpenMP Version       -- ${LIBOMP_OMP_VERSION}")
    say("LIBOMP: Lib Type             -- ${LIBOMP_LIB_TYPE}")
    say("LIBOMP: Fortran Modules      -- ${LIBOMP_FORTRAN_MODULES}")
    # will say development if all zeros
    if("${build_number}" STREQUAL "00000000")
        set(build "development")
    else()
        set(build "${build_number}")
    endif()
    say("LIBOMP: Build                -- ${build}")
    say("LIBOMP: Stats-Gathering      -- ${LIBOMP_STATS}")
    say("LIBOMP: OMPT-support         -- ${LIBOMP_OMPT_SUPPORT}")
    if(${LIBOMP_OMPT_SUPPORT})
        say("LIBOMP: OMPT-blame           -- ${LIBOMP_OMPT_BLAME}")
        say("LIBOMP: OMPT-trace           -- ${LIBOMP_OMPT_TRACE}")
    endif()
    say("LIBOMP: Use build.pl rules   -- ${LIBOMP_USE_BUILDPL_RULES}")
    say("LIBOMP: Adaptive locks       -- ${LIBOMP_USE_ADAPTIVE_LOCKS}")
    say("LIBOMP: Use predefined linker flags      -- ${LIBOMP_USE_PREDEFINED_LINKER_FLAGS}")
    say("LIBOMP: Compiler supports quad precision -- ${LIBOMP_COMPILER_SUPPORTS_QUAD_PRECISION}")
endif()

add_subdirectory(src)

