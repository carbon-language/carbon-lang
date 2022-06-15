/* This generated file is for internal use. Do not include it from headers. */

#ifdef CLANG_CONFIG_H
#error config.h can only be included once
#else
#define CLANG_CONFIG_H

/* Bug report URL. */
#define BUG_REPORT_URL "${BUG_REPORT_URL}"

/* Default to -fPIE and -pie on Linux. */
#cmakedefine01 CLANG_DEFAULT_PIE_ON_LINUX

/* Default linker to use. */
#define CLANG_DEFAULT_LINKER "${CLANG_DEFAULT_LINKER}"

/* Default C/ObjC standard to use. */
#cmakedefine CLANG_DEFAULT_STD_C LangStandard::lang_${CLANG_DEFAULT_STD_C}
/* Always #define something so that missing the config.h #include at use sites
 * becomes a compile error.
 */
#ifndef CLANG_DEFAULT_STD_C
#define CLANG_DEFAULT_STD_C LangStandard::lang_unspecified
#endif

/* Default C++/ObjC++ standard to use. */
#cmakedefine CLANG_DEFAULT_STD_CXX LangStandard::lang_${CLANG_DEFAULT_STD_CXX}
/* Always #define something so that missing the config.h #include at use sites
 * becomes a compile error.
 */
#ifndef CLANG_DEFAULT_STD_CXX
#define CLANG_DEFAULT_STD_CXX LangStandard::lang_unspecified
#endif

/* Default C++ stdlib to use. */
#define CLANG_DEFAULT_CXX_STDLIB "${CLANG_DEFAULT_CXX_STDLIB}"

/* Default runtime library to use. */
#define CLANG_DEFAULT_RTLIB "${CLANG_DEFAULT_RTLIB}"

/* Default unwind library to use. */
#define CLANG_DEFAULT_UNWINDLIB "${CLANG_DEFAULT_UNWINDLIB}"

/* Default objcopy to use */
#define CLANG_DEFAULT_OBJCOPY "${CLANG_DEFAULT_OBJCOPY}"

/* Default OpenMP runtime used by -fopenmp. */
#define CLANG_DEFAULT_OPENMP_RUNTIME "${CLANG_DEFAULT_OPENMP_RUNTIME}"

/* Default architecture for OpenMP offloading to Nvidia GPUs. */
#define CLANG_OPENMP_NVPTX_DEFAULT_ARCH "${CLANG_OPENMP_NVPTX_DEFAULT_ARCH}"

/* Default architecture for SystemZ. */
#define CLANG_SYSTEMZ_DEFAULT_ARCH "${CLANG_SYSTEMZ_DEFAULT_ARCH}"

/* Multilib suffix for libdir. */
#define CLANG_LIBDIR_SUFFIX "${CLANG_LIBDIR_SUFFIX}"

/* Relative directory for resource files */
#define CLANG_RESOURCE_DIR "${CLANG_RESOURCE_DIR}"

/* Directories clang will search for headers */
#define C_INCLUDE_DIRS "${C_INCLUDE_DIRS}"

/* Directories clang will search for configuration files */
#cmakedefine CLANG_CONFIG_FILE_SYSTEM_DIR "${CLANG_CONFIG_FILE_SYSTEM_DIR}"
#cmakedefine CLANG_CONFIG_FILE_USER_DIR "${CLANG_CONFIG_FILE_USER_DIR}"

/* Default <path> to all compiler invocations for --sysroot=<path>. */
#define DEFAULT_SYSROOT "${DEFAULT_SYSROOT}"

/* Directory where gcc is installed. */
#define GCC_INSTALL_PREFIX "${GCC_INSTALL_PREFIX}"

/* Define if we have libxml2 */
#cmakedefine CLANG_HAVE_LIBXML ${CLANG_HAVE_LIBXML}

/* Define if we have sys/resource.h (rlimits) */
#cmakedefine CLANG_HAVE_RLIMITS ${CLANG_HAVE_RLIMITS}

/* The LLVM product name and version */
#define BACKEND_PACKAGE_STRING "${BACKEND_PACKAGE_STRING}"

/* Linker version detected at compile time. */
#cmakedefine HOST_LINK_VERSION "${HOST_LINK_VERSION}"

/* pass --build-id to ld */
#cmakedefine ENABLE_LINKER_BUILD_ID

/* enable x86 relax relocations by default */
#cmakedefine01 ENABLE_X86_RELAX_RELOCATIONS

/* Enable IEEE binary128 as default long double format on PowerPC Linux. */
#cmakedefine01 PPC_LINUX_DEFAULT_IEEELONGDOUBLE

/* Enable each functionality of modules */
#cmakedefine01 CLANG_ENABLE_ARCMT
#cmakedefine01 CLANG_ENABLE_OBJC_REWRITER
#cmakedefine01 CLANG_ENABLE_STATIC_ANALYZER

/* Spawn a new process clang.exe for the CC1 tool invocation, when necessary */
#cmakedefine01 CLANG_SPAWN_CC1

/* Whether to enable opaque pointers by default */
#cmakedefine01 CLANG_ENABLE_OPAQUE_POINTERS_INTERNAL

#endif
