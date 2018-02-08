/* This generated file is for internal use. Do not include it from headers. */

#ifdef FLANG_CONFIG_H
#error config.h can only be included once
#else
#define FLANG_CONFIG_H

/* Bug report URL. */
#define BUG_REPORT_URL "${BUG_REPORT_URL}"

/* Default linker to use. */
#define FLANG_DEFAULT_LINKER "${FLANG_DEFAULT_LINKER}"

/* Default C++ stdlib to use. */
#define FLANG_DEFAULT_CXX_STDLIB "${FLANG_DEFAULT_CXX_STDLIB}"

/* Default runtime library to use. */
#define FLANG_DEFAULT_RTLIB "${FLANG_DEFAULT_RTLIB}"

/* Default OpenMP runtime used by -fopenmp. */
#define FLANG_DEFAULT_OPENMP_RUNTIME "${FLANG_DEFAULT_OPENMP_RUNTIME}"

/* Multilib suffix for libdir. */
#define FLANG_LIBDIR_SUFFIX "${FLANG_LIBDIR_SUFFIX}"

/* Relative directory for resource files */
#define FLANG_RESOURCE_DIR "${FLANG_RESOURCE_DIR}"

/* Directories clang will search for headers */
#define C_INCLUDE_DIRS "${C_INCLUDE_DIRS}"

/* Default <path> to all compiler invocations for --sysroot=<path>. */
#define DEFAULT_SYSROOT "${DEFAULT_SYSROOT}"

/* Directory where gcc is installed. */
#define GCC_INSTALL_PREFIX "${GCC_INSTALL_PREFIX}"

/* Define if we have libxml2 */
#cmakedefine FLANG_HAVE_LIBXML ${FLANG_HAVE_LIBXML}

/* Define if we have z3 and want to build it */
#cmakedefine FLANG_ANALYZER_WITH_Z3 ${FLANG_ANALYZER_WITH_Z3}

/* Define if we have sys/resource.h (rlimits) */
#cmakedefine FLANG_HAVE_RLIMITS ${FLANG_HAVE_RLIMITS}

/* The LLVM product name and version */
#define BACKEND_PACKAGE_STRING "${BACKEND_PACKAGE_STRING}"

/* Linker version detected at compile time. */
#cmakedefine HOST_LINK_VERSION "${HOST_LINK_VERSION}"

/* pass --build-id to ld */
#cmakedefine ENABLE_LINKER_BUILD_ID

/* enable x86 relax relocations by default */
#cmakedefine01 ENABLE_X86_RELAX_RELOCATIONS

/* Enable each functionality of modules */
#cmakedefine FLANG_ENABLE_ARCMT
#cmakedefine FLANG_ENABLE_OBJC_REWRITER
#cmakedefine FLANG_ENABLE_STATIC_ANALYZER

#endif
