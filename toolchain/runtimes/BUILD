# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@rules_python//python:defs.bzl", "py_binary")
load(
    "//toolchain/base:runtimes_build_info.bzl",
    "BUILTINS_SRCS_FILEGROUPS",
    "BUILTINS_TEXTUAL_SRCS_FILEGROUPS",
    "CRT_FILES",
)
load("configure_cmake_file.bzl", "configure_cmake_file")

package(default_visibility = ["//visibility:public"])

exports_files(["carbon_runtimes.bzl"])

# Collect the runtime sources that are collectively installed into the
# `builtins` directory.
filegroup(
    name = "clang_builtins_runtimes",
    srcs = (
        CRT_FILES.values() +
        BUILTINS_SRCS_FILEGROUPS +
        BUILTINS_TEXTUAL_SRCS_FILEGROUPS
    ),
)

py_binary(
    name = "configure_cmake_file_impl",
    srcs = ["configure_cmake_file_impl.py"],
)

configure_cmake_file(
    name = "libcxx_site_config_gen",
    src = "@llvm-project//libcxx:include/__config_site.in",
    out = "staging_libcxx/include/__config_site",
    defines = {
        # We can inject custom logic at the end of the site configuration with
        # the ABI defines. This can custom set (or re-set) any of the relevant
        # configuration defines. Note that while this is sorted here in the
        # BUILD file, it is expanded at the _end_ of the configuration header
        # and so overrides the other configuration settings.
        #
        # TODO: This is a lot of C++ code to embed into a BUILD file. Even
        # though it moves it farther from the interacting CMake defines, we
        # should look at factoring this into a header that is included.
        "_LIBCPP_ABI_DEFINES": "\n".join([
            # We want to install a single header that works in all build modes,
            # so we define the ABI namespace based on how the header is used
            # rather than a fixed one. However, we only support use with Clang
            # and so we assume `__has_feature` is available and works.
            #
            # Note that generally, we don't rely on different ABI namespaces for
            # functionality -- the distinction is more to make errors when
            # linking with the wrong build of the standard library obvious and
            # immediate. We only can achieve this for sanitizers that have a
            # preprocessor detectable model.
            "#if __has_feature(address_sanitizer)",
            "# undef _LIBCPP_ABI_NAMESPACE",
            "# define _LIBCPP_ABI_NAMESPACE __asan",
            # Also mark that libc++ will be instrumented.
            "# undef _LIBCPP_INSTRUMENTED_WITH_ASAN",
            "# define _LIBCPP_INSTRUMENTED_WITH_ASAN 1",
            "#elif __has_feature(memory_sanitizer)",
            # TODO: If a track-origins macro becomes available, we should
            # distinguish that case, too.
            "# undef _LIBCPP_ABI_NAMESPACE",
            "# define _LIBCPP_ABI_NAMESPACE __msan",
            "#elif __has_feature(thread_sanitizer)",
            "# undef _LIBCPP_ABI_NAMESPACE",
            "# define _LIBCPP_ABI_NAMESPACE __tsan",
            "#elif __has_feature(cfi_sanitizer)",
            "# undef _LIBCPP_ABI_NAMESPACE",
            "# define _LIBCPP_ABI_NAMESPACE __cfi",
            "#endif",
            "",

            # Establish a default hardening mode where possible.
            "#ifndef _LIBCPP_HARDENING_MODE",
            "# ifndef NDEBUG",
            # !NDEBUG has significant overhead anyway and is explicitly a
            # debugging build rather than a production build.
            "#  define _LIBCPP_HARDENING_MODE _LIBCPP_HARDENING_MODE_DEBUG",
            "# else",
            # Default to the fast hardening checks.
            "#  define _LIBCPP_HARDENING_MODE _LIBCPP_HARDENING_MODE_FAST",
            "# endif",
            "#endif",
            "",

            # CUDA can't call any existing abort implementations, so disable
            # hardening there.
            "#ifdef __CUDA__",
            "# undef _LIBCPP_HARDENING_MODE",
            "# define _LIBCPP_HARDENING_MODE _LIBCPP_HARDENING_MODE_NONE",
            "#endif",
            "",

            # Setup platform-dependent features using preprocessor logic.
            "#ifdef __linux__",
            "# undef _LIBCPP_HAS_TIME_ZONE_DATABASE",
            "# define _LIBCPP_HAS_TIME_ZONE_DATABASE 1",
            "#endif",
            "",

            # Mixing translation units compiled with different versions of
            # libc++ is unsupported. Disable ABI tags to decrease symbol
            # lengths.
            "#define _LIBCPP_NO_ABI_TAG",
        ]),

        # No forced ABI.
        "_LIBCPP_ABI_FORCE_ITANIUM": "OFF",
        "_LIBCPP_ABI_FORCE_MICROSOFT": "OFF",

        # We use the unstable ABI and define a custom, Carbon-specific ABI
        # namespace. This also matches the mangling prefix used for Carbon
        # symbols.
        "_LIBCPP_ABI_NAMESPACE": "_C",
        # TODO: Fix the need to define _LIBCPP_ABI_VERSION when the unstable
        # ABI is selected.
        "_LIBCPP_ABI_VERSION": "999",

        # Follow hardening mode for the assertion semantics.
        "_LIBCPP_ASSERTION_SEMANTIC_DEFAULT": "_LIBCPP_ASSERTION_SEMANTIC_HARDENING_DEPENDENT",

        # Enable various features in libc++ available across platforms. We
        # describe these in a block to allow the BUILD file to sort them.
        #
        # - We enable threads, and use auto-detection rather than forcing an
        #   API.
        # - Availability annotations do not apply to Carbon's libc++, so those
        #   are disabled.
        #
        # Where there are platform differences in the features, we disable them
        # here and re-enable them in the `_LIBCPP_ABI_DEFINES` section using
        # custom logic to detect the relevant platform.
        "_LIBCPP_HAS_FILESYSTEM": "ON",
        "_LIBCPP_HAS_LOCALIZATION": "ON",
        "_LIBCPP_HAS_MONOTONIC_CLOCK": "ON",
        "_LIBCPP_HAS_NO_VENDOR_AVAILABILITY_ANNOTATIONS": "ON",
        "_LIBCPP_HAS_RANDOM_DEVICE": "ON",
        "_LIBCPP_HAS_TERMINAL": "ON",
        "_LIBCPP_HAS_THREADS": "ON",
        "_LIBCPP_HAS_THREAD_API_EXTERNAL": "OFF",
        "_LIBCPP_HAS_THREAD_API_PTHREAD": "OFF",
        "_LIBCPP_HAS_THREAD_API_WIN32": "OFF",
        "_LIBCPP_HAS_TIME_ZONE_DATABASE": "OFF",
        "_LIBCPP_HAS_UNICODE": "ON",
        "_LIBCPP_HAS_VENDOR_AVAILABILITY_ANNOTATIONS": "OFF",
        "_LIBCPP_HAS_WIDE_CHARACTERS": "ON",

        # When ASan is enabled, we ensure that libc++ is built with it as well.
        # However, we can't set this more carefully here so we set it to off and
        # override it below when using ASan.
        "_LIBCPP_INSTRUMENTED_WITH_ASAN": "OFF",

        # Set the parallel backend to serial.
        # TODO: We should revisit this.
        "_LIBCPP_PSTL_BACKEND_SERIAL": "1",
    },
)

configure_cmake_file(
    name = "libcxx_assertion_handler_gen",
    src = "@llvm-project//libcxx:vendor/llvm/default_assertion_handler.in",
    out = "staging_libcxx/include/__assertion_handler",
    defines = {
        # Currently the default handler needs no substitutions.
    },
)

filegroup(
    name = "libcxx",
    srcs = [
        "@llvm-project//libcxx:libcxx_all_srcs",
        "@llvm-project//libcxx:libcxx_hdrs",
    ],
)

filegroup(
    name = "libcxx_gen_files",
    srcs = [
        "staging_libcxx/include/__assertion_handler",
        "staging_libcxx/include/__config_site",
    ],
)

filegroup(
    name = "libcxxabi",
    srcs = [
        "@llvm-project//libcxxabi:libcxxabi_hdrs",
        "@llvm-project//libcxxabi:libcxxabi_srcs",
        "@llvm-project//libcxxabi:libcxxabi_textual_srcs",
    ],
)

filegroup(
    name = "libunwind",
    srcs = [
        "@llvm-project//libunwind:libunwind_hdrs",
        "@llvm-project//libunwind:libunwind_srcs",
    ],
)

# Currently, we're only installing the subset of LLVM's libc internals needed to
# build libc++. At some point, we should ship LLVM's libc itself, and that will
# likely expand this to cover more of the source. However, we'll still want to
# distinguish between the _internal_ installation and the generated set of
# headers that we inject into the include search for user compiles. The
# `include` subdirectory in this file group is _not_ intended to be exposed to
# user compiles, only to compilation of runtimes.
filegroup(
    name = "libc_internal",
    srcs = [
        "@llvm-project//libc:libcxx_shared_headers_hdrs",
    ],
)
